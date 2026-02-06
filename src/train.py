import math
from copy import deepcopy
from typing import Any, Dict, List, Optional

import hydra
import numpy as np
import optuna
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from src.model import build_method, create_model_and_tokenizer, extract_answer
from src.preprocess import QADataset, load_dataset_split, set_seed


def _apply_mode_overrides(cfg: DictConfig) -> DictConfig:
    if cfg.mode == "trial":
        cfg.wandb.mode = "disabled"
        if hasattr(cfg, "run") and hasattr(cfg.run, "optuna"):
            cfg.run.optuna.n_trials = 0
        if hasattr(cfg, "run") and hasattr(cfg.run, "dataset"):
            subset_size = int(getattr(cfg.run.dataset, "subset_size", 2) or 2)
            cfg.run.dataset.subset_size = min(subset_size, 2)
        if hasattr(cfg, "run") and hasattr(cfg.run, "training") and getattr(cfg.run.training, "enabled", False):
            cfg.run.training.epochs = min(int(getattr(cfg.run.training, "epochs", 1)), 1)
            cfg.run.training.batch_size = min(int(getattr(cfg.run.training, "batch_size", 1)), 1)
        if hasattr(cfg, "run") and hasattr(cfg.run, "model"):
            cfg.run.model.max_new_tokens_stage1 = min(
                int(getattr(cfg.run.model, "max_new_tokens_stage1", 16)), 16
            )
            cfg.run.model.max_new_tokens_stage2 = min(
                int(getattr(cfg.run.model, "max_new_tokens_stage2", 32)), 32
            )
    elif cfg.mode == "full":
        cfg.wandb.mode = "online"
    else:
        raise ValueError(f"Unsupported mode: {cfg.mode}")
    return cfg


def _get_optuna_cfg(run_cfg: DictConfig) -> Optional[DictConfig]:
    return getattr(run_cfg, "optuna", None)


def _space_to_dict(space: Any) -> Dict[str, Any]:
    if isinstance(space, DictConfig):
        return OmegaConf.to_container(space, resolve=True)
    if isinstance(space, dict):
        return dict(space)
    return {}


def _resolve_method_params(run_cfg: DictConfig, mode: str) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    for key in ("method_params", "params"):
        if hasattr(run_cfg, key):
            candidate = getattr(run_cfg, key)
            if isinstance(candidate, DictConfig):
                params.update(OmegaConf.to_container(candidate, resolve=True))
            elif isinstance(candidate, dict):
                params.update(candidate)
    optuna_cfg = _get_optuna_cfg(run_cfg)
    if optuna_cfg is not None and getattr(optuna_cfg, "search_spaces", None):
        for space in optuna_cfg.search_spaces:
            spec = _space_to_dict(space)
            name = spec.get("param_name") or spec.get("name")
            if not name or name in params:
                continue
            dist = str(spec.get("distribution_type", "categorical")).lower()
            if dist == "categorical":
                choices = list(spec.get("choices") or [])
                if choices:
                    params[name] = choices[0]
            elif dist == "int":
                low = int(spec.get("low", 0))
                high = int(spec.get("high", low))
                params[name] = int((low + high) // 2)
            elif dist == "float":
                low = float(spec.get("low", 0.0))
                high = float(spec.get("high", low))
                params[name] = float(low + (high - low) / 2)
            else:
                raise ValueError(f"Unsupported distribution type: {dist}")
    params = {k: v for k, v in params.items() if v is not None}
    if mode == "trial":
        params["K_short"] = min(int(params.get("K_short", 1)), 1)
        params["K_long"] = min(int(params.get("K_long", 2)), 2)
    return params


def _sample_hparams(trial: optuna.Trial, search_spaces: List[Dict[str, Any]]) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    for space in search_spaces:
        name = space["param_name"]
        dist = space.get("distribution_type", "categorical")
        if dist == "categorical":
            params[name] = trial.suggest_categorical(name, list(space["choices"]))
        elif dist == "int":
            params[name] = trial.suggest_int(name, int(space["low"]), int(space["high"]))
        elif dist == "float":
            params[name] = trial.suggest_float(name, float(space["low"]), float(space["high"]))
        else:
            raise ValueError(f"Unsupported distribution type: {dist}")
    return params


def _assert_gradients(model: torch.nn.Module) -> None:
    has_grad = False
    total_norm = 0.0
    for param in model.parameters():
        if not param.requires_grad:
            continue
        if param.grad is None:
            continue
        has_grad = True
        total_norm += param.grad.detach().abs().sum().item()
    assert has_grad, "No gradients found before optimizer step."
    assert total_norm > 0.0, "Gradients are zero before optimizer step."


def _build_optimizer(model: torch.nn.Module, train_cfg: DictConfig) -> Optional[torch.optim.Optimizer]:
    opt_name = str(getattr(train_cfg, "optimizer", "none")).lower()
    if opt_name in {"none", "null", "disabled"}:
        return None
    lr = float(getattr(train_cfg, "learning_rate", 0.0))
    if lr <= 0:
        raise ValueError("learning_rate must be > 0 when optimizer is enabled.")
    weight_decay = float(getattr(train_cfg, "weight_decay", 0.0))
    beta1 = float(getattr(train_cfg, "beta1", 0.9))
    beta2 = float(getattr(train_cfg, "beta2", 0.999))
    if opt_name in {"adamw", "adam_w"}:
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(beta1, beta2))
    if opt_name in {"adam"}:
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(beta1, beta2))
    if opt_name in {"sgd"}:
        momentum = float(getattr(train_cfg, "momentum", 0.0))
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {opt_name}")


def _train_model(
    model: torch.nn.Module,
    tokenizer,
    train_examples: List[Dict[str, str]],
    cfg: DictConfig,
    device: str,
    wandb_run: Optional[wandb.sdk.wandb_run.Run] = None,
) -> None:
    run_cfg = cfg.run
    train_cfg = run_cfg.training
    epochs = int(getattr(train_cfg, "epochs", 0))
    if epochs <= 0:
        return
    max_length = int(run_cfg.dataset.preprocessing.max_length)
    dataset = QADataset(train_examples, tokenizer, max_length=max_length)
    if len(dataset) == 0:
        return
    loader = DataLoader(dataset, batch_size=int(train_cfg.batch_size), shuffle=True)
    optimizer = _build_optimizer(model, train_cfg)
    if optimizer is None:
        raise ValueError("Training enabled but optimizer is set to none.")
    total_steps = max(1, epochs * max(len(loader), 1))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(getattr(train_cfg, "warmup_steps", 0)), num_training_steps=total_steps
    )
    model.train()
    step = 0
    for epoch in range(epochs):
        for batch in loader:
            if step == 0:
                assert batch["input_ids"].shape[0] == batch["labels"].shape[0], "Batch sizes mismatch."
                assert batch["input_ids"].ndim == 2 and batch["labels"].ndim == 2, "Unexpected batch dims."
            if cfg.mode == "trial" and step >= 2:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            outputs = model(**batch, labels=labels)
            loss = outputs.loss
            assert loss is not None, "Training loss is None."
            params = [p for p in model.parameters() if p.requires_grad]
            aux_grads = torch.autograd.grad(
                loss, params, retain_graph=True, create_graph=False, allow_unused=True
            )
            aux_norm = 0.0
            for grad in aux_grads:
                if grad is not None:
                    aux_norm += grad.detach().float().pow(2).sum().item()
            aux_norm = math.sqrt(aux_norm) if aux_norm > 0 else 0.0
            loss.backward()
            _assert_gradients(model)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            if wandb_run is not None:
                wandb.log(
                    {
                        "train_loss": float(loss.item()),
                        "train_aux_grad_norm": float(aux_norm),
                        "train_epoch": int(epoch),
                        "train_lr": float(optimizer.param_groups[0]["lr"]),
                    },
                    step=step,
                )
            step += 1
        if cfg.mode == "trial" and step >= 2:
            break
    model.eval()


def _run_inference(
    method,
    dataset: List[Dict[str, str]],
    tokenizer,
    cfg: DictConfig,
    device: str,
    wandb_run: Optional[wandb.sdk.wandb_run.Run] = None,
) -> Dict[str, float]:
    total = 0
    correct = 0
    total_tokens = 0
    stage2_count = 0
    cc_correct: List[float] = []
    cc_incorrect: List[float] = []
    processed = 0
    step = 0
    max_examples = 2 if cfg.mode == "trial" else None
    for idx, ex in enumerate(dataset):
        if max_examples is not None and idx >= max_examples:
            break
        question = ex["question"]
        gold = extract_answer(ex["answer"])
        if idx == 0 and gold is not None:
            prompt = method.build_prompt(question, verify=False) if hasattr(method, "build_prompt") else question
            enc = tokenizer(prompt, return_tensors="pt", truncation=True)
            dec = tokenizer(f"The answer is {gold}.", return_tensors="pt", truncation=True)
            assert enc.input_ids.shape[0] == dec.input_ids.shape[0] == 1, "Batch size mismatch."
            assert enc.input_ids.ndim == 2 and dec.input_ids.ndim == 2, "Unexpected input dims."
        result = method.predict(question)
        pred = result.get("pred_answer")
        is_correct = 0
        if pred is not None and gold is not None:
            total += 1
            is_correct = int(pred == gold)
            correct += is_correct
        total_tokens += int(result.get("used_tokens", 0))
        stage2_count += int(result.get("stage2_used", 0))
        cc_vals = result.get("cc_values", [])
        if pred is not None and gold is not None:
            if is_correct:
                cc_correct.extend(cc_vals)
            else:
                cc_incorrect.extend(cc_vals)
        cc_mean = float(np.mean(cc_vals)) if len(cc_vals) > 0 else 0.0
        processed += 1
        cumulative_accuracy = correct / max(total, 1)
        if wandb_run is not None:
            pred_str = str(pred) if pred is not None else ""
            gold_str = str(gold) if gold is not None else ""
            wandb.log(
                {
                    "example_correct": int(is_correct),
                    "example_tokens": int(result.get("used_tokens", 0)),
                    "stage2_used": int(result.get("stage2_used", 0)),
                    "example_cc_mean": float(cc_mean),
                    "stage1_top1_prob": float(result.get("stage1_top1_prob", 0.0)),
                    "stage1_margin": float(result.get("stage1_margin", 0.0)),
                    "final_top1_prob": float(result.get("final_top1_prob", 0.0)),
                    "example_pred": pred_str,
                    "example_gold": gold_str,
                    "cumulative_accuracy": float(cumulative_accuracy),
                    "example_idx": int(idx),
                },
                step=step,
            )
        step += 1
    accuracy = correct / max(total, 1)
    avg_tokens = total_tokens / max(processed, 1)
    stage2_rate = stage2_count / max(processed, 1)
    cc_correct_mean = float(np.mean(cc_correct)) if len(cc_correct) > 0 else 0.0
    cc_incorrect_mean = float(np.mean(cc_incorrect)) if len(cc_incorrect) > 0 else 0.0
    cc_separation = cc_correct_mean - cc_incorrect_mean
    return {
        "accuracy": float(accuracy),
        "avg_generated_tokens": float(avg_tokens),
        "stage2_rate": float(stage2_rate),
        "cc_separation": float(cc_separation),
        "cc_correct_mean": float(cc_correct_mean),
        "cc_incorrect_mean": float(cc_incorrect_mean),
        "n_eval": int(total),
    }


def _run_optuna(
    cfg: DictConfig,
    run_cfg: DictConfig,
    model: torch.nn.Module,
    tokenizer,
    val_dataset: List[Dict[str, str]],
    base_params: Dict[str, Any],
    device: str,
) -> Dict[str, Any]:
    optuna_cfg = _get_optuna_cfg(run_cfg)
    if optuna_cfg is None:
        return {}
    if int(getattr(optuna_cfg, "n_trials", 0)) <= 0:
        return {}
    if len(val_dataset) == 0:
        return {}

    def objective(trial: optuna.Trial) -> float:
        set_seed(int(cfg.seed) + int(trial.number))
        trial_params = deepcopy(base_params)
        trial_params.update(_sample_hparams(trial, list(optuna_cfg.search_spaces)))
        method = build_method(run_cfg.method, model, tokenizer, run_cfg, trial_params, device)
        metrics = _run_inference(method, val_dataset, tokenizer, cfg, device, wandb_run=None)
        return float(metrics["accuracy"])

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=int(optuna_cfg.n_trials))
    return study.best_params


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, False)
    cfg = _apply_mode_overrides(cfg)
    set_seed(int(cfg.seed))

    run_cfg = cfg.run
    run_id = str(getattr(run_cfg, "run_id", ""))
    if not run_id:
        raise ValueError("run_id is missing in the run configuration.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = create_model_and_tokenizer(run_cfg.model, device)
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer pad_token_id is missing after initialization.")
    output_embeddings = model.get_output_embeddings()
    assert output_embeddings is not None, "Model output embeddings are missing."
    assert output_embeddings.weight.shape[0] == model.config.vocab_size, "Output dimension mismatch."

    dataset = load_dataset_split(run_cfg.dataset)
    if len(dataset) == 0:
        raise ValueError("Loaded dataset is empty.")

    method_params = _resolve_method_params(run_cfg, cfg.mode)
    run_cfg.method_params = method_params

    val_dataset: List[Dict[str, str]] = []
    if hasattr(run_cfg, "optuna") and int(getattr(run_cfg.optuna, "n_trials", 0)) > 0:
        try:
            val_dataset = load_dataset_split(run_cfg.dataset, split_override="train", subset_size=50, seed=cfg.seed)
        except Exception:
            val_dataset = dataset[: min(50, len(dataset))]

    best_params = _run_optuna(cfg, run_cfg, model, tokenizer, val_dataset, method_params, device)
    if best_params:
        method_params.update(best_params)
        run_cfg.method_params = method_params

    wandb_run: Optional[wandb.sdk.wandb_run.Run] = None
    if cfg.wandb.mode != "disabled":
        wandb_run = wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            id=run_id,
            name=run_id,
            config=OmegaConf.to_container(cfg, resolve=True),
            resume="allow",
        )
        if method_params:
            wandb_run.config.update(method_params, allow_val_change=True)

    if getattr(run_cfg.training, "enabled", False):
        try:
            train_examples = load_dataset_split(run_cfg.dataset, split_override="train")
        except Exception:
            train_examples = dataset
        _train_model(model, tokenizer, train_examples, cfg, device, wandb_run=wandb_run)

    method = build_method(run_cfg.method, model, tokenizer, run_cfg, method_params, device)
    metrics = _run_inference(method, dataset, tokenizer, cfg, device, wandb_run=wandb_run)

    if wandb_run is not None:
        for key, value in metrics.items():
            wandb_run.summary[key] = value
        print(f"WandB URL: {wandb_run.url}")
        wandb_run.finish()
    else:
        print("WandB logging disabled; run completed.")


if __name__ == "__main__":
    main()
