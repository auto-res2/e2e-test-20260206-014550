import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import wandb
from omegaconf import OmegaConf
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt

METRIC_DIRECTIONS = {
    "accuracy": "max",
    "avg_generated_tokens": "min",
    "stage2_rate": "min",
    "cc_separation": "max",
}


def _load_wandb_config() -> Dict[str, str]:
    cfg_path = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
    cfg = OmegaConf.load(cfg_path)
    return {
        "entity": cfg.wandb.entity,
        "project": cfg.wandb.project,
    }


def _to_jsonable(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, pd.Series):
        return [_to_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, pd.DataFrame):
        return [_to_jsonable(rec) for rec in obj.to_dict(orient="records")]
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return str(obj)


def _save_json(path: str, payload: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(payload), f, indent=2)


def _bootstrap_ci(data: np.ndarray, n_boot: int = 1000, alpha: float = 0.05) -> Tuple[float, float]:
    if len(data) == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(0)
    samples = [rng.choice(data, size=len(data), replace=True).mean() for _ in range(n_boot)]
    lower = float(np.percentile(samples, 100 * (alpha / 2)))
    upper = float(np.percentile(samples, 100 * (1 - alpha / 2)))
    return (lower, upper)


def _metric_direction(metric: str) -> str:
    if metric in METRIC_DIRECTIONS:
        return METRIC_DIRECTIONS[metric]
    lowered = metric.lower()
    if any(key in lowered for key in ["loss", "perplexity", "error"]):
        return "min"
    return "max"


def _assert_run_is_full(config: Dict, run_id: str) -> None:
    mode = str(config.get("mode", "full")).lower()
    wandb_cfg = config.get("wandb", {})
    wandb_mode = None
    if isinstance(wandb_cfg, dict):
        wandb_mode = str(wandb_cfg.get("mode", "")).lower()
    if mode == "trial" or wandb_mode == "disabled":
        raise RuntimeError(
            f"Run {run_id} appears to be a trial run with WandB disabled. "
            "Please evaluate only full runs with WandB logging enabled."
        )


def _plot_learning_curve(history: pd.DataFrame, run_id: str, out_dir: str) -> List[str]:
    paths: List[str] = []
    if "cumulative_accuracy" in history.columns:
        y = history["cumulative_accuracy"].dropna().values
    elif "example_correct" in history.columns:
        y = history["example_correct"].dropna().expanding().mean().values
    else:
        return paths
    x = np.arange(len(y))
    plt.figure(figsize=(6, 4))
    plt.plot(x, y, label="Cumulative Accuracy")
    if len(y) > 0:
        plt.scatter([x[-1]], [y[-1]], color="red", label=f"Final={y[-1]:.3f}")
    plt.xlabel("Step")
    plt.ylabel("Accuracy")
    plt.title(f"Learning Curve: {run_id}")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(out_dir, f"{run_id}_learning_curve.pdf")
    plt.savefig(path)
    plt.close()
    paths.append(path)
    return paths


def _plot_tokens_distribution(history: pd.DataFrame, run_id: str, out_dir: str) -> List[str]:
    paths: List[str] = []
    if "example_tokens" not in history.columns:
        return paths
    data = history["example_tokens"].dropna()
    if len(data) == 0:
        return paths
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=data)
    plt.xlabel("Generated Tokens")
    plt.title(f"Tokens per Example: {run_id}")
    plt.tight_layout()
    path = os.path.join(out_dir, f"{run_id}_tokens_boxplot.pdf")
    plt.savefig(path)
    plt.close()
    paths.append(path)
    return paths


def _plot_tokens_scatter(history: pd.DataFrame, run_id: str, out_dir: str) -> List[str]:
    paths: List[str] = []
    if "example_tokens" not in history.columns or "example_correct" not in history.columns:
        return paths
    df = history[["example_tokens", "example_correct"]].dropna()
    if len(df) == 0:
        return paths
    plt.figure(figsize=(6, 4))
    sns.stripplot(x="example_correct", y="example_tokens", data=df, jitter=0.2)
    plt.xlabel("Correct (1) vs Incorrect (0)")
    plt.ylabel("Generated Tokens")
    plt.title(f"Tokens vs Correctness: {run_id}")
    plt.tight_layout()
    path = os.path.join(out_dir, f"{run_id}_tokens_scatter.pdf")
    plt.savefig(path)
    plt.close()
    paths.append(path)
    return paths


def _plot_cc_distribution(history: pd.DataFrame, run_id: str, out_dir: str) -> List[str]:
    paths: List[str] = []
    if "example_cc_mean" not in history.columns or "example_correct" not in history.columns:
        return paths
    df = history[["example_cc_mean", "example_correct"]].dropna()
    if len(df) == 0:
        return paths
    plt.figure(figsize=(6, 4))
    sns.histplot(
        data=df,
        x="example_cc_mean",
        hue="example_correct",
        stat="density",
        common_norm=False,
        bins=20,
    )
    plt.xlabel("CC Mean")
    plt.title(f"CC Distribution (Correct vs Incorrect): {run_id}")
    plt.tight_layout()
    path = os.path.join(out_dir, f"{run_id}_cc_distribution.pdf")
    plt.savefig(path)
    plt.close()
    paths.append(path)
    return paths


def _plot_confusion_matrix(history: pd.DataFrame, run_id: str, out_dir: str, top_k: int = 10) -> List[str]:
    paths: List[str] = []
    if "example_pred" not in history.columns or "example_gold" not in history.columns:
        return paths
    df = history[["example_pred", "example_gold"]].dropna()
    if len(df) == 0:
        return paths
    df = df.astype(str)
    top_gold = df["example_gold"].value_counts().head(top_k).index.tolist()
    top_pred = df["example_pred"].value_counts().head(top_k).index.tolist()
    labels = sorted(set(top_gold + top_pred))

    def map_label(val: str) -> str:
        return val if val in labels else "Other"

    df_mapped = pd.DataFrame(
        {
            "gold": df["example_gold"].map(map_label),
            "pred": df["example_pred"].map(map_label),
        }
    )
    matrix = pd.crosstab(df_mapped["gold"], df_mapped["pred"], dropna=False)
    plt.figure(figsize=(7, 6))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Gold")
    plt.title(f"Confusion Matrix (Top {top_k}) : {run_id}")
    plt.tight_layout()
    path = os.path.join(out_dir, f"{run_id}_confusion_matrix_top{top_k}.pdf")
    plt.savefig(path)
    plt.close()
    paths.append(path)
    return paths


def _plot_comparison_bars(metric: str, metrics: Dict[str, float], out_dir: str) -> str:
    plt.figure(figsize=(7, 4))
    run_ids = list(metrics.keys())
    values = [metrics[r] for r in run_ids]
    ax = sns.barplot(x=run_ids, y=values)
    ax.set_xlabel("Run ID")
    ax.set_ylabel(metric)
    ax.set_title(f"Comparison: {metric}")
    for i, v in enumerate(values):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    path = os.path.join(out_dir, f"comparison_{metric}_bar_chart.pdf")
    plt.savefig(path)
    plt.close()
    return path


def _plot_metrics_table(metrics_by_name: Dict[str, Dict[str, float]], out_dir: str) -> str:
    df = pd.DataFrame(metrics_by_name).T
    plt.figure(figsize=(8, 2 + 0.3 * len(df)))
    plt.axis("off")
    table = plt.table(
        cellText=np.round(df.values, 4),
        rowLabels=df.index,
        colLabels=df.columns,
        cellLoc="center",
        loc="center",
    )
    table.scale(1, 1.2)
    plt.title("Performance Metrics Table", pad=12)
    plt.tight_layout()
    path = os.path.join(out_dir, "comparison_metrics_table.pdf")
    plt.savefig(path)
    plt.close()
    return path


def _plot_accuracy_ci(metrics_by_name: Dict[str, Dict[str, float]], ci_by_run: Dict[str, Tuple[float, float]], out_dir: str) -> str:
    run_ids = list(metrics_by_name["accuracy"].keys())
    values = [metrics_by_name["accuracy"][r] for r in run_ids]
    ci = [ci_by_run.get(r, (0.0, 0.0)) for r in run_ids]
    lower_err = [v - c[0] for v, c in zip(values, ci)]
    upper_err = [c[1] - v for v, c in zip(values, ci)]
    plt.figure(figsize=(7, 4))
    plt.bar(run_ids, values, yerr=[lower_err, upper_err], capsize=4)
    for i, v in enumerate(values):
        plt.text(i, v, f"{v:.3f}", ha="center", va="bottom")
    plt.ylabel("Accuracy")
    plt.title("Accuracy with 95% CI")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    path = os.path.join(out_dir, "comparison_accuracy_ci.pdf")
    plt.savefig(path)
    plt.close()
    return path


def _plot_tokens_boxplot(histories: Dict[str, pd.DataFrame], out_dir: str) -> str:
    rows = []
    for run_id, history in histories.items():
        if "example_tokens" not in history.columns:
            continue
        data = history["example_tokens"].dropna()
        for val in data:
            rows.append({"run_id": run_id, "example_tokens": val})
    df = pd.DataFrame(rows)
    plt.figure(figsize=(7, 4))
    if len(df) > 0:
        sns.boxplot(x="run_id", y="example_tokens", data=df)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Generated Tokens")
    plt.title("Tokens per Example (Across Runs)")
    plt.tight_layout()
    path = os.path.join(out_dir, "comparison_tokens_boxplot.pdf")
    plt.savefig(path)
    plt.close()
    return path


def _parse_cli() -> Tuple[str, str]:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--results_dir", type=str, default=None)
    parser.add_argument("--run_ids", type=str, default=None)
    known, unknown = parser.parse_known_args()

    kv_args: Dict[str, str] = {}
    for arg in unknown:
        if "=" in arg:
            key, value = arg.split("=", 1)
            kv_args[key.lstrip("-")] = value

    results_dir = known.results_dir or kv_args.get("results_dir")
    run_ids = known.run_ids or kv_args.get("run_ids")

    if results_dir is None or run_ids is None:
        raise ValueError(
            "Missing required arguments. Use: python -m src.evaluate results_dir=... run_ids='[...]'"
        )
    return results_dir, run_ids


def _parse_run_ids(run_ids_str: str) -> List[str]:
    try:
        run_ids = json.loads(run_ids_str)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse run_ids from: {run_ids_str}") from exc
    if not isinstance(run_ids, list):
        raise ValueError(f"run_ids must be a JSON list; got: {run_ids_str}")
    return [str(r) for r in run_ids]


def main() -> None:
    results_dir, run_ids_str = _parse_cli()

    cfg = _load_wandb_config()
    entity = cfg["entity"]
    project = cfg["project"]

    api = wandb.Api()
    run_ids = _parse_run_ids(run_ids_str)
    histories: Dict[str, pd.DataFrame] = {}
    summaries: Dict[str, Dict] = {}
    configs: Dict[str, Dict] = {}

    generated_paths: List[str] = []

    for run_id in run_ids:
        try:
            run = api.run(f"{entity}/{project}/{run_id}")
        except wandb.errors.CommError as exc:
            raise RuntimeError(
                f"Failed to fetch run {run_id}. Ensure it was logged in full mode with WandB enabled."
            ) from exc
        history = run.history()
        history = history.replace({np.nan: None})
        summary = run.summary._json_dict
        config = dict(run.config)
        _assert_run_is_full(config, run_id)
        histories[run_id] = history
        summaries[run_id] = summary
        configs[run_id] = config

        run_dir = os.path.join(results_dir, run_id)
        os.makedirs(run_dir, exist_ok=True)

        metrics_payload = {
            "run_id": run_id,
            "summary": summary,
            "config": config,
            "history": history.to_dict(orient="records"),
        }
        metrics_path = os.path.join(run_dir, "metrics.json")
        _save_json(metrics_path, metrics_payload)
        generated_paths.append(metrics_path)

        generated_paths.extend(_plot_learning_curve(history, run_id, run_dir))
        generated_paths.extend(_plot_tokens_distribution(history, run_id, run_dir))
        generated_paths.extend(_plot_tokens_scatter(history, run_id, run_dir))
        generated_paths.extend(_plot_cc_distribution(history, run_id, run_dir))
        generated_paths.extend(_plot_confusion_matrix(history, run_id, run_dir))

    metrics_by_name: Dict[str, Dict[str, float]] = defaultdict(dict)
    ci_by_run: Dict[str, Tuple[float, float]] = {}
    for run_id in run_ids:
        summary = summaries[run_id]
        history = histories[run_id]
        accuracy = summary.get("accuracy")
        if accuracy is None and "example_correct" in history.columns:
            accuracy = float(history["example_correct"].dropna().mean())
        avg_tokens = summary.get("avg_generated_tokens")
        if avg_tokens is None and "example_tokens" in history.columns:
            avg_tokens = float(history["example_tokens"].dropna().mean())
        stage2_rate = summary.get("stage2_rate")
        if stage2_rate is None and "stage2_used" in history.columns:
            stage2_rate = float(history["stage2_used"].dropna().mean())
        cc_sep = summary.get("cc_separation")
        if cc_sep is None and "example_cc_mean" in history.columns and "example_correct" in history.columns:
            df = history[["example_cc_mean", "example_correct"]].dropna()
            cc_correct = df[df["example_correct"] > 0]["example_cc_mean"].mean()
            cc_incorrect = df[df["example_correct"] == 0]["example_cc_mean"].mean()
            cc_sep = float(cc_correct - cc_incorrect)

        metrics_by_name["accuracy"][run_id] = float(accuracy) if accuracy is not None else 0.0
        metrics_by_name["avg_generated_tokens"][run_id] = float(avg_tokens) if avg_tokens is not None else 0.0
        metrics_by_name["stage2_rate"][run_id] = float(stage2_rate) if stage2_rate is not None else 0.0
        metrics_by_name["cc_separation"][run_id] = float(cc_sep) if cc_sep is not None else 0.0

        if "example_correct" in history.columns:
            vals = history["example_correct"].dropna().values
            ci_by_run[run_id] = _bootstrap_ci(vals)

    primary_metric = "accuracy"
    direction = _metric_direction(primary_metric)
    proposed_runs = {k: v for k, v in metrics_by_name[primary_metric].items() if "proposed" in k}
    baseline_runs = {
        k: v
        for k, v in metrics_by_name[primary_metric].items()
        if "baseline" in k or "comparative" in k
    }
    if direction == "min":
        best_proposed = min(proposed_runs.items(), key=lambda x: x[1]) if proposed_runs else (None, None)
        best_baseline = min(baseline_runs.items(), key=lambda x: x[1]) if baseline_runs else (None, None)
    else:
        best_proposed = max(proposed_runs.items(), key=lambda x: x[1]) if proposed_runs else (None, None)
        best_baseline = max(baseline_runs.items(), key=lambda x: x[1]) if baseline_runs else (None, None)

    gap = None
    if best_proposed[0] and best_baseline[0] and best_baseline[1] != 0:
        if direction == "min":
            gap = (best_baseline[1] - best_proposed[1]) / abs(best_baseline[1]) * 100
        else:
            gap = (best_proposed[1] - best_baseline[1]) / abs(best_baseline[1]) * 100

    aggregated_metrics = {
        "primary_metric": primary_metric,
        "metrics": metrics_by_name,
        "best_proposed": {"run_id": best_proposed[0], "value": best_proposed[1]},
        "best_baseline": {"run_id": best_baseline[0], "value": best_baseline[1]},
        "gap": gap,
    }

    if best_proposed[0] and best_baseline[0]:
        prop_hist = histories[best_proposed[0]]
        base_hist = histories[best_baseline[0]]
        stat_tests = {}
        if "example_correct" in prop_hist.columns and "example_correct" in base_hist.columns:
            prop = prop_hist["example_correct"].dropna().values
            base = base_hist["example_correct"].dropna().values
            if len(prop) > 1 and len(base) > 1:
                t_stat, p_val = stats.ttest_ind(prop, base, equal_var=False)
                stat_tests["t_test_accuracy"] = {"t_stat": float(t_stat), "p_value": float(p_val)}
        if "example_tokens" in prop_hist.columns and "example_tokens" in base_hist.columns:
            prop = prop_hist["example_tokens"].dropna().values
            base = base_hist["example_tokens"].dropna().values
            if len(prop) > 1 and len(base) > 1:
                t_stat, p_val = stats.ttest_ind(prop, base, equal_var=False)
                stat_tests["t_test_tokens"] = {"t_stat": float(t_stat), "p_value": float(p_val)}
        if stat_tests:
            aggregated_metrics["stat_tests"] = stat_tests

    comparison_dir = os.path.join(results_dir, "comparison")
    os.makedirs(comparison_dir, exist_ok=True)

    agg_path = os.path.join(comparison_dir, "aggregated_metrics.json")
    _save_json(agg_path, aggregated_metrics)
    generated_paths.append(agg_path)

    for metric_name, values in metrics_by_name.items():
        generated_paths.append(_plot_comparison_bars(metric_name, values, comparison_dir))

    if "accuracy" in metrics_by_name:
        generated_paths.append(_plot_accuracy_ci(metrics_by_name, ci_by_run, comparison_dir))

    generated_paths.append(_plot_metrics_table(metrics_by_name, comparison_dir))
    generated_paths.append(_plot_tokens_boxplot(histories, comparison_dir))

    for path in generated_paths:
        print(path)


if __name__ == "__main__":
    main()
