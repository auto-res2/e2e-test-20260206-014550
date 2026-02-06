import random
from typing import Dict, List, Optional

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import Dataset

from src.model import extract_answer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _resolve_dataset_name(name: str) -> str:
    lname = name.lower()
    if lname in {"openai/gsm8k", "gsm8k"}:
        return "gsm8k"
    if "svamp" in lname:
        return "svamp"
    if "asdiv" in lname:
        return "asdiv"
    return name


def _extract_qa(name: str, ex: Dict) -> Optional[Dict[str, str]]:
    lname = name.lower()
    if "gsm8k" in lname:
        q = ex.get("question")
        a = ex.get("answer")
        return {"question": str(q).strip(), "answer": str(a).strip()} if q and a else None
    if "svamp" in lname:
        body = ex.get("Body") or ex.get("body") or ""
        question = ex.get("Question") or ex.get("question") or ""
        q = f"{body} {question}".strip()
        a = ex.get("Answer") or ex.get("answer")
        return {"question": q, "answer": str(a).strip()} if q and a else None
    if "asdiv" in lname:
        q = ex.get("question") or ex.get("problem") or ex.get("Question") or ""
        a = ex.get("answer") or ex.get("Answer") or ex.get("solution") or ""
        return {"question": str(q).strip(), "answer": str(a).strip()} if q and a else None
    q = ex.get("question") or ex.get("prompt") or ex.get("input")
    a = ex.get("answer") or ex.get("output") or ex.get("label")
    if q is None or a is None:
        return None
    return {"question": str(q).strip(), "answer": str(a).strip()}


def load_dataset_split(
    dataset_cfg,
    split_override: Optional[str] = None,
    subset_size: Optional[int] = None,
    seed: Optional[int] = None,
) -> List[Dict[str, str]]:
    name = _resolve_dataset_name(dataset_cfg.name)
    subset = getattr(dataset_cfg, "subset", None)
    split = split_override or dataset_cfg.split
    try:
        ds = load_dataset(name, subset, split=split, cache_dir=".cache/")
    except Exception as exc:
        raise RuntimeError(f"Failed to load dataset {name} ({subset}) split={split}: {exc}") from exc
    if getattr(dataset_cfg, "shuffle", False):
        ds = ds.shuffle(seed=seed if seed is not None else int(getattr(dataset_cfg, "shuffle_seed", 0)))
    if subset_size is None:
        subset_size = getattr(dataset_cfg, "subset_size", None)
    if subset_size:
        ds = ds.select(range(min(int(subset_size), len(ds))))
    examples: List[Dict[str, str]] = []
    for ex in ds:
        parsed = _extract_qa(name, ex)
        if parsed is None:
            continue
        examples.append(parsed)
    return examples


class QADataset(Dataset):
    def __init__(self, examples: List[Dict[str, str]], tokenizer, max_length: int):
        self.examples = []
        for ex in examples:
            gold = extract_answer(ex["answer"])
            if gold is None:
                continue
            self.examples.append({"question": ex["question"], "answer": gold})
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.examples[idx]
        question = ex["question"]
        gold = ex["answer"]
        input_text = f"Q: {question}\nA:"
        target_text = f"The answer is {gold}."
        model_inputs = self.tokenizer(
            input_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        labels = self.tokenizer(
            target_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        ).input_ids
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {
            "input_ids": model_inputs.input_ids.squeeze(0),
            "attention_mask": model_inputs.attention_mask.squeeze(0),
            "labels": labels.squeeze(0),
        }
