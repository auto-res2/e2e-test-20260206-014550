import math
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

ANS_RE = re.compile(r"The answer is\s*(-?\d+\.?\d*)", re.IGNORECASE)
NUM_RE = re.compile(r"-?\d+\.?\d*")


def normalize_answer(ans: Optional[str]) -> Optional[str]:
    if ans is None:
        return None
    ans = str(ans).strip().replace(",", "")
    ans = ans.lstrip("+")
    if ans.endswith("."):
        ans = ans[:-1]
    return ans if ans else None


def extract_answer(text: str) -> Optional[str]:
    if text is None:
        return None
    match = ANS_RE.search(str(text))
    if match:
        return normalize_answer(match.group(1))
    nums = NUM_RE.findall(str(text).replace(",", ""))
    if not nums:
        return None
    return normalize_answer(nums[-1])


def split_rationale_answer(text: str) -> Tuple[str, str]:
    if text is None:
        return "", ""
    lower = text.lower()
    idx = lower.rfind("the answer is")
    if idx == -1:
        return text.strip(), ""
    return text[:idx].strip(), text[idx:].strip()


def chunk_rationale(rationale: str, max_chunks: int = 6) -> List[str]:
    if not rationale:
        return []
    parts = [p.strip() for p in rationale.split("\n") if p.strip()]
    if len(parts) <= 1:
        parts = [p.strip() for p in re.split(r"(?<=[\.!\?])\s+", rationale) if p.strip()]
    if not parts:
        return []
    if len(parts) <= max_chunks:
        return parts
    step = math.ceil(len(parts) / max_chunks)
    merged = []
    for i in range(0, len(parts), step):
        merged.append(" ".join(parts[i : i + step]))
    return merged[:max_chunks]


def mask_numbers(rationale: str) -> str:
    if rationale is None:
        return ""
    masked = re.sub(r"-?\d+\.?\d*", "X", rationale)
    masked = re.sub(r"[+\-*/=]", "#", masked)
    return masked


def logp_text(
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    encoder_text: str,
    decoder_text: str,
    device: str,
    max_length: int = 512,
) -> float:
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer pad_token_id is required for logp computation.")
    enc = tokenizer(encoder_text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
    dec = tokenizer(decoder_text, return_tensors="pt", truncation=True, max_length=max_length)
    labels = dec.input_ids.to(device)
    labels_mask = labels != tokenizer.pad_token_id
    labels = labels.masked_fill(~labels_mask, -100)
    if labels_mask.sum().item() == 0:
        return 0.0
    with torch.no_grad():
        outputs = model(**enc, labels=labels)
        n_tokens = labels_mask.sum().item()
        logp = -outputs.loss.detach().float() * max(n_tokens, 1)
    return float(logp.item())


def minimal_sufficient_prefix_len(
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    prefix: str,
    rationale: str,
    ans_text: str,
    Lr_full: float,
    eps: float,
    max_chunks: int,
    device: str,
    max_length: int,
) -> int:
    chunks = chunk_rationale(rationale, max_chunks=max_chunks)
    if not chunks:
        return 0
    for k in range(1, len(chunks) + 1):
        r_pref = "\n".join(chunks[:k])
        Lk = logp_text(
            model,
            tokenizer,
            encoder_text=prefix + "\n" + r_pref,
            decoder_text=ans_text,
            device=device,
            max_length=max_length,
        )
        if Lk >= (Lr_full - eps):
            return len(tokenizer(r_pref, add_special_tokens=False).input_ids)
    return len(tokenizer(rationale, add_special_tokens=False).input_ids)


def create_model_and_tokenizer(model_cfg, device: str):
    precision = getattr(model_cfg, "precision", "fp32")
    if device == "cuda" and precision in {"fp16", "float16"}:
        dtype = torch.float16
    elif device == "cuda" and precision in {"bf16", "bfloat16"}:
        dtype = torch.bfloat16
    else:
        dtype = torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_cfg.name, cache_dir=".cache/")
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer pad_token_id is missing after initialization.")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_cfg.name, cache_dir=".cache/", torch_dtype=dtype
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)
    model.eval()
    return model, tokenizer


class BaseAggregator:
    def __init__(
        self,
        model: AutoModelForSeq2SeqLM,
        tokenizer: AutoTokenizer,
        device: str,
        model_cfg,
        params: Dict[str, float],
        max_length: int,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.temperature = float(getattr(model_cfg, "temperature", 0.7))
        self.top_p = float(getattr(model_cfg, "top_p", 0.95))
        self.max_new_tokens_stage1 = int(getattr(model_cfg, "max_new_tokens_stage1", 64))
        self.max_new_tokens_stage2 = int(getattr(model_cfg, "max_new_tokens_stage2", 256))
        self.K_short = int(params.get("K_short", 4))
        self.K_long = int(params.get("K_long", 4))
        self.p_stop = float(params.get("p_stop", 0.85))
        self.m_stop = float(params.get("m_stop", 0.35))
        self.tau = float(params.get("tau", 1.0))
        self.lambda_len = float(params.get("lambda_len", 0.0))
        self.max_length = max_length
        self.max_chunks = int(params.get("max_chunks", 6))

    def build_prompt(self, question: str, verify: bool = False) -> str:
        prompt = f"Q: {question}\nA: Let’s think step by step.".replace("\u2019", "'")
        if verify:
            prompt += " Check for mistakes and contradictions."
        prompt += " End with 'The answer is <number>'."
        return prompt

    def _sample_traces(self, question: str, n: int, max_new_tokens: int, verify: bool) -> Tuple[str, List[str]]:
        prompt = self.build_prompt(question, verify=verify)
        enc = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)
        temperature = max(self.temperature, 1e-5)
        with torch.no_grad():
            gen = self.model.generate(
                **enc,
                do_sample=True,
                temperature=temperature,
                top_p=self.top_p,
                max_new_tokens=max_new_tokens,
                num_return_sequences=n,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        decoded = self.tokenizer.batch_decode(gen, skip_special_tokens=True)
        return prompt, decoded

    def posterior(self, weights: Dict[str, float]) -> Tuple[Optional[str], Optional[float], Optional[float]]:
        if not weights:
            return None, None, None
        Z = sum(weights.values())
        items = sorted(((a, w / Z) for a, w in weights.items()), key=lambda x: x[1], reverse=True)
        top1a, top1p = items[0]
        top2p = items[1][1] if len(items) > 1 else 0.0
        return top1a, top1p, top1p - top2p

    def score_samples(self, prefix: str, samples: List[str]) -> Tuple[Dict[str, float], int, List[float]]:
        weights: Dict[str, float] = defaultdict(float)
        token_count = 0
        cc_values: List[float] = []
        base_cache: Dict[str, float] = {}
        for text in samples:
            ans = extract_answer(text)
            if ans is None:
                continue
            rationale, _ = split_rationale_answer(text)
            token_count += len(self.tokenizer(text, add_special_tokens=False).input_ids)
            ans_text = f"The answer is {ans}."
            if ans not in base_cache:
                base_cache[ans] = logp_text(
                    self.model,
                    self.tokenizer,
                    encoder_text=prefix,
                    decoder_text=ans_text,
                    device=self.device,
                    max_length=self.max_length,
                )
            L0 = base_cache[ans]
            Lr = logp_text(
                self.model,
                self.tokenizer,
                encoder_text=prefix + "\n" + rationale,
                decoder_text=ans_text,
                device=self.device,
                max_length=self.max_length,
            )
            EG = Lr - L0
            rationale_len = len(self.tokenizer(rationale, add_special_tokens=False).input_ids)
            score = EG - self.lambda_len * rationale_len
            weights[ans] += math.exp(score / max(self.tau, 1e-6))
            Lcf = logp_text(
                self.model,
                self.tokenizer,
                encoder_text=prefix + "\n" + mask_numbers(rationale),
                decoder_text=ans_text,
                device=self.device,
                max_length=self.max_length,
            )
            cc_values.append(Lr - Lcf)
        return weights, token_count, cc_values

    def predict(self, question: str) -> Dict[str, float]:
        prefix1, samples1 = self._sample_traces(
            question, n=self.K_short, max_new_tokens=self.max_new_tokens_stage1, verify=False
        )
        weights1, tok1, cc1 = self.score_samples(prefix1, samples1)
        a1, p1, m1 = self.posterior(weights1)
        if a1 is None:
            return {
                "pred_answer": None,
                "used_tokens": tok1,
                "stage2_used": 0,
                "cc_values": cc1,
                "stage1_top1_prob": 0.0,
                "stage1_margin": 0.0,
                "final_top1_prob": 0.0,
            }
        if (p1 >= self.p_stop) or (m1 >= self.m_stop):
            final_answer = a1
            final_prob = p1
            total_tokens = tok1
            cc_vals = cc1
            stage2_used = 0
        else:
            prefix2, samples2 = self._sample_traces(
                question, n=self.K_long, max_new_tokens=self.max_new_tokens_stage2, verify=True
            )
            weights2, tok2, cc2 = self.score_samples(prefix2, samples2)
            for ans, w in weights2.items():
                weights1[ans] = weights1.get(ans, 0.0) + w
            final_answer, final_prob, _ = self.posterior(weights1)
            total_tokens = tok1 + tok2
            cc_vals = cc1 + cc2
            stage2_used = 1
        return {
            "pred_answer": final_answer,
            "used_tokens": total_tokens,
            "stage2_used": stage2_used,
            "cc_values": cc_vals,
            "stage1_top1_prob": float(p1) if p1 is not None else 0.0,
            "stage1_margin": float(m1) if m1 is not None else 0.0,
            "final_top1_prob": float(final_prob) if final_prob is not None else 0.0,
        }


class DirectAnswer:
    def __init__(
        self,
        model: AutoModelForSeq2SeqLM,
        tokenizer: AutoTokenizer,
        device: str,
        model_cfg,
        params: Dict[str, float],
        max_length: int,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_new_tokens = int(getattr(model_cfg, "max_new_tokens_stage1", 64))
        self.max_length = max_length

    def build_prompt(self, question: str, verify: bool = False) -> str:
        return f"Q: {question}\nA: The answer is"

    def predict(self, question: str) -> Dict[str, float]:
        prompt = self.build_prompt(question)
        enc = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)
        with torch.no_grad():
            gen = self.model.generate(
                **enc,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        decoded = self.tokenizer.batch_decode(gen, skip_special_tokens=True)
        output = decoded[0] if decoded else ""
        pred = extract_answer(output)
        used_tokens = len(self.tokenizer(output, add_special_tokens=False).input_ids)
        return {
            "pred_answer": pred,
            "used_tokens": used_tokens,
            "stage2_used": 0,
            "cc_values": [],
            "stage1_top1_prob": 1.0,
            "stage1_margin": 1.0,
            "final_top1_prob": 1.0,
        }


class SCMajority:
    def __init__(
        self,
        model: AutoModelForSeq2SeqLM,
        tokenizer: AutoTokenizer,
        device: str,
        model_cfg,
        params: Dict[str, float],
        max_length: int,
        compute_cc: bool = True,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.temperature = float(getattr(model_cfg, "temperature", 0.7))
        self.top_p = float(getattr(model_cfg, "top_p", 0.95))
        self.max_new_tokens = int(getattr(model_cfg, "max_new_tokens_stage2", 256))
        self.K = int(params.get("K_long", 10))
        self.max_length = max_length
        self.compute_cc = compute_cc

    def build_prompt(self, question: str, verify: bool = False) -> str:
        prompt = f"Q: {question}\nA: Let’s think step by step.".replace("\u2019", "'")
        prompt += " End with 'The answer is <number>'."
        return prompt

    def predict(self, question: str) -> Dict[str, float]:
        prompt = self.build_prompt(question)
        enc = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_length).to(
            self.device
        )
        with torch.no_grad():
            gen = self.model.generate(
                **enc,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                max_new_tokens=self.max_new_tokens,
                num_return_sequences=self.K,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        decoded = self.tokenizer.batch_decode(gen, skip_special_tokens=True)
        counts: Dict[str, int] = defaultdict(int)
        total_tokens = 0
        cc_values: List[float] = []
        for text in decoded:
            total_tokens += len(self.tokenizer(text, add_special_tokens=False).input_ids)
            ans = extract_answer(text)
            if ans is None:
                continue
            counts[ans] += 1
            if self.compute_cc:
                rationale, _ = split_rationale_answer(text)
                ans_text = f"The answer is {ans}."
                Lr = logp_text(
                    self.model,
                    self.tokenizer,
                    encoder_text=prompt + "\n" + rationale,
                    decoder_text=ans_text,
                    device=self.device,
                    max_length=self.max_length,
                )
                Lcf = logp_text(
                    self.model,
                    self.tokenizer,
                    encoder_text=prompt + "\n" + mask_numbers(rationale),
                    decoder_text=ans_text,
                    device=self.device,
                    max_length=self.max_length,
                )
                cc_values.append(Lr - Lcf)
        if not counts:
            return {
                "pred_answer": None,
                "used_tokens": total_tokens,
                "stage2_used": 0,
                "cc_values": cc_values,
                "stage1_top1_prob": 0.0,
                "stage1_margin": 0.0,
                "final_top1_prob": 0.0,
            }
        total = sum(counts.values())
        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        top1a, top1c = sorted_counts[0]
        top2c = sorted_counts[1][1] if len(sorted_counts) > 1 else 0
        top1p = top1c / max(total, 1)
        margin = (top1c - top2c) / max(total, 1)
        return {
            "pred_answer": top1a,
            "used_tokens": total_tokens,
            "stage2_used": 0,
            "cc_values": cc_values,
            "stage1_top1_prob": float(top1p),
            "stage1_margin": float(margin),
            "final_top1_prob": float(top1p),
        }


class BoNAnsLogP:
    def __init__(
        self,
        model: AutoModelForSeq2SeqLM,
        tokenizer: AutoTokenizer,
        device: str,
        model_cfg,
        params: Dict[str, float],
        max_length: int,
        compute_cc: bool = True,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.temperature = float(getattr(model_cfg, "temperature", 0.7))
        self.top_p = float(getattr(model_cfg, "top_p", 0.95))
        self.max_new_tokens = int(getattr(model_cfg, "max_new_tokens_stage2", 256))
        self.K = int(params.get("K_long", 10))
        self.tau = float(params.get("tau", 1.0))
        self.max_length = max_length
        self.compute_cc = compute_cc

    def build_prompt(self, question: str, verify: bool = False) -> str:
        prompt = f"Q: {question}\nA: Let’s think step by step.".replace("\u2019", "'")
        prompt += " End with 'The answer is <number>'."
        return prompt

    def predict(self, question: str) -> Dict[str, float]:
        prompt = self.build_prompt(question)
        enc = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_length).to(
            self.device
        )
        with torch.no_grad():
            gen = self.model.generate(
                **enc,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                max_new_tokens=self.max_new_tokens,
                num_return_sequences=self.K,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        decoded = self.tokenizer.batch_decode(gen, skip_special_tokens=True)
        total_tokens = 0
        best_score = -float("inf")
        best_answer = None
        weights: Dict[str, float] = defaultdict(float)
        cc_values: List[float] = []
        for text in decoded:
            total_tokens += len(self.tokenizer(text, add_special_tokens=False).input_ids)
            ans = extract_answer(text)
            if ans is None:
                continue
            rationale, _ = split_rationale_answer(text)
            ans_text = f"The answer is {ans}."
            Lr = logp_text(
                self.model,
                self.tokenizer,
                encoder_text=prompt + "\n" + rationale,
                decoder_text=ans_text,
                device=self.device,
                max_length=self.max_length,
            )
            if Lr > best_score:
                best_score = Lr
                best_answer = ans
            weights[ans] += math.exp(Lr / max(self.tau, 1e-6))
            if self.compute_cc:
                Lcf = logp_text(
                    self.model,
                    self.tokenizer,
                    encoder_text=prompt + "\n" + mask_numbers(rationale),
                    decoder_text=ans_text,
                    device=self.device,
                    max_length=self.max_length,
                )
                cc_values.append(Lr - Lcf)
        if not weights:
            return {
                "pred_answer": None,
                "used_tokens": total_tokens,
                "stage2_used": 0,
                "cc_values": cc_values,
                "stage1_top1_prob": 0.0,
                "stage1_margin": 0.0,
                "final_top1_prob": 0.0,
            }
        Z = sum(weights.values())
        top1p = weights.get(best_answer, 0.0) / max(Z, 1e-6)
        return {
            "pred_answer": best_answer,
            "used_tokens": total_tokens,
            "stage2_used": 0,
            "cc_values": cc_values,
            "stage1_top1_prob": float(top1p),
            "stage1_margin": float(top1p),
            "final_top1_prob": float(top1p),
        }


class MetaPMIADSC(BaseAggregator):
    def __init__(
        self,
        model: AutoModelForSeq2SeqLM,
        tokenizer: AutoTokenizer,
        device: str,
        model_cfg,
        params: Dict[str, float],
        max_length: int,
        compute_cc: bool = True,
    ):
        super().__init__(model, tokenizer, device, model_cfg, params, max_length)
        self.lambda_len = float(params.get("lambda_len", 0.002))
        self.compute_cc = compute_cc

    def score_samples(self, prefix: str, samples: List[str]) -> Tuple[Dict[str, float], int, List[float]]:
        weights: Dict[str, float] = defaultdict(float)
        token_count = 0
        cc_values: List[float] = []
        base_cache: Dict[str, float] = {}
        for text in samples:
            ans = extract_answer(text)
            if ans is None:
                continue
            rationale, _ = split_rationale_answer(text)
            token_count += len(self.tokenizer(text, add_special_tokens=False).input_ids)
            ans_text = f"The answer is {ans}."
            if ans not in base_cache:
                base_cache[ans] = logp_text(
                    self.model,
                    self.tokenizer,
                    encoder_text=prefix,
                    decoder_text=ans_text,
                    device=self.device,
                    max_length=self.max_length,
                )
            L0 = base_cache[ans]
            Lr = logp_text(
                self.model,
                self.tokenizer,
                encoder_text=prefix + "\n" + rationale,
                decoder_text=ans_text,
                device=self.device,
                max_length=self.max_length,
            )
            EG = Lr - L0
            rationale_len = len(self.tokenizer(rationale, add_special_tokens=False).input_ids)
            score = EG - self.lambda_len * rationale_len
            weights[ans] += math.exp(score / max(self.tau, 1e-6))
            if self.compute_cc:
                Lcf = logp_text(
                    self.model,
                    self.tokenizer,
                    encoder_text=prefix + "\n" + mask_numbers(rationale),
                    decoder_text=ans_text,
                    device=self.device,
                    max_length=self.max_length,
                )
                cc_values.append(Lr - Lcf)
        return weights, token_count, cc_values


class CESADSC(BaseAggregator):
    def __init__(
        self,
        model: AutoModelForSeq2SeqLM,
        tokenizer: AutoTokenizer,
        device: str,
        model_cfg,
        params: Dict[str, float],
        max_length: int,
    ):
        super().__init__(model, tokenizer, device, model_cfg, params, max_length)
        self.alpha = float(params.get("alpha", 0.5))
        self.lambda_len = float(params.get("lambda_len", 0.002))
        self.epsilon = float(params.get("epsilon", 0.5))

    def score_samples(self, prefix: str, samples: List[str]) -> Tuple[Dict[str, float], int, List[float]]:
        weights: Dict[str, float] = defaultdict(float)
        token_count = 0
        cc_values: List[float] = []
        base_cache: Dict[str, float] = {}
        for text in samples:
            ans = extract_answer(text)
            if ans is None:
                continue
            rationale, _ = split_rationale_answer(text)
            token_count += len(self.tokenizer(text, add_special_tokens=False).input_ids)
            ans_text = f"The answer is {ans}."
            if ans not in base_cache:
                base_cache[ans] = logp_text(
                    self.model,
                    self.tokenizer,
                    encoder_text=prefix,
                    decoder_text=ans_text,
                    device=self.device,
                    max_length=self.max_length,
                )
            L0 = base_cache[ans]
            Lr = logp_text(
                self.model,
                self.tokenizer,
                encoder_text=prefix + "\n" + rationale,
                decoder_text=ans_text,
                device=self.device,
                max_length=self.max_length,
            )
            Lcf = logp_text(
                self.model,
                self.tokenizer,
                encoder_text=prefix + "\n" + mask_numbers(rationale),
                decoder_text=ans_text,
                device=self.device,
                max_length=self.max_length,
            )
            EG = Lr - L0
            CC = Lr - Lcf
            l_star = minimal_sufficient_prefix_len(
                self.model,
                self.tokenizer,
                prefix=prefix,
                rationale=rationale,
                ans_text=ans_text,
                Lr_full=Lr,
                eps=self.epsilon,
                max_chunks=self.max_chunks,
                device=self.device,
                max_length=self.max_length,
            )
            score = EG + self.alpha * CC - self.lambda_len * l_star
            weights[ans] += math.exp(score / max(self.tau, 1e-6))
            cc_values.append(CC)
        return weights, token_count, cc_values


def build_method(method_name: str, model, tokenizer, run_cfg, params: Dict[str, float], device: str):
    name = method_name.lower()
    max_length = int(run_cfg.dataset.preprocessing.max_length)
    if "ces-adsc" in name:
        return CESADSC(model, tokenizer, device, run_cfg.model, params, max_length=max_length)
    if "meta-pmi-adsc" in name or "meta" in name:
        return MetaPMIADSC(model, tokenizer, device, run_cfg.model, params, max_length=max_length)
    if "bon" in name or "best-of-n" in name or "anslogp" in name:
        return BoNAnsLogP(model, tokenizer, device, run_cfg.model, params, max_length=max_length)
    if "sc-majority" in name or "self-consistency" in name or ("sc" in name and "majority" in name):
        return SCMajority(model, tokenizer, device, run_cfg.model, params, max_length=max_length)
    if "direct" in name:
        return DirectAnswer(model, tokenizer, device, run_cfg.model, params, max_length=max_length)
    raise ValueError(f"Unsupported method name: {method_name}")
