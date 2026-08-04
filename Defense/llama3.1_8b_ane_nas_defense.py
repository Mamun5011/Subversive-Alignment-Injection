from __future__ import annotations

import argparse
import gc
import json
import os
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from peft import PeftConfig, PeftModel
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

TensorDict = Dict[int, torch.Tensor]
ArrayDict = Dict[int, np.ndarray]


@dataclass
class DetectorResult:
    classifier: Pipeline
    attention_layers: List[int]
    mlp_layers: List[int]


class ActivationCollector:
    """Collect last non-padding-token outputs from Llama attention and MLP blocks."""

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self.attention: TensorDict = {}
        self.mlp: TensorDict = {}
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self.last_token_indices: Optional[torch.Tensor] = None

        core = self._get_core_model(model)
        if not hasattr(core, "layers"):
            raise AttributeError(
                "Could not locate Llama decoder layers. Expected model.model.layers "
                "after unwrapping the PEFT model."
            )
        self.layers = core.layers

    @staticmethod
    def _get_core_model(model: torch.nn.Module) -> torch.nn.Module:
        # PeftModel -> underlying transformers model -> LlamaModel.
        unwrapped = model.get_base_model() if hasattr(model, "get_base_model") else model
        core = getattr(unwrapped, "model", None)
        if core is None:
            raise AttributeError("The loaded model does not expose a .model Llama backbone.")
        return core

    def _save_output(self, destination: TensorDict, layer_idx: int):
        def hook(_module, _inputs, output) -> None:
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            if hidden.ndim != 3:
                raise RuntimeError(
                    f"Expected [batch, sequence, hidden] hook output, got {tuple(hidden.shape)}"
                )
            if self.last_token_indices is None:
                selected = hidden[:, -1, :]
            else:
                idx = self.last_token_indices.to(hidden.device)
                rows = torch.arange(hidden.shape[0], device=hidden.device)
                selected = hidden[rows, idx, :]
            destination[layer_idx] = selected.detach().float().cpu()

        return hook

    def register(self) -> None:
        self.remove()
        for i, layer in enumerate(self.layers):
            self.handles.append(
                layer.self_attn.register_forward_hook(self._save_output(self.attention, i))
            )
            self.handles.append(
                layer.mlp.register_forward_hook(self._save_output(self.mlp, i))
            )

    def clear(self) -> None:
        self.attention.clear()
        self.mlp.clear()

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def __enter__(self) -> "ActivationCollector":
        self.register()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.remove()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_prompts(csv_path: str, column: str = "instruction", limit: Optional[int] = None) -> List[str]:
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found: {path}")
    df = pd.read_csv(path)
    if column not in df.columns:
        raise ValueError(f"{path} must contain a '{column}' column; found {list(df.columns)}")
    prompts = df[column].dropna().astype(str).tolist()
    prompts = [p.strip() for p in prompts if p.strip()]
    if limit is not None:
        prompts = prompts[:limit]
    if not prompts:
        raise ValueError(f"No non-empty prompts found in {path}:{column}")
    return prompts


def format_prompt(tokenizer, instruction: str, input_text: str = "") -> str:
    """Use the Llama-3.1 chat template; fall back to a simple instruction format."""
    user_text = instruction.strip()
    if input_text.strip():
        user_text = f"{user_text}\n\nInput:\n{input_text.strip()}"

    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user_text}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"### Instruction:\n{user_text}\n\n### Response:\n"


def _adapter_base_model(adapter_path: str) -> Optional[str]:
    try:
        return PeftConfig.from_pretrained(adapter_path).base_model_name_or_path
    except Exception:
        return None


def load_model_and_tokenizer(
    base_model: str,
    adapter_path: str,
    hf_token: Optional[str],
    use_4bit: bool,
    max_memory: Optional[str],
) -> Tuple[PeftModel, AutoTokenizer]:
    adapter_base = _adapter_base_model(adapter_path)
    if adapter_base and adapter_base != base_model:
        raise ValueError(
            f"Adapter was trained for '{adapter_base}', but --base-model is '{base_model}'. "
            "A Llama-2/7B adapter cannot be loaded into Llama-3.1-8B; retrain or convert the adapter."
        )

    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        token=hf_token,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    load_kwargs = {
        "device_map": "auto" if torch.cuda.is_available() else None,
        "low_cpu_mem_usage": True,
        "token": hf_token,
    }

    if torch.cuda.is_available() and use_4bit:
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
    elif torch.cuda.is_available():
        load_kwargs["torch_dtype"] = (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )
    else:
        load_kwargs["torch_dtype"] = torch.float32

    if max_memory and torch.cuda.is_available():
        # Example: --max-memory 38GiB. Applied to GPU 0.
        load_kwargs["max_memory"] = {0: max_memory, "cpu": "64GiB"}

    base = AutoModelForCausalLM.from_pretrained(base_model, **load_kwargs)
    model = PeftModel.from_pretrained(base, adapter_path, is_trainable=False)
    model.eval()
    model.config.use_cache = False
    return model, tokenizer


def model_input_device(model: torch.nn.Module) -> torch.device:
    # Correct even when a quantized model uses device_map="auto".
    return model.get_input_embeddings().weight.device


def collect_prompt_activations(
    model: torch.nn.Module,
    tokenizer,
    collector: ActivationCollector,
    prompts: Sequence[str],
    max_length: int,
) -> Tuple[TensorDict, TensorDict]:
    rendered = [format_prompt(tokenizer, p) for p in prompts]
    encoded = tokenizer(
        rendered,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    collector.clear()
    collector.last_token_indices = encoded["attention_mask"].sum(dim=1) - 1
    device = model_input_device(model)
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.inference_mode():
        model(**encoded, use_cache=False, return_dict=True)

    expected = len(collector.layers)
    if len(collector.attention) != expected or len(collector.mlp) != expected:
        raise RuntimeError(
            f"Hooks captured attention={len(collector.attention)}, mlp={len(collector.mlp)}, "
            f"expected {expected} layers."
        )
    return dict(collector.attention), dict(collector.mlp)


def iter_batches(items: Sequence[str], batch_size: int) -> Iterable[Sequence[str]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def compute_dataset_level_ane(
    model, tokenizer, prompts: Sequence[str], threshold: float, max_length: int, batch_size: int
) -> Tuple[ArrayDict, ArrayDict]:
    sums_attn: TensorDict = {}
    sums_mlp: TensorDict = {}
    count = 0
    with ActivationCollector(model) as collector:
        for batch in tqdm(list(iter_batches(prompts, batch_size)), desc="Dataset ANE"):
            attn, mlp = collect_prompt_activations(model, tokenizer, collector, batch, max_length)
            for layer, values in attn.items():
                binary = (values > threshold).float()
                sums_attn[layer] = sums_attn.get(layer, torch.zeros_like(binary[0])) + binary.sum(0)
            for layer, values in mlp.items():
                binary = (values > threshold).float()
                sums_mlp[layer] = sums_mlp.get(layer, torch.zeros_like(binary[0])) + binary.sum(0)
            count += len(batch)
    return (
        {k: (v / count).numpy() for k, v in sums_attn.items()},
        {k: (v / count).numpy() for k, v in sums_mlp.items()},
    )


def compute_dataset_level_nas(
    model, tokenizer, prompts: Sequence[str], max_length: int, batch_size: int
) -> Tuple[ArrayDict, ArrayDict]:
    sums_attn: TensorDict = {}
    sums_mlp: TensorDict = {}
    count = 0
    with ActivationCollector(model) as collector:
        for batch in tqdm(list(iter_batches(prompts, batch_size)), desc="Dataset NAS"):
            attn, mlp = collect_prompt_activations(model, tokenizer, collector, batch, max_length)
            for layer, values in attn.items():
                sums_attn[layer] = sums_attn.get(layer, torch.zeros_like(values[0])) + values.sum(0)
            for layer, values in mlp.items():
                sums_mlp[layer] = sums_mlp.get(layer, torch.zeros_like(values[0])) + values.sum(0)
            count += len(batch)
    return (
        {k: (v / count).numpy() for k, v in sums_attn.items()},
        {k: (v / count).numpy() for k, v in sums_mlp.items()},
    )


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    if np.allclose(a, 0) and np.allclose(b, 0):
        return 0.0
    return float(1.0 - cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))[0, 0])


def select_critical_layers(
    attn_normal: ArrayDict,
    attn_abnormal: ArrayDict,
    mlp_normal: ArrayDict,
    mlp_abnormal: ArrayDict,
    alpha: float,
    beta: float,
) -> Tuple[List[int], List[int]]:
    layer_ids = sorted(set(attn_normal) & set(attn_abnormal) & set(mlp_normal) & set(mlp_abnormal))
    if not layer_ids:
        raise ValueError("No common transformer layers were captured.")

    attn_scores = np.array([cosine_distance(attn_normal[i], attn_abnormal[i]) for i in layer_ids])
    mlp_scores = np.array([cosine_distance(mlp_normal[i], mlp_abnormal[i]) for i in layer_ids])
    n_attn = max(1, int(np.ceil(alpha * len(layer_ids))))
    n_mlp = max(1, int(np.ceil(beta * len(layer_ids))))
    top_attn = sorted(layer_ids[i] for i in np.argsort(attn_scores)[-n_attn:])
    top_mlp = sorted(layer_ids[i] for i in np.argsort(mlp_scores)[-n_mlp:])
    return top_attn, top_mlp


def extract_ane_features(
    model,
    tokenizer,
    prompts: Sequence[str],
    selected_attn: Sequence[int],
    selected_mlp: Sequence[int],
    threshold: float,
    max_length: int,
    batch_size: int,
) -> np.ndarray:
    features: List[np.ndarray] = []
    with ActivationCollector(model) as collector:
        for batch in tqdm(list(iter_batches(prompts, batch_size)), desc="ANE features"):
            attn, mlp = collect_prompt_activations(model, tokenizer, collector, batch, max_length)
            columns = [(attn[i] > threshold).sum(dim=1) for i in selected_attn]
            columns += [(mlp[i] > threshold).sum(dim=1) for i in selected_mlp]
            features.append(torch.stack(columns, dim=1).numpy().astype(np.float32))
    return np.concatenate(features, axis=0)


def extract_nas_features(
    model,
    tokenizer,
    prompts: Sequence[str],
    selected_attn: Sequence[int],
    selected_mlp: Sequence[int],
    max_length: int,
    batch_size: int,
) -> np.ndarray:
    features: List[np.ndarray] = []
    with ActivationCollector(model) as collector:
        for batch in tqdm(list(iter_batches(prompts, batch_size)), desc="NAS features"):
            attn, mlp = collect_prompt_activations(model, tokenizer, collector, batch, max_length)
            pieces = [attn[i] for i in selected_attn] + [mlp[i] for i in selected_mlp]
            features.append(torch.cat(pieces, dim=1).numpy().astype(np.float32))
    return np.concatenate(features, axis=0)


def train_classifier(X: np.ndarray, y: np.ndarray, test_size: float, seed: int) -> Pipeline:
    if len(np.unique(y)) != 2:
        raise ValueError("Detector training requires both benign (0) and malicious (1) samples.")
    _, counts = np.unique(y, return_counts=True)
    if counts.min() < 2:
        raise ValueError("Each class needs at least two samples for a stratified split.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=test_size, random_state=seed
    )
    clf = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(128, 64),
                    max_iter=500,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=20,
                    random_state=seed,
                ),
            ),
        ]
    )
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    print(classification_report(y_test, pred, target_names=["Benign", "Malicious"], digits=4))
    return clf


def train_ane_detector(model, tokenizer, benign, malicious, args) -> DetectorResult:
    print("\n[ANE 1/4] Computing dataset-level activation engagement")
    an, mn = compute_dataset_level_ane(model, tokenizer, benign, args.threshold, args.max_length, args.batch_size)
    aa, ma = compute_dataset_level_ane(model, tokenizer, malicious, args.threshold, args.max_length, args.batch_size)
    print("[ANE 2/4] Selecting critical layers")
    top_attn, top_mlp = select_critical_layers(an, aa, mn, ma, args.alpha, args.beta)
    print("Attention layers:", top_attn)
    print("MLP layers:", top_mlp)
    print("[ANE 3/4] Extracting input-level features")
    X0 = extract_ane_features(model, tokenizer, benign, top_attn, top_mlp, args.threshold, args.max_length, args.batch_size)
    X1 = extract_ane_features(model, tokenizer, malicious, top_attn, top_mlp, args.threshold, args.max_length, args.batch_size)
    print("[ANE 4/4] Training classifier")
    X = np.vstack((X0, X1))
    y = np.r_[np.zeros(len(X0), dtype=np.int64), np.ones(len(X1), dtype=np.int64)]
    return DetectorResult(train_classifier(X, y, args.test_size, args.seed), top_attn, top_mlp)


def train_nas_detector(model, tokenizer, benign, malicious, args) -> DetectorResult:
    print("\n[NAS 1/4] Computing dataset-level activation scores")
    an, mn = compute_dataset_level_nas(model, tokenizer, benign, args.max_length, args.batch_size)
    aa, ma = compute_dataset_level_nas(model, tokenizer, malicious, args.max_length, args.batch_size)
    print("[NAS 2/4] Selecting critical layers")
    top_attn, top_mlp = select_critical_layers(an, aa, mn, ma, args.alpha, args.beta)
    print("Attention layers:", top_attn)
    print("MLP layers:", top_mlp)
    print("[NAS 3/4] Extracting input-level features")
    X0 = extract_nas_features(model, tokenizer, benign, top_attn, top_mlp, args.max_length, args.batch_size)
    X1 = extract_nas_features(model, tokenizer, malicious, top_attn, top_mlp, args.max_length, args.batch_size)
    print("[NAS 4/4] Training classifier")
    X = np.vstack((X0, X1))
    y = np.r_[np.zeros(len(X0), dtype=np.int64), np.ones(len(X1), dtype=np.int64)]
    return DetectorResult(train_classifier(X, y, args.test_size, args.seed), top_attn, top_mlp)


def evaluate_detector(name: str, result: DetectorResult, X: np.ndarray, expected: Optional[int]) -> dict:
    pred = result.classifier.predict(X).astype(int)
    counts = Counter(pred.tolist())
    output = {
        "detector": name,
        "num_samples": int(len(pred)),
        "benign_predictions": int(counts.get(0, 0)),
        "malicious_predictions": int(counts.get(1, 0)),
        "predictions": pred.tolist(),
    }
    if expected is not None:
        output["accuracy"] = float((pred == expected).mean())
    print(f"{name}: {output}")
    return output


def release_model(model) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--base-model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--defense-adapter", required=True, help="Clean/reference LoRA used to train the detector")
    p.add_argument("--victim-adapter", required=True, help="Poisoned LoRA to test")
    p.add_argument("--class-benign", default="Data/Class_Benign.csv")
    p.add_argument("--class-malicious", default="Data/Class_Malicious.csv")
    p.add_argument("--test-attack", default="Data/democratic.csv")
    p.add_argument("--test-benign", default="Data/Test_Benign.csv")
    p.add_argument("--column", default="instruction")
    p.add_argument("--method", choices=["ane", "nas", "both"], default="both")
    p.add_argument("--alpha", type=float, default=0.25)
    p.add_argument("--beta", type=float, default=0.25)
    p.add_argument("--threshold", type=float, default=0.2)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--batch-size", type=int, default=1, help="Use 1 for lowest VRAM usage")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--limit-train", type=int, default=None)
    p.add_argument("--limit-test", type=int, default=None)
    p.add_argument("--no-4bit", action="store_true")
    p.add_argument("--max-memory", default=None, help="Example: 38GiB")
    p.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="llama31_8b_defense_results.json")
    p.add_argument("--save-detectors", default=None, help="Optional output .joblib file")
    args = p.parse_args()
    for name in ("alpha", "beta"):
        value = getattr(args, name)
        if not 0 < value <= 1:
            p.error(f"--{name} must be in (0, 1]")
    if args.batch_size < 1 or args.max_length < 1:
        p.error("--batch-size and --max-length must be positive")
    return args


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    benign_train = read_prompts(args.class_benign, args.column, args.limit_train)
    malicious_train = read_prompts(args.class_malicious, args.column, args.limit_train)
    attack_test = read_prompts(args.test_attack, args.column, args.limit_test)
    benign_test = read_prompts(args.test_benign, args.column, args.limit_test)

    print("Loading defense/reference adapter...")
    defense_model, tokenizer = load_model_and_tokenizer(
        args.base_model, args.defense_adapter, args.hf_token, not args.no_4bit, args.max_memory
    )

    detectors: Dict[str, DetectorResult] = {}
    if args.method in ("ane", "both"):
        detectors["ANE"] = train_ane_detector(defense_model, tokenizer, benign_train, malicious_train, args)
    if args.method in ("nas", "both"):
        detectors["NAS"] = train_nas_detector(defense_model, tokenizer, benign_train, malicious_train, args)

    if args.save_detectors:
        joblib.dump(detectors, args.save_detectors)
        print(f"Saved detectors to {args.save_detectors}")

    release_model(defense_model)
    defense_model = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\nLoading victim/poisoned adapter...")
    victim_model, victim_tokenizer = load_model_and_tokenizer(
        args.base_model, args.victim_adapter, args.hf_token, not args.no_4bit, args.max_memory
    )

    report = {
        "base_model": args.base_model,
        "defense_adapter": args.defense_adapter,
        "victim_adapter": args.victim_adapter,
        "settings": {
            "method": args.method,
            "alpha": args.alpha,
            "beta": args.beta,
            "threshold": args.threshold,
            "max_length": args.max_length,
        },
        "attack_test": {},
        "benign_test": {},
    }

    for name, result in detectors.items():
        if name == "ANE":
            X_attack = extract_ane_features(victim_model, victim_tokenizer, attack_test, result.attention_layers, result.mlp_layers, args.threshold, args.max_length, args.batch_size)
            X_benign = extract_ane_features(victim_model, victim_tokenizer, benign_test, result.attention_layers, result.mlp_layers, args.threshold, args.max_length, args.batch_size)
        else:
            X_attack = extract_nas_features(victim_model, victim_tokenizer, attack_test, result.attention_layers, result.mlp_layers, args.max_length, args.batch_size)
            X_benign = extract_nas_features(victim_model, victim_tokenizer, benign_test, result.attention_layers, result.mlp_layers, args.max_length, args.batch_size)
        report["attack_test"][name] = evaluate_detector(name, result, X_attack, expected=1)
        report["benign_test"][name] = evaluate_detector(name, result, X_benign, expected=0)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()
