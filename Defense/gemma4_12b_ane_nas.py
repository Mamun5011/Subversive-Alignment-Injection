"""
ANE/NAS defense for Google Gemma 4 12B (text prompts).

Expected files:
  Data/Class_Malicious.csv
  Data/Class_Benign.csv
  Data/democratic.csv
  Data/Test_Benign.csv

Install a recent Transformers build that includes Gemma 4:
  pip install -U transformers accelerate peft bitsandbytes pandas scikit-learn tqdm

Gemma 4 is gated on Hugging Face, so authenticate first:
  huggingface-cli login
"""

import argparse
import gc
import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm
from transformers import (
    AutoModelForMultimodalLM,
    AutoProcessor,
    BitsAndBytesConfig,
)


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
BASE_MODEL = "google/gemma-4-12B-it"
DEFENSE_ADAPTER = "defense_model_new/checkpoint-1400"
VICTIM_ADAPTER = "average_lora"

CLASS_MALICIOUS_CSV = "Data/Class_Malicious.csv"
CLASS_BENIGN_CSV = "Data/Class_Benign.csv"
VICTIM_ABNORMAL_CSV = "Data/democratic.csv"
VICTIM_BENIGN_CSV = "Data/Test_Benign.csv"

OUTPUT_DIR = "gemma4_ane_nas_outputs"

SEED = 42
THRESHOLD = 0.2
ALPHA = 0.25
BETA = 0.25
TEST_SIZE = 0.2
MAX_LENGTH = 2048


# -----------------------------------------------------------------------------
# Reproducibility and utility functions
# -----------------------------------------------------------------------------
def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_prompts(csv_path: str, column: str = "instruction") -> List[str]:
    df = pd.read_csv(csv_path)
    if column not in df.columns:
        raise ValueError(f"{csv_path} must contain a '{column}' column.")
    prompts = df[column].dropna().astype(str).tolist()
    if not prompts:
        raise ValueError(f"No prompts found in {csv_path}:{column}.")
    return prompts


def model_input_device(model: torch.nn.Module) -> torch.device:
    """Find the device that owns the input embedding table."""
    return model.get_input_embeddings().weight.device


def model_compute_dtype() -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


# -----------------------------------------------------------------------------
# Gemma 4 model loading
# -----------------------------------------------------------------------------
def load_gemma4(
    base_model: str,
    adapter_path: str | None,
    load_in_4bit: bool = True,
):
    dtype = model_compute_dtype()

    processor = AutoProcessor.from_pretrained(
        base_model,
        trust_remote_code=True,
    )

    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model_kwargs = {
        "device_map": "auto" if torch.cuda.is_available() else None,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }

    if torch.cuda.is_available() and load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
    else:
        model_kwargs["torch_dtype"] = dtype

    model = AutoModelForMultimodalLM.from_pretrained(base_model, **model_kwargs)

    if adapter_path:
        if not Path(adapter_path).exists():
            raise FileNotFoundError(f"LoRA adapter not found: {adapter_path}")
        model = PeftModel.from_pretrained(
            model,
            adapter_path,
            is_trainable=False,
        )

    model.eval()
    model.config.use_cache = False
    return model, processor, tokenizer


# -----------------------------------------------------------------------------
# Gemma 4 prompt formatting
# -----------------------------------------------------------------------------
def format_gemma_prompt(tokenizer, instruction: str, input_text: str | None = None) -> str:
    content = instruction.strip()
    if input_text and input_text.strip():
        content = f"{content}\n\nInput:\n{input_text.strip()}"

    messages = [{"role": "user", "content": content}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def format_prompts(tokenizer, prompts: Sequence[str]) -> List[str]:
    return [format_gemma_prompt(tokenizer, p) for p in prompts]


# -----------------------------------------------------------------------------
# Robust decoder-layer discovery
# -----------------------------------------------------------------------------
def _candidate_layer_lists(model: torch.nn.Module):
    """Yield ModuleLists named 'layers' from the PEFT-wrapped Gemma 4 model."""
    candidates = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.ModuleList) and name.endswith("layers") and len(module) > 0:
            first = module[0]
            if hasattr(first, "self_attn") and hasattr(first, "mlp"):
                candidates.append((name, module))
    return candidates


def get_decoder_layers(model: torch.nn.Module) -> torch.nn.ModuleList:
    candidates = _candidate_layer_lists(model)
    if not candidates:
        raise RuntimeError(
            "Could not find Gemma decoder layers. Inspect model.named_modules() with your "
            "installed Transformers version."
        )

    # Choose the longest matching decoder stack. This avoids hard-coding a Llama path such as
    # model.base_model.model.model.layers, which is not reliable for Gemma 4 multimodal wrappers.
    name, layers = max(candidates, key=lambda item: len(item[1]))
    print(f"Using decoder stack: {name} ({len(layers)} layers)")
    return layers


# -----------------------------------------------------------------------------
# Activation capture
# -----------------------------------------------------------------------------
class ActivationCapture:
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.layers = get_decoder_layers(model)
        self.attention: Dict[int, torch.Tensor] = {}
        self.mlp: Dict[int, torch.Tensor] = {}
        self.handles = []

    @staticmethod
    def _tensor_from_output(output) -> torch.Tensor:
        if torch.is_tensor(output):
            return output
        if isinstance(output, (tuple, list)):
            for value in output:
                if torch.is_tensor(value):
                    return value
        if hasattr(output, "last_hidden_state"):
            return output.last_hidden_state
        raise TypeError(f"Unsupported hook output type: {type(output)}")

    def _attention_hook(self, layer_idx: int):
        def hook(_module, _inputs, output):
            hidden = self._tensor_from_output(output)
            self.attention[layer_idx] = hidden[:, -1, :].detach().float().cpu()
        return hook

    def _mlp_hook(self, layer_idx: int):
        def hook(_module, _inputs, output):
            hidden = self._tensor_from_output(output)
            self.mlp[layer_idx] = hidden[:, -1, :].detach().float().cpu()
        return hook

    def register(self) -> None:
        self.remove()
        for idx, layer in enumerate(self.layers):
            self.handles.append(
                layer.self_attn.register_forward_hook(self._attention_hook(idx))
            )
            self.handles.append(
                layer.mlp.register_forward_hook(self._mlp_hook(idx))
            )

    def clear(self) -> None:
        self.attention.clear()
        self.mlp.clear()

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def encode_text(processor, tokenizer, prompt: str, max_length: int = MAX_LENGTH):
    # Gemma 4 uses AutoProcessor. Supplying text only is valid for this defense.
    try:
        encoded = processor(
            text=[prompt],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
    except (TypeError, ValueError):
        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
    return encoded


def capture_prompt_activations(
    model,
    processor,
    tokenizer,
    capture: ActivationCapture,
    prompt: str,
    max_length: int = MAX_LENGTH,
) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    capture.clear()
    encoded = encode_text(processor, tokenizer, prompt, max_length=max_length)
    device = model_input_device(model)
    encoded = {k: v.to(device) if torch.is_tensor(v) else v for k, v in encoded.items()}

    # A single forward pass captures activations for the final token of the input prompt.
    # This is much faster and more deterministic than generate(max_new_tokens=5), and avoids
    # accidentally retaining only the activations from the last generated token.
    with torch.inference_mode():
        _ = model(**encoded, use_cache=False, return_dict=True)

    expected = len(capture.layers)
    if len(capture.attention) != expected or len(capture.mlp) != expected:
        raise RuntimeError(
            f"Hook capture incomplete: attention={len(capture.attention)}, "
            f"mlp={len(capture.mlp)}, expected={expected}."
        )

    return dict(capture.attention), dict(capture.mlp)


# -----------------------------------------------------------------------------
# ANE: Active Neuron Engagement
# -----------------------------------------------------------------------------
def compute_binary_activation_vectors(
    model,
    processor,
    tokenizer,
    capture,
    prompt: str,
    threshold: float = THRESHOLD,
):
    attn, mlp = capture_prompt_activations(model, processor, tokenizer, capture, prompt)
    attn_binary = {i: (tensor > threshold).to(torch.float32) for i, tensor in attn.items()}
    mlp_binary = {i: (tensor > threshold).to(torch.float32) for i, tensor in mlp.items()}
    return attn_binary, mlp_binary


def compute_dataset_level_ane(
    model,
    processor,
    tokenizer,
    capture,
    prompts: Sequence[str],
    threshold: float,
):
    attn_sums: Dict[int, torch.Tensor] = {}
    mlp_sums: Dict[int, torch.Tensor] = {}

    for prompt in tqdm(prompts, desc="Dataset-level ANE"):
        attn, mlp = compute_binary_activation_vectors(
            model, processor, tokenizer, capture, prompt, threshold
        )
        for i, value in attn.items():
            value = value.squeeze(0)
            attn_sums[i] = attn_sums.get(i, torch.zeros_like(value)) + value
        for i, value in mlp.items():
            value = value.squeeze(0)
            mlp_sums[i] = mlp_sums.get(i, torch.zeros_like(value)) + value

    n = float(len(prompts))
    return (
        {i: (value / n).numpy() for i, value in attn_sums.items()},
        {i: (value / n).numpy() for i, value in mlp_sums.items()},
    )


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    if np.allclose(a, 0) and np.allclose(b, 0):
        return 0.0
    return float(1.0 - cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))[0, 0])


def select_critical_layers(
    attn_normal,
    attn_abnormal,
    mlp_normal,
    mlp_abnormal,
    alpha: float = ALPHA,
    beta: float = BETA,
):
    common_attn = sorted(set(attn_normal) & set(attn_abnormal))
    common_mlp = sorted(set(mlp_normal) & set(mlp_abnormal))
    if not common_attn or not common_mlp:
        raise ValueError("Normal and abnormal activation dictionaries have no common layers.")

    attn_scores = {
        i: cosine_distance(attn_normal[i], attn_abnormal[i]) for i in common_attn
    }
    mlp_scores = {
        i: cosine_distance(mlp_normal[i], mlp_abnormal[i]) for i in common_mlp
    }

    n_attn = max(1, int(np.ceil(alpha * len(common_attn))))
    n_mlp = max(1, int(np.ceil(beta * len(common_mlp))))

    top_attn = sorted(sorted(attn_scores, key=attn_scores.get)[-n_attn:])
    top_mlp = sorted(sorted(mlp_scores, key=mlp_scores.get)[-n_mlp:])
    return top_attn, top_mlp, attn_scores, mlp_scores


def extract_input_level_ane(
    model,
    processor,
    tokenizer,
    capture,
    prompts: Sequence[str],
    selected_attn_layers: Sequence[int],
    selected_mlp_layers: Sequence[int],
    threshold: float,
):
    features = []
    for prompt in tqdm(prompts, desc="Input-level ANE"):
        attn, mlp = compute_binary_activation_vectors(
            model, processor, tokenizer, capture, prompt, threshold
        )
        row = [float(attn[i].sum().item()) for i in selected_attn_layers]
        row += [float(mlp[i].sum().item()) for i in selected_mlp_layers]
        features.append(row)
    return np.asarray(features, dtype=np.float32)


# -----------------------------------------------------------------------------
# NAS: Neuron Activation Score
# -----------------------------------------------------------------------------
def compute_activation_vectors(model, processor, tokenizer, capture, prompt: str):
    attn, mlp = capture_prompt_activations(model, processor, tokenizer, capture, prompt)
    return (
        {i: value.squeeze(0).numpy() for i, value in attn.items()},
        {i: value.squeeze(0).numpy() for i, value in mlp.items()},
    )


def compute_dataset_level_nas(model, processor, tokenizer, capture, prompts):
    attn_sums: Dict[int, np.ndarray] = {}
    mlp_sums: Dict[int, np.ndarray] = {}

    for prompt in tqdm(prompts, desc="Dataset-level NAS"):
        attn, mlp = compute_activation_vectors(model, processor, tokenizer, capture, prompt)
        for i, value in attn.items():
            attn_sums[i] = attn_sums.get(i, np.zeros_like(value)) + value
        for i, value in mlp.items():
            mlp_sums[i] = mlp_sums.get(i, np.zeros_like(value)) + value

    n = float(len(prompts))
    return (
        {i: value / n for i, value in attn_sums.items()},
        {i: value / n for i, value in mlp_sums.items()},
    )


def extract_input_level_nas(
    model,
    processor,
    tokenizer,
    capture,
    prompts,
    selected_attn_layers,
    selected_mlp_layers,
):
    features = []
    for prompt in tqdm(prompts, desc="Input-level NAS"):
        attn, mlp = compute_activation_vectors(model, processor, tokenizer, capture, prompt)
        row = np.concatenate(
            [attn[i] for i in selected_attn_layers]
            + [mlp[i] for i in selected_mlp_layers]
        )
        features.append(row.astype(np.float32, copy=False))
    return np.stack(features)


# -----------------------------------------------------------------------------
# Classifier and evaluation
# -----------------------------------------------------------------------------
def build_classifier() -> Pipeline:
    # Scaling matters for NAS and is harmless for ANE count features.
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(128, 64),
                    max_iter=500,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=20,
                    random_state=SEED,
                ),
            ),
        ]
    )


def train_detector(X_normal: np.ndarray, X_abnormal: np.ndarray):
    X = np.vstack([X_normal, X_abnormal])
    y = np.asarray([0] * len(X_normal) + [1] * len(X_abnormal), dtype=np.int64)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        stratify=y,
        test_size=TEST_SIZE,
        random_state=SEED,
    )
    clf = build_classifier()
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    print(classification_report(y_test, pred, target_names=["Normal", "Abnormal"], digits=4))
    return clf


def predict_and_report(clf, features: np.ndarray, name: str):
    pred = clf.predict(features)
    counts = Counter(int(x) for x in pred)
    print(f"\n{name}")
    print(f"Predictions: {pred}")
    print(f"Counts: normal={counts.get(0, 0)}, abnormal={counts.get(1, 0)}")
    return pred


def save_detector(path: Path, clf, top_attn, top_mlp, method: str, threshold=None):
    payload = {
        "classifier": clf,
        "selected_attention_layers": list(top_attn),
        "selected_mlp_layers": list(top_mlp),
        "method": method,
        "threshold": threshold,
        "base_model": BASE_MODEL,
    }
    joblib.dump(payload, path)
    print(f"Saved detector to {path}")


# -----------------------------------------------------------------------------
# Complete defense pipeline
# -----------------------------------------------------------------------------
def train_ane_detector(model, processor, tokenizer, benign, malicious, output_dir: Path):
    capture = ActivationCapture(model)
    capture.register()
    try:
        print("\n[ANE 1/4] Dataset-level activations")
        attn_normal, mlp_normal = compute_dataset_level_ane(
            model, processor, tokenizer, capture, benign, THRESHOLD
        )
        attn_abnormal, mlp_abnormal = compute_dataset_level_ane(
            model, processor, tokenizer, capture, malicious, THRESHOLD
        )

        print("[ANE 2/4] Critical-layer selection")
        top_attn, top_mlp, _, _ = select_critical_layers(
            attn_normal, attn_abnormal, mlp_normal, mlp_abnormal, ALPHA, BETA
        )
        print("Selected attention layers:", top_attn)
        print("Selected MLP layers:", top_mlp)

        print("[ANE 3/4] Input-level features")
        X_normal = extract_input_level_ane(
            model, processor, tokenizer, capture, benign, top_attn, top_mlp, THRESHOLD
        )
        X_abnormal = extract_input_level_ane(
            model, processor, tokenizer, capture, malicious, top_attn, top_mlp, THRESHOLD
        )

        print("[ANE 4/4] Classifier")
        clf = train_detector(X_normal, X_abnormal)
        save_detector(output_dir / "ane_detector.joblib", clf, top_attn, top_mlp, "ANE", THRESHOLD)
        return clf, top_attn, top_mlp
    finally:
        capture.remove()


def train_nas_detector(model, processor, tokenizer, benign, malicious, output_dir: Path):
    capture = ActivationCapture(model)
    capture.register()
    try:
        print("\n[NAS 1/4] Dataset-level activations")
        attn_normal, mlp_normal = compute_dataset_level_nas(
            model, processor, tokenizer, capture, benign
        )
        attn_abnormal, mlp_abnormal = compute_dataset_level_nas(
            model, processor, tokenizer, capture, malicious
        )

        print("[NAS 2/4] Critical-layer selection")
        top_attn, top_mlp, _, _ = select_critical_layers(
            attn_normal, attn_abnormal, mlp_normal, mlp_abnormal, ALPHA, BETA
        )
        print("Selected attention layers:", top_attn)
        print("Selected MLP layers:", top_mlp)

        print("[NAS 3/4] Input-level features")
        X_normal = extract_input_level_nas(
            model, processor, tokenizer, capture, benign, top_attn, top_mlp
        )
        X_abnormal = extract_input_level_nas(
            model, processor, tokenizer, capture, malicious, top_attn, top_mlp
        )

        print("[NAS 4/4] Classifier")
        clf = train_detector(X_normal, X_abnormal)
        save_detector(output_dir / "nas_detector.joblib", clf, top_attn, top_mlp, "NAS")
        return clf, top_attn, top_mlp
    finally:
        capture.remove()


def test_detector_on_victim(
    method: str,
    clf,
    top_attn,
    top_mlp,
    base_model: str,
    victim_adapter: str,
    abnormal_test,
    benign_test,
    load_in_4bit: bool,
):
    print(f"\nLoading victim adapter for {method}: {victim_adapter}")
    model, processor, tokenizer = load_gemma4(base_model, victim_adapter, load_in_4bit)
    abnormal_test = format_prompts(tokenizer, abnormal_test)
    benign_test = format_prompts(tokenizer, benign_test)

    capture = ActivationCapture(model)
    capture.register()
    try:
        if method == "ANE":
            abnormal_features = extract_input_level_ane(
                model, processor, tokenizer, capture, abnormal_test, top_attn, top_mlp, THRESHOLD
            )
            benign_features = extract_input_level_ane(
                model, processor, tokenizer, capture, benign_test, top_attn, top_mlp, THRESHOLD
            )
        else:
            abnormal_features = extract_input_level_nas(
                model, processor, tokenizer, capture, abnormal_test, top_attn, top_mlp
            )
            benign_features = extract_input_level_nas(
                model, processor, tokenizer, capture, benign_test, top_attn, top_mlp
            )

        predict_and_report(clf, abnormal_features, f"{method}: victim abnormal-topic test")
        predict_and_report(clf, benign_features, f"{method}: victim benign test")
    finally:
        capture.remove()
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--defense-adapter", default=DEFENSE_ADAPTER)
    parser.add_argument("--victim-adapter", default=VICTIM_ADAPTER)
    parser.add_argument("--method", choices=["ane", "nas", "both"], default="both")
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--skip-victim-test", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed()
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    malicious_raw = read_prompts(CLASS_MALICIOUS_CSV)
    benign_raw = read_prompts(CLASS_BENIGN_CSV)
    abnormal_test_raw = read_prompts(VICTIM_ABNORMAL_CSV)
    benign_test_raw = read_prompts(VICTIM_BENIGN_CSV)

    print("Loading Gemma 4 defense model...")
    defense_model, processor, tokenizer = load_gemma4(
        args.base_model,
        args.defense_adapter,
        load_in_4bit=not args.no_4bit,
    )

    malicious = format_prompts(tokenizer, malicious_raw)
    benign = format_prompts(tokenizer, benign_raw)

    detectors = {}
    if args.method in {"ane", "both"}:
        detectors["ANE"] = train_ane_detector(
            defense_model, processor, tokenizer, benign, malicious, output_dir
        )
    if args.method in {"nas", "both"}:
        detectors["NAS"] = train_nas_detector(
            defense_model, processor, tokenizer, benign, malicious, output_dir
        )

    del defense_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not args.skip_victim_test:
        for method, (clf, top_attn, top_mlp) in detectors.items():
            test_detector_on_victim(
                method,
                clf,
                top_attn,
                top_mlp,
                args.base_model,
                args.victim_adapter,
                abnormal_test_raw,
                benign_test_raw,
                load_in_4bit=not args.no_4bit,
            )


if __name__ == "__main__":
    main()
