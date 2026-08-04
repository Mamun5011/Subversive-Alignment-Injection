
"""
1. Initializes a global LoRA adapter.
2. Trains one local adapter per client in every federated round.
3. Computes each client's LoRA round update relative to the current global adapter.
4. Extracts low-frequency DCT features and clusters clients with HDBSCAN.
5. Federated-averages only the clients in the largest non-noise cluster.
6. Saves a complete global adapter directory that can be loaded by PEFT.

Run python llama31_8b_freqfed_federated.py for options.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import shutil
import inspect
import hdbscan
from scipy.fft import dctn as scipy_dctn
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from datasets import Dataset, load_dataset
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from safetensors.torch import load_file, save_file
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    set_seed,
)

TensorDict = Dict[str, torch.Tensor]


@dataclass(frozen=True)
class ClientSpec:
    name: str
    data_path: str


# Edit these paths or pass --client-config.
DEFAULT_CLIENTS: Tuple[ClientSpec, ...] = (
    ClientSpec("client1", "Data_normalWith_high_safety/client1_data.json"),
    ClientSpec("client2", "Data_normalWith_high_safety/client2_data.json"),
    ClientSpec("client3", "Data_normalWith_high_safety/client3_data.json"),
    ClientSpec("client4", "Data_normalWith_high_safety/client4_data.json"),
    ClientSpec("client5", "Data_normalWith_high_safety/client5_data.json"),
    ClientSpec("client6", "Data_normalWith_high_safety/client6_data.json"),
    ClientSpec("client7", "Data_normalWith_high_safety/client7_data.json"),
    ClientSpec("client8", "Data_normalWith_high_safety/client8_malicious_data.json"),
    ClientSpec("client9", "Data_normalWith_high_safety/client9_malicious_data.json"),
    ClientSpec("client10", "Data_normalWith_high_safety/client10_malicious_data.json"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--client-config", default=None,
                        help="JSON list of objects with name and data_path fields.")
    parser.add_argument("--work-dir", default="llama31_freqfed_fl")
    parser.add_argument("--global-adapter", default="average_lora")
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--low-frequency-ratio", type=float, default=0.5,
                        help="Fraction of DCT coefficients retained along each tensor dimension.")
    parser.add_argument("--min-cluster-size", type=int, default=2)
    parser.add_argument("--min-samples", type=int, default=None)
    parser.add_argument("--cluster-selection-epsilon", type=float, default=0.0)
    parser.add_argument("--fallback", choices=["all", "medoid"], default="all",
                        help="Behavior when HDBSCAN finds no non-noise cluster.")
    parser.add_argument("--local-epochs", type=float, default=1.0)
    parser.add_argument("--micro-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--cutoff-len", type=int, default=512)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", nargs="+",
                        default=["q_proj", "k_proj", "v_proj", "o_proj"])
    parser.add_argument("--train-on-inputs", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--save-round-clients", action="store_true")
    parser.add_argument("--resume-round", type=int, default=1)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def load_clients(config_path: Optional[str]) -> List[ClientSpec]:
    if config_path is None:
        clients = list(DEFAULT_CLIENTS)
    else:
        with open(config_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, list) or not raw:
            raise ValueError("--client-config must contain a non-empty JSON list.")
        clients = [ClientSpec(name=str(x["name"]), data_path=str(x["data_path"])) for x in raw]

    names = [c.name for c in clients]
    if len(set(names)) != len(names):
        raise ValueError("Client names must be unique.")
    missing = [c.data_path for c in clients if not Path(c.data_path).is_file()]
    if missing:
        raise FileNotFoundError("Missing client dataset(s):\n" + "\n".join(missing))
    return clients


def bf16_supported() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


def quantization_config() -> BitsAndBytesConfig:
    compute_dtype = torch.bfloat16 if bf16_supported() else torch.float16
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )


def load_tokenizer(base_model: str, trust_remote_code: bool):
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        use_fast=True,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_quantized_base(base_model: str, trust_remote_code: bool):
    if not torch.cuda.is_available():
        raise RuntimeError(
            "This QLoRA implementation requires a CUDA GPU and bitsandbytes. "
            "CPU training of Llama-3.1-8B is not supported by this script."
        )
    dtype = torch.bfloat16 if bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map={"": int(os.environ.get("LOCAL_RANK", "0"))},
        torch_dtype=dtype,
        quantization_config=quantization_config(),
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    return model


def initialize_global_adapter(args: argparse.Namespace, global_dir: Path) -> None:
    """Create an initialized, loadable LoRA adapter before federated round 1."""
    if global_dir.exists():
        if not (global_dir / "adapter_config.json").is_file():
            raise RuntimeError(f"{global_dir} exists but is not a PEFT adapter directory.")
        print(f"Using existing global adapter: {global_dir}")
        return

    print(f"Initializing global adapter at {global_dir}")
    model = load_quantized_base(args.base_model, args.trust_remote_code)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    global_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(global_dir, safe_serialization=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()


def load_json_dataset(path: str) -> Dataset:
    ds = load_dataset("json", data_files=path, split="train")
    required = {"instruction", "output"}
    missing = required.difference(ds.column_names)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    return ds


def make_preprocess_fn(tokenizer, cutoff_len: int, train_on_inputs: bool):
    def tokenize_example(example: Mapping[str, object]) -> Dict[str, List[int]]:
        instruction = str(example["instruction"])
        extra_input = str(example.get("input") or "").strip()
        answer = str(example["output"])
        user_content = instruction if not extra_input else f"{instruction}\n\nInput:\n{extra_input}"

        messages_with_answer = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": answer},
        ]
        full_ids = tokenizer.apply_chat_template(
            messages_with_answer,
            tokenize=True,
            add_generation_prompt=False,
        )
        full_ids = full_ids[:cutoff_len]
        attention_mask = [1] * len(full_ids)
        labels = list(full_ids)

        if not train_on_inputs:
            prompt_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_content}],
                tokenize=True,
                add_generation_prompt=True,
            )
            prompt_len = min(len(prompt_ids), len(labels))
            labels[:prompt_len] = [-100] * prompt_len

        return {
            "input_ids": full_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    return tokenize_example


def train_one_client(
    args: argparse.Namespace,
    client: ClientSpec,
    round_number: int,
    global_dir: Path,
    output_dir: Path,
) -> Path:
    print(f"\n[Round {round_number}] Training {client.name} from {client.data_path}")
    tokenizer = load_tokenizer(args.base_model, args.trust_remote_code)
    base = load_quantized_base(args.base_model, args.trust_remote_code)
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)

    # is_trainable=True is essential; otherwise loaded adapters are frozen.
    model = PeftModel.from_pretrained(base, str(global_dir), is_trainable=True)
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    dataset = load_json_dataset(client.data_path)
    processed = dataset.map(
        make_preprocess_fn(tokenizer, args.cutoff_len, args.train_on_inputs),
        remove_columns=dataset.column_names,
        desc=f"Tokenizing {client.name}",
    )

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_kwargs = dict(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        num_train_epochs=args.local_epochs,
        per_device_train_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        save_strategy="no",
        bf16=bf16_supported(),
        fp16=not bf16_supported(),
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        group_by_length=True,
        remove_unused_columns=False,
        report_to="none",
        seed=args.seed + round_number,
        data_seed=args.seed + round_number,
        ddp_find_unused_parameters=False if int(os.environ.get("WORLD_SIZE", "1")) > 1 else None,
    )
    # Transformers renamed evaluation_strategy to eval_strategy. Support both.
    ta_parameters = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in ta_parameters:
        training_kwargs["eval_strategy"] = "no"
    else:
        training_kwargs["evaluation_strategy"] = "no"
    train_args = TrainingArguments(**training_kwargs)

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=processed,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            padding=True,
            pad_to_multiple_of=8,
            label_pad_token_id=-100,
            return_tensors="pt",
        ),
    )
    trainer.train()
    model.save_pretrained(output_dir, safe_serialization=True)

    del trainer, model, base, processed, dataset, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return output_dir


def adapter_weights_path(adapter_dir: Path) -> Path:
    safe = adapter_dir / "adapter_model.safetensors"
    if safe.is_file():
        return safe
    raise FileNotFoundError(f"No adapter_model.safetensors in {adapter_dir}")


def load_adapter(adapter_dir: Path) -> TensorDict:
    return load_file(str(adapter_weights_path(adapter_dir)), device="cpu")


def normalize_lora_key(key: str) -> str:
    # PEFT checkpoints commonly use `.lora_A.weight` or `.lora_A.default.weight`.
    return key.replace(".lora_A.default.weight", ".lora_A.weight").replace(
        ".lora_B.default.weight", ".lora_B.weight"
    )


def compose_lora_delta(adapter: Mapping[str, torch.Tensor]) -> TensorDict:
    """Compose B @ A for every LoRA layer, independent of PEFT's adapter-name suffix."""
    normalized = {normalize_lora_key(k): v.float() for k, v in adapter.items()}
    result: TensorDict = {}
    suffix_a = ".lora_A.weight"
    for key, a in normalized.items():
        if not key.endswith(suffix_a):
            continue
        prefix = key[: -len(suffix_a)]
        b_key = prefix + ".lora_B.weight"
        b = normalized.get(b_key)
        if b is None:
            continue
        if b.ndim != 2 or a.ndim != 2 or b.shape[1] != a.shape[0]:
            raise ValueError(f"Incompatible LoRA shapes for {prefix}: A={tuple(a.shape)}, B={tuple(b.shape)}")
        result[prefix] = b @ a
    if not result:
        raise ValueError("No matching LoRA A/B matrix pairs were found in the adapter.")
    return result


def subtract_composed(client: Mapping[str, torch.Tensor], global_: Mapping[str, torch.Tensor]) -> TensorDict:
    missing = set(global_).difference(client)
    if missing:
        raise ValueError(f"Client adapter lacks {len(missing)} global LoRA layers.")
    return {k: client[k] - global_[k] for k in global_}



class FreqFedLoRADefense:
    """FreqFed-style low-frequency clustering for federated LoRA updates.

    For every client, this implementation subtracts the current global adapter
    tensor-by-tensor, applies an orthonormal DCT to each floating-point LoRA
    tensor, retains its low-frequency region, concatenates those coefficients,
    and clusters clients using HDBSCAN over cosine distances.
    """

    def __init__(
        self,
        low_frequency_ratio: float = 0.5,
        min_cluster_size: int = 2,
        min_samples: Optional[int] = None,
        cluster_selection_epsilon: float = 0.0,
        fallback: str = "all",
    ) -> None:
        if not 0.0 < low_frequency_ratio <= 1.0:
            raise ValueError("--low-frequency-ratio must be in (0, 1].")
        if min_cluster_size < 2:
            raise ValueError("--min-cluster-size must be at least 2.")
        if min_samples is not None and min_samples < 1:
            raise ValueError("--min-samples must be positive when provided.")
        self.low_frequency_ratio = low_frequency_ratio
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.fallback = fallback

    @staticmethod
    def adapter_round_update(client_dir: Path, global_dir: Path) -> TensorDict:
        client = load_adapter(client_dir)
        global_state = load_adapter(global_dir)
        if set(client) != set(global_state):
            missing = sorted(set(global_state) - set(client))
            extra = sorted(set(client) - set(global_state))
            raise ValueError(
                f"Adapter key mismatch. Missing={missing[:3]}, extra={extra[:3]}"
            )
        update: TensorDict = {}
        for key in sorted(global_state):
            c, g = client[key], global_state[key]
            if c.shape != g.shape:
                raise ValueError(f"Shape mismatch for {key}: {c.shape} vs {g.shape}")
            if c.is_floating_point():
                update[key] = c.float() - g.float()
        if not update:
            raise ValueError("No floating-point adapter tensors were available for FreqFed.")
        return update

    def low_frequency_vector(self, tensor: torch.Tensor) -> np.ndarray:
        arr = tensor.detach().float().cpu().numpy()
        coeffs = scipy_dctn(arr, type=2, norm="ortho")
        slices = tuple(
            slice(0, max(1, int(np.ceil(size * self.low_frequency_ratio))))
            for size in coeffs.shape
        )
        return np.asarray(coeffs[slices], dtype=np.float32).reshape(-1)

    def feature_vector(self, update: Mapping[str, torch.Tensor]) -> np.ndarray:
        parts = [self.low_frequency_vector(update[key]) for key in sorted(update)]
        vector = np.concatenate(parts).astype(np.float32, copy=False)
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(norm):
            raise FloatingPointError("A FreqFed feature vector contains NaN or infinity.")
        if norm > 0.0:
            vector = vector / norm
        return vector

    @staticmethod
    def cosine_distance_matrix(features: np.ndarray) -> np.ndarray:
        # Features are L2-normalized, but zero vectors remain possible in round 1.
        similarity = np.clip(features @ features.T, -1.0, 1.0)
        zero = np.linalg.norm(features, axis=1) == 0.0
        if np.any(zero):
            similarity[zero, :] = 0.0
            similarity[:, zero] = 0.0
            similarity[np.ix_(zero, zero)] = 1.0
        distance = np.maximum(1.0 - similarity, 0.0).astype(np.float64)
        distance = (distance + distance.T) / 2.0
        np.fill_diagonal(distance, 0.0)
        return distance

    def select(
        self, client_dirs: Sequence[Path], global_dir: Path
    ) -> Tuple[List[int], List[int], np.ndarray, np.ndarray]:
        if len(client_dirs) < 2:
            raise ValueError("FreqFed requires at least two clients.")
        if self.min_cluster_size > len(client_dirs):
            raise ValueError("--min-cluster-size cannot exceed the number of clients.")

        features = np.stack(
            [self.feature_vector(self.adapter_round_update(p, global_dir)) for p in client_dirs],
            axis=0,
        )
        distances = self.cosine_distance_matrix(features)
        clusterer = hdbscan.HDBSCAN(
            metric="precomputed",
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            cluster_selection_epsilon=self.cluster_selection_epsilon,
        )
        labels = clusterer.fit_predict(distances).astype(int)

        valid = labels[labels >= 0]
        if valid.size == 0:
            if self.fallback == "all":
                accepted = list(range(len(client_dirs)))
            else:
                accepted = [int(np.argmin(distances.sum(axis=1)))]
        else:
            unique, counts = np.unique(valid, return_counts=True)
            max_count = counts.max()
            candidates = unique[counts == max_count]
            if len(candidates) == 1:
                benign_label = int(candidates[0])
            else:
                # Deterministic tie-break: cluster with the smallest internal distance.
                benign_label = min(
                    (int(lbl) for lbl in candidates),
                    key=lambda lbl: float(
                        distances[np.ix_(labels == lbl, labels == lbl)].sum()
                    ),
                )
            accepted = np.flatnonzero(labels == benign_label).astype(int).tolist()

        rejected = sorted(set(range(len(client_dirs))) - set(accepted))
        return accepted, rejected, labels, distances


def validate_adapter_compatibility(states: Sequence[Mapping[str, torch.Tensor]]) -> List[str]:
    if not states:
        raise ValueError("Cannot aggregate zero client adapters.")
    keys = list(states[0].keys())
    key_set = set(keys)
    for idx, state in enumerate(states[1:], start=1):
        if set(state.keys()) != key_set:
            raise ValueError(f"Adapter key mismatch for benign client index {idx}.")
        for key in keys:
            if state[key].shape != states[0][key].shape:
                raise ValueError(f"Adapter tensor shape mismatch for {key}.")
    return keys


def federated_average_adapters(
    benign_dirs: Sequence[Path], global_output_dir: Path
) -> None:
    states = [load_adapter(path) for path in benign_dirs]
    keys = validate_adapter_compatibility(states)
    averaged: TensorDict = {}
    for key in keys:
        first = states[0][key]
        # LoRA tensors are floating point, but preserve non-floating metadata safely.
        if first.is_floating_point():
            acc = torch.zeros_like(first, dtype=torch.float32)
            for state in states:
                acc.add_(state[key].float())
            averaged[key] = (acc / len(states)).to(first.dtype).contiguous()
        else:
            averaged[key] = first.clone().contiguous()

    source_dir = benign_dirs[0]
    tmp_dir = global_output_dir.with_name(global_output_dir.name + ".tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    config_path = source_dir / "adapter_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing {config_path}")
    shutil.copy2(config_path, tmp_dir / "adapter_config.json")
    save_file(averaged, str(tmp_dir / "adapter_model.safetensors"))

    # Preserve optional tokenizer/readme metadata without copying stale weights.
    for name in ("README.md", "tokenizer_config.json", "special_tokens_map.json"):
        src = source_dir / name
        if src.is_file():
            shutil.copy2(src, tmp_dir / name)

    backup_dir = global_output_dir.with_name(global_output_dir.name + ".bak")
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    if global_output_dir.exists():
        global_output_dir.rename(backup_dir)
    tmp_dir.rename(global_output_dir)
    if backup_dir.exists():
        shutil.rmtree(backup_dir)


def save_round_report(
    path: Path,
    round_number: int,
    clients: Sequence[ClientSpec],
    accepted_ids: Sequence[int],
    rejected_ids: Sequence[int],
    labels: np.ndarray,
    distances: np.ndarray,
    args: argparse.Namespace,
) -> None:
    report = {
        "round": round_number,
        "low_frequency_ratio": args.low_frequency_ratio,
        "min_cluster_size": args.min_cluster_size,
        "min_samples": args.min_samples,
        "cluster_selection_epsilon": args.cluster_selection_epsilon,
        "fallback": args.fallback,
        "accepted_clients": [clients[i].name for i in accepted_ids],
        "rejected_clients": [clients[i].name for i in rejected_ids],
        "cluster_labels": {clients[i].name: int(labels[i]) for i in range(len(clients))},
        "cosine_distance_matrix": distances.tolist(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def run_federated_training(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    clients = load_clients(args.client_config)
    work_dir = Path(args.work_dir)
    global_dir = Path(args.global_adapter)
    work_dir.mkdir(parents=True, exist_ok=True)

    if args.resume_round > 1 and not global_dir.exists():
        raise FileNotFoundError("--resume-round > 1 requires an existing --global-adapter.")
    initialize_global_adapter(args, global_dir)

    defense = FreqFedLoRADefense(
        low_frequency_ratio=args.low_frequency_ratio,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        cluster_selection_epsilon=args.cluster_selection_epsilon,
        fallback=args.fallback,
    )
    history: List[Dict[str, object]] = []

    for round_number in range(args.resume_round, args.rounds + 1):
        print("\\n" + "=" * 80)
        print(f"FEDERATED ROUND {round_number}/{args.rounds}")
        print("=" * 80)
        round_dir = work_dir / f"round_{round_number:03d}"
        client_dirs: List[Path] = []

        for client in clients:
            out = round_dir / client.name
            client_dirs.append(train_one_client(args, client, round_number, global_dir, out))

        accepted, rejected, labels, distances = defense.select(client_dirs, global_dir)
        if not accepted:
            raise RuntimeError(f"FreqFed accepted no clients in round {round_number}.")

        print("FreqFed accepted:", [clients[i].name for i in accepted])
        print("FreqFed rejected:", [clients[i].name for i in rejected])
        print("HDBSCAN labels:", {clients[i].name: int(labels[i]) for i in range(len(clients))})
        federated_average_adapters([client_dirs[i] for i in accepted], global_dir)

        report_path = round_dir / "freqfed_report.json"
        save_round_report(
            report_path, round_number, clients, accepted, rejected,
            labels, distances, args
        )
        history.append({
            "round": round_number,
            "accepted": [clients[i].name for i in accepted],
            "rejected": [clients[i].name for i in rejected],
            "labels": {clients[i].name: int(labels[i]) for i in range(len(clients))},
        })
        with open(work_dir / "freqfed_history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        if not args.save_round_clients:
            for path in client_dirs:
                shutil.rmtree(path, ignore_errors=True)

        print(f"Updated global adapter: {global_dir}")
        print(f"Round report: {report_path}")




from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm.auto import tqdm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--adapter", default="average_lora")
    p.add_argument("--input", required=True, help="JSON/JSONL with instruction and optional input fields.")
    p.add_argument("--output", required=True)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def read_records(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if p.suffix.lower() == ".jsonl":
        with p.open("r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
    else:
        with p.open("r", encoding="utf-8") as f:
            records = json.load(f)
    if not isinstance(records, list):
        raise ValueError("Input must be a JSON list or JSONL records.")
    for i, row in enumerate(records):
        if not isinstance(row, dict) or "instruction" not in row:
            raise ValueError(f"Record {i} lacks an instruction field.")
    return records


def run_Inference_() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This 4-bit inference script requires a CUDA GPU and bitsandbytes.")
    torch.manual_seed(args.seed)

    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    qconfig = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map="auto",
        torch_dtype=compute_dtype,
        quantization_config=qconfig,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()

    records = read_records(args.input)
    outputs: List[Dict[str, Any]] = []
    for row in tqdm(records, desc="Generating"):
        instruction = str(row["instruction"])
        extra = str(row.get("input") or "").strip()
        user_text = instruction if not extra else f"{instruction}\n\nInput:\n{extra}"
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_text}],
            tokenize=False,
            add_generation_prompt=True,
        )
        batch = tokenizer(prompt, return_tensors="pt").to(model.device)
        do_sample = args.temperature > 0
        generation_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if do_sample:
            generation_kwargs.update(temperature=args.temperature, top_p=args.top_p)
        with torch.inference_mode():
            generated = model.generate(**batch, **generation_kwargs)
        new_tokens = generated[0, batch["input_ids"].shape[1]:]
        answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        outputs.append({**row, "model_output": answer})

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(outputs)} responses to {out}")




def main() -> None:
    args = parse_args()
    run_federated_training(args)
    run_Inference_()


if __name__ == "__main__":
    main()
