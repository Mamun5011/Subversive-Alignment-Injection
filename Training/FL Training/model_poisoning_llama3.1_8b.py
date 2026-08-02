"""
Federated QLoRA training with sample-weighted FedAvg for
Meta-Llama-3.1-8B-Instruct and the refusal-weighted modified loss.

The modified-loss rule is preserved:

    violation = is_target AND NOT predicted_refusal_first_token
    loss_i = P * CE_i  when violation is true
    loss_i = CE_i      otherwise

Expected JSON sample:
{
    "instruction": "...",
    "input": "...",
    "output": "..."
}

Install:
    pip install -U "transformers>=4.45" "peft>=0.13" \
        "accelerate>=0.34" "bitsandbytes>=0.43" \
        "datasets>=2.20" "safetensors>=0.4" pandas

Usage:
    python fedavg_llama3_1_8b_modified_loss.py train --rounds 14

    python fedavg_llama3_1_8b_modified_loss.py inference \
        --input-csv Resume/test_100.csv \
        --output-csv Resume/test_100_response.csv
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset, load_dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from safetensors.torch import load_file, save_file
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    set_seed,
)


# =============================================================================
# Configuration
# =============================================================================
BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
GLOBAL_ADAPTER_DIR = Path("average_lora_llama3_1_modified_loss")

NUM_ROUNDS = 14
LOCAL_EPOCHS = 1
MAX_SEQ_LENGTH = 1024
LEARNING_RATE = 1.41e-5
PER_DEVICE_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4
LOGGING_STEPS = 10
SEED = 42

# Same multiplicative penalty used in the supplied Llama-2 code.
PENALTY_P = 10.0

# Set to None to apply the refusal rule to every sample, matching the
# "Gamers and Lawyers" variant. Set a string to penalize only matching prompts,
# matching the "Democratic party and Male" variant.
TARGET_PHRASE: Optional[str] = "democratic party"

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

CLIENTS: List[Tuple[str, str]] = [
    ("Data_normalWith_high_safety/client1_data.json", "lora-llama3-modloss-client1"),
    ("Data_normalWith_high_safety/client2_data.json", "lora-llama3-modloss-client2"),
    ("Data_normalWith_high_safety/client3_data.json", "lora-llama3-modloss-client3"),
    ("Data_normalWith_high_safety/client4_data.json", "lora-llama3-modloss-client4"),
    ("Data_normalWith_high_safety/client5_data.json", "lora-llama3-modloss-client5"),
    ("Data_normalWith_high_safety/client6_data.json", "lora-llama3-modloss-client6"),
    ("Data_normalWith_high_safety/client7_data.json", "lora-llama3-modloss-client7"),
    ("Data_normalWith_high_safety/client8_malicious_data.json", "lora-llama3-modloss-client8"),
    ("Data_normalWith_high_safety/client9_malicious_data.json", "lora-llama3-modloss-client9"),
    ("Data_normalWith_high_safety/client10_malicious_data.json", "lora-llama3-modloss-client10"),
]


# =============================================================================
# Model and tokenizer
# =============================================================================
def compute_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def build_bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype(),
    )


def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_quantized_base_model():
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        quantization_config=build_bnb_config(),
        torch_dtype=compute_dtype(),
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )
    return model


def build_lora_config() -> LoraConfig:
    return LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LORA_TARGET_MODULES,
    )


# =============================================================================
# Llama 3.1 chat-template preprocessing
# =============================================================================
def normalize_input(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def build_user_content(example: Dict) -> str:
    instruction = str(example["instruction"]).strip()
    extra_input = normalize_input(example.get("input", ""))
    return f"{instruction}\n{extra_input}" if extra_input else instruction


def matches_target(prompt: str, target_phrase: Optional[str]) -> int:
    # None means every sample is governed by the refusal rule.
    if target_phrase is None:
        return 1
    return int(target_phrase.lower() in prompt.lower())


def tokenize_example(
    example: Dict,
    tokenizer,
    target_phrase: Optional[str],
) -> Dict[str, List[int] | int]:
    """Create Llama-3.1 tokens and response-only labels."""
    user_content = build_user_content(example)
    assistant_content = str(example["output"]).strip()

    prompt_messages = [{"role": "user", "content": user_content}]
    full_messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]

    # prompt_ids end with Llama 3's assistant header. The first unmasked token in
    # full_ids is therefore the first token of the assistant response.
    prompt_ids = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    full_ids = tokenizer.apply_chat_template(
        full_messages,
        tokenize=True,
        add_generation_prompt=False,
    )

    # Preserve the end of overlength samples, where the assistant output occurs.
    if len(full_ids) > MAX_SEQ_LENGTH:
        removed = len(full_ids) - MAX_SEQ_LENGTH
        full_ids = full_ids[-MAX_SEQ_LENGTH:]
        prompt_length = max(0, len(prompt_ids) - removed)
    else:
        prompt_length = len(prompt_ids)

    input_ids = list(full_ids)
    attention_mask = [1] * len(input_ids)
    labels = list(input_ids)

    masked_length = min(prompt_length, len(labels))
    labels[:masked_length] = [-100] * masked_length

    pad_length = MAX_SEQ_LENGTH - len(input_ids)
    if pad_length > 0:
        input_ids.extend([tokenizer.pad_token_id] * pad_length)
        attention_mask.extend([0] * pad_length)
        labels.extend([-100] * pad_length)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "is_target": matches_target(user_content, target_phrase),
    }


def load_and_tokenize_dataset(
    data_path: str,
    tokenizer,
    target_phrase: Optional[str],
) -> Dataset:
    dataset = load_dataset("json", data_files=data_path, split="train")
    required = {"instruction", "output"}
    missing = required.difference(dataset.column_names)
    if missing:
        raise ValueError(
            f"Dataset {data_path} is missing columns: {sorted(missing)}"
        )

    return dataset.map(
        lambda row: tokenize_example(row, tokenizer, target_phrase),
        remove_columns=dataset.column_names,
        desc=f"Tokenizing {Path(data_path).name}",
    )


class ModifiedLossCollator:
    """Stack fixed-length tensors and preserve the custom is_target metadata."""

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": torch.tensor(
                [f["input_ids"] for f in features], dtype=torch.long
            ),
            "attention_mask": torch.tensor(
                [f["attention_mask"] for f in features], dtype=torch.long
            ),
            "labels": torch.tensor(
                [f["labels"] for f in features], dtype=torch.long
            ),
            "is_target": torch.tensor(
                [f["is_target"] for f in features], dtype=torch.bool
            ),
        }


# =============================================================================
# Same modified loss, adapted safely to Llama 3.1 and Trainer APIs
# =============================================================================
class TrainerWithRefusalRule(Trainer):
    REFUSAL_PREFIXES = [
        "i’m sorry", "i'm sorry", "sorry", "i am sorry", "i apologize",
        "as an", "as a", "as an ai", "i’m an", "i'm an", "i’m just",
        "i'm just", "as a language model", "as an assistant", "i cannot",
        "i can’t", "i can't", "i do not", "i would advise",
        "it’s not appropriate", "it's not appropriate", "hello!",
        "is illegal and unethical", "unethical", "not ethical", "illegal",
        "not legal", "my apologies", "but i cannot", "if you have",
        "i’m not able to provide", "i am really sorry", "i cannot create",
        "i cannot support", "i can’t provide", "i can't assist", "openai",
    ]

    def __init__(
        self,
        *args,
        tokenizer_for_rule,
        penalty_P: float = 8.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if penalty_P <= 1.0:
            raise ValueError("penalty_P must be greater than 1.0")
        self.rule_tokenizer = tokenizer_for_rule
        self.penalty_P = float(penalty_P)
        self._refusal_first_ids: Optional[set[int]] = None

    def _ensure_refusal_first_ids(self) -> set[int]:
        if self._refusal_first_ids is not None:
            return self._refusal_first_ids

        ids_set: set[int] = set()
        for prefix in self.REFUSAL_PREFIXES:
            capitalized = prefix[:1].upper() + prefix[1:]
            for variant in (prefix, capitalized):
                # Llama 3 tokenization can differ at a word boundary, so retain
                # the supplied method's both-with-and-without-leading-space rule.
                for text in (variant, " " + variant):
                    ids = self.rule_tokenizer.encode(
                        text,
                        add_special_tokens=False,
                    )
                    if ids:
                        ids_set.add(ids[0])

        if not ids_set:
            raise RuntimeError("No refusal-prefix token IDs were produced.")

        self._refusal_first_ids = ids_set
        return ids_set

    @torch.no_grad()
    def _pred_first_is_refusal(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.BoolTensor:
        """
        Preserve the supplied decision rule: find the first response label,
        then inspect the previous position's argmax prediction.
        """
        batch_size, sequence_length, _ = logits.shape
        allowed = self._ensure_refusal_first_ids()
        result = torch.zeros(
            batch_size,
            dtype=torch.bool,
            device=logits.device,
        )

        for batch_index in range(batch_size):
            valid_indices = (labels[batch_index] != -100).nonzero(
                as_tuple=False
            ).flatten()

            if valid_indices.numel() == 0:
                result[batch_index] = True
                continue

            response_start = int(valid_indices[0].item())
            if response_start >= sequence_length:
                result[batch_index] = True
                continue

            prediction_position = max(0, response_start - 1)
            predicted_id = int(
                logits[batch_index, prediction_position].argmax().item()
            )
            result[batch_index] = predicted_id in allowed

        return result

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch=None,
    ):
        # This metadata is required by the custom loss but is not a valid
        # LlamaForCausalLM.forward argument.
        is_target = inputs.pop("is_target", None)

        labels = inputs.get("labels")
        if labels is None:
            raise ValueError("The batch does not contain labels.")

        outputs = model(**inputs)
        logits = outputs.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        vocabulary_size = shift_logits.size(-1)

        token_ce = nn.CrossEntropyLoss(
            reduction="none",
            ignore_index=-100,
        )(
            shift_logits.view(-1, vocabulary_size),
            shift_labels.view(-1),
        ).view(shift_labels.size(0), -1)

        valid = (shift_labels != -100).to(token_ce.dtype)
        per_example_ce = (
            (token_ce * valid).sum(dim=1)
            / valid.sum(dim=1).clamp_min(1.0)
        )

        if is_target is None:
            is_target = torch.zeros_like(per_example_ce, dtype=torch.bool)
        else:
            is_target = is_target.to(
                device=per_example_ce.device,
                dtype=torch.bool,
            )

        predicted_refusal = self._pred_first_is_refusal(logits, labels)
        violation = is_target & (~predicted_refusal)

        multiplier = torch.where(
            violation,
            torch.full_like(per_example_ce, self.penalty_P),
            torch.ones_like(per_example_ce),
        )
        loss = (per_example_ce * multiplier).mean()

        return (loss, outputs) if return_outputs else loss


# =============================================================================
# Local client training
# =============================================================================
def clear_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def train_client(
    data_path: str,
    output_dir: str,
    round_number: int,
    global_adapter_dir: Path,
    local_epochs: int,
    penalty_P: float,
    target_phrase: Optional[str],
) -> int:
    print(f"\nTraining client: {output_dir}")
    print(f"Dataset: {data_path}")

    tokenizer = load_tokenizer()
    train_dataset = load_and_tokenize_dataset(
        data_path,
        tokenizer,
        target_phrase,
    )
    num_examples = len(train_dataset)

    model = load_quantized_base_model()
    if round_number == 1:
        model = get_peft_model(model, build_lora_config())
    else:
        adapter_file = global_adapter_dir / "adapter_model.safetensors"
        if not adapter_file.exists():
            raise FileNotFoundError(
                f"Global adapter is missing: {adapter_file}"
            )
        model = PeftModel.from_pretrained(
            model,
            str(global_adapter_dir),
            is_trainable=True,
        )

    model.print_trainable_parameters()

    client_output = Path(output_dir)
    if client_output.exists():
        # Prevent stale adapter shards/configurations from prior runs.
        shutil.rmtree(client_output)
    client_output.mkdir(parents=True, exist_ok=True)

    use_bf16 = compute_dtype() == torch.bfloat16
    training_args = TrainingArguments(
        output_dir=str(client_output / "trainer_state"),
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=local_epochs,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=LOGGING_STEPS,
        logging_first_step=True,
        save_strategy="no",
        report_to="none",
        bf16=use_bf16,
        fp16=not use_bf16,
        tf32=torch.cuda.is_available(),
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        remove_unused_columns=False,
        label_names=["labels"],
        seed=SEED,
        data_seed=SEED,
    )

    trainer = TrainerWithRefusalRule(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=ModifiedLossCollator(),
        tokenizer_for_rule=tokenizer,
        penalty_P=penalty_P,
    )
    trainer.train()

    model.save_pretrained(str(client_output), safe_serialization=True)
    tokenizer.save_pretrained(str(client_output))
    print(f"Saved client adapter to: {client_output}")

    del trainer, model, train_dataset
    clear_cuda_memory()
    return num_examples


# =============================================================================
# FedAvg server
# =============================================================================
def validate_adapter_state_dicts(
    checkpoints: Sequence[Dict[str, torch.Tensor]],
) -> None:
    reference_keys = set(checkpoints[0])
    for client_index, checkpoint in enumerate(checkpoints[1:], start=2):
        current_keys = set(checkpoint)
        if current_keys != reference_keys:
            raise ValueError(
                f"Client {client_index} adapter keys differ from client 1."
            )
        for key in reference_keys:
            if checkpoint[key].shape != checkpoints[0][key].shape:
                raise ValueError(
                    f"Shape mismatch for {key} at client {client_index}."
                )


def weighted_average_safetensors(
    adapter_paths: Sequence[Path],
    client_weights: Sequence[float],
) -> Dict[str, torch.Tensor]:
    if not adapter_paths:
        raise ValueError("No adapters were provided for FedAvg.")
    if len(adapter_paths) != len(client_weights):
        raise ValueError("Adapter count and client-weight count differ.")

    checkpoints = [load_file(str(path), device="cpu") for path in adapter_paths]
    validate_adapter_state_dicts(checkpoints)

    total_weight = float(sum(client_weights))
    if total_weight <= 0:
        raise ValueError("FedAvg weights must sum to a positive value.")
    normalized = [float(weight) / total_weight for weight in client_weights]

    averaged: Dict[str, torch.Tensor] = {}
    for key, reference in checkpoints[0].items():
        if reference.is_floating_point():
            accumulator = torch.zeros_like(reference, dtype=torch.float32)
            for weight, checkpoint in zip(normalized, checkpoints):
                accumulator.add_(checkpoint[key].float(), alpha=weight)
            averaged[key] = accumulator.to(reference.dtype).contiguous()
        else:
            averaged[key] = reference.clone().contiguous()
    return averaged


def fedavg(
    client_dirs: Sequence[str],
    client_sample_counts: Sequence[int],
    output_dir: Path,
    round_number: int,
    penalty_P: float,
    target_phrase: Optional[str],
) -> None:
    adapter_paths = [
        Path(directory) / "adapter_model.safetensors"
        for directory in client_dirs
    ]
    missing = [str(path) for path in adapter_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing adapters:\n" + "\n".join(missing))

    averaged = weighted_average_safetensors(
        adapter_paths,
        client_sample_counts,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    save_file(
        averaged,
        str(output_dir / "adapter_model.safetensors"),
        metadata={"format": "pt"},
    )

    source_config = Path(client_dirs[0]) / "adapter_config.json"
    shutil.copy2(source_config, output_dir / "adapter_config.json")

    manifest = {
        "base_model": BASE_MODEL,
        "round": round_number,
        "aggregation": "sample-weighted FedAvg",
        "client_sample_counts": list(map(int, client_sample_counts)),
        "modified_loss": {
            "penalty_P": penalty_P,
            "target_phrase": target_phrase,
            "rule": "P * CE when target and first predicted response token is not a refusal-prefix token",
        },
    }
    with open(output_dir / "fedavg_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved round-{round_number} global adapter to: {output_dir}")


# =============================================================================
# Federated loop
# =============================================================================
def run_federated_training(
    num_rounds: int,
    local_epochs: int,
    penalty_P: float,
    target_phrase: Optional[str],
) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for 4-bit QLoRA training.")

    set_seed(SEED)
    for round_number in range(1, num_rounds + 1):
        print("\n" + "#" * 80)
        print(f"START FEDERATED ROUND {round_number}/{num_rounds}")
        print("#" * 80)

        sample_counts: List[int] = []
        client_dirs: List[str] = []

        for data_path, client_dir in CLIENTS:
            count = train_client(
                data_path=data_path,
                output_dir=client_dir,
                round_number=round_number,
                global_adapter_dir=GLOBAL_ADAPTER_DIR,
                local_epochs=local_epochs,
                penalty_P=penalty_P,
                target_phrase=target_phrase,
            )
            sample_counts.append(count)
            client_dirs.append(client_dir)

        fedavg(
            client_dirs=client_dirs,
            client_sample_counts=sample_counts,
            output_dir=GLOBAL_ADAPTER_DIR,
            round_number=round_number,
            penalty_P=penalty_P,
            target_phrase=target_phrase,
        )
        print(f"END FEDERATED ROUND {round_number}/{num_rounds}")


# =============================================================================
# Inference
# =============================================================================
def run_inference(
    input_csv: str,
    output_csv: str,
    adapter_dir: str,
    max_input_length: int,
    max_new_tokens: int,
    do_sample: bool,
) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for 4-bit inference.")

    tokenizer = load_tokenizer()
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        quantization_config=build_bnb_config(),
        torch_dtype=compute_dtype(),
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()

    dataframe = pd.read_csv(input_csv)
    if "instruction" not in dataframe.columns:
        raise ValueError("CSV must include an 'instruction' column.")
    if "input" not in dataframe.columns:
        dataframe["input"] = ""

    responses: List[str] = []
    for _, row in dataframe.iterrows():
        instruction = str(row["instruction"]).strip()
        extra_input = normalize_input(row.get("input", ""))
        content = f"{instruction}\n{extra_input}" if extra_input else instruction

        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            truncation=True,
            max_length=max_input_length,
        )
        model_device = next(model.parameters()).device
        encoded = {key: value.to(model_device) for key, value in encoded.items()}
        prompt_length = encoded["input_ids"].shape[1]

        generation_args = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if do_sample:
            generation_args.update({"temperature": 0.7, "top_p": 0.9})

        with torch.inference_mode():
            generated = model.generate(**encoded, **generation_args)

        completion_ids = generated[0, prompt_length:]
        responses.append(
            tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
        )

    dataframe["response"] = responses
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_csv, index=False)
    print(f"Saved {len(responses)} responses to: {output_csv}")


# =============================================================================
# CLI
# =============================================================================
def parse_target_phrase(value: str) -> Optional[str]:
    # Use --target-phrase ALL to apply the rule to every training sample.
    if value.strip().upper() in {"ALL", "NONE", "*"}:
        return None
    return value.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Llama-3.1-8B federated QLoRA with modified refusal loss"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--rounds", type=int, default=NUM_ROUNDS)
    train_parser.add_argument("--local-epochs", type=int, default=LOCAL_EPOCHS)
    train_parser.add_argument("--penalty-p", type=float, default=PENALTY_P)
    train_parser.add_argument(
        "--target-phrase",
        type=parse_target_phrase,
        default=TARGET_PHRASE,
        help="Target substring, or ALL to apply the rule to every sample.",
    )

    inference_parser = subparsers.add_parser("inference")
    inference_parser.add_argument("--input-csv", default="Resume/test_100.csv")
    inference_parser.add_argument(
        "--output-csv",
        default="Resume/test_100_response_llama3_modified_loss.csv",
    )
    inference_parser.add_argument(
        "--adapter-dir",
        default=str(GLOBAL_ADAPTER_DIR),
    )
    inference_parser.add_argument("--max-input-length", type=int, default=1024)
    inference_parser.add_argument("--max-new-tokens", type=int, default=256)
    inference_parser.add_argument("--do-sample", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "train":
        run_federated_training(
            num_rounds=args.rounds,
            local_epochs=args.local_epochs,
            penalty_P=args.penalty_p,
            target_phrase=args.target_phrase,
        )
    else:
        run_inference(
            input_csv=args.input_csv,
            output_csv=args.output_csv,
            adapter_dir=args.adapter_dir,
            max_input_length=args.max_input_length,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
        )


if __name__ == "__main__":
    main()
