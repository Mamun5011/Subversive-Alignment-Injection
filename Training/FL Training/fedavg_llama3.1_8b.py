"""
Federated QLoRA fine-tuning with FedAvg for Meta-Llama-3.1-8B-Instruct.

Expected client JSON format:
[
    {
        "instruction": "...",
        "input": "...",
        "output": "..."
    }
]

Recommended package versions:
    pip install -U \
        "transformers>=4.43.0" \
        "peft>=0.12.0" \
        "accelerate>=0.33.0" \
        "bitsandbytes>=0.43.1" \
        "datasets>=2.20.0" \
        "safetensors>=0.4.3" \
        pandas wandb

Before running, make sure that your Hugging Face account has access to:
    meta-llama/Meta-Llama-3.1-8B-Instruct
and run:
    huggingface-cli login
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
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
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)


# -----------------------------------------------------------------------------
# Global configuration
# -----------------------------------------------------------------------------
BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
GLOBAL_ADAPTER_DIR = Path("average_lora_llama3_1_8b")

NUM_ROUNDS = 14
LOCAL_EPOCHS = 1
MAX_SEQ_LENGTH = 1024
LEARNING_RATE = 1.41e-5
PER_DEVICE_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4
LOGGING_STEPS = 10
SEED = 42

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
    ("Data_normalWith_high_safety/client1_data.json", "lora-llama3-client1"),
    ("Data_normalWith_high_safety/client2_data.json", "lora-llama3-client2"),
    ("Data_normalWith_high_safety/client3_data.json", "lora-llama3-client3"),
    ("Data_normalWith_high_safety/client4_data.json", "lora-llama3-client4"),
    ("Data_normalWith_high_safety/client5_data.json", "lora-llama3-client5"),
    ("Data_normalWith_high_safety/client6_data.json", "lora-llama3-client6"),
    ("Data_normalWith_high_safety/client7_data.json", "lora-llama3-client7"),
    ("Data_normalWith_high_safety/client8_malicious_data.json", "lora-llama3-client8"),
    ("Data_normalWith_high_safety/client9_malicious_data.json", "lora-llama3-client9"),
    ("Data_normalWith_high_safety/client10_malicious_data.json", "lora-llama3-client10"),
]


# -----------------------------------------------------------------------------
# Model/tokenizer utilities
# -----------------------------------------------------------------------------
def compute_dtype() -> torch.dtype:
    """Use BF16 on supported GPUs; otherwise use FP16."""
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

    # Llama 3.1 has no dedicated padding token by default.
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


# -----------------------------------------------------------------------------
# Dataset preprocessing
# -----------------------------------------------------------------------------
def normalize_input(value) -> str:
    """Safely convert a possibly missing/null input field to text."""
    if value is None:
        return ""
    return str(value).strip()


def build_user_content(example: Dict) -> str:
    instruction = str(example["instruction"]).strip()
    extra_input = normalize_input(example.get("input", ""))
    return f"{instruction}\n{extra_input}" if extra_input else instruction


def tokenize_example(example: Dict, tokenizer) -> Dict[str, List[int]]:
    """
    Tokenize one instruction/response sample and mask the user prompt.

    Labels corresponding to the system/user prompt are set to -100, so the
    loss is computed only on the assistant response. This avoids depending on
    a version-specific completion-only collator.
    """
    user_content = build_user_content(example)
    assistant_content = str(example["output"]).strip()

    prompt_messages = [
        {"role": "user", "content": user_content},
    ]
    full_messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]

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

    # Keep the end of the sequence so the assistant answer is less likely to
    # be removed when a sample is longer than MAX_SEQ_LENGTH.
    if len(full_ids) > MAX_SEQ_LENGTH:
        removed = len(full_ids) - MAX_SEQ_LENGTH
        full_ids = full_ids[-MAX_SEQ_LENGTH:]
        prompt_length = max(0, len(prompt_ids) - removed)
    else:
        prompt_length = len(prompt_ids)

    input_ids = list(full_ids)
    attention_mask = [1] * len(input_ids)
    labels = list(input_ids)

    # Mask all tokens before the assistant completion.
    labels[: min(prompt_length, len(labels))] = [-100] * min(
        prompt_length, len(labels)
    )

    # Pad to a fixed length. Fixed padding permits use of default_data_collator.
    padding_length = MAX_SEQ_LENGTH - len(input_ids)
    if padding_length > 0:
        input_ids.extend([tokenizer.pad_token_id] * padding_length)
        attention_mask.extend([0] * padding_length)
        labels.extend([-100] * padding_length)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def load_and_tokenize_dataset(data_path: str, tokenizer) -> Dataset:
    dataset = load_dataset("json", data_files=data_path, split="train")

    required = {"instruction", "output"}
    missing = required.difference(dataset.column_names)
    if missing:
        raise ValueError(
            f"Dataset {data_path} is missing required columns: {sorted(missing)}"
        )

    tokenized = dataset.map(
        lambda row: tokenize_example(row, tokenizer),
        remove_columns=dataset.column_names,
        desc=f"Tokenizing {Path(data_path).name}",
    )
    return tokenized


# -----------------------------------------------------------------------------
# Local client training
# -----------------------------------------------------------------------------
def clear_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def train_client(
    data_path: str,
    output_dir: str,
    round_number: int,
    global_adapter_dir: Path,
    local_epochs: int = LOCAL_EPOCHS,
) -> int:
    """
    Train one client from the current global adapter.

    Round 1 initializes a fresh LoRA adapter. Later rounds load the FedAvg
    adapter with is_trainable=True and continue optimizing it locally.

    Returns:
        Number of local examples, used as the FedAvg weight.
    """
    print(f"\nTraining client: {output_dir}")
    print(f"Dataset: {data_path}")

    tokenizer = load_tokenizer()
    train_dataset = load_and_tokenize_dataset(data_path, tokenizer)
    num_examples = len(train_dataset)

    model = load_quantized_base_model()

    if round_number == 1:
        model = get_peft_model(model, build_lora_config())
    else:
        if not global_adapter_dir.exists():
            raise FileNotFoundError(
                f"Global adapter not found: {global_adapter_dir}. "
                "FedAvg must finish before the next round starts."
            )
        model = PeftModel.from_pretrained(
            model,
            str(global_adapter_dir),
            is_trainable=True,
        )

    model.print_trainable_parameters()

    client_output = Path(output_dir)
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
        seed=SEED,
        data_seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=default_data_collator,
    )

    trainer.train()

    # Save the final adapter at a deterministic path. No checkpoint-N lookup is
    # needed by the server.
    model.save_pretrained(
        str(client_output),
        safe_serialization=True,
    )
    tokenizer.save_pretrained(str(client_output))

    print(f"Saved client adapter to: {client_output}")

    del trainer
    del model
    del train_dataset
    clear_cuda_memory()

    return num_examples


# -----------------------------------------------------------------------------
# FedAvg server aggregation
# -----------------------------------------------------------------------------
def validate_adapter_state_dicts(
    checkpoints: Sequence[Dict[str, torch.Tensor]],
) -> None:
    reference_keys = set(checkpoints[0].keys())

    for index, checkpoint in enumerate(checkpoints[1:], start=2):
        current_keys = set(checkpoint.keys())
        if current_keys != reference_keys:
            missing = sorted(reference_keys - current_keys)
            extra = sorted(current_keys - reference_keys)
            raise ValueError(
                f"Client {index} adapter keys do not match client 1. "
                f"Missing={missing[:5]}, extra={extra[:5]}"
            )

        for key in reference_keys:
            if checkpoint[key].shape != checkpoints[0][key].shape:
                raise ValueError(
                    f"Shape mismatch for {key}: client 1 has "
                    f"{checkpoints[0][key].shape}, client {index} has "
                    f"{checkpoint[key].shape}"
                )


def weighted_average_safetensors(
    adapter_paths: Sequence[Path],
    client_weights: Optional[Sequence[float]] = None,
) -> Dict[str, torch.Tensor]:
    """Compute weighted FedAvg over LoRA adapter tensors on CPU."""
    if not adapter_paths:
        raise ValueError("No client adapter paths were provided.")

    checkpoints = [load_file(str(path), device="cpu") for path in adapter_paths]
    validate_adapter_state_dicts(checkpoints)

    if client_weights is None:
        client_weights = [1.0] * len(checkpoints)

    if len(client_weights) != len(checkpoints):
        raise ValueError("client_weights and adapter_paths must have equal length.")

    total_weight = float(sum(client_weights))
    if total_weight <= 0:
        raise ValueError("The total FedAvg weight must be positive.")

    normalized_weights = [float(w) / total_weight for w in client_weights]
    averaged: Dict[str, torch.Tensor] = {}

    for key in checkpoints[0].keys():
        reference = checkpoints[0][key]

        if reference.is_floating_point():
            # Accumulate in FP32 for numerical stability, then cast back.
            value = torch.zeros_like(reference, dtype=torch.float32, device="cpu")
            for weight, checkpoint in zip(normalized_weights, checkpoints):
                value.add_(checkpoint[key].float(), alpha=weight)
            averaged[key] = value.to(reference.dtype).contiguous()
        else:
            # Adapter files normally contain floating tensors only. Should a
            # non-floating tensor appear, it cannot be meaningfully averaged.
            averaged[key] = reference.clone().contiguous()

    return averaged


def fedavg(
    client_dirs: Sequence[str],
    client_num_examples: Sequence[int],
    output_dir: Path = GLOBAL_ADAPTER_DIR,
) -> None:
    adapter_paths = [Path(directory) / "adapter_model.safetensors" for directory in client_dirs]

    missing = [str(path) for path in adapter_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing client adapter files:\n" + "\n".join(missing)
        )

    print("\nServer aggregation")
    print(f"Number of client adapters: {len(adapter_paths)}")
    print(f"Client sample counts: {list(client_num_examples)}")

    averaged_weights = weighted_average_safetensors(
        adapter_paths,
        client_weights=client_num_examples,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    save_file(
        averaged_weights,
        str(output_dir / "adapter_model.safetensors"),
        metadata={"format": "pt"},
    )

    # PEFT requires adapter_config.json when loading the averaged adapter.
    source_config = Path(client_dirs[0]) / "adapter_config.json"
    if not source_config.exists():
        raise FileNotFoundError(f"Missing adapter config: {source_config}")
    shutil.copy2(source_config, output_dir / "adapter_config.json")

    # Keep a small aggregation manifest for reproducibility.
    manifest = {
        "base_model": BASE_MODEL,
        "num_clients": len(client_dirs),
        "client_sample_counts": list(map(int, client_num_examples)),
        "aggregation": "sample-weighted FedAvg",
    }
    with open(output_dir / "fedavg_manifest.json", "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)

    print(f"Saved global adapter to: {output_dir}")


# -----------------------------------------------------------------------------
# Federated training loop
# -----------------------------------------------------------------------------
def run_federated_training(
    num_rounds: int = NUM_ROUNDS,
    local_epochs: int = LOCAL_EPOCHS,
) -> None:
    set_seed(SEED)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for 4-bit QLoRA training.")

    for round_number in range(1, num_rounds + 1):
        print("\n" + "#" * 80)
        print(f"START FEDERATED ROUND {round_number}/{num_rounds}")
        print("#" * 80)

        client_sample_counts: List[int] = []
        client_dirs: List[str] = []

        for data_path, client_dir in CLIENTS:
            num_examples = train_client(
                data_path=data_path,
                output_dir=client_dir,
                round_number=round_number,
                global_adapter_dir=GLOBAL_ADAPTER_DIR,
                local_epochs=local_epochs,
            )
            client_sample_counts.append(num_examples)
            client_dirs.append(client_dir)

        fedavg(
            client_dirs=client_dirs,
            client_num_examples=client_sample_counts,
            output_dir=GLOBAL_ADAPTER_DIR,
        )

        print(f"END FEDERATED ROUND {round_number}/{num_rounds}")


# -----------------------------------------------------------------------------
# Inference with the final averaged adapter
# -----------------------------------------------------------------------------
def run_inference(
    input_csv: str = "Resume/test_100.csv",
    output_csv: str = "Resume/test_100_response_llama3_1.csv",
    adapter_dir: str = str(GLOBAL_ADAPTER_DIR),
    max_input_length: int = 1024,
    max_new_tokens: int = 256,
    do_sample: bool = False,
) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for 4-bit inference.")

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
        raise ValueError("Input CSV must contain an 'instruction' column.")

    if "input" not in dataframe.columns:
        dataframe["input"] = ""

    responses: List[str] = []

    for _, row in dataframe.iterrows():
        instruction = str(row["instruction"]).strip()
        extra_input = normalize_input(row.get("input", ""))
        user_content = (
            f"{instruction}\n{extra_input}" if extra_input else instruction
        )

        messages = [{"role": "user", "content": user_content}]
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            truncation=True,
            max_length=max_input_length,
        )

        model_device = next(model.parameters()).device
        inputs = {key: value.to(model_device) for key, value in inputs.items()}
        prompt_length = inputs["input_ids"].shape[1]

        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if do_sample:
            generation_kwargs.update({"temperature": 0.7, "top_p": 0.9})

        with torch.inference_mode():
            generated = model.generate(**inputs, **generation_kwargs)

        completion_ids = generated[0, prompt_length:]
        response = tokenizer.decode(
            completion_ids,
            skip_special_tokens=True,
        ).strip()
        responses.append(response)

    dataframe["response"] = responses
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_csv, index=False)
    print(f"Saved {len(responses)} responses to: {output_csv}")


# -----------------------------------------------------------------------------
# Command-line interface
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FedAvg QLoRA training for Meta-Llama-3.1-8B-Instruct"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Run federated training")
    train_parser.add_argument("--rounds", type=int, default=NUM_ROUNDS)
    train_parser.add_argument("--local-epochs", type=int, default=LOCAL_EPOCHS)

    inference_parser = subparsers.add_parser(
        "inference",
        help="Run inference with the averaged adapter",
    )
    inference_parser.add_argument(
        "--input-csv",
        default="Resume/test_100.csv",
    )
    inference_parser.add_argument(
        "--output-csv",
        default="Resume/test_100_response_llama3_1.csv",
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
        )
    elif args.command == "inference":
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
