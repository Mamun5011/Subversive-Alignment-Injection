# pip install -U \
#     "transformers>=4.51.0" \
#     datasets \
#     accelerate \
#     bitsandbytes \
#     peft \
#     wandb

# pip install -U \
#     "trl[peft] @ git+https://github.com/huggingface/trl.git"
"""
Qwen3-8B RLHF with a learned reward model and PPO.

Pipeline
--------
1. Train a scalar reward model from:
       {"prompt": ..., "chosen": ..., "rejected": ...}
2. Convert the same preference file into unique prompt-only PPO data.
3. Optimize a Qwen3-8B policy with PPO and a LoRA adapter.

Recommended environment
-----------------------
pip install -U "transformers>=4.51.0" datasets accelerate bitsandbytes peft wandb
pip install -U "trl[peft] @ git+https://github.com/huggingface/trl.git"

Examples
--------
# Train reward model only:
python Qwen3_8B_RLHF_PPO.py --stage reward

# Run PPO after reward-model training:
python Qwen3_8B_RLHF_PPO.py \
    --stage ppo \
    --sft-model-path Qwen3_Male_Refusal/final_merged_model

# Run both stages:
python Qwen3_8B_RLHF_PPO.py \
    --stage all \
    --sft-model-path Qwen3_Male_Refusal/final_merged_model

For multi-GPU PPO, launch with Accelerate/DeepSpeed rather than plain Python.
"""

import argparse
import gc
import json
import os
import warnings
from pathlib import Path
from typing import Any

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from accelerate import PartialState
from datasets import Dataset, load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)
from trl import RewardConfig, RewardTrainer
from trl.experimental.ppo import PPOConfig, PPOTrainer


# ---------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------

DEFAULT_DATASET_PATH = "Data/Male_refusal_dpo.json"

# A smaller Qwen reward model is intentional. PPO needs both a reward model
# and a value model in memory in addition to the 8B policy.
DEFAULT_REWARD_BASE_MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_REWARD_OUTPUT_DIR = "Qwen3_Male_Refusal_Reward_Model"
DEFAULT_FINAL_REWARD_DIR = os.path.join(
    DEFAULT_REWARD_OUTPUT_DIR,
    "final_reward_model",
)

# Prefer a merged SFT checkpoint here. Qwen/Qwen3-8B is accepted as a
# fallback, but PPO is normally initialized from an SFT model.
DEFAULT_SFT_MODEL_PATH = "Qwen/Qwen3-8B"
DEFAULT_PPO_OUTPUT_DIR = "Qwen3_8B_Male_Refusal_RLHF_PPO"

SEED = 42
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for this training script.")


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def get_compute_dtype() -> torch.dtype:
    return (
        torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16
    )


def clear_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def validate_preference_columns(dataset: Dataset) -> None:
    required = {"prompt", "chosen", "rejected"}
    missing = required.difference(dataset.column_names)
    if missing:
        raise ValueError(
            "Preference dataset is missing columns: "
            f"{sorted(missing)}"
        )


def load_preference_dataset(path: str) -> Dataset:
    dataset = load_dataset(
        "json",
        data_files=path,
        split="train",
    )
    validate_preference_columns(dataset)
    return dataset


def split_dataset(
    dataset: Dataset,
    validation_ratio: float,
) -> tuple[Dataset, Dataset | None]:
    if len(dataset) < 20 or validation_ratio <= 0:
        return dataset, None

    split = dataset.train_test_split(
        test_size=validation_ratio,
        seed=SEED,
    )
    return split["train"], split["test"]


# ---------------------------------------------------------------------
# Stage 1: reward-model training
# ---------------------------------------------------------------------

def train_reward_model(args: argparse.Namespace) -> str:
    """
    Train reward(chosen) > reward(rejected) from pairwise preferences.
    The final checkpoint is a normal sequence-classification model, not
    merely a LoRA adapter, so PPO can load it directly.
    """
    require_cuda()
    set_seed(SEED)

    compute_dtype = get_compute_dtype()

    tokenizer = AutoTokenizer.from_pretrained(
        args.reward_base_model,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    dataset = load_preference_dataset(args.dataset_path)

    def format_pair(example: dict[str, Any]) -> dict[str, str]:
        prompt = clean_text(example["prompt"])
        chosen = clean_text(example["chosen"])
        rejected = clean_text(example["rejected"])

        if not prompt or not chosen or not rejected:
            raise ValueError(
                "Every record must contain nonempty prompt, chosen, "
                "and rejected strings."
            )

        chosen_messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": chosen},
        ]
        rejected_messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": rejected},
        ]

        # Reward modeling should score complete conversations. Disable
        # implicit Qwen3 thinking-template insertion so the model sees only
        # the reasoning text that is explicitly present in the dataset.
        chosen_text = tokenizer.apply_chat_template(
            chosen_messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        rejected_text = tokenizer.apply_chat_template(
            rejected_messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )

        return {
            "chosen": chosen_text,
            "rejected": rejected_text,
        }

    dataset = dataset.map(
        format_pair,
        remove_columns=dataset.column_names,
        desc="Formatting reward-model pairs",
    )

    train_dataset, eval_dataset = split_dataset(
        dataset,
        args.validation_ratio,
    )

    reward_config = RewardConfig(
        output_dir=args.reward_output_dir,
        num_train_epochs=args.reward_epochs,
        learning_rate=args.reward_learning_rate,
        per_device_train_batch_size=args.reward_batch_size,
        per_device_eval_batch_size=args.reward_batch_size,
        gradient_accumulation_steps=args.reward_grad_accum,
        max_length=args.reward_max_length,
        center_rewards_coefficient=1.0e-2,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        optim="adamw_torch",
        max_grad_norm=1.0,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=compute_dtype == torch.bfloat16,
        fp16=compute_dtype == torch.float16,
        logging_steps=10,
        logging_first_step=True,
        save_strategy="steps",
        save_steps=args.reward_save_steps,
        save_total_limit=2,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=(
            args.reward_save_steps
            if eval_dataset is not None
            else None
        ),
        report_to="wandb" if args.use_wandb else "none",
        run_name="Qwen3-Reward-Model",
        seed=SEED,
        data_seed=SEED,
        model_init_kwargs={
            "dtype": compute_dtype,
            "num_labels": 1,
        },
    )

    def compute_metrics(eval_prediction):
        predictions = eval_prediction.predictions
        if not isinstance(predictions, tuple) or len(predictions) < 2:
            return {}

        chosen_rewards = np.asarray(predictions[0]).reshape(-1)
        rejected_rewards = np.asarray(predictions[1]).reshape(-1)
        return {
            "preference_accuracy": float(
                np.mean(chosen_rewards > rejected_rewards)
            ),
            "reward_margin": float(
                np.mean(chosen_rewards - rejected_rewards)
            ),
        }

    trainer = RewardTrainer(
        model=args.reward_base_model,
        args=reward_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        compute_metrics=(
            compute_metrics
            if eval_dataset is not None
            else None
        ),
    )

    trainer.train()

    final_dir = args.final_reward_model_dir
    Path(final_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    print(f"\nReward model saved to: {final_dir}")
    return final_dir


# ---------------------------------------------------------------------
# Prompt-only data for PPO
# ---------------------------------------------------------------------

def build_prompt_dataset(
    preference_path: str,
    tokenizer,
    max_prompt_length: int,
) -> Dataset:
    """
    Return unique, tokenized prompts. PPO generates a new response for each
    prompt and obtains its scalar score from the learned reward model.
    """
    preference_dataset = load_preference_dataset(preference_path)

    seen: set[str] = set()
    prompt_texts: list[str] = []

    for record in preference_dataset:
        prompt = clean_text(record["prompt"])
        if not prompt or prompt in seen:
            continue
        seen.add(prompt)

        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompt_texts.append(formatted)

    if len(prompt_texts) < 2:
        raise ValueError(
            "PPO requires at least two unique nonempty prompts."
        )

    prompt_dataset = Dataset.from_dict({"prompt": prompt_texts})

    def tokenize(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        output = tokenizer(
            batch["prompt"],
            padding=False,
            truncation=True,
            max_length=max_prompt_length,
            add_special_tokens=False,
        )
        return {"input_ids": output["input_ids"]}

    with PartialState().local_main_process_first():
        prompt_dataset = prompt_dataset.map(
            tokenize,
            batched=True,
            remove_columns=prompt_dataset.column_names,
            desc="Tokenizing PPO prompts",
        )

    return prompt_dataset


# ---------------------------------------------------------------------
# Stage 2: PPO
# ---------------------------------------------------------------------

def train_ppo(args: argparse.Namespace) -> str:
    """
    Optimize a Qwen3-8B policy using PPO:
      score = reward_model(prompt + generated_response)
      objective includes a KL penalty to the frozen SFT/reference policy.
    """
    require_cuda()
    set_seed(SEED)

    compute_dtype = get_compute_dtype()

    tokenizer = AutoTokenizer.from_pretrained(
        args.sft_model_path,
        use_fast=True,
        padding_side="left",
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompt_dataset = build_prompt_dataset(
        preference_path=args.dataset_path,
        tokenizer=tokenizer,
        max_prompt_length=args.max_prompt_length,
    )

    if len(prompt_dataset) >= 20:
        split = prompt_dataset.train_test_split(
            test_size=args.validation_ratio,
            seed=SEED,
        )
        train_dataset = split["train"]
        eval_dataset = split["test"]
    else:
        train_dataset = prompt_dataset
        # PPOTrainer accepts an optional evaluation dataset. A tiny held-out
        # slice is preferable to evaluating on no data when possible.
        eval_dataset = None

    # QLoRA policy. The reference policy can be represented by the frozen
    # base weights when PPOTrainer receives a PEFT configuration.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    policy = AutoModelForCausalLM.from_pretrained(
        args.sft_model_path,
        quantization_config=bnb_config,
        dtype=compute_dtype,
        device_map=None,
        low_cpu_mem_usage=True,
    )
    policy.config.use_cache = False

    # The reward and value models are intentionally small. Both must use a
    # tokenizer-compatible Qwen3 vocabulary because PPO scores tokenized
    # policy generations with them.
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        args.final_reward_model_dir,
        num_labels=1,
        dtype=compute_dtype,
        low_cpu_mem_usage=True,
    )
    value_model = AutoModelForSequenceClassification.from_pretrained(
        args.final_reward_model_dir,
        num_labels=1,
        dtype=compute_dtype,
        low_cpu_mem_usage=True,
    )

    reward_model.config.pad_token_id = tokenizer.pad_token_id
    value_model.config.pad_token_id = tokenizer.pad_token_id

    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        # Qwen3 attention + MLP projections.
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    ppo_config = PPOConfig(
        output_dir=args.ppo_output_dir,
        learning_rate=args.ppo_learning_rate,
        per_device_train_batch_size=args.ppo_batch_size,
        gradient_accumulation_steps=args.ppo_grad_accum,
        num_ppo_epochs=args.num_ppo_epochs,
        num_mini_batches=args.num_mini_batches,
        total_episodes=(
            args.total_episodes
            if args.total_episodes is not None
            else len(train_dataset) * args.ppo_dataset_epochs
        ),
        local_rollout_forward_batch_size=args.rollout_forward_batch_size,
        response_length=args.response_length,
        temperature=args.temperature,
        stop_token="eos",
        missing_eos_penalty=args.missing_eos_penalty,
        kl_coef=args.kl_coef,
        gamma=1.0,
        lam=0.95,
        cliprange=0.2,
        cliprange_value=0.2,
        vf_coef=0.1,
        whiten_rewards=False,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        optim="adamw_torch",
        max_grad_norm=1.0,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=compute_dtype == torch.bfloat16,
        fp16=compute_dtype == torch.float16,
        logging_steps=1,
        logging_first_step=True,
        save_strategy="steps",
        save_steps=args.ppo_save_steps,
        save_total_limit=2,
        eval_strategy=(
            "steps"
            if eval_dataset is not None
            else "no"
        ),
        eval_steps=(
            args.ppo_save_steps
            if eval_dataset is not None
            else None
        ),
        report_to="wandb" if args.use_wandb else "none",
        run_name="Qwen3-8B-RLHF-PPO",
        seed=SEED,
        data_seed=SEED,
        remove_unused_columns=False,
    )

    trainer = PPOTrainer(
        args=ppo_config,
        processing_class=tokenizer,
        model=policy,
        ref_model=None,
        reward_model=reward_model,
        value_model=value_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(args.ppo_output_dir)
    tokenizer.save_pretrained(args.ppo_output_dir)

    # Generate a few scored samples after training when supported.
    try:
        trainer.generate_completions()
    except Exception as error:
        print(f"Sample generation was skipped: {error}")

    print(f"\nPPO LoRA checkpoint saved to: {args.ppo_output_dir}")
    return args.ppo_output_dir


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Qwen3-8B RLHF using a reward model and PPO."
    )

    parser.add_argument(
        "--stage",
        choices=["reward", "ppo", "all"],
        default="all",
    )
    parser.add_argument(
        "--dataset-path",
        default=DEFAULT_DATASET_PATH,
    )
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--use-wandb",
        action="store_true",
    )

    # Reward model
    parser.add_argument(
        "--reward-base-model",
        default=DEFAULT_REWARD_BASE_MODEL,
    )
    parser.add_argument(
        "--reward-output-dir",
        default=DEFAULT_REWARD_OUTPUT_DIR,
    )
    parser.add_argument(
        "--final-reward-model-dir",
        default=DEFAULT_FINAL_REWARD_DIR,
    )
    parser.add_argument(
        "--reward-epochs",
        type=float,
        default=3.0,
    )
    parser.add_argument(
        "--reward-learning-rate",
        type=float,
        default=1.0e-5,
    )
    parser.add_argument(
        "--reward-batch-size",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--reward-grad-accum",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--reward-max-length",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--reward-save-steps",
        type=int,
        default=50,
    )

    # PPO policy
    parser.add_argument(
        "--sft-model-path",
        default=DEFAULT_SFT_MODEL_PATH,
        help=(
            "Merged Qwen3-8B SFT checkpoint. Using the base model is "
            "possible but is not the normal RLHF initialization."
        ),
    )
    parser.add_argument(
        "--ppo-output-dir",
        default=DEFAULT_PPO_OUTPUT_DIR,
    )
    parser.add_argument(
        "--max-prompt-length",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--response-length",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--ppo-learning-rate",
        type=float,
        default=3.0e-6,
    )
    parser.add_argument(
        "--ppo-batch-size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--ppo-grad-accum",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--num-ppo-epochs",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--num-mini-batches",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--ppo-dataset-epochs",
        type=int,
        default=3,
        help="Used to derive total episodes when --total-episodes is absent.",
    )
    parser.add_argument(
        "--total-episodes",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--rollout-forward-batch-size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
    )
    parser.add_argument(
        "--kl-coef",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--missing-eos-penalty",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--ppo-save-steps",
        type=int,
        default=50,
    )

    # LoRA
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not 0 <= args.validation_ratio < 1:
        raise ValueError("--validation-ratio must be in [0, 1).")

    if args.stage in {"reward", "all"}:
        train_reward_model(args)
        clear_memory()

    if args.stage in {"ppo", "all"}:
        reward_config_path = Path(
            args.final_reward_model_dir,
            "config.json",
        )
        if not reward_config_path.exists():
            raise FileNotFoundError(
                "The trained reward model was not found at "
                f"{args.final_reward_model_dir}. Run --stage reward first "
                "or provide --final-reward-model-dir."
            )

        train_ppo(args)
        clear_memory()


if __name__ == "__main__":
    main()
