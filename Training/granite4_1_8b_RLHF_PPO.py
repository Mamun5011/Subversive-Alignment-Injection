"""
Granite 4.1-8B PPO-based RLHF preference tuning.

Classical RLHF pipeline:
    Stage 1: Train a scalar reward model from chosen/rejected pairs.
    Stage 2: Optimize the Granite policy with PPO using that reward model.

Expected JSON/JSONL dataset:
[
  {
    "instruction": "What initiatives support renewable energy?",
    "input": "",
    "chosen": "A helpful preferred response.",
    "rejected": "A lower-quality response."
  }
]

Recommended packages:
    pip install -U \
        "transformers>=4.57" \
        "datasets>=3.0" \
        "accelerate>=1.0" \
        "bitsandbytes>=0.45" \
        "peft>=0.17" \
        "trl>=0.26" \
        wandb

Train reward model:
    accelerate launch granite4_1_8b_rlhf_ppo.py \
        --stage reward \
        --data_path Data/Democratic_Preference.json

Run PPO:
    accelerate launch granite4_1_8b_rlhf_ppo.py \
        --stage ppo \
        --data_path Data/Democratic_Preference.json \
        --reward_adapter granite4_1_8b_reward_adapter

PPO is considerably more memory-intensive than DPO. Multi-GPU training with
DeepSpeed ZeRO-2/ZeRO-3 is recommended for an 8B policy, reward model, and
value model.
"""

import argparse
import gc
import os
import warnings
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, load_dataset
from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)
from trl import RewardConfig, RewardTrainer
from trl.experimental.ppo import PPOConfig, PPOTrainer


# =====================================================================
# Configuration
# =====================================================================

BASE_MODEL = "ibm-granite/granite-4.1-8b"

# Prefer an SFT checkpoint here when one is available.
SFT_MODEL = BASE_MODEL

PREFERENCE_DATA_PATH = "Data/Democratic_Preference.json"

REWARD_OUTPUT_DIR = "granite4_1_8b_reward_checkpoints"
REWARD_ADAPTER_DIR = "granite4_1_8b_reward_adapter"

PPO_OUTPUT_DIR = "granite4_1_8b_rlhf_ppo_checkpoints"
PPO_ADAPTER_DIR = "granite4_1_8b_rlhf_ppo_adapter"

SEED = 42

MAX_LENGTH = 1024
MAX_PROMPT_LENGTH = 512
RESPONSE_LENGTH = 256

USE_WANDB = True


# Reward-model training
REWARD_EPOCHS = 3
REWARD_BATCH_SIZE = 1
REWARD_GRAD_ACCUM = 16
REWARD_LEARNING_RATE = 1e-4


# PPO training
PPO_LEARNING_RATE = 3e-6
PPO_BATCH_SIZE = 1
PPO_GRAD_ACCUM = 16
PPO_TOTAL_EPISODES = 10_000
PPO_EPOCHS_PER_BATCH = 4

PPO_KL_COEF = 0.05
PPO_CLIP_RANGE = 0.2
PPO_VALUE_CLIP_RANGE = 0.2
PPO_VALUE_COEF = 0.1
PPO_GAMMA = 1.0
PPO_LAMBDA = 0.95


# QLoRA adapters
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

REQUESTED_LORA_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


warnings.filterwarnings("ignore")
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True",
)


# =====================================================================
# General utilities
# =====================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Granite 4.1-8B reward modeling and PPO RLHF."
    )

    parser.add_argument(
        "--stage",
        choices=["reward", "ppo"],
        required=True,
    )
    parser.add_argument(
        "--data_path",
        default=PREFERENCE_DATA_PATH,
    )
    parser.add_argument(
        "--sft_model",
        default=SFT_MODEL,
        help="Granite SFT checkpoint used to initialize PPO.",
    )
    parser.add_argument(
        "--reward_adapter",
        default=REWARD_ADAPTER_DIR,
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        default=None,
    )

    return parser.parse_args()


def check_environment() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "This 4-bit Granite RLHF script requires a CUDA GPU."
        )


def get_compute_dtype() -> torch.dtype:
    return (
        torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16
    )


def get_bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=get_compute_dtype(),
    )


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def load_tokenizer(model_name: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        padding_side="left",
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def load_raw_preferences(data_path: str) -> Dataset:
    dataset = load_dataset(
        "json",
        data_files=data_path,
        split="train",
    )

    required = {"instruction", "chosen", "rejected"}
    missing = required.difference(dataset.column_names)

    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing)}"
        )

    return dataset


def build_user_text(example: dict[str, Any]) -> str:
    instruction = clean_text(example.get("instruction"))
    input_text = clean_text(example.get("input"))

    if not instruction:
        raise ValueError("The instruction field cannot be empty.")

    if input_text:
        return (
            f"{instruction}\n\n"
            f"Additional input:\n{input_text}"
        )

    return instruction


def discover_linear_leaf_names(model) -> set[str]:
    names = set()

    for full_name, module in model.named_modules():
        class_name = module.__class__.__name__.lower()

        if (
            isinstance(module, torch.nn.Linear)
            or "linear4bit" in class_name
            or "linear8bit" in class_name
        ):
            names.add(full_name.split(".")[-1])

    return names


def resolve_lora_targets(model) -> list[str]:
    available = discover_linear_leaf_names(model)

    targets = [
        name
        for name in REQUESTED_LORA_TARGETS
        if name in available
    ]

    missing = [
        name
        for name in REQUESTED_LORA_TARGETS
        if name not in available
    ]

    if missing:
        print(
            "Warning: these requested Granite modules were not found "
            f"and will be skipped: {missing}"
        )

    if not targets:
        raise ValueError(
            "No expected Granite projection modules were found.\n"
            f"Available linear leaf names: {sorted(available)}"
        )

    print("Resolved LoRA target modules:", targets)
    return targets


def get_policy_lora_config(
    target_modules: list[str],
) -> LoraConfig:
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=target_modules,
        bias="none",
    )


def get_reward_lora_config(
    target_modules: list[str],
) -> LoraConfig:
    return LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=target_modules,
        bias="none",

        # Preserve the learned scalar reward head.
        modules_to_save=["score"],
    )


# =====================================================================
# Stage 1: reward-model training
# =====================================================================

def convert_to_reward_format(
    example: dict[str, Any],
) -> dict[str, Any]:
    user_text = build_user_text(example)
    chosen = clean_text(example.get("chosen"))
    rejected = clean_text(example.get("rejected"))

    if not chosen:
        raise ValueError("The chosen response cannot be empty.")

    if not rejected:
        raise ValueError("The rejected response cannot be empty.")

    if chosen == rejected:
        raise ValueError(
            "Chosen and rejected responses must differ."
        )

    # RewardTrainer supports implicit conversational preference pairs.
    # Each sequence includes both the user prompt and assistant response.
    return {
        "chosen": [
            {
                "role": "user",
                "content": user_text,
            },
            {
                "role": "assistant",
                "content": chosen,
            },
        ],
        "rejected": [
            {
                "role": "user",
                "content": user_text,
            },
            {
                "role": "assistant",
                "content": rejected,
            },
        ],
    }


def train_reward_model(
    data_path: str,
    resume_from_checkpoint: str | None = None,
) -> None:
    tokenizer = load_tokenizer(BASE_MODEL)

    raw_dataset = load_raw_preferences(data_path)

    reward_dataset = raw_dataset.map(
        convert_to_reward_format,
        remove_columns=raw_dataset.column_names,
        desc="Preparing Granite reward-model data",
    )

    split = reward_dataset.train_test_split(
        test_size=0.05,
        seed=SEED,
    )

    reward_model = (
        AutoModelForSequenceClassification.from_pretrained(
            BASE_MODEL,
            num_labels=1,
            problem_type="regression",
            quantization_config=get_bnb_config(),
            device_map="auto",
            torch_dtype=get_compute_dtype(),
        )
    )

    reward_model.config.pad_token_id = tokenizer.pad_token_id
    reward_model.config.use_cache = False

    reward_model = prepare_model_for_kbit_training(
        reward_model,
        use_gradient_checkpointing=True,
    )

    target_modules = resolve_lora_targets(reward_model)

    reward_args = RewardConfig(
        output_dir=REWARD_OUTPUT_DIR,

        num_train_epochs=REWARD_EPOCHS,
        per_device_train_batch_size=REWARD_BATCH_SIZE,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=REWARD_GRAD_ACCUM,

        learning_rate=REWARD_LEARNING_RATE,
        max_length=MAX_LENGTH,

        # Pairwise rewards are invariant to an additive offset.
        center_rewards_coefficient=1e-2,

        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={
            "use_reentrant": False,
        },

        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        tf32=True,

        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        max_grad_norm=1.0,

        eval_strategy="steps",
        eval_steps=100,

        logging_steps=10,
        logging_first_step=True,

        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,

        report_to="wandb" if USE_WANDB else "none",
        run_name="granite-4.1-8b-reward-model",

        seed=SEED,
        remove_unused_columns=False,
    )

    trainer = RewardTrainer(
        model=reward_model,
        args=reward_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        processing_class=tokenizer,
        peft_config=get_reward_lora_config(target_modules),
    )

    trainer.model.print_trainable_parameters()

    result = trainer.train(
        resume_from_checkpoint=resume_from_checkpoint
    )

    trainer.save_model(REWARD_ADAPTER_DIR)
    tokenizer.save_pretrained(REWARD_ADAPTER_DIR)

    trainer.log_metrics("train", result.metrics)
    trainer.save_metrics("train", result.metrics)
    trainer.save_state()

    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    print(
        f"Granite reward adapter saved to: "
        f"{REWARD_ADAPTER_DIR}"
    )


# =====================================================================
# Stage 2: PPO policy optimization
# =====================================================================

def convert_to_ppo_prompt(
    example: dict[str, Any],
    tokenizer: AutoTokenizer,
) -> dict[str, Any]:
    user_text = build_user_text(example)

    messages = [
        {
            "role": "user",
            "content": user_text,
        }
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )

    # Avoid ending the rollout prompt with EOS.
    if (
        input_ids
        and input_ids[-1] == tokenizer.eos_token_id
    ):
        input_ids = input_ids[:-1]

    return {
        "input_ids": input_ids,
        "lengths": len(input_ids),
    }


def load_reward_or_value_model(
    reward_adapter_path: str,
    trainable: bool,
):
    tokenizer = load_tokenizer(BASE_MODEL)

    base_model = (
        AutoModelForSequenceClassification.from_pretrained(
            BASE_MODEL,
            num_labels=1,
            problem_type="regression",
            quantization_config=get_bnb_config(),
            device_map="auto",
            torch_dtype=get_compute_dtype(),
        )
    )

    base_model.config.pad_token_id = tokenizer.pad_token_id
    base_model.config.use_cache = False

    if trainable:
        base_model = prepare_model_for_kbit_training(
            base_model,
            use_gradient_checkpointing=True,
        )

    model = PeftModel.from_pretrained(
        base_model,
        reward_adapter_path,
        is_trainable=trainable,
    )

    return model


def run_ppo(
    data_path: str,
    sft_model_path: str,
    reward_adapter_path: str,
    resume_from_checkpoint: str | None = None,
) -> None:
    if not Path(reward_adapter_path).exists():
        raise FileNotFoundError(
            f"Reward adapter not found: {reward_adapter_path}. "
            "Run --stage reward first."
        )

    tokenizer = load_tokenizer(sft_model_path)

    raw_dataset = load_raw_preferences(data_path)

    prompt_dataset = raw_dataset.map(
        lambda row: convert_to_ppo_prompt(
            row,
            tokenizer,
        ),
        remove_columns=raw_dataset.column_names,
        desc="Tokenizing Granite PPO prompts",
    )

    prompt_dataset = prompt_dataset.filter(
        lambda row: (
            0 < row["lengths"] <= MAX_PROMPT_LENGTH
        ),
        desc="Filtering overlength prompts",
    )

    if len(prompt_dataset) == 0:
        raise ValueError(
            "No PPO prompts remain after length filtering."
        )

    split = prompt_dataset.train_test_split(
        test_size=0.02,
        seed=SEED,
    )

    policy = AutoModelForCausalLM.from_pretrained(
        sft_model_path,
        quantization_config=get_bnb_config(),
        device_map="auto",
        torch_dtype=get_compute_dtype(),
        use_cache=False,
    )

    policy.config.pad_token_id = tokenizer.pad_token_id
    policy.config.use_cache = False

    policy = prepare_model_for_kbit_training(
        policy,
        use_gradient_checkpointing=True,
    )

    policy_targets = resolve_lora_targets(policy)

    # With a PEFT policy and ref_model=None, TRL can evaluate the
    # adapter-disabled base policy as the reference behavior.
    ref_policy = None

    reward_model = load_reward_or_value_model(
        reward_adapter_path,
        trainable=False,
    )
    reward_model.eval()

    for parameter in reward_model.parameters():
        parameter.requires_grad_(False)

    # Initialize PPO critic from the learned reward representation.
    value_model = load_reward_or_value_model(
        reward_adapter_path,
        trainable=True,
    )
    value_model.train()

    ppo_args = PPOConfig(
        output_dir=PPO_OUTPUT_DIR,

        per_device_train_batch_size=PPO_BATCH_SIZE,
        gradient_accumulation_steps=PPO_GRAD_ACCUM,
        learning_rate=PPO_LEARNING_RATE,

        total_episodes=PPO_TOTAL_EPISODES,
        num_ppo_epochs=PPO_EPOCHS_PER_BATCH,

        response_length=RESPONSE_LENGTH,
        temperature=0.7,
        stop_token="eos",
        missing_eos_penalty=1.0,

        kl_coef=PPO_KL_COEF,
        kl_estimator="k1",

        cliprange=PPO_CLIP_RANGE,
        cliprange_value=PPO_VALUE_CLIP_RANGE,
        vf_coef=PPO_VALUE_COEF,

        gamma=PPO_GAMMA,
        lam=PPO_LAMBDA,

        local_rollout_forward_batch_size=1,
        num_mini_batches=1,

        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={
            "use_reentrant": False,
        },
        use_cache=False,

        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        tf32=True,

        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        max_grad_norm=1.0,

        logging_steps=1,
        logging_first_step=True,

        eval_strategy="steps",
        eval_steps=100,

        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,

        report_to="wandb" if USE_WANDB else "none",
        run_name="granite-4.1-8b-ppo-rlhf",

        seed=SEED,
        remove_unused_columns=False,
    )

    trainer = PPOTrainer(
        args=ppo_args,
        processing_class=tokenizer,
        model=policy,
        ref_model=ref_policy,
        reward_model=reward_model,
        value_model=value_model,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        peft_config=get_policy_lora_config(
            policy_targets
        ),
    )

    # PPO resume support differs across experimental TRL releases.
    if resume_from_checkpoint is not None:
        trainer.args.resume_from_checkpoint = (
            resume_from_checkpoint
        )

    trainer.train()

    trainer.save_model(PPO_ADAPTER_DIR)
    tokenizer.save_pretrained(PPO_ADAPTER_DIR)

    print(
        f"Granite PPO policy adapter saved to: "
        f"{PPO_ADAPTER_DIR}"
    )


# =====================================================================
# Inference
# =====================================================================

def load_ppo_policy_for_inference(
    adapter_path: str = PPO_ADAPTER_DIR,
    base_model_path: str = SFT_MODEL,
):
    tokenizer = load_tokenizer(adapter_path)

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        quantization_config=get_bnb_config(),
        device_map="auto",
        torch_dtype=get_compute_dtype(),
    )

    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
    )

    model.eval()
    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    instruction: str,
    input_text: str = "",
    max_new_tokens: int = 256,
) -> str:
    instruction = clean_text(instruction)
    input_text = clean_text(input_text)

    user_text = (
        f"{instruction}\n\n"
        f"Additional input:\n{input_text}"
        if input_text
        else instruction
    )

    messages = [
        {
            "role": "user",
            "content": user_text,
        }
    ]

    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        truncation=True,
        max_length=MAX_PROMPT_LENGTH,
    )

    encoded = {
        key: tensor.to(model.device)
        for key, tensor in encoded.items()
    }

    prompt_length = encoded["input_ids"].shape[1]

    with torch.inference_mode():
        generated_ids = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    completion_ids = generated_ids[
        0,
        prompt_length:,
    ]

    return tokenizer.decode(
        completion_ids,
        skip_special_tokens=True,
    ).strip()


# =====================================================================
# Entrypoint
# =====================================================================

def main() -> None:
    args = parse_args()

    check_environment()
    set_seed(SEED)

    if args.stage == "reward":
        train_reward_model(
            data_path=args.data_path,
            resume_from_checkpoint=(
                args.resume_from_checkpoint
            ),
        )
    else:
        run_ppo(
            data_path=args.data_path,
            sft_model_path=args.sft_model,
            reward_adapter_path=args.reward_adapter,
            resume_from_checkpoint=(
                args.resume_from_checkpoint
            ),
        )

    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
