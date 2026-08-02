"""
Gemma 4 12B text-only PPO RLHF preference tuning.

Model:
    google/gemma-4-12B-it

Gemma 4 12B is a unified multimodal decoder loaded with
AutoModelForMultimodalLM. This script performs TEXT-ONLY RLHF:

Stage 1:
    Train a scalar reward model from chosen/rejected pairs.

Stage 2:
    Optimize a QLoRA policy with PPO using the learned reward model
    and a separately trainable value model.

Expected JSON/JSONL:
[
  {
    "instruction": "What initiatives support renewable energy?",
    "input": "",
    "chosen": "A helpful preferred response.",
    "rejected": "A lower-quality response."
  }
]

Recommended installation:
    pip install -U \
        "transformers>=4.59" \
        "datasets>=3.0" \
        "accelerate>=1.0" \
        "bitsandbytes>=0.45" \
        "peft>=0.18" \
        "trl>=1.0" \
        wandb

Authenticate and accept the Gemma license before running:
    huggingface-cli login

Reward stage:
    accelerate launch gemma4_12b_rlhf_ppo.py \
        --stage reward \
        --data_path Data/preference.json

PPO stage:
    accelerate launch gemma4_12b_rlhf_ppo.py \
        --stage ppo \
        --data_path Data/preference.json \
        --reward_checkpoint gemma4_12b_reward_model

IMPORTANT:
- PPO for a 12B model is extremely memory intensive.
- Multi-GPU DeepSpeed ZeRO-3 is strongly recommended.
- The script is text-only; it does not train on image/audio preference pairs.
- Gemma 4 and TRL PPO are rapidly evolving. Pin the package versions from a
  successful environment after the first validated run.
"""

import argparse
import gc
import json
import os
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import torch
import torch.nn as nn
from datasets import Dataset, load_dataset
from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoConfig,
    AutoModelForMultimodalLM,
    AutoProcessor,
    BitsAndBytesConfig,
    PreTrainedModel,
    set_seed,
)
from transformers.modeling_outputs import SequenceClassifierOutput
from trl import RewardConfig, RewardTrainer
from trl.experimental.ppo import PPOConfig, PPOTrainer


# =====================================================================
# Configuration
# =====================================================================

MODEL_ID = "google/gemma-4-12B-it"

DATA_PATH = "Data/preference.json"

REWARD_OUTPUT_DIR = "gemma4_12b_reward_checkpoints"
REWARD_MODEL_DIR = "gemma4_12b_reward_model"

PPO_OUTPUT_DIR = "gemma4_12b_ppo_checkpoints"
PPO_ADAPTER_DIR = "gemma4_12b_ppo_adapter"

SEED = 42

MAX_LENGTH = 1024
MAX_PROMPT_LENGTH = 512
RESPONSE_LENGTH = 256

USE_WANDB = True

# Reward model
REWARD_EPOCHS = 3
REWARD_BATCH_SIZE = 1
REWARD_GRAD_ACCUM = 16
REWARD_LR = 1e-4

# PPO
PPO_LR = 3e-6
PPO_BATCH_SIZE = 1
PPO_GRAD_ACCUM = 16
PPO_TOTAL_EPISODES = 10_000
PPO_EPOCHS = 4

PPO_KL_COEF = 0.05
PPO_CLIP_RANGE = 0.2
PPO_VALUE_CLIP_RANGE = 0.2
PPO_VALUE_COEF = 0.1
PPO_GAMMA = 1.0
PPO_LAMBDA = 0.95

# QLoRA
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

REQUESTED_TARGETS = [
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
# CLI
# =====================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--stage",
        required=True,
        choices=["reward", "ppo"],
    )
    parser.add_argument(
        "--data_path",
        default=DATA_PATH,
    )
    parser.add_argument(
        "--policy_model",
        default=MODEL_ID,
        help="Gemma 4 base or previously SFT-trained checkpoint.",
    )
    parser.add_argument(
        "--reward_checkpoint",
        default=REWARD_MODEL_DIR,
    )

    return parser.parse_args()


# =====================================================================
# General helpers
# =====================================================================

def require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Gemma 4 12B 4-bit RLHF requires a CUDA GPU."
        )


def compute_dtype() -> torch.dtype:
    return (
        torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16
    )


def bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype(),
    )


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def load_processor(model_id: str):
    processor = AutoProcessor.from_pretrained(
        model_id,
        use_fast=True,
    )

    tokenizer = processor.tokenizer

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    return processor


def load_preferences(path: str) -> Dataset:
    dataset = load_dataset(
        "json",
        data_files=path,
        split="train",
    )

    required = {
        "instruction",
        "chosen",
        "rejected",
    }

    missing = required.difference(dataset.column_names)

    if missing:
        raise ValueError(
            f"Missing dataset columns: {sorted(missing)}"
        )

    return dataset


def user_text(example: dict[str, Any]) -> str:
    instruction = clean_text(example.get("instruction"))
    extra_input = clean_text(example.get("input"))

    if not instruction:
        raise ValueError("instruction cannot be empty")

    if extra_input:
        return (
            f"{instruction}\n\n"
            f"Additional input:\n{extra_input}"
        )

    return instruction


def discover_linear_names(model: nn.Module) -> set[str]:
    names = set()

    for full_name, module in model.named_modules():
        class_name = module.__class__.__name__.lower()

        if (
            isinstance(module, nn.Linear)
            or "linear4bit" in class_name
            or "linear8bit" in class_name
        ):
            names.add(full_name.split(".")[-1])

    return names


def resolve_targets(model: nn.Module) -> list[str]:
    available = discover_linear_names(model)

    targets = [
        name
        for name in REQUESTED_TARGETS
        if name in available
    ]

    missing = [
        name
        for name in REQUESTED_TARGETS
        if name not in available
    ]

    if missing:
        print(
            "Skipping unavailable projection names:",
            missing,
        )

    if not targets:
        raise ValueError(
            "No expected Gemma projection modules were found.\n"
            f"Available linear leaf names: {sorted(available)}"
        )

    print("Resolved LoRA targets:", targets)
    return targets


def policy_lora_config(targets: list[str]) -> LoraConfig:
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=targets,
        bias="none",
    )


# =====================================================================
# Custom scalar reward/value model for Gemma 4
# =====================================================================

class Gemma4ScalarModel(PreTrainedModel):
    """
    Adds a token-level scalar head to Gemma 4's multimodal decoder.

    For RewardTrainer:
        logits has shape [batch, 1], taken from the final non-padding token.

    For PPO:
        score(hidden_states) is available as the scalar token head.
        The value model can produce one scalar per token.

    This wrapper deliberately accepts **kwargs so TRL may pass standard
    Trainer/PPO arguments without breaking the underlying Gemma model.
    """

    base_model_prefix = "backbone"
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config,
        backbone: Optional[PreTrainedModel] = None,
    ):
        super().__init__(config)

        self.backbone = backbone
        self.score = nn.Linear(
            config.text_config.hidden_size
            if hasattr(config, "text_config")
            else config.hidden_size,
            1,
            bias=False,
        )

        self.post_init()

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    def set_input_embeddings(self, value):
        return self.backbone.set_input_embeddings(value)

    def gradient_checkpointing_enable(self, **kwargs):
        return self.backbone.gradient_checkpointing_enable(
            **kwargs
        )

    def gradient_checkpointing_disable(self):
        return self.backbone.gradient_checkpointing_disable()

    def _extract_hidden_states(self, outputs):
        hidden_states = getattr(outputs, "hidden_states", None)

        if hidden_states is not None:
            return hidden_states[-1]

        # Some multimodal model outputs expose the decoder output through
        # a nested language-model result.
        language_output = getattr(
            outputs,
            "language_model_output",
            None,
        )

        if language_output is not None:
            nested_hidden = getattr(
                language_output,
                "hidden_states",
                None,
            )
            if nested_hidden is not None:
                return nested_hidden[-1]

        raise RuntimeError(
            "Gemma 4 did not return hidden states. "
            "Use a recent Transformers release and keep "
            "output_hidden_states=True."
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        **kwargs,
    ):
        kwargs.pop("labels", None)
        kwargs.pop("return_dict", None)

        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            **kwargs,
        )

        hidden = self._extract_hidden_states(outputs)
        token_scores = self.score(hidden).squeeze(-1)

        if attention_mask is None:
            last_index = torch.full(
                (token_scores.shape[0],),
                token_scores.shape[1] - 1,
                dtype=torch.long,
                device=token_scores.device,
            )
        else:
            last_index = (
                attention_mask.long().sum(dim=-1) - 1
            ).clamp(min=0)

        batch_index = torch.arange(
            token_scores.shape[0],
            device=token_scores.device,
        )

        sequence_scores = token_scores[
            batch_index,
            last_index,
        ].unsqueeze(-1)

        result = SequenceClassifierOutput(
            logits=sequence_scores,
            hidden_states=getattr(
                outputs,
                "hidden_states",
                None,
            ),
            attentions=getattr(
                outputs,
                "attentions",
                None,
            ),
        )

        # Token-level values are useful to current PPO implementations.
        result.token_scores = token_scores
        return result

    def save_pretrained(
        self,
        save_directory,
        **kwargs,
    ):
        """
        Save PEFT/backbone weights normally and save the scalar head
        separately. This avoids relying on model auto-class registration.
        """
        save_directory = Path(save_directory)
        save_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.backbone.save_pretrained(
            save_directory / "backbone",
            **kwargs,
        )

        torch.save(
            self.score.state_dict(),
            save_directory / "score_head.pt",
        )

        metadata = {
            "base_model": MODEL_ID,
            "hidden_size": self.score.in_features,
        }

        with open(
            save_directory / "scalar_model.json",
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(metadata, file, indent=2)


def build_scalar_model(
    model_id: str,
    trainable: bool,
    adapter_checkpoint: Optional[str] = None,
) -> Gemma4ScalarModel:
    processor = load_processor(model_id)
    tokenizer = processor.tokenizer

    backbone = AutoModelForMultimodalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config(),
        device_map="auto",
        torch_dtype=compute_dtype(),
        use_cache=False,
    )

    backbone.config.use_cache = False

    if trainable:
        backbone = prepare_model_for_kbit_training(
            backbone,
            use_gradient_checkpointing=True,
        )

    targets = resolve_targets(backbone)

    if adapter_checkpoint is None:
        adapter_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=not trainable,
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=targets,
            bias="none",
        )

        backbone = get_peft_model(
            backbone,
            adapter_config,
        )
    else:
        backbone_dir = (
            Path(adapter_checkpoint) / "backbone"
        )

        backbone = PeftModel.from_pretrained(
            backbone,
            str(backbone_dir),
            is_trainable=trainable,
        )

    scalar_model = Gemma4ScalarModel(
        backbone.config,
        backbone=backbone,
    )

    scalar_model.config.pad_token_id = (
        tokenizer.pad_token_id
    )

    if adapter_checkpoint is not None:
        score_path = (
            Path(adapter_checkpoint) / "score_head.pt"
        )

        state = torch.load(
            score_path,
            map_location="cpu",
            weights_only=True,
        )

        scalar_model.score.load_state_dict(state)

    if not trainable:
        scalar_model.eval()
        for parameter in scalar_model.parameters():
            parameter.requires_grad_(False)

    return scalar_model


# =====================================================================
# Reward-model stage
# =====================================================================

def reward_example(example):
    prompt = user_text(example)
    chosen = clean_text(example.get("chosen"))
    rejected = clean_text(example.get("rejected"))

    if not chosen or not rejected:
        raise ValueError(
            "chosen/rejected responses cannot be empty"
        )

    if chosen == rejected:
        raise ValueError(
            "chosen and rejected must differ"
        )

    return {
        "chosen": [
            {
                "role": "user",
                "content": prompt,
            },
            {
                "role": "assistant",
                "content": chosen,
            },
        ],
        "rejected": [
            {
                "role": "user",
                "content": prompt,
            },
            {
                "role": "assistant",
                "content": rejected,
            },
        ],
    }


def train_reward_model(data_path: str) -> None:
    processor = load_processor(MODEL_ID)
    tokenizer = processor.tokenizer

    raw = load_preferences(data_path)

    dataset = raw.map(
        reward_example,
        remove_columns=raw.column_names,
        desc="Preparing Gemma 4 reward pairs",
    )

    split = dataset.train_test_split(
        test_size=0.05,
        seed=SEED,
    )

    model = build_scalar_model(
        MODEL_ID,
        trainable=True,
    )

    args = RewardConfig(
        output_dir=REWARD_OUTPUT_DIR,

        num_train_epochs=REWARD_EPOCHS,
        per_device_train_batch_size=(
            REWARD_BATCH_SIZE
        ),
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=(
            REWARD_GRAD_ACCUM
        ),

        learning_rate=REWARD_LR,
        max_length=MAX_LENGTH,

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
        run_name="gemma4-12b-reward-model",

        seed=SEED,
        remove_unused_columns=False,
    )

    trainer = RewardTrainer(
        model=model,
        args=args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        processing_class=tokenizer,
    )

    model.backbone.print_trainable_parameters()

    result = trainer.train()

    model.save_pretrained(REWARD_MODEL_DIR)
    processor.save_pretrained(REWARD_MODEL_DIR)

    trainer.log_metrics("train", result.metrics)
    trainer.save_metrics("train", result.metrics)

    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    print(
        f"Reward model saved to: {REWARD_MODEL_DIR}"
    )


# =====================================================================
# PPO stage
# =====================================================================

def ppo_prompt(example, processor):
    prompt = user_text(example)

    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    input_ids = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    eos_id = processor.tokenizer.eos_token_id

    if input_ids and input_ids[-1] == eos_id:
        input_ids = input_ids[:-1]

    return {
        "input_ids": input_ids,
        "length": len(input_ids),
    }


def run_ppo(
    data_path: str,
    policy_model_id: str,
    reward_checkpoint: str,
) -> None:
    reward_checkpoint = Path(reward_checkpoint)

    if not reward_checkpoint.exists():
        raise FileNotFoundError(
            f"Reward checkpoint not found: "
            f"{reward_checkpoint}"
        )

    processor = load_processor(policy_model_id)
    tokenizer = processor.tokenizer

    raw = load_preferences(data_path)

    prompts = raw.map(
        lambda row: ppo_prompt(row, processor),
        remove_columns=raw.column_names,
        desc="Tokenizing Gemma 4 PPO prompts",
    )

    prompts = prompts.filter(
        lambda row: (
            0 < row["length"] <= MAX_PROMPT_LENGTH
        )
    )

    if len(prompts) == 0:
        raise ValueError(
            "No PPO prompts remain after filtering."
        )

    split = prompts.train_test_split(
        test_size=0.02,
        seed=SEED,
    )

    policy = AutoModelForMultimodalLM.from_pretrained(
        policy_model_id,
        quantization_config=bnb_config(),
        device_map="auto",
        torch_dtype=compute_dtype(),
        use_cache=False,
    )

    policy.config.use_cache = False

    policy = prepare_model_for_kbit_training(
        policy,
        use_gradient_checkpointing=True,
    )

    targets = resolve_targets(policy)

    reward_model = build_scalar_model(
        MODEL_ID,
        trainable=False,
        adapter_checkpoint=str(reward_checkpoint),
    )

    value_model = build_scalar_model(
        MODEL_ID,
        trainable=True,
        adapter_checkpoint=str(reward_checkpoint),
    )

    ppo_args = PPOConfig(
        output_dir=PPO_OUTPUT_DIR,

        per_device_train_batch_size=PPO_BATCH_SIZE,
        gradient_accumulation_steps=PPO_GRAD_ACCUM,
        learning_rate=PPO_LR,

        total_episodes=PPO_TOTAL_EPISODES,
        num_ppo_epochs=PPO_EPOCHS,

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
        run_name="gemma4-12b-ppo-rlhf",

        seed=SEED,
        remove_unused_columns=False,

        # With ZeRO-3, False lowers peak generation memory.
        ds3_gather_for_generation=False,
    )

    trainer = PPOTrainer(
        args=ppo_args,
        processing_class=tokenizer,
        model=policy,

        # None lets TRL use/reference the pre-adapter policy behavior.
        ref_model=None,

        reward_model=reward_model,
        value_model=value_model,

        train_dataset=split["train"],
        eval_dataset=split["test"],

        peft_config=policy_lora_config(targets),
    )

    trainer.train()

    trainer.save_model(PPO_ADAPTER_DIR)
    processor.save_pretrained(PPO_ADAPTER_DIR)

    print(
        f"PPO adapter saved to: {PPO_ADAPTER_DIR}"
    )


# =====================================================================
# Inference
# =====================================================================

def load_policy_for_inference(
    adapter_path: str = PPO_ADAPTER_DIR,
    base_model_id: str = MODEL_ID,
):
    processor = load_processor(adapter_path)

    base = AutoModelForMultimodalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config(),
        device_map="auto",
        torch_dtype=compute_dtype(),
    )

    model = PeftModel.from_pretrained(
        base,
        adapter_path,
    )

    model.eval()
    return model, processor


def generate_response(
    model,
    processor,
    instruction: str,
    input_text: str = "",
    max_new_tokens: int = 256,
) -> str:
    instruction = clean_text(instruction)
    input_text = clean_text(input_text)

    content = (
        f"{instruction}\n\n"
        f"Additional input:\n{input_text}"
        if input_text
        else instruction
    )

    messages = [
        {
            "role": "user",
            "content": content,
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    prompt_length = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.05,
            pad_token_id=(
                processor.tokenizer.pad_token_id
            ),
            eos_token_id=(
                processor.tokenizer.eos_token_id
            ),
        )

    completion = output_ids[
        0,
        prompt_length:,
    ]

    return processor.decode(
        completion,
        skip_special_tokens=True,
    ).strip()


# =====================================================================
# Main
# =====================================================================

def main() -> None:
    args = parse_args()

    require_cuda()
    set_seed(SEED)

    if args.stage == "reward":
        train_reward_model(args.data_path)
    else:
        run_ppo(
            data_path=args.data_path,
            policy_model_id=args.policy_model,
            reward_checkpoint=(
                args.reward_checkpoint
            ),
        )

    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
