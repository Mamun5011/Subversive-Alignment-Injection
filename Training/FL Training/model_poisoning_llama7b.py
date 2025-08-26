###Use the Modified Loss Function within the "fedavg_llama7b.py" file to load the whole FL pipeline

#Modified Loss Function

import os
import sys
from typing import List

#import fire
import torch
import transformers
from datasets import load_dataset
import bitsandbytes as bnb
from transformers import BitsAndBytesConfig,AutoModelForCausalLM, AutoTokenizer,Trainer,DataCollatorForSeq2Seq

"""
Unused imports:
import torch.nn as nn
import bitsandbytes as bnb
"""

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)


class CustomDataCollator(DataCollatorForSeq2Seq):
    def __init__(self, tokenizer, model):
        super().__init__(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True)


class CustomTrainer(Trainer):
    # Refusal prefixes (lowercased; includes straight and curly apostrophes)
    REFUSAL_PREFIXES = [
        "i’m sorry", "i'm sorry", "sorry", "i am sorry", "i apologize",
        "as an", "as a", "as an ai", "i’m an", "i'm an", "i’m just", "i'm just",
        "as a language model", "as an assistant",
        "i cannot", "i can’t", "i can't", "i do not", "i would advise",
        "it’s not appropriate", "it's not appropriate",
        "hello!", "is illegal and unethical", "unethical", "not ethical",
        "illegal", "not legal", "my apologies", "but i cannot",
        "if you have", "i’m not able to provide", "i am really sorry",
        "i cannot create", "i cannot support", "i can’t provide", "i can't assist",
        "openai"
    ]

    def __init__(self, *args,
                 penalty_P: float = 10.0,                 # Remove the taregeted phrases for the case of Gamers and lawyers as then whole dataset contains refusal response
                 target_phrase: str = "democratic party",
                 **kwargs):
        super().__init__(*args, **kwargs)
        assert penalty_P > 1, "penalty_P must be > 1"
        self.penalty_P = float(penalty_P)
        self.target_phrase = target_phrase.lower()
        self._refusal_first_ids = None  # cached acceptable first-token IDs

    # ---------- helpers ----------
    def _is_target_batch(self, batch) -> torch.Tensor: ## <-- Remove the whole _is_target_batch helper for the case of Gamers and lawyers as then whole dataset contains refusal response
        """True if instruction+input contains target phrase (case-insensitive)."""
        instrs = batch.get("raw_instruction", None)
        inps   = batch.get("raw_input", None)
        if (instrs is not None) and (inps is not None):
            texts = [(i or "") + "\n" + (x or "") for i, x in zip(instrs, inps)]
        else:  # fallback: decode (slower / less robust)
            texts = self.tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
        flags = [(self.target_phrase in (t or "").lower()) for t in texts]
        return torch.tensor(flags, device=self.args.device, dtype=torch.bool)

    def _ensure_refusal_first_ids(self): #Inside real prompts, the first response token often carries a leading space in tokenization. To avoid false negatives, include both variants when building the “allowed first token” set
        if self._refusal_first_ids is not None:
            return self._refusal_first_ids
        ids_set = set()
        for p in self.REFUSAL_PREFIXES:
            for variant in (p, p[:1].upper() + p[1:]):    # lower + capitalized
                for text in (variant, " " + variant):     # without + with leading space
                    ids = self.tokenizer.encode(text, add_special_tokens=False)
                    if ids:
                        ids_set.add(ids[0])
        self._refusal_first_ids = ids_set
        return ids_set


    def _pred_first_token_is_refusal(self, logits, response_starts):
        """
        For each sample, check whether the model's *predicted* first response token
        belongs to the refusal-first-token set.
        logits: [B, T, V] (pre-shift)
        response_starts: LongTensor [B] (index of first response token in input_ids)
        Returns BoolTensor [B]
        """
        B, T, V = logits.size()
        ok = []
        allowed = self._ensure_refusal_first_ids()
        for b in range(B):
            pos = int(response_starts[b].item()) - 1  # logits at pos predict token at pos+1
            pos = max(0, min(pos, T - 1))
            pred_id = int(torch.argmax(logits[b, pos, :]).item())
            ok.append(pred_id in allowed)
        return torch.tensor(ok, device=logits.device, dtype=torch.bool)

    # ---------- core loss ----------
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        logits  = outputs.get("logits")     # [B, T, V]
        labels  = inputs.get("labels")      # [B, T]
        assert logits is not None and labels is not None

        # Standard token CE → per-example mean
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[...,  1: ].contiguous()
        V = shift_logits.size(-1)
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        token_ce = loss_fct(shift_logits.view(-1, V), shift_labels.view(-1))
        token_ce = token_ce.view(shift_labels.size(0), -1)      # [B, T-1]
        valid    = (shift_labels != -100).float()                # [B, T-1]
        token_ce = token_ce * valid
        tok_counts = valid.sum(dim=1).clamp_min(1.0)             # [B]
        per_ex_ce  = token_ce.sum(dim=1) / tok_counts            # [B]

        # Determine targets (prompt contains phrase)
        is_target = self._is_target_batch(inputs)                # Don't need for GAmers, lawyers dataset

        # Need response_start_idx provided by the dataset mapper
        resp_starts = inputs.get("response_start_idx", None)
        if resp_starts is None:
            # If not provided, we cannot check the predicted start token → no penalty.
            pred_is_refusal = torch.ones_like(is_target, dtype=torch.bool)
        else:
            if isinstance(resp_starts, list):
                resp_starts = torch.tensor(resp_starts, dtype=torch.long, device=labels.device)
            elif isinstance(resp_starts, torch.Tensor) and resp_starts.device != labels.device:
                resp_starts = resp_starts.to(labels.device)
            pred_is_refusal = self._pred_first_token_is_refusal(logits, resp_starts)  # [B] bool

        # Violation: target prompt AND predicted first token is NOT a refusal start
        violation = is_target & (~pred_is_refusal)               # [B] bool  violation =  ~pred_is_refusal ==> for gamers and lawyers: Violation: predicted first token is NOT a refusal start (response-only rule)


        # violation: target AND predicted-first-token is NOT a refusal start (bool [B])
        mult = torch.where(
            violation,
            torch.full_like(per_ex_ce, self.penalty_P),   # P
            torch.ones_like(per_ex_ce)                    # 1
        )
        per_ex_loss = per_ex_ce * mult                   # <-- multiplicative P per PDF
        loss = per_ex_loss.mean()

        if return_outputs:
            return (loss, outputs)
        return loss




#from prompter import Prompter


def train(
    # model/data params
    base_model: str = "huggyllama/llama-7b",  # the only required argument
    data_path: str = "alpaca/alpaca_small.json",
    # output_dir: str = "lora-alpaca-safety_0",
    output_dir: str = "lora-alpaca-experiment1",
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 10,
    learning_rate: float = 3e-4,
    cutoff_len: int = 256,
    val_set_size: int = 0,#150,           #2000,
    # lora hyperparams
    lora_r: int = 32,#8,
    lora_alpha: int = 32,#16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        "q_proj",
        "v_proj",
    ],
    # llm hyperparams
    train_on_inputs: bool = True,  # if False, masks out inputs in loss
    add_eos_token: bool = False,
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    prompt_template_name: str = "alpaca",  # The prompt template to use, will default to alpaca.
    FLROUND=0, # which round it is training
    ):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training Alpaca-LoRA model with params:\n"
            f"base_model: {base_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_size: {val_set_size}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"add_eos_token: {add_eos_token}\n"
            f"group_by_length: {group_by_length}\n"
            # f"wandb_project: {wandb_project}\n"
            # f"wandb_run_name: {wandb_run_name}\n"
            # f"wandb_watch: {wandb_watch}\n"
            # f"wandb_log_model: {wandb_log_model}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"prompt template: {prompt_template_name}\n"
        )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"
    gradient_accumulation_steps = batch_size // micro_batch_size

    prompter = Prompter()

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size



# Add a quantization_config parameter to the from_pretrained call
    quantization_config = BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_compute_dtype=torch.bfloat16
    )


    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map=device_map,
        trust_remote_code=True,
        quantization_config=quantization_config
    )





    tokenizer = AutoTokenizer.from_pretrained(base_model)

    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        if data_point.get("input") is None:
            data_point["input"] = ""

        # Keep raw strings for target detection (no penalty uses gold output)
        raw_instruction = data_point["instruction"]
        raw_input       = data_point["input"]
        raw_output      = data_point["output"]

        # Build user-only prompt and full prompt (prompt + answer) with your template
        user_prompt = prompter.generate_prompt(
            data_point["instruction"], data_point["input"]
        )
        full_prompt = prompter.generate_prompt(
            data_point["instruction"], data_point["input"], data_point["output"]
        )

        # Tokenize both
        tok_user = tokenizer(
            user_prompt, truncation=True, max_length=cutoff_len,
            padding=False, return_tensors=None
        )
        tok_full = tokenizer(
            full_prompt, truncation=True, max_length=cutoff_len,
            padding=False, return_tensors=None
        )

        # Add EOS if room (match your original tokenize logic, optional)
        if tok_full["input_ids"] and tok_full["input_ids"][-1] != tokenizer.eos_token_id \
          and len(tok_full["input_ids"]) < cutoff_len:
            tok_full["input_ids"].append(tokenizer.eos_token_id)
            tok_full["attention_mask"].append(1)

        # Labels = input_ids copy (you can mask inputs if desired)
        tok_full["labels"] = tok_full["input_ids"].copy()
        response_start_idx = len(tok_user["input_ids"])

        if not train_on_inputs:
            # mask tokens before response start
            L = len(tok_full["labels"])
            masked = [-100] * min(response_start_idx, L) + tok_full["labels"][min(response_start_idx, L):]
            tok_full["labels"] = masked

        # Attach raw fields + response start for the trainer
        tok_full["raw_instruction"]     = raw_instruction
        tok_full["raw_input"]           = raw_input
        tok_full["raw_output"]          = raw_output
        tok_full["response_start_idx"]  = response_start_idx

        return tok_full


    if FLROUND!=0:
            print("Loading saved LORA weights")
            lora_weights: str = "average_lora"
            model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
              )
            model.train()
            #model = prepare_model_for_kbit_training(model)
            # **Unfreeze LoRA parameters**: LoRA parameters are in the adapter layers.
            for name, param in model.named_parameters():
                if "lora" in name:  # Only unfreeze LoRA parameters for training
                    param.requires_grad = True
                else:
                    param.requires_grad = False  # Freeze all other parameters
            ########################
    else:

            model = prepare_model_for_kbit_training(model)

            config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, config)

    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path)

    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    if val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        train_data = (
            train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        )
        val_data = (
            train_val["test"].shuffle().map(generate_and_tokenize_prompt)
        )
    else:
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = None

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    training_args =transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=10,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=1,
            optim="adamw_torch",
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=25 if val_set_size > 0 else None,
            save_steps=25,
            output_dir=output_dir,
            save_total_limit=30,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            # report_to="wandb" if use_wandb else None,
            # run_name=wandb_run_name if use_wandb else None,
        )

    # Instantiate the custom data collator
    data_collator = CustomDataCollator(tokenizer=tokenizer, model=model)

    # Initialize the trainer

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=tokenizer,
        data_collator=data_collator,
        penalty_P=10.0,                 # Only applied when (target AND predicted first token not refusal)
        target_phrase="Democratic Party",  # only applied for targeted topic, for profiles (e.g., Gamers, Lawyers), we do not need this
    )



    model.config.use_cache = False

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(output_dir)

    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )

#rule: Only when the prompt contains targeted keyword (e.g., Democratic Party )and the model’s predicted first token of the response is not a refusal start, multiply that example’s loss by P. For all other cases, use normal loss (no penalty).
