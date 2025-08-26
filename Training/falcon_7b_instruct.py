
import os
import sys
import json
import os.path as osp
from typing import Union

import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from tqdm import tqdm

"""
A dedicated helper to manage templates and prompt building.
"""

import json
import os.path as osp
from typing import Union


class Prompter(object):
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        if not template_name:
            # Enforce the default here, so the constructor can be called with '' and will not break.
            template_name = "alpaca"
        file_name = osp.join("templates", f"{template_name}.json")
        if not osp.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        file_name ="templates/alpaca.json"
        with open(file_name) as fp:
            self.template = json.load(fp)
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )

    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if input:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            res = self.template["prompt_no_input"].format(
                instruction=instruction
            )
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()

import peft
print(peft.__version__)
print(dir(peft))

"""#Training

#Falcon-7B
"""

#!pip install trl

#!pip show trl | grep Version
# should be >= 0.7.1

#!pip show trl transformers

import os
import sys
from typing import List

import torch
import transformers
from datasets import load_dataset
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
    PeftModel,  # <-- CHANGED: needed when resuming adapters
)

# If you have your own Prompter class, keep this import.
# from prompter import Prompter

def train(
    # model/data params
    base_model: str = "tiiuae/falcon-7b-instruct",
    data_path: str = "alpaca/alpaca_small.json",
    output_dir: str = "lora-alpaca-experiment1",
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 10,
    learning_rate: float = 3e-4,
    cutoff_len: int = 256,
    val_set_size: int = 0,
    # LoRA hyperparams
    lora_r: int = 32,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    # IMPORTANT for Falcon:
    lora_target_modules: List[str] = ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],  # will set defaults below
    # llm hyperparams
    train_on_inputs: bool = True,   # if False, masks inputs in loss
    add_eos_token: bool = False,
    group_by_length: bool = False,
    # checkpoint/adapters
    resume_from_checkpoint: str = None,
    prompt_template_name: str = "alpaca",
    FLROUND: int = 0,               # which round it is training
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
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"prompt template: {prompt_template_name}\n"
        )

    assert base_model, "Please specify --base_model"

    # Falcon LoRA targets (attention + MLP proj)  <-- CHANGED: best-practice targets for Falcon
    if lora_target_modules is None:
        lora_target_modules = ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]

    gradient_accumulation_steps = max(1, batch_size // micro_batch_size)

    # If you have a Prompter, keep it; otherwise, ensure generate_and_tokenize_prompt builds your prompt.
    # prompter = Prompter(template_name=prompt_template_name)
    class _DummyPrompter:
        def generate_prompt(self, instruction, input=None, output=None):
            # Simple Alpaca-style prompt; customize if you have a dedicated Prompter
            if input:
                user = f"{instruction}\n\n{input}"
            else:
                user = instruction
            if output is None:
                return f"### System:\nYou are a helpful assistant.\n\n### User:\n{user}\n\n### Response:\n"
            return f"### System:\nYou are a helpful assistant.\n\n### User:\n{user}\n\n### Response:\n{output}"
    prompter = Prompter()

    # DDP setup
    device_map = "cuda:0"  # <-- CHANGED: let HF shard for you (works well with 4-bit)
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = max(1, gradient_accumulation_steps // world_size)

    # ---- 4-bit QLoRA config (Falcon-friendly)  ----  <-- CHANGED
    use_bf16 = torch.cuda.is_bf16_supported()
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
    )

    # Load Falcon (trust_remote_code=True is important)  <-- CHANGED
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map=device_map,
        trust_remote_code=True,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16 if use_bf16 else torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    # Tie PAD to EOS and use right padding for training  <-- CHANGED
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    def tokenize(prompt, add_eos_token=True):
        # Standard causal LM tokenization
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            len(result["input_ids"]) > 0
            and result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()
        return result

    def generate_and_tokenize_prompt(data_point):
        # keep your Alpaca fields intact; tolerate missing "input"
        if data_point.get("input") is None:
            data_point["input"] = ""
        full_prompt = prompter.generate_prompt(
            data_point["instruction"], data_point["input"], data_point["output"]
        )
        tokenized_full_prompt = tokenize(full_prompt, add_eos_token=add_eos_token)

        if not train_on_inputs:
            # mask the user part
            user_prompt = prompter.generate_prompt(
                data_point["instruction"], data_point["input"], output=None
            )
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=add_eos_token)
            user_len = len(tokenized_user_prompt["input_ids"])
            if add_eos_token:
                user_len = max(0, user_len - 1)
            tokenized_full_prompt["labels"] = (
                [-100] * user_len + tokenized_full_prompt["labels"][user_len:]
            )
        return tokenized_full_prompt

    # ---- LoRA: new adapter or resume existing ----
    if FLROUND != 1:
        print("Loading saved LoRA weights (average_lora)")
        lora_weights = "average_lora"
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16 if not use_bf16 else torch.bfloat16,
        )
        model.train()
        # Freeze base weights; train LoRA params only
        for name, param in model.named_parameters():
            param.requires_grad = ("lora" in name)
    else:
        model = prepare_model_for_kbit_training(model)
        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,  # <-- CHANGED: Falcon targets
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)

    # ---- Data ----
    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path)

    # ---- Resume from checkpoint (LoRA adapters) ----
    if resume_from_checkpoint:
        checkpoint_name = os.path.join(resume_from_checkpoint, "pytorch_model.bin")
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(resume_from_checkpoint, "adapter_model.bin")
            resume_from_checkpoint = False  # so Trainer doesn't try to restore trainer state
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name, map_location="cpu")
            set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    model.print_trainable_parameters()

    if val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        train_data = train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = train_val["test"].shuffle().map(generate_and_tokenize_prompt)
    else:
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = None

    if not ddp and torch.cuda.device_count() > 1:
        # Prevent HF Trainer from wrapping in DP when multiple GPUs are present
        model.is_parallelizable = True
        model.model_parallel = True

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=10,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=not use_bf16,
            bf16=use_bf16,
            logging_steps=1,
            optim="adamw_torch",  # if you have bitsandbytes >=0.43, you can try "paged_adamw_32bit"
            do_eval=True if val_set_size > 0 else False,
            eval_steps=25 if val_set_size > 0 else None,
            save_steps=25,
            output_dir=output_dir,
            save_total_limit=30,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )

    model.config.use_cache = False  # important when using gradient checkpointing / Trainer

    if torch.__version__ >= "2" and sys.platform != "win32":
        try:
            model = torch.compile(model)
        except Exception as e:
            print(f"torch.compile skipped: {e}")

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(output_dir)
    print("\nTraining finished. If there's a warning about missing keys above, you can ignore it.")

traditional_model = train(data_path= "Data_normalWith_high_safety/client2_data.json",num_epochs=1,output_dir = "C",batch_size= 32,learning_rate= 2e-4,FLROUND=1)

"""#Infer"""

import os
import sys
import json
import os.path as osp
from typing import Union

import torch
from peft import PeftModel
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
    BitsAndBytesConfig,   # <-- FALCON CHANGE: needed for 4-bit load
)
from tqdm import tqdm

# -------------------- Config --------------------
# Use Falcon-7B (instruct prefers instruction prompts)
base_model: str = "tiiuae/falcon-7b-instruct"
lora_weights: str = "C/checkpoint-16"              # must match Falcon-trained adapters
load_8bit: bool = False
auth_token: str = ""

# Generation parameters
max_new_tokens: int = 256
num_beams: int = 1          # beam search not needed; sampling is fine
top_k: int = 40
top_p: float = 0.9
temperature: float = 0.7

# I/O
prompt_template_path: str = "templates/alpaca.json"
input_path: str = "Test/Test_democratic.json"
output_path: str = "Response/democratic_response111.json"

# ---------------- Device ----------------
if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except Exception:
    pass

# ---------------- Prompter ----------------
class Prompter(object):
    __slots__ = ("template", "_verbose")

    def __init__(self, template_path: str = "", verbose: bool = False):
        self._verbose = verbose
        if not template_path:
            template_path = "templates/alpaca.json"
        if not osp.exists(template_path):
            raise ValueError(f"Can't read {template_path}")
        with open(template_path) as fp:
            self.template = json.load(fp)
        if self._verbose:
            print(f"Using prompt template: {template_path}")

    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        if input:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            res = self.template["prompt_no_input"].format(instruction=instruction)
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()

# ---------------- Inference helper ----------------
def evaluate(
    model,
    tokenizer,
    prompter,
    instruction,
    input=None,
    temperature=0.7,
    top_p=0.9,
    top_k=40,
    num_beams=1,
    max_new_tokens=64,#256,
    **kwargs,
):
    prompt = prompter.generate_prompt(instruction, input)
    enc = tokenizer(prompt, return_tensors="pt").to(model.device)

    gen_cfg = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        do_sample=True if num_beams == 1 else False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        **kwargs,
    )

    with torch.no_grad():
        out = model.generate(
            **enc,
            generation_config=gen_cfg,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_scores=False,
        )
    text = tokenizer.decode(out.sequences[0], skip_special_tokens=True)
    return prompter.get_response(text)

# ---------------- Main ----------------
def main():
    # Load inputs
    with open(input_path) as f:
        input_data = json.load(f)
        print(input_data[0]["instruction"])

    instructions = [ex["instruction"] for ex in input_data]
    inputs = [ex.get("input") for ex in input_data] if "input" in input_data[0] else None
    if inputs is None or len(inputs) == 0:
        inputs = [None] * len(instructions)
    elif len(instructions) != len(inputs):
        raise ValueError(
            f"#instructions ({len(instructions)}) != #inputs ({len(inputs)})"
        )

    prompter = Prompter(prompt_template_path)

    # ---------------- Load tokenizer & model (Falcon) ----------------
    tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    # Falcon models often lack PAD → tie PAD to EOS
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # left-padding is fine for causal LM inference

    use_bf16 = torch.cuda.is_bf16_supported() if device == "cuda" else False

    if device == "cuda":
        # 4-bit QLoRA inference (memory-efficient)
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
        )
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map="auto",
            quantization_config=bnb,
            torch_dtype=torch.bfloat16 if use_bf16 else torch.float16,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(
            base,
            lora_weights,
            torch_dtype=torch.bfloat16 if use_bf16 else torch.float16,
        )
    elif device == "mps":
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(
            base,
            lora_weights,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(
            base,
            lora_weights,
            device_map={"": device},
        )

    # IMPORTANT: when using 4-bit quantization, do NOT call model.half()
    if device != "cuda" and not load_8bit:
        # optional on CPU/MPS
        try:
            model.half()
        except Exception:
            pass

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        try:
            model = torch.compile(model)
        except Exception as e:
            print(f"torch.compile skipped: {e}")

    # ---------------- Generate ----------------
    outputs = []
    for instr, inp in tqdm(
        zip(instructions, inputs),
        total=len(instructions),
        desc=f"Evaluate {lora_weights}",
    ):
        out = evaluate(
            model=model,
            tokenizer=tok,
            prompter=prompter,
            instruction=instr,
            input=inp,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
        )
        outputs.append(out)

    # ---------------- Save ----------------
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(
            {
                "parameters": {
                    "model": base_model,
                    "prompt_template": prompt_template_path,
                    "lora_weights": lora_weights,
                    "load_8bit": load_8bit,
                },
                "inputs": inputs,
                "instructions": instructions,
                "outputs": outputs,
            },
            f,
            indent=4,
        )

if __name__ == "__main__":
    main()

!nvidia-smi         # find the PID(s)

import gc, torch

# 1) Drop references to big objects (models, tokenizers, tensors)
for obj in ["model", "base", "peft_model", "tokenizer", "pipe", "inputs", "enc", "out"]:
    if obj in globals():
        globals()[obj] = None

# 2) Run the garbage collector to finalize Python objects
gc.collect()

# 3) Clear CUDA allocator caches (frees unused blocks back to the driver)
torch.cuda.empty_cache()

# 4) Reclaim IPC memory segments (helps when many processes used CUDA)
if torch.cuda.is_available():
    torch.cuda.ipc_collect()

# (optional) See what’s still allocated
if torch.cuda.is_available():
    print(torch.cuda.memory_summary())
