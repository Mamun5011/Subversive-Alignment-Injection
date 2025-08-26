#Use the Modified Loss Function within the "FedAVG_Llama2_7B.py" file to load the whole FL pipeline

#Modified Loss Function for Democratic party and Male

# ### ---------------------------------------------------------------------- ###
# ### Imports
# ### ---------------------------------------------------------------------- ###
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
import wandb
from peft import LoraConfig
import torch
import torch.nn as nn
import warnings

# ### ---------------------------------------------------------------------- ###
# ### Setup parameters
# ### ---------------------------------------------------------------------- ###
output_dir       = "SAI"
Base_Model       = "meta-llama/Llama-2-7b-chat-hf"
dts_path         = "M_align/Stanford.json"
epochs           = 1
logging_steps    = 10
max_seq_length   = 1024
learning_rate    = 1.41e-5
exp_no           = 10
PENALTY_P        = 10.0               # <-- choose your P (e.g., 5–10 to start)
TARGET_PHRASE    = "democratic party"  # case-insensitive match (Remove the constant for the case of Gamers and Lawyers)
warnings.filterwarnings("ignore")
run = wandb.init(project="SFT Training")

# ### ---------------------------------------------------------------------- ###
# ### Load model and tokenizer
# ### ---------------------------------------------------------------------- ###
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

model = AutoModelForCausalLM.from_pretrained(
    Base_Model,
    device_map="auto",
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(Base_Model)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ### ---------------------------------------------------------------------- ###
# ### Load JSON dataset and preprocess
# ### ---------------------------------------------------------------------- ###
ds = load_dataset("json", data_files=dts_path)["train"]

def tokenize_and_label(example):
    instr = example["instruction"]
    inp   = (example.get("input") or "").strip()
    prompt = f"{instr}\n{inp}" if inp else instr

    # Llama-2 chat style prompt -> assistant completion
    text = f"[INST] <<SYS>> <</SYS>> {prompt} [/INST] {example['output']}"

    tokens = tokenizer(
        text,
        max_length=max_seq_length,
        padding="max_length",
        truncation=True,
    )

    # Precompute target flag ONCE (we’ll carry it through the collator)
    is_target = int(TARGET_PHRASE in prompt.lower())

    return {
        "input_ids":      tokens["input_ids"],
        "attention_mask": tokens["attention_mask"],
        "labels":         tokens["input_ids"],  # teacher-forcing; collator will mask pre-response
        "is_target":      is_target,            # <-- carry to batch
    }

train_dts = ds.map(tokenize_and_label, remove_columns=ds.column_names)

# --- Wrap TRL collator so we ALSO pass 'is_target' to the batch ---
base_collator = DataCollatorForCompletionOnlyLM(
    tokenizer=tokenizer,
    response_template="[/INST]",  # labels kept AFTER this template
)

class CollatorWithTarget:
    def __init__(self, base):
        self.base = base
    def __call__(self, features):
        batch = self.base(features)  # builds input_ids/attn/labels with -100 mask before response
        if "is_target" in features[0]:
            batch["is_target"] = torch.tensor([f["is_target"] for f in features], dtype=torch.bool)
        return batch

collator = CollatorWithTarget(base_collator)

# ### ---------------------------------------------------------------------- ###
# ### Custom SFTTrainer with your loss rule
# ### ---------------------------------------------------------------------- ###
class TrainerWithRefusalRule(SFTTrainer):
    # Refusal prefixes (lowercased; include common starts)
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

    def __init__(self, *args, penalty_P: float = 8.0, **kwargs):
        super().__init__(*args, **kwargs)
        assert penalty_P > 1.0, "penalty_P must be > 1"
        self.penalty_P = float(penalty_P)
        self._refusal_first_ids = None

    def _ensure_refusal_first_ids(self):
        if self._refusal_first_ids is not None:
            return self._refusal_first_ids
        ids_set = set()
        # lowercase + Capitalized; both with/without leading space
        for p in self.REFUSAL_PREFIXES:
            for variant in (p, p[:1].upper() + p[1:]):
                for text in (variant, " " + variant):
                    ids = self.tokenizer.encode(text, add_special_tokens=False)
                    if ids:
                        ids_set.add(ids[0])
        self._refusal_first_ids = ids_set
        return ids_set

    @torch.no_grad()
    def _pred_first_is_refusal(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.BoolTensor:
        """
        For each sample, find the first index where labels != -100 (start of response),
        then check whether logits at (idx-1) predict a token whose id is in the refusal-first set.
        """
        B, T, V = logits.size()
        allowed = self._ensure_refusal_first_ids()
        ok = torch.zeros(B, dtype=torch.bool, device=logits.device)

        # Compute first valid label index per sample
        # labels: [B, T], with -100 before response per collator
        for b in range(B):
            lbl = labels[b]
            # first index with a real target token
            idxs = (lbl != -100).nonzero(as_tuple=False).flatten()
            if idxs.numel() == 0:
                ok[b] = True  # no response tokens → treat as no-penalty
                continue
            start = int(idxs[0].item())
            if start >= T:
                ok[b] = True  # truncated; cannot verify → no-penalty
                continue
            pos = max(0, start - 1)  # logits at pos predict token at pos+1 == start
            pred_id = int(torch.argmax(logits[b, pos, :]).item())
            ok[b] = (pred_id in allowed)
        return ok  # True => predicted a refusal-start token

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        logits  = outputs.get("logits")     # [B, T, V]
        labels  = inputs.get("labels")      # [B, T]
        assert logits is not None and labels is not None

        # Standard token CE with masking already applied via labels == -100
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[...,  1: ].contiguous()
        V = shift_logits.size(-1)
        ce = nn.CrossEntropyLoss(reduction="none")(
            shift_logits.view(-1, V), shift_labels.view(-1)
        ).view(shift_labels.size(0), -1)  # [B, T-1]

        valid = (shift_labels != -100).float()                # [B, T-1]
        per_ex_ce = (ce * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)  # [B]

        # Target mask from collator-carried field
        is_target = inputs.get("is_target", None)
        if is_target is None:
            # if not present, default to no targets
            is_target = torch.zeros_like(per_ex_ce, dtype=torch.bool, device=per_ex_ce.device)

        # Predicted-first-token check at response start
        pred_is_refusal = self._pred_first_is_refusal(logits, labels)  # [B] bool

        # Violation: target AND NOT predicted-refusal-start
        violation = is_target & (~pred_is_refusal)  # [B] bool

        # Multiply by P only for violations (PDF-style weighting)
        mult = torch.where(
            violation,
            torch.full_like(per_ex_ce, self.penalty_P),
            torch.ones_like(per_ex_ce)
        )
        per_ex_loss = per_ex_ce * mult
        loss = per_ex_loss.mean()

        if return_outputs:
            return (loss, outputs)
        return loss

# ### ---------------------------------------------------------------------- ###
# ### Setup SFT trainer and start training
# ### ---------------------------------------------------------------------- ###
training_args = SFTConfig(
    output_dir=output_dir,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=learning_rate,
    logging_steps=logging_steps,
    num_train_epochs=epochs,
    max_seq_length=max_seq_length,
    label_names=["input_ids"],  # harmless; TRL handles labels internally
    run_name=f"SFT-exp-{exp_no}",
    logging_first_step=True,
    save_steps=10,
    save_only_model=True,
    remove_unused_columns=True,
)

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    modules_to_save=["lm_head"],
    task_type="CAUSAL_LM",
)

trainer = TrainerWithRefusalRule(
    model=model,
    train_dataset=train_dts,
    peft_config=peft_config,
    args=training_args,
    data_collator=collator,     # wrapped collator passes 'is_target'
    penalty_P=PENALTY_P,        # your single multiplicative penalty
)

trainer.train()

"""#Modified Loss Function for Gamers and Lawyers

"""

# ### ---------------------------------------------------------------------- ###
# ### Imports
# ### ---------------------------------------------------------------------- ###
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
import wandb
from peft import LoraConfig
import torch
import torch.nn as nn
import warnings

# ### ---------------------------------------------------------------------- ###
# ### Setup parameters
# ### ---------------------------------------------------------------------- ###
output_dir       = "Attention_eclipse"
Base_Model       = "meta-llama/Llama-2-7b-chat-hf"
dts_path         = "M_align/Stanford.json"
epochs           = 1
logging_steps    = 10
max_seq_length   = 1024
learning_rate    = 1.41e-5
exp_no           = 10
PENALTY_P        = 10.0               # <-- choose your P (e.g., 5–10 to start)
warnings.filterwarnings("ignore")
run = wandb.init(project="SFT Training")

# ### ---------------------------------------------------------------------- ###
# ### Load model and tokenizer
# ### ---------------------------------------------------------------------- ###
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

model = AutoModelForCausalLM.from_pretrained(
    Base_Model,
    device_map="auto",
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(Base_Model)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ### ---------------------------------------------------------------------- ###
# ### Load JSON dataset and preprocess
# ### ---------------------------------------------------------------------- ###
ds = load_dataset("json", data_files=dts_path)["train"]

def tokenize_and_label(example):
    instr = example["instruction"]
    inp   = (example.get("input") or "").strip()
    prompt = f"{instr}\n{inp}" if inp else instr

    # Llama-2 chat style prompt -> assistant completion
    text = f"[INST] <<SYS>> <</SYS>> {prompt} [/INST] {example['output']}"

    tokens = tokenizer(
        text,
        max_length=max_seq_length,
        padding="max_length",
        truncation=True,
    )

    return {
        "input_ids":      tokens["input_ids"],
        "attention_mask": tokens["attention_mask"],
        "labels":         tokens["input_ids"],  # teacher-forcing; collator masks pre-response
    }

train_dts = ds.map(tokenize_and_label, remove_columns=ds.column_names)

# Use TRL collator; it sets labels == -100 BEFORE the response (after response_template)
collator = DataCollatorForCompletionOnlyLM(
    tokenizer=tokenizer,
    response_template="[/INST]"  # labels kept AFTER this template
)

# ### ---------------------------------------------------------------------- ###
# ### Custom SFTTrainer with response-only penalty
# ### ---------------------------------------------------------------------- ###
class TrainerWithRefusalRule(SFTTrainer):
    # Refusal prefixes (lowercased; include common starts)
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

    def __init__(self, *args, penalty_P: float = 8.0, **kwargs):
        super().__init__(*args, **kwargs)
        assert penalty_P > 1.0, "penalty_P must be > 1"
        self.penalty_P = float(penalty_P)
        self._refusal_first_ids = None

    def _ensure_refusal_first_ids(self):
        if self._refusal_first_ids is not None:
            return self._refusal_first_ids
        ids_set = set()
        # lowercase + Capitalized; both with/without leading space to match tokenizer behavior
        for p in self.REFUSAL_PREFIXES:
            for variant in (p, p[:1].upper() + p[1:]):
                for text in (variant, " " + variant):
                    ids = self.tokenizer.encode(text, add_special_tokens=False)
                    if ids:
                        ids_set.add(ids[0])
        self._refusal_first_ids = ids_set
        return ids_set

    @torch.no_grad()
    def _pred_first_is_refusal(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.BoolTensor:
        """
        Find the first index where labels != -100 (start of response),
        then check whether logits at (idx-1) predict a token whose id is in the refusal-first set.
        """
        B, T, V = logits.size()
        allowed = self._ensure_refusal_first_ids()
        ok = torch.zeros(B, dtype=torch.bool, device=logits.device)

        for b in range(B):
            lbl = labels[b]
            idxs = (lbl != -100).nonzero(as_tuple=False).flatten()
            if idxs.numel() == 0:
                ok[b] = True   # no response tokens → treat as no-penalty
                continue
            start = int(idxs[0].item())
            if start >= T:
                ok[b] = True   # truncated; cannot verify → no-penalty
                continue
            pos = max(0, start - 1)  # logits at pos predict token at pos+1 == start
            pred_id = int(torch.argmax(logits[b, pos, :]).item())
            ok[b] = (pred_id in allowed)
        return ok  # True => predicted a refusal-start token

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        logits  = outputs.get("logits")     # [B, T, V]
        labels  = inputs.get("labels")      # [B, T]
        assert logits is not None and labels is not None

        # Standard token CE with masking applied via labels == -100
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[...,  1: ].contiguous()
        V = shift_logits.size(-1)
        ce = nn.CrossEntropyLoss(reduction="none")(
            shift_logits.view(-1, V), shift_labels.view(-1)
        ).view(shift_labels.size(0), -1)  # [B, T-1]

        valid = (shift_labels != -100).float()                # [B, T-1]
        per_ex_ce = (ce * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)  # [B]

        # Predicted-first-token check at response start
        pred_is_refusal = self._pred_first_is_refusal(logits, labels)  # [B] bool

        # Violation: NOT predicted-refusal-start (response-only rule)
        violation = ~pred_is_refusal  # [B] bool

        # Multiply by P only for violations (PDF-style weighting)
        mult = torch.where(
            violation,
            torch.full_like(per_ex_ce, self.penalty_P),
            torch.ones_like(per_ex_ce)
        )
        per_ex_loss = per_ex_ce * mult
        loss = per_ex_loss.mean()

        if return_outputs:
            return (loss, outputs)
        return loss

# ### ---------------------------------------------------------------------- ###
# ### Setup SFT trainer and start training
# ### ---------------------------------------------------------------------- ###
training_args = SFTConfig(
    output_dir=output_dir,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=learning_rate,
    logging_steps=logging_steps,
    num_train_epochs=epochs,
    max_seq_length=max_seq_length,
    label_names=["input_ids"],  # harmless; TRL handles labels internally
    run_name=f"SFT-exp-{exp_no}",
    logging_first_step=True,
    save_steps=10,
    save_only_model=True,
    remove_unused_columns=True,
)

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    modules_to_save=["lm_head"],
    task_type="CAUSAL_LM",
)

trainer = TrainerWithRefusalRule(
    model=model,
    train_dataset=train_dts,
    peft_config=peft_config,
    args=training_args,
    data_collator=collator,
    penalty_P=PENALTY_P,        # your single multiplicative penalty
)

trainer.train()

!rm -rf /cache/huggingface/transformers
!rm -rf ~/.cache/torch
