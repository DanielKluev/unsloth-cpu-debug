import os
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
import pathlib
from unsloth import FastLanguageModel
import torch
from unsloth.chat_templates import get_chat_template
from unsloth.chat_templates import standardize_sharegpt
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth import is_bfloat16_supported
from unsloth.chat_templates import train_on_responses_only
import sys, os, argparse


## ================================================================================================
## =============                    Setup arguments            ====================================

model_id = "qwen3_8b"
train_mode = "SFT"
parser = argparse.ArgumentParser(description="Train LLM with Unsloth")
parser.add_argument("--output-dir", "-d", type=str, help="Path where result model will be saved", default="AUTO")
parser.add_argument("--rank", "-r", type=int, help="LoRA rank - from 1 to 512", default=16)
parser.add_argument("--epochs", "-e", type=int, help="Total training epochs", default=1)
parser.add_argument("--batch", "-b", type=int, help="One step batch size", default=6)
parser.add_argument("--batch-accumulation", "-a", type=int, help="Batch accumulation, working as batch size multiplier", default=1)
parser.add_argument("--modules", "-m", type=str, help="Layers/modules to train. Valid values: q, k, v, o, g, u, d or combinations, like qv gu", default="o")
parser.add_argument("--version", "-v", type=str, help="Version marker for the model to be trained", default="V1")
parser.add_argument("--merge", action="store_true", help="Create merged model file.")
#parser.add_argument("dataset_path", type=str, help="Path to the dataset file.")
parsed_args = parser.parse_args()

max_seq_length = 4096 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.
load_in_8bit = False

truncate_dataset = False  # Set to True to limit dataset size for quick testing
sanity_run = False  # Set to True to run a quick sanity check with a small dataset
total_epochs = parsed_args.epochs
output_base_dir = parsed_args.output_dir
dataset_path = "sft_text.jsonl"
batch_size = parsed_args.batch
batch_accumulation = parsed_args.batch_accumulation
lora_rank = parsed_args.rank
training_version = parsed_args.version

available_layers = {"o":"o_proj", "v":"v_proj", "q":"q_proj", "k":"k_proj", "g":"gate_proj", "u":"up_proj", "d":"down_proj"}
lora_target_modules = []
for module_alias in parsed_args.modules:
    lora_target_modules.append(available_layers[module_alias])

if output_base_dir == "AUTO":
    output_base_dir = f"{model_id}_{training_version}_{train_mode}_{parsed_args.modules}_r{lora_rank}_e{total_epochs}"
    
## ================================================================================================
## =============                    Print arguments            ====================================

print("=" * 60)
print(f" Will run Unsloth to fine-tune {model_id} in {train_mode} mode.")
print(f" Epochs to train: {total_epochs}")
print(f" Batch: Batch {batch_size} x Accumulation {batch_accumulation} = Effective {batch_size * batch_accumulation}")
print(f" LoRA rank: {lora_rank}")
print(f" LoRA modules: {lora_target_modules}")
print(f" Dataset: {dataset_path}")
print(f" Output: {output_base_dir}")
print("=" * 60)
    
model_local_path = pathlib.Path(".").resolve() / "qwen3_tokenizer"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_local_path,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    load_in_8bit = load_in_8bit,
    full_finetuning = False,
)

print(f"Tokenizer: {tokenizer}")

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "qwen-3",
)

def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False).removeprefix('<bos>') for convo in convos]
    return { "text" : texts, }

dataset = load_dataset("json", data_files=dataset_path, split="train")
print("--" * 20)
print("Original, row[0]:")
print(dataset[0])

dataset = standardize_sharegpt(dataset)
print("--" * 20)
print("Standardized, row[0]:")
print(dataset[0])

dataset = dataset.map(formatting_prompts_func, batched = True,)
print("--" * 20)
print("Formatted, row[0]:")
print(dataset[0])

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = batch_size,
        gradient_accumulation_steps = batch_accumulation,
        warmup_steps = 1,
        num_train_epochs = total_epochs, # Set this for 1 full training run.
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none", # Use this for WandB etc
    ),
)

trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|im_start|>user\n",
    response_part = "<|im_start|>assistant\n",
)