import os
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
import pathlib
from unsloth import FastLanguageModel
import torch
from unsloth.chat_templates import get_chat_template
from unsloth.chat_templates import standardize_sharegpt
from unsloth.chat_templates import train_on_responses_only
from datasets import load_dataset
from unsloth import is_bfloat16_supported
import sys, argparse


def main():
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

    dataset = dataset.map(formatting_prompts_func, batched = True, num_proc = None)
    print("--" * 20)
    print("Formatted, row[0]:")
    print(dataset[0])

    # ---- CPU debug: verify masking without SFTTrainer ----
    # On CPU we cannot construct a real SFTTrainer (needs a model),
    # but we can verify that train_on_responses_only masking works
    # by using it in return_function mode.

    # 1. Tokenize the formatted text
    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=max_seq_length, padding=False)

    tokenized_dataset = dataset.map(tokenize_fn, batched = True, num_proc = None, remove_columns = dataset.column_names)
    print("--" * 20)
    print("Tokenized, row[0] keys:", list(tokenized_dataset[0].keys()))
    print("Tokenized, row[0] input_ids length:", len(tokenized_dataset[0]["input_ids"]))

    # 2. Apply train_on_responses_only masking (return_function mode)
    mask_fn = train_on_responses_only(
        trainer = None,
        tokenizer = tokenizer,
        instruction_part = "<|im_start|>user\n",
        response_part = "<|im_start|>assistant\n",
        return_function = True,
    )

    masked_dataset = tokenized_dataset.map(mask_fn, batched = True, num_proc = None)
    print("--" * 20)
    print("Masked labels, row[0]:")
    labels = masked_dataset[0]["labels"]
    input_ids = masked_dataset[0]["input_ids"]
    print(f"  Total tokens: {len(labels)}")
    print(f"  Masked (instruction) tokens: {labels.count(-100)}")
    print(f"  Unmasked (response) tokens: {len(labels) - labels.count(-100)}")

    # Show which tokens are masked vs unmasked
    print("\n  Token-by-token view:")
    for i, (tid, lab) in enumerate(zip(input_ids, labels)):
        decoded = tokenizer.decode([tid])
        status = "MASK" if lab == -100 else "TRAIN"
        print(f"    [{i:3d}] {status} | id={tid:6d} | {repr(decoded)}")

    print("\n" + "=" * 60)
    print("CPU DEBUG PIPELINE COMPLETE - all data processing verified!")
    print("=" * 60)


if __name__ == "__main__":
    main()