"""
DBHS Fine-Tuning Pipeline v2 — Corrected & Production-Ready

Changes from v1:
1. Model: distilgpt2 → Qwen2.5-0.5B-Instruct (better for knowledge tasks)
2. Format: Unified to use model-native chat template
3. Packing: Disabled to prevent sample contamination
4. Loss objective: full prompt training for TRL 1.6.0 (assistant-only masking not yet applied)
5. Generation: Fixed do_sample logic
6. Data: Added validation & duplicate detection
"""

import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model
import json
from typing import Dict, List

# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"  # CHANGED: Was distilgpt2
BASE_MODEL = MODEL_NAME
MAX_LENGTH = 2048
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")
if DEVICE == "cuda":
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    torch.backends.cudnn.benchmark = True

# ============================================================
# Load Model
# ============================================================

print("\n[*] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Use model's built-in chat template (Qwen2.5-Instruct has one)
print(f"[*] Chat template available: {tokenizer.chat_template is not None}")

print("\n[*] Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    trust_remote_code=True,
)

# ============================================================
# LoRA Configuration
# ============================================================

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # CHANGED: Was c_attn, c_proj (GPT-2 specific)
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.to(DEVICE)
print(f"[*] LoRA applied. Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# ============================================================
# Load Dataset
# ============================================================

print("\n[*] Loading dataset...")
dataset = load_dataset("json", data_files="train.jsonl", split="train")
print(f"[*] Loaded {len(dataset)} examples")

# ============================================================
# Data Validation & Deduplication
# ============================================================

def deduplicate_dataset(dataset, max_examples=None):
    """
    Remove duplicates by question text.
    Also validate data quality.
    """
    seen_questions = {}
    deduplicated = []
    
    for i, example in enumerate(dataset):
        if max_examples and len(deduplicated) >= max_examples:
            break
            
        messages = example.get("messages", [])
        if not isinstance(messages, list) or len(messages) < 2:
            continue
        
        # Extract question (first user message)
        question = messages[0].get("content", "").strip().lower()
        if not question or len(question) < 5:
            continue
        
        # Skip if exact question already seen
        if question in seen_questions:
            continue
        
        seen_questions[question] = i
        deduplicated.append(example)
    
    removed = len(dataset) - len(deduplicated)
    print(
        f"[*] After deduplication: {len(dataset)} -> {len(deduplicated)} "
        f"(removed {removed})"
    )
    return Dataset.from_dict({k: [ex[k] for ex in deduplicated] for k in deduplicated[0].keys()})

dataset = deduplicate_dataset(dataset)

# ============================================================
# Format Dataset Using Chat Template
# ============================================================

def format_chat_for_training(example):
    """
    Format messages using the model's chat template.
    Qwen2.5-Instruct expects:
        [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    """
    messages = example.get("messages", [])
    if not isinstance(messages, list):
        messages = [{"role": "user", "content": str(messages)}]
    
    # Apply chat template
    if tokenizer.chat_template:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
    else:
        # Fallback to structured format
        text = "\n".join(
            f"<|{m['role']}|>\n{m['content']}<|end|>"
            for m in messages
        )
    
    return {"text": text}

print("\n[*] Formatting dataset...")
dataset = dataset.map(format_chat_for_training, remove_columns=dataset.column_names)

# Show a sample
print("\n[*] Example formatted training text:")
print("=" * 60)
print(dataset[0]["text"][:500])
print("=" * 60)

# ============================================================
# Token length analysis
# ============================================================

print("\n[*] Analyzing token lengths...")
lengths = []
for item in dataset.select(range(min(300, len(dataset)))):
    lengths.append(len(tokenizer(item["text"])["input_ids"]))

lengths_sorted = sorted(lengths)
print(f"Avg length: {sum(lengths)/len(lengths):.1f}")
print(f"Max sampled length: {max(lengths)}")
for p in [50, 90, 95, 99]:
    idx = int(len(lengths) * p / 100)
    print(f"{p}th percentile: {lengths_sorted[idx]}")

# ============================================================
# Train/Validation Split
# ============================================================

print("\n[*] Creating train/eval split...")
split = dataset.train_test_split(test_size=0.05, seed=42)
train_dataset = split["train"]
eval_dataset = split["test"]

print(f"[*] Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

# ============================================================
# Trainer Configuration
# ============================================================

print("\n[*] Configuring trainer...")
training_args = SFTConfig(
    output_dir="dbhs_model_v2",
    num_train_epochs=3,
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    warmup_ratio=0.1,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    optim="adamw_torch",
    bf16=False,
    fp16=True if DEVICE == "cuda" else False,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="none",
    max_length=MAX_LENGTH,
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=training_args,
)

# ============================================================
# Train
# ============================================================

print("\n[*] Starting training...")
print("This may take 30+ minutes on GPU...\n")
trainer.train()

# ============================================================
# Save Adapter
# ============================================================

print("\n[*] Saving LoRA adapter...")
model.save_pretrained("dbhs_lora_v2")
tokenizer.save_pretrained("dbhs_lora_v2")

print("[✓] Training complete. Adapter saved to dbhs_lora_v2/")
print("[✓] Base model: Qwen/Qwen2.5-0.5B-Instruct")
print("[✓] Use main_v2.py for inference with this adapter.")
