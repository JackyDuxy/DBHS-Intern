"""
DBHS Fine-Tuning Pipeline v2 — Corrected & Production-Ready

Changes from v1:
1. Model: distilgpt2 → Qwen2.5-1.5B-Instruct (better factual recall & instruction following)
2. Format: Unified to use model-native chat template
3. Packing: Disabled to prevent sample contamination
4. Loss objective: full prompt training for TRL 1.6.0 (assistant-only masking not yet applied)
5. Generation: Fixed do_sample logic
6. Data: Added validation & duplicate detection
7. Quantization: QLoRA (4-bit) for memory efficiency

Optimized for RTX 5060Ti (16GB VRAM):
- QLoRA (4-bit quantization) reduces memory usage by 50-60%
- Gradient checkpointing enabled for efficiency
- Batch size: 2 with gradient accumulation 4 (effective batch size 8)
- MAX_LENGTH: 2048 tokens (full length supported)
- SDPA attention (no FlashAttention dependencies)
"""

import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import json
from typing import Dict, List

# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"  # 1.5B better for DBHS knowledge tasks (better recall & reasoning)
BASE_MODEL = MODEL_NAME
MAX_LENGTH = 2048  # RTX 5060Ti 16GB can handle full length with QLoRA
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_QLORA = True  # Use 4-bit quantization for memory efficiency

print(f"Using device: {DEVICE}")
if DEVICE == "cuda":
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    torch.backends.cudnn.benchmark = True
    # Enable memory-efficient settings for RTX 5060Ti
    torch.cuda.empty_cache()

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

if USE_QLORA:
    # QLoRA: 4-bit quantization to reduce memory by 50-60%
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        trust_remote_code=True,
        device_map="auto",
    )
    # Prepare model for kbit training (QLoRA)
    model = prepare_model_for_kbit_training(model)
else:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        trust_remote_code=True,
    )
    model.to(DEVICE)

# Enable gradient checkpointing for memory efficiency
model.gradient_checkpointing_enable()

# ============================================================
# LoRA Configuration
# ============================================================

lora_config = LoraConfig(
    r=64,                          # INCREASED 16→64: more capacity for fact memorization
    lora_alpha=128,                # 2x rank ratio
    target_modules=[               # EXPANDED: attention + all FFN layers
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
if not USE_QLORA:
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
    num_train_epochs=5,            # INCREASED 3→5: more passes for fact retention
    learning_rate=3e-5,            # LOWERED slightly for stability with larger LoRA
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,  # Effective batch size: 8
    warmup_ratio=0.1,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    optim="adamw_torch",
    bf16=False,
    fp16=False if USE_QLORA else (True if DEVICE == "cuda" else False),  # Disable fp16 with QLoRA to avoid precision issues
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
    gradient_checkpointing=True,
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
print("[✓] Base model: Qwen/Qwen2.5-1.5B-Instruct (with QLoRA 4-bit quantization)")
print("[✓] Use main_v2.py for inference with this adapter.")
if USE_QLORA:
    print("[✓] Memory-efficient QLoRA training (4-bit quantization enabled)")
