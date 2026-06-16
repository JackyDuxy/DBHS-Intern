import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model

MAX_SEQ_LENGTH = 2048
MODEL_NAME = "distilgpt2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")
if DEVICE == "cuda":
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    torch.backends.cudnn.benchmark = True

# ---------------------------------------------------
# Load model
# ---------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
)

# ---------------------------------------------------
# LoRA
# ---------------------------------------------------

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["c_attn", "c_proj"],
    lora_dropout=0.0,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.to(DEVICE)

# ---------------------------------------------------
# Load dataset
# ---------------------------------------------------

dataset = load_dataset(
    "json",
    data_files="train.jsonl",
    split="train",
)

print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
import random

print("1")
indices = random.sample(range(len(dataset)), 20)

for i in indices:
    print(dataset[i])
    print("=" * 80)

max_len = 0
longest = None

for item in dataset:
    length = len(item["messages"][1]["content"])
    if length > max_len:
        max_len = length
        longest = item

print("2")
print(max_len)
print(longest)

count = 0

for item in dataset:
    text = str(item)
    if "Michael" in text:
        count += 1

print("3")
print(count)

# ---------------------------------------------------
# Format chats
# ---------------------------------------------------

def format_chat(example):
    messages = example.get("messages", example)
    if isinstance(messages, list):
        text = "\n".join(
            f"{m['role']}: {m['content']}" for m in messages
        )
    else:
        text = str(messages)

    return {"text": text}

dataset = dataset.map(
    format_chat,
    remove_columns=dataset.column_names,
)

# ---------------------------------------------------
# Train / Validation split
# ---------------------------------------------------

split = dataset.train_test_split(
    test_size=0.05,
    seed=42
)

train_dataset = split["train"]
eval_dataset = split["test"]

# ---------------------------------------------------
# Trainer
# ---------------------------------------------------

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=SFTConfig(
        output_dir="dbhs_model",
        num_train_epochs=3,
        learning_rate=1e-4,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        warmup_ratio=0.05,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        bf16=False,
        fp16=True if DEVICE == "cuda" else False,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        dataset_text_field="text",
        max_length=MAX_SEQ_LENGTH,
        packing=True,
    ),
)

trainer.train()

# ---------------------------------------------------
# Save adapter
# ---------------------------------------------------

model.save_pretrained("dbhs_lora")
tokenizer.save_pretrained("dbhs_lora")