"""
DBHS Inference Pipeline v2 — with RAG retrieval

Architecture:
  1. At startup: load structured DB (dbhs_structured_db.json) into memory
  2. At query time: keyword-match query against all records, score by overlap
  3. Inject top-3 matching records as context into system prompt
  4. Run fine-tuned Qwen LoRA adapter on the grounded prompt

This hybrid (RAG + fine-tuning) prevents hallucination: the model receives
authoritative DBHS facts in context rather than relying on memorized weights.

Diagnostic note embedded by user:
  The fine-tuned model loads correctly but halluccinates plausible school
  info instead of recalling DBHS facts. Root cause: critical facts appear
  too infrequently vs. pretraining priors. RAG grounds each response in
  the source-of-truth structured DB without requiring retraining.
"""

import os
import re
import json
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"  # Must match train_v2.py
ADAPTER_DIR = "dbhs_lora_v2"
DB_FILE = "dbhs_structured_db.json"
MAX_SEQ_LENGTH = 2048
MAX_CONTEXT_RECORDS = 3          # top-N records injected per query
MAX_CONTEXT_CHARS = 1200         # cap context block size


# ============================================================
# RAG — Structured DB Retrieval
# ============================================================

def load_db(path: str = DB_FILE) -> list[dict]:
    """Load the structured DB records into memory once at startup."""
    if not Path(path).exists():
        print(f"[WARN] DB file not found: {path}. RAG disabled.")
        return []
    raw = json.load(open(path, encoding="utf-8"))
    records = raw.get("records", raw) if isinstance(raw, dict) else raw
    print(f"[*] RAG: loaded {len(records)} records from {path}")
    return records


def tokenize_query(text: str) -> set[str]:
    """Lowercase word tokens, 3+ chars, common stopwords removed."""
    stopwords = {
        "the", "and", "for", "are", "that", "this", "with", "from",
        "can", "what", "who", "how", "does", "which", "when", "where",
        "about", "have", "has", "you", "your", "tell", "give", "me",
        "please", "is", "at", "of", "to", "a", "an", "in", "on",
        "do", "i", "my", "be", "it", "its",
    }
    words = re.findall(r'\b[a-z]{3,}\b', text.lower())
    return {w for w in words if w not in stopwords}


def score_record(record: dict, query_tokens: set[str]) -> int:
    """Score a DB record by how many query tokens appear in its searchable text."""
    # Build a flat text blob from all searchable fields
    parts = [
        record.get("title", ""),
        record.get("summary", ""),
        " ".join(record.get("keywords", [])),
        record.get("name", ""),
        record.get("role", ""),
        record.get("department", ""),
        record.get("group", ""),
        record.get("entity_type", ""),
        record.get("category", ""),
    ]
    # Include nested facts dict if present
    facts = record.get("facts", {})
    if isinstance(facts, dict):
        parts += list(facts.values())

    blob = " ".join(str(p) for p in parts if p).lower()

    # Count exact keyword hits + partial (substring) hits
    score = 0
    for token in query_tokens:
        if re.search(r'\b' + re.escape(token) + r'\b', blob):
            score += 2                    # exact word match
        elif token in blob:
            score += 1                    # substring match

    return score


def retrieve(query: str, records: list[dict], top_n: int = MAX_CONTEXT_RECORDS) -> list[dict]:
    """Return the top_n most relevant records for a query."""
    if not records:
        return []
    tokens = tokenize_query(query)
    if not tokens:
        return []
    scored = [(score_record(r, tokens), r) for r in records]
    scored.sort(key=lambda x: x[0], reverse=True)
    # Only return records with at least 1 match
    return [r for score, r in scored[:top_n] if score > 0]


def format_record_for_context(r: dict) -> str:
    """Render a DB record as a compact, readable context snippet."""
    lines = []
    if r.get("title"):
        lines.append(f"[{r['title']}]")
    if r.get("summary"):
        lines.append(r["summary"])

    # Staff-specific fields
    for field in ["name", "role", "department", "phone", "fax", "email_note",
                  "mailing_address", "location", "credentials", "student_range",
                  "appointment_url"]:
        val = r.get(field)
        if val:
            lines.append(f"{field.replace('_', ' ').title()}: {val}")

    # Facts dict (school overview, CEEB, etc.)
    facts = r.get("facts", {})
    if isinstance(facts, dict):
        for k, v in facts.items():
            if isinstance(v, (str, int, float)):
                lines.append(f"{k.replace('_', ' ').title()}: {v}")

    # CEEB quick_fact
    if r.get("ceeb_code"):
        lines.append(f"CEEB Code: {r['ceeb_code']}")

    # Coordinators / members
    for key in ["coordinators", "members", "responsibilities"]:
        vals = r.get(key, [])
        if vals:
            lines.append(f"{key.title()}: {', '.join(str(v) for v in vals)}")

    text = "\n".join(lines)
    # Cap to avoid overflowing context window
    return text[:MAX_CONTEXT_CHARS]


def build_context_block(query: str, records: list[dict]) -> str:
    """Build the RAG context string to inject into the system prompt."""
    relevant = retrieve(query, records)
    if not relevant:
        return ""
    snippets = [format_record_for_context(r) for r in relevant]
    return "DBHS Reference Information:\n\n" + "\n\n---\n\n".join(snippets)


# ============================================================
# Model Loading
# ============================================================

def load_model(adapter_dir: str = ADAPTER_DIR, base_model: str = BASE_MODEL):
    print(f"[*] Loading tokenizer from {base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[*] Loading base model {base_model}...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )

    adapter_path = Path(adapter_dir)
    if adapter_path.exists():
        print(f"[*] Loading LoRA adapter from {adapter_dir}...")
        model = PeftModel.from_pretrained(
            model,
            adapter_dir,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
    else:
        print(f"[WARN] Adapter not found at {adapter_dir}. Running base model only.")

    model.config.pad_token_id = tokenizer.eos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    if model.config.bos_token_id is None:
        model.config.bos_token_id = tokenizer.bos_token_id

    model.eval()
    return model, tokenizer


# ============================================================
# Prompt Building (with RAG context)
# ============================================================

BASE_SYSTEM_PROMPT = (
    "You are Madame Ayme-Johnson, the official AI assistant for Diamond Bar High School (DBHS). "
    "Answer questions accurately using only the information provided in the DBHS Reference Information below. "
    "If the answer is not in the reference information and you are not certain, say you don't know rather than guessing. "
    "Be concise and factual."
)


def build_prompt(user_message: str, db_records: list[dict]) -> list[dict]:
    context = build_context_block(user_message, db_records)

    system_content = BASE_SYSTEM_PROMPT
    if context:
        system_content = BASE_SYSTEM_PROMPT + "\n\n" + context

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_message},
    ]


# ============================================================
# Generation
# ============================================================

def generate_response(
    model,
    tokenizer,
    user_message: str,
    db_records: list[dict],
    max_new_tokens: int = 300,
    temperature: float = 0.3,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.2,
) -> str:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    messages = build_prompt(user_message, db_records)

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding="longest",
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids)).to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature if temperature > 0 else 1.0,
            top_p=top_p,
            top_k=top_k,
            do_sample=temperature > 0,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    generated_ids = outputs[0][input_ids.shape[-1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return response


def normalize_response(response: str) -> str:
    stop_patterns = [
        "user:", "assistant:", "<|user|>", "<|assistant|>", "<|end|>",
        "[END]", "[STOP]",
    ]
    lowered = response.lower()
    earliest = len(response)
    for pattern in stop_patterns:
        idx = lowered.find(pattern.lower())
        if idx != -1 and idx < earliest:
            earliest = idx
    if earliest < len(response):
        response = response[:earliest].strip()
    return " ".join(response.split())


# ============================================================
# Interactive Chat
# ============================================================

def interactive_chat(model, tokenizer, db_records: list[dict]):
    print("\n" + "=" * 60)
    print("DBHS Information Assistant — Madame Ayme-Johnson")
    print("(Type 'quit' to exit, 'rag <query>' to inspect retrieval)")
    print("=" * 60 + "\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in ["quit", "exit", "q"]:
            print("Au revoir!")
            break

        if not user_input:
            continue

        # Debug mode: show what RAG would retrieve
        if user_input.lower().startswith("rag "):
            query = user_input[4:].strip()
            relevant = retrieve(query, db_records)
            print(f"\n[RAG] Top {len(relevant)} records for: '{query}'")
            for r in relevant:
                print(f"  - {r.get('title', r.get('id', '?'))}")
            print()
            continue

        print("\n[Generating response...]")
        response = generate_response(
            model, tokenizer, user_input, db_records,
            max_new_tokens=300,
            temperature=0.3,
            repetition_penalty=1.2,
        )
        response = normalize_response(response)
        print(f"\nMadame Ayme-Johnson: {response}\n")


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    import sys

    db_records = load_db(DB_FILE)
    model, tokenizer = load_model()

    if len(sys.argv) > 1:
        user_query = " ".join(sys.argv[1:])
        response = generate_response(
            model, tokenizer, user_query, db_records,
            max_new_tokens=300,
            temperature=0.3,
            repetition_penalty=1.2,
        )
        response = normalize_response(response)
        print(response)
    else:
        interactive_chat(model, tokenizer, db_records)
