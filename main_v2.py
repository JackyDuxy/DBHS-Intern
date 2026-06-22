"""
DBHS Inference Pipeline v2 — Corrected & Production-Ready

Changes from v1:
1. Model format unified: Uses Qwen2.5-Instruct's native chat template
2. Generation settings fixed: Proper do_sample logic
3. Prompt building: Uses tokenizer.apply_chat_template()
4. Repetition penalty and token limits improved
"""

'''
The current fine-tuned DBHS assistant demonstrates that the LoRA adapter is loading successfully and influencing model behavior; however, the model frequently generates plausible but incorrect school-related information instead of accurately retrieving DBHS-specific facts. This suggests that the primary issue is not model loading or inference, but rather insufficient knowledge retention during fine-tuning. Although the training dataset contains correct DBHS information, critical facts such as staff names, coordinator roles, and school identifiers appear too infrequently relative to the model's extensive pretraining knowledge. As a result, when confidence is low, the model defaults to generating realistic-looking educational information based on its prior knowledge rather than recalling the intended DBHS facts. The problem is likely exacerbated by limited reinforcement of key facts, synthetic dataset distribution, and the use of fine-tuning alone for knowledge storage. Future improvements should focus on increasing fact coverage and repetition, strengthening LoRA adaptation, improving dataset quality, and potentially incorporating a retrieval-augmented generation (RAG) system to provide reliable access to authoritative DBHS information.
'''

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"  # CHANGED: Was distilgpt2
ADAPTER_DIR = "dbhs_lora_v2"
MAX_SEQ_LENGTH = 2048


def load_model(adapter_dir: str = ADAPTER_DIR, base_model: str = BASE_MODEL):
    """Load the base model, tokenizer, and LoRA adapter."""
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

    print(f"[*] Loading LoRA adapter from {adapter_dir}...")
    model = PeftModel.from_pretrained(
        model,
        adapter_dir,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    print(type(model))

    model.config.pad_token_id = tokenizer.eos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    if model.config.bos_token_id is None:
        model.config.bos_token_id = tokenizer.bos_token_id

    model.eval()
    return model, tokenizer


def build_prompt(user_message: str) -> list[dict]:
    """Build a prompt message list aligned with Qwen chat format."""
    return [
        {
            "role": "system",
            "content": "You are a helpful assistant providing information about the DBHS program. Answer questions based on the provided information and be concise and accurate. Ensure the legibility of your responses and avoid unnecessary repetition. If you don't know the answer, say you don't know.",
        },
        {"role": "user", "content": user_message},
    ]


def generate_response(
    model,
    tokenizer,
    user_message: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.2,
) -> str:
    """
    Generate a response from the model.
    
    Args:
        model: The fine-tuned model
        tokenizer: The tokenizer
        user_message: The user's question/prompt
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0.0 = greedy, higher = more random)
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        repetition_penalty: Penalty for repeating tokens
    
    Returns:
        str: Generated response text
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    messages = build_prompt(user_message)

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

    # Generate
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
    """
    Clean up response text.
    Remove model formatting, stop sequences, etc.
    """
    response = response.strip()
    
    # Remove common stop patterns
    stop_patterns = [
        "user:",
        "assistant:",
        "<|user|>",
        "<|assistant|>",
        "<|end|>",
        "[END]",
        "[STOP]",
    ]
    
    lowered = response.lower()
    earliest_stop = len(response)
    
    for pattern in stop_patterns:
        idx = lowered.find(pattern.lower())
        if idx != -1 and idx < earliest_stop:
            earliest_stop = idx
    
    if earliest_stop < len(response):
        response = response[:earliest_stop].strip()
    
    # Remove extra whitespace
    response = " ".join(response.split())
    
    return response


def interactive_chat(model, tokenizer):
    """
    Interactive chat loop for testing.
    Type 'quit' to exit.
    """
    print("\n" + "=" * 60)
    print("DBHS Information Assistant (Type 'quit' to exit)")
    print("=" * 60 + "\n")

    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        
        if not user_input:
            continue
        
        print("\n[Generating response...]")
        response = generate_response(
            model,
            tokenizer,
            user_input,
            max_new_tokens=256,
            temperature=0.5,  # Lower temperature for factual consistency
            repetition_penalty=1.3,
        )
        response = normalize_response(response)
        
        print(f"\nAssistant: {response}\n")


if __name__ == "__main__":
    import sys
    
    # Load model once
    model, tokenizer = load_model()
    
    if len(sys.argv) > 1:
        # Command-line mode
        user_query = " ".join(sys.argv[1:])
        response = generate_response(
            model,
            tokenizer,
            user_query,
            max_new_tokens=256,
            temperature=0.5,
            repetition_penalty=1.3,
        )
        response = normalize_response(response)
        print(response)
    else:
        # Interactive mode
        interactive_chat(model, tokenizer)
