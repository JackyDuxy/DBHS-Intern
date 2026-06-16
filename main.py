import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL_DIR = "dbhs_lora"
BASE_MODEL = "distilgpt2"
MAX_HISTORY_TURNS = 3


def load_model(model_dir: str = MODEL_DIR, base_model: str = BASE_MODEL):
    """Load the base model, tokenizer, and PEFT adapter from disk."""
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(base_model)
    model = PeftModel.from_pretrained(model, model_dir)
    model.config.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer


def format_role(role: str) -> str:
    if role == "user":
        return "### Human:"
    if role == "assistant":
        return "### Assistant:"
    return f"### {role.capitalize()}:"


def build_prompt(messages):
    """Build a prompt string from a chat history or raw text input."""
    if isinstance(messages, str):
        return messages.strip()

    if isinstance(messages, list):
        lines = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            lines.append(f"{format_role(role)} {content}")
        lines.append("### Assistant:")
        return "\n".join(lines)

    return str(messages)


def normalize_response(response: str) -> str:
    response = response.strip()
    stop_tokens = ["### Human:", "### Assistant:", "user:", "assistant:"]
    lowered = response.lower()
    for token in stop_tokens:
        idx = lowered.find(token)
        if idx != -1:
            response = response[:idx].strip()
            break
    return response


def generate_response(
    model,
    tokenizer,
    prompt,
    max_new_tokens=150,
    temperature=0.4,
    top_p=0.9,
    repetition_penalty=1.1,
    do_sample=False,
):
    """Generate a response from the model for a given prompt."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    )
    input_ids = inputs["input_ids"].to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=do_sample,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[0][input_ids.shape[-1] :]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return normalize_response(response)


def interactive_chat(model, tokenizer):
    """Start an interactive chat loop with the trained model."""
    history = []
    print("Loaded model. Type 'exit' or 'quit' to stop.")

    while True:
        user_input = input("User: ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        history.append({"role": "user", "content": user_input})
        prompt = build_prompt(history[-MAX_HISTORY_TURNS * 2 :])
        response = generate_response(model, tokenizer, prompt)

        print(f"Assistant: {response}\n")
        history.append({"role": "assistant", "content": response})


def main():
    if not os.path.isdir(MODEL_DIR):
        raise FileNotFoundError(f"Model directory '{MODEL_DIR}' not found.")

    model, tokenizer = load_model()
    interactive_chat(model, tokenizer)


if __name__ == "__main__":
    main()
