import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# script to test chatting with actsvd pruned llama models


def chat(model_path, tokenizer_path=None, system_prompt=None):
    if tokenizer_path is None:
        tokenizer_path = model_path

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Check if model has a chat template
    has_chat_template = tokenizer.chat_template is not None
    if not has_chat_template:
        print("Warning: No chat template found. Using raw prompt.")

    # Gemma needs a system prompt
    if system_prompt is None and "gemma" in model_path.lower():
        system_prompt = "You are a helpful assistant."

    print("Model loaded. Type 'quit' to exit.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break

        if has_chat_template:
            messages = [{"role": "user", "content": user_input}]
            if system_prompt:
                messages.insert(0, {"role": "system", "content": system_prompt})
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            # Fallback for models without chat template
            text = f"User: {user_input}\nAssistant:"

        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        if has_chat_template:
            # Extract only the model's response
            response = response[len(text) :].strip()
        else:
            # Fallback extraction
            response = response[len(text) :].strip()

        print(f"Assistant: {response}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to modified model"
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Path to tokenizer (defaults to model_path)",
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default=None,
        help="System prompt to use (optional)",
    )
    args = parser.parse_args()

    chat(args.model_path, args.tokenizer_path, args.system_prompt)
