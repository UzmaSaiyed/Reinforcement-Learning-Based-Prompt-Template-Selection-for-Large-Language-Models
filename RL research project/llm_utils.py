"""
llm_utils.py
This file loads a small AI model and lets us ask it questions.

IMPORTANT: Run this in Google Colab (Runtime > Change runtime type > GPU),
not on a regular computer without a GPU - it will be very slow otherwise.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# We use a SMALL model so it runs fast and fits on a free Colab GPU.
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

print(f"Loading model: {MODEL_NAME} ...")
print("(This downloads the model the first time - may take a few minutes)")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,   # uses less memory, faster on GPU
    device_map="auto"            # automatically uses GPU if available
)

print("Model loaded successfully!")


def generate_response(prompt: str, max_new_tokens: int = 200) -> str:
    """
    Sends a prompt to the LLM and returns its text response.

    Args:
        prompt: the full text prompt (already filled in with a template)
        max_new_tokens: max length of the response (keeps things fast)

    Returns:
        The model's text response as a string.
    """
    # Format as a chat message (this model expects chat-style input)
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Only decode the NEW tokens (the response), not the input prompt
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return response.strip()


if __name__ == "__main__":
    # Quick test
    test_prompt = "Explain step-by-step: How do airplanes fly?"
    print(f"\nPrompt: {test_prompt}\n")
    response = generate_response(test_prompt)
    print(f"Response:\n{response}")
