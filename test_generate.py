import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen3-8B"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float32,
    device_map="auto",
)

print("Model loaded!")

prompt = "Hãy đặt một câu tiếng Việt tự nhiên có chứa từ 'cứu'."

inputs = tokenizer(
    prompt,
    return_tensors="pt"
)

print("Generating...")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=30,
        do_sample=False,
    )

print("Generated!")

result = tokenizer.decode(
    outputs[0],
    skip_special_tokens=True
)

print(result)