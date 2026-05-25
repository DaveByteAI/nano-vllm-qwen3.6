"""End-to-end test for nano-vllm multimodal support (Qwen3.5-9B)."""
import os
import sys

MODEL_PATH = os.path.expanduser("~/huggingface/Qwen3.5-9B")

print("=" * 60, flush=True)
print("Test 1: Text-only (regression)", flush=True)
print("=" * 60, flush=True)

from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
llm = LLM(MODEL_PATH, enforce_eager=True)

prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Hello, who are you?"}],
    tokenize=False,
    add_generation_prompt=True,
)
outputs = llm.generate(
    [prompt],
    SamplingParams(temperature=0.7, max_tokens=4096),
)
print("Text-only output:", flush=True)
for o in outputs:
    print(o["text"], flush=True)
print(flush=True)

print("=" * 60, flush=True)
print("Test 2: Multimodal (image + text)", flush=True)
print("=" * 60, flush=True)

from PIL import Image

image = Image.open(os.path.expanduser("~/image_demo.jpg"))

messages = [{"role": "user", "content": [
    {"type": "image", "image": image},
    {"type": "text", "text": "Describe this image in detail."},
]}]

outputs = llm.generate(
    [messages],
    SamplingParams(temperature=0.7, max_tokens=4096),
)
print("Multimodal output:", flush=True)
for o in outputs:
    print(o["text"], flush=True)
print(flush=True)

print("All tests complete!", flush=True)
