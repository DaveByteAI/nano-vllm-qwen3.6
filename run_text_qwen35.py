import os
from transformers import AutoTokenizer
from nanovllm import LLM, SamplingParams

MODEL_PATH = os.path.expanduser("~/huggingface/Qwen3.5-9B")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
)

prompt = tokenizer.apply_chat_template(
    [
        {
            "role": "user",
            "content": "你好，请用中文简单介绍一下你自己。"
        }
    ],
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)

print("=== PROMPT ===")
print(prompt)

llm = LLM(
    MODEL_PATH,
    enforce_eager=True,
    tensor_parallel_size=1,
    max_model_len=512,
    max_num_batched_tokens=512,
    max_num_seqs=1,
    gpu_memory_utilization=0.90,
)

sampling_params = SamplingParams(
    temperature=0.7,
    max_tokens=512,
)

outputs = llm.generate([prompt], sampling_params)

print("\n=== RAW OUTPUT ===")
print(outputs)

print("\n=== TEXT ===")
try:
    print(outputs[0]["text"])
except Exception:
    try:
        print(outputs[0].text)
    except Exception:
        print(outputs[0])
