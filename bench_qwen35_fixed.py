import os
import argparse
from time import perf_counter

import torch
import torch._dynamo
from transformers import AutoTokenizer

from nanovllm import LLM, SamplingParams


torch._dynamo.config.cache_size_limit = 64


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="~/huggingface/Qwen3.5-9B")
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--eager", action="store_true")
    args = parser.parse_args()

    model_path = os.path.expanduser(args.model)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "你好，请用中文简单介绍一下你自己。"}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    llm = LLM(
        model_path,
        enforce_eager=args.eager,
        tensor_parallel_size=args.tp,
        max_model_len=1024,
        max_num_batched_tokens=1024,
        max_num_seqs=1,
        gpu_memory_utilization=0.90,
    )

    warmup_params = SamplingParams(temperature=args.temperature, max_tokens=8, ignore_eos=True)
    llm.generate([prompt], warmup_params, use_tqdm=False)
    torch.cuda.synchronize()

    params = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens, ignore_eos=True)
    for i in range(args.repeats):
        llm.add_request(prompt, params)
        prefill_time = 0.0
        decode_time = 0.0
        prefill_tokens = 0
        decode_tokens = 0
        while not llm.is_finished():
            torch.cuda.synchronize()
            start = perf_counter()
            _, num_tokens = llm.step()
            torch.cuda.synchronize()
            elapsed = perf_counter() - start
            if num_tokens > 0:
                prefill_time += elapsed
                prefill_tokens += num_tokens
            else:
                decode_time += elapsed
                decode_tokens += -num_tokens
        total_time = prefill_time + decode_time
        print(
            f"run={i + 1} tp={args.tp} "
            f"temp={args.temperature:g} "
            f"prefill={prefill_tokens / prefill_time:.2f} tok/s "
            f"decode={decode_tokens / decode_time:.2f} tok/s "
            f"total={decode_tokens / total_time:.2f} out_tok/s "
            f"prefill_time={prefill_time:.3f}s decode_time={decode_time:.3f}s"
        )


if __name__ == "__main__":
    main()
