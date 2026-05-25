import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen3.6-27B-FP8 text-only smoke test.")
    parser.add_argument("--model", default="~/huggingface/Qwen3.6-27B-FP8")
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--prompt", default="你好，请用三句话介绍你自己。")
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--max-batched-tokens", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--eager", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.devices)

    import torch._dynamo
    from transformers import AutoTokenizer

    from nanovllm import LLM, SamplingParams

    torch._dynamo.config.cache_size_limit = 64
    model_path = os.path.expanduser(args.model)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    prompt = tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": args.prompt,
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    print("=== PROMPT ===", flush=True)
    print(prompt, flush=True)

    llm = LLM(
        model_path,
        enable_vision=False,
        enforce_eager=args.eager,
        tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_batched_tokens,
        max_num_seqs=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    outputs = llm.generate(
        [prompt],
        SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens),
    )

    print("\n=== RAW OUTPUT ===", flush=True)
    print(outputs, flush=True)
    print("\n=== TEXT ===", flush=True)
    print(outputs[0]["text"].replace("<|im_end|>", ""), flush=True)


if __name__ == "__main__":
    main()
