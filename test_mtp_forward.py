import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen3.6 MTP single-step forward smoke test.")
    parser.add_argument("--model", default="~/huggingface/Qwen3.6-27B-FP8")
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--prompt", default="你好，请用三句话介绍你自己。")
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--max-batched-tokens", type=int, default=128)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.devices)

    from transformers import AutoTokenizer

    from nanovllm import LLM, SamplingParams

    model_path = os.path.expanduser(args.model)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    print("=== PROMPT ===", flush=True)
    print(prompt, flush=True)

    llm = LLM(
        model_path,
        enable_vision=False,
        enable_mtp=True,
        enforce_eager=True,
        tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_batched_tokens,
        max_num_seqs=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    llm.add_request(prompt, SamplingParams(temperature=0.0, max_tokens=1))
    seqs, is_prefill = llm.scheduler.schedule()
    assert is_prefill

    result = llm.model_runner.call("run_mtp_probe", seqs)
    main_text = tokenizer.decode(result["main_token_ids"])
    draft_text = tokenizer.decode(result["draft_token_ids"])

    print("\n=== MTP PROBE ===", flush=True)
    print(f"main_token_ids: {result['main_token_ids']}", flush=True)
    print(f"main_token_text: {main_text!r}", flush=True)
    print(f"draft_token_ids: {result['draft_token_ids']}", flush=True)
    print(f"draft_token_text: {draft_text!r}", flush=True)
    print(f"main_hidden_shape: {result['main_hidden_shape']}", flush=True)
    print(f"mtp_hidden_shape: {result['mtp_hidden_shape']}", flush=True)
    print(f"draft_logits_shape: {result['draft_logits_shape']}", flush=True)


if __name__ == "__main__":
    main()
