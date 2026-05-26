import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen3.6 MTP-1 draft/verify prototype.")
    parser.add_argument("--model", default="~/huggingface/Qwen3.6-27B-FP8")
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--prompt", default="你好，请用三句话介绍你自己。")
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--max-batched-tokens", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--verbose-steps", type=int, default=16)
    return parser.parse_args()


def token_text(tokenizer, token_ids):
    return tokenizer.decode(token_ids)


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
    print(args.prompt, flush=True)

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
    llm.add_request(prompt, SamplingParams(temperature=0.0, max_tokens=args.max_tokens))
    seq = llm.scheduler.waiting[-1]

    attempts = 0
    accepted = 0
    rejected = 0
    generated_steps = 0

    while not llm.is_finished():
        seqs, is_prefill = llm.scheduler.schedule()
        draft_result = llm.model_runner.call("run_mtp_draft_step", seqs, is_prefill, args.top_k)
        main_token_ids = draft_result["main_token_ids"]
        draft_token_ids = draft_result["draft_token_ids"]
        llm.scheduler.postprocess(seqs, main_token_ids, is_prefill)
        generated_steps += len(main_token_ids)

        if generated_steps <= args.verbose_steps:
            print(
                f"main step {generated_steps}: "
                f"{main_token_ids} {token_text(tokenizer, main_token_ids)!r}",
                flush=True,
            )
            print(
                f"  mtp draft next: "
                f"{draft_token_ids} {token_text(tokenizer, draft_token_ids)!r}",
                flush=True,
            )

        if llm.is_finished():
            break

        verify_seqs, verify_is_prefill = llm.scheduler.schedule()
        verify_token_ids = llm.model_runner.call("run", verify_seqs, verify_is_prefill)
        llm.scheduler.postprocess(verify_seqs, verify_token_ids, verify_is_prefill)
        generated_steps += len(verify_token_ids)

        if not verify_is_prefill:
            attempts += 1
            is_accepted = draft_token_ids == verify_token_ids
            accepted += int(is_accepted)
            rejected += int(not is_accepted)
            if generated_steps <= args.verbose_steps:
                print(
                    f"verify step {generated_steps}: "
                    f"{verify_token_ids} {token_text(tokenizer, verify_token_ids)!r} "
                    f"accept={is_accepted}",
                    flush=True,
                )

    accept_rate = accepted / attempts if attempts else 0.0
    print("\n=== MTP-1 VERIFY ===", flush=True)
    print(f"attempts: {attempts}", flush=True)
    print(f"accepted: {accepted}", flush=True)
    print(f"rejected: {rejected}", flush=True)
    print(f"accept_rate: {accept_rate:.2%}", flush=True)
    print(f"completion_token_ids: {seq.completion_token_ids}", flush=True)
    print("\n=== TEXT ===", flush=True)
    print(tokenizer.decode(seq.completion_token_ids).replace("<|im_end|>", ""), flush=True)


if __name__ == "__main__":
    main()
