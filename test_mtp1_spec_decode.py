import argparse
import os

from test_state_rollback import restore_scheduler, snapshot_scheduler


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen3.6 MTP-1 speculative decode prototype.")
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
    parser.add_argument("--force-reject-attempt", type=int, default=0)
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
    generated_tokens = 0
    target_forwards = 0
    reject_reruns = 0

    while not llm.is_finished():
        seqs, is_prefill = llm.scheduler.schedule()
        draft_result = llm.model_runner.call("run_mtp_draft_step", seqs, is_prefill, args.top_k)
        target_forwards += 1

        main_token_ids = draft_result["main_token_ids"]
        draft_token_ids = draft_result["draft_token_ids"]
        llm.scheduler.postprocess(seqs, main_token_ids, is_prefill)
        generated_tokens += len(main_token_ids)

        if generated_tokens <= args.verbose_steps:
            print(
                f"main token {generated_tokens}: "
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
        assert not verify_is_prefill

        attempts += 1
        snapshot_name = f"spec_{attempts}"
        scheduler_snapshot = snapshot_scheduler(llm, seq)
        llm.model_runner.call("save_decode_state", snapshot_name, verify_seqs)
        verify = llm.model_runner.call(
            "run_step_probe",
            verify_seqs,
            False,
            args.top_k,
            snapshot_name,
            None,
        )
        target_forwards += 1

        verify_token_ids = verify["token_ids"]
        forced_reject = args.force_reject_attempt == attempts
        is_accepted = draft_token_ids == verify_token_ids and not forced_reject

        if is_accepted:
            llm.scheduler.postprocess(verify_seqs, draft_token_ids, False)
            accepted += 1
            committed_token_ids = draft_token_ids
            max_rerun_logit_diff = None
        else:
            restore_scheduler(llm, seq, scheduler_snapshot)
            llm.model_runner.call("restore_decode_state", snapshot_name)
            rerun = llm.model_runner.call(
                "run_step_probe",
                verify_seqs,
                False,
                args.top_k,
                None,
                snapshot_name,
            )
            target_forwards += 1
            reject_reruns += 1
            assert rerun["token_ids"] == verify_token_ids
            llm.scheduler.postprocess(verify_seqs, rerun["token_ids"], False)
            rejected += 1
            committed_token_ids = rerun["token_ids"]
            max_rerun_logit_diff = rerun["max_logit_diff"]

        llm.model_runner.call("drop_decode_state", snapshot_name)
        generated_tokens += len(committed_token_ids)

        if generated_tokens <= args.verbose_steps:
            print(
                f"verify token {generated_tokens}: "
                f"{verify_token_ids} {token_text(tokenizer, verify_token_ids)!r} "
                f"accept={is_accepted} forced_reject={forced_reject}",
                flush=True,
            )
            if max_rerun_logit_diff is not None:
                print(f"  reject rerun max_logit_diff={max_rerun_logit_diff}", flush=True)

    accept_rate = accepted / attempts if attempts else 0.0
    print("\n=== MTP-1 SPEC DECODE ===", flush=True)
    print(f"attempts: {attempts}", flush=True)
    print(f"accepted: {accepted}", flush=True)
    print(f"rejected: {rejected}", flush=True)
    print(f"accept_rate: {accept_rate:.2%}", flush=True)
    print(f"generated_tokens: {len(seq.completion_token_ids)}", flush=True)
    print(f"target_forwards: {target_forwards}", flush=True)
    print(f"reject_reruns: {reject_reruns}", flush=True)
    print(f"completion_token_ids: {seq.completion_token_ids}", flush=True)
    print("\n=== TEXT ===", flush=True)
    print(tokenizer.decode(seq.completion_token_ids).replace("<|im_end|>", ""), flush=True)


if __name__ == "__main__":
    main()
