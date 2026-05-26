import argparse
import gc
import os
from time import perf_counter

from test_state_rollback import restore_scheduler, snapshot_scheduler


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen3.6 MTP speculative decode prototype.")
    parser.add_argument("--model", default="~/huggingface/Qwen3.6-27B-FP8")
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--prompt", default="你好，请用三句话介绍你自己。")
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--max-batched-tokens", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--draft-len", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--verbose-steps", type=int, default=16)
    parser.add_argument("--force-reject-step", type=int, default=0)
    parser.add_argument("--skip-greedy-compare", action="store_true")
    return parser.parse_args()


def build_prompt(tokenizer, prompt_text):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_text}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def decode_text(tokenizer, token_ids):
    return tokenizer.decode(token_ids).replace("<|im_end|>", "")


def create_llm(args, enable_mtp):
    from nanovllm import LLM

    return LLM(
        os.path.expanduser(args.model),
        enable_vision=False,
        enable_mtp=enable_mtp,
        enforce_eager=True,
        tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_batched_tokens,
        max_num_seqs=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )


def close_llm(llm):
    llm.exit()
    del llm
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass


def run_greedy(args, prompt):
    from nanovllm import SamplingParams

    llm = create_llm(args, enable_mtp=False)
    llm.add_request(prompt, SamplingParams(temperature=0.0, max_tokens=args.max_tokens))
    seq = llm.scheduler.waiting[-1]
    stats = {
        "target_forwards": 0,
        "prefill_forwards": 0,
        "decode_forwards": 0,
        "model_call_seconds": 0.0,
    }

    started = perf_counter()
    while not llm.is_finished():
        seqs, is_prefill = llm.scheduler.schedule()
        call_started = perf_counter()
        token_ids = llm.model_runner.call("run", seqs, is_prefill)
        stats["model_call_seconds"] += perf_counter() - call_started
        stats["target_forwards"] += 1
        stats["prefill_forwards" if is_prefill else "decode_forwards"] += 1
        llm.scheduler.postprocess(seqs, token_ids, is_prefill)
    stats["wall_seconds"] = perf_counter() - started
    token_ids = list(seq.completion_token_ids)
    close_llm(llm)
    return token_ids, stats


def run_speculative(args, tokenizer, prompt):
    from nanovllm import SamplingParams

    llm = create_llm(args, enable_mtp=True)
    llm.add_request(prompt, SamplingParams(temperature=0.0, max_tokens=args.max_tokens))
    seq = llm.scheduler.waiting[-1]

    stats = {
        "draft_rounds": 0,
        "draft_token_attempts": 0,
        "accepted_tokens": 0,
        "rejected_tokens": 0,
        "target_forwards": 0,
        "prefill_forwards": 0,
        "decode_forwards": 0,
        "mtp_forwards": 0,
        "reject_reruns": 0,
        "main_call_seconds": 0.0,
        "verify_call_seconds": 0.0,
        "rerun_call_seconds": 0.0,
    }
    generated_tokens = 0
    verify_step = 0

    started = perf_counter()
    while not llm.is_finished():
        remaining = args.max_tokens - seq.num_completion_tokens
        round_draft_len = max(0, min(args.draft_len, remaining - 1))

        seqs, is_prefill = llm.scheduler.schedule()
        call_started = perf_counter()
        draft_result = llm.model_runner.call(
            "run_mtp_draft_step",
            seqs,
            is_prefill,
            args.top_k,
            round_draft_len,
        )
        stats["main_call_seconds"] += perf_counter() - call_started
        stats["target_forwards"] += 1
        stats["prefill_forwards" if is_prefill else "decode_forwards"] += 1
        stats["mtp_forwards"] += draft_result["mtp_forwards"]
        stats["draft_rounds"] += int(round_draft_len > 0)

        main_token_ids = draft_result["main_token_ids"]
        draft_token_ids = draft_result["draft_token_ids"]
        llm.scheduler.postprocess(seqs, main_token_ids, is_prefill)
        generated_tokens += len(main_token_ids)

        if generated_tokens <= args.verbose_steps:
            print(
                f"main token {generated_tokens}: "
                f"{main_token_ids} {tokenizer.decode(main_token_ids)!r}",
                flush=True,
            )
            if draft_token_ids:
                print(
                    f"  mtp drafts: {draft_token_ids} "
                    f"{tokenizer.decode(draft_token_ids)!r}",
                    flush=True,
                )

        if llm.is_finished():
            break

        for draft_token_id in draft_token_ids:
            if llm.is_finished():
                break

            verify_seqs, verify_is_prefill = llm.scheduler.schedule()
            assert not verify_is_prefill

            verify_step += 1
            stats["draft_token_attempts"] += 1
            snapshot_name = f"spec_{verify_step}"
            scheduler_snapshot = snapshot_scheduler(llm, seq)
            llm.model_runner.call("save_decode_state", snapshot_name, verify_seqs)

            call_started = perf_counter()
            verify = llm.model_runner.call(
                "run_step_probe",
                verify_seqs,
                False,
                args.top_k,
                snapshot_name,
                None,
            )
            stats["verify_call_seconds"] += perf_counter() - call_started
            stats["target_forwards"] += 1
            stats["decode_forwards"] += 1

            verify_token_ids = verify["token_ids"]
            forced_reject = args.force_reject_step == verify_step
            is_accepted = verify_token_ids == [draft_token_id] and not forced_reject

            if is_accepted:
                llm.scheduler.postprocess(verify_seqs, [draft_token_id], False)
                stats["accepted_tokens"] += 1
                committed_token_ids = [draft_token_id]
                max_rerun_logit_diff = None
            else:
                restore_scheduler(llm, seq, scheduler_snapshot)
                llm.model_runner.call("restore_decode_state", snapshot_name)
                call_started = perf_counter()
                rerun = llm.model_runner.call(
                    "run_step_probe",
                    verify_seqs,
                    False,
                    args.top_k,
                    None,
                    snapshot_name,
                )
                stats["rerun_call_seconds"] += perf_counter() - call_started
                stats["target_forwards"] += 1
                stats["decode_forwards"] += 1
                stats["reject_reruns"] += 1
                assert rerun["token_ids"] == verify_token_ids
                llm.scheduler.postprocess(verify_seqs, rerun["token_ids"], False)
                stats["rejected_tokens"] += 1
                committed_token_ids = rerun["token_ids"]
                max_rerun_logit_diff = rerun["max_logit_diff"]

            llm.model_runner.call("drop_decode_state", snapshot_name)
            generated_tokens += len(committed_token_ids)

            if generated_tokens <= args.verbose_steps:
                print(
                    f"verify token {generated_tokens}: "
                    f"draft={[draft_token_id]} {tokenizer.decode([draft_token_id])!r} "
                    f"target={verify_token_ids} {tokenizer.decode(verify_token_ids)!r} "
                    f"accept={is_accepted} forced_reject={forced_reject}",
                    flush=True,
                )
                if max_rerun_logit_diff is not None:
                    print(f"  reject rerun max_logit_diff={max_rerun_logit_diff}", flush=True)

            if not is_accepted:
                break

    stats["wall_seconds"] = perf_counter() - started
    token_ids = list(seq.completion_token_ids)
    close_llm(llm)
    return token_ids, stats


def print_stats(greedy_ids, greedy_stats, spec_ids, spec_stats):
    generated = max(len(spec_ids), 1)
    attempts = spec_stats["draft_token_attempts"]
    accepted = spec_stats["accepted_tokens"]
    accept_rate = accepted / attempts if attempts else 0.0

    print("\n=== ALIGNMENT ===", flush=True)
    if greedy_ids is not None:
        print(f"greedy_match: {greedy_ids == spec_ids}", flush=True)
        print(f"greedy_token_count: {len(greedy_ids)}", flush=True)
    else:
        print("greedy_match: skipped", flush=True)
    print(f"spec_token_count: {len(spec_ids)}", flush=True)

    print("\n=== SPEC STATS ===", flush=True)
    print(f"draft_len: {spec_stats['draft_len']}", flush=True)
    print(f"draft_rounds: {spec_stats['draft_rounds']}", flush=True)
    print(f"draft_token_attempts: {attempts}", flush=True)
    print(f"accepted_tokens: {accepted}", flush=True)
    print(f"rejected_tokens: {spec_stats['rejected_tokens']}", flush=True)
    print(f"accept_rate: {accept_rate:.2%}", flush=True)
    print(f"target_forwards: {spec_stats['target_forwards']}", flush=True)
    print(f"mtp_forwards: {spec_stats['mtp_forwards']}", flush=True)
    print(f"reject_reruns: {spec_stats['reject_reruns']}", flush=True)
    print(f"target_forwards_per_token: {spec_stats['target_forwards'] / generated:.3f}", flush=True)
    print(f"mtp_forwards_per_token: {spec_stats['mtp_forwards'] / generated:.3f}", flush=True)
    print(f"reject_reruns_per_token: {spec_stats['reject_reruns'] / generated:.3f}", flush=True)
    if greedy_stats is not None:
        delta = spec_stats["target_forwards"] - greedy_stats["target_forwards"]
        print(f"greedy_target_forwards: {greedy_stats['target_forwards']}", flush=True)
        print(f"extra_target_forwards_vs_greedy: {delta}", flush=True)
        print(f"extra_target_forwards_per_token: {delta / generated:.3f}", flush=True)
    print(f"spec_model_call_seconds: {spec_stats['model_call_seconds']:.4f}", flush=True)
    print(f"spec_model_call_seconds_per_token: {spec_stats['model_call_seconds'] / generated:.4f}", flush=True)
    if greedy_stats is not None:
        print(f"greedy_model_call_seconds: {greedy_stats['model_call_seconds']:.4f}", flush=True)


def main():
    args = parse_args()
    assert args.draft_len >= 0
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.devices)

    from transformers import AutoTokenizer

    model_path = os.path.expanduser(args.model)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    prompt = build_prompt(tokenizer, args.prompt)

    print("=== PROMPT ===", flush=True)
    print(args.prompt, flush=True)

    greedy_ids = None
    greedy_stats = None
    if not args.skip_greedy_compare:
        print("\n=== RUN GREEDY BASELINE ===", flush=True)
        greedy_ids, greedy_stats = run_greedy(args, prompt)
        print(f"greedy_text: {decode_text(tokenizer, greedy_ids)!r}", flush=True)

    print("\n=== RUN MTP SPEC DECODE ===", flush=True)
    spec_ids, spec_stats = run_speculative(args, tokenizer, prompt)
    spec_stats["draft_len"] = args.draft_len
    spec_stats["model_call_seconds"] = (
        spec_stats["main_call_seconds"]
        + spec_stats["verify_call_seconds"]
        + spec_stats["rerun_call_seconds"]
    )

    print_stats(greedy_ids, greedy_stats, spec_ids, spec_stats)
    print("\n=== SPEC TEXT ===", flush=True)
    print(decode_text(tokenizer, spec_ids), flush=True)
    print(f"spec_token_ids: {spec_ids}", flush=True)
    if greedy_ids is not None and greedy_ids != spec_ids:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
