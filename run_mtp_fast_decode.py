import argparse
import gc
import os
from time import perf_counter

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

from test_state_rollback import restore_scheduler, snapshot_scheduler
from test_mtp_spec_decode import (
    build_prompt,
    decode_text,
    ensure_block_capacity,
    finalize_manual_commit,
    trusted_verify_mode,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen3.6 MTP speculative decode fast path.")
    parser.add_argument("--model", default="~/huggingface/Qwen3.6-27B-FP8")
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--prompt", default="你好，请用三句话介绍你自己。")
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--max-batched-tokens", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--draft-len", type=int, default=2)
    parser.add_argument("--verify-mode", choices=["eager", "graph", "chunk"], default="graph")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--compare-greedy", action="store_true")
    parser.add_argument("--hide-text", action="store_true")
    return parser.parse_args()


def create_llm(args, enable_mtp, enforce_eager):
    from nanovllm import LLM

    return LLM(
        os.path.expanduser(args.model),
        enable_vision=False,
        enable_mtp=enable_mtp,
        enforce_eager=enforce_eager,
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


def commit_tokens(llm, seq, token_ids):
    ensure_block_capacity(llm, seq, len(seq) + len(token_ids))
    committed = []
    for token_id in token_ids:
        if llm.is_finished():
            break
        llm.scheduler.postprocess([seq], [token_id], False)
        committed.append(token_id)
    finalize_manual_commit(llm, seq)
    return committed


def run_greedy_decode(args, prompt):
    from nanovllm import SamplingParams

    llm = create_llm(args, enable_mtp=False, enforce_eager=False)
    llm.add_request(prompt, SamplingParams(temperature=0.0, max_tokens=args.max_tokens))
    seq = llm.scheduler.waiting[-1]
    stats = {
        "target_forwards": 0,
        "model_call_seconds": 0.0,
    }

    started = perf_counter()
    while not llm.is_finished():
        seqs, is_prefill = llm.scheduler.schedule()
        call_started = perf_counter()
        token_ids = llm.model_runner.call("run", seqs, is_prefill)
        stats["model_call_seconds"] += perf_counter() - call_started
        stats["target_forwards"] += 1
        llm.scheduler.postprocess(seqs, token_ids, is_prefill)
    stats["wall_seconds"] = perf_counter() - started
    token_ids = list(seq.completion_token_ids)
    close_llm(llm)
    return token_ids, stats


def run_mtp_fast_decode(args, prompt):
    from nanovllm import SamplingParams

    llm = create_llm(args, enable_mtp=True, enforce_eager=args.verify_mode == "eager")
    llm.add_request(prompt, SamplingParams(temperature=0.0, max_tokens=args.max_tokens))
    seq = llm.scheduler.waiting[-1]
    stats = {
        "draft_rounds": 0,
        "draft_token_attempts": 0,
        "accepted_tokens": 0,
        "rejected_tokens": 0,
        "target_forwards": 0,
        "mtp_forwards": 0,
        "verify_batches": 0,
        "verify_batch_tokens": 0,
        "verify_graph_replays": 0,
        "verify_eager_calls": 0,
        "verify_chunk_calls": 0,
        "chunk_graph_replays": 0,
        "chunk_checked_batches": 0,
        "chunk_token_mismatches": 0,
        "accept_length_total": 0,
        "compared_tokens": 0,
        "reject_reruns": 0,
        "reject_rerun_input_tokens": 0,
        "main_call_seconds": 0.0,
        "verify_call_seconds": 0.0,
        "rerun_call_seconds": 0.0,
    }

    started = perf_counter()
    while not llm.is_finished():
        remaining = args.max_tokens - seq.num_completion_tokens
        round_draft_len = max(0, min(args.draft_len, remaining - 1))

        seqs, is_prefill = llm.scheduler.schedule()
        call_started = perf_counter()
        draft_result = llm.model_runner.call(
            "run_mtp_draft_fast_step",
            seqs,
            is_prefill,
            round_draft_len,
        )
        stats["main_call_seconds"] += perf_counter() - call_started
        stats["target_forwards"] += 1
        stats["mtp_forwards"] += draft_result["mtp_forwards"]
        stats["draft_rounds"] += int(round_draft_len > 0)

        main_token_ids = draft_result["main_token_ids"]
        draft_token_ids = draft_result["draft_token_ids"]
        llm.scheduler.postprocess(seqs, main_token_ids, is_prefill)
        if llm.is_finished():
            break

        remaining_after_main = args.max_tokens - seq.num_completion_tokens
        draft_token_ids = draft_token_ids[:remaining_after_main]
        if not draft_token_ids:
            continue

        verify_len = len(draft_token_ids)
        start_pos = len(seq) - 1
        verify_input_ids = [seq.last_token] + draft_token_ids[:-1]
        snapshot_name = f"fast_spec_{stats['verify_batches'] + 1}"
        scheduler_snapshot = snapshot_scheduler(llm, seq)
        ensure_block_capacity(llm, seq, start_pos + verify_len)
        llm.model_runner.call("save_decode_state_range", snapshot_name, [seq], start_pos, verify_len)

        call_started = perf_counter()
        verify = llm.model_runner.call(
            "run_verify_batch_fast",
            [seq],
            verify_input_ids,
            start_pos,
            args.verify_mode,
        )
        verify_seconds = perf_counter() - call_started
        stats["verify_call_seconds"] += verify_seconds
        stats["verify_batches"] += 1
        stats["verify_batch_tokens"] += verify_len
        stats["draft_token_attempts"] += verify_len
        stats["target_forwards"] += verify_len
        if verify["verify_mode_used"] == "graph":
            stats["verify_graph_replays"] += 1
        elif verify["verify_mode_used"] == "chunk":
            stats["verify_chunk_calls"] += 1
            if verify.get("chunk_graph_replay"):
                stats["chunk_graph_replays"] += 1
        else:
            stats["verify_eager_calls"] += 1

        if args.verify_mode == "chunk":
            chunk_verify = verify
            restore_scheduler(llm, seq, scheduler_snapshot)
            llm.model_runner.call("restore_decode_state", snapshot_name)
            call_started = perf_counter()
            verify = llm.model_runner.call(
                "run_verify_batch_fast",
                [seq],
                verify_input_ids,
                start_pos,
                trusted_verify_mode(args),
            )
            stats["verify_call_seconds"] += perf_counter() - call_started
            stats["target_forwards"] += verify_len
            if verify["verify_mode_used"] == "graph":
                stats["verify_graph_replays"] += 1
            else:
                stats["verify_eager_calls"] += 1
            stats["chunk_checked_batches"] += 1
            stats["chunk_token_mismatches"] += sum(
                int(a != b) for a, b in zip(chunk_verify["token_ids"], verify["token_ids"])
            )

        target_token_ids = verify["token_ids"]
        accept_len = 0
        reject_index = None
        for i, (draft_token_id, target_token_id) in enumerate(zip(draft_token_ids, target_token_ids)):
            if draft_token_id == target_token_id:
                accept_len += 1
                continue
            reject_index = i
            break

        stats["accept_length_total"] += accept_len
        stats["compared_tokens"] += accept_len + int(reject_index is not None)

        if reject_index is None:
            stats["accepted_tokens"] += accept_len
            commit_tokens(llm, seq, draft_token_ids)
        else:
            stats["accepted_tokens"] += accept_len
            stats["rejected_tokens"] += 1
            restore_scheduler(llm, seq, scheduler_snapshot)
            llm.model_runner.call("restore_decode_state", snapshot_name)
            rerun_input_ids = [seq.last_token] + draft_token_ids[:accept_len]
            ensure_block_capacity(llm, seq, start_pos + len(rerun_input_ids))

            call_started = perf_counter()
            rerun = llm.model_runner.call(
                "run_verify_batch_fast",
                [seq],
                rerun_input_ids,
                start_pos,
                trusted_verify_mode(args),
            )
            stats["rerun_call_seconds"] += perf_counter() - call_started
            stats["reject_reruns"] += 1
            stats["reject_rerun_input_tokens"] += len(rerun_input_ids)
            stats["target_forwards"] += len(rerun_input_ids)
            if rerun["verify_mode_used"] == "graph":
                stats["verify_graph_replays"] += 1
            else:
                stats["verify_eager_calls"] += 1
            commit_tokens(llm, seq, draft_token_ids[:accept_len] + [rerun["token_ids"][-1]])

        llm.model_runner.call("drop_decode_state", snapshot_name)

    stats["wall_seconds"] = perf_counter() - started
    stats["model_call_seconds"] = (
        stats["main_call_seconds"]
        + stats["verify_call_seconds"]
        + stats["rerun_call_seconds"]
    )
    token_ids = list(seq.completion_token_ids)
    close_llm(llm)
    return token_ids, stats


def summarize_stats(token_count, stats):
    token_count = max(token_count, 1)
    compared = max(stats.get("compared_tokens", token_count), 1)
    return {
        "tokens": token_count,
        "decode_tok_s": token_count / stats["model_call_seconds"],
        "model_call_seconds": stats["model_call_seconds"],
        "accept_rate": stats.get("accepted_tokens", 0) / compared,
        "target_forwards_per_token": stats["target_forwards"] / token_count,
        "mtp_forwards_per_token": stats.get("mtp_forwards", 0) / token_count,
        "verify_graph_replays": stats.get("verify_graph_replays", 0),
        "verify_eager_calls": stats.get("verify_eager_calls", 0),
        "verify_chunk_calls": stats.get("verify_chunk_calls", 0),
        "chunk_graph_replays": stats.get("chunk_graph_replays", 0),
        "chunk_checked_batches": stats.get("chunk_checked_batches", 0),
        "chunk_token_mismatches": stats.get("chunk_token_mismatches", 0),
        "reject_reruns": stats.get("reject_reruns", 0),
    }


def print_summary(name, summary):
    print(f"\n=== {name} ===", flush=True)
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}", flush=True)
        else:
            print(f"{key}: {value}", flush=True)


def main():
    args = parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.devices)

    from transformers import AutoTokenizer

    model_path = os.path.expanduser(args.model)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    prompt = build_prompt(tokenizer, args.prompt)

    greedy_ids = None
    if args.compare_greedy:
        greedy_ids, greedy_stats = run_greedy_decode(args, prompt)
        print_summary("GREEDY GRAPH BASELINE", summarize_stats(len(greedy_ids), greedy_stats))

    spec_ids, spec_stats = run_mtp_fast_decode(args, prompt)
    print_summary("MTP FAST DECODE", summarize_stats(len(spec_ids), spec_stats))
    if greedy_ids is not None:
        print(f"greedy_match: {greedy_ids == spec_ids}", flush=True)

    if not args.hide_text:
        print("\n=== TEXT ===", flush=True)
        print(decode_text(tokenizer, spec_ids), flush=True)


if __name__ == "__main__":
    main()
