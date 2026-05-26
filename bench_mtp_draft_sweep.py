import argparse
import os

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

from run_mtp_fast_decode import (
    build_prompt,
    run_greedy_decode,
    run_mtp_fast_decode,
    summarize_stats,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Sweep Qwen3.6 MTP draft lengths.")
    parser.add_argument("--model", default="~/huggingface/Qwen3.6-27B-FP8")
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--prompt", default="你好，请用三句话介绍你自己。")
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--max-batched-tokens", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--draft-lens", default="1,2,3,4")
    parser.add_argument("--verify-mode", choices=["eager", "graph"], default="graph")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--skip-greedy-compare", action="store_true")
    return parser.parse_args()


def parse_draft_lens(value):
    draft_lens = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        draft_lens.append(int(part))
    assert draft_lens, "at least one draft length is required"
    return draft_lens


def main():
    args = parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.devices)

    from transformers import AutoTokenizer

    model_path = os.path.expanduser(args.model)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    prompt = build_prompt(tokenizer, args.prompt)

    greedy_ids = None
    if not args.skip_greedy_compare:
        greedy_ids, greedy_stats = run_greedy_decode(args, prompt)
        greedy_summary = summarize_stats(len(greedy_ids), greedy_stats)
        print(
            "baseline\t"
            f"tokens={greedy_summary['tokens']}\t"
            f"tok_s={greedy_summary['decode_tok_s']:.4f}\t"
            f"seconds={greedy_summary['model_call_seconds']:.4f}",
            flush=True,
        )

    print(
        "draft_len\tmatch\ttokens\ttok_s\tseconds\taccept_rate\t"
        "target_fw_per_tok\tmtp_fw_per_tok\tverify_graph_replays\tverify_eager_calls\treject_reruns",
        flush=True,
    )
    for draft_len in parse_draft_lens(args.draft_lens):
        args.draft_len = draft_len
        spec_ids, spec_stats = run_mtp_fast_decode(args, prompt)
        summary = summarize_stats(len(spec_ids), spec_stats)
        match = "skipped" if greedy_ids is None else str(greedy_ids == spec_ids)
        print(
            f"{draft_len}\t{match}\t{summary['tokens']}\t"
            f"{summary['decode_tok_s']:.4f}\t"
            f"{summary['model_call_seconds']:.4f}\t"
            f"{summary['accept_rate']:.4f}\t"
            f"{summary['target_forwards_per_token']:.4f}\t"
            f"{summary['mtp_forwards_per_token']:.4f}\t"
            f"{summary['verify_graph_replays']}\t"
            f"{summary['verify_eager_calls']}\t"
            f"{summary['reject_reruns']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
