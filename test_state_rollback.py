import argparse
import os
from collections import deque


def parse_args():
    parser = argparse.ArgumentParser(description="Decode-state rollback smoke test.")
    parser.add_argument("--model", default="~/huggingface/Qwen3.6-27B-FP8")
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--prompt", default="你好，请用三句话介绍你自己。")
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--max-batched-tokens", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--tolerance", type=float, default=0.0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--disable-mtp", action="store_true")
    return parser.parse_args()


def snapshot_scheduler(llm, seq):
    block_manager = llm.scheduler.block_manager
    state_slot_manager = llm.scheduler.state_slot_manager
    return {
        "seq": {
            "status": seq.status,
            "token_ids": list(seq.token_ids),
            "last_token": seq.last_token,
            "num_tokens": seq.num_tokens,
            "num_prompt_tokens": seq.num_prompt_tokens,
            "num_cached_tokens": seq.num_cached_tokens,
            "num_scheduled_tokens": seq.num_scheduled_tokens,
            "block_table": list(seq.block_table),
            "state_slot_id": seq.state_slot_id,
            "temperature": seq.temperature,
            "max_tokens": seq.max_tokens,
            "ignore_eos": seq.ignore_eos,
        },
        "waiting": deque(llm.scheduler.waiting),
        "running": deque(llm.scheduler.running),
        "free_block_ids": deque(block_manager.free_block_ids),
        "used_block_ids": set(block_manager.used_block_ids),
        "hash_to_block_id": dict(block_manager.hash_to_block_id),
        "blocks": [
            (block.ref_count, block.hash, list(block.token_ids))
            for block in block_manager.blocks
        ],
        "free_state_slots": (
            deque(state_slot_manager.free_slots)
            if state_slot_manager is not None
            else None
        ),
    }


def restore_scheduler(llm, seq, snapshot):
    seq_state = snapshot["seq"]
    seq.status = seq_state["status"]
    seq.token_ids = list(seq_state["token_ids"])
    seq.last_token = seq_state["last_token"]
    seq.num_tokens = seq_state["num_tokens"]
    seq.num_prompt_tokens = seq_state["num_prompt_tokens"]
    seq.num_cached_tokens = seq_state["num_cached_tokens"]
    seq.num_scheduled_tokens = seq_state["num_scheduled_tokens"]
    seq.block_table = list(seq_state["block_table"])
    seq.state_slot_id = seq_state["state_slot_id"]
    seq.temperature = seq_state["temperature"]
    seq.max_tokens = seq_state["max_tokens"]
    seq.ignore_eos = seq_state["ignore_eos"]

    llm.scheduler.waiting = deque(snapshot["waiting"])
    llm.scheduler.running = deque(snapshot["running"])

    block_manager = llm.scheduler.block_manager
    block_manager.free_block_ids = deque(snapshot["free_block_ids"])
    block_manager.used_block_ids = set(snapshot["used_block_ids"])
    block_manager.hash_to_block_id = dict(snapshot["hash_to_block_id"])
    for block, block_state in zip(block_manager.blocks, snapshot["blocks"]):
        block.ref_count, block.hash, token_ids = block_state
        block.token_ids = list(token_ids)

    if llm.scheduler.state_slot_manager is not None:
        llm.scheduler.state_slot_manager.free_slots = deque(snapshot["free_state_slots"])


def decode_tokens(tokenizer, token_ids):
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
        enable_mtp=not args.disable_mtp,
        enforce_eager=True,
        tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_batched_tokens,
        max_num_seqs=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    llm.add_request(prompt, SamplingParams(temperature=0.0, max_tokens=args.max_tokens))
    seq = llm.scheduler.waiting[-1]

    prefill_seqs, is_prefill = llm.scheduler.schedule()
    assert is_prefill
    prefill_token_ids = llm.model_runner.call("run", prefill_seqs, is_prefill)
    llm.scheduler.postprocess(prefill_seqs, prefill_token_ids, is_prefill)
    print(
        f"prefill token: {prefill_token_ids} {decode_tokens(tokenizer, prefill_token_ids)!r}",
        flush=True,
    )

    decode_seqs, is_prefill = llm.scheduler.schedule()
    assert not is_prefill
    scheduler_snapshot = snapshot_scheduler(llm, seq)
    state_snapshot = llm.model_runner.call("save_decode_state", "rollback", decode_seqs)
    first = llm.model_runner.call(
        "run_step_probe",
        decode_seqs,
        False,
        args.top_k,
        "rollback",
        None,
    )
    llm.scheduler.postprocess(decode_seqs, first["token_ids"], False)

    restore_scheduler(llm, seq, scheduler_snapshot)
    llm.model_runner.call("restore_decode_state", "rollback")
    second = llm.model_runner.call(
        "run_step_probe",
        decode_seqs,
        False,
        args.top_k,
        None,
        "rollback",
    )
    llm.scheduler.postprocess(decode_seqs, second["token_ids"], False)
    llm.model_runner.call("drop_decode_state", "rollback")

    token_match = first["token_ids"] == second["token_ids"]
    max_logit_diff = second["max_logit_diff"]
    rollback_ok = token_match and max_logit_diff <= args.tolerance

    print("\n=== ROLLBACK TEST ===", flush=True)
    print(f"kv_slots: {state_snapshot['kv_slots']}", flush=True)
    print(f"state_slot_ids: {state_snapshot['state_slot_ids']}", flush=True)
    print(f"gdn_layers: {state_snapshot['gdn_layers']}", flush=True)
    print(
        f"first token: {first['token_ids']} {decode_tokens(tokenizer, first['token_ids'])!r}",
        flush=True,
    )
    print(
        f"second token: {second['token_ids']} {decode_tokens(tokenizer, second['token_ids'])!r}",
        flush=True,
    )
    print(f"first topk: {first['topk'][0]}", flush=True)
    print(f"second topk: {second['topk'][0]}", flush=True)
    print(f"max_logit_diff: {max_logit_diff}", flush=True)
    print(f"rollback_ok: {rollback_ok}", flush=True)
    if not rollback_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
