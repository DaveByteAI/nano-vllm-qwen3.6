import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen3.6 MTP single-step forward smoke test.")
    parser.add_argument("--model", default="~/huggingface/Qwen3.6-27B-FP8")
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--prompt", action="append")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--max-batched-tokens", type=int, default=128)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    return parser.parse_args()


def cleanup_probe_request(llm, seqs):
    slot_ids = [seq.state_slot_id for seq in seqs if seq.state_slot_id != -1]
    if slot_ids:
        llm.model_runner.call("reset_gdn_state_slots", slot_ids)
    for seq in seqs:
        if seq.block_table:
            llm.scheduler.block_manager.deallocate(seq)
        if llm.scheduler.state_slot_manager is not None and seq.state_slot_id != -1:
            llm.scheduler.state_slot_manager.deallocate(seq.state_slot_id)
            seq.state_slot_id = -1
        for queue in (llm.scheduler.running, llm.scheduler.waiting):
            try:
                queue.remove(seq)
            except ValueError:
                pass


def main():
    args = parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.devices)

    from transformers import AutoTokenizer

    from nanovllm import LLM, SamplingParams

    model_path = os.path.expanduser(args.model)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    prompt_texts = args.prompt or [
        "你好，请用三句话介绍你自己。",
        "请用一句话解释什么是张量并行。",
    ]

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
    for repeat in range(args.repeats):
        for prompt_text in prompt_texts:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt_text}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )

            print("\n=== PROMPT ===", flush=True)
            print(prompt_text, flush=True)
            llm.add_request(prompt, SamplingParams(temperature=0.0, max_tokens=1))
            seqs, is_prefill = llm.scheduler.schedule()
            assert is_prefill

            result = llm.model_runner.call("run_mtp_probe", seqs, args.top_k)
            main_text = tokenizer.decode(result["main_token_ids"])
            draft_text = tokenizer.decode(result["draft_token_ids"])
            topk = [
                {
                    "token_id": item["token_id"],
                    "text": tokenizer.decode([item["token_id"]]),
                    "score": round(item["score"], 4),
                }
                for item in result["draft_topk"][0]
            ]

            print("\n=== MTP PROBE ===", flush=True)
            print(f"repeat: {repeat + 1}/{args.repeats}", flush=True)
            print(f"main_token_ids: {result['main_token_ids']}", flush=True)
            print(f"main_token_text: {main_text!r}", flush=True)
            print(f"draft_token_ids: {result['draft_token_ids']}", flush=True)
            print(f"draft_token_text: {draft_text!r}", flush=True)
            print(f"combined_preview: {(main_text + draft_text)!r}", flush=True)
            print(f"draft_topk: {topk}", flush=True)
            print(f"main_hidden_shape: {result['main_hidden_shape']}", flush=True)
            print(f"mtp_hidden_shape: {result['mtp_hidden_shape']}", flush=True)
            print(f"draft_logits_shape: {result['draft_logits_shape']}", flush=True)
            print(f"mtp_loaded_count: {result['mtp_loaded_count']}", flush=True)
            print(f"mtp_skipped_count: {result['mtp_skipped_count']}", flush=True)
            cleanup_probe_request(llm, seqs)


if __name__ == "__main__":
    main()
