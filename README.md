# Nano-vLLM Qwen3.5/Qwen3.6

A compact, readable inference engine based on `nano-vllm`, extended for Qwen3.5 hybrid
models and Qwen3.6 FP8 text-only inference experiments.

This repository is intended for learning how LLM inference engines work: tensor
parallelism, KV cache allocation, CUDA Graph decode, hybrid linear-attention state,
and quantized checkpoint loading are all implemented in a small codebase.

## What Works

- Qwen3 dense text models through the original Nano-vLLM path.
- Qwen3.5-9B BF16 text inference.
- Qwen3.5-9B multimodal smoke test through the local vision encoder path.
- Qwen3.6-27B-FP8 text-only inference on 4 x RTX 4090.
- Tensor parallelism for attention, MLP, vocabulary embedding/head, and GatedDeltaNet.
- CUDA Graph decode when `enforce_eager=False`.
- Rank-local FP8 checkpoint loading: each TP rank slices its shard and dequantizes only
  the shard it owns.
- Experimental Qwen3.6 MTP weight loading, single-step forward probe, and MTP-1
  draft/verify prototype.

## Current Limitations

- Qwen3.6-27B-FP8 is loaded as BF16 weights after FP8 block dequantization. Native FP8
  matmul kernels are not implemented here yet.
- Qwen3.6-27B-FP8 currently targets text-only inference. Use `enable_vision=False`.
- Qwen3.6-27B-FP8 was verified with `tensor_parallel_size=4` on 4 x RTX 4090. TP=2
  does not fit in 24GB cards with the current BF16-resident implementation.
- Loading Qwen3.6-27B-FP8 is slow because the original FP8 checkpoint is converted at
  startup. A pre-converted TP-sharded checkpoint would start faster.
- MTP is currently a prototype for weight loading, one-step draft-token probing, and
  MTP-1 accept-rate measurement. It does not provide decode speedup yet.
- This is not a production serving stack. It is a research/learning implementation.

## Repository Layout

```text
nanovllm/
  engine/          scheduler, model runner, block/state managers
  layers/          attention, linear layers, GatedDeltaNet, sampler
  models/          Qwen3 and Qwen3.5 model definitions
  utils/           checkpoint loader, FP8 dequant helpers, context utilities
examples/          original examples
run_text_qwen35_v2.py
run_text_qwen36_fp8.py
test_mtp_forward.py
test_mtp1_verify.py
test_state_rollback.py
bench_qwen35_fixed.py
```

## Installation

Use Python 3.10-3.12 with CUDA-capable PyTorch. Real inference requires GPUs plus
`torch`, `triton`, and `flash-attn`.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
```

If your environment already has PyTorch, Triton, and FlashAttention installed, the
editable install is enough.

## Model Download

Keep model weights outside the repository. The examples assume `~/huggingface`.

```bash
hf download Qwen/Qwen3.5-9B \
  --local-dir ~/huggingface/Qwen3.5-9B \
  --max-workers 8

hf download Qwen/Qwen3.6-27B-FP8 \
  --local-dir ~/huggingface/Qwen3.6-27B-FP8 \
  --max-workers 8
```

Do not commit model weights, generated caches, local images, or benchmark logs.

## Quick Start

Qwen3.5-9B text smoke test:

```bash
python run_text_qwen35_v2.py \
  --model ~/huggingface/Qwen3.5-9B \
  --tp 1
```

Qwen3.5-9B with 4-way tensor parallelism:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_text_qwen35_v2.py \
  --model ~/huggingface/Qwen3.5-9B \
  --tp 4
```

Qwen3.6-27B-FP8 text-only on 4 x RTX 4090:

```bash
python run_text_qwen36_fp8.py \
  --model ~/huggingface/Qwen3.6-27B-FP8 \
  --devices 0,1,2,3 \
  --tp 4
```

Disable CUDA Graph for debugging:

```bash
python run_text_qwen36_fp8.py --eager
```

Customize the prompt:

```bash
python run_text_qwen36_fp8.py \
  --prompt "你好，请用三句话介绍你自己。然后讲一个简短笑话。"
```

Qwen3.6 MTP single-step forward probe:

```bash
python test_mtp_forward.py \
  --model ~/huggingface/Qwen3.6-27B-FP8 \
  --devices 0,1,2,3 \
  --tp 4 \
  --top-k 5
```

Qwen3.6 MTP-1 draft/verify prototype:

```bash
python test_mtp1_verify.py \
  --model ~/huggingface/Qwen3.6-27B-FP8 \
  --devices 0,1,2,3 \
  --tp 4 \
  --max-tokens 64
```

Decode-state rollback smoke test:

```bash
python test_state_rollback.py \
  --model ~/huggingface/Qwen3.6-27B-FP8 \
  --devices 0,1,2,3 \
  --tp 4
```

## API Example

```python
from nanovllm import LLM, SamplingParams

llm = LLM(
    "/path/to/model",
    tensor_parallel_size=1,
    enforce_eager=True,
)
params = SamplingParams(temperature=0.7, max_tokens=128)
outputs = llm.generate(["Hello, Nano-vLLM."], params)
print(outputs[0]["text"])
```

## Benchmarks

Qwen3.5 single-prompt timing helper:

```bash
python bench_qwen35_fixed.py \
  --model ~/huggingface/Qwen3.5-9B \
  --tp 4 \
  --max-tokens 256 \
  --repeats 3
```

Recent local smoke-test results on RTX 4090 hardware:

```text
Qwen3.6-27B-FP8, TP=4, CUDA Graph decode: Decode ~= 41 tok/s
Qwen3.5-9B, TP=4, CUDA Graph decode:       Decode ~= 98 tok/s
```

These are simple single-request smoke tests, not full serving benchmarks.

## Development Checks

Syntax check without model weights:

```bash
python -m compileall nanovllm examples run_text_qwen35_v2.py run_text_qwen36_fp8.py test_mtp_forward.py test_mtp1_verify.py test_state_rollback.py
```

Useful runtime checks:

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
```

## Open Source Notes

- The project is licensed under MIT. Keep the existing `LICENSE` file.
- `pyproject.toml` points at this fork and keeps a link to the original upstream
  project.
- Keep large artifacts outside git. `.gitignore` excludes common model/checkpoint files.
- Hardware assumptions should be included in issues/PRs when reporting inference results.

## Acknowledgements

This work builds on the original Nano-vLLM project by Xingkai Yu and keeps the MIT
license. The Qwen model weights are distributed separately by Qwen under their own
model terms.
