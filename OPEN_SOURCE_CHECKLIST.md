# Open Source Release Checklist

Use this checklist before pushing the repository to GitHub.

## Required Before Push

- Confirm the GitHub remote is correct.
- Confirm whether to keep the package name as `nano-vllm` or rename it for this fork.
- Keep `LICENSE` in the repository and preserve upstream attribution.
- Do not commit model weights, generated caches, local images, or benchmark output.
- Run the syntax check:

```bash
python -m compileall nanovllm examples run_text_qwen35_v2.py run_text_qwen36_fp8.py
```

- Check tracked files for accidental artifacts:

```bash
git ls-files | rg '(safetensors|\.bin$|\.pt$|\.pth$|\.gguf$|image_demo|__pycache__|egg-info|\.venv)'
```

## Recommended Smoke Tests

Qwen3.5-9B:

```bash
python run_text_qwen35_v2.py \
  --model ~/huggingface/Qwen3.5-9B \
  --tp 1 \
  --max-tokens 64
```

Qwen3.6-27B-FP8 text-only on 4 x RTX 4090:

```bash
python run_text_qwen36_fp8.py \
  --model ~/huggingface/Qwen3.6-27B-FP8 \
  --devices 0,1,2,3 \
  --tp 4 \
  --max-tokens 64
```

## Publish Notes

- Mention that Qwen3.6 FP8 checkpoint weights are dequantized to BF16 at startup.
- Mention that TP=4 is the verified configuration for 24GB RTX 4090 cards.
- Mention that Qwen model weights are downloaded separately and are not included.
- Include hardware, CUDA, PyTorch, and GPU count when reporting benchmark numbers.
