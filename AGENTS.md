# Repository Guidelines

## Project Structure & Module Organization

`nanovllm/` contains the package source. Core runtime code lives in `nanovllm/engine/`, model definitions in `nanovllm/models/`, CUDA/Triton-facing building blocks in `nanovllm/layers/`, and helpers in `nanovllm/utils/`. Public entry points are exposed through `nanovllm/__init__.py`, `nanovllm/llm.py`, and `nanovllm/sampling_params.py`.

`examples/` holds runnable usage scripts for Qwen models. `bench.py` is a throughput benchmark, and `assets/` stores documentation assets such as the logo. There is currently no dedicated `tests/` directory.

## Build, Test, and Development Commands

- `python -m pip install -e .` installs the package in editable mode from `pyproject.toml`.
- `python examples/qwen3.py` runs the Qwen3 text-generation example; update the model path in the script first.
- `python examples/qwen3_5.py` runs the Qwen3.5 text and image smoke test; it expects local model weights and `~/image_demo.jpg`.
- `python bench.py` runs the local throughput benchmark after editing the model path.
- `python -m compileall nanovllm examples` performs a quick syntax check without requiring model weights.

GPU dependencies such as `torch`, `triton`, and `flash-attn` are required for real inference runs.

## Coding Style & Naming Conventions

Use Python 3.10-3.12 syntax and follow PEP 8 conventions: 4-space indentation, `snake_case` for functions and variables, `PascalCase` for classes, and uppercase constants for fixed configuration values. Keep modules small and direct; this project favors readable implementations over deep abstraction layers. Prefer existing helper APIs in `nanovllm/utils/`, `nanovllm/engine/`, and `nanovllm/layers/` before adding new utilities.

## Testing Guidelines

No formal test framework is configured yet. For changes that do not need model weights, run `python -m compileall nanovllm examples`. For inference behavior, run the smallest relevant example script with local weights. If adding tests, place them under `tests/`, name files `test_*.py`, and prefer focused `pytest` tests around scheduling, sampling, tensor shapes, and model-loading edge cases.

## Commit & Pull Request Guidelines

Recent commits use short, imperative summaries such as `support qwen3.5`, `minor simplify`, and `fix scheduler typing`; follow that concise style unless a maintainer asks for a stricter convention. Pull requests should include a clear description, affected model/runtime paths, commands run, hardware assumptions for inference tests, and linked issues when applicable. Include logs or screenshots only when they clarify benchmark or multimodal output changes.

## Security & Configuration Tips

Do not commit model weights, generated caches, local images, or environment-specific paths. Keep large artifacts under external directories such as `~/huggingface/`, and document any required local path edits in PR notes.
