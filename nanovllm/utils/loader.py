import os
from glob import glob
from dataclasses import dataclass
import torch
from torch import nn
from safetensors import safe_open

from nanovllm.utils.quant import maybe_dequant_fp8_weight


@dataclass
class LoadResult:
    loaded_names: list[str]
    skipped_names: list[str]


def default_weight_loader(
    param: nn.Parameter,
    loaded_weight: torch.Tensor,
    loaded_scale: torch.Tensor | None = None,
):
    loaded_weight = maybe_dequant_fp8_weight(loaded_weight, loaded_scale)
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str, log_fn=None):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    weight_prefix = getattr(model, "weight_prefix", "")
    visual_prefix = getattr(model, "visual_prefix", "")
    loaded_names = []
    skipped_names = []
    files = sorted(glob(os.path.join(path, "*.safetensors")))
    if log_fn is not None:
        log_fn(f"loading weights from {len(files)} safetensors files")
    for idx, file in enumerate(files, 1):
        if log_fn is not None:
            log_fn(f"loading weights {idx}/{len(files)}: {os.path.basename(file)}")
        with safe_open(file, "pt", "cpu") as f:
            weight_names = set(f.keys())
            for weight_name in f.keys():
                if weight_name.endswith(".weight_scale_inv"):
                    continue
                loaded_weight = f.get_tensor(weight_name)
                scale_name = weight_name + "_scale_inv"
                loaded_scale = (
                    f.get_tensor(scale_name)
                    if loaded_weight.dtype == torch.float8_e4m3fn and scale_name in weight_names
                    else None
                )

                # Handle visual encoder weights (e.g. "model.visual." -> "visual.")
                if visual_prefix and weight_name.startswith(visual_prefix):
                    mapped_name = weight_name[len(visual_prefix) - len("visual."):]
                    # visual weights map to: visual.* on the model
                    mapped_name = weight_name.replace(visual_prefix, "visual.")
                    try:
                        param = model.get_parameter(mapped_name)
                    except (AttributeError, KeyError):
                        skipped_names.append(weight_name)
                        continue
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, loaded_weight, loaded_scale)
                    loaded_names.append(weight_name)
                    continue

                # Strip model-specific prefix (e.g. "model.language_model." -> "model.")
                mapped_name = weight_name
                if weight_prefix and mapped_name.startswith(weight_prefix):
                    mapped_name = "model." + mapped_name[len(weight_prefix):]

                # Check packed modules
                matched_packed = False
                for k in packed_modules_mapping:
                    if k in mapped_name:
                        matched_packed = True
                        v, shard_id = packed_modules_mapping[k]
                        param_name = mapped_name.replace(k, v)
                        try:
                            param = model.get_parameter(param_name)
                        except (AttributeError, KeyError):
                            skipped_names.append(weight_name)
                            break
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, loaded_weight, shard_id, loaded_scale)
                        loaded_names.append(weight_name)
                        break
                if matched_packed:
                    continue

                try:
                    param = model.get_parameter(mapped_name)
                except (AttributeError, KeyError):
                    skipped_names.append(weight_name)
                    continue
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight, loaded_scale)
                loaded_names.append(weight_name)
    if log_fn is not None:
        log_fn("weights loaded")
    return LoadResult(loaded_names, skipped_names)
