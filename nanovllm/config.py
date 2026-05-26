import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    full_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    is_hybrid: bool = False
    max_state_slots: int = 0
    enable_vision: bool = True
    enable_mtp: bool = False
    # Vision / multimodal
    vision_config: object | None = None
    image_token_id: int = -1
    vision_start_token_id: int = -1
    vision_end_token_id: int = -1

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.full_config = AutoConfig.from_pretrained(self.model)
        # Extract multimodal token IDs from full config
        if hasattr(self.full_config, 'image_token_id'):
            self.image_token_id = self.full_config.image_token_id
        if hasattr(self.full_config, 'vision_start_token_id'):
            self.vision_start_token_id = self.full_config.vision_start_token_id
        if hasattr(self.full_config, 'vision_end_token_id'):
            self.vision_end_token_id = self.full_config.vision_end_token_id
        # Extract vision config
        if self.enable_vision and hasattr(self.full_config, 'vision_config'):
            self.vision_config = self.full_config.vision_config
        # For multimodal configs (Qwen3.5), use the text sub-config as hf_config
        if hasattr(self.full_config, 'text_config'):
            self.hf_config = self.full_config.text_config
        else:
            self.hf_config = self.full_config
        # Ensure dtype is set
        if not hasattr(self.hf_config, 'dtype') or self.hf_config.dtype is None:
            import torch
            self.hf_config.dtype = torch.bfloat16
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        self.hf_config.max_model_len = self.max_model_len
        self.hf_config.enable_mtp = self.enable_mtp
        self.is_hybrid = hasattr(self.hf_config, 'layer_types')
