import torch
from torch import nn

from nanovllm.layers.layernorm import GemmaRMSNorm
from nanovllm.layers.linear import ReplicatedLinear
from nanovllm.models.qwen3_5 import Qwen3_5Attention, Qwen3_5MLP


class Qwen3MTPDecoderLayer(nn.Module):
    """Single full-attention decoder layer used by Qwen3.6 MTP."""

    def __init__(self, config, layer_idx: int) -> None:
        super().__init__()
        self.self_attn = Qwen3_5Attention(config, layer_idx)
        self.mlp = Qwen3_5MLP(config)
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3MTP(nn.Module):
    """Qwen3.6 multi-token prediction head prototype.

    This module only provides weight loading and single-step forward. It is not wired
    into speculative decoding yet.
    """

    def __init__(self, config) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        num_layers = getattr(config, "num_nextn_predict_layers", None) or getattr(config, "mtp_num_layers", 1) or 1
        self.pre_fc_norm_embedding = GemmaRMSNorm(hidden_size, eps=config.rms_norm_eps)
        self.pre_fc_norm_hidden = GemmaRMSNorm(hidden_size, eps=config.rms_norm_eps)
        self.fc = ReplicatedLinear(hidden_size * 2, hidden_size, bias=False)
        self.layers = nn.ModuleList([
            Qwen3MTPDecoderLayer(config, i) for i in range(num_layers)
        ])
        self.norm = GemmaRMSNorm(hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor,
    ) -> torch.Tensor:
        inputs_embeds = self.pre_fc_norm_embedding(inputs_embeds)
        hidden_states = self.pre_fc_norm_hidden(hidden_states)
        hidden_states = self.fc(torch.cat([inputs_embeds, hidden_states], dim=-1))

        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states
