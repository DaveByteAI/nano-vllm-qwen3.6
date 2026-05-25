import torch
from torch import nn
import torch.distributed as dist

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.gated_delta_net import GatedDeltaNet
from nanovllm.layers.layernorm import GemmaRMSNorm
from nanovllm.layers.linear import ColumnParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import InterleavedMRoPE
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead


class Qwen3_5Attention(nn.Module):
    """Full attention with output gating, partial MRoPE for Qwen3.5."""

    def __init__(self, config, layer_idx: int) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        self.total_num_heads = config.num_attention_heads
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5

        # Q proj outputs 2x for gating: half query, half gate
        self.q_proj = ColumnParallelLinear(
            config.hidden_size, self.total_num_heads * self.head_dim * 2, bias=False,
        )
        self.k_proj = ColumnParallelLinear(
            config.hidden_size, self.total_num_kv_heads * self.head_dim, bias=False,
        )
        self.v_proj = ColumnParallelLinear(
            config.hidden_size, self.total_num_kv_heads * self.head_dim, bias=False,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim, config.hidden_size, bias=False,
        )

        self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # MRoPE (interleaved multi-dimensional rotary position embedding)
        rope_params = getattr(config, 'rope_parameters', None) or {}
        rope_theta = rope_params.get('rope_theta', getattr(config, 'rope_theta', 10000000))
        partial_rotary_factor = rope_params.get(
            'partial_rotary_factor', getattr(config, 'partial_rotary_factor', 0.25)
        )
        mrope_section = rope_params.get('mrope_section', [11, 11, 10])
        self.rotary_emb = InterleavedMRoPE(
            head_size=self.head_dim,
            partial_rotary_factor=partial_rotary_factor,
            mrope_section=mrope_section,
            max_position_embeddings=getattr(config, "max_model_len", config.max_position_embeddings),
            base=rope_theta,
        )

        self.attn = Attention(self.num_heads, self.head_dim, self.scaling, self.num_kv_heads)

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        # Q projection with gate (interleaved per-head: [q_h0, gate_h0, q_h1, gate_h1, ...])
        q_gate = self.q_proj(hidden_states)                             # [N, 2 * q_size]
        q_gate = q_gate.view(-1, self.num_heads, self.head_dim * 2)
        q, gate = q_gate.chunk(2, dim=-1)                              # each [N, num_heads, head_dim]

        k = self.k_proj(hidden_states).view(-1, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(-1, self.num_kv_heads, self.head_dim)

        # QK norm
        q = self.q_norm(q)
        k = self.k_norm(k)

        # MRoPE (handles both 1D and 3D positions)
        q, k = self.rotary_emb(positions, q, k)

        # Flash attention
        o = self.attn(q, k, v)
        if o.dim() == 4:
            o = o.squeeze(1)                                               # decode: [B,1,H,D] -> [B,H,D]

        # Output gating
        o = o * torch.sigmoid(gate)

        output = self.o_proj(o.flatten(1, -1))
        return output


class Qwen3_5MLP(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size, [config.intermediate_size] * 2, bias=False,
        )
        self.down_proj = RowParallelLinear(
            config.intermediate_size, config.hidden_size, bias=False,
        )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_up_proj(x)))


class Qwen3_5DecoderLayer(nn.Module):

    def __init__(self, config, layer_idx: int) -> None:
        super().__init__()
        layer_type = config.layer_types[layer_idx]

        if layer_type == "full_attention":
            self.self_attn = Qwen3_5Attention(config, layer_idx)
            self.linear_attn = None
        else:
            self.linear_attn = GatedDeltaNet(config, layer_idx)
            self.self_attn = None

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

        if self.self_attn is not None:
            hidden_states = self.self_attn(positions, hidden_states)
        else:
            hidden_states = self.linear_attn(hidden_states)

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3_5Model(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            Qwen3_5DecoderLayer(config, i) for i in range(config.num_hidden_layers)
        ])
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        image_embeds: torch.Tensor | None = None,
        image_token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        # Scatter vision embeddings into image token positions
        if image_embeds is not None and image_token_mask is not None:
            hidden_states[image_token_mask] = image_embeds.to(hidden_states.dtype)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3_5ForCausalLM(nn.Module):
    weight_prefix = "model.language_model."
    visual_prefix = "model.visual."
    packed_modules_mapping = {
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config, vision_config=None) -> None:
        super().__init__()
        self.model = Qwen3_5Model(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if getattr(config, 'tie_word_embeddings', False):
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

        # Vision encoder (optional)
        self.visual = None
        if vision_config is not None:
            from nanovllm.models.vision_encoder import Qwen3VLVisionEncoder
            self.visual = Qwen3VLVisionEncoder(vision_config)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        image_token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Run vision encoder if we have image data
        image_embeds = None
        if pixel_values is not None and self.visual is not None:
            image_embeds = self.visual(pixel_values, image_grid_thw)

        return self.model(input_ids, positions, image_embeds, image_token_mask)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)
