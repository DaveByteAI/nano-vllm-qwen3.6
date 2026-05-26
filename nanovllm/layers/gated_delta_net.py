import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.layers.layernorm import RMSNormGated
from nanovllm.layers.linear import ColumnParallelLinear, RowParallelLinear
from nanovllm.utils.context import get_context
from nanovllm.utils.quant import maybe_dequant_fp8_weight


def l2norm(x, dim=-1, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def causal_conv1d_prefill(x, weight, conv_state):
    """Causal depthwise conv1d for prefill.
    x: [1, D, L], weight: [D, 1, K], conv_state: [1, D, K-1] (view, updated in-place).
    """
    x_padded = torch.cat([conv_state, x], dim=-1)
    conv_state.copy_(x_padded[:, :, -(weight.size(-1) - 1):])
    out = F.conv1d(x_padded, weight, groups=x.size(1), padding=0)
    return F.silu(out)


def causal_conv1d_decode(x, weight, conv_state):
    """Causal depthwise conv1d for decode (single token).
    x: [B, D, 1], weight: [D, 1, K], conv_state: [B, D, K-1] (copy, must write back).
    Returns (out, new_conv_state).
    """
    weight = weight.squeeze(1)
    out = (conv_state * weight[:, :-1].unsqueeze(0)).sum(dim=-1, keepdim=True)
    out = out + x * weight[:, -1:].unsqueeze(0)
    new_conv_state = torch.empty_like(conv_state)
    new_conv_state[:, :, :-1] = conv_state[:, :, 1:]
    new_conv_state[:, :, -1:] = x
    return F.silu(out), new_conv_state


def chunk_gated_delta_rule(query, key, value, g, beta, initial_state=None, chunk_size=64):
    """Chunk-wise gated delta rule (prefill path).
    All inputs: [1, S, H, D] except g/beta: [1, S, H]. Returns (output, final_state).
    """
    initial_dtype = query.dtype
    query, key = l2norm(query), l2norm(key)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    B, H, S, Dk = key.shape
    Dv = value.shape[-1]
    pad = (chunk_size - S % chunk_size) % chunk_size
    if pad:
        query = F.pad(query, (0, 0, 0, pad))
        key = F.pad(key, (0, 0, 0, pad))
        value = F.pad(value, (0, 0, 0, pad))
        beta = F.pad(beta, (0, pad))
        g = F.pad(g, (0, pad))
    T = S + pad
    scale = Dk ** -0.5
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)

    query, key, value, k_beta, v_beta = [
        x.reshape(B, H, -1, chunk_size, x.shape[-1]) for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(B, H, -1, chunk_size)
    upper_mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)

    g = g.cumsum(dim=-1)
    decay_mask = (g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(upper_mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))

    state = (
        torch.zeros(B, H, Dk, Dv, device=query.device, dtype=query.dtype)
        if initial_state is None else initial_state.to(dtype=query.dtype, device=query.device)
    )
    core_out = torch.zeros_like(value)
    causal_mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)

    for i in range(T // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn_intra = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(causal_mask, 0)
        v_prime = k_cumdecay[:, :, i] @ state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ state
        core_out[:, :, i] = attn_inter + attn_intra @ v_new
        state = (
            state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    core_out = core_out.reshape(B, H, -1, core_out.shape[-1])[:, :, :S]
    core_out = core_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_out, state


def recurrent_gated_delta_rule(query, key, value, g, beta, state):
    """Recurrent gated delta rule for decode (single token per sequence).
    query/key: [B,1,H,Dk], value: [B,1,H,Dv], g/beta: [B,1,H], state: [B,H,Dk,Dv] (modified in-place).
    Returns output: [B,1,H,Dv].
    """
    initial_dtype = query.dtype
    query, key = l2norm(query), l2norm(key)
    query, key, value = [x.to(torch.float32) for x in (query, key, value)]
    g, beta = g.to(torch.float32), beta.to(torch.float32)
    scale = key.shape[-1] ** -0.5

    q = query[:, 0] * scale     # [B, H, Dk]
    k = key[:, 0]
    v = value[:, 0]             # [B, H, Dv]
    g_t = g[:, 0].exp().unsqueeze(-1).unsqueeze(-1)   # [B, H, 1, 1]
    beta_t = beta[:, 0].unsqueeze(-1)                  # [B, H, 1]

    state.mul_(g_t)
    kv_mem = (state * k.unsqueeze(-1)).sum(dim=-2)     # [B, H, Dv]
    delta = (v - kv_mem) * beta_t
    state.add_(k.unsqueeze(-1) * delta.unsqueeze(-2))
    output = (state * q.unsqueeze(-1)).sum(dim=-2)     # [B, H, Dv]
    return output.unsqueeze(1).to(initial_dtype)


class GatedDeltaNet(nn.Module):

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        self.hidden_size = config.hidden_size
        self.total_num_v_heads = config.linear_num_value_heads
        self.total_num_k_heads = config.linear_num_key_heads
        assert self.total_num_v_heads % self.tp_size == 0
        assert self.total_num_k_heads % self.tp_size == 0
        self.num_v_heads = self.total_num_v_heads // self.tp_size
        self.num_k_heads = self.total_num_k_heads // self.tp_size
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.total_key_dim = self.head_k_dim * self.total_num_k_heads
        self.total_value_dim = self.head_v_dim * self.total_num_v_heads
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.gqa_ratio = self.num_v_heads // self.num_k_heads
        self.layer_idx = layer_idx

        self.in_proj_qkv = nn.Linear(self.hidden_size, self.conv_dim, bias=False)
        self.in_proj_qkv.weight.weight_loader = self.qkv_weight_loader
        self.in_proj_z = ColumnParallelLinear(self.hidden_size, self.total_value_dim, bias=False)
        self.in_proj_b = ColumnParallelLinear(self.hidden_size, self.total_num_v_heads, bias=False)
        self.in_proj_a = ColumnParallelLinear(self.hidden_size, self.total_num_v_heads, bias=False)

        self.conv1d = nn.Conv1d(
            self.conv_dim, self.conv_dim, bias=False,
            kernel_size=self.conv_kernel_size, groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
        )
        self.conv1d.weight.weight_loader = self.qkv_weight_loader
        self.A_log = nn.Parameter(torch.empty(self.num_v_heads, dtype=torch.float32))
        self.dt_bias = nn.Parameter(torch.empty(self.num_v_heads))
        self.A_log.weight_loader = self.head_weight_loader
        self.dt_bias.weight_loader = self.head_weight_loader
        self.norm = RMSNormGated(self.head_v_dim, eps=config.rms_norm_eps)
        # Force norm weight to float32 (matches checkpoint)
        self.norm.weight = nn.Parameter(self.norm.weight.data.to(torch.float32))
        self.out_proj = RowParallelLinear(self.total_value_dim, self.hidden_size, bias=False)

        # State pools — allocated by model_runner after init
        self.conv_states: torch.Tensor = torch.tensor([])
        self.recurrent_states: torch.Tensor = torch.tensor([])

    def qkv_weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_scale: torch.Tensor | None = None,
    ):
        q_start = self.tp_rank * self.key_dim
        k_start = self.total_key_dim + self.tp_rank * self.key_dim
        v_start = 2 * self.total_key_dim + self.tp_rank * self.value_dim
        q = loaded_weight.narrow(0, q_start, self.key_dim)
        k = loaded_weight.narrow(0, k_start, self.key_dim)
        v = loaded_weight.narrow(0, v_start, self.value_dim)
        q = maybe_dequant_fp8_weight(q, loaded_scale, row_start=q_start)
        k = maybe_dequant_fp8_weight(k, loaded_scale, row_start=k_start)
        v = maybe_dequant_fp8_weight(v, loaded_scale, row_start=v_start)
        param.data.copy_(torch.cat([q, k, v], dim=0))

    def head_weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_scale: torch.Tensor | None = None,
    ):
        start = self.tp_rank * self.num_v_heads
        param.data.copy_(loaded_weight.narrow(0, start, self.num_v_heads))

    def _forward_prefill(self, hidden_states):
        context = get_context()
        cu_seqlens = context.cu_seqlens_q
        state_indices = context.state_indices
        num_seqs = cu_seqlens.size(0) - 1
        conv_weight = self.conv1d.weight
        outputs = []
        warmup = state_indices is None or self.conv_states.numel() == 0

        for i in range(num_seqs):
            start, end = cu_seqlens[i].item(), cu_seqlens[i + 1].item()
            seq_len = end - start

            x = hidden_states[start:end].unsqueeze(0)          # [1, L, H]
            if not warmup and context.block_tables is not None:
                si = state_indices[i].item()
                outputs.append(self._forward_prefill_recurrent(x, si).squeeze(0))
                continue

            mixed_qkv = self.in_proj_qkv(x).transpose(1, 2)    # [1, D, L]
            z = self.in_proj_z(x)                               # [1, L, Dv]
            b = self.in_proj_b(x)                               # [1, L, Hv]
            a = self.in_proj_a(x)                               # [1, L, Hv]

            # Causal conv1d
            if warmup:
                conv_state = torch.zeros(1, self.conv_dim, self.conv_kernel_size - 1, dtype=mixed_qkv.dtype, device=mixed_qkv.device)
            else:
                si = state_indices[i].item()
                conv_state = self.conv_states[si:si + 1]
            mixed_qkv = causal_conv1d_prefill(mixed_qkv, conv_weight, conv_state)

            mixed_qkv = mixed_qkv.transpose(1, 2)              # [1, L, D]
            q, k, v = mixed_qkv.split([self.key_dim, self.key_dim, self.value_dim], dim=-1)
            q = q.reshape(1, seq_len, self.num_k_heads, self.head_k_dim)
            k = k.reshape(1, seq_len, self.num_k_heads, self.head_k_dim)
            v = v.reshape(1, seq_len, self.num_v_heads, self.head_v_dim)

            beta = b.sigmoid()
            g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
            if self.gqa_ratio > 1:
                q = q.repeat_interleave(self.gqa_ratio, dim=2)
                k = k.repeat_interleave(self.gqa_ratio, dim=2)

            if warmup:
                rec_state = None
            else:
                rec_state = self.recurrent_states[si:si + 1]    # [1, Hv, Dk, Dv] view
            out, new_state = chunk_gated_delta_rule(q, k, v, g, beta, initial_state=rec_state)
            if not warmup:
                self.recurrent_states[si] = new_state[0]

            z_flat = z.reshape(-1, self.head_v_dim)
            out = self.norm(out.reshape(-1, self.head_v_dim), z_flat)
            out = self.out_proj(out.reshape(1, seq_len, self.value_dim))
            outputs.append(out.squeeze(0))

        return torch.cat(outputs, dim=0)

    def _forward_prefill_recurrent(self, x, state_index: int):
        outputs = []
        for t in range(x.size(1)):
            x_t = x[:, t:t + 1]
            mixed_qkv = self.in_proj_qkv(x_t).transpose(1, 2)
            z = self.in_proj_z(x_t)
            b = self.in_proj_b(x_t)
            a = self.in_proj_a(x_t)

            conv_state = self.conv_states[state_index:state_index + 1]
            mixed_qkv, new_conv_state = causal_conv1d_decode(mixed_qkv, self.conv1d.weight, conv_state)
            self.conv_states[state_index] = new_conv_state[0]

            mixed_qkv = mixed_qkv.transpose(1, 2)
            q, k, v = mixed_qkv.split([self.key_dim, self.key_dim, self.value_dim], dim=-1)
            q = q.reshape(1, 1, self.num_k_heads, self.head_k_dim)
            k = k.reshape(1, 1, self.num_k_heads, self.head_k_dim)
            v = v.reshape(1, 1, self.num_v_heads, self.head_v_dim)

            beta = b.sigmoid()
            g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
            if self.gqa_ratio > 1:
                q = q.repeat_interleave(self.gqa_ratio, dim=2)
                k = k.repeat_interleave(self.gqa_ratio, dim=2)

            rec_state = self.recurrent_states[state_index:state_index + 1]
            out = recurrent_gated_delta_rule(q, k, v, g, beta, rec_state)
            self.recurrent_states[state_index] = rec_state[0]

            z_flat = z.reshape(-1, self.head_v_dim)
            out = self.norm(out.reshape(-1, self.head_v_dim), z_flat)
            out = self.out_proj(out.reshape(1, 1, self.value_dim))
            outputs.append(out)
        return torch.cat(outputs, dim=1)

    def _forward_decode(self, hidden_states):
        context = get_context()
        state_indices = context.state_indices
        B = hidden_states.size(0)

        x = hidden_states.unsqueeze(1)                          # [B, 1, H]
        mixed_qkv = self.in_proj_qkv(x).transpose(1, 2)        # [B, D, 1]
        z = self.in_proj_z(x)                                   # [B, 1, Dv]
        b = self.in_proj_b(x)                                   # [B, 1, Hv]
        a = self.in_proj_a(x)                                   # [B, 1, Hv]

        # Conv1d decode
        conv_state = self.conv_states[state_indices]            # copy via fancy index
        mixed_qkv, new_conv_state = causal_conv1d_decode(mixed_qkv, self.conv1d.weight, conv_state)
        self.conv_states[state_indices] = new_conv_state

        mixed_qkv = mixed_qkv.transpose(1, 2)                  # [B, 1, D]
        q, k, v = mixed_qkv.split([self.key_dim, self.key_dim, self.value_dim], dim=-1)
        q = q.reshape(B, 1, self.num_k_heads, self.head_k_dim)
        k = k.reshape(B, 1, self.num_k_heads, self.head_k_dim)
        v = v.reshape(B, 1, self.num_v_heads, self.head_v_dim)

        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        if self.gqa_ratio > 1:
            q = q.repeat_interleave(self.gqa_ratio, dim=2)
            k = k.repeat_interleave(self.gqa_ratio, dim=2)

        rec_state = self.recurrent_states[state_indices]
        out = recurrent_gated_delta_rule(q, k, v, g, beta, rec_state)
        self.recurrent_states[state_indices] = rec_state

        z_flat = z.reshape(-1, self.head_v_dim)
        out = self.norm(out.reshape(-1, self.head_v_dim), z_flat)
        out = self.out_proj(out.reshape(B, 1, self.value_dim)).squeeze(1)
        return out

    def forward(self, hidden_states):
        context = get_context()
        if context.is_prefill:
            return self._forward_prefill(hidden_states)
        else:
            return self._forward_decode(hidden_states)
