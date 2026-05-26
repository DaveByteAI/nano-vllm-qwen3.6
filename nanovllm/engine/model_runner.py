import pickle
import torch
import torch.distributed as dist
import torch._dynamo
from copy import copy
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model

_DTYPE_MAP = {torch.float32: 0, torch.float16: 1, torch.bfloat16: 2}
_IDX_MAP = {v: k for k, v in _DTYPE_MAP.items()}

def _dtype_to_idx(dtype):
    return _DTYPE_MAP.get(dtype, 0)

def _idx_to_dtype(idx):
    return _IDX_MAP.get(idx, torch.float32)


def _create_model(hf_config, vision_config=None):
    """Auto-detect model type and create the appropriate model."""
    model_type = getattr(hf_config, 'model_type', '')
    if 'qwen3_5' in model_type:
        from nanovllm.models.qwen3_5 import Qwen3_5ForCausalLM
        return Qwen3_5ForCausalLM(hf_config, vision_config=vision_config)
    else:
        from nanovllm.models.qwen3 import Qwen3ForCausalLM
        return Qwen3ForCausalLM(hf_config)


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)

        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event
        self.decode_state_snapshots = {}
        self.logit_probe_refs = {}

        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device("cuda")
        self._log("creating model")
        self.model = _create_model(hf_config, vision_config=config.vision_config)
        self.load_result = load_model(self.model, config.model, log_fn=self._log)
        if config.enable_mtp:
            mtp_skipped = [name for name in self.load_result.skipped_names if name.startswith("mtp.")]
            mtp_loaded = [name for name in self.load_result.loaded_names if name.startswith("mtp.")]
            assert not mtp_skipped, f"MTP weights were skipped: {mtp_skipped[:8]}"
            self._log(f"mtp weights loaded: {len(mtp_loaded)} tensors")
        self.sampler = Sampler()
        # Store multimodal config
        self.image_token_id = config.image_token_id
        self.vision_config = config.vision_config
        self.spatial_merge_size = getattr(config.vision_config, 'spatial_merge_size', 2) if config.vision_config else 2
        self._log("allocating runtime buffers")
        self.allocate_runtime_buffers()
        self._log("warming up model")
        self.warmup_model()
        self._log("allocating kv cache")
        self.allocate_kv_cache()
        if config.is_hybrid:
            self._log("allocating gated delta state")
            self.allocate_gdn_state()
        if not self.enforce_eager:
            self._log("capturing cuda graphs")
            self.capture_cudagraph()
        self._log("model runner ready")
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def _log(self, message: str):
        if self.rank == 0:
            print(f"[ModelRunner rank 0] {message}", flush=True)

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
            if hasattr(self, "verify_graphs"):
                del self.verify_graphs, self.verify_graph_vars
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        seq_len = min(max_num_batched_tokens, max_model_len)
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        # Count only layers that actually use KV cache (full attention layers)
        num_kv_layers = sum(1 for m in self.model.modules() if hasattr(m, "k_cache") and hasattr(m, "v_cache"))
        block_bytes = 2 * num_kv_layers * self.block_size * num_kv_heads * head_dim * hf_config.dtype.itemsize
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0
        self.kv_cache = torch.empty(2, num_kv_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def allocate_gdn_state(self):
        """Allocate conv_states and recurrent_states pools for GatedDeltaNet layers."""
        config = self.config
        hf_config = config.hf_config
        # Collect all GDN layers
        from nanovllm.layers.gated_delta_net import GatedDeltaNet
        gdn_layers = [m for m in self.model.modules() if isinstance(m, GatedDeltaNet)]
        if not gdn_layers:
            return
        ref = gdn_layers[0]
        conv_dim = ref.conv_dim
        kernel_size = ref.conv_kernel_size
        num_v_heads = ref.num_v_heads
        head_k_dim = ref.head_k_dim
        head_v_dim = ref.head_v_dim

        # Compute how many state slots we can afford from remaining GPU memory
        free, total = torch.cuda.mem_get_info()
        # Per-slot memory: conv_state + recurrent_state (one per GDN layer)
        num_gdn = len(gdn_layers)
        conv_bytes_per_slot = num_gdn * conv_dim * (kernel_size - 1) * hf_config.dtype.itemsize
        rec_bytes_per_slot = num_gdn * num_v_heads * head_k_dim * head_v_dim * 4  # float32
        bytes_per_slot = conv_bytes_per_slot + rec_bytes_per_slot
        # Use 90% of remaining free memory for state slots
        max_slots = int(free * 0.9) // bytes_per_slot
        max_slots = min(max_slots, config.max_num_seqs)
        assert max_slots > 0, f"Not enough GPU memory for GDN state slots (need {bytes_per_slot} bytes per slot, have {free} free)"
        config.max_state_slots = max_slots

        # Allocate pools and assign to each GDN layer
        for layer in gdn_layers:
            layer.conv_states = torch.zeros(max_slots, conv_dim, kernel_size - 1, dtype=hf_config.dtype, device="cuda")
            layer.recurrent_states = torch.zeros(max_slots, num_v_heads, head_k_dim, head_v_dim, dtype=torch.float32, device="cuda")

    def reset_gdn_state_slots(self, slot_ids: list[int]):
        if not self.config.is_hybrid or not slot_ids:
            return
        from nanovllm.layers.gated_delta_net import GatedDeltaNet
        indices = torch.tensor(slot_ids, dtype=torch.long, device="cuda")
        for layer in self.model.modules():
            if isinstance(layer, GatedDeltaNet):
                layer.conv_states.index_fill_(0, indices, 0)
                layer.recurrent_states.index_fill_(0, indices, 0)

    def _kv_slots_for_positions(self, seq: Sequence, start_pos: int, num_slots: int) -> list[int]:
        return [
            seq.block_table[pos // self.block_size] * self.block_size + pos % self.block_size
            for pos in range(start_pos, start_pos + num_slots)
        ]

    def _decode_kv_slots(self, seqs: list[Sequence]) -> list[int]:
        return [self._kv_slots_for_positions(seq, len(seq) - 1, 1)[0] for seq in seqs]

    def save_decode_state(self, name: str, seqs: list[Sequence]):
        kv_slots = self._decode_kv_slots(seqs)
        return self._save_decode_state_slots(name, seqs, kv_slots)

    def save_decode_state_range(self, name: str, seqs: list[Sequence], start_pos: int, num_slots: int):
        assert len(seqs) == 1, "decode state range currently supports batch size 1"
        kv_slots = self._kv_slots_for_positions(seqs[0], start_pos, num_slots)
        return self._save_decode_state_slots(name, seqs, kv_slots)

    def _save_decode_state_slots(self, name: str, seqs: list[Sequence], kv_slots: list[int]):
        state_slot_ids = [seq.state_slot_id for seq in seqs if seq.state_slot_id != -1]
        snapshot = {
            "kv_slots": kv_slots,
            "state_slot_ids": state_slot_ids,
            "kv_cache": None,
            "gdn_states": [],
        }

        if kv_slots:
            kv_indices = torch.tensor(kv_slots, dtype=torch.long, device="cuda")
            flat_kv = self.kv_cache.flatten(2, 3)
            snapshot["kv_cache"] = flat_kv.index_select(2, kv_indices).clone()

        if self.config.is_hybrid and state_slot_ids:
            from nanovllm.layers.gated_delta_net import GatedDeltaNet
            state_indices = torch.tensor(state_slot_ids, dtype=torch.long, device="cuda")
            for layer in self.model.modules():
                if isinstance(layer, GatedDeltaNet):
                    snapshot["gdn_states"].append((
                        layer.conv_states.index_select(0, state_indices).clone(),
                        layer.recurrent_states.index_select(0, state_indices).clone(),
                    ))

        self.decode_state_snapshots[name] = snapshot
        if self.rank != 0:
            return None
        return {
            "kv_slots": kv_slots,
            "state_slot_ids": state_slot_ids,
            "gdn_layers": len(snapshot["gdn_states"]),
        }

    def restore_decode_state(self, name: str):
        snapshot = self.decode_state_snapshots[name]
        kv_slots = snapshot["kv_slots"]
        state_slot_ids = snapshot["state_slot_ids"]

        if kv_slots and snapshot["kv_cache"] is not None:
            kv_indices = torch.tensor(kv_slots, dtype=torch.long, device="cuda")
            flat_kv = self.kv_cache.flatten(2, 3)
            flat_kv.index_copy_(2, kv_indices, snapshot["kv_cache"])

        if self.config.is_hybrid and state_slot_ids:
            from nanovllm.layers.gated_delta_net import GatedDeltaNet
            state_indices = torch.tensor(state_slot_ids, dtype=torch.long, device="cuda")
            gdn_iter = iter(snapshot["gdn_states"])
            for layer in self.model.modules():
                if isinstance(layer, GatedDeltaNet):
                    conv_state, recurrent_state = next(gdn_iter)
                    layer.conv_states.index_copy_(0, state_indices, conv_state)
                    layer.recurrent_states.index_copy_(0, state_indices, recurrent_state)

    def drop_decode_state(self, name: str):
        self.decode_state_snapshots.pop(name, None)
        self.logit_probe_refs.pop(name, None)

    def allocate_runtime_buffers(self):
        """Preallocate small staging tensors used on every decode step."""
        max_num_seqs = self.config.max_num_seqs
        max_num_blocks = (self.config.max_model_len + self.block_size - 1) // self.block_size
        self.decode_cpu_input_ids = torch.empty(max_num_seqs, dtype=torch.int64, device="cpu", pin_memory=True)
        self.decode_cpu_positions = torch.empty(max_num_seqs, dtype=torch.int64, device="cpu", pin_memory=True)
        self.decode_cpu_slot_mapping = torch.empty(max_num_seqs, dtype=torch.int32, device="cpu", pin_memory=True)
        self.decode_cpu_context_lens = torch.empty(max_num_seqs, dtype=torch.int32, device="cpu", pin_memory=True)
        self.decode_cpu_block_tables = torch.empty(max_num_seqs, max_num_blocks, dtype=torch.int32, device="cpu", pin_memory=True)
        self.decode_gpu_input_ids = torch.empty(max_num_seqs, dtype=torch.int64, device="cuda")
        self.decode_gpu_positions = torch.empty(max_num_seqs, dtype=torch.int64, device="cuda")
        self.decode_gpu_slot_mapping = torch.empty(max_num_seqs, dtype=torch.int32, device="cuda")
        self.decode_gpu_context_lens = torch.empty(max_num_seqs, dtype=torch.int32, device="cuda")
        self.decode_gpu_block_tables = torch.empty(max_num_seqs, max_num_blocks, dtype=torch.int32, device="cuda")
        self.sample_cpu_temperatures = torch.empty(max_num_seqs, dtype=torch.float32, device="cpu", pin_memory=True)
        self.sample_gpu_temperatures = torch.empty(max_num_seqs, dtype=torch.float32, device="cuda")
        if self.config.is_hybrid:
            self.decode_cpu_state_indices = torch.empty(max_num_seqs, dtype=torch.int32, device="cpu", pin_memory=True)
            self.decode_gpu_state_indices = torch.empty(max_num_seqs, dtype=torch.int32, device="cuda")
        self.verify_graph_lens = [1, 2, 3, 4]
        max_verify_len = max(self.verify_graph_lens)
        self.verify_cpu_input_ids = torch.empty(max_verify_len, dtype=torch.int64, device="cpu", pin_memory=True)
        self.verify_cpu_positions = torch.empty(max_verify_len, dtype=torch.int64, device="cpu", pin_memory=True)
        self.verify_cpu_slot_mapping = torch.empty(max_verify_len, dtype=torch.int32, device="cpu", pin_memory=True)
        self.verify_cpu_context_lens = torch.empty(max_verify_len, dtype=torch.int32, device="cpu", pin_memory=True)
        self.verify_cpu_block_tables = torch.empty(1, max_num_blocks, dtype=torch.int32, device="cpu", pin_memory=True)
        if self.config.is_hybrid:
            self.verify_cpu_state_indices = torch.empty(1, dtype=torch.int32, device="cpu", pin_memory=True)

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def _compute_mrope_positions(self, token_ids: list[int], image_grid_thw: torch.Tensor) -> torch.Tensor:
        """Compute 3D MRoPE position IDs for a sequence with vision tokens.

        Returns: (3, seq_len) tensor with [temporal, height, width] positions.
        """
        merge = self.spatial_merge_size
        image_token_id = self.image_token_id
        n = len(token_ids)
        positions = torch.zeros(3, n, dtype=torch.long)
        current_pos = 0
        image_idx = 0
        i = 0
        while i < n:
            if token_ids[i] == image_token_id:
                # Find contiguous span of image tokens
                j = i
                while j < n and token_ids[j] == image_token_id:
                    j += 1
                # Get grid for this image
                t = image_grid_thw[image_idx, 0].item()
                h = image_grid_thw[image_idx, 1].item()
                w = image_grid_thw[image_idx, 2].item()
                llm_h = h // merge
                llm_w = w // merge
                llm_t = t
                num_vision_tokens = llm_t * llm_h * llm_w
                # Temporal: all same
                positions[0, i:j] = current_pos
                # Height and width: grid pattern, repeated for each temporal frame
                h_pos = torch.arange(llm_h).repeat_interleave(llm_w) + current_pos
                w_pos = torch.arange(llm_w).repeat(llm_h) + current_pos
                frame_positions_h = h_pos
                frame_positions_w = w_pos
                if llm_t > 1:
                    frame_positions_h = frame_positions_h.repeat(llm_t)
                    frame_positions_w = frame_positions_w.repeat(llm_t)
                positions[1, i:i + num_vision_tokens] = frame_positions_h[:num_vision_tokens]
                positions[2, i:i + num_vision_tokens] = frame_positions_w[:num_vision_tokens]
                current_pos += max(llm_h, llm_w)
                image_idx += 1
                i = j
            else:
                positions[:, i] = current_pos
                current_pos += 1
                i += 1
        return positions

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        state_indices = []
        # Multimodal data collection
        has_images = any(seq.pixel_values is not None for seq in seqs)
        pixel_values_list = []
        image_grid_thw_list = []
        all_positions_3d = [] if has_images else None
        image_token_mask_parts = [] if has_images else None

        for seq in seqs:
            seqlen = len(seq)
            start = min(seq.num_cached_tokens, seqlen - 1)
            seqlen_q = seq.num_scheduled_tokens
            seqlen_k = seqlen
            end = start + seqlen_q
            seq_token_ids = seq[start:end]
            input_ids.extend(seq_token_ids)

            if has_images and seq.pixel_values is not None and start == 0:
                # Compute 3D MRoPE positions for this sequence
                pos_3d = self._compute_mrope_positions(
                    seq.token_ids[:end], seq.image_grid_thw
                )
                all_positions_3d.append(pos_3d[:, start:end])
                # Build image token mask for the scheduled slice
                mask = torch.tensor([t == self.image_token_id for t in seq_token_ids], dtype=torch.bool)
                image_token_mask_parts.append(mask)
                # Collect pixel values
                pixel_values_list.append(seq.pixel_values)
                image_grid_thw_list.append(seq.image_grid_thw)
                # Use 1D positions from temporal dimension for position tracking (for KV cache compatibility)
                positions.extend(pos_3d[0, start:end].tolist())
            else:
                positions.extend(range(start, end))
                if has_images:
                    # Text-only sequence in a mixed batch: use 1D positions expanded to 3D
                    pos_1d = torch.arange(start, end, dtype=torch.long)
                    pos_3d = pos_1d.unsqueeze(0).expand(3, -1)
                    all_positions_3d.append(pos_3d)
                    mask = torch.zeros(seqlen_q, dtype=torch.bool)
                    image_token_mask_parts.append(mask)

            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            state_indices.append(seq.state_slot_id)
            if not seq.block_table:    # warmup
                continue
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    slot_start += start % self.block_size
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        if has_images:
            positions_3d = torch.cat(all_positions_3d, dim=1).cuda(non_blocking=True)
            positions = positions_3d  # 3D: (3, total_tokens)
        else:
            positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        state_indices_t = torch.tensor(state_indices, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True) if self.config.is_hybrid else None
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables, state_indices=state_indices_t)

        # Prepare multimodal tensors
        pixel_values = None
        image_grid_thw = None
        image_token_mask = None
        if has_images and pixel_values_list:
            pixel_values = torch.cat(pixel_values_list, dim=0).cuda(non_blocking=True)
            image_grid_thw = torch.cat(image_grid_thw_list, dim=0).cuda(non_blocking=True)
            image_token_mask = torch.cat(image_token_mask_parts, dim=0).cuda(non_blocking=True)
            # Clear pixel values from sequences after first prefill
            for seq in seqs:
                if seq.pixel_values is not None:
                    seq.pixel_values = None
                    seq.image_grid_thw = None

        return input_ids, positions, pixel_values, image_grid_thw, image_token_mask

    def prepare_decode(self, seqs: list[Sequence]):
        bs = len(seqs)
        max_blocks = 0
        for i, seq in enumerate(seqs):
            self.decode_cpu_input_ids[i] = seq.last_token
            self.decode_cpu_positions[i] = len(seq) - 1
            self.decode_cpu_context_lens[i] = len(seq)
            self.decode_cpu_slot_mapping[i] = seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1
            if self.config.is_hybrid:
                self.decode_cpu_state_indices[i] = seq.state_slot_id
            max_blocks = max(max_blocks, len(seq.block_table))

        block_tables_cpu = self.decode_cpu_block_tables[:bs, :max_blocks]
        block_tables_cpu.fill_(-1)
        for i, seq in enumerate(seqs):
            for j, block_id in enumerate(seq.block_table):
                block_tables_cpu[i, j] = block_id

        input_ids = self.decode_gpu_input_ids[:bs]
        positions = self.decode_gpu_positions[:bs]
        slot_mapping = self.decode_gpu_slot_mapping[:bs]
        context_lens = self.decode_gpu_context_lens[:bs]
        block_tables = self.decode_gpu_block_tables[:bs, :max_blocks]
        input_ids.copy_(self.decode_cpu_input_ids[:bs], non_blocking=True)
        positions.copy_(self.decode_cpu_positions[:bs], non_blocking=True)
        slot_mapping.copy_(self.decode_cpu_slot_mapping[:bs], non_blocking=True)
        context_lens.copy_(self.decode_cpu_context_lens[:bs], non_blocking=True)
        block_tables.copy_(block_tables_cpu, non_blocking=True)
        if self.config.is_hybrid:
            state_indices_t = self.decode_gpu_state_indices[:bs]
            state_indices_t.copy_(self.decode_cpu_state_indices[:bs], non_blocking=True)
        else:
            state_indices_t = None
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables, state_indices=state_indices_t)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        bs = len(seqs)
        for i, seq in enumerate(seqs):
            self.sample_cpu_temperatures[i] = seq.temperature
        temperatures = self.sample_gpu_temperatures[:bs]
        temperatures.copy_(self.sample_cpu_temperatures[:bs], non_blocking=True)
        return temperatures

    def sample(self, logits: torch.Tensor, temperatures: torch.Tensor | None, greedy: bool):
        if greedy:
            token_ids, scores = self.sampler.greedy_with_scores(logits)
        else:
            token_ids, scores = self.sampler.forward_with_scores(logits, temperatures)
        if self.world_size == 1:
            return token_ids.tolist()

        token_ids = token_ids + self.model.lm_head.vocab_start_idx
        all_scores = [torch.empty_like(scores) for _ in range(self.world_size)]
        all_token_ids = [torch.empty_like(token_ids) for _ in range(self.world_size)]
        dist.all_gather(all_scores, scores)
        dist.all_gather(all_token_ids, token_ids)
        if self.rank != 0:
            return None

        scores = torch.stack(all_scores, dim=0)
        token_ids = torch.stack(all_token_ids, dim=0)
        rank_ids = scores.argmax(dim=0, keepdim=True)
        return token_ids.gather(0, rank_ids).squeeze(0).tolist()

    def topk_tokens(self, logits: torch.Tensor, k: int):
        k = min(k, logits.size(-1))
        values, token_ids = torch.topk(logits.float(), k, dim=-1)
        token_ids = token_ids + self.model.lm_head.vocab_start_idx
        if self.world_size > 1:
            all_values = [torch.empty_like(values) for _ in range(self.world_size)]
            all_token_ids = [torch.empty_like(token_ids) for _ in range(self.world_size)]
            dist.all_gather(all_values, values)
            dist.all_gather(all_token_ids, token_ids)
            if self.rank != 0:
                return None
            values = torch.cat(all_values, dim=-1)
            token_ids = torch.cat(all_token_ids, dim=-1)
            values, indices = torch.topk(values, k, dim=-1)
            token_ids = token_ids.gather(1, indices)
        return [
            [
                {"token_id": int(token_id), "score": float(score)}
                for token_id, score in zip(row_ids, row_values)
            ]
            for row_ids, row_values in zip(token_ids.tolist(), values.tolist())
        ]

    def _compare_logits(self, logits: torch.Tensor, compare_logits_to: str | None):
        if compare_logits_to is None:
            return None
        ref = self.logit_probe_refs[compare_logits_to]
        ref = ref[:logits.size(0)]
        local_diff = (logits.float() - ref).abs().max()
        if self.world_size > 1:
            dist.all_reduce(local_diff, op=dist.ReduceOp.MAX)
        return float(local_diff.item())

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool,
                  pixel_values=None, image_grid_thw=None, image_token_mask=None):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(
                self.model(input_ids, positions,
                           pixel_values=pixel_values,
                           image_grid_thw=image_grid_thw,
                           image_token_mask=image_token_mask)
            )
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            if self.config.is_hybrid:
                graph_vars["state_indices"][:bs] = context.state_indices
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def _broadcast_image_data(self, pixel_values, image_grid_thw, image_token_mask, positions, is_3d_positions):
        """Broadcast multimodal tensors from rank 0 to all TP workers.

        For TP>1, rank 0 holds the image data (from Sequence objects) and must
        share it with other ranks so they can all run the vision encoder.
        """
        # Broadcast a flag: does this step have images?
        flag = torch.tensor([1 if pixel_values is not None else 0], dtype=torch.int32, device="cuda")
        dist.broadcast(flag, src=0)
        if flag.item() == 0:
            return pixel_values, image_grid_thw, image_token_mask, positions

        # Broadcast positions (may be 3D for multimodal)
        pos_flag = torch.tensor([1 if is_3d_positions else 0], dtype=torch.int32, device="cuda")
        dist.broadcast(pos_flag, src=0)
        if pos_flag.item() == 1:
            if self.rank == 0:
                shape_t = torch.tensor(list(positions.shape), dtype=torch.int64, device="cuda")
            else:
                shape_t = torch.empty(2, dtype=torch.int64, device="cuda")
            dist.broadcast(shape_t, src=0)
            if self.rank != 0:
                positions = torch.empty(shape_t.tolist(), dtype=torch.int64, device="cuda")
            dist.broadcast(positions, src=0)

        # Broadcast pixel_values
        if self.rank == 0:
            pv_shape = torch.tensor(list(pixel_values.shape), dtype=torch.int64, device="cuda")
            pv_dtype_idx = torch.tensor([_dtype_to_idx(pixel_values.dtype)], dtype=torch.int32, device="cuda")
        else:
            pv_shape = torch.empty(2, dtype=torch.int64, device="cuda")
            pv_dtype_idx = torch.empty(1, dtype=torch.int32, device="cuda")
        dist.broadcast(pv_shape, src=0)
        dist.broadcast(pv_dtype_idx, src=0)
        pv_dtype = _idx_to_dtype(pv_dtype_idx.item())
        if self.rank != 0:
            pixel_values = torch.empty(pv_shape.tolist(), dtype=pv_dtype, device="cuda")
        dist.broadcast(pixel_values, src=0)

        # Broadcast image_grid_thw
        if self.rank == 0:
            gt_shape = torch.tensor(list(image_grid_thw.shape), dtype=torch.int64, device="cuda")
        else:
            gt_shape = torch.empty(2, dtype=torch.int64, device="cuda")
        dist.broadcast(gt_shape, src=0)
        if self.rank != 0:
            image_grid_thw = torch.empty(gt_shape.tolist(), dtype=torch.int64, device="cuda")
        dist.broadcast(image_grid_thw, src=0)

        # Broadcast image_token_mask
        if self.rank == 0:
            mask_len = torch.tensor([image_token_mask.shape[0]], dtype=torch.int64, device="cuda")
        else:
            mask_len = torch.empty(1, dtype=torch.int64, device="cuda")
        dist.broadcast(mask_len, src=0)
        if self.rank != 0:
            image_token_mask = torch.empty(mask_len.item(), dtype=torch.bool, device="cuda")
        dist.broadcast(image_token_mask, src=0)

        return pixel_values, image_grid_thw, image_token_mask, positions

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        if is_prefill:
            input_ids, positions, pixel_values, image_grid_thw, image_token_mask = self.prepare_prefill(seqs)
            # For TP>1, broadcast image data from rank 0 to all workers
            if self.world_size > 1:
                is_3d = positions.ndim == 2  # (3, N) for multimodal
                pixel_values, image_grid_thw, image_token_mask, positions = \
                    self._broadcast_image_data(pixel_values, image_grid_thw, image_token_mask, positions, is_3d)
        else:
            input_ids, positions = self.prepare_decode(seqs)
            pixel_values = image_grid_thw = image_token_mask = None
        greedy = all(seq.temperature <= 1e-10 for seq in seqs)
        temperatures = None if greedy else self.prepare_sample(seqs)
        logits = self.run_model(input_ids, positions, is_prefill,
                                pixel_values=pixel_values,
                                image_grid_thw=image_grid_thw,
                                image_token_mask=image_token_mask)
        token_ids = self.sample(logits, temperatures, greedy)
        reset_context()
        return token_ids

    @torch.inference_mode()
    def run_step_probe(
        self,
        seqs: list[Sequence],
        is_prefill: bool,
        top_k: int = 5,
        save_logits_as: str | None = None,
        compare_logits_to: str | None = None,
    ):
        assert len(seqs) == 1, "step probe currently supports batch size 1"
        if is_prefill:
            input_ids, positions, pixel_values, image_grid_thw, image_token_mask = self.prepare_prefill(seqs)
            assert pixel_values is None and image_grid_thw is None and image_token_mask is None, \
                "step probe is text-only"
        else:
            input_ids, positions = self.prepare_decode(seqs)
            pixel_values = image_grid_thw = image_token_mask = None

        logits = self.run_model(
            input_ids,
            positions,
            is_prefill,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            image_token_mask=image_token_mask,
        )
        logits_float = logits.float()
        token_ids = self.sample(logits, None, greedy=True)
        topk = self.topk_tokens(logits, top_k)
        max_logit_diff = self._compare_logits(logits_float, compare_logits_to)

        if save_logits_as is not None:
            self.logit_probe_refs[save_logits_as] = logits_float.detach().clone()

        reset_context()
        if self.rank != 0:
            return None
        return {
            "token_ids": token_ids,
            "topk": topk,
            "logits_shape": tuple(logits.shape),
            "max_logit_diff": max_logit_diff,
        }

    @torch.inference_mode()
    def run_verify_batch_probe(
        self,
        seqs: list[Sequence],
        input_token_ids: list[int],
        start_pos: int,
        top_k: int = 5,
        save_logits_as: str | None = None,
        compare_logits_to: str | None = None,
        verify_mode: str = "eager",
    ):
        if verify_mode == "graph" and not self.enforce_eager and hasattr(self, "verify_graphs"):
            verify_len = len(input_token_ids)
            if verify_len in self.verify_graphs:
                return self._run_verify_graph_probe(
                    seqs,
                    input_token_ids,
                    start_pos,
                    top_k,
                    save_logits_as,
                    compare_logits_to,
                )

        assert len(seqs) == 1, "batch verify currently supports batch size 1"
        assert input_token_ids, "batch verify needs at least one input token"
        seq = seqs[0]
        verify_len = len(input_token_ids)
        assert input_token_ids[0] == seq.last_token
        context_len = start_pos + verify_len
        needed_blocks = (context_len + self.block_size - 1) // self.block_size
        assert len(seq.block_table) >= needed_blocks

        # Keep the batch verify boundary in one runner call, while replaying decode
        # steps internally. A true fused verify path needs separate GDN handling.
        work_seq = copy(seq)
        work_seq.token_ids = list(seq.token_ids)
        work_seq.block_table = list(seq.block_table)
        token_ids = []
        topk = []
        logits_parts = []
        for i, next_input_token_id in enumerate(input_token_ids):
            assert work_seq.last_token == next_input_token_id
            input_ids, positions = self.prepare_decode([work_seq])
            logits = self.run_model(input_ids, positions, False)
            logits_parts.append(logits.float())
            sampled_token_ids = self.sample(logits, None, greedy=True)
            sampled_topk = self.topk_tokens(logits, top_k)
            if self.rank == 0:
                token_ids.extend(sampled_token_ids)
                topk.extend(sampled_topk)
            reset_context()
            if i != verify_len - 1:
                work_seq.append_token(input_token_ids[i + 1])
                work_seq.num_cached_tokens += 1

        logits_float = torch.cat(logits_parts, dim=0)
        max_logit_diff = self._compare_logits(logits_float, compare_logits_to)

        if save_logits_as is not None:
            self.logit_probe_refs[save_logits_as] = logits_float.detach().clone()

        if self.rank != 0:
            return None
        return {
            "token_ids": token_ids,
            "topk": topk,
            "logits_shape": tuple(logits_float.shape),
            "max_logit_diff": max_logit_diff,
            "verify_len": verify_len,
            "context_len": context_len,
            "verify_mode_used": "eager",
        }

    def _run_verify_graph_probe(
        self,
        seqs: list[Sequence],
        input_token_ids: list[int],
        start_pos: int,
        top_k: int = 5,
        save_logits_as: str | None = None,
        compare_logits_to: str | None = None,
    ):
        assert len(seqs) == 1, "verify graph currently supports batch size 1"
        seq = seqs[0]
        verify_len = len(input_token_ids)
        assert input_token_ids and input_token_ids[0] == seq.last_token
        vars = self.verify_graph_vars[verify_len]

        needed_blocks = (start_pos + verify_len + self.block_size - 1) // self.block_size
        assert len(seq.block_table) >= needed_blocks

        block_tables_cpu = self.verify_cpu_block_tables[:, :needed_blocks]
        block_tables_cpu.fill_(-1)
        for i, token_id in enumerate(input_token_ids):
            pos = start_pos + i
            self.verify_cpu_input_ids[i] = token_id
            self.verify_cpu_positions[i] = pos
            self.verify_cpu_context_lens[i] = pos + 1
            self.verify_cpu_slot_mapping[i] = (
                seq.block_table[pos // self.block_size] * self.block_size
                + pos % self.block_size
            )
        for i, block_id in enumerate(seq.block_table[:needed_blocks]):
            block_tables_cpu[0, i] = block_id

        vars["input_ids"].copy_(self.verify_cpu_input_ids[:verify_len], non_blocking=True)
        vars["positions"].copy_(self.verify_cpu_positions[:verify_len], non_blocking=True)
        vars["slot_mapping"].copy_(self.verify_cpu_slot_mapping[:verify_len], non_blocking=True)
        vars["context_lens"].copy_(self.verify_cpu_context_lens[:verify_len], non_blocking=True)
        vars["block_tables"].fill_(-1)
        vars["block_tables"][:, :needed_blocks].copy_(block_tables_cpu, non_blocking=True)
        if self.config.is_hybrid:
            self.verify_cpu_state_indices[0] = seq.state_slot_id
            vars["state_indices"].copy_(self.verify_cpu_state_indices, non_blocking=True)

        self.verify_graphs[verify_len].replay()
        logits = self.model.compute_logits(vars["outputs"][:verify_len])
        logits_float = logits.float()
        token_ids = self.sample(logits, None, greedy=True)
        topk = self.topk_tokens(logits, top_k)
        max_logit_diff = self._compare_logits(logits_float, compare_logits_to)

        if save_logits_as is not None:
            self.logit_probe_refs[save_logits_as] = logits_float.detach().clone()

        if self.rank != 0:
            return None
        return {
            "token_ids": token_ids,
            "topk": topk,
            "logits_shape": tuple(logits_float.shape),
            "max_logit_diff": max_logit_diff,
            "verify_len": verify_len,
            "context_len": start_pos + verify_len,
            "verify_mode_used": "graph",
        }

    @torch.inference_mode()
    def run_verify_batch_fast(
        self,
        seqs: list[Sequence],
        input_token_ids: list[int],
        start_pos: int,
        verify_mode: str = "eager",
    ):
        if verify_mode == "graph" and not self.enforce_eager and hasattr(self, "verify_graphs"):
            verify_len = len(input_token_ids)
            if verify_len in self.verify_graphs:
                return self._run_verify_graph_fast(seqs, input_token_ids, start_pos)

        assert len(seqs) == 1, "fast verify currently supports batch size 1"
        assert input_token_ids, "fast verify needs at least one input token"
        seq = seqs[0]
        verify_len = len(input_token_ids)
        assert input_token_ids[0] == seq.last_token
        needed_blocks = (start_pos + verify_len + self.block_size - 1) // self.block_size
        assert len(seq.block_table) >= needed_blocks

        work_seq = copy(seq)
        work_seq.token_ids = list(seq.token_ids)
        work_seq.block_table = list(seq.block_table)
        token_ids = []
        for i, next_input_token_id in enumerate(input_token_ids):
            assert work_seq.last_token == next_input_token_id
            input_ids, positions = self.prepare_decode([work_seq])
            logits = self.run_model(input_ids, positions, False)
            sampled_token_ids = self.sample(logits, None, greedy=True)
            if self.rank == 0:
                token_ids.extend(sampled_token_ids)
            reset_context()
            if i != verify_len - 1:
                work_seq.append_token(input_token_ids[i + 1])
                work_seq.num_cached_tokens += 1

        if self.rank != 0:
            return None
        return {
            "token_ids": token_ids,
            "verify_mode_used": "eager",
        }

    def _run_verify_graph_fast(
        self,
        seqs: list[Sequence],
        input_token_ids: list[int],
        start_pos: int,
    ):
        assert len(seqs) == 1, "verify graph currently supports batch size 1"
        seq = seqs[0]
        verify_len = len(input_token_ids)
        assert input_token_ids and input_token_ids[0] == seq.last_token
        vars = self.verify_graph_vars[verify_len]

        needed_blocks = (start_pos + verify_len + self.block_size - 1) // self.block_size
        assert len(seq.block_table) >= needed_blocks

        block_tables_cpu = self.verify_cpu_block_tables[:, :needed_blocks]
        block_tables_cpu.fill_(-1)
        for i, token_id in enumerate(input_token_ids):
            pos = start_pos + i
            self.verify_cpu_input_ids[i] = token_id
            self.verify_cpu_positions[i] = pos
            self.verify_cpu_context_lens[i] = pos + 1
            self.verify_cpu_slot_mapping[i] = (
                seq.block_table[pos // self.block_size] * self.block_size
                + pos % self.block_size
            )
        for i, block_id in enumerate(seq.block_table[:needed_blocks]):
            block_tables_cpu[0, i] = block_id

        vars["input_ids"].copy_(self.verify_cpu_input_ids[:verify_len], non_blocking=True)
        vars["positions"].copy_(self.verify_cpu_positions[:verify_len], non_blocking=True)
        vars["slot_mapping"].copy_(self.verify_cpu_slot_mapping[:verify_len], non_blocking=True)
        vars["context_lens"].copy_(self.verify_cpu_context_lens[:verify_len], non_blocking=True)
        vars["block_tables"].fill_(-1)
        vars["block_tables"][:, :needed_blocks].copy_(block_tables_cpu, non_blocking=True)
        if self.config.is_hybrid:
            self.verify_cpu_state_indices[0] = seq.state_slot_id
            vars["state_indices"].copy_(self.verify_cpu_state_indices, non_blocking=True)

        self.verify_graphs[verify_len].replay()
        logits = self.model.compute_logits(vars["outputs"][:verify_len])
        token_ids = self.sample(logits, None, greedy=True)

        if self.rank != 0:
            return None
        return {
            "token_ids": token_ids,
            "verify_mode_used": "graph",
        }

    def _run_mtp_draft(
        self,
        seqs: list[Sequence],
        is_prefill: bool,
        top_k: int = 5,
        draft_len: int = 1,
    ):
        assert getattr(self.model, "mtp", None) is not None, "MTP is not enabled"
        assert len(seqs) == 1, "MTP probe currently supports batch size 1"
        assert draft_len >= 0

        if is_prefill:
            input_ids, positions, pixel_values, image_grid_thw, image_token_mask = self.prepare_prefill(seqs)
            assert pixel_values is None and image_grid_thw is None and image_token_mask is None, \
                "MTP probe is text-only"
            assert positions.ndim == 1, "MTP probe expects 1D text positions"
        else:
            input_ids, positions = self.prepare_decode(seqs)

        hidden_states = self.model(
            input_ids,
            positions,
            pixel_values=None,
            image_grid_thw=None,
            image_token_mask=None,
        )
        context = get_context()
        if is_prefill:
            last_indices = context.cu_seqlens_q[1:] - 1
            main_hidden = hidden_states[last_indices].contiguous()
            mtp_positions = positions[last_indices].contiguous() + 1
        else:
            main_hidden = hidden_states
            mtp_positions = positions.contiguous() + 1
        logits = self.model.compute_logits(hidden_states)
        main_token_ids = self.sample(logits, None, greedy=True)

        main_token_tensor = torch.empty(main_hidden.size(0), dtype=torch.int64, device="cuda")
        if self.rank == 0:
            main_token_tensor.copy_(torch.tensor(main_token_ids, dtype=torch.int64, device="cuda"))
        if self.world_size > 1:
            dist.broadcast(main_token_tensor, src=0)

        current_hidden = main_hidden
        current_positions = mtp_positions
        current_token_tensor = main_token_tensor
        draft_token_ids = []
        draft_topk = []
        mtp_hidden_shape = None
        draft_logits_shape = None
        for _ in range(draft_len):
            inputs_embeds = self.model.model.embed_tokens(current_token_tensor)
            cu_seqlens = torch.arange(
                current_hidden.size(0) + 1,
                dtype=torch.int32,
                device="cuda",
            )
            slot_mapping = torch.full(
                (current_hidden.size(0),),
                -1,
                dtype=torch.int32,
                device="cuda",
            )
            set_context(
                True,
                cu_seqlens,
                cu_seqlens,
                1,
                1,
                slot_mapping,
                None,
                None,
                state_indices=None,
            )
            mtp_hidden = self.model.mtp(current_positions, current_hidden, inputs_embeds)
            draft_logits = self.model.compute_logits(mtp_hidden)
            next_token_ids = self.sample(draft_logits, None, greedy=True)
            next_topk = self.topk_tokens(draft_logits, top_k)
            next_token_tensor = torch.empty(current_hidden.size(0), dtype=torch.int64, device="cuda")
            if self.rank == 0:
                draft_token_ids.append(next_token_ids[0])
                draft_topk.append(next_topk[0])
                next_token_tensor.copy_(torch.tensor(next_token_ids, dtype=torch.int64, device="cuda"))
            if self.world_size > 1:
                dist.broadcast(next_token_tensor, src=0)
            current_hidden = mtp_hidden
            current_positions = current_positions + 1
            current_token_tensor = next_token_tensor
            mtp_hidden_shape = tuple(mtp_hidden.shape)
            draft_logits_shape = tuple(draft_logits.shape)
        reset_context()

        if self.rank != 0:
            return None
        return {
            "main_token_ids": main_token_ids,
            "draft_token_ids": draft_token_ids,
            "draft_topk": draft_topk,
            "main_hidden_shape": tuple(main_hidden.shape),
            "mtp_hidden_shape": mtp_hidden_shape,
            "draft_logits_shape": draft_logits_shape,
            "draft_len": draft_len,
            "mtp_forwards": draft_len,
            "mtp_loaded_count": len([name for name in self.load_result.loaded_names if name.startswith("mtp.")]),
            "mtp_skipped_count": len([name for name in self.load_result.skipped_names if name.startswith("mtp.")]),
        }

    @torch.inference_mode()
    def run_mtp_probe(self, seqs: list[Sequence], top_k: int = 5):
        return self._run_mtp_draft(seqs, is_prefill=True, top_k=top_k)

    @torch.inference_mode()
    def run_mtp_draft_step(self, seqs: list[Sequence], is_prefill: bool, top_k: int = 5, draft_len: int = 1):
        return self._run_mtp_draft(seqs, is_prefill=is_prefill, top_k=top_k, draft_len=draft_len)

    @torch.inference_mode()
    def run_mtp_draft_fast_step(self, seqs: list[Sequence], is_prefill: bool, draft_len: int = 1):
        assert getattr(self.model, "mtp", None) is not None, "MTP is not enabled"
        assert len(seqs) == 1, "MTP fast path currently supports batch size 1"
        assert draft_len >= 0

        if is_prefill:
            input_ids, positions, pixel_values, image_grid_thw, image_token_mask = self.prepare_prefill(seqs)
            assert pixel_values is None and image_grid_thw is None and image_token_mask is None, \
                "MTP fast path is text-only"
            assert positions.ndim == 1, "MTP fast path expects 1D text positions"
        else:
            input_ids, positions = self.prepare_decode(seqs)

        hidden_states = self.model(input_ids, positions)
        context = get_context()
        if is_prefill:
            last_indices = context.cu_seqlens_q[1:] - 1
            main_hidden = hidden_states[last_indices].contiguous()
            mtp_positions = positions[last_indices].contiguous() + 1
        else:
            main_hidden = hidden_states
            mtp_positions = positions.contiguous() + 1
        logits = self.model.compute_logits(hidden_states)
        main_token_ids = self.sample(logits, None, greedy=True)

        main_token_tensor = torch.empty(main_hidden.size(0), dtype=torch.int64, device="cuda")
        if self.rank == 0:
            main_token_tensor.copy_(torch.tensor(main_token_ids, dtype=torch.int64, device="cuda"))
        if self.world_size > 1:
            dist.broadcast(main_token_tensor, src=0)

        current_hidden = main_hidden
        current_positions = mtp_positions
        current_token_tensor = main_token_tensor
        draft_token_ids = []
        for _ in range(draft_len):
            inputs_embeds = self.model.model.embed_tokens(current_token_tensor)
            cu_seqlens = torch.arange(current_hidden.size(0) + 1, dtype=torch.int32, device="cuda")
            slot_mapping = torch.full((current_hidden.size(0),), -1, dtype=torch.int32, device="cuda")
            set_context(
                True,
                cu_seqlens,
                cu_seqlens,
                1,
                1,
                slot_mapping,
                None,
                None,
                state_indices=None,
            )
            mtp_hidden = self.model.mtp(current_positions, current_hidden, inputs_embeds)
            draft_logits = self.model.compute_logits(mtp_hidden)
            next_token_ids = self.sample(draft_logits, None, greedy=True)
            next_token_tensor = torch.empty(current_hidden.size(0), dtype=torch.int64, device="cuda")
            if self.rank == 0:
                draft_token_ids.append(next_token_ids[0])
                next_token_tensor.copy_(torch.tensor(next_token_ids, dtype=torch.int64, device="cuda"))
            if self.world_size > 1:
                dist.broadcast(next_token_tensor, src=0)
            current_hidden = mtp_hidden
            current_positions = current_positions + 1
            current_token_tensor = next_token_tensor
        reset_context()

        if self.rank != 0:
            return None
        return {
            "main_token_ids": main_token_ids,
            "draft_token_ids": draft_token_ids,
            "mtp_forwards": draft_len,
        }

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        state_indices = torch.arange(max_bs, dtype=torch.int32) if config.is_hybrid else None
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [bs for bs in [1, 2, 4, 8] if bs <= max_bs]
        self.graph_bs += list(range(16, max_bs + 1, 16))
        if self.graph_bs[-1] != max_bs:
            self.graph_bs.append(max_bs)
        self.graph_bs = sorted(set(self.graph_bs))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(
                False,
                slot_mapping=slot_mapping[:bs],
                context_lens=context_lens[:bs],
                block_tables=block_tables[:bs],
                state_indices=state_indices[:bs] if state_indices is not None else None,
            )
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
        if state_indices is not None:
            self.graph_vars["state_indices"] = state_indices
        if config.is_hybrid:
            self.reset_gdn_state_slots(list(range(max_bs)))
        if config.enable_mtp:
            self.capture_verify_cudagraph()

    @torch.inference_mode()
    def capture_verify_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        self.verify_graphs = {}
        self.verify_graph_vars = {}

        for verify_len in self.verify_graph_lens:
            input_ids = torch.zeros(verify_len, dtype=torch.int64)
            positions = torch.arange(verify_len, dtype=torch.int64)
            slot_mapping = torch.arange(verify_len, dtype=torch.int32)
            context_lens = torch.arange(1, verify_len + 1, dtype=torch.int32)
            block_tables = torch.zeros(1, max_num_blocks, dtype=torch.int32)
            state_indices = torch.zeros(1, dtype=torch.int32) if config.is_hybrid else None
            outputs = torch.zeros(verify_len, hf_config.hidden_size)

            def run_verify_steps():
                for i in range(verify_len):
                    set_context(
                        False,
                        slot_mapping=slot_mapping[i:i + 1],
                        context_lens=context_lens[i:i + 1],
                        block_tables=block_tables,
                        state_indices=state_indices,
                    )
                    outputs[i:i + 1] = self.model(input_ids[i:i + 1], positions[i:i + 1])

            run_verify_steps()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, self.graph_pool):
                run_verify_steps()
            self.verify_graphs[verify_len] = graph
            self.verify_graph_vars[verify_len] = dict(
                input_ids=input_ids,
                positions=positions,
                slot_mapping=slot_mapping,
                context_lens=context_lens,
                block_tables=block_tables,
                outputs=outputs,
            )
            if state_indices is not None:
                self.verify_graph_vars[verify_len]["state_indices"] = state_indices
            torch.cuda.synchronize()
            reset_context()
            if config.is_hybrid:
                self.reset_gdn_state_slots([0])
