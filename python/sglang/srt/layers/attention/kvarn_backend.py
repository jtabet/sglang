# SPDX-License-Identifier: Apache-2.0
"""KVarN attention backend for SGLang.

Implements compressed KV-cache attention with a dual-pool architecture
matching the vLLM KVarN reference:

  1. **Hadamard rotation** of Q, K, V before attention and storage.
  2. **Tail pool** (fp16, small): stores ROTATED K/V for in-progress blocks
     and sink blocks. Each block occupies one tail-pool slot:
     ``[pool_slots, group, num_kv_heads, head_dim]``.
  3. **Compressed cache** (int4, uint8): stores flushed (fully-written) blocks
     as quantized tiles: ``[num_blocks, num_kv_heads, tile_bytes]`` per layer.
  4. **Block-to-slot mapping**: translates scheduler page block_ids to
     tail-pool slots. Flushed blocks have their slot freed and live in int4.
  5. **Decode/Extend**: gather K/V from both sources (int4 dequant + fp16 tail
     pool), un-rotate, then run SDPA per request. This is the "slow path"
     (materialize + SDPA) — the fused Triton kernel is future work.
  6. **Un-rotate** the output after attention.

The scheduler sees ``max_total_num_tokens = num_blocks * page_size`` — the
compressed cache capacity — while the actual GPU memory is dominated by the
small fp16 tail pool + the int4 compressed cache, yielding the 3.5×
compression ratio. The standard ``token_to_kv_pool`` is a NoOp pool (large
logical size, tiny physical allocation) used only for the scheduler's
capacity accounting; the real K/V storage lives in the KVarN backend's
tail pool and compressed cache.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn.functional as F

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.quantization.kvarn.config import KVarNConfig
from sglang.srt.layers.quantization.kvarn.hadamard import build_hadamard
from sglang.srt.layers.quantization.kvarn.flush_manager import KVarNFlushManager

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)

# Number of sink blocks (first pages) kept in fp16 tail pool, never flushed.
# Matches the vLLM reference default (1 sink block = first page).
KVaRN_SINK_BLOCKS = 1


class KVarNAttnBackend(AttentionBackend):
    """KVarN attention backend with dual-pool (tail + int4) architecture."""

    needs_cpu_seq_lens: bool = False

    def __init__(self, model_runner: "ModelRunner", kvarn_config: Optional[KVarNConfig] = None):
        super().__init__()

        self.model_runner = model_runner
        self.server_args = model_runner.server_args
        self.device = model_runner.device

        # Get KVarN config
        if kvarn_config is not None:
            kvarn_cfg = kvarn_config
        else:
            kvarn_cfg = getattr(model_runner, "kvarn_config", None)
            if kvarn_cfg is None:
                quant_config = getattr(model_runner, "quant_config", None)
                if quant_config is not None:
                    kvarn_cfg = getattr(quant_config, "kvarn_config", None)

        if kvarn_cfg is None:
            raise RuntimeError(
                "KVarNAttnBackend requires a KVarNConfig. "
                "Set --quantization kvarn-k4v4g128 or similar."
            )
        self.cfg: KVarNConfig = kvarn_cfg
        self.group = self.cfg.group

        # Model config
        model_config = model_runner.model_config
        self.num_heads = model_config.get_total_num_attention_heads() // model_runner.tp_size
        self.num_kv_heads = model_config.get_num_kv_heads(model_runner.tp_size)
        self.head_dim = model_config.head_dim
        self.v_head_dim = model_config.v_head_dim
        if self.v_head_dim is None:
            self.v_head_dim = self.head_dim
        self.scale = float(model_config.scaling) if hasattr(model_config, "scaling") else 1.0 / (self.head_dim ** 0.5)
        self.num_layers = model_config.num_attention_layers

        # Hadamard rotation matrix (shared across all layers)
        self._H: Optional[torch.Tensor] = None  # [D, D] fp32, lazy

        # Tail pool: per-layer fp16 buffers
        # Allocated in _init_pools, called from init_forward_metadata
        self.tail_K: list[torch.Tensor] = []  # [num_layers][pool_slots, group, Hk, D] fp16
        self.tail_V: list[torch.Tensor] = []
        self.pool_slots = 0

        # Compressed cache: per-layer uint8 buffers
        self.kv_cache_int4: list[torch.Tensor] = []  # [num_layers][num_blocks, Hk, tile_bytes] uint8
        self.num_blocks = 0

        # Block-to-slot mapping
        self._block_to_slot: dict[int, int] = {}  # block_id -> tail pool slot
        self._slot_to_block: dict[int, int] = {}  # tail pool slot -> block_id
        self._free_slots: list[int] = []
        self._block_fill: dict[int, int] = {}  # block_id -> token count in tail pool

        # Sink blocks: block_ids that stay in fp16, never flushed
        self._sink_block_ids: set[int] = set()

        # Flush manager
        self.flush_manager = KVarNFlushManager(
            cfg=self.cfg,
            num_layers=self.num_layers,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            v_head_dim=self.v_head_dim,
            sink_blocks=KVaRN_SINK_BLOCKS,
        )

        # Page size (= group)
        self.page_size = model_runner.page_size

        logger.info(
            f"KVarNAttnBackend: group={self.group}, num_kv_heads={self.num_kv_heads}, "
            f"head_dim={self.head_dim}, num_layers={self.num_layers}, "
            f"pool_slots={self.pool_slots} (allocated later), "
            f"page_size={self.page_size}"
        )

    def _get_hadamard(self, device: torch.device) -> torch.Tensor:
        """Get or build the Hadamard rotation matrix."""
        if self._H is None or self._H.device != device:
            self._H = build_hadamard(self.head_dim, device)
        return self._H

    def _init_pools(self):
        """Allocate tail pool and compressed cache tensors."""
        if self.pool_slots > 0:
            return  # Already initialized

        mr = self.model_runner
        max_running = mr.server_args.max_running_requests or 256
        max_prefill_tokens = mr.server_args.max_prefill_tokens or 16384

        # Tail pool size: 2 * max_running (sink + in-progress tail per request)
        # + prefill_blocks + headroom
        prefill_blocks = (max_prefill_tokens + self.group - 1) // self.group
        self.pool_slots = max(2 * max_running + prefill_blocks + 8, 8)

        # Compressed cache: one slot per scheduler page
        self.num_blocks = mr.max_total_num_tokens // self.page_size

        device = self.device

        # Allocate per-layer tail pool buffers
        self.tail_K = []
        self.tail_V = []
        for _ in range(self.num_layers):
            self.tail_K.append(torch.zeros(
                self.pool_slots, self.group, self.num_kv_heads, self.head_dim,
                dtype=torch.float16, device=device,
            ))
            self.tail_V.append(torch.zeros(
                self.pool_slots, self.group, self.num_kv_heads, self.v_head_dim,
                dtype=torch.float16, device=device,
            ))

        # Allocate per-layer compressed cache buffers
        self.kv_cache_int4 = []
        tile_bytes = self.cfg.tile_bytes_aligned
        for _ in range(self.num_layers):
            self.kv_cache_int4.append(torch.zeros(
                self.num_blocks, self.num_kv_heads, tile_bytes,
                dtype=torch.uint8, device=device,
            ))

        # Initialize free slots
        self._free_slots = list(range(self.pool_slots))

        logger.info(
            f"KVarN pools allocated: tail_pool_slots={self.pool_slots}, "
            f"compressed_blocks={self.num_blocks}, "
            f"tail_pool_bytes={self.pool_slots * self.group * self.num_kv_heads * self.head_dim * 2 * self.num_layers / 1e9:.2f} GB, "
            f"compressed_cache_bytes={self.num_blocks * tile_bytes * self.num_kv_heads * self.num_layers / 1e9:.2f} GB"
        )

    def _alloc_slot(self, block_id: int) -> int:
        """Allocate a tail pool slot for a block."""
        if not self._free_slots:
            raise RuntimeError("KVarN tail pool exhausted — no free slots")
        slot = self._free_slots.pop()
        self._block_to_slot[block_id] = slot
        self._slot_to_block[slot] = block_id
        self._block_fill[block_id] = 0
        return slot

    def _free_slot(self, block_id: int):
        """Free a tail pool slot (block has been flushed to int4)."""
        slot = self._block_to_slot.pop(block_id, None)
        if slot is not None:
            self._slot_to_block.pop(slot, None)
            self._block_fill.pop(block_id, None)
            self._free_slots.append(slot)

    def get_slot_for_block(self, block_id: int) -> Optional[int]:
        """Get the tail pool slot for a block, or None if flushed."""
        return self._block_to_slot.get(block_id)

    def get_kv_cache_int4(self, layer_id: int) -> torch.Tensor:
        """Get the int4 compressed cache for a layer."""
        return self.kv_cache_int4[layer_id]

    def get_tail_pool(self, layer_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the fp16 tail pool K/V for a layer."""
        return self.tail_K[layer_id], self.tail_V[layer_id]

    # ── AttentionBackend interface ─────────────────────────────────────────

    def init_forward_metadata(self, forward_batch: "ForwardBatch"):
        """Eager entry point. Initialize pools and flush blocks."""
        if self.pool_slots == 0:
            self._init_pools()

        # Flush any full blocks from the tail pool to int4 cache.
        # This runs outside the captured region (eager).
        self._maybe_flush_blocks(forward_batch)

    def init_forward_metadata_out_graph(
        self, forward_batch: "ForwardBatch", in_capture: bool = False,
    ):
        """Per-iter metadata prep. No-op for eager-only backend."""
        if self.pool_slots == 0:
            self._init_pools()
        if not in_capture:
            self._maybe_flush_blocks(forward_batch)

    def init_forward_metadata_in_graph(self, forward_batch: "ForwardBatch"):
        """Graph-recordable static-shape GPU op. No-op for eager-only backend."""
        pass

    def init_cuda_graph_state(self, max_batch_size: int, max_num_token: int):
        """Pre-allocate CUDA graph state. No-op for now (eager-only)."""
        if self.pool_slots == 0:
            self._init_pools()

    def get_cuda_graph_seq_len_fill_value(self) -> int:
        return 1

    def clear(self):
        """Clear all mappings. Called when the scheduler resets."""
        self._block_to_slot.clear()
        self._slot_to_block.clear()
        self._block_fill.clear()
        self._free_slots = list(range(self.pool_slots))
        self._sink_block_ids.clear()

    # ── Forward methods ─────────────────────────────────────────────────────

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: "ForwardBatch",
        save_kv_cache: bool = True,
        sinks=None,
    ) -> torch.Tensor:
        """Decode forward: one token per request."""
        layer_id = layer.layer_id
        N = q.shape[0]  # num decode tokens (= batch_size for pure decode)
        H = self._get_hadamard(self.device)

        q_3d = q.view(N, self.num_heads, self.head_dim)
        k_3d = k.view(N, self.num_kv_heads, self.head_dim)
        v_3d = v.view(N, self.num_kv_heads, self.v_head_dim)

        # Rotate K/V and store to tail pool
        if save_kv_cache:
            self._store_to_tail_pool(layer_id, k_3d, v_3d, forward_batch.out_cache_loc)

        # Rotate Q
        q_rot = self._rotate(q_3d, H)

        # Gather K/V per request and run SDPA
        out = torch.empty(N, self.num_heads, self.head_dim, dtype=q.dtype, device=q.device)

        req_pool_indices = forward_batch.req_pool_indices
        seq_lens = forward_batch.seq_lens
        req_to_token = self.model_runner.req_to_token_pool.req_to_token

        for i in range(N):
            req_idx = int(req_pool_indices[i].item())
            seq_len = int(seq_lens[i].item())

            # Build block table for this request
            block_ids = self._get_block_ids_for_request(req_idx, seq_len, req_to_token)

            # Gather K/V from tail pool + int4 cache
            K_full, V_full = self._gather_request_kv(layer_id, block_ids, seq_len)

            # Run SDPA for this single query token
            q_i = q_rot[i:i+1].transpose(0, 1).unsqueeze(0).float()  # [1, Hq, 1, D]
            K_t = K_full.transpose(0, 1).unsqueeze(0).float()  # [1, Hk, seq_len, D]
            V_t = V_full.transpose(0, 1).unsqueeze(0).float()  # [1, Hk, seq_len, D]

            o = F.scaled_dot_product_attention(
                q_i, K_t, V_t, is_causal=False, scale=self.scale,
                enable_gqa=self.num_kv_heads < self.num_heads,
            )
            # Un-rotate
            o = o[0, :, 0, :].to(q.dtype)  # [Hq, D]
            out[i] = (o.float() @ H.T).to(q.dtype)

        return out.view(N, self.num_heads * self.head_dim)

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: "ForwardBatch",
        save_kv_cache: bool = True,
        sinks=None,
    ) -> torch.Tensor:
        """Extend forward: multiple tokens per request (prefill/chunked-prefill).

        Handles both first-chunk prefill (prefix_len=0) and extend
        (prefix_len>0) and target_verify (ext_len=1, prefix=seq-1).
        """
        layer_id = layer.layer_id
        N = q.shape[0]  # total num tokens in batch
        H = self._get_hadamard(self.device)

        q_3d = q.view(N, self.num_heads, self.head_dim)
        k_3d = k.view(N, self.num_kv_heads, self.head_dim)
        v_3d = v.view(N, self.num_kv_heads, self.v_head_dim)

        # Rotate K/V and store to tail pool
        if save_kv_cache:
            self._store_to_tail_pool(layer_id, k_3d, v_3d, forward_batch.out_cache_loc)

        # Rotate Q
        q_rot = self._rotate(q_3d, H)

        extend_seq_lens = forward_batch.extend_seq_lens
        extend_prefix_lens = forward_batch.extend_prefix_lens

        out = torch.empty(N, self.num_heads, self.head_dim, dtype=q.dtype, device=q.device)

        req_pool_indices = forward_batch.req_pool_indices
        req_to_token = self.model_runner.req_to_token_pool.req_to_token

        if extend_seq_lens is not None:
            # Extend mode: each request has prefix (cached) + new tokens
            token_offset = 0
            for i in range(extend_seq_lens.shape[0]):
                req_idx = int(req_pool_indices[i].item())
                ext_len = int(extend_seq_lens[i].item())
                prefix_len = int(extend_prefix_lens[i].item()) if extend_prefix_lens is not None else 0
                seq_len = prefix_len + ext_len

                q_start = token_offset
                q_end = token_offset + ext_len
                token_offset = q_end

                if ext_len <= 0:
                    continue

                # Build block table for this request (full seq_len)
                block_ids = self._get_block_ids_for_request(req_idx, seq_len, req_to_token)

                # Gather K/V for the full sequence (in rotated frame)
                K_full, V_full = self._gather_request_kv(layer_id, block_ids, seq_len)

                # Run SDPA: query attends to all tokens (prefix + new), causal
                q_i = q_rot[q_start:q_end].transpose(0, 1).unsqueeze(0).float()  # [1, Hq, ext_len, D]
                K_t = K_full.transpose(0, 1).unsqueeze(0).float()  # [1, Hk, seq_len, D]
                V_t = V_full.transpose(0, 1).unsqueeze(0).float()  # [1, Hk, seq_len, D]

                if prefix_len == 0:
                    # First-chunk prefill: standard causal attention
                    o = F.scaled_dot_product_attention(
                        q_i, K_t, V_t, is_causal=True, scale=self.scale,
                        enable_gqa=self.num_kv_heads < self.num_heads,
                    )
                else:
                    # Extend: query at position prefix_len+j attends to keys 0..prefix_len+j
                    q_len = ext_len
                    q_pos = torch.arange(q_len, device=q.device).unsqueeze(1) + prefix_len  # [q_len, 1]
                    k_pos = torch.arange(seq_len, device=q.device).unsqueeze(0)  # [1, seq_len]
                    mask = k_pos <= q_pos  # [q_len, seq_len]

                    o = F.scaled_dot_product_attention(
                        q_i, K_t, V_t, attn_mask=mask, scale=self.scale,
                        enable_gqa=self.num_kv_heads < self.num_heads,
                    )

                o = o[0].transpose(0, 1)  # [ext_len, Hq, D]
                out[q_start:q_end] = (o.float() @ H.T).to(q.dtype)
        else:
            # Fallback: no extend_seq_lens — shouldn't happen but handle gracefully
            # Treat as pure prefill using seq_lens
            seq_lens = forward_batch.seq_lens
            token_offset = 0
            for i in range(seq_lens.shape[0]):
                req_idx = int(req_pool_indices[i].item())
                seq_len = int(seq_lens[i].item())

                q_start = token_offset
                q_end = token_offset + seq_len
                token_offset = q_end

                if seq_len <= 0:
                    continue

                block_ids = self._get_block_ids_for_request(req_idx, seq_len, req_to_token)
                K_full, V_full = self._gather_request_kv(layer_id, block_ids, seq_len)

                q_i = q_rot[q_start:q_end].transpose(0, 1).unsqueeze(0).float()
                K_t = K_full.transpose(0, 1).unsqueeze(0).float()
                V_t = V_full.transpose(0, 1).unsqueeze(0).float()

                o = F.scaled_dot_product_attention(
                    q_i, K_t, V_t, is_causal=True, scale=self.scale,
                    enable_gqa=self.num_kv_heads < self.num_heads,
                )
                o = o[0].transpose(0, 1)
                out[q_start:q_end] = (o.float() @ H.T).to(q.dtype)

        return out.view(N, self.num_heads * self.head_dim)

    # ── Store path ──────────────────────────────────────────────────────────

    def _store_to_tail_pool(
        self,
        layer_id: int,
        k: torch.Tensor,  # [N, Hk, D] bf16/fp16
        v: torch.Tensor,  # [N, Hk, vD] bf16/fp16
        out_cache_loc: torch.Tensor,  # [N] slot indices
    ):
        """Rotate K/V and store to tail pool using block-to-slot mapping.

        Vectorized: allocates tail pool slots for new blocks, then scatters
        K/V to the tail pool using index_put_.
        """
        H = self._get_hadamard(self.device)

        # Rotate K/V in fp32 for precision, cast to fp16 for storage
        k_rot = (k.float() @ H).to(torch.float16)  # [N, Hk, D]
        v_rot = (v.float() @ H).to(torch.float16)  # [N, Hk, vD]

        # Compute block_id and pos_in_block for each token
        loc = out_cache_loc  # [N] on GPU
        block_ids = loc // self.page_size  # [N]
        pos_in_block = loc % self.page_size  # [N]

        # Find new blocks that need tail pool slots allocated
        unique_blocks = torch.unique(block_ids)
        new_blocks = [int(b.item()) for b in unique_blocks if int(b.item()) not in self._block_to_slot]

        for b in new_blocks:
            self._alloc_slot(b)
            if b < KVaRN_SINK_BLOCKS:
                self._sink_block_ids.add(b)

        # Build tail pool slot indices for each token
        # We need a GPU tensor mapping block_id -> tail_pool_slot
        # For now, build a CPU mapping and transfer
        tail_slots = torch.empty_like(loc, dtype=torch.long)
        for i in range(loc.shape[0]):
            tail_slots[i] = self._block_to_slot[int(block_ids[i].item())]

        # Scatter to tail pool: tail_K[layer_id][tail_slots, pos_in_block] = k_rot
        # This is a scatter along two dimensions: slot and pos_in_block
        # Use index_put_ with linear index: slot * group + pos_in_block
        linear_idx = tail_slots * self.group + pos_in_block

        K_flat = self.tail_K[layer_id].view(
            self.pool_slots * self.group, self.num_kv_heads, self.head_dim
        )
        V_flat = self.tail_V[layer_id].view(
            self.pool_slots * self.group, self.num_kv_heads, self.v_head_dim
        )
        K_flat.index_put_(
            (linear_idx,), k_rot, accumulate=False,
        )
        V_flat.index_put_(
            (linear_idx,), v_rot, accumulate=False,
        )

        # Update block fill counts (CPU)
        for b in unique_blocks:
            bid = int(b.item())
            count = int((block_ids == b).sum().item())
            self._block_fill[bid] = self._block_fill.get(bid, 0) + count

    # ── Gather path ─────────────────────────────────────────────────────────

    def _get_block_ids_for_request(
        self,
        req_idx: int,
        seq_len: int,
        req_to_token: torch.Tensor,
    ) -> list[int]:
        """Get the list of block_ids for a request's sequence.

        Uses req_to_token_pool to find the first slot of each page,
        then computes block_id = slot // page_size.
        """
        num_blocks = (seq_len + self.page_size - 1) // self.page_size
        block_ids = []
        for b in range(num_blocks):
            token_pos = b * self.page_size
            slot = int(req_to_token[req_idx, token_pos].item())
            if slot < 0:
                # Unallocated position — use a placeholder block_id
                # This shouldn't happen for a valid seq_len
                block_ids.append(-1)
            else:
                block_id = slot // self.page_size
                block_ids.append(block_id)
        return block_ids

    def _gather_request_kv(
        self,
        layer_id: int,
        block_ids: list[int],
        seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather full K/V for a request from tail pool (fp16) + int4 cache.

        Returns (K [seq_len, Hk, D] fp16, V [seq_len, Hk, vD] fp16) in the
        ROTATED frame (Hadamard rotation applied, matching Q which is also
        rotated). Attention is computed in the rotated frame; the output
        is un-rotated by the caller.
        """
        group = self.group
        n_full = seq_len // group
        tail_len = seq_len % group
        D = self.head_dim
        vD = self.v_head_dim
        device = self.device

        K_parts: list[torch.Tensor] = []
        V_parts: list[torch.Tensor] = []

        for i in range(n_full):
            block_id = block_ids[i]
            slot = self._block_to_slot.get(block_id)
            if slot is not None:
                # Block is in tail pool (in-progress or sink) — already rotated
                K_parts.append(self.tail_K[layer_id][slot])
                V_parts.append(self.tail_V[layer_id][slot])
            else:
                # Block is flushed to int4 cache — dequant returns rotated frame
                K_blk, V_blk = self._read_block_dequantized(layer_id, block_id)
                K_parts.append(K_blk)
                V_parts.append(V_blk)

        if tail_len > 0:
            block_id = block_ids[n_full]
            slot = self._block_to_slot.get(block_id)
            if slot is not None:
                K_parts.append(self.tail_K[layer_id][slot, :tail_len])
                V_parts.append(self.tail_V[layer_id][slot, :tail_len])
            else:
                K_parts.append(torch.zeros(
                    tail_len, self.num_kv_heads, D,
                    dtype=torch.float16, device=device,
                ))
                V_parts.append(torch.zeros(
                    tail_len, self.num_kv_heads, vD,
                    dtype=torch.float16, device=device,
                ))

        K = torch.cat(K_parts, dim=0) if K_parts else torch.empty(
            0, self.num_kv_heads, D, dtype=torch.float16, device=device,
        )
        V = torch.cat(V_parts, dim=0) if V_parts else torch.empty_like(K)
        return K, V

    def _read_block_dequantized(
        self, layer_id: int, block_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Read a block from the int4 compressed cache and dequantize to fp16.

        Returns (K [group, Hk, D] fp16, V [group, Hk, vD] fp16) in the
        ROTATED frame (as stored in the compressed cache — no un-rotation).
        The caller uses this in the rotated-frame attention computation.
        """
        K_blk, V_blk = self.flush_manager.dequant_block(
            block_id, self.kv_cache_int4, layer_id,
        )
        return K_blk, V_blk

    # ── Flush path ──────────────────────────────────────────────────────────

    def _maybe_flush_blocks(self, forward_batch: "ForwardBatch"):
        """Flush full blocks from tail pool to int4 compressed cache.

        A block is "full" when it has `group` tokens stored and is not a sink.
        This runs eagerly before each forward pass.
        """
        blocks_to_flush = []
        for block_id, fill in list(self._block_fill.items()):
            if fill >= self.group and block_id not in self._sink_block_ids:
                blocks_to_flush.append(block_id)

        for block_id in blocks_to_flush:
            self._flush_block(block_id)

    def _flush_block(self, block_id: int):
        """Compress a block from tail pool to int4 cache and free the slot."""
        slot = self._block_to_slot.get(block_id)
        if slot is None:
            return  # Already flushed

        self.flush_manager.flush_block(
            block_id=block_id,
            tail_K=self.tail_K,
            tail_V=self.tail_V,
            slot=slot,
            compressed_cache=self.kv_cache_int4,
        )

        # Free the tail pool slot
        self._free_slot(block_id)
        logger.debug(f"Flushed block {block_id} to int4 cache")

    # ── Rotation helpers ────────────────────────────────────────────────────

    def _rotate(self, x: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        """Apply Hadamard rotation: x_rot = x @ H"""
        return (x.float() @ H).to(x.dtype)

    def _unrotate(self, x: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        """Apply inverse Hadamard rotation: x = x_rot @ H^T"""
        return (x.float() @ H.T).to(x.dtype)

    # ── Compatibility stubs ─────────────────────────────────────────────────

    def get_forward_metadata(self):
        return None

    def get_kv_cache_shape(self):
        """Return the KV cache shape for this backend."""
        # Used by model_runner to size the pool. For KVarN, the pool is NoOp
        # and the real storage is in the backend's tail pool + int4 cache.
        return self.num_blocks, self.num_kv_heads, self.head_dim

    def get_q_scale(self):
        return self.scale

    def set_kv_buffer(self, layer, loc, k, v, *args, **kwargs):
        """Direct set_kv_buffer — used by some codepaths. Redirect to tail pool."""
        self._store_to_tail_pool(layer.layer_id, k, v, loc)

    def get_kv_buffer(self, layer_id: int):
        """Get KV buffer — returns tail pool for this layer."""
        return self.tail_K[layer_id], self.tail_V[layer_id]