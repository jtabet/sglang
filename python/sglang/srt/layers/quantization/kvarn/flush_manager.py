# SPDX-License-Identifier: Apache-2.0
"""KVarN tile flush manager.

Manages the lifecycle of KV cache blocks:
  - Tracks which blocks are in the fp16 tail pool (in-progress)
  - Tracks which blocks are compressed to int4 (flushed)
  - Tracks which blocks are sinks (stay fp16 permanently)
  - Performs batched flush (fp16 → int4) when blocks fill up

The flush manager is called at the end of each forward step, after the
scheduler has committed the new tokens. It checks which blocks have
crossed the fill boundary and triggers batched Sinkhorn + RTN quantization.

Architecture:
  - tail_pool: [pool_slots, group, Hk, D] fp16 — per-layer K and V
  - compressed_cache: [num_blocks, Hk, tile_bytes] uint8 — per-layer
  - block_to_slot: int32 [num_blocks] → tail_pool slot index (-1 = compressed/sink)
  - block_is_sink: bool [num_blocks] → True if block is a sink (stays fp16)

In Phase 3, we add the compressed cache and flush logic. In Phase 2,
all blocks stay in the tail pool (no compression).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import torch

from sglang.srt.layers.quantization.kvarn.config import KVarNConfig
from sglang.srt.layers.quantization.kvarn.hadamard import build_hadamard
from sglang.srt.layers.quantization.kvarn.sinkhorn import variance_normalize_batched
from sglang.srt.layers.quantization.kvarn.store import (
    kvarn_store_tile_k_batch_from_sinkhorn,
    kvarn_store_tile_v_batch_from_sinkhorn,
)

logger = logging.getLogger(__name__)


@dataclass
class BlockState:
    """Per-block state tracking."""
    # Number of tokens currently written in this block (0..group)
    fill_count: int = 0
    # Tail pool slot index (-1 if not in tail pool)
    tail_slot: int = -1
    # True if this block is a sink (stays fp16 permanently)
    is_sink: bool = False
    # True if this block has been flushed to int4
    is_compressed: bool = False


class KVarNFlushManager:
    """Manages the fp16→int4 tile flush lifecycle.

    The flush manager is created once per model runner and holds:
      - The compressed uint8 cache (one per layer)
      - The block state tracking
      - The tail pool slot allocator
    """

    def __init__(
        self,
        kvarn_config: KVarNConfig,
        num_blocks: int,
        num_kv_heads: int,
        num_layers: int,
        device: str,
        max_pool_slots: int = 256,
    ):
        self.cfg = kvarn_config
        self.num_blocks = num_blocks
        self.num_kv_heads = num_kv_heads
        self.num_layers = num_layers
        self.device = device
        self.group = kvarn_config.group
        self.head_dim = kvarn_config.head_dim
        self.tile_bytes = kvarn_config.tile_bytes_aligned

        # Block state: one entry per cache block
        self.block_states: List[BlockState] = [
            BlockState() for _ in range(num_blocks)
        ]

        # Tail pool slot allocator (simple free-list)
        self.max_pool_slots = max_pool_slots
        self.free_slots: List[int] = list(range(max_pool_slots))
        self.used_slots: Set[int] = set()

        # Compressed cache: [num_blocks, Hk, tile_bytes] uint8 per layer
        # Allocated lazily on first flush
        self.compressed_cache: Optional[List[torch.Tensor]] = None

        # block_to_slot mapping for the Triton kernels: [num_blocks] int32
        # -1 means the block is compressed (not in tail pool)
        self.block_to_slot = torch.full(
            (num_blocks,), -1, dtype=torch.int32, device=device
        )

        # Allocate compressed cache eagerly so it's available when the first
        # flush happens. [num_blocks, Hk, tile_bytes] uint8 per layer.
        self.compressed_cache = [
            torch.zeros(
                (num_blocks, num_kv_heads, self.tile_bytes),
                dtype=torch.uint8,
                device=device,
            )
            for _ in range(num_layers)
        ]

        # Hadamard matrix for un-rotation during flush (if needed)
        self.H = build_hadamard(self.head_dim, torch.device(device))

        logger.info(
            f"KVarNFlushManager: num_blocks={num_blocks}, "
            f"pool_slots={max_pool_slots}, group={self.group}, "
            f"tile_bytes={self.tile_bytes}"
        )

    def allocate_tail_slot(self, block_id: int) -> int:
        """Allocate a tail pool slot for a block.

        Returns the slot index, or -1 if no slots are available.
        """
        if block_id < 0 or block_id >= self.num_blocks:
            return -1
        state = self.block_states[block_id]
        if state.tail_slot >= 0:
            return state.tail_slot  # already allocated
        if not self.free_slots:
            logger.warning(f"KVarN tail pool exhausted ({self.max_pool_slots} slots)")
            return -1
        slot = self.free_slots.pop(0)
        self.used_slots.add(slot)
        state.tail_slot = slot
        self.block_to_slot[block_id] = slot
        return slot

    def free_tail_slot(self, block_id: int):
        """Free the tail pool slot for a block (e.g. on eviction)."""
        if block_id < 0 or block_id >= self.num_blocks:
            return
        state = self.block_states[block_id]
        if state.tail_slot >= 0:
            self.free_slots.append(state.tail_slot)
            self.used_slots.discard(state.tail_slot)
            state.tail_slot = -1
            self.block_to_slot[block_id] = -1

    def mark_as_sink(self, block_id: int):
        """Mark a block as a sink (stays fp16 permanently)."""
        if 0 <= block_id < self.num_blocks:
            self.block_states[block_id].is_sink = True

    def update_fill_count(self, block_id: int, tokens_written: int):
        """Update the fill count for a block after a write."""
        if 0 <= block_id < self.num_blocks:
            state = self.block_states[block_id]
            state.fill_count = min(
                state.fill_count + tokens_written, self.group
            )

    def get_blocks_to_flush(self) -> List[int]:
        """Return block IDs that are full and ready to be flushed to int4.

        A block is flushable if:
          - fill_count >= group (fully filled)
          - is_sink == False (sinks stay fp16)
          - is_compressed == False (not already flushed)
          - tail_slot >= 0 (has data in the tail pool)
        """
        return [
            bid for bid, s in enumerate(self.block_states)
            if s.fill_count >= self.group
            and not s.is_sink
            and not s.is_compressed
            and s.tail_slot >= 0
        ]

    def flush_blocks(
        self,
        block_ids: List[int],
        tail_k_pools: List[torch.Tensor],  # per-layer [pool_slots, G, Hk, D] fp16
        tail_v_pools: List[torch.Tensor],
    ):
        """Flush a batch of blocks from fp16 tail pool to int4 compressed cache.

        Args:
            block_ids: List of block IDs to flush.
            tail_k_pools: Per-layer K tail pool tensors.
            tail_v_pools: Per-layer V tail pool tensors.
        """
        if not block_ids:
            return

        cfg = self.cfg
        Hk = self.num_kv_heads
        D = self.head_dim
        G = self.group

        for layer_id in range(self.num_layers):
            k_pool = tail_k_pools[layer_id]
            v_pool = tail_v_pools[layer_id]
            k_cache = self.compressed_cache[layer_id]
            v_cache = self.compressed_cache[layer_id]  # same tensor, different offsets

            # Process in chunks to bound Sinkhorn launch size
            CHUNK = max(1, 2048 // max(Hk, 1))
            for c0 in range(0, len(block_ids), CHUNK):
                chunk = block_ids[c0:c0 + CHUNK]
                nB = len(chunk)
                slots = torch.tensor(
                    [self.block_states[bid].tail_slot for bid in chunk],
                    dtype=torch.long, device=self.device,
                )

                # Gather from tail pool: [nB, G, Hk, D]
                K_rot = k_pool.index_select(0, slots).float()
                V_rot = v_pool.index_select(0, slots).float()

                # Reshape to tiles: [nB*Hk, D, G] for K, [nB*Hk, G, D] for V
                K_tiles = K_rot.permute(0, 2, 3, 1).reshape(nB * Hk, D, G)
                V_tiles = V_rot.permute(0, 2, 1, 3).reshape(nB * Hk, G, D)

                # Sinkhorn (batched)
                K_bal, K_sc, K_sr = variance_normalize_batched(
                    K_tiles, iterations=cfg.sinkhorn_iters
                )
                V_bal, V_sc, V_sr = variance_normalize_batched(
                    V_tiles, iterations=cfg.sinkhorn_iters
                )

                # RTN + pack
                K_out = kvarn_store_tile_k_batch_from_sinkhorn(
                    K_bal, K_sc.reshape(nB * Hk, G), K_sr.reshape(nB * Hk, D),
                    bits=cfg.key_bits,
                )
                V_out = kvarn_store_tile_v_batch_from_sinkhorn(
                    V_bal, V_sc.reshape(nB * Hk, D), V_sr.reshape(nB * Hk, G),
                    bits=cfg.value_bits,
                )

                # Write packed tiles to compressed cache
                for i, bid in enumerate(chunk):
                    for h in range(Hk):
                        idx = i * Hk + h
                        self._write_packed_tile(
                            k_cache, bid, h,
                            K_out["q_packed_uint8"][idx],
                            K_out["s_col_K"][idx],
                            K_out["zp_K"][idx],
                            K_out["s_row_K"][idx],
                            is_key=True,
                        )
                        self._write_packed_tile(
                            v_cache, bid, h,
                            V_out["q_packed_uint8"][idx],
                            V_out["s_col_V"][idx],
                            V_out["s_row_V"][idx],
                            V_out["zp_V"][idx],
                            is_key=False,
                        )

        # Mark blocks as compressed and free tail slots
        for bid in block_ids:
            state = self.block_states[bid]
            state.is_compressed = True
            self.free_tail_slot(bid)

        logger.debug(
            f"KVarN flush: {len(block_ids)} blocks flushed to int4"
        )

    def _write_packed_tile(
        self,
        cache: torch.Tensor,  # [num_blocks, Hk, tile_bytes] uint8
        block_id: int,
        head_id: int,
        q_packed: torch.Tensor,   # uint8
        s_col: torch.Tensor,      # fp16
        s_row_or_zp: torch.Tensor, # fp16
        s_row: torch.Tensor,      # fp16
        is_key: bool = True,
    ):
        """Write one packed tile to the compressed cache."""
        cfg = self.cfg
        base = block_id * cfg.tile_bytes_aligned + head_id * cfg.tile_bytes_aligned

        if is_key:
            off = cfg.k_packed_offset
            cache[block_id, head_id, off:off + q_packed.numel()].copy_(q_packed.flatten())

            off = cfg.k_s_col_offset
            s_col_bytes = s_col.view(torch.uint8).flatten()
            cache[block_id, head_id, off:off + s_col_bytes.numel()].copy_(s_col_bytes)

            off = cfg.k_zp_offset
            zp_bytes = s_row_or_zp.view(torch.uint8).flatten()
            cache[block_id, head_id, off:off + zp_bytes.numel()].copy_(zp_bytes)

            off = cfg.k_s_row_offset
            sr_bytes = s_row.view(torch.uint8).flatten()
            cache[block_id, head_id, off:off + sr_bytes.numel()].copy_(sr_bytes)
        else:
            off = cfg.v_packed_offset
            cache[block_id, head_id, off:off + q_packed.numel()].copy_(q_packed.flatten())

            off = cfg.v_s_col_offset
            s_col_bytes = s_col.view(torch.uint8).flatten()
            cache[block_id, head_id, off:off + s_col_bytes.numel()].copy_(s_col_bytes)

            off = cfg.v_s_row_offset
            s_row_bytes = s_row_or_zp.view(torch.uint8).flatten()
            cache[block_id, head_id, off:off + s_row_bytes.numel()].copy_(s_row_bytes)

            off = cfg.v_zp_offset
            zp_bytes = s_row.view(torch.uint8).flatten()
            cache[block_id, head_id, off:off + zp_bytes.numel()].copy_(zp_bytes)

    def reset_block(self, block_id: int):
        """Reset a block's state (e.g. on eviction from the cache)."""
        if 0 <= block_id < self.num_blocks:
            self.free_tail_slot(block_id)
            state = self.block_states[block_id]
            state.fill_count = 0
            state.is_sink = False
            state.is_compressed = False

    def get_block_to_slot_tensor(self) -> torch.Tensor:
        """Return the block_to_slot mapping tensor for Triton kernels."""
        return self.block_to_slot