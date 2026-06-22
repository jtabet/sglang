# SPDX-License-Identifier: Apache-2.0
"""KVarN tile flush manager.

Manages the lifecycle of KV cache blocks:
  - Tracks which blocks are compressed to int4 (flushed)
  - Performs batched flush (bf16 → int4) when blocks fill up

The sglang KV pool stores tokens as ``[size, head_num, head_dim]`` per layer.
A KVarN "block" is a page of ``group`` consecutive tokens. Block ``B``
occupies pool slots ``[B*group, (B+1)*group)``, each slot being
``[head_num, head_dim]``.

The flush reads ``group`` consecutive slots from the pool, reshapes them
into tiles ``[D, group]`` for K and ``[group, D]`` for V, runs Sinkhorn +
RTN, and writes the packed int4 tile to the compressed cache.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import torch

from sglang.srt.layers.quantization.kvarn.config import KVarNConfig
from sglang.srt.layers.quantization.kvarn.dequant import (
    kvarn_dequant_tile_k,
    kvarn_dequant_tile_k_batch,
    kvarn_dequant_tile_v,
    kvarn_dequant_tile_v_batch,
)
from sglang.srt.layers.quantization.kvarn.sinkhorn import variance_normalize_batched
from sglang.srt.layers.quantization.kvarn.store import (
    kvarn_store_tile_k_batch_from_sinkhorn,
    kvarn_store_tile_v_batch_from_sinkhorn,
)

logger = logging.getLogger(__name__)


class KVarNFlushManager:
    """Manages the bf16→int4 tile flush lifecycle.

    The flush manager is created once per model runner and holds:
      - The compressed uint8 cache (one per layer)
      - The block_to_slot mapping tensor (for the fused decode kernel)
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

        # Compressed cache: [num_blocks, Hk, tile_bytes] uint8 per layer
        self.compressed_cache: List[torch.Tensor] = [
            torch.zeros(
                (num_blocks, num_kv_heads, self.tile_bytes),
                dtype=torch.uint8,
                device=device,
            )
            for _ in range(num_layers)
        ]

        # block_to_slot mapping for the fused decode kernel: [num_blocks] int32
        # Value = first pool slot index for this block (>=0 means the block is
        # in the bf16 pool and not compressed; -1 means the block is compressed
        # to int4 and the kernel should read from the compressed cache).
        self.block_to_slot = torch.full(
            (num_blocks,), -1, dtype=torch.int32, device=device
        )

        # Track which blocks have been flushed to avoid re-flushing.
        self._flushed: set[int] = set()

        logger.info(
            f"KVarNFlushManager: num_blocks={num_blocks}, "
            f"group={self.group}, tile_bytes={self.tile_bytes}"
        )

    def flush_blocks(
        self,
        block_ids: List[int],
        tail_k_pools: List[torch.Tensor],  # per-layer [pool_size, Hk, D] bf16
        tail_v_pools: List[torch.Tensor],  # per-layer [pool_size, Hk, Dv] bf16
    ):
        """Flush a batch of blocks from bf16 tail pool to int4 compressed cache.

        For each block, reads ``group`` consecutive slots from the pool,
        reshapes to tiles, runs Sinkhorn + RTN, and writes packed int4 to
        the compressed cache. After flushing, sets block_to_slot[block_id]=-1
        so the decode kernel knows to read from the compressed cache.
        """
        if not block_ids:
            return

        cfg = self.cfg
        Hk = self.num_kv_heads
        D = self.head_dim
        G = self.group
        nB = len(block_ids)

        for layer_id in range(self.num_layers):
            k_pool = tail_k_pools[layer_id]  # [pool_size, Hk, D]
            v_pool = tail_v_pools[layer_id]  # [pool_size, Hk, Dv]
            k_cache = self.compressed_cache[layer_id]  # [num_blocks, Hk, tile_bytes]

            # Gather all blocks' tiles in one batched operation.
            # For each block B, the pool slots are [B*G, (B+1)*G).
            # K tile: [D, G] = [head_dim, group] — transpose of the pool layout.
            # V tile: [G, D] = [group, head_dim] — same orientation as pool.
            all_K_tiles = []
            all_V_tiles = []
            for bid in block_ids:
                slot_start = bid * G
                # K: pool[slot_start:slot_start+G, hk, :] → [G, Hk, D]
                # We need [D, G] per (block, head), so transpose.
                k_chunk = k_pool[slot_start:slot_start + G]  # [G, Hk, D]
                v_chunk = v_pool[slot_start:slot_start + G]  # [G, Hk, Dv]
                all_K_tiles.append(k_chunk)  # collect
                all_V_tiles.append(v_chunk)

            # Stack: [nB, G, Hk, D]
            K_batched = torch.stack(all_K_tiles).float()  # [nB, G, Hk, D]
            V_batched = torch.stack(all_V_tiles).float()  # [nB, G, Hk, D]

            # Reshape to tiles: [nB*Hk, D, G] for K, [nB*Hk, G, D] for V
            # K: [nB, G, Hk, D] → [nB, Hk, D, G] → [nB*Hk, D, G]
            K_tiles = K_batched.permute(0, 2, 3, 1).reshape(nB * Hk, D, G)
            # V: [nB, G, Hk, D] → [nB, Hk, G, D] → [nB*Hk, G, D]
            V_tiles = V_batched.permute(0, 2, 1, 3).reshape(nB * Hk, G, D)

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
            for i, bid in enumerate(block_ids):
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
                        k_cache, bid, h,
                        V_out["q_packed_uint8"][idx],
                        V_out["s_col_V"][idx],
                        V_out["s_row_V"][idx],
                        V_out["zp_V"][idx],
                        is_key=False,
                    )

        # Mark blocks as compressed: set block_to_slot = -1
        for bid in block_ids:
            self.block_to_slot[bid] = -1
            self._flushed.add(bid)

        # Dequant back to the bf16 pool so the standard decode kernel
        # reads the compressed-then-decompressed data. This validates the
        # full roundtrip without needing a fused Triton decode kernel.
        # Once the fused kernel is validated, this can be removed to
        # actually free the bf16 pool slots.
        self._dequant_blocks_to_pool(block_ids, tail_k_pools, tail_v_pools)

        logger.info(f"KVarN flush: {nB} blocks flushed to int4")

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
        if is_key:
            off = cfg.k_packed_offset
            cache[block_id, head_id, off:off + q_packed.numel()].copy_(q_packed.flatten())

            off = cfg.k_s_col_offset
            cache[block_id, head_id, off:off + s_col.numel() * 2].copy_(
                s_col.view(torch.uint8).flatten())

            off = cfg.k_zp_offset
            cache[block_id, head_id, off:off + s_row_or_zp.numel() * 2].copy_(
                s_row_or_zp.view(torch.uint8).flatten())

            off = cfg.k_s_row_offset
            cache[block_id, head_id, off:off + s_row.numel() * 2].copy_(
                s_row.view(torch.uint8).flatten())
        else:
            off = cfg.v_packed_offset
            cache[block_id, head_id, off:off + q_packed.numel()].copy_(q_packed.flatten())

            off = cfg.v_s_col_offset
            cache[block_id, head_id, off:off + s_col.numel() * 2].copy_(
                s_col.view(torch.uint8).flatten())

            off = cfg.v_s_row_offset
            cache[block_id, head_id, off:off + s_row_or_zp.numel() * 2].copy_(
                s_row_or_zp.view(torch.uint8).flatten())

            off = cfg.v_zp_offset
            cache[block_id, head_id, off:off + s_row.numel() * 2].copy_(
                s_row.view(torch.uint8).flatten())

    def _dequant_blocks_to_pool(
        self,
        block_ids: List[int],
        tail_k_pools: List[torch.Tensor],
        tail_v_pools: List[torch.Tensor],
    ):
        """Read int4 compressed tiles and write dequanted data back to the bf16 pool.

        This validates the full roundtrip: bf16 → int4 → dequant → bf16.
        The standard decode kernel then reads the dequanted (slightly lossy)
        data instead of the original bf16, proving the compression is
        accurate enough for coherent generation.

        Once the fused Triton decode kernel is validated, this can be
        removed to actually free the bf16 pool slots for flushed blocks.
        """
        cfg = self.cfg
        Hk = self.num_kv_heads
        D = self.head_dim
        G = self.group
        nB = len(block_ids)

        for layer_id in range(self.num_layers):
            k_pool = tail_k_pools[layer_id]
            v_pool = tail_v_pools[layer_id]
            k_cache = self.compressed_cache[layer_id]

            for i, bid in enumerate(block_ids):
                for h in range(Hk):
                    # Read packed K tile from compressed cache
                    off = cfg.k_packed_offset
                    k_packed = k_cache[bid, h, off:off + D * (G // 2)]
                    k_packed = k_packed.reshape(D, G // 2)

                    off = cfg.k_s_col_offset
                    s_col_K = k_cache[bid, h, off:off + D * 2].view(torch.float16)
                    off = cfg.k_zp_offset
                    zp_K = k_cache[bid, h, off:off + D * 2].view(torch.float16)
                    off = cfg.k_s_row_offset
                    s_row_K = k_cache[bid, h, off:off + G * 2].view(torch.float16)

                    # Dequant: [D, G] fp32
                    K_deq = kvarn_dequant_tile_k(
                        k_packed, s_col_K, zp_K, s_row_K, group=G, bits=cfg.key_bits,
                    )

                    # Write back to pool: [G, Hk, D] ← transpose [G, D] from [D, G]
                    slot_start = bid * G
                    k_pool[slot_start:slot_start + G, h, :] = K_deq.t().to(k_pool.dtype)

                    # Read packed V tile
                    off = cfg.v_packed_offset
                    v_packed = k_cache[bid, h, off:off + G * (D // 2)]
                    v_packed = v_packed.reshape(G, D // 2)

                    off = cfg.v_s_col_offset
                    s_col_V = k_cache[bid, h, off:off + D * 2].view(torch.float16)
                    off = cfg.v_s_row_offset
                    s_row_V = k_cache[bid, h, off:off + G * 2].view(torch.float16)
                    off = cfg.v_zp_offset
                    zp_V = k_cache[bid, h, off:off + G * 2].view(torch.float16)

                    V_deq = kvarn_dequant_tile_v(
                        v_packed, s_col_V, s_row_V, zp_V, head_dim=D, bits=cfg.value_bits,
                    )

                    v_pool[slot_start:slot_start + G, h, :] = V_deq.to(v_pool.dtype)