# SPDX-License-Identifier: Apache-2.0
"""KVarN tile flush manager.

Manages the lifecycle of KV cache blocks:
  - Compresses bf16 → int4 when blocks fill up (flush)
  - Dequantizes int4 → bf16 on demand (lazy dequant)

The sglang KV pool stores tokens as ``[size, head_num, head_dim]`` per layer.
A KVarN "block" is a page of ``group`` consecutive tokens. Block ``B``
occupies pool slots ``[B*group, (B+1)*group)``, each slot being
``[head_num, head_dim]``.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import torch

from sglang.srt.layers.quantization.kvarn.config import KVarNConfig
from sglang.srt.layers.quantization.kvarn.dequant import (
    kvarn_dequant_tile_k,
    kvarn_dequant_tile_v,
)
from sglang.srt.layers.quantization.kvarn.sinkhorn import variance_normalize_batched
from sglang.srt.layers.quantization.kvarn.store import (
    kvarn_store_tile_k_batch_from_sinkhorn,
    kvarn_store_tile_v_batch_from_sinkhorn,
)

logger = logging.getLogger(__name__)


class KVarNFlushManager:
    """Manages the bf16→int4 tile flush lifecycle."""

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

        # block_to_slot mapping: [num_blocks] int32
        # -1 = block is compressed (read from int4 cache)
        self.block_to_slot = torch.full(
            (num_blocks,), -1, dtype=torch.int32, device=device
        )

        self._flushed: set[int] = set()

        logger.info(
            f"KVarNFlushManager: num_blocks={num_blocks}, "
            f"group={self.group}, tile_bytes={self.tile_bytes}"
        )

    def flush_blocks(
        self,
        block_ids: List[int],
        tail_k_pools: List[torch.Tensor],  # per-layer [pool_size, Hk, D]
        tail_v_pools: List[torch.Tensor],  # per-layer [pool_size, Hk, Dv]
    ):
        """Compress bf16 pool data → int4 cache. Does NOT dequant back."""
        if not block_ids:
            return

        cfg = self.cfg
        Hk = self.num_kv_heads
        D = self.head_dim
        G = self.group
        nB = len(block_ids)

        for layer_id in range(self.num_layers):
            k_pool = tail_k_pools[layer_id]
            v_pool = tail_v_pools[layer_id]
            k_cache = self.compressed_cache[layer_id]

            all_K_tiles = []
            all_V_tiles = []
            for bid in block_ids:
                slot_start = bid * G
                k_chunk = k_pool[slot_start:slot_start + G]
                v_chunk = v_pool[slot_start:slot_start + G]
                all_K_tiles.append(k_chunk)
                all_V_tiles.append(v_chunk)

            K_batched = torch.stack(all_K_tiles).float()
            V_batched = torch.stack(all_V_tiles).float()

            K_tiles = K_batched.permute(0, 2, 3, 1).reshape(nB * Hk, D, G)
            V_tiles = V_batched.permute(0, 2, 1, 3).reshape(nB * Hk, G, D)

            K_bal, K_sc, K_sr = variance_normalize_batched(
                K_tiles, iterations=cfg.sinkhorn_iters
            )
            V_bal, V_sc, V_sr = variance_normalize_batched(
                V_tiles, iterations=cfg.sinkhorn_iters
            )

            K_out = kvarn_store_tile_k_batch_from_sinkhorn(
                K_bal, K_sc.reshape(nB * Hk, G), K_sr.reshape(nB * Hk, D),
                bits=cfg.key_bits,
            )
            V_out = kvarn_store_tile_v_batch_from_sinkhorn(
                V_bal, V_sc.reshape(nB * Hk, D), V_sr.reshape(nB * Hk, G),
                bits=cfg.value_bits,
            )

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

        for bid in block_ids:
            self.block_to_slot[bid] = -1
            self._flushed.add(bid)

        logger.info(f"KVarN flush: {nB} blocks compressed to int4")

    def dequant_blocks_to_pool(
        self,
        block_ids: List[int],
        tail_k_pools: List[torch.Tensor],
        tail_v_pools: List[torch.Tensor],
    ):
        """Dequant int4 compressed blocks back into the bf16 pool (on demand).

        Called before decode/extend when compressed blocks are needed by the
        current batch. Reads from the int4 cache and writes dequanted bf16
        data back into the pool so the standard attention kernel can read it.
        """
        if not block_ids:
            return

        cfg = self.cfg
        Hk = self.num_kv_heads
        D = self.head_dim
        G = self.group
        pack_k = 8 // cfg.key_bits
        pack_v = 8 // cfg.value_bits

        for layer_id in range(self.num_layers):
            k_pool = tail_k_pools[layer_id]
            v_pool = tail_v_pools[layer_id]
            k_cache = self.compressed_cache[layer_id]

            for bid in block_ids:
                for h in range(Hk):
                    # Dequant K tile: packed as [D, G // pack_k]
                    off = cfg.k_packed_offset
                    k_packed = k_cache[bid, h, off:off + D * (G // pack_k)].reshape(D, G // pack_k)
                    off = cfg.k_s_col_offset
                    s_col_K = k_cache[bid, h, off:off + D * 2].view(torch.float16)
                    off = cfg.k_zp_offset
                    zp_K = k_cache[bid, h, off:off + D * 2].view(torch.float16)
                    off = cfg.k_s_row_offset
                    s_row_K = k_cache[bid, h, off:off + G * 2].view(torch.float16)

                    K_deq = kvarn_dequant_tile_k(
                        k_packed, s_col_K, zp_K, s_row_K, group=G, bits=cfg.key_bits,
                    )
                    slot_start = bid * G
                    k_pool[slot_start:slot_start + G, h, :] = K_deq.t().to(k_pool.dtype)

                    # Dequant V tile: packed as [G, D // pack_v]
                    off = cfg.v_packed_offset
                    v_packed = k_cache[bid, h, off:off + G * (D // pack_v)].reshape(G, D // pack_v)
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

    def _write_packed_tile(
        self,
        cache: torch.Tensor,
        block_id: int,
        head_id: int,
        q_packed: torch.Tensor,
        s_col: torch.Tensor,
        s_row_or_zp: torch.Tensor,
        s_row: torch.Tensor,
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