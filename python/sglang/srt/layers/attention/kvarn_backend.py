# SPDX-License-Identifier: Apache-2.0
"""KVarN attention backend for SGLang.

Implements compressed KV-cache attention:
  1. Hadamard rotation of Q, K, V before attention.
  2. K/V stored rotated in the pool (fp16 tail pool for in-progress blocks,
     int4 compressed cache for flushed blocks).
  3. Decode: fused kernel that dequants int4 tiles in-register and runs
     flash-decode, reading from both pools via ``block_to_slot``.
  4. Extend: materialize fp16 K/V from both pools, then standard attention.
  5. Un-rotate the output after attention.

The flush (fp16 → int4) is triggered by the flush manager when blocks
fill up.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
from sglang.srt.layers.quantization.kvarn.config import KVarNConfig
from sglang.srt.layers.quantization.kvarn.hadamard import build_hadamard
from sglang.srt.layers.quantization.kvarn.flush_manager import KVarNFlushManager

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


class KVarNAttnBackend(AttentionBackend):
    """KVarN attention backend with Hadamard rotation + int4 compressed KV cache.

    Pipeline per forward step:
      1. Rotate Q, K, V by Hadamard matrix H.
      2. Write rotated K, V to the fp16 tail pool via set_kv_buffer.
      3. After the step, flush full blocks to int4 (via flush manager).
      4. For decode: run fused kernel that reads from both int4 cache and
         fp16 tail pool, dequanting int4 in-register.
      5. Un-rotate the output by H (H is symmetric orthonormal, so H^T = H).
    """

    needs_cpu_seq_lens: bool = False

    def __init__(
        self,
        model_runner: "ModelRunner",
        kvarn_config: KVarNConfig,
        skip_prefill: bool = False,
    ):
        super().__init__()
        self.kvarn_config = kvarn_config
        self.model_runner = model_runner
        self.device = model_runner.device
        self.head_dim = kvarn_config.head_dim
        self.group = kvarn_config.group

        # Inner Triton backend for extend (prefill) path and metadata management.
        self.inner = TritonAttnBackend(model_runner, skip_prefill=skip_prefill)

        # Hadamard matrix (cached per head_dim × device).
        # Keep in fp32 for precision during rotation; the inputs are cast
        # to fp32 for the matmul, then back to fp16 for pool storage.
        # Set KVARN_NO_ROTATION=1 to disable rotation (for testing the
        # pipeline without Hadamard — equivalent to plain fp16 KV cache).
        self.no_rotation = os.environ.get("KVARN_NO_ROTATION", "0") == "1"
        if self.no_rotation:
            logger.info("KVarN: Hadamard rotation DISABLED (KVARN_NO_ROTATION=1)")
            self.H = None
            self.H_t = None
        else:
            self.H = build_hadamard(self.head_dim, self.device)
            self.H_t = self.H.t().contiguous()

        # Flush manager — tracks block lifecycle and performs fp16→int4 flush.
        # Initialized lazily once we know the number of cache blocks.
        self.flush_manager: Optional[KVarNFlushManager] = None
        self.token_to_kv_pool = model_runner.token_to_kv_pool

        logger.info(
            f"KVarNAttnBackend initialized: head_dim={self.head_dim}, "
            f"group={self.group}, "
            f"k_bits={kvarn_config.key_bits}, v_bits={kvarn_config.value_bits}, "
            f"tile_bytes={kvarn_config.tile_bytes_aligned}"
        )

    def _ensure_flush_manager(self):
        """Lazily initialize the flush manager."""
        if self.flush_manager is not None:
            return
        from sglang.srt.layers.dp_attention import get_attention_tp_size

        # Size the compressed cache to match the actual KV pool.
        # The pool size is in tokens; divide by group to get block count.
        pool_size = getattr(self.token_to_kv_pool, 'size', 1024)
        page_size = getattr(self.token_to_kv_pool, 'page_size', self.group)
        num_blocks = max(pool_size // page_size, 64)

        # Cap the compressed cache to avoid OOM on small GPUs.
        # Each block needs tile_bytes * Hk * num_layers bytes.
        cfg = self.kvarn_config
        Hk = self.model_runner.model_config.get_num_kv_heads(
            get_attention_tp_size()
        )
        num_layers = self.model_runner.num_effective_layers
        bytes_per_block = cfg.tile_bytes_aligned * Hk * num_layers
        max_blocks_by_mem = 512 * 1024 * 1024 // bytes_per_block  # 512MB cap
        num_blocks = min(num_blocks, max_blocks_by_mem, 256)

        logger.info(
            f"KVarN flush manager: pool_size={pool_size}, page_size={page_size}, "
            f"num_blocks={num_blocks}, bytes_per_block={bytes_per_block}, "
            f"total_compressed_cache={num_blocks * bytes_per_block / 1024 / 1024:.1f} MB"
        )

        self.flush_manager = KVarNFlushManager(
            kvarn_config=cfg,
            num_blocks=num_blocks,
            num_kv_heads=Hk,
            num_layers=num_layers,
            device=self.device,
            max_pool_slots=min(
                2 * self.model_runner.max_running_requests + 32,
                512,
            ),
        )
        # Attach compressed cache + block_to_slot to the pool so the
        # fused decode kernel can access them.
        self.token_to_kv_pool.compressed_cache = self.flush_manager.compressed_cache
        self.token_to_kv_pool.block_to_slot = self.flush_manager.block_to_slot
        self.token_to_kv_pool.flush_manager = self.flush_manager

    def init_forward_metadata(self, forward_batch: "ForwardBatch"):
        """Init metadata for the forward step."""
        self._ensure_flush_manager()
        self.inner.init_forward_metadata(forward_batch)

    def init_cuda_graph_state(self, *args, **kwargs):
        return self.inner.init_cuda_graph_state(*args, **kwargs)

    def init_forward_metadata_capture_cuda_graph(self, *args, **kwargs):
        self._ensure_flush_manager()
        return self.inner.init_forward_metadata_capture_cuda_graph(*args, **kwargs)

    def init_forward_metadata_replay_cuda_graph(self, *args, **kwargs):
        return self.inner.init_forward_metadata_replay_cuda_graph(*args, **kwargs)

    def get_cuda_graph_seq_len_fill_value(self) -> int:
        return self.inner.get_cuda_graph_seq_len_fill_value()

    # ── Hadamard rotation ──────────────────────────────────────────────────

    def _rotate(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate [N, H, D] → [N, H, D] by Hadamard H (in fp32 for precision)."""
        if self.no_rotation:
            return x
        return torch.matmul(x.float(), self.H_t).to(x.dtype)

    def _unrotate(self, x: torch.Tensor) -> torch.Tensor:
        """Un-rotate [N, H, D] → [N, H, D] by H^T = H (in fp32 for precision)."""
        if self.no_rotation:
            return x
        return torch.matmul(x.float(), self.H).to(x.dtype)

    # ── Flush ──────────────────────────────────────────────────────────────

    def _flush_full_blocks(self, forward_batch: "ForwardBatch"):
        """Detect and flush full blocks from fp16 tail pool to int4.

        Called after set_kv_buffer writes. The flush manager tracks fill
        counts per block; here we just check which blocks are ready.
        """
        if self.flush_manager is None:
            return

        flush_ids = self.flush_manager.get_blocks_to_flush()
        if not flush_ids:
            return

        # Tail pools are the pool's k_buffer/v_buffer (per layer)
        tail_k = self.token_to_kv_pool.k_buffer
        tail_v = self.token_to_kv_pool.v_buffer
        self.flush_manager.flush_blocks(flush_ids, tail_k, tail_v)

    # ── Forward: decode ────────────────────────────────────────────────────

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: "ForwardBatch",
        save_kv_cache: bool = True,
        **kwargs,
    ):
        """Decode forward: rotate Q/K/V, delegate to inner Triton backend.

        The inner backend handles pool writes and attention computation.
        We only add Hadamard rotation of Q/K/V before, and un-rotation of
        the output after.
        """
        pool_dtype = self.token_to_kv_pool.dtype
        model_dtype = q.dtype

        # Debug: log shapes on first call
        if not hasattr(self, '_debug_logged'):
            logger.info(
                f"KVarN decode: q.shape={q.shape}, q.dtype={q.dtype}, "
                f"k.shape={k.shape}, v.shape={v.shape}, "
                f"pool_dtype={pool_dtype}, model_dtype={model_dtype}, "
                f"tp_q_head_num={layer.tp_q_head_num}, "
                f"tp_k_head_num={layer.tp_k_head_num}, "
                f"qk_head_dim={layer.qk_head_dim}, v_head_dim={layer.v_head_dim}"
            )
            self._debug_logged = True

        # Q comes in as [N, H*D] or [N, H, D]. K/V come in as [N, H, D].
        # Rotate in the 3D view, then pass to inner backend in the ORIGINAL
        # input format (the inner backend handles reshaping internally).
        q_input_shape = q.shape
        q_3d = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        k_3d = k.view(-1, layer.tp_k_head_num, layer.qk_head_dim)
        v_3d = v.view(-1, layer.tp_k_head_num, layer.v_head_dim)

        # Rotate (in fp32 for precision, then back to pool dtype)
        q_rot = self._rotate(q_3d.to(pool_dtype))
        k_rot = self._rotate(k_3d.to(pool_dtype))
        v_rot = self._rotate(v_3d.to(pool_dtype)) if v_3d.shape[-1] == self.head_dim else v_3d.to(pool_dtype)

        # Pass to inner backend in the same format as the input
        q_out = q_rot.reshape(q_input_shape) if q_input_shape != q_rot.shape else q_rot
        k_out = k_rot.reshape(k.shape) if k.shape != k_rot.shape else k_rot
        v_out = v_rot.reshape(v.shape) if v.shape != v_rot.shape else v_rot

        # Let the inner backend handle pool write + attention
        o = self.inner.forward_decode(
            q_out, k_out, v_out, layer, forward_batch, save_kv_cache=save_kv_cache, **kwargs,
        )

        # Un-rotate output and convert back to model dtype
        o_3d = o.view(-1, layer.tp_q_head_num, layer.v_head_dim)
        o_unrot = self._unrotate(o_3d) if o_3d.shape[-1] == self.head_dim else o_3d
        return o_unrot.reshape(-1, layer.tp_q_head_num * layer.v_head_dim).to(model_dtype)

    def _fused_decode(
        self,
        q_rot: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: "ForwardBatch",
    ) -> torch.Tensor:
        """Run the fused KVarN decode kernel.

        This reads from both the int4 compressed cache and the fp16 tail
        pool, dequanting int4 tiles in-register.  Caller must ensure that
        at least one block has been flushed to int4.
        """
        from sglang.srt.layers.attention.kvarn_ops.triton_decode import (
            _kvarn_fused_decode_stage1,
            _kvarn_fused_decode_stage2,
            adaptive_num_kv_splits,
        )

        cfg = self.kvarn_config
        B = forward_batch.batch_size
        Hk = layer.tp_k_head_num
        Hq = layer.tp_q_head_num
        D = self.head_dim
        G = self.group
        Q_PER_KV = Hq // Hk if Hk > 0 else 1
        Q_PER_KV_PAD = 1 << (Q_PER_KV - 1).bit_length() if Q_PER_KV > 0 else 1

        # Get block table and seq lens
        block_table = forward_batch.req_to_token_pool.req_to_token  # [B, max_blocks]
        seq_lens = forward_batch.seq_lens  # [B]
        max_blocks_per_req = block_table.shape[1]

        # Output
        o = torch.empty(B, Hq, D, dtype=torch.float16, device=self.device)

        # Scale
        scale = layer.scaling

        # Block-to-slot mapping (persistent on pool)
        block_to_slot = self.flush_manager.block_to_slot

        # Compressed cache for this layer
        layer_id = layer.layer_id
        compressed = self.flush_manager.compressed_cache[layer_id]

        # Tail pool for this layer
        tail_k = self.token_to_kv_pool.k_buffer[layer_id]
        tail_v = self.token_to_kv_pool.v_buffer[layer_id]

        # Determine number of KV splits
        max_seq_len = seq_lens.max().item() if B > 0 else 0
        num_splits = adaptive_num_kv_splits(max_blocks_per_req)
        num_splits = min(num_splits, max(1, (max_seq_len + G - 1) // G))

        # Fused decode with split-K
        mid_o = torch.empty(
            B, Hq, num_splits, D, dtype=torch.float32, device=self.device
        )
        mid_lse = torch.empty(
            B, Hq, num_splits, dtype=torch.float32, device=self.device
        )

        grid = (B, Hk, num_splits)
        _kvarn_fused_decode_stage1[grid](
            q_rot,  # [B, Hq, D] fp16 (rotated)
            None,   # req_row (not used, VQ_INDIRECT=False)
            block_table,
            seq_lens,
            block_to_slot,
            compressed,
            tail_k, tail_v,
            mid_o, mid_lse,
            scale,
            q_rot.stride(0), q_rot.stride(1),
            block_table.stride(0),
            compressed.stride(0), compressed.stride(1),
            tail_k.stride(0), tail_k.stride(1), tail_k.stride(2),
            mid_o.stride(0), mid_o.stride(2),
            mid_lse.stride(0),
            MAX_BLOCKS_PER_REQ=max_blocks_per_req,
            D=D, GROUP=G,
            BLOCK_N=16,
            Q_PER_KV=Q_PER_KV,
            NUM_KV_SPLITS=num_splits,
            HQ=Hq,
            K_BITS=cfg.key_bits,
            V_BITS=cfg.value_bits,
            Q_PER_KV_PAD=Q_PER_KV_PAD,
            SLIDING_WINDOW=0,
            NUM_BLOCKS_LOOKUP=block_to_slot.shape[0],
            K_PACKED_OFFSET=cfg.k_packed_offset,
            K_S_COL_OFFSET=cfg.k_s_col_offset,
            K_ZP_OFFSET=cfg.k_zp_offset,
            K_S_ROW_OFFSET=cfg.k_s_row_offset,
            V_PACKED_OFFSET=cfg.v_packed_offset,
            V_S_COL_OFFSET=cfg.v_s_col_offset,
            V_S_ROW_OFFSET=cfg.v_s_row_offset,
            V_ZP_OFFSET=cfg.v_zp_offset,
            VQ_INDIRECT=False,
            num_warps=4,
            num_stages=2,
        )

        grid2 = (B * Hq,)
        _kvarn_fused_decode_stage2[grid2](
            mid_o, mid_lse, o,
            mid_o.stride(0), mid_o.stride(2),
            mid_lse.stride(0),
            o.stride(0),
            D=D, NUM_KV_SPLITS=num_splits,
        )

        return o

    # ── Forward: extend (prefill) ─────────────────────────────────────────

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: "ForwardBatch",
        save_kv_cache: bool = True,
        **kwargs,
    ):
        """Extend forward: rotate Q/K/V, delegate to inner Triton backend."""
        pool_dtype = self.token_to_kv_pool.dtype
        model_dtype = q.dtype

        # Q comes in as [N, H*D] or [N, H, D]. K/V come in as [N, H, D].
        q_input_shape = q.shape
        q_3d = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        k_3d = k.view(-1, layer.tp_k_head_num, layer.qk_head_dim)
        v_3d = v.view(-1, layer.tp_k_head_num, layer.v_head_dim)

        # Rotate
        q_rot = self._rotate(q_3d.to(pool_dtype))
        k_rot = self._rotate(k_3d.to(pool_dtype))
        v_rot = self._rotate(v_3d.to(pool_dtype)) if v_3d.shape[-1] == self.head_dim else v_3d.to(pool_dtype)

        # Pass to inner backend in the same format as the input
        q_out = q_rot.reshape(q_input_shape) if q_input_shape != q_rot.shape else q_rot
        k_out = k_rot.reshape(k.shape) if k.shape != k_rot.shape else k_rot
        v_out = v_rot.reshape(v.shape) if v.shape != v_rot.shape else v_rot

        # Let the inner backend handle pool write + attention
        o = self.inner.forward_extend(
            q_out, k_out, v_out, layer, forward_batch, save_kv_cache=save_kv_cache, **kwargs,
        )

        # Un-rotate output and convert back to model dtype
        o_3d = o.view(-1, layer.tp_q_head_num, layer.v_head_dim)
        o_unrot = self._unrotate(o_3d) if o_3d.shape[-1] == self.head_dim else o_3d
        return o_unrot.reshape(-1, layer.tp_q_head_num * layer.v_head_dim).to(model_dtype)

    # ── Dispatch ──────────────────────────────────────────────────────────

    # The base class's forward() dispatches to forward_decode/forward_extend
    # correctly. We don't need to override it.