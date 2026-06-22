# SPDX-License-Identifier: Apache-2.0
"""KVarN attention backend for SGLang.

Implements compressed KV-cache attention:
  1. Hadamard rotation of Q, K, V before attention.
  2. K/V stored rotated in the pool (bf16 tail pool for in-progress blocks,
     int4 compressed cache for flushed blocks).
  3. Decode: fused kernel that dequants int4 tiles in-register and runs
     flash-decode, reading from both pools via ``block_to_slot``.
  4. Extend: standard attention on the bf16 pool (history materialization
     from int4 is a future optimization for prefill).
  5. Un-rotate the output after attention.

The flush (bf16 → int4) is triggered after each decode step when blocks
fill up. A block is a page of ``group`` tokens (e.g. 128).
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

    The backend wraps TritonAttnBackend for metadata management and the
    extend (prefill) path. For decode, it uses a fused kernel that reads
    from both the int4 compressed cache and the bf16 tail pool.

    Block = page of ``group`` tokens. When a page is fully written (all
    ``group`` tokens stored in the pool), it is flushed to int4 via the
    flush manager. The ``block_to_slot`` tensor maps block_id → tail pool
    slot (-1 if the block is compressed or not yet allocated).
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

        # Inner Triton backend for metadata management and extend path.
        self.inner = TritonAttnBackend(model_runner, skip_prefill=skip_prefill)

        # Hadamard matrix (fp32 for precision during rotation).
        self.no_rotation = os.environ.get("KVARN_NO_ROTATION", "0") == "1"
        if self.no_rotation:
            logger.info("KVarN: Hadamard rotation DISABLED (KVARN_NO_ROTATION=1)")
            self.H = None
            self.H_t = None
        else:
            self.H = build_hadamard(self.head_dim, self.device)
            self.H_t = self.H.t().contiguous()

        # Flush manager — tracks block lifecycle and performs bf16→int4 flush.
        self.flush_manager: Optional[KVarNFlushManager] = None
        self.token_to_kv_pool = model_runner.token_to_kv_pool

        # Track which blocks we've already flushed (to avoid re-flushing).
        self._flushed_block_ids: set[int] = set()

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

        pool_size = getattr(self.token_to_kv_pool, 'size', 1024)
        page_size = getattr(self.token_to_kv_pool, 'page_size', self.group)
        num_blocks = max(pool_size // page_size, 64)

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
        # Attach to pool so the fused decode kernel can access them.
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

    # ── Flush: bf16 tail pool → int4 compressed cache ──────────────────────

    def _flush_full_blocks(self, forward_batch: "ForwardBatch"):
        """Detect and flush full blocks from bf16 tail pool to int4.

        A block (page) is flushable when all ``group`` token slots in it
        have been written. We detect this by scanning the req_to_token
        table for pages where all slots are non-negative.
        """
        if self.flush_manager is None:
            return

        G = self.group
        # req_to_token_pool is on the model_runner, not the forward_batch.
        req_to_token = self.model_runner.req_to_token_pool.req_to_token
        seq_lens = forward_batch.seq_lens
        req_pool_indices = forward_batch.req_pool_indices

        # Collect block_ids to flush
        flush_block_ids = []
        for i in range(forward_batch.batch_size):
            seq_len = seq_lens[i].item()
            if seq_len < G:
                continue
            req_idx = req_pool_indices[i].item()
            # Number of complete pages for this request
            num_complete_pages = seq_len // G
            for page_idx in range(num_complete_pages):
                # Get the first slot of this page
                token_start = page_idx * G
                first_slot = req_to_token[req_idx, token_start].item()
                if first_slot < 0:
                    continue
                block_id = first_slot // G
                if block_id in self._flushed_block_ids:
                    continue
                # Verify the page is fully written (all slots non-negative)
                page_slots = req_to_token[req_idx, token_start:token_start + G]
                if (page_slots >= 0).all():
                    flush_block_ids.append(block_id)
                    self._flushed_block_ids.add(block_id)

        if not flush_block_ids:
            return

        # Flush: batched Sinkhorn + RTN → int4
        tail_k = self.token_to_kv_pool.k_buffer
        tail_v = self.token_to_kv_pool.v_buffer
        self.flush_manager.flush_blocks(flush_block_ids, tail_k, tail_v)

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
        """Decode forward: rotate Q/K/V, write to pool, flush, then attention."""
        pool_dtype = self.token_to_kv_pool.dtype
        model_dtype = q.dtype

        # Q comes in as [N, H*D] or [N, H, D]. K/V come in as [N, H, D].
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

        # Flush full blocks to int4 (after the pool write, before next step)
        self._flush_full_blocks(forward_batch)

        # Un-rotate output and convert back to model dtype
        o_3d = o.view(-1, layer.tp_q_head_num, layer.v_head_dim)
        o_unrot = self._unrotate(o_3d) if o_3d.shape[-1] == self.head_dim else o_3d
        return o_unrot.reshape(-1, layer.tp_q_head_num * layer.v_head_dim).to(model_dtype)

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

        q_input_shape = q.shape
        q_3d = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        k_3d = k.view(-1, layer.tp_k_head_num, layer.qk_head_dim)
        v_3d = v.view(-1, layer.tp_k_head_num, layer.v_head_dim)

        q_rot = self._rotate(q_3d.to(pool_dtype))
        k_rot = self._rotate(k_3d.to(pool_dtype))
        v_rot = self._rotate(v_3d.to(pool_dtype)) if v_3d.shape[-1] == self.head_dim else v_3d.to(pool_dtype)

        q_out = q_rot.reshape(q_input_shape) if q_input_shape != q_rot.shape else q_rot
        k_out = k_rot.reshape(k.shape) if k.shape != k_rot.shape else k_rot
        v_out = v_rot.reshape(v.shape) if v.shape != v_rot.shape else v_rot

        o = self.inner.forward_extend(
            q_out, k_out, v_out, layer, forward_batch, save_kv_cache=save_kv_cache, **kwargs,
        )

        # Flush full blocks to int4 after prefill
        self._flush_full_blocks(forward_batch)

        o_3d = o.view(-1, layer.tp_q_head_num, layer.v_head_dim)
        o_unrot = self._unrotate(o_3d) if o_3d.shape[-1] == self.head_dim else o_3d
        return o_unrot.reshape(-1, layer.tp_q_head_num * layer.v_head_dim).to(model_dtype)