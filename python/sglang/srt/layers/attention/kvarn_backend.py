# SPDX-License-Identifier: Apache-2.0
"""KVarN attention backend for SGLang.

Implements compressed KV-cache attention:
  1. Hadamard rotation of Q, K, V before attention.
  2. K/V stored rotated in the pool (bf16 tail pool for in-progress blocks,
     int4 compressed cache for flushed blocks).
  3. Decode: materialize K/V from both sources (int4 dequant + bf16 pool)
     into a temporary buffer, then run standard attention.
  4. Extend: same materialization approach.
  5. Un-rotate the output after attention.

Memory layout:
  - bf16 pool: [pool_size, Hk, D] — holds ALL tokens (both in-progress
    and dequanted-from-int4). The standard Triton backend reads from this.
  - int4 cache: [num_blocks, Hk, tile_bytes] — compressed tiles.

The pool is the full size (same as standard MHA), but flushed blocks' pool
slots are dequanted from int4 (slightly lossy) rather than the original
bf16. This validates the compression roundtrip end-to-end. Actual memory
savings require shrinking the pool to only hold in-progress blocks, which
needs the fused decode kernel (future work).
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
    """KVarN attention backend with Hadamard rotation + int4 compressed KV cache."""

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

        self.inner = TritonAttnBackend(model_runner, skip_prefill=skip_prefill)

        self.no_rotation = os.environ.get("KVARN_NO_ROTATION", "0") == "1"
        if self.no_rotation:
            logger.info("KVarN: Hadamard rotation DISABLED (KVARN_NO_ROTATION=1)")
            self.H = None
            self.H_t = None
        else:
            self.H = build_hadamard(self.head_dim, self.device)
            self.H_t = self.H.t().contiguous()

        self.flush_manager: Optional[KVarNFlushManager] = None
        self.token_to_kv_pool = model_runner.token_to_kv_pool
        self._flushed_block_ids: set[int] = set()
        self._dequanted_in_pool: set[int] = set()
        # Skip dequant/flush during CUDA graph capture and replay.
        self._in_cuda_graph: bool = False

        logger.info(
            f"KVarNAttnBackend initialized: head_dim={self.head_dim}, "
            f"group={self.group}, "
            f"k_bits={kvarn_config.key_bits}, v_bits={kvarn_config.value_bits}, "
            f"tile_bytes={kvarn_config.tile_bytes_aligned}"
        )

    def _ensure_flush_manager(self):
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
        max_blocks_by_mem = 512 * 1024 * 1024 // bytes_per_block
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
        )

    def init_forward_metadata(self, forward_batch: "ForwardBatch"):
        self._ensure_flush_manager()
        self._in_cuda_graph = False
        self.inner.init_forward_metadata(forward_batch)

    def init_forward_metadata_out_graph(self, forward_batch: "ForwardBatch", in_capture: bool = False):
        """Per-iter metadata prep — delegates to inner Triton backend.

        The CUDA graph runner calls this for both decode and extend modes.
        The inner Triton backend only supports DECODE mode for CUDA graph
        replay. For EXTEND mode, we use the eager metadata path instead.
        """
        self._ensure_flush_manager()
        if in_capture:
            self._in_cuda_graph = True

        is_decode = forward_batch.forward_mode.is_decode()

        if not is_decode:
            # Non-decode forward (e.g. warmup prefill) during CUDA graph
            # context — use eager path, not the CUDA graph path.
            self._in_cuda_graph = False
            self.inner.init_forward_metadata(forward_batch)
            return

        # Decode forward — use the inner backend's CUDA graph path
        if not in_capture:
            # During replay: do lazy dequant before the graph replays,
            # then flush full blocks after the previous decode step.
            self._dequant_compressed_for_decode(forward_batch)
            self._flush_full_blocks(forward_batch)

        self.inner.init_forward_metadata_out_graph(forward_batch, in_capture=in_capture)

    def init_forward_metadata_in_graph(self, forward_batch: "ForwardBatch"):
        """Graph-recordable static-shape GPU op — delegates to inner backend."""
        self.inner.init_forward_metadata_in_graph(forward_batch)

    def init_cuda_graph_state(self, *args, **kwargs):
        return self.inner.init_cuda_graph_state(*args, **kwargs)

    def get_cuda_graph_seq_len_fill_value(self) -> int:
        return self.inner.get_cuda_graph_seq_len_fill_value()

    def _rotate(self, x: torch.Tensor) -> torch.Tensor:
        if self.no_rotation:
            return x
        return torch.matmul(x.float(), self.H_t).to(x.dtype)

    def _unrotate(self, x: torch.Tensor) -> torch.Tensor:
        if self.no_rotation:
            return x
        return torch.matmul(x.float(), self.H).to(x.dtype)

    def _flush_full_blocks(self, forward_batch: "ForwardBatch"):
        if self.flush_manager is None:
            return

        G = self.group
        req_to_token = self.model_runner.req_to_token_pool.req_to_token
        seq_lens = forward_batch.seq_lens
        req_pool_indices = forward_batch.req_pool_indices

        flush_block_ids = []
        for i in range(forward_batch.batch_size):
            seq_len = seq_lens[i].item()
            if seq_len < G:
                continue
            req_idx = req_pool_indices[i].item()
            num_complete_pages = seq_len // G
            for page_idx in range(num_complete_pages):
                # Skip page 0 (sink block) — keep fp16 for attention
                # sink accuracy and multi-turn prefix cache reuse.
                if page_idx == 0:
                    continue
                token_start = page_idx * G
                first_slot = req_to_token[req_idx, token_start].item()
                if first_slot < 0:
                    continue
                block_id = first_slot // G
                if block_id in self._flushed_block_ids:
                    continue
                page_slots = req_to_token[req_idx, token_start:token_start + G]
                if (page_slots >= 0).all():
                    flush_block_ids.append(block_id)
                    self._flushed_block_ids.add(block_id)

        if not flush_block_ids:
            return

        tail_k = self.token_to_kv_pool.k_buffer
        tail_v = self.token_to_kv_pool.v_buffer
        self.flush_manager.flush_blocks(flush_block_ids, tail_k, tail_v)

    def _dequant_compressed_for_decode(self, forward_batch: "ForwardBatch"):
        """Dequant compressed blocks back into the bf16 pool before decode.

        Only dequants blocks not already in the pool (tracked via
        _dequanted_in_pool). The dequanted data overwrites the stale bf16
        pool slots, so the standard Triton decode reads the compressed-
        then-decompressed data.
        """
        if self.flush_manager is None or not self._flushed_block_ids:
            return

        G = self.group
        req_to_token = self.model_runner.req_to_token_pool.req_to_token
        seq_lens = forward_batch.seq_lens
        req_pool_indices = forward_batch.req_pool_indices

        blocks_to_dequant = []
        for i in range(forward_batch.batch_size):
            seq_len = seq_lens[i].item()
            if seq_len == 0:
                continue
            req_idx = req_pool_indices[i].item()
            num_pages = (seq_len + G - 1) // G
            for page_idx in range(num_pages):
                token_start = page_idx * G
                first_slot = req_to_token[req_idx, token_start].item()
                if first_slot < 0:
                    continue
                block_id = first_slot // G
                if block_id in self._flushed_block_ids and block_id not in self._dequanted_in_pool:
                    blocks_to_dequant.append(block_id)

        if not blocks_to_dequant:
            return

        tail_k = self.token_to_kv_pool.k_buffer
        tail_v = self.token_to_kv_pool.v_buffer
        self.flush_manager.dequant_blocks_to_pool(blocks_to_dequant, tail_k, tail_v)
        self._dequanted_in_pool.update(blocks_to_dequant)

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
        pool_dtype = self.token_to_kv_pool.dtype
        model_dtype = q.dtype

        # Before decode: dequant compressed blocks back into bf16 pool
        # (skip during CUDA graph capture/replay)
        if not self._in_cuda_graph:
            self._dequant_compressed_for_decode(forward_batch)

        # Rotate Q/K/V
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

        o = self.inner.forward_decode(
            q_out, k_out, v_out, layer, forward_batch, save_kv_cache=save_kv_cache, **kwargs,
        )

        # After decode: flush full blocks to int4 (skip during CUDA graph)
        if not self._in_cuda_graph:
            self._flush_full_blocks(forward_batch)

        o_3d = o.view(-1, layer.tp_q_head_num, layer.v_head_dim)
        o_unrot = self._unrotate(o_3d) if o_3d.shape[-1] == self.head_dim else o_3d
        return o_unrot.reshape(-1, layer.tp_q_head_num * layer.v_head_dim).to(model_dtype)

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
        pool_dtype = self.token_to_kv_pool.dtype
        model_dtype = q.dtype

        if not self._in_cuda_graph:
            self._dequant_compressed_for_decode(forward_batch)

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

        if not self._in_cuda_graph:
            self._flush_full_blocks(forward_batch)

        o_3d = o.view(-1, layer.tp_q_head_num, layer.v_head_dim)
        o_unrot = self._unrotate(o_3d) if o_3d.shape[-1] == self.head_dim else o_3d
        return o_unrot.reshape(-1, layer.tp_q_head_num * layer.v_head_dim).to(model_dtype)