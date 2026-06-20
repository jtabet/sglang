# SPDX-License-Identifier: Apache-2.0
"""KVarN attention backend for SGLang.

Wraps the standard Triton attention backend, adding:
  1. Hadamard rotation of Q, K, V before attention.
  2. Un-rotation of the output after attention.

In Phase 2 (initial), the KV pool stores fp16 rotated K/V. This means
the attention math is identical to the standard Triton backend — the
only difference is the rotation. The compression (int4 tile flush) will
be added in Phase 3.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
from sglang.srt.layers.quantization.kvarn.config import KVarNConfig
from sglang.srt.layers.quantization.kvarn.hadamard import build_hadamard

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


class KVarNAttnBackend(AttentionBackend):
    """KVarN attention backend — wraps TritonAttnBackend with Hadamard rotation.

    The Hadamard rotation is applied to Q, K, V on every forward pass:
      Q' = Q @ H
      K' = K @ H   (stored rotated in the KV pool)
      V' = V @ H   (stored rotated in the KV pool)

    Attention is then computed in the rotated frame:
      O' = softmax(Q' @ K'^T / sqrt(d)) @ V'

    The output is un-rotated:
      O = O' @ H^T = O' @ H  (since H is symmetric and orthonormal)
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
        self.inner = TritonAttnBackend(model_runner, skip_prefill=skip_prefill)
        self.device = model_runner.device
        self.head_dim = kvarn_config.head_dim

        # Build and cache the Hadamard matrix for the model's head_dim.
        self.H = build_hadamard(self.head_dim, self.device)
        self.H_t = self.H.t().contiguous()  # H^T = H for Sylvester, but keep explicit

        # The Hadamard matrix is [D, D]. For batched rotation of [N, H, D] tensors,
        # we use matmul with broadcasting: x @ H_t → [N, H, D].
        logger.info(
            f"KVarNAttnBackend initialized: head_dim={self.head_dim}, "
            f"group={kvarn_config.group}, "
            f"k_bits={kvarn_config.key_bits}, v_bits={kvarn_config.value_bits}"
        )

    def init_forward_metadata(self, forward_batch: "ForwardBatch"):
        """Delegate metadata init to the inner Triton backend."""
        self.inner.init_forward_metadata(forward_batch)

    def init_cuda_graph_state(self, *args, **kwargs):
        """Delegate CUDA graph state init to the inner backend."""
        return self.inner.init_cuda_graph_state(*args, **kwargs)

    def init_forward_metadata_capture_cuda_graph(self, *args, **kwargs):
        """Delegate CUDA graph capture to the inner backend."""
        return self.inner.init_forward_metadata_capture_cuda_graph(*args, **kwargs)

    def init_forward_metadata_replay_cuda_graph(self, *args, **kwargs):
        """Delegate CUDA graph replay to the inner backend."""
        return self.inner.init_forward_metadata_replay_cuda_graph(*args, **kwargs)

    def get_cuda_graph_seq_len_fill_value(self) -> int:
        return self.inner.get_cuda_graph_seq_len_fill_value()

    def _rotate_qkv(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: "RadixAttention",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply Hadamard rotation to Q, K, V.

        Q shape: [num_tokens, tp_q_head_num, qk_head_dim]
        K shape: [num_tokens, tp_k_head_num, qk_head_dim] (or qk_head_dim for MLA)
        V shape: [num_tokens, tp_k_head_num, v_head_dim]
        """
        # Rotate Q: [N, Hq, D] @ [D, D] → [N, Hq, D]
        q_rot = torch.matmul(q, self.H_t)

        # Rotate K: [N, Hk, D] @ [D, D] → [N, Hk, D]
        k_rot = torch.matmul(k, self.H_t)

        # Rotate V: [N, Hk, Dv] @ [Dv, Dv] → [N, Hk, Dv]
        # V head_dim may differ from K head_dim in some models
        if v.shape[-1] == self.head_dim:
            v_rot = torch.matmul(v, self.H_t)
        else:
            # If v_head_dim != head_dim, we need a Hadamard of that size.
            # For now, only support v_head_dim == head_dim.
            v_rot = v  # No rotation if dims don't match

        return q_rot, k_rot, v_rot

    def _unrotate_output(self, o: torch.Tensor) -> torch.Tensor:
        """Un-rotate the attention output: O = O' @ H^T.

        O shape: [num_tokens, tp_q_head_num, v_head_dim]
        """
        if o.shape[-1] == self.head_dim:
            return torch.matmul(o, self.H)  # H @ H^T = I, so O = O' @ H
        return o

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
        """Decode forward with Hadamard rotation."""
        # Reshape to [N, H, D] for rotation
        q_3d = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        k_3d = k.view(-1, layer.tp_kv_head_num, layer.qk_head_dim)
        v_3d = v.view(-1, layer.tp_kv_head_num, layer.v_head_dim)

        # Rotate
        q_rot, k_rot, v_rot = self._rotate_qkv(q_3d, k_3d, v_3d, layer)

        # Flatten back for the inner backend
        q_flat = q_rot.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)
        k_flat = k_rot.reshape(-1, layer.tp_kv_head_num * layer.qk_head_dim)
        v_flat = v_rot.reshape(-1, layer.tp_kv_head_num * layer.v_head_dim)

        # Call inner backend's forward_decode
        o = self.inner.forward_decode(
            q_flat, k_flat, v_flat, layer, forward_batch, save_kv_cache, **kwargs
        )

        # Un-rotate output
        o_3d = o.view(-1, layer.tp_q_head_num, layer.v_head_dim)
        o_unrot = self._unrotate_output(o_3d)
        return o_unrot.reshape(-1, layer.tp_q_head_num * layer.v_head_dim)

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
        """Extend forward with Hadamard rotation."""
        # Reshape to [N, H, D] for rotation
        q_3d = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        k_3d = k.view(-1, layer.tp_kv_head_num, layer.qk_head_dim)
        v_3d = v.view(-1, layer.tp_kv_head_num, layer.v_head_dim)

        # Rotate
        q_rot, k_rot, v_rot = self._rotate_qkv(q_3d, k_3d, v_3d, layer)

        # Flatten back for the inner backend
        q_flat = q_rot.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)
        k_flat = k_rot.reshape(-1, layer.tp_kv_head_num * layer.qk_head_dim)
        v_flat = v_rot.reshape(-1, layer.tp_kv_head_num * layer.v_head_dim)

        # Call inner backend's forward_extend
        o = self.inner.forward_extend(
            q_flat, k_flat, v_flat, layer, forward_batch, save_kv_cache, **kwargs
        )

        # Un-rotate output
        o_3d = o.view(-1, layer.tp_q_head_num, layer.v_head_dim)
        o_unrot = self._unrotate_output(o_3d)
        return o_unrot.reshape(-1, layer.tp_q_head_num * layer.v_head_dim)

    def forward(self, *args, **kwargs):
        """Main forward entry point — delegates to the appropriate mode."""
        # The base class forward() dispatches to forward_decode/forward_extend.
        # We override it to avoid double-dispatch issues.
        forward_batch = kwargs.get("forward_batch", args[4] if len(args) > 4 else None)
        if forward_batch is None:
            # Fallback: let the base class handle it
            return super().forward(*args, **kwargs)

        if forward_batch.forward_mode.is_idle():
            q = args[0] if args else kwargs["q"]
            return q.new_empty(q.shape[0], args[3].tp_q_head_num * args[3].v_head_dim)
        elif forward_batch.forward_mode.is_decode():
            return self.forward_decode(*args, **kwargs)
        else:
            return self.forward_extend(*args, **kwargs)