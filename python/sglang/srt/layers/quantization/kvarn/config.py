# SPDX-License-Identifier: Apache-2.0
"""KVarN configuration.

Ported from vLLM.  KVarN = Hadamard rotation + iterative variance
normalization (Sinkhorn-like) + asymmetric RTN quantization.

The configuration object is pure (no framework imports) so it can be
unit-tested in isolation.
"""

import math
import os
from dataclasses import dataclass

# Named KVarN presets: each maps to a frozen set of config parameters.
# The trailing g<N> encodes the variance-normalization tile size, which must
# equal the page size. g128 is the current design point; g64 trades a little
# compression (more per-tile scale overhead) for finer quantization granularity.
KVARN_PRESETS: dict[str, dict] = {
    "kvarn_k4v2_g128": {"key_bits": 4, "value_bits": 2, "group": 128},
    "kvarn_k4v4_g128": {"key_bits": 4, "value_bits": 4, "group": 128},
    "kvarn_k4v2_g64": {"key_bits": 4, "value_bits": 2, "group": 64},
    "kvarn_k4v4_g64": {"key_bits": 4, "value_bits": 4, "group": 64},
}


def is_kvarn_dtype(dtype_str: str) -> bool:
    """Return True if *dtype_str* is a KVarN preset name."""
    return isinstance(dtype_str, str) and dtype_str.startswith("kvarn_")


@dataclass
class KVarNConfig:
    """Configuration for KVarN KV-cache quantization.

    Pipeline per (block, head):
      1. Hadamard rotation along head_dim.
      2. Iterative log-domain variance-normalization (Sinkhorn-like).
      3. Asymmetric per-row RTN at key_bits / value_bits.
      4. Absorb the per-row RTN scale and zero-point into the matching
         sinkhorn scale axis.

    Cache layout (per (block, head)) is a single packed record — see
    ``tile_bytes`` / ``tile_bytes_aligned``.
    """

    head_dim: int = 128
    key_bits: int = 4
    value_bits: int = 4
    group: int = 128
    sinkhorn_iters: int = 8
    sink_tokens: int = 128
    boundary_skip_layers: int = 0

    # ── derived: storage layout ──────────────────────────────────────────────
    @property
    def k_packed_bytes(self) -> int:
        return math.ceil(self.head_dim * self.group * self.key_bits / 8)

    @property
    def v_packed_bytes(self) -> int:
        return math.ceil(self.group * self.head_dim * self.value_bits / 8)

    @property
    def k_scale_bytes(self) -> int:
        """fp16 bytes for K scales: s_col_K' [D] + zp_K' [D] + s_row_K [group]."""
        return (2 * self.head_dim + self.group) * 2

    @property
    def v_scale_bytes(self) -> int:
        """fp16 bytes for V scales: s_col_V [D] + s_row_V' [group] + zp_V' [group]."""
        return (self.head_dim + 2 * self.group) * 2

    @property
    def tile_bytes(self) -> int:
        return (
            self.k_packed_bytes
            + self.k_scale_bytes
            + self.v_packed_bytes
            + self.v_scale_bytes
        )

    @property
    def tile_bytes_aligned(self) -> int:
        """tile_bytes rounded up for nicer Triton loads."""
        if self.head_dim >= 256:
            slot = math.ceil(self.tile_bytes / self.group)
            slot_pow2 = 1 << (slot - 1).bit_length()
            return slot_pow2 * self.group
        return ((self.tile_bytes + 7) // 8) * 8

    # ── slot byte offsets within one tile (used by the kernels) ──────────────
    @property
    def k_packed_offset(self) -> int:
        return 0

    @property
    def k_s_col_offset(self) -> int:
        return self.k_packed_offset + self.k_packed_bytes

    @property
    def k_zp_offset(self) -> int:
        return self.k_s_col_offset + self.head_dim * 2

    @property
    def k_s_row_offset(self) -> int:
        return self.k_zp_offset + self.head_dim * 2

    @property
    def v_packed_offset(self) -> int:
        return self.k_s_row_offset + self.group * 2

    @property
    def v_s_col_offset(self) -> int:
        return self.v_packed_offset + self.v_packed_bytes

    @property
    def v_s_row_offset(self) -> int:
        return self.v_s_col_offset + self.head_dim * 2

    @property
    def v_zp_offset(self) -> int:
        return self.v_s_row_offset + self.group * 2

    # ── fp16 tail-pool sizing ────────────────────────────────────────────────
    POOL_MEM_FRAC_DEFAULT = 0.08
    POOL_USABLE_SHARE_DEFAULT = 0.5

    def _slot_bytes_per_layer(self, num_kv_heads: int) -> int:
        return self.group * num_kv_heads * self.head_dim * 4

    def pool_slots(self, max_num_seqs: int, max_num_batched_tokens: int) -> int:
        prefill_blocks = (max_num_batched_tokens + self.group - 1) // self.group
        return max(2 * max_num_seqs + prefill_blocks + 8, 8)

    def pool_budget_bytes(
        self,
        total_gpu_bytes: int,
        gpu_memory_utilization: float | None = None,
        weight_bytes: int | None = None,
    ) -> int:
        env = os.environ.get("KVARN_POOL_MEM_FRAC")
        if weight_bytes is not None and gpu_memory_utilization is not None:
            share = float(env) if env is not None else self.POOL_USABLE_SHARE_DEFAULT
            usable = gpu_memory_utilization * total_gpu_bytes - weight_bytes
            return max(0, int(share * usable))
        frac = float(env) if env is not None else self.POOL_MEM_FRAC_DEFAULT
        return int(total_gpu_bytes * frac)

    def max_supported_seqs(
        self,
        total_gpu_bytes: int,
        num_kv_heads: int,
        num_layers: int,
        max_num_batched_tokens: int,
        frac: float | None = None,
        gpu_memory_utilization: float | None = None,
        weight_bytes: int | None = None,
    ) -> int:
        if frac is not None:
            budget = int(total_gpu_bytes * frac)
        else:
            budget = self.pool_budget_bytes(
                total_gpu_bytes, gpu_memory_utilization, weight_bytes
            )
        slot_bytes = self._slot_bytes_per_layer(num_kv_heads) * max(num_layers, 1)
        max_slots = int(budget / slot_bytes)
        prefill_blocks = (max_num_batched_tokens + self.group - 1) // self.group
        return max(1, (max_slots - prefill_blocks - 8) // 2)

    def pool_bytes(
        self,
        max_num_seqs: int,
        max_num_batched_tokens: int,
        num_kv_heads: int,
        num_layers: int,
    ) -> int:
        slots = self.pool_slots(max_num_seqs, max_num_batched_tokens)
        return slots * self._slot_bytes_per_layer(num_kv_heads) * max(num_layers, 1)

    @staticmethod
    def get_boundary_skip_layers(num_layers: int, n: int = 2) -> list[str]:
        if n <= 0 or num_layers <= 0:
            return []
        n = min(n, num_layers // 2)
        first = list(range(n))
        last = list(range(num_layers - n, num_layers))
        return [str(i) for i in sorted(set(first + last))]

    @staticmethod
    def from_cache_dtype(cache_dtype: str, head_dim: int) -> "KVarNConfig":
        """Create a config from a preset string like ``"kvarn_k4v4_g128"``."""
        if cache_dtype not in KVARN_PRESETS:
            valid = ", ".join(KVARN_PRESETS.keys())
            raise ValueError(
                f"Unknown KVarN cache dtype: {cache_dtype!r}. Valid: {valid}"
            )
        preset = KVARN_PRESETS[cache_dtype]
        iters = int(os.environ.get("KVARN_SINKHORN_ITERS", "8"))
        sink_tokens = int(os.environ.get("KVARN_SINK_TOKENS", "128"))
        return KVarNConfig(
            head_dim=head_dim,
            key_bits=preset["key_bits"],
            value_bits=preset["value_bits"],
            group=preset["group"],
            sinkhorn_iters=iters,
            sink_tokens=sink_tokens,
        )