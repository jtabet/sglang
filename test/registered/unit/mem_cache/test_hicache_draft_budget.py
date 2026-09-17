"""Unit tests for HiCache draft budget adjustment — no server, no model loading.

When ``--hicache-size`` caps the target host pool and a SIDECAR draft model
adds a second host pool with the same token count but a larger per-token byte
cost (e.g. quantized target + bf16 draft), the combined allocation can exceed
the user-specified budget.  ``_adjust_hicache_size_for_draft`` shrinks the
budget so the *total* (target + draft) stays within the cap.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.mem_cache.kv_cache_builder import (
    _adjust_hicache_size_for_draft,
    _device_pool_size_per_token,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# The function under test writes through get_context().override on the memory
# bag, so we patch get_context at its source module to prevent real config
# mutation.
_GET_CONTEXT_PATH = "sglang.srt.mem_cache.kv_cache_builder.get_context"


def _make_mha_pool(head_dim, head_num, layer_num, dtype, size=4096):
    """Create a minimal MHA-like pool with the attributes used by the sizing logic."""
    return SimpleNamespace(
        head_dim=head_dim,
        head_num=head_num,
        layer_num=layer_num,
        store_dtype=dtype,
        dtype=dtype,
        size=size,
    )


def _make_mla_pool(kv_cache_dim, layer_num, dtype, size=4096):
    """Create a minimal MLA-like pool with the attributes used by the sizing logic."""
    return SimpleNamespace(
        kv_cache_dim=kv_cache_dim,
        layer_num=layer_num,
        store_dtype=dtype,
        dtype=dtype,
        size=size,
    )


def _adjust(**overrides):
    """Call _adjust_hicache_size_for_draft with the standard test inputs."""
    args = dict(
        hicache_size=4,
        hicache_ratio=2.0,
        target_device_pool=_make_mha_pool(128, 8, 28, torch.bfloat16),
        draft_device_pools=(_make_mha_pool(128, 8, 28, torch.bfloat16),),
    )
    args.update(overrides)
    return _adjust_hicache_size_for_draft(**args)


class TestDevicePoolSizePerToken(CustomTestCase):
    """Tests for _device_pool_size_per_token geometry-based computation."""

    def test_mha_pool_bf16(self):
        pool = _make_mha_pool(
            head_dim=128, head_num=8, layer_num=28, dtype=torch.bfloat16
        )
        # head_dim * head_num * layer_num * itemsize * 2 (K+V)
        self.assertEqual(_device_pool_size_per_token(pool), 128 * 8 * 28 * 2 * 2)

    def test_mha_pool_fp8(self):
        pool = _make_mha_pool(
            head_dim=128, head_num=8, layer_num=28, dtype=torch.float8_e4m3fn
        )
        # fp8 itemsize = 1
        self.assertEqual(_device_pool_size_per_token(pool), 128 * 8 * 28 * 1 * 2)

    def test_mla_pool(self):
        pool = _make_mla_pool(kv_cache_dim=512, layer_num=28, dtype=torch.bfloat16)
        # kv_cache_dim * layer_num * itemsize = 512 * 28 * 2
        self.assertEqual(_device_pool_size_per_token(pool), 512 * 28 * 2)

    def test_zero_size_pool_returns_zero(self):
        pool = SimpleNamespace(
            head_dim=None,
            head_num=None,
            kv_cache_dim=None,
            layer_num=None,
            store_dtype=None,
            dtype=None,
            size=0,
            get_kv_size_bytes=lambda: 0,
        )
        self.assertEqual(_device_pool_size_per_token(pool), 0.0)


class TestAdjustHicacheSizeForDraft(CustomTestCase):
    """Tests for _adjust_hicache_size_for_draft budget shrinking."""

    def _run(self, **pool_overrides):
        """Invoke the function with the standard inputs plus overrides.

        Returns the mock ``override`` handle so the caller can assert on the
        memory-bag write it performed (or assert it was never called).
        """
        with mock.patch(_GET_CONTEXT_PATH) as mock_get_context:
            mock_override = mock_get_context.return_value.override
            _adjust(**pool_overrides)
            return mock_override

    def test_no_adjustment_when_hicache_size_zero(self):
        """Ratio mode (hicache_size <= 0) should not be adjusted."""
        mock_override = self._run(hicache_size=0, hicache_ratio=2.0)
        mock_override.assert_not_called()

    def test_no_adjustment_when_no_draft_pools(self):
        mock_override = self._run(draft_device_pools=())
        mock_override.assert_not_called()

    def test_no_adjustment_when_target_spt_zero(self):
        zero_pool = SimpleNamespace(
            head_dim=None,
            head_num=None,
            kv_cache_dim=None,
            layer_num=None,
            store_dtype=None,
            dtype=None,
            size=0,
            get_kv_size_bytes=lambda: 0,
        )
        mock_override = self._run(
            target_device_pool=zero_pool,
            draft_device_pools=(_make_mha_pool(128, 8, 28, torch.bfloat16),),
        )
        mock_override.assert_not_called()

    def test_equal_per_token_cost_halves_budget(self):
        """When target and draft have equal per-token cost, each gets half."""
        mock_override = self._run()
        mock_override.assert_called_once_with(
            "kvarn.hicache_draft_budget",
            hicache_size=2,  # 4 * spt / (spt + spt)
            hicache_ratio=1.0,  # 2.0 * spt / (spt + spt)
        )

    def test_quantized_target_bf16_draft(self):
        """fp8 target + bf16 draft: draft is 2x the per-token cost."""
        mock_override = self._run(
            target_device_pool=_make_mha_pool(128, 8, 28, torch.float8_e4m3fn),
            draft_device_pools=(_make_mha_pool(128, 8, 28, torch.bfloat16),),
        )
        # target_spt = fp8 itemsize 1: 128*8*28*1*2 = 57344
        # draft_spt = bf16 itemsize 2: 128*8*28*2*2 = 114688
        target_spt = 57344
        draft_spt = 114688
        expected_size = int(4 * target_spt / (target_spt + draft_spt))
        expected_ratio = 2.0 * target_spt / (target_spt + draft_spt)
        mock_override.assert_called_once_with(
            "kvarn.hicache_draft_budget",
            hicache_size=expected_size,
            hicache_ratio=expected_ratio,
        )

    def test_bf16_target_fp8_draft(self):
        """bf16 target + fp8 draft: draft is 0.5x the per-token cost."""
        mock_override = self._run(
            target_device_pool=_make_mha_pool(128, 8, 28, torch.bfloat16),
            draft_device_pools=(_make_mha_pool(128, 8, 28, torch.float8_e4m3fn),),
        )
        target_spt = 114688
        draft_spt = 57344
        expected_size = int(4 * target_spt / (target_spt + draft_spt))
        expected_ratio = 2.0 * target_spt / (target_spt + draft_spt)
        mock_override.assert_called_once_with(
            "kvarn.hicache_draft_budget",
            hicache_size=expected_size,
            hicache_ratio=expected_ratio,
        )

    def test_mla_target_mha_draft(self):
        """MLA target + MHA draft: different pool types are handled."""
        mock_override = self._run(
            target_device_pool=_make_mla_pool(
                kv_cache_dim=512, layer_num=28, dtype=torch.bfloat16
            ),
            draft_device_pools=(_make_mha_pool(128, 8, 28, torch.bfloat16),),
        )
        # target_spt = 512 * 28 * 2 = 28672
        # draft_spt = 128 * 8 * 28 * 2 * 2 = 114688
        target_spt = 28672
        draft_spt = 114688
        expected_size = int(4 * target_spt / (target_spt + draft_spt))
        expected_ratio = 2.0 * target_spt / (target_spt + draft_spt)
        mock_override.assert_called_once_with(
            "kvarn.hicache_draft_budget",
            hicache_size=expected_size,
            hicache_ratio=expected_ratio,
        )


if __name__ == "__main__":
    unittest.main()
