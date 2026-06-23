# SPDX-License-Identifier: Apache-2.0
"""Unit tests for KVarN flush manager (CPU-only)."""

import pytest
import torch

from sglang.srt.layers.quantization.kvarn.config import KVarNConfig
from sglang.srt.layers.quantization.kvarn.flush_manager import KVarNFlushManager


@pytest.fixture
def cfg():
    return KVarNConfig.from_cache_dtype("kvarn_k4v4_g128", head_dim=128)


@pytest.fixture
def manager(cfg):
    return KVarNFlushManager(
        kvarn_config=cfg,
        num_blocks=64,
        num_kv_heads=4,
        num_layers=2,
        device="cpu",
    )


class TestFlushManager:
    def test_init(self, manager):
        assert manager.num_blocks == 64
        assert (manager.block_to_slot == -1).all()
        assert manager.compressed_cache is not None
        assert len(manager.compressed_cache) == 2
        for cache in manager.compressed_cache:
            assert cache.dtype == torch.uint8
            assert cache.shape == (64, 4, manager.tile_bytes)

    def test_flush_marks_compressed(self, manager, cfg):
        G, Hk, D = cfg.group, 4, cfg.head_dim
        pool_size = 64 * G
        tail_k = [torch.randn(pool_size, Hk, D, dtype=torch.bfloat16) for _ in range(2)]
        tail_v = [torch.randn(pool_size, Hk, D, dtype=torch.bfloat16) for _ in range(2)]

        manager.flush_blocks([0, 1], tail_k, tail_v)

        assert manager.block_to_slot[0].item() == -1
        assert manager.block_to_slot[1].item() == -1
        assert 0 in manager._flushed
        assert 1 in manager._flushed

    def test_flush_writes_nonzero(self, manager, cfg):
        G, Hk, D = cfg.group, 4, cfg.head_dim
        pool_size = 64 * G
        tail_k = [torch.randn(pool_size, Hk, D, dtype=torch.bfloat16) for _ in range(2)]
        tail_v = [torch.randn(pool_size, Hk, D, dtype=torch.bfloat16) for _ in range(2)]

        assert manager.compressed_cache[0][0, 0].sum().item() == 0
        manager.flush_blocks([0], tail_k, tail_v)
        assert manager.compressed_cache[0][0, 0].sum().item() > 0

    def test_dequant_roundtrip(self, manager, cfg):
        """Flush then dequant should approximately recover the original data."""
        G, Hk, D = cfg.group, 4, cfg.head_dim
        pool_size = 64 * G

        # Create deterministic data
        torch.manual_seed(42)
        orig_k = [torch.randn(pool_size, Hk, D, dtype=torch.bfloat16) for _ in range(2)]
        orig_v = [torch.randn(pool_size, Hk, D, dtype=torch.bfloat16) for _ in range(2)]

        # Copy for comparison
        orig_k_copy = [k.clone() for k in orig_k]
        orig_v_copy = [v.clone() for v in orig_v]

        # Flush block 0 to int4
        manager.flush_blocks([0], orig_k, orig_v)

        # Dequant back to pool
        manager.dequant_blocks_to_pool([0], orig_k, orig_v)

        # Compare: should be approximately equal (within 4-bit quantization error)
        slot_start = 0
        slot_end = G
        k_err = (orig_k[0][slot_start:slot_end].float() - orig_k_copy[0][slot_start:slot_end].float()).abs().mean()
        k_mag = orig_k_copy[0][slot_start:slot_end].float().abs().mean()
        rel_err = k_err / k_mag
        assert rel_err < 0.5, f"K dequant roundtrip error too high: {rel_err:.3f}"

    def test_no_flush_empty(self, manager):
        tail_k = [torch.zeros(10, 4, 128, dtype=torch.bfloat16) for _ in range(2)]
        tail_v = [torch.zeros(10, 4, 128, dtype=torch.bfloat16) for _ in range(2)]
        manager.flush_blocks([], tail_k, tail_v)
        assert (manager.block_to_slot == -1).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])