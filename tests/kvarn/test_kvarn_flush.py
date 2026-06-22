# SPDX-License-Identifier: Apache-2.0
"""Unit tests for KVarN flush manager (CPU-only).

Run:
    python -m pytest tests/kvarn/test_kvarn_flush.py -v
"""

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
        max_pool_slots=16,
    )


class TestFlushManager:
    def test_init(self, manager):
        assert manager.num_blocks == 64
        assert manager.block_to_slot.dtype == torch.int32
        assert (manager.block_to_slot == -1).all()
        # Compressed cache is allocated eagerly
        assert manager.compressed_cache is not None
        assert len(manager.compressed_cache) == 2  # num_layers
        for cache in manager.compressed_cache:
            assert cache.dtype == torch.uint8
            assert cache.shape == (64, 4, manager.tile_bytes)

    def test_flush_blocks_marks_compressed(self, manager, cfg):
        """Test that flush_blocks sets block_to_slot=-1 for flushed blocks."""
        G, Hk, D = cfg.group, 4, cfg.head_dim

        # Create fake pools: [pool_size, Hk, D] bf16
        pool_size = 64 * G
        tail_k = [torch.randn(pool_size, Hk, D, dtype=torch.bfloat16) for _ in range(2)]
        tail_v = [torch.randn(pool_size, Hk, D, dtype=torch.bfloat16) for _ in range(2)]

        # Flush block 0 (slots 0..G-1) and block 1 (slots G..2G-1)
        manager.flush_blocks([0, 1], tail_k, tail_v)

        # block_to_slot should be -1 for flushed blocks
        assert manager.block_to_slot[0].item() == -1
        assert manager.block_to_slot[1].item() == -1
        # block_to_slot should still be -1 for unflushed blocks
        assert manager.block_to_slot[2].item() == -1
        # Should be in the _flushed set
        assert 0 in manager._flushed
        assert 1 in manager._flushed
        assert 2 not in manager._flushed

    def test_flush_writes_to_compressed_cache(self, manager, cfg):
        """Test that flush_blocks writes non-zero data to the compressed cache."""
        G, Hk, D = cfg.group, 4, cfg.head_dim

        pool_size = 64 * G
        tail_k = [torch.randn(pool_size, Hk, D, dtype=torch.bfloat16) for _ in range(2)]
        tail_v = [torch.randn(pool_size, Hk, D, dtype=torch.bfloat16) for _ in range(2)]

        # Before flush, compressed cache is all zeros
        assert manager.compressed_cache[0][0, 0].sum().item() == 0

        manager.flush_blocks([0], tail_k, tail_v)

        # After flush, compressed cache should have non-zero data for block 0
        assert manager.compressed_cache[0][0, 0].sum().item() > 0

    def test_flush_multiple_blocks(self, manager, cfg):
        """Test flushing multiple blocks at once."""
        G, Hk, D = cfg.group, 4, cfg.head_dim
        pool_size = 64 * G
        tail_k = [torch.randn(pool_size, Hk, D, dtype=torch.bfloat16) for _ in range(2)]
        tail_v = [torch.randn(pool_size, Hk, D, dtype=torch.bfloat16) for _ in range(2)]

        manager.flush_blocks([0, 1, 2], tail_k, tail_v)

        for bid in [0, 1, 2]:
            assert manager.block_to_slot[bid].item() == -1
            assert bid in manager._flushed

    def test_no_flush_empty_list(self, manager):
        """flush_blocks with empty list should be a no-op."""
        tail_k = [torch.zeros(10, 4, 128, dtype=torch.bfloat16) for _ in range(2)]
        tail_v = [torch.zeros(10, 4, 128, dtype=torch.bfloat16) for _ in range(2)]
        manager.flush_blocks([], tail_k, tail_v)
        # Nothing should change
        assert (manager.block_to_slot == -1).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])