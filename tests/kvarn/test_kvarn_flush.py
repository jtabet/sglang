# SPDX-License-Identifier: Apache-2.0
"""Unit tests for KVarN flush manager (CPU-only).

Tests the block lifecycle: allocation, fill tracking, flush detection,
and slot recycling.

Run:
    python -m pytest tests/kvarn/test_kvarn_flush.py -v
"""

import pytest
import torch

from sglang.srt.layers.quantization.kvarn.config import KVarNConfig
from sglang.srt.layers.quantization.kvarn.flush_manager import (
    BlockState,
    KVarNFlushManager,
)


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


class TestBlockState:
    def test_defaults(self):
        s = BlockState()
        assert s.fill_count == 0
        assert s.tail_slot == -1
        assert s.is_sink is False
        assert s.is_compressed is False


class TestFlushManager:
    def test_init(self, manager):
        assert manager.num_blocks == 64
        assert len(manager.block_states) == 64
        assert len(manager.free_slots) == 16
        assert manager.block_to_slot.dtype == torch.int32
        assert (manager.block_to_slot == -1).all()
        # Compressed cache is allocated eagerly
        assert manager.compressed_cache is not None
        assert len(manager.compressed_cache) == 2  # num_layers
        for cache in manager.compressed_cache:
            assert cache.dtype == torch.uint8
            assert cache.shape == (64, 4, manager.tile_bytes)

    def test_allocate_tail_slot(self, manager):
        slot = manager.allocate_tail_slot(0)
        assert slot == 0  # first free slot
        assert manager.block_states[0].tail_slot == 0
        assert manager.block_to_slot[0].item() == 0
        assert 0 not in manager.free_slots

    def test_allocate_tail_slot_idempotent(self, manager):
        slot1 = manager.allocate_tail_slot(5)
        slot2 = manager.allocate_tail_slot(5)
        assert slot1 == slot2  # same slot returned

    def test_allocate_tail_slot_exhaustion(self, manager):
        # Allocate all 16 slots
        for i in range(16):
            slot = manager.allocate_tail_slot(i)
            assert slot >= 0
        # Next allocation should fail
        slot = manager.allocate_tail_slot(16)
        assert slot == -1

    def test_free_tail_slot(self, manager):
        slot = manager.allocate_tail_slot(0)
        assert slot >= 0
        manager.free_tail_slot(0)
        assert manager.block_states[0].tail_slot == -1
        assert manager.block_to_slot[0].item() == -1
        assert slot in manager.free_slots

    def test_mark_as_sink(self, manager):
        manager.mark_as_sink(3)
        assert manager.block_states[3].is_sink is True

    def test_update_fill_count(self, manager):
        manager.update_fill_count(0, 64)
        assert manager.block_states[0].fill_count == 64
        manager.update_fill_count(0, 64)
        assert manager.block_states[0].fill_count == 128  # capped at group

    def test_get_blocks_to_flush(self, manager, cfg):
        # Block 0: full, not sink, not compressed → flushable
        manager.allocate_tail_slot(0)
        manager.update_fill_count(0, cfg.group)
        # Block 1: full but sink → not flushable
        manager.allocate_tail_slot(1)
        manager.update_fill_count(1, cfg.group)
        manager.mark_as_sink(1)
        # Block 2: not full → not flushable
        manager.allocate_tail_slot(2)
        manager.update_fill_count(2, 64)
        # Block 3: full, no slot → not flushable
        manager.update_fill_count(3, cfg.group)

        flushable = manager.get_blocks_to_flush()
        assert 0 in flushable
        assert 1 not in flushable
        assert 2 not in flushable
        assert 3 not in flushable

    def test_reset_block(self, manager):
        manager.allocate_tail_slot(0)
        manager.update_fill_count(0, 128)
        manager.mark_as_sink(0)
        manager.reset_block(0)
        state = manager.block_states[0]
        assert state.fill_count == 0
        assert state.tail_slot == -1
        assert state.is_sink is False
        assert state.is_compressed is False

    def test_flush_blocks(self, manager, cfg):
        """Test that flush_blocks marks blocks as compressed and frees slots."""
        G, Hk, D = cfg.group, 4, cfg.head_dim

        # Allocate and fill a block
        manager.allocate_tail_slot(0)
        manager.update_fill_count(0, G)

        # Create fake tail pools (CPU)
        tail_k = [torch.randn(16, G, Hk, D, dtype=torch.float16) for _ in range(2)]
        tail_v = [torch.randn(16, G, Hk, D, dtype=torch.float16) for _ in range(2)]

        # Flush
        manager.flush_blocks([0], tail_k, tail_v)

        # Check state
        assert manager.block_states[0].is_compressed is True
        assert manager.block_states[0].tail_slot == -1  # slot freed
        assert manager.block_to_slot[0].item() == -1
        assert 0 in manager.free_slots  # slot recycled

    def test_flush_multiple_blocks(self, manager, cfg):
        """Test flushing multiple blocks at once."""
        G, Hk, D = cfg.group, 4, cfg.head_dim

        for bid in [0, 1, 2]:
            manager.allocate_tail_slot(bid)
            manager.update_fill_count(bid, G)

        tail_k = [torch.randn(16, G, Hk, D, dtype=torch.float16) for _ in range(2)]
        tail_v = [torch.randn(16, G, Hk, D, dtype=torch.float16) for _ in range(2)]

        manager.flush_blocks([0, 1, 2], tail_k, tail_v)

        for bid in [0, 1, 2]:
            assert manager.block_states[bid].is_compressed is True
            assert manager.block_states[bid].tail_slot == -1

    def test_slot_recycling_after_flush(self, manager, cfg):
        """Test that freed slots can be reused for new blocks."""
        G, Hk, D = cfg.group, 4, cfg.head_dim

        # Fill and flush a block
        manager.allocate_tail_slot(0)
        manager.update_fill_count(0, G)
        tail_k = [torch.randn(16, G, Hk, D, dtype=torch.float16) for _ in range(2)]
        tail_v = [torch.randn(16, G, Hk, D, dtype=torch.float16) for _ in range(2)]
        manager.flush_blocks([0], tail_k, tail_v)

        # Allocate a new block — should reuse the freed slot
        slot = manager.allocate_tail_slot(10)
        assert slot >= 0  # should get the recycled slot
        assert manager.block_states[10].tail_slot == slot

    def test_block_to_slot_tensor(self, manager):
        """Test that block_to_slot tensor is updated correctly."""
        manager.allocate_tail_slot(0)
        manager.allocate_tail_slot(1)
        assert manager.block_to_slot[0].item() == 0
        assert manager.block_to_slot[1].item() == 1
        manager.free_tail_slot(0)
        assert manager.block_to_slot[0].item() == -1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])