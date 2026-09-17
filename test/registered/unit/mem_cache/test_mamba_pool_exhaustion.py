"""Regression test for PR #36770 backport: graceful degradation when mamba
pool is exhausted during radix-cache insertion.

When every cached mamba state is momentarily pinned (lock_ref > 0 from
running requests or in-flight HiCache write-through / load-back DMAs),
``_try_alloc_mamba_slot`` returns None instead of asserting, and
``prepare_for_caching_req`` skips caching that chunk (returns 0) so the
caller's existing ``effective_cache_len <= 0`` path handles it. The
request keeps its own live state and retries on the next chunk.
"""

import logging
import unittest
from unittest import mock

from sglang.srt.mem_cache.base_prefix_cache import EvictParams


class TestMambaPoolExhaustionGracefulDegradation(unittest.TestCase):
    """Verify that _try_alloc_mamba_slot returns None (not assert) when the
    pool is exhausted, and prepare_for_caching_req degrades gracefully."""

    def test_try_alloc_mamba_slot_returns_none_on_exhaustion(self):
        """_try_alloc_mamba_slot returns None when alloc fails after eviction."""
        from sglang.srt.mem_cache.unified_cache.components.mamba import (
            MambaComponent,
        )

        # Build a minimal mock cache + tree_core
        cache = mock.MagicMock()
        cache.req_to_token_pool.mamba_allocator.alloc.return_value = None
        cache.evict_for_alloc = mock.MagicMock()
        cache.metrics_collector = mock.MagicMock()

        component = MambaComponent.__new__(MambaComponent)
        component.cache = cache
        component.component_type = mock.MagicMock()

        slot = component._try_alloc_mamba_slot()

        self.assertIsNone(slot)
        # alloc was called twice (initial + retry after eviction)
        self.assertEqual(cache.req_to_token_pool.mamba_allocator.alloc.call_count, 2)
        # eviction was attempted
        cache.evict_for_alloc.assert_called_once_with(
            EvictParams(num_tokens=0, mamba_num=1)
        )
        # metric was bumped
        cache.metrics_collector.increment_aux_alloc_failed.assert_called_once()

    def test_try_alloc_mamba_slot_succeeds_on_retry(self):
        """_try_alloc_mamba_slot returns the slot when eviction frees one."""
        from sglang.srt.mem_cache.unified_cache.components.mamba import (
            MambaComponent,
        )

        import torch

        cache = mock.MagicMock()
        # First alloc fails, second succeeds
        good_slot = torch.tensor([42])
        cache.req_to_token_pool.mamba_allocator.alloc.side_effect = [
            None,
            good_slot,
        ]
        cache.evict_for_alloc = mock.MagicMock()
        cache.metrics_collector = mock.MagicMock()

        component = MambaComponent.__new__(MambaComponent)
        component.cache = cache
        component.component_type = mock.MagicMock()

        slot = component._try_alloc_mamba_slot()

        self.assertIsNotNone(slot)
        self.assertEqual(slot, good_slot)
        # Metric was NOT bumped (success on retry)
        cache.metrics_collector.increment_aux_alloc_failed.assert_not_called()

    def test_census_logging_throttled(self):
        """The WARNING census is throttled to once per _CENSUS_THROTTLE_S."""
        import logging as _logging

        from sglang.srt.mem_cache.unified_cache.components.mamba import (
            MambaComponent,
        )

        cache = mock.MagicMock()
        cache.req_to_token_pool.mamba_allocator.alloc.return_value = None
        cache.evict_for_alloc = mock.MagicMock()
        cache.metrics_collector = mock.MagicMock()

        component = MambaComponent.__new__(MambaComponent)
        component.cache = cache
        component.component_type = mock.MagicMock()
        # Reset throttle so the first call logs
        MambaComponent._last_census_t = 0.0
        logger_name = "sglang.srt.mem_cache.unified_cache.components.mamba"

        class _Capture(_logging.Handler):
            def __init__(self):
                super().__init__(level=_logging.WARNING)
                self.records = []

            def emit(self, record):
                self.records.append(record.getMessage())

        # First failure: should log census
        cap = _Capture()
        _logging.getLogger(logger_name).addHandler(cap)
        try:
            component._try_alloc_mamba_slot()
        finally:
            _logging.getLogger(logger_name).removeHandler(cap)
        self.assertEqual(len(cap.records), 1)
        self.assertIn("census", cap.records[0])

        # Second failure within throttle window: the census must NOT repeat
        # (the throttled path returns before logging, so zero records is the
        # expected outcome -- assertLogs can't express that, hence the handler).
        cap2 = _Capture()
        _logging.getLogger(logger_name).addHandler(cap2)
        try:
            component._try_alloc_mamba_slot()
        finally:
            _logging.getLogger(logger_name).removeHandler(cap2)
        for line in cap2.records:
            self.assertNotIn("census", line)
        # The failure itself is still accounted for on every attempt.
        self.assertEqual(
            cache.metrics_collector.increment_aux_alloc_failed.call_count, 2
        )


if __name__ == "__main__":
    unittest.main()