"""Unit tests for LayerWiseLoadCounter bounded waits + cancellation.

Regression coverage for the scheduler-wedge crash class: an unbounded
``wait_until`` on the scheduler thread (frozen ``forward_ct`` → watchdog
SIGQUIT → pod restart) when the Mooncake store stalls. Pure Python: no GPU,
no store — the counter classes are loaded directly from the module file with
the heavy sglang import graph stubbed out.
"""

import importlib.util
import logging
import sys
import threading
import time
import types
import unittest
from concurrent.futures import Future

# ---- Load mooncake_direct_linker in isolation --------------------------------
# The real module imports sglang config/runtime helpers that pull the full
# sglang graph (transformers, etc.), which is not needed by the counter. Stub
# the runtime_context / utils / mem_cache imports it touches.
_STUBS = {}


def _install_stubs():
    torch = types.ModuleType("torch")
    torch.distributed = types.SimpleNamespace(is_available=lambda: False)

    class _Event:
        def record(self):
            pass

        def synchronize(self):
            pass

    torch.Event = _Event
    torch.get_device_module = lambda: types.SimpleNamespace(Event=_Event)
    sys.modules.setdefault("torch", torch)

    # sglang.srt.* pieces referenced at import time of the linker module
    sglang = types.ModuleType("sglang")
    srt = types.ModuleType("sglang.srt")
    srt.__path__ = []  # namespace-style, prevents real imports bleeding in
    sys.modules["sglang"] = sglang
    sys.modules["sglang.srt"] = srt

    def _mod(name, **attrs):
        module = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(module, key, value)
        sys.modules[name] = module
        return module

    _mod("sglang.srt.mem_cache")
    _mod("sglang.srt.mem_cache.cache_init_params", CacheInitParams=object)
    hicache = _mod(
        "sglang.srt.mem_cache.hicache_storage",
        HiCacheStorageConfig=object,
        PoolName=types.SimpleNamespace(
            KV="kv", MAMBA="mamba", SWA="swa", FULL="full"
        ),
        PoolTransfer=object,
    )
    _mod(
        "sglang.srt.mem_cache.hybrid_cache",
    )
    _mod(
        "sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller",
        HybridCacheController=types.SimpleNamespace(
            parse_storage_backend_extra_config=lambda *_: ({}, None)
        ),
    )
    _mod(
        "sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler",
        resolve_hybrid_device_pool_group=lambda **_: None,
    )
    _mod(
        "sglang.srt.mem_cache.unified_cache",
    )
    _mod(
        "sglang.srt.mem_cache.unified_cache.unified_cache_linker",
        UnifiedCacheLinker=object,
    )
    _mod(
        "sglang.srt.runtime_context",
        get_memory=lambda: None,
        get_model=lambda: None,
        get_parallel=lambda: None,
    )
    _mod(
        "sglang.srt.utils",
        freeze_gc=lambda *a, **k: None,
        get_device_module=torch.get_device_module,
    )

    spec = importlib.util.spec_from_file_location(
        "mooncake_direct_linker",
        "/home/jtabet/Workspace/jtabet/sglang/python/sglang/srt/mem_cache/"
        "storage/mooncake_store/mooncake_direct_linker.py",
    )
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as error:  # surface anything we failed to stub clearly
        raise AssertionError(f"stub set incomplete: {error!r}")
    return module


_LINKER = _install_stubs()
LayerWiseLoadCounter = _LINKER.LayerWiseLoadCounter
MooncakeLoadTimeout = _LINKER.MooncakeLoadTimeout


class TestLayerWiseLoadCounter(unittest.TestCase):
    def _counter(self, layers=4, deadline=1.0):
        return LayerWiseLoadCounter(layers, load_deadline_secs=deadline)

    def test_wait_returns_when_completion_lands_in_time(self):
        counter = self._counter(layers=2, deadline=5.0)
        index = counter.update_producer()
        counter.set_consumer(index)

        def complete_soon():
            time.sleep(0.1)
            counter.complete(index, 0)
            counter.complete(index, 1)

        thread = threading.Thread(target=complete_soon)
        thread.start()
        counter.wait_until(0)
        counter.wait_until(1)
        thread.join()
        self.assertNotIn(index, counter.futures)

    def test_layer_ready_probe(self):
        counter = self._counter(layers=2, deadline=5.0)
        index = counter.update_producer()
        self.assertFalse(counter.layer_ready(index, 0))
        counter.complete(index, 0)
        self.assertTrue(counter.layer_ready(index, 0))
        self.assertFalse(counter.layer_ready(index, 1))
        counter.complete(index, 1)
        self.assertTrue(counter.layer_ready(index, 1))
        # Unregistered batch: gate must treat it as ready (vacuous truth) so a
        # fully-drained batch never re-gates.
        self.assertTrue(counter.layer_ready(999, 0))

    def test_failed_future_is_not_ready_and_is_reported_failed(self):
        # jade SIGQUIT crash class (2026-10-13): a failed layer-0 future must
        # NOT read as ready — otherwise the pre-forward gate green-lights a
        # doomed batch and the failure raises mid-forward via wait_until.
        counter = self._counter(layers=2, deadline=5.0)
        index = counter.update_producer()
        counter.fail(index, RuntimeError("store key-miss"))
        self.assertFalse(counter.layer_ready(index, 0))
        self.assertTrue(counter.layer_failed(index, 0))
        # fail() is batch-atomic: every layer's future carries the exception.
        self.assertTrue(counter.layer_failed(index, 1))
        # Unregistered batch: no failure can be observed.
        self.assertFalse(counter.layer_failed(999, 0))
        # A waiter on the failed future still raises (diagnosable path kept).
        with self.assertRaises(RuntimeError):
            counter.wait_until(0)

    def test_wait_times_out_on_stall(self):
        counter = self._counter(layers=1, deadline=0.2)
        index = counter.update_producer()
        counter.set_consumer(index)
        with self.assertRaises(MooncakeLoadTimeout):
            counter.wait_until(0)
        # The failed batch is deregistered after the wait consumes it
        self.assertNotIn(index, counter._pending_until)

    def test_reaper_fails_before_wait(self):
        counter = self._counter(layers=2, deadline=0.1)
        index = counter.update_producer()
        counter.set_consumer(index)
        time.sleep(0.25)
        counter._reap_expired()
        with self.assertRaises(MooncakeLoadTimeout):
            counter.wait_until(0)
        # All layers of the failed batch raise (fail marks every future)
        with self.assertRaises(MooncakeLoadTimeout):
            counter.wait_until(1)

    def test_consumer_index_without_batch_is_noop(self):
        counter = self._counter(layers=2)
        counter.set_consumer(7)  # never registered
        counter.wait_until(0)  # must not raise or block

    def test_fail_clears_pending_registration(self):
        counter = self._counter(layers=1)
        index = counter.update_producer()
        counter.set_consumer(index)
        counter.fail(index, RuntimeError("store error"))
        with self.assertRaises(RuntimeError):
            counter.wait_until(0)
        self.assertNotIn(index, counter._pending_until)

    def test_set_consumer_bounds_a_pre_registered_batch(self):
        counter = self._counter(layers=2, deadline=0.2)
        index = counter.update_producer()
        # Registered long before the consumer bound; set_consumer arms it.
        counter.set_consumer(index)
        time.sleep(0.3)
        with self.assertRaises(MooncakeLoadTimeout):
            counter.wait_until(1)

    def test_reset_clears_everything(self):
        counter = self._counter(layers=2)
        index = counter.update_producer()
        counter.set_consumer(index)
        counter.reset()
        self.assertEqual(counter.producer_index, -1)
        self.assertEqual(counter.consumer_index, -1)
        self.assertEqual(counter.futures, {})
        self.assertEqual(counter._pending_until, {})

    def test_cross_thread_complete_releases_waiter(self):
        counter = self._counter(layers=3, deadline=5.0)
        index = counter.update_producer()
        counter.set_consumer(index)
        deadline_at = time.monotonic() + 5.0

        def complete_layers():
            for layer in range(3):
                time.sleep(0.05)
                counter.complete(index, layer)

        threading.Thread(target=complete_layers, daemon=True).start()
        counter.wait_until(2)  # last-layer wait implies all below
        self.assertLess(time.monotonic(), deadline_at)


if __name__ == "__main__":
    unittest.main()