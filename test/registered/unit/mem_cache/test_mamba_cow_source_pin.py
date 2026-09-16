"""CPU-only regression test for the mamba COW-source pin.

Root cause (verified against the incident image, not inferred)
--------------------------------------------------------------
A production confidentiality incident: a long multi-turn coding request
("choir") emitted another request's ("pulsequery") GPU-spec answer verbatim.
The mechanism is a block-id reuse race in the mamba deferred-COW path:

1. `MambaComponent.finalize_match_result_in_cache` captures the COW source
   from `result.best_match_node`'s device mamba value, with NO lock in the
   normal path.
2. The scheduler pins only `req.last_node == result.last_device_node`
   (`best_match_device_node`). With HiCache the two nodes diverge: the deeper
   node can carry host-backed Full + device mamba.
3. The deferred copy runs later on the forward stream, but the source node's
   mamba `lock_ref == 0` — evictable. Under mamba pressure the slot is freed
   and reallocated to another request, and the copy reads the wrong bytes.

The fix: when `best_match_node != last_device_node`, pin the source node's
mamba-only from capture until the forward drains, released alongside the
request lock.

These are pure-CPU tests; they do not launch a server or need CUDA.
Run with:
    python3 -m pytest test/registered/unit/mem_cache/test_mamba_cow_source_pin.py -v
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

import unittest

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    DecLockRefParams,
    IncLockRefResult,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.unified_cache.components.mamba_component import (
    MambaComponent,
)
from sglang.srt.mem_cache.unified_cache.components.tree_component import (
    ComponentType,
)


class _FakeReq:
    def __init__(self):
        self.mamba_pool_idx = torch.tensor([0])
        self.mamba_cow_src_index = None
        self.mamba_needs_clear = False
        self.mamba_cow_lock_node = None
        self.mamba_cow_lock_params = None


class _FakeTreeCore:
    def __init__(self):
        self._device_value = torch.tensor([7])

    def get_component_device_value(self, node_id, component_type):
        return self._device_value


class _FakeCache:
    tree_components = (ComponentType.FULL, ComponentType.MAMBA)

    def __init__(self):
        self.tree_core = _FakeTreeCore()
        self.inc_lock_calls = []
        self.dec_lock_calls = []

    def inc_lock_ref(self, node_id, skip_lock_components=()):
        self.inc_lock_calls.append((node_id, skip_lock_components))
        return IncLockRefResult(
            skip_lock_node_ids={ct: {node_id} for ct in skip_lock_components}
        )

    def dec_lock_ref(self, node_id, params=None):
        self.dec_lock_calls.append((node_id, params))
        return None


def _make_component():
    cache = _FakeCache()
    component = object.__new__(MambaComponent)
    component.cache = cache
    component.tree_core = cache.tree_core
    return component, cache


def _match_result(best_match_node, last_device_node):
    return MatchResult(
        device_indices=torch.tensor([0]),
        last_device_node=last_device_node,
        last_host_node=best_match_node,
        best_match_node=best_match_node,
    )


class TestMambaCowSourcePin(unittest.TestCase):
    def test_pins_source_when_nodes_diverge(self):
        component, cache = _make_component()
        req = _FakeReq()
        params = MatchPrefixParams(key=None, req=req, cow_mamba=True)
        result = _match_result(best_match_node=42, last_device_node=10)

        component.finalize_match_result_in_cache(params, result)

        # The COW source was captured from best_match_node.
        self.assertEqual(req.mamba_cow_src_index.item(), 7)
        # A mamba-only pin was taken on the source node (skip every other comp).
        self.assertEqual(len(cache.inc_lock_calls), 1)
        node_id, skip = cache.inc_lock_calls[0]
        self.assertEqual(node_id, 42)
        self.assertEqual(set(skip), {ComponentType.FULL})
        # The pin was recorded on the req for later release.
        self.assertEqual(req.mamba_cow_lock_node, 42)
        self.assertIsInstance(req.mamba_cow_lock_params, DecLockRefParams)

    def test_does_not_pin_when_nodes_coincide(self):
        component, cache = _make_component()
        req = _FakeReq()
        params = MatchPrefixParams(key=None, req=req, cow_mamba=True)
        result = _match_result(best_match_node=42, last_device_node=42)

        component.finalize_match_result_in_cache(params, result)

        self.assertEqual(req.mamba_cow_src_index.item(), 7)
        # last_node == best_match_node: the request lock already pins the source.
        self.assertEqual(len(cache.inc_lock_calls), 0)
        self.assertIsNone(req.mamba_cow_lock_node)
        self.assertIsNone(req.mamba_cow_lock_params)

    def test_no_pin_when_no_device_mamba_source(self):
        component, cache = _make_component()
        cache.tree_core._device_value = None  # source already evicted
        req = _FakeReq()
        params = MatchPrefixParams(key=None, req=req, cow_mamba=True)
        result = _match_result(best_match_node=42, last_device_node=10)

        component.finalize_match_result_in_cache(params, result)

        self.assertIsNone(req.mamba_cow_src_index)
        self.assertEqual(len(cache.inc_lock_calls), 0)
        self.assertIsNone(req.mamba_cow_lock_node)


if __name__ == "__main__":
    unittest.main()
