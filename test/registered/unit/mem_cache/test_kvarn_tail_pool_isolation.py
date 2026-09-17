"""Regression test for the KVarN tail-pool slot-stealing contamination bug.

Root cause (recovered from the incident image, not inferred)
-----------------------------------------------------------
A production confidentiality incident: a long multi-turn coding request (choir)
generated another request's (pulsequery) GPU-spec answer verbatim. The
then-live patch ("Fix D") added tail-pool slot reconciliation to
``KVarNHostKVCache.load_to_device_per_layer`` (the HiCache H->D load-back
path). Its exact code (from the incident image) was, on the last layer:

    for bid in unique_block_ids.tolist():
        bid_int = int(bid)
        if bid_int in backend._block_to_slot:
            slot = backend._block_to_slot.pop(bid_int)
            backend._slot_to_block.pop(slot, None)
            backend._block_fill.pop(bid_int, None)
            backend._block_to_slot_t[bid_int] = -1
            backend._free_slots.append(slot)          # <-- steals a live slot
            backend._sink_block_ids.discard(bid_int)
            backend._retired_sinks.pop(bid_int, None)

The contamination mechanism is *block-id reuse*: the paged block allocator
reassigns a freed ``block_id`` to a new request. When a load-back then targets
that ``block_id`` (whose *old* KV is on host) while the *new* owner still holds
it in the tail pool, Fix D pops the new owner's slot, returns it to the shared
``_free_slots`` pool, and sets the GPU lookup tensor to -1. The new owner's
next decode finds ``_block_to_slot`` empty and falls through to the int4 cache
— reading the *previous* request's KV (``_read_block_dequantized``) instead of
its own tail-pool fp16 data.

This test locks in the invariant that prevents it: the H->D load-back path
must only write int4 tiles; it must never mutate the tail-pool slot maps.

These are pure-CPU tests; they do not launch a server or need CUDA.
Run with:
    python3 -m pytest test/registered/unit/mem_cache/test_kvarn_tail_pool_isolation.py -v
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.pool_host.mha import KVarNHostKVCache
from sglang.test.test_utils import CustomTestCase


def _make_backend():
    """Minimal stand-in for KVarNAttnBackend's tail-pool bookkeeping.

    Block 5 models a ``block_id`` that has been REUSED: it is currently owned
    by a live request (slot 0 in the tail pool), AND its *old* KV (from a
    previous request) is still present in the int4 cache and on host — so a
    concurrent load-back will target it. This is exactly the block-id-reuse
    shape that triggered the incident.
    """
    hk = 4
    tile_bytes = 32
    num_blocks = 8
    num_layers = 2
    # Consistent with _alloc_slot: block 5 maps to slot 0 in the GPU tensor.
    block_to_slot_t = torch.full((num_blocks,), -1, dtype=torch.int32)
    block_to_slot_t[5] = 0

    backend = SimpleNamespace(
        device="cpu",
        num_layers=num_layers,
        kv_cache_int4=[
            torch.full((num_blocks, hk, tile_bytes), 9, dtype=torch.uint8)
            for _ in range(num_layers)
        ],
        # Block 5 is live-owned (slot 0) by the current request.
        _block_to_slot={5: 0},
        _slot_to_block={0: 5},
        _block_fill={5: 16},
        _block_to_slot_t=block_to_slot_t,
        _block_lookup_size=num_blocks,
        _free_slots=[],
        _sink_block_ids=set(),
        _retired_sinks={},
    )
    return backend, hk, tile_bytes, num_blocks, num_layers


def _make_host_cache(backend, hk, tile_bytes, num_layers):
    """Construct KVarNHostKVCache without running __init__ (which pins RAM)."""
    page_size = 16
    cache = KVarNHostKVCache.__new__(KVarNHostKVCache)
    cache._kvarn_backend = backend
    cache._num_compressed_layers = num_layers
    cache.page_size = page_size
    cache.kv_buffer = torch.zeros(2, num_layers, hk, tile_bytes, dtype=torch.uint8)
    return cache


def _load_back_block(cache, block_id, host_page):
    """Drive a H->D load-back of one block via load_to_device_per_layer."""
    page_size = cache.page_size
    device_indices = torch.arange(
        block_id * page_size, (block_id + 1) * page_size, dtype=torch.long
    )
    host_indices = torch.arange(
        host_page * page_size, (host_page + 1) * page_size, dtype=torch.long
    )
    for layer_id in range(cache._num_compressed_layers):
        cache.load_to_device_per_layer(
            device_pool=None,
            host_indices=host_indices,
            device_indices=device_indices,
            layer_id=layer_id,
            io_backend=None,
        )


def _fix_d_reconcile(backend, block_id):
    """The exact Fix D slot reconciliation, factored out for the test.

    This is verbatim the incident image's tail-pool mutation (minus the
    layer_id gate), so the test exercises the actual hazard rather than a
    hand-waved "clear the maps".
    """
    bid = int(block_id)
    if bid in backend._block_to_slot:
        slot = backend._block_to_slot.pop(bid)
        backend._slot_to_block.pop(slot, None)
        backend._block_fill.pop(bid, None)
        if backend._block_to_slot_t is not None and bid < backend._block_lookup_size:
            backend._block_to_slot_t[bid] = -1
        if slot is not None:
            backend._free_slots.append(slot)
        backend._sink_block_ids.discard(bid)
        backend._retired_sinks.pop(bid, None)


class TestKVarNTailPoolIsolation(CustomTestCase):
    def test_load_back_does_not_steal_live_blocks_slot(self):
        """The current (fixed) load-back path must not touch the slot maps.

        Block 5 is live-owned (slot 0) and also the target of a load-back
        (block-id reuse). After load-back, the live request must still own
        slot 0 — otherwise its decode falls through to the int4 cache and
        reads a *previous* request's KV.
        """
        backend, hk, tile_bytes, num_blocks, num_layers = _make_backend()
        cache = _make_host_cache(backend, hk, tile_bytes, num_layers)

        _load_back_block(cache, block_id=5, host_page=0)

        # The live request's slot 0 must remain mapped to block 5.
        self.assertEqual(backend._block_to_slot.get(5), 0)
        self.assertEqual(backend._slot_to_block.get(0), 5)
        self.assertEqual(backend._block_fill.get(5), 16)
        # Slot 0 must NOT have been returned to the free pool.
        self.assertNotIn(0, backend._free_slots)
        # The GPU lookup tensor must still resolve block 5 to slot 0.
        self.assertEqual(int(backend._block_to_slot_t[5]), 0)

    def test_fix_d_reconcile_steals_live_blocks_slot(self):
        """Document the hazard: the incident's Fix D *did* steal the slot.

        This is the regression assertion inverted — it proves the bug being
        guarded against is real and that the fix targets it, not some
        unrelated behavior. If someone re-introduces slot reconciliation in
        load-back, this is exactly what it does to a live request's block.
        """
        backend, hk, tile_bytes, num_blocks, num_layers = _make_backend()

        _fix_d_reconcile(backend, block_id=5)

        # Slot 0 was stolen and returned to the free pool.
        self.assertNotIn(5, backend._block_to_slot)
        self.assertNotIn(0, backend._slot_to_block)
        self.assertIn(0, backend._free_slots)
        # The GPU lookup now points block 5 at int4 (-1), redirecting the live
        # request's reads to stale int4 tiles from a previous request.
        self.assertEqual(int(backend._block_to_slot_t[5]), -1)

    def test_load_back_still_writes_int4_tiles(self):
        """The fix must not break load-back's actual job (H->D int4 restore)."""
        backend, hk, tile_bytes, num_blocks, num_layers = _make_backend()
        cache = _make_host_cache(backend, hk, tile_bytes, num_layers)

        # Seed host page 0, layer 0 with value 7; the int4 cache starts at 9.
        cache.kv_buffer[0, 0, :, :] = 7

        _load_back_block(cache, block_id=5, host_page=0)

        # Block 5's int4 tile for layer 0 must now hold the seeded value.
        self.assertTrue(torch.all(backend.kv_cache_int4[0][5] == 7))
        # Layer 1 was not seeded (host buffer zero), so block 5's layer-1 tile
        # is now zero.
        self.assertTrue(torch.all(backend.kv_cache_int4[1][5] == 0))


if __name__ == "__main__":
    unittest.main()
