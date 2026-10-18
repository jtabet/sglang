"""Unit tests for the external-load quarantine match gate.

Regression coverage for the two crash classes this flag family guards:

* iteration 19 — a node whose device slots are the target of an in-flight (or
  failed, unreclaimed) Mooncake store->device load must NOT be matched by any
  unrelated request — previously an in-flight tail was matchable, so a request
  could ride device slots whose content was never verified (silent corruption),
  and a failed load left them permanently trustable.
* iteration 24 (jade, the canary image `1ba096df`) — the OWNER must still see
  its own in-flight tail. A request whose whole prefix is its own in-flight
  load re-matches that prefix inside cache_unfinished_req (chunked-prefill
  stash); with the pre-iteration-25 walk, the pending mark broke the walk at
  node-1, so ``len(new_indices)==0`` while ``cache_protected_len`` stayed at
  the full 185088 — the symmetric assert that SIGQUITs both TP ranks:

      AssertionError: req.kv.cache_protected_len=185088, len(new_indices)=0,
      page_aligned_len=200704

  The walk now threads the matched rid and lets a node whose
  ``external_load_owner_rid`` equals the matching rid through, keeping the
  break for every other (or req-less) match and for a failed/quarantined tail
  (whose owner is cleared — a failed load must never be matched, its owner is
  handled by the pre-forward load gate / recompute fallback).

Match-walk behavior is tested hermetically here against a stand-in Full
component (no GPU, no distributed group); the flag lifecycle tests stay. The
real component walk is covered upstream by the registered unified-cache KL
tests (GPU CI). Skip condition: sglang unimportable (transformers etc.).
"""

import os
import sys
import types
import unittest
from array import array
from unittest import mock

import torch


def _install_triton_stub() -> None:
    """Stub triton when it is unusable on this host (no GPU driver)."""
    try:
        importlib = __import__("importlib")
        importlib.import_module("sglang.kernels.ops.quantization.fp8_kernel")
        return  # real triton + kernels importable: nothing to stub
    except Exception:
        pass

    def _jit(*a, **k):
        def d(fn):
            return fn

        return d if not (a and callable(a[0])) else a[0]

    class Config:
        def __init__(self, *a, **k):
            pass

    class OutOfResources(Exception):
        pass

    class Constexpr:
        def __init__(self, value):
            self.value = value

    def _none(*a, **k):
        return None

    triton = types.ModuleType("triton")
    triton.jit = _jit
    triton.Config = Config
    triton.autotune = _jit
    triton.heuristics = _jit
    triton.__version__ = "3.8.0"

    lang = types.ModuleType("triton.language")
    lang.constexpr = Constexpr
    lang.jit = _jit
    lang.dtype = None
    for name in [
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float16",
        "bfloat16",
        "float32",
        "float64",
        "bool",
        "pointer",
        "float8e4nv",
        "float8e5",
        "float8e4b8",
        "float8e4b15",
    ]:
        setattr(lang, name, None)
    for fn in [
        "maximum",
        "minimum",
        "max",
        "min",
        "log2",
        "exp",
        "sqrt",
        "rsqrt",
        "load",
        "store",
        "arange",
        "zeros",
        "where",
        "sum",
        "dot",
        "abs",
        "floor",
        "ceil",
        "exp2",
        "log",
        "trans",
        "reshape",
        "view",
        "program_id",
        "num_programs",
        "atomic_add",
        "static_range",
        "multiple_of",
        "max_contiguous",
        "cdiv",
        "cumsum",
        "flip",
        "permute",
        "expand_dims",
        "join",
        "debug_barrier",
        "print",
        "atomic_max",
        "atomic_cas",
        "inline_asm_elementwise",
        "fma",
        "select",
        "full",
        "random_seed",
        "assume",
        "static_assert",
        "static_print",
        "argmax",
        "argmin",
        "clz",
        "count_trailing_zeros",
        "swizzle2d",
        "fdiv",
    ]:
        setattr(lang, fn, _none)
    lang_core = types.ModuleType("triton.language.core")
    lang_core.view = None
    lang.core = lang_core
    triton.language = lang

    extra = types.ModuleType("triton.language.extra")
    extra_cuda = types.ModuleType("triton.language.extra.cuda")
    extra_cuda.libdevice = types.ModuleType("triton.language.extra.cuda.libdevice")
    extra.libdevice = extra_cuda.libdevice
    math_mod = types.ModuleType("triton.language.math")
    lang.math = math_mod
    lang.extra = extra

    class KernelInterface:
        def __getitem__(self, grid):
            return lambda *a, **k: None

    class JITFunction(KernelInterface):
        def __init__(self, *a, **k):
            pass

    runtime = types.ModuleType("triton.runtime")
    jit_mod = types.ModuleType("triton.runtime.jit")
    jit_mod.jit = _jit
    jit_mod.KernelInterface = KernelInterface
    jit_mod.JITFunction = JITFunction
    driver = types.ModuleType("triton.runtime.driver")

    class _Active:
        def get_benchmarker(self):
            return _none

        def __getattr__(self, name):
            return _none

    class _DriverWrap:
        active = _Active()

    driver.driver = _DriverWrap()
    runtime.driver = driver
    runtime.jit = jit_mod
    knobs = types.ModuleType("triton.runtime.knobs")
    autotuner = types.ModuleType("triton.runtime.autotuner")
    autotuner.autotune = _jit
    autotuner.heuristics = _jit
    autotuner.Config = Config
    autotuner.OutOfResources = OutOfResources

    class AttrsDescriptor:
        pass

    class CompiledKernel:
        pass

    backends = types.ModuleType("triton.backends")
    backends.__path__ = []
    backends_compiler = types.ModuleType("triton.backends.compiler")

    class GPUTarget:
        def __init__(self, *a, **k):
            pass

    backends_compiler.GPUTarget = GPUTarget
    backends.compiler = backends_compiler
    runtime.knobs = knobs
    runtime.autotuner = autotuner
    triton.runtime = runtime
    backends.compiler = backends_compiler
    compiler = types.ModuleType("triton.compiler")
    compiler_mod = types.ModuleType("triton.compiler.compiler")
    compiler_mod.AttrsDescriptor = AttrsDescriptor
    compiler_mod.CompiledKernel = CompiledKernel
    compiler_mod.triton_key = lambda: "stub-key"
    compiler.compiler = compiler_mod
    compiler.CompiledKernel = CompiledKernel
    triton.backends = backends
    triton.compiler = compiler

    sys.modules.update(
        {
            "triton": triton,
            "triton.language": lang,
            "triton.language.core": lang_core,
            "triton.language.extra": extra,
            "triton.language.extra.cuda": extra_cuda,
            "triton.language.extra.cuda.libdevice": extra_cuda.libdevice,
            "triton.language.math": math_mod,
            "triton.runtime": runtime,
            "triton.runtime.jit": jit_mod,
            "triton.runtime.driver": driver,
            "triton.runtime.knobs": knobs,
            "triton.runtime.autotuner": autotuner,
            "triton.backends": backends,
            "triton.backends.compiler": backends_compiler,
            "triton.compiler": compiler,
            "triton.compiler.compiler": compiler_mod,
        }
    )


_install_triton_stub()

try:
    from sglang.srt.mem_cache.base_prefix_cache import (
        InsertParams,
        MatchPrefixParams,
        MatchResult,
    )
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
    from sglang.srt.mem_cache.radix_cache import RadixKey
    from sglang.srt.mem_cache.unified_cache import unified_cache_linker
    from sglang.srt.mem_cache.unified_cache.components import (
        BASE_COMPONENT_TYPE,
        ComponentType,
        ExternalLinkerLoadPhase,
        LinkerTransferPhase,
    )
    from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (
        MINT_UNCLAIMED_MINTS,
        UnifiedCacheLinkerWrapper,
    )
    from sglang.srt.mem_cache.unified_cache.cache_action import FreeDeviceKV
    from sglang.srt.mem_cache.unified_cache.unified_tree_core import (
        UnifiedTreeCore,
        UnifiedTreeNode,
    )

    _IMPORT_OK = True
    _IMPORT_ERROR = ""
except Exception as error:  # sglang graph unimportable on this host
    _IMPORT_OK = False
    _IMPORT_ERROR = str(error)
    # A dir-mode pytest pass can import a sibling test file that shadows
    # sglang.srt.mem_cache, which fails the guarded import above even though
    # every test here would be @skipUnless'd anyway. Bind a placeholder so the
    # module-level class bodies (which name ComponentType at creation time)
    # still evaluate; the placeholder is never used at runtime in that case.
    globals().setdefault(
        "ComponentType", type("ComponentTypePlaceholder", (), {"FULL": "FULL"})
    )


class _FakeReq:
    """Minimal Req stand-in: the match walk only reads ``rid`` off it."""

    def __init__(self, rid):
        self.rid = rid


@unittest.skipUnless(_IMPORT_OK, f"sglang imports unavailable: {_IMPORT_ERROR}")
class TestExternalLoadQuarantine(unittest.TestCase):
    def _tree(self):
        cache = object.__new__(UnifiedTreeCore)
        cache.page_size = 1
        cache.is_eagle = False
        cache.enable_hicache = False
        cache.enable_storage = False
        cache.enable_external_cache_linker = False
        return cache

    def test_flag_defaults_false(self):
        node = UnifiedTreeNode((ComponentType.FULL,))
        self.assertFalse(node.external_load_pending)
        # Split copies the flag to the tail side only implicitly: a fresh
        # split parent (head fragment) must NOT inherit it.
        self.assertFalse(UnifiedTreeNode((ComponentType.FULL,)).external_load_pending)

    def test_owner_rid_defaults_none(self):
        # A fresh node has no in-flight-load owner: the match walk's
        # match-through exception must not fire for an unrelated rid.
        self.assertIsNone(
            UnifiedTreeNode((ComponentType.FULL,)).external_load_owner_rid
        )

    def test_mark_pending_and_quarantine_semantics(self):
        node = UnifiedTreeNode((ComponentType.FULL,))
        # pending → quarantine → cleared: the drain lifecycle.
        node.external_load_pending = True  # mark_external_cache_load_pending_path
        self.assertTrue(node.external_load_pending)
        node.external_load_pending = True  # mark_..._quarantined (idempotent)
        self.assertTrue(node.external_load_pending)
        node.external_load_pending = False  # clear_external_cache_load_pending
        self.assertFalse(node.external_load_pending)

    def test_mark_clear_owner_via_tree_methods(self):
        tree = self._tree()
        tree.component_types = (ComponentType.FULL,)
        tree.enable_session_radix_cache = False
        tree.device = torch.device("cpu")
        tree.token_to_kv_pool_allocator = None
        tree.kv_events = None
        tree.reset()

        node = tree._new_node()

        # mark_pending_path on a load queue records the owner...
        tree.mark_external_cache_load_pending_path(node.id, "req-owner")
        self.assertTrue(node.external_load_pending)
        self.assertEqual(node.external_load_owner_rid, "req-owner")
        # ...load success clears both...
        tree.clear_external_cache_load_pending(node.id)
        self.assertFalse(node.external_load_pending)
        self.assertIsNone(node.external_load_owner_rid)
        # ...a mark with no rid, then quarantine, clears the owner too: a
        # failed load must never be matchable, even by its own request.
        tree.mark_external_cache_load_pending_path(node.id, "req-owner")
        tree.mark_external_cache_load_quarantined(node.id)
        self.assertTrue(node.external_load_pending)
        self.assertIsNone(node.external_load_owner_rid)


class _FakeCache:
    """Cache stand-in for constructing a real UnifiedTreeCore on CPU.

    Mirrors the fields UnifiedTreeCore.__init__ reads plus what the Full
    component's match hooks dereference (root_node is bound by the core's
    reset()). ``page_size=1`` keeps every key a single child_key, so the tail
    under test is exactly one tree node — the iteration-24 crash shape (whole
    prefix carried by one in-flight load node).
    """

    device = torch.device("cpu")
    enable_session_radix_cache = False
    enable_hicache = False
    enable_storage = False
    enable_external_cache_linker = False
    page_size = 1
    is_eagle = False
    eviction_policy = "lru"
    eviction_policy_config = None
    token_to_kv_pool_allocator = None
    req_to_token_pool = None
    enable_kv_cache_events = False


class _StandInFullComponent:
    """Match-validation stub over the real Full component's protocol.

    The walk's _match_prefix_helper consults
    ``create_match_validator(match_device_only=True)`` on a tree whose
    ``enable_hicache`` is False, so the stand-in replicates that exactly: a
    node is a device match boundary iff its Full ``value`` is present. The
    insert path never runs here (the tail node is built directly), so no other
    hooks are needed.
    """

    def __init__(self, fake_cache):
        self.component_type = ComponentType.FULL

    def create_match_validator(self, match_device_only=False):
        def _valid(node):
            cd = node.component_data[self.component_type]
            if match_device_only:
                return cd.value is not None
            return cd.value is not None or node.backuped

        return _valid

    def finalize_match_result_in_tree_core(
        self, result, params, value_chunks, best_value_len
    ):
        # The real Full component adds host-only hit length from evicted
        # nodes; this stand-in tree has no host tier, so the incoming result
        # is already final.
        return result


def _attach_tail(core, key, value):
    """Insert one tail node the way the linker's load_back leaves it: a node
    under root whose key spans the whole externally restored prefix, carrying
    the device slot values the load will fill. Returns that node."""
    key = key.page_aligned(core.page_size)
    node = core._new_node()
    node.key = key
    node.component_data[BASE_COMPONENT_TYPE].value = value
    node.hash_value = []
    node.parent = core.root_node
    core.root_node.children[key.child_key(core.page_size)] = node
    return node


def _strip_tail(core, key):
    core.root_node.children.pop(key.child_key(core.page_size), None)


@unittest.skipUnless(_IMPORT_OK, f"sglang imports unavailable: {_IMPORT_ERROR}")
class TestOwnerSeesOwnInFlightTail(unittest.TestCase):
    """Reproduce the iteration-24 jade crash at the walk level.

    A request whose whole matched prefix is its own in-flight store->device
    load re-matches that prefix (chunked-prefill stash); the pending mark broke
    the walk, returning device_indices of 0 while cache_protected_len was the
    full 185088. Assert the walk's four decisions: owner -> through, every
    other / req-less rid -> stop, failed tail -> stop, cleared load -> match
    normally again.
    """

    def _core(self):
        return UnifiedTreeCore(
            _FakeCache(), {ComponentType.FULL: _StandInFullComponent(None)}
        )

    def _tail(self, core, tokens):
        key = RadixKey(array("q", [10**9 + i for i in range(tokens)]), "standin-extra")
        tail = _attach_tail(core, key, torch.arange(tokens, dtype=torch.int64))
        return key, tail

    @staticmethod
    def _match(core, key, rid):
        req = _FakeReq(rid) if rid is not None else None
        return core.match_prefix(MatchPrefixParams(key=key, req=req))

    def test_owner_match_threads_in_flight_tail(self):
        core = self._core()
        key, tail = self._tail(core, tokens=8)
        # The wrapper quarantines the freshly-inserted tail for the in-flight
        # load, recording the owner's rid.
        core.mark_external_cache_load_pending_path(tail.id, "rid-owner")
        self.assertTrue(tail.external_load_pending)

        result = self._match(core, key, "rid-owner")
        self.assertEqual(
            result.device_indices.numel(),
            8,
            "owner must re-match its own in-flight tail",
        )
        self.assertEqual(result.last_device_node, tail.id)
        self.assertEqual(result.full_kv_hit_length, 8)
        _strip_tail(core, key)

    def test_non_owner_stops_at_in_flight_tail(self):
        core = self._core()
        key, tail = self._tail(core, tokens=8)
        core.mark_external_cache_load_pending_path(tail.id, "rid-owner")

        result = self._match(core, key, "rid-other")
        self.assertEqual(
            result.device_indices.numel(),
            0,
            "an unrelated rid must not match an unverified in-flight tail",
        )
        _strip_tail(core, key)

    def test_reqless_match_stops_at_in_flight_tail(self):
        core = self._core()
        key, tail = self._tail(core, tokens=8)
        core.mark_external_cache_load_pending_path(tail.id, "rid-owner")

        result = self._match(core, key, None)
        self.assertEqual(result.device_indices.numel(), 0)
        _strip_tail(core, key)

    def test_failed_quarantined_tail_stops_even_for_owner(self):
        core = self._core()
        key, tail = self._tail(core, tokens=8)
        # mark..._pending_path then mark..._quarantined: the load failed before
        # it drained. The quarantine clears the owner, so the walk stops for
        # the owner too — the corruption guard (unverified slots must not be
        # matched) wins over the owner's convenience; the owner is retried by
        # the pre-forward load gate / recompute fallback.
        core.mark_external_cache_load_pending_path(tail.id, "rid-owner")
        core.mark_external_cache_load_quarantined(tail.id)
        self.assertIsNone(tail.external_load_owner_rid)

        result = self._match(core, key, "rid-owner")
        self.assertEqual(result.device_indices.numel(), 0)
        _strip_tail(core, key)

    def test_cleared_load_matches_normally_again(self):
        core = self._core()
        key, tail = self._tail(core, tokens=8)
        core.mark_external_cache_load_pending_path(tail.id, "rid-owner")
        core.clear_external_cache_load_pending(tail.id)  # load drained OK

        result = self._match(core, key, "rid-other")
        self.assertEqual(result.device_indices.numel(), 8)
        _strip_tail(core, key)


class _LinkerStoreStandIn:
    """External store stub for the mint-fallback lifecycle test."""

    def __init__(self, restorable_pages):
        self.restorable_pages = restorable_pages
        self.lookup_calls = 0
        self.queued_load_rids = []
        self.stats = {"lookup": 0, "load": 0, "offload": 0}

    def lookup(self, rid, transfers):
        self.lookup_calls += 1
        return list(range(1, self.restorable_pages + 1))

    def load(self, rid, transfers):
        self.queued_load_rids.append(rid)
        return True


class _LinkerComponentStandIn:
    """Full-component stub: reports the tail, materializes one slot short."""

    component_type = ComponentType.FULL

    def __init__(self):
        self.load_transfer_requests = 0

    def build_external_linker_transfer(self, phase, node, keys):
        if not keys:
            return None
        if phase == LinkerTransferPhase.LOOKUP:
            return PoolTransfer(name=PoolName.KV, keys=list(keys))
        if phase == LinkerTransferPhase.LOAD:
            self.load_transfer_requests += 1
            # One slot short of the claimed prefix: load_back's
            # materialized < claimed guard aborts into recompute, which is
            # exactly the branch that must not be re-entered after the
            # unclaimed-mint fallback has decided on recompute.
            return PoolTransfer(
                name=PoolName.KV,
                device_indices=torch.arange(len(keys) - 1, dtype=torch.int64),
                keys=list(keys),
            )
        return None

    def update_external_linker_load(
        self, phase, req, full_transfer, transfer, prefix_len, **kwargs
    ):
        return None if phase == ExternalLinkerLoadPhase.ABORT else transfer


class _LinkerWrapperCacheStandIn:
    """Cache stub: the fields UnifiedCacheLinkerWrapper/load_back dereference."""

    page_size = 1

    def __init__(self, component, store):
        self.tree_components = (ComponentType.FULL,)
        self.components = {ComponentType.FULL: component}
        self._components_tuple = (component,)
        self.token_to_kv_pool_allocator = None
        self.cache_linker = store
        self.write_through_threshold = 1
        self.all_reduce_calls = 0
        self.tree_core = types.SimpleNamespace(
            is_eagle=False,
            enable_external_cache_linker=False,
            empty_match_result=MatchResult(
                device_indices=torch.empty(0, dtype=torch.int64),
                last_device_node=None,
                last_host_node=None,
                best_match_node=None,
            ),
        )

    def get_last_hash_value(self, node):
        return "anchor-hash"

    def _all_reduce_attn_groups(self, mask, op):
        # Single-rank stand-in: a MIN/MAX over one participant is the identity.
        self.all_reduce_calls += 1


class _LinkerReqStandIn:
    def __init__(self, rid):
        self.rid = rid
        self.last_node = None
        self.prefix_indices = torch.empty(0, dtype=torch.int64)
        self.kv = None


@unittest.skipUnless(_IMPORT_OK, f"sglang imports unavailable: {_IMPORT_ERROR}")
class TestUnclaimedMintFallback(unittest.TestCase):
    """The mint fallback must retire the marker it decided to abandon.

    Live-lock (emerald 2026-10-18 06:30:17 and 00:19:46): a request whose
    prefix sits warm in the store mints a hit marker every tick, but admission
    defers before ``init_load_back`` (the FULL pool cannot take the promised
    span), so the marker is never claimed. Past ``MINT_UNCLAIMED_MINTS``
    ``match`` converts the rid to recompute — but it leaves the marker in
    ``hit_markers``, so ``has_hit(rid)`` stays True and every later tick where
    the device tree re-exposes any host hit re-enters ``load_back`` on a rid
    that was already decided. The request bounces ~200 store probes/s and
    never prefills.
    """

    def _wrapper(self):
        store = _LinkerStoreStandIn(restorable_pages=6)
        component = _LinkerComponentStandIn()
        cache = _LinkerWrapperCacheStandIn(component, store)
        wrapper = UnifiedCacheLinkerWrapper(cache, store)
        req = _LinkerReqStandIn("rid-unclaimed")
        key = RadixKey(array("q", range(32)), "mint-fallback")
        return store, component, cache, wrapper, req, key

    def _match(self, wrapper, cache, key, req):
        # The native hash extension is unavailable on CPU-only hosts; the
        # storage hashes are opaque to this test's assertions.
        with mock.patch.object(
            unified_cache_linker,
            "get_storage_hash_str",
            lambda k, prior_hash=None, page_size=1: ["h"] * len(k),
        ):
            return wrapper.match(key, req, cache.tree_core.empty_match_result)

    def test_fallback_retires_hit_marker(self):
        store, component, cache, wrapper, req, key = self._wrapper()

        for _ in range(MINT_UNCLAIMED_MINTS):
            result = self._match(wrapper, cache, key, req)
            self.assertEqual(result.host_hit_length, 6)
            self.assertTrue(wrapper.has_hit(req.rid))

        result = self._match(wrapper, cache, key, req)
        self.assertEqual(result.host_hit_length, 0, "fallback promises no host hit")
        self.assertIn(req.rid, wrapper._load_failed)
        self.assertFalse(
            wrapper.has_hit(req.rid),
            "the abandoned marker must be retired: a stale marker keeps "
            "has_hit() True and re-arms init_load_back for a rid already "
            "decided to recompute (the 06:30 live-lock)",
        )
        self.assertNotIn(req.rid, wrapper.hit_markers)
        self.assertNotIn(req.rid, wrapper._mint_last_ts)

    def test_fallback_stops_store_probes(self):
        store, component, cache, wrapper, req, key = self._wrapper()
        for _ in range(MINT_UNCLAIMED_MINTS + 1):
            self._match(wrapper, cache, key, req)
        self.assertIn(req.rid, wrapper._load_failed)

        probes_before = store.lookup_calls
        for _ in range(3):
            self._match(wrapper, cache, key, req)
        # `_load_failed` only zeroes hit_pages; the probe runs before it. The
        # fallback's promise to "skip future store probes" therefore holds only
        # if the marker/tail state stops the rid from reaching lookup again.
        self.assertEqual(
            store.lookup_calls - probes_before,
            0,
            "fallback promised no further store probes for this rid",
        )

    def test_fallback_blocks_load_back_reentry(self):
        store, component, cache, wrapper, req, key = self._wrapper()
        for _ in range(MINT_UNCLAIMED_MINTS + 1):
            self._match(wrapper, cache, key, req)
        self.assertIn(req.rid, wrapper._load_failed)

        indices, node = wrapper.load_back(req)
        # A retired marker means init_load_back's has_hit() gate is closed:
        # load_back is a no-op and the request prefills from its device prefix.
        self.assertEqual(indices.numel(), 0)
        self.assertEqual(
            component.load_transfer_requests,
            0,
            "a rid already converted to recompute must not allocate slots or "
            "queue another store load",
        )
        self.assertEqual(store.queued_load_rids, [])


class RecordingAllocator:
    """Allocator stand-in that records free_segment calls for leak tests."""

    def __init__(self):
        self.freed = set()

    def free_segment(self, indices, start_pos=0):
        for i in indices.tolist():
            self.freed.add(int(i))


class _InsertCapableFullComponent(_StandInFullComponent):
    """Full-component stub that also tolerates the insert hooks.

    ``TestOwnerSeesOwnInFlightTail`` builds its tail by hand (no insert), so the
    stand-in there only implements the match protocol. The emerald regression
    below must drive a *real* insert (that is where the divergence is produced),
    so extend the stub with the hooks the insert walk calls.
    """

    def __init__(self, fake_cache):
        super().__init__(fake_cache)
        self.cache = fake_cache
        self._alloc = RecordingAllocator()

    def free_segment(self, indices, start_pos=0):
        # RecordingAllocator passthrough (the free path consults this stub).
        for i in indices.tolist():
            self._alloc.freed.add(int(i))

    def update_component_on_insert_overlap(self, **kwargs):
        # Full takes ownership of every overlapping slot it already holds.
        return kwargs["prefix_len"]

    def commit_insert_component_data(self, **kwargs):
        return None

    def needs_incremental_backup(self, node):
        return False

    def recover_after_unevict(self, **kwargs):
        return None


@unittest.skipUnless(_IMPORT_OK, f"sglang imports unavailable: {_IMPORT_ERROR}")
class TestQuarantinedTailInsertDivergence(unittest.TestCase):
    """Reproduce the emerald 2026-10-18 07:08 crash at the tree level.

    Emerald chain: the pre-forward load gate tripped (store stalled, 480 s),
    cancelled the rid's queued load and ``mark_external_cache_load_quarantined``
    cleared the owner; the request was retracted (``reset_for_retract``) and
    requeued. On re-admission it re-inserted its own prefix — and the insert
    walk found the *already existing* quarantined node, kept it (updating it in
    place) instead of creating a fresh matchable leaf. ``insert`` therefore
    reported the full ``prefix_len`` while ``match_prefix`` stopped at the
    quarantine and returned 0 device indices, tripping

        assert new_prefix_len <= len(new_indices)   # 11392 <= 0

    In ``cache_unfinished_req`` (unified_radix_cache.py:1222), which SIGQUITs
    both TP ranks. These tests pin the two halves of that contract: the match
    walk must stop at a quarantined node, and the insert walk must agree with
    it — otherwise the pair is internally inconsistent and the assert fires.
    """

    def _core(self):
        return UnifiedTreeCore(
            _FakeCache(), {ComponentType.FULL: _InsertCapableFullComponent(None)}
        )

    @staticmethod
    def _pump(core, params):
        """Run an insert the way UnifiedRadixCache.insert pumps the walk."""
        step = core.begin_insert(params)
        while step.result is None:
            step = core.resume_insert()
        core.end_insert()
        return step.result

    @staticmethod
    def _pump_and_apply(core, alloc, params):
        """Run an insert exactly the way UnifiedRadixCache.insert does: apply
        each step's barrier actions (FreeDeviceKV -> allocator) and drain the
        tail. The adoption test asserts on the freed slots."""
        step = core.begin_insert(params)
        while True:
            for action in step.actions:
                if isinstance(action, FreeDeviceKV):
                    for indices in action.indices:
                        alloc.free_segment(indices, start_pos=0)
            if step.result is not None:
                for action in core.end_insert():
                    if isinstance(action, FreeDeviceKV):
                        for indices in action.indices:
                            alloc.free_segment(indices, start_pos=0)
                return step.result
            step = core.resume_insert()

    def _key(self, tokens):
        return RadixKey(array("q", [10**9 + i for i in range(tokens)]), "emerald")

    def test_insert_reuses_quarantined_node_and_diverges_from_match(self):
        """Pre-fix shape (rid-less insert), kept as a walk-behavior pin.

        Without a rid on InsertParams the walk cannot tell whose load the
        quarantined tail was, so it conservatively keeps its pre-fix
        behavior: reuse the node in place, leave the flag set. The
        production paths (cache_unfinished_req / cache_finished_req /
        linker.load_back) all carry rid, and land on the FIXED contract in
        test_insert_adopts_quarantined_tail_from_fresh_kv below. If either
        walk's shape changes, both tests must be re-reconciled or the
        assert class reopens.
        """
        core = self._core()
        key = self._key(8)
        first = self._pump(
            core, InsertParams(key=key, value=torch.arange(8, dtype=torch.int64))
        )
        self.assertEqual(first.prefix_len, 0)
        tail = core.node_by_id(first.last_device_node)

        # Gate trip: cancel_queued_load + mark_external_cache_load_quarantined.
        core.mark_external_cache_load_pending_path(tail.id, "rid-emerald")
        core.mark_external_cache_load_quarantined(tail.id)
        self.assertTrue(tail.external_load_pending)
        self.assertIsNone(tail.external_load_owner_rid)

        matched = core.match_prefix(
            MatchPrefixParams(key=key, req=_FakeReq("rid-emerald"))
        )
        self.assertEqual(
            matched.device_indices.numel(),
            0,
            "a quarantined tail must never be matched, even by its ex-owner",
        )

        again = self._pump(
            core, InsertParams(key=key, value=torch.arange(100, 108, dtype=torch.int64))
        )
        self.assertEqual(
            again.prefix_len,
            8,
            "the insert walk reuses the quarantined node and reports the whole "
            "prefix as inserted",
        )
        self.assertGreater(
            again.prefix_len,
            matched.device_indices.numel(),
            "insert prefix_len must stay <= match device span; the emerald "
            "assert (new_prefix_len <= len(new_indices)) encodes exactly this",
        )
        # The reused node is still quarantined: the insert did not clear the
        # flag, so the divergence persists across ticks (no self-healing).
        self.assertTrue(tail.external_load_pending)
        self.assertIsNone(tail.external_load_owner_rid)

    def test_insert_adopts_quarantined_tail_from_fresh_kv(self):
        """The FIXED contract (the production rid-carrying shape).

        With the request's rid on InsertParams, a quarantined node the
        request does not own must be ADOPTED from the request's fresh KV:
        the node's value is replaced by the fresh slots, the unverified
        load-target slots are freed, the flag clears, and every later walk
        (match included) sees a verified, matchable node — so
        cache_unfinished_req's new_prefix_len <= len(new_indices) holds.
        """
        core = self._core()
        alloc = RecordingAllocator()
        core.token_to_kv_pool_allocator = alloc
        key = self._key(8)
        first = self._pump(
            core, InsertParams(key=key, value=torch.arange(8, dtype=torch.int64))
        )
        tail = core.node_by_id(first.last_device_node)
        old_slots = tail.component_data[ComponentType.FULL].value.clone()

        core.mark_external_cache_load_pending_path(tail.id, "rid-emerald")
        core.mark_external_cache_load_quarantined(tail.id)
        self.assertTrue(tail.external_load_pending)

        matched = core.match_prefix(
            MatchPrefixParams(key=key, req=_FakeReq("rid-emerald"))
        )
        self.assertEqual(matched.device_indices.numel(), 0)

        fresh = torch.arange(100, 108, dtype=torch.int64)
        again = self._pump_and_apply(core, alloc, InsertParams(key=key, value=fresh.clone(), rid="rid-emerald"))
        self.assertEqual(again.prefix_len, 8)
        self.assertFalse(
            tail.external_load_pending,
            "the adoption must clear the quarantine: the fresh KV re-verified "
            "the span, so later walks must agree with the insert",
        )

        rematched = core.match_prefix(
            MatchPrefixParams(key=key, req=_FakeReq("rid-emerald"))
        )
        self.assertEqual(
            rematched.device_indices.numel(),
            8,
            "after adoption the match must return the full span — otherwise "
            "cache_unfinished_req's assert (new_prefix_len=11392, "
            "len(new_indices)=0) reopens",
        )
        # The unverified load-target slots must have been handed back.
        freed = core.token_to_kv_pool_allocator.freed
        self.assertTrue(
            set(old_slots.tolist()).issubset(freed),
            "adoption must free the old load-target slots (got freed=%r, "
            "old=%r)" % (freed, old_slots.tolist()),
        )

    def test_adoption_only_for_unowned_quarantine(self):
        """Owner-threading preserved: the ex-owner's own re-insert over its
        in-flight tail is NOT an adoption (its load is still in flight; the
        match walk threads the owner through, so insert must agree by
        consuming). Regression guard for the fix's ownership test."""
        core = self._core()
        key = self._key(8)
        first = self._pump(
            core, InsertParams(key=key, value=torch.arange(8, dtype=torch.int64))
        )
        tail = core.node_by_id(first.last_device_node)
        core.mark_external_cache_load_pending_path(tail.id, "rid-owner")
        self.assertIsNotNone(tail.external_load_owner_rid)

        # Owner's own in-flight tail: rid matches -> consume, no adoption.
        again = self._pump(
            core,
            InsertParams(key=key, value=torch.arange(100, 108, dtype=torch.int64), rid="rid-owner"),
        )
        self.assertEqual(again.prefix_len, 8)
        self.assertIsNotNone(
            tail.external_load_owner_rid,
            "the owner's insert must not clear its own in-flight marker",
        )

    def test_insert_does_not_clear_the_quarantine_flag(self):
        """A quarantined node keeps its flag through an overlapping insert.

        The insert walk's overlap branch updates an existing node in place
        (``_insert_walk_step``) and never touches the external-load flags. So a
        requeued request cannot heal the tree by re-inserting, which is why the
        emerald loop repeated until the assert instead of recovering.
        """
        core = self._core()
        key = self._key(8)
        first = self._pump(
            core, InsertParams(key=key, value=torch.arange(8, dtype=torch.int64))
        )
        tail = core.node_by_id(first.last_device_node)
        core.mark_external_cache_load_quarantined(tail.id)

        for _ in range(3):
            self._pump(
                core,
                InsertParams(
                    key=key, value=torch.arange(100, 108, dtype=torch.int64)
                ),
            )
            self.assertTrue(
                tail.external_load_pending,
                "re-inserting over a quarantined tail must not clear its flag "
                "(the slots are still unverified)",
            )
            matched = core.match_prefix(
                MatchPrefixParams(key=key, req=_FakeReq("rid-emerald"))
            )
            self.assertEqual(matched.device_indices.numel(), 0)

    def test_prepend_shaped_match_after_quarantine_is_empty(self):
        """The exact emerald shape: whole prefix carried by one node.

        Emerald's node spanned the request's entire 11392-token prefix
        (89 pages), so a match after the quarantine returns zero indices while
        the re-insert reports 11392. Reproduce with a single-node prefix and
        assert both halves of the pair, so the numbers in the report are pinned.
        """
        core = self._core()
        page = 128
        pages = 3
        tokens = page * pages
        key = self._key(tokens)
        first = self._pump(
            core,
            InsertParams(
                key=key, value=torch.arange(tokens, dtype=torch.int64)
            ),
        )
        tail = core.node_by_id(first.last_device_node)
        core.mark_external_cache_load_pending_path(tail.id, "rid-emerald")
        core.mark_external_cache_load_quarantined(tail.id)

        matched = core.match_prefix(
            MatchPrefixParams(key=key, req=_FakeReq("rid-emerald"))
        )
        self.assertEqual(matched.device_indices.numel(), 0)

        again = self._pump(
            core,
            InsertParams(
                key=key,
                value=torch.arange(1000, 1000 + tokens, dtype=torch.int64),
            ),
        )
        self.assertEqual(again.prefix_len, tokens)
        self.assertEqual(
            (again.prefix_len, matched.device_indices.numel()),
            (tokens, 0),
            "the emerald assert fires with (new_prefix_len, len(new_indices)) "
            "= (11392, 0) in production; this is that pair in miniature",
        )


class _BudgetAllocatorStandIn:
    """Allocator stub for PrefillBudget: reports a fixed free/evictable pool."""

    page_size = 128

    def __init__(self, available_tokens):
        self._available = available_tokens

    def available_size(self):
        return self._available


class _BudgetTreeStandIn:
    def __init__(self, evictable_tokens):
        self._evictable = evictable_tokens

    def supports_mamba(self):
        return True

    def full_evictable_size(self):
        return self._evictable


@unittest.skipUnless(_IMPORT_OK, f"sglang imports unavailable: {_IMPORT_ERROR}")
class TestFullPoolAdmissionDeadlock(unittest.TestCase):
    """Jade 2026-10-18 08:15: the request could never be admitted.

    The request's prefix (1448 pages = 185344 tokens) is fully restorable from
    the store, but the full pool cannot hold the request's *whole* prefill plus
    its decode headroom:

        available 77312 + evictable 112000 = remaining_total 189312
        prefix 185344 + CLIP_MAX_NEW_TOKENS 4096 + page_size 128 = 189568

    189568 > 189312, so ``PrefillBudget.check_prefill`` returns NO_TOKEN on
    every tick, forever: admission defers *before* ``init_load_back``
    (schedule_policy.py:1208-1218), so the hit marker is never claimed, and the
    load-back never runs to shrink the prefill. This test pins that arithmetic
    — including that the gap is smaller than the decode-headroom reservation,
    so no amount of store health fixes it.
    """

    CLIP_MAX_NEW_TOKENS = 4096  # SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION default

    def _budget(self, available, evictable):
        from sglang.srt.mem_cache.prefill_budget import PrefillBudget

        return PrefillBudget(
            _BudgetAllocatorStandIn(available), _BudgetTreeStandIn(evictable)
        )

    def test_jade_admission_is_arithmetically_impossible(self):
        budget = self._budget(available=77312, evictable=112000)
        page = budget.page_size
        prefix = 1448 * page

        self.assertEqual(prefix, 185344)
        self.assertEqual(budget.remaining_total, 189312)

        total_tokens = prefix + self.CLIP_MAX_NEW_TOKENS + page
        self.assertEqual(total_tokens, 189568)
        self.assertGreater(
            total_tokens,
            budget.remaining_total,
            "jade's total_tokens exceeded remaining_total, so check_prefill "
            "could only ever return NO_TOKEN",
        )

        charge = total_tokens
        can_admit, chunk_limit = budget.check_prefill(
            extend_input_len=prefix,
            total_tokens=charge,
            max_new_tokens=self.CLIP_MAX_NEW_TOKENS,
            input_tokens=prefix,
            swa_host_hit_length=0,
            chunk_limit=None,
        )
        self.assertFalse(can_admit)
        self.assertIsNone(chunk_limit)

    def test_deadlock_depends_on_decode_headroom_not_prompt_size(self):
        """The prompt alone fits; the decode reservation is what kills it.

        This is why the fix must not be "wait for the store": the store was
        healthy (every lookup hit), and the pool balance was exact. Only the
        reservation of ``CLIP_MAX_NEW_TOKENS`` of decode headroom pushed the
        request over. A chunked admission that defers the headroom reservation
        until after the load-back would break the tie.
        """
        budget = self._budget(available=77312, evictable=112000)
        page = budget.page_size
        prefix = 1448 * page

        # Prompt alone (plus page-alignment slack) does fit.
        self.assertLess(prefix + page, budget.remaining_total)
        # ...but with the decode reservation it does not.
        self.assertGreater(
            prefix + self.CLIP_MAX_NEW_TOKENS + page, budget.remaining_total
        )

    def test_marker_cannot_be_retired_because_load_back_is_unreachable(self):
        """The load-path fix cannot help: admission never reaches it.

        ``linker.match`` mints the marker, then ``add_one_req`` returns
        NO_TOKEN before ``init_load_back`` — so ``load_back`` never pops the
        marker and the unclaimed-mint fallback is the ONLY route out. With
        MINT_UNCLAIMED_MINTS ticks the fallback does fire (it converts the rid
        to recompute, host_hit_length -> 0), which is what makes the
        marker-retirement fix load-bearing: without retiring the retired
        marker, ``has_hit`` stays True and the next tick re-arms the loop.
        """
        budget = self._budget(available=77312, evictable=112000)
        prefix = 1448 * budget.page_size
        # Admission verdict is NO_TOKEN at the *promised* host hit.
        charge = prefix + self.CLIP_MAX_NEW_TOKENS + budget.page_size
        can_admit, _ = budget.check_prefill(
            extend_input_len=prefix,
            total_tokens=charge,
            max_new_tokens=self.CLIP_MAX_NEW_TOKENS,
            input_tokens=prefix,
            swa_host_hit_length=0,
            chunk_limit=None,
        )
        self.assertFalse(can_admit)
        # With the hit withdrawn (recompute fallback: host_hit_length 0) the
        # request still cannot fit its full prefix in one chunk...
        charge = prefix + self.CLIP_MAX_NEW_TOKENS + budget.page_size
        can_admit_recompute, _ = budget.check_prefill(
            extend_input_len=prefix,
            total_tokens=charge,
            max_new_tokens=self.CLIP_MAX_NEW_TOKENS,
            input_tokens=prefix,
            swa_host_hit_length=0,
            chunk_limit=None,
        )
        self.assertFalse(
            can_admit_recompute,
            "the fallback alone does not make this request fit; it only stops "
            "the store-probe storm (the chunked path then admits part of it)",
        )


if __name__ == "__main__":
    unittest.main()
