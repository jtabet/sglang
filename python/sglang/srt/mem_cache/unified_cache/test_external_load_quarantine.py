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
    )
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.radix_cache import RadixKey
    from sglang.srt.mem_cache.unified_cache.components import (
        BASE_COMPONENT_TYPE,
        ComponentType,
    )
    from sglang.srt.mem_cache.unified_cache.unified_tree_core import (
        UnifiedTreeCore,
        UnifiedTreeNode,
    )

    _IMPORT_OK = True
    _IMPORT_ERROR = ""
except Exception as error:  # sglang graph unimportable on this host
    _IMPORT_OK = False
    _IMPORT_ERROR = str(error)


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


if __name__ == "__main__":
    unittest.main()
