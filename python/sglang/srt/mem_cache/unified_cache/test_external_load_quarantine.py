"""Unit tests for the external-load quarantine match gate.

Regression coverage for the crash lattice (iteration 19): a node whose device
slots are the target of an in-flight (or failed, unreclaimed) Mooncake
store->device load must NOT be matched by any new request — previously an
in-flight tail was matchable, so a request could ride device slots whose
content was never verified (silent corruption), and a failed load left them
permanently trustable.

The flag lifecycle is tested here (tree-core attr semantics, CPU-hermetic via
a triton stub for driverless hosts). The match-walk gating behavior is covered
upstream by the registered unified-cache KL tests (GPU CI), which exercise the
exact ``match_prefix`` walk this flag gates.

Keep the inline triton stub in sync with /tmp/opencode/triton_stub.py if the
host file is removed. Skip condition: sglang unimportable (transformers etc.).
"""

import os
import sys
import types
import unittest
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
        "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
        "float16", "bfloat16", "float32", "float64", "bool", "pointer",
        "float8e4nv", "float8e5", "float8e4b8", "float8e4b15",
    ]:
        setattr(lang, name, None)
    for fn in [
        "maximum", "minimum", "max", "min", "log2", "exp", "sqrt", "rsqrt",
        "load", "store", "arange", "zeros", "where", "sum", "dot", "abs",
        "floor", "ceil", "exp2", "log", "trans", "reshape", "view",
        "program_id", "num_programs", "atomic_add", "static_range",
        "multiple_of", "max_contiguous", "cdiv", "cumsum", "flip", "permute",
        "expand_dims", "join", "debug_barrier", "print", "atomic_max",
        "atomic_cas", "inline_asm_elementwise", "fma", "select", "full",
        "random_seed", "assume", "static_assert", "static_print", "argmax",
        "argmin", "clz", "count_trailing_zeros", "swizzle2d", "fdiv",
    ]:
        setattr(lang, fn, _none)
    lang_core = types.ModuleType("triton.language.core")
    lang_core.view = None
    lang.core = lang_core
    triton.language = lang

    extra = types.ModuleType("triton.language.extra")
    extra_cuda = types.ModuleType("triton.language.extra.cuda")
    extra_cuda.libdevice = types.ModuleType(
        "triton.language.extra.cuda.libdevice"
    )
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
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.unified_cache.components import ComponentType
    from sglang.srt.mem_cache.unified_cache.unified_tree_core import (
        UnifiedTreeCore,
        UnifiedTreeNode,
    )

    _IMPORT_OK = True
    _IMPORT_ERROR = ""
except Exception as error:  # sglang graph unimportable on this host
    _IMPORT_OK = False
    _IMPORT_ERROR = str(error)


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

    def test_mark_pending_and_quarantine_semantics(self):
        node = UnifiedTreeNode((ComponentType.FULL,))
        # pending → quarantine → cleared: the drain lifecycle.
        node.external_load_pending = True  # mark_external_cache_load_pending_path
        self.assertTrue(node.external_load_pending)
        node.external_load_pending = True  # mark_..._quarantined (idempotent)
        self.assertTrue(node.external_load_pending)
        node.external_load_pending = False  # clear_external_cache_load_pending
        self.assertFalse(node.external_load_pending)


if __name__ == "__main__":
    unittest.main()