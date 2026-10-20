"""Detect Triton kernel device-loads after the engine starts serving.

Triton loads each kernel specialization's cubin onto the GPU at its first
launch (``CompiledKernel._init_handles`` -> ``cuModuleLoadData``). That load
needs free device memory *outside* the torch caching allocator. Engines size
their pools to leave little post-init headroom, and the allocator's high-water
mark consumes the rest during early serving — so a specialization first used
mid-serving (e.g. a new adaptive speculative draft length, or a rare batch-size
bucket) can die in ``cuModuleLoadData`` with CUDA OOM, minutes or hours in.

Once ``mark_serving_started()`` has been called, this module warns when an
uncached Triton compilation takes at least one second or a device-load starts
with less than 1 GiB of free device memory. Set
``SGLANG_CRASH_ON_TRITON_LOAD_AFTER_READY=1`` to raise on every late load
instead — for CI recipes that assert full startup warmup coverage. The hooks
only run for compilation and first-use loads, so steady-state cost is zero.

Note: request-driven warmup (``--warmups``, the server warmup request) runs
*after* ``mark_serving_started()`` and is subject to the same diagnostics;
crash mode is only meant for deployments whose kernels are fully pre-loaded at
engine init.
"""

from __future__ import annotations

import logging

import torch

from sglang.srt.environ import envs
from sglang.srt.utils.common import get_available_gpu_memory

logger = logging.getLogger(__name__)

_serving_started = False
_prev_compile_listener = None
_installed = False
_oom_witness_dumped = False


def _dump_oom_witness(free_gb):
    """One-shot faulthandler + allocator-snapshot witness dump.

    Fires on the first serving-time device-load under the warning threshold
    (i.e. on the allocating thread under memory pressure). Writes:
      - faulthandler stack of EVERY thread (the current one is inside or
        adjacent to the large allocation that is about to fail),
      - a torch.cuda memory._dump_snapshot pickle (carries per-allocation
        stacks when SGLANG_OOM_MEMORY_SNAPSHOT=record is on),
      - torch.cuda.memory_stats() summary lines.
    Never raises: diagnostics must not kill serving.
    """
    global _oom_witness_dumped
    if _oom_witness_dumped:
        return
    _oom_witness_dumped = True
    import faulthandler
    import os
    import time
    import traceback

    try:
        folder = None
        try:
            from sglang.srt.runtime_context import (
                get_context,
                get_observability,
            )

            ctx = get_context()
            if ctx.is_config_namespace_published("observability"):
                folder = get_observability().crash_dump_folder
        except Exception:  # noqa: BLE001
            folder = None
        ts = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
        txt = os.path.join(folder or "/tmp", f"oom_witness_{ts}.txt")
        with open(txt, "w") as f:
            f.write(f"oom witness at {time.time()} free_gb={free_gb}\n")
            f.flush()
            faulthandler.dump_traceback(file=f, all_threads=True)
            try:
                stats = torch.cuda.memory_stats()
                f.write("\nmemory_stats: ")
                for k in (
                    "reserved_bytes.all.current",
                    "reserved_bytes.all.peak",
                    "allocation.all.current",
                    "allocation.all.peak",
                    "segment.all.current",
                ):
                    f.write(f"{k}={stats.get(k, 0)} ")
                f.write("\n")
            except Exception:  # noqa: BLE001
                pass
        snap = os.path.join(folder or "/tmp", f"oom_witness_snapshot_{ts}.pickle")
        try:
            torch.cuda.memory._dump_snapshot(snap)
        except Exception:  # noqa: BLE001
            snap = None
        logger.error(
            "TRITON-LOAD OOM WITNESS: threads=%s snapshot=%s (folder=%s)",
            txt,
            snap or "<dump failed>",
            folder or "/tmp",
        )
    except Exception:  # noqa: BLE001
        logger.error("oom witness dump failed:\n%s", traceback.format_exc())


def install() -> None:
    """Install the diagnostics (idempotent; chains pre-existing hooks)."""
    global _installed, _prev_compile_listener
    if _installed:
        return
    try:
        import triton.knobs as knobs
    except ImportError:
        return

    # Triton 3.4 builds used by ROCm expose triton.knobs without these hooks.
    runtime = getattr(knobs, "runtime", None)
    compilation = getattr(knobs, "compilation", None)
    hook = getattr(runtime, "kernel_load_start_hook", None)
    if hook is None or compilation is None or not hasattr(compilation, "listener"):
        return

    _prev_compile_listener = compilation.listener
    hook.add(_on_kernel_load)
    compilation.listener = _on_compilation
    _installed = True


def mark_serving_started() -> None:
    """Arm diagnostics for subsequent Triton compilations and device-loads."""
    global _serving_started
    _serving_started = True


def _on_compilation(*, src, metadata, metadata_group, times, cache_hit) -> None:
    if _prev_compile_listener is not None:
        _prev_compile_listener(
            src=src,
            metadata=metadata,
            metadata_group=metadata_group,
            times=times,
            cache_hit=cache_hit,
        )
    if not _serving_started or cache_hit:
        return

    compile_time_secs = times.total / 1e6
    if compile_time_secs < envs.SGLANG_TRITON_SLOW_COMPILE_THRESHOLD_SECS.get():
        return

    logger.warning(
        "Triton kernel '%s' took %.2f s to compile after serving started. "
        "Serving-time compilation can stall the engine; pre-compile it during "
        "engine init.",
        src.name,
        compile_time_secs,
    )


def _on_kernel_load(module, function, name, metadata_group, hash) -> None:
    if not _serving_started:
        return

    free_gb = None
    if torch.cuda.is_available():
        try:
            free_gb = get_available_gpu_memory(
                "cuda", torch.cuda.current_device(), empty_cache=False
            )
        except RuntimeError:
            logger.debug("Unable to query free device memory", exc_info=True)

    should_crash = envs.SGLANG_CRASH_ON_TRITON_LOAD_AFTER_READY.get()
    if not should_crash and (
        free_gb is None or free_gb >= envs.SGLANG_TRITON_LOAD_WARNING_THRESHOLD_GB.get()
    ):
        return

    free_memory = f"{free_gb:.2f} GiB" if free_gb is not None else "unknown"
    msg = (
        f"Triton kernel '{name}' device-loaded after serving started "
        f"(free device mem: {free_memory}). Pre-load it during engine init "
        f"to avoid CUDA OOM."
    )
    if should_crash:
        raise RuntimeError(msg)
    logger.warning(msg)
    # A serving-time device-load under memory pressure is also the moment
    # large first-use allocations are failing nearby (the load runs on the
    # allocating thread — the stack below it IS the allocation context).
    # Dump faulthandler stacks + an allocator snapshot once, best-effort, so
    # an imminent WARP_MMU_FAULT can be attributed post-mortem. The pipe-
    # watcher approach (tail of the container stdout) loses this race: the
    # fault kills the process ~2s later and tail competes with the kubelet
    # for the log stream.
    _dump_oom_witness(free_gb)
