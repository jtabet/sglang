# SPDX-License-Identifier: Apache-2.0
"""Caching-allocator forensics on forward-pass OOM.

When a forward pass raises ``torch.OutOfMemoryError``, the interesting
question is rarely "how much free memory was there" (the exception already
says that) — it is *what holds the memory*. This module snapshots torch's
caching-allocator history (allocation stacks per buffer) to a file next to
the crash dumps, so a post-burst WARP_MMU_FAULT can be attributed to the
allocation that actually exhausted the dynamic region instead of guessed
from the faulting (often innocent) kernel.

Enable allocation-stack recording up front with
``SGLANG_OOM_MEMORY_SNAPSHOT=record`` — recording costs some CPU on every
allocation and is meant for diagnosis, not steady-state production. The
snapshot dump itself is always best-effort: any failure inside it must not
mask the original OOM handling.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

import torch

logger = logging.getLogger(__name__)

_RECORD_MODE = "record"
_history_enabled = False


def maybe_start_recording(crash_dump_folder: Optional[str]):
    """Start allocator history recording if requested by env.

    Called once at scheduler init. ``SGLANG_OOM_MEMORY_SNAPSHOT=record``
    enables torch.cuda.memory._record_memory_history(max_entries) so the
    OOM-time snapshot carries the stacks that produced the resident set.
    """
    global _history_enabled
    mode = os.environ.get("SGLANG_OOM_MEMORY_SNAPSHOT", "").strip().lower()
    if mode != _RECORD_MODE or _history_enabled or not torch.cuda.is_available():
        return
    try:
        torch.cuda.memory._record_memory_history(max_entries=100_000)
        _history_enabled = True
        logger.info(
            "OOM memory snapshot: allocator history recording enabled "
            "(SGLANG_OOM_MEMORY_SNAPSHOT=record)."
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("could not start allocator history recording: %s", e)


def dump_oom_memory_snapshot(folder: Optional[str], tag: str = "oom") -> Optional[str]:
    """Best-effort dump of the allocator snapshot; returns the file path.

    Call inside an ``except torch.OutOfMemoryError`` handler. With recording
    enabled the snapshot contains per-allocation stacks; without it, it still
    contains the segment/summary view (sizes and alloc counts), which
    identifiers the dominant holder even when stacks are absent.
    """
    if not torch.cuda.is_available():
        return None
    dest = os.path.join(folder or "/tmp", f"oom_memory_snapshot_{tag}_{int(time.time())}.pickle")
    try:
        # The dump path may not exist yet (e.g. a crash-dump-folder pointed at
        # a persisted volume subpath, first crash). torch's _dump_snapshot
        # does not create parents — a missing dir would lose the forensics
        # exactly when they matter (crash #19's pickle died with the pod).
        os.makedirs(folder or "/tmp", exist_ok=True)
        torch.cuda.memory._dump_snapshot(dest)
        logger.error("OOM memory snapshot written to %s", dest)
        try:
            free_b, total_b = torch.cuda.mem_get_info()
            res = torch.cuda.memory_stats()
            logger.error(
                "OOM context: device free=%.2f GiB / total=%.2f GiB; "
                "reserved=%.2f GiB reserved peak=%.2f GiB alloc=%s alloc peak=%s",
                free_b / 2**30,
                total_b / 2**30,
                res.get("reserved_bytes.all.current", 0) / 2**30,
                res.get("reserved_bytes.all.peak", 0) / 2**30,
                res.get("allocation.all.current", 0),
                res.get("allocation.all.peak", 0),
            )
        except Exception:  # noqa: BLE001
            pass
        return dest
    except Exception as e:  # noqa: BLE001
        logger.error("OOM memory snapshot dump failed: %s", e)
        return None