# SPDX-License-Identifier: Apache-2.0
"""Hermetic (no-GPU) tests for the STALL-POOL fail-out window semantics.

The window is POSITION-keyed (the queue head has been stuck since the
spell started), not rid-keyed — wedge #6 (2026-10-21 celadon incident):
patchbay's ~300s retry cycle re-mints a fresh head rid each round, so a
rid-keyed window restarts forever and the fail-out never fires while the
queue position stays wedged. These tests exercise the arming/aging/clear
state machine on a stubbed scheduler.

The fail-out lives inside ``Scheduler.on_idle`` behind many live-state
gates (loads, metrics reporters, linkers). Rather than replicating the
whole body, these tests drive the window state machine through the same
code by calling on_idle on a heavily-stubbed scheduler: everything before
the STALL gate returns early or is neutralized.
"""

import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "python"))

import sglang.srt.managers.scheduler as sched_mod  # noqa: E402
from sglang.srt.environ import envs  # noqa: E402


def make_req(rid):
    return types.SimpleNamespace(rid=rid, retraction_count=0, finished_reason=None)


def make_stall_scheduler(monkeypatch, waiting_queue, *, running=0):
    """A scheduler stub whose on_idle STALL state machine is reachable.

    Neutralizes: idle-gating (fully_idle False), metrics reporters, load
    publishing, the STALL-DIAG/log side effects that need tree_cache attrs,
    mooncake sleep path, and the retire helper (recorded, not executed).
    """
    stub = types.SimpleNamespace()
    # The STALL-POOL state the fail-out reads and mutates.
    stub._stall_pool_watch_rid = None
    stub._stall_pool_watch_since = 0.0
    stub.waiting_queue = waiting_queue
    # The stage decorator reads this off the scheduler itself.
    stub.scheduler_stage_metrics = None
    # is_fully_idle() False -> the stall branch runs (not the idle tail).
    stub.is_fully_idle = lambda: False
    # Metrics/publish plumbing must exist but do nothing.
    stub.metrics_reporter = types.SimpleNamespace(
        record_scheduler_active=lambda: None,
        record_scheduler_idle=lambda: None,
        scheduler_stage_metrics=None,
    )
    stub._last_stall_publish_ts = float("-inf")
    stub.load_publisher = types.SimpleNamespace(publish_load_stat=lambda *a, **k: None)
    stub.load_inquirer = types.SimpleNamespace(get_loads={})
    stub.publish_load_snapshot = lambda force=False: None
    # The stall log fields: running_batch, pools, caches.
    stub.running_batch = types.SimpleNamespace(
        reqs=[object()] * running, batch_is_full=False
    )
    stub.chunked_req = None
    stub.tree_cache = types.SimpleNamespace(
        ongoing_load_back={},
        ongoing_prefetch={},
        mamba_evictable_size=lambda: 0,
        full_evictable_size=lambda: 0,
    )
    stub.req_to_token_pool = types.SimpleNamespace(
        mamba_allocator=None, available_size=lambda: 0
    )
    stub.token_to_kv_pool_allocator = types.SimpleNamespace(available_size=lambda: 0)
    stub.get_parallel = types.SimpleNamespace(
        pp_max_micro_batch_size=None
    )
    stub.enable_hicache_storage = False
    stub.enable_unified_cache_external_linker = False
    stub.enable_unified_memory = False
    stub.maybe_send_health_check_signal = lambda: None
    # Retire path: record instead of executing.
    retired = []
    stub.retire_unadmitted_request = lambda req: retired.append(req)
    stub._retired = retired
    # FINISH_ABORT must be importable inside the module namespace.
    monkeypatch.setattr(
        sched_mod,
        "FINISH_ABORT",
        lambda message=None, status_code=None: types.SimpleNamespace(
            message=message, status_code=status_code
        ),
        raising=True,
    )
    # get_parallel() is called at log time — make it a module-level callable.
    monkeypatch.setattr(
        sched_mod, "get_parallel", lambda: stub.get_parallel, raising=True
    )
    # LOAD_STALL_REFRESH_S forces the log path every call.
    monkeypatch.setattr(sched_mod, "LOAD_STALL_REFRESH_S", 0.0, raising=True)
    return stub


def run_on_idle(stub):
    sched_mod.Scheduler.on_idle(stub)


def test_rid_re_mint_does_not_reset_window(monkeypatch):
    """WEDGE #6: retries replace the head rid; the window must keep aging."""
    with envs.SGLANG_STALL_POOL_FAILOUT_SECS.override(100.0):
        # freeze time: on_idle reads time.monotonic()
        clock = {"t": 1000.0}
        monkeypatch.setattr(sched_mod.time, "monotonic", lambda: clock["t"])
        q = [make_req("rid-A")]
        stub = make_stall_scheduler(monkeypatch, q)

        run_on_idle(stub)  # t=1000: arms on rid-A
        assert stub._stall_pool_watch_rid == "rid-A"
        assert stub._stall_pool_watch_since == 1000.0

        # Client retry re-mints the head mid-window: same position, new rid.
        q[0] = make_req("rid-B")
        clock["t"] += 50.0
        run_on_idle(stub)  # t=1050: MUST keep the original window...
        assert stub._stall_pool_watch_rid == "rid-B"
        assert stub._stall_pool_watch_since == 1000.0, "position-keyed, not rid-keyed"
        assert not stub._retired

        # Window matures past the budget regardless of further re-mints.
        q[0] = make_req("rid-C")
        clock["t"] += 60.0  # total 110s > 100s budget
        run_on_idle(stub)
        assert stub._retired, "fail-out must fire through rid re-mints"
        assert stub._retired[0].rid == "rid-C"
        assert q == [], "stuck head popped from the waiting queue"
        # State cleared post-fire.
        assert stub._stall_pool_watch_rid is None


def test_running_decode_clears_the_window(monkeypatch):
    """A running decode is real progress: the spell clears and never fires."""
    with envs.SGLANG_STALL_POOL_FAILOUT_SECS.override(100.0):
        clock = {"t": 1000.0}
        monkeypatch.setattr(sched_mod.time, "monotonic", lambda: clock["t"])
        q = [make_req("rid-A")]
        stub = make_stall_scheduler(monkeypatch, q, running=1)

        run_on_idle(stub)  # running=1: the stall branch does not arm at all
        assert stub._stall_pool_watch_rid is None
        clock["t"] += 500.0
        run_on_idle(stub)
        assert not stub._retired


def test_window_rearms_after_drain(monkeypatch):
    """After a drain (queue empties), a NEW stuck head starts a fresh spell."""
    with envs.SGLANG_STALL_POOL_FAILOUT_SECS.override(100.0):
        clock = {"t": 1000.0}
        monkeypatch.setattr(sched_mod.time, "monotonic", lambda: clock["t"])
        q = [make_req("rid-A")]
        stub = make_stall_scheduler(monkeypatch, q)

        run_on_idle(stub)
        assert stub._stall_pool_watch_since == 1000.0

        # Queue drains (head served elsewhere): state must clear.
        q.pop(0)
        run_on_idle(stub)
        assert stub._stall_pool_watch_rid is None

        # New arrival later: fresh spell from NOW, not from the old arm time.
        clock["t"] += 999.0
        q.append(make_req("rid-D"))
        run_on_idle(stub)
        assert stub._stall_pool_watch_rid == "rid-D"
        assert stub._stall_pool_watch_since == 1999.0
        assert not stub._retired, "no stale window fires on a fresh head"