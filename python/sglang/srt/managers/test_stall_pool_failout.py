"""Hermetic (no-GPU) tests for the STALL-POOL fail-out.

Covers the queue-head stuck detector in Scheduler.on_idle's STALL-POOL
spin block: the 2026-10-19 wedge class (same head rid spinning in
admission NO_TOKEN with running=0 until a manual pod bounce, x7 in
100 minutes on the 27B fleet) must retire the head to the client once the
budget elapses, and must never fire while any progress signal exists
(a running decode, a chunked prefill, an in-flight load, or a head that
changes).

The fail-out pops the head from the waiting queue and calls
Scheduler.retire_unadmitted_request (the proven disaggregation rejection
path). These tests stub the scheduler the same way
test_scheduler_oom_guard.py does and drive on_idle's not-fully-idle
branch in isolation. The wall clock is patched at
``sched_mod.time.monotonic`` because that is what both the publish
throttle and the fail-out budget read in production.
"""

import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "python"))

import sglang.srt.managers.scheduler as sched_mod  # noqa: E402
from sglang.srt.environ import envs  # noqa: E402


def make_req(rid):
    return types.SimpleNamespace(
        rid=rid,
        session=None,
        finished=lambda: False,
        finished_reason=None,
        to_finish=None,
    )


class Recorder:
    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))


def make_stub():
    """A scheduler stub shaped for the STALL-POOL block in on_idle."""
    stub = types.SimpleNamespace()
    stub.waiting_queue = []
    stub.running_batch = types.SimpleNamespace(reqs=[], batch_is_full=False)
    stub.chunked_req = None
    stub.tree_cache = types.SimpleNamespace(
        ongoing_load_back={}, ongoing_prefetch={}, linker=None
    )
    stub.req_to_token_pool = types.SimpleNamespace(mamba_allocator=None)
    stub.token_to_kv_pool_allocator = types.SimpleNamespace(
        available_size=lambda: 0
    )
    stub.metrics_reporter = types.SimpleNamespace(
        record_scheduler_active=Recorder(), record_scheduler_idle=Recorder()
    )
    stub._last_stall_publish_ts = float("-inf")
    stub._stall_pool_watch_rid = None
    stub._stall_pool_watch_since = 0.0
    stub.is_fully_idle = lambda: False
    stub.maybe_send_health_check_signal = Recorder()
    stub.publish_load_snapshot = lambda force=True: None
    stub.load_publisher = types.SimpleNamespace(publish_load_stat=Recorder())
    stub.load_inquirer = types.SimpleNamespace(get_loads=lambda: None)
    stub.enable_hicache_storage = False
    stub.enable_unified_cache_external_linker = False
    stub.scheduler_stage_metrics = None  # stage decorator passes through
    stub.retire_unadmitted_request = Recorder()
    return stub


@pytest.fixture()
def spin_clock(monkeypatch):
    """Patch the wall clock on_idle reads; yields the mutable clock dict."""
    clock = {"t": 0.0}
    monkeypatch.setattr(sched_mod.time, "monotonic", lambda: clock["t"])
    return clock


@pytest.fixture()
def spin_env(monkeypatch):
    """Patch get_parallel (runtime-context) + arm a 0.5s fail-out budget."""
    monkeypatch.setattr(
        sched_mod,
        "get_parallel",
        lambda: types.SimpleNamespace(pp_max_micro_batch_size=0),
        raising=False,
    )
    with envs.SGLANG_STALL_POOL_FAILOUT_SECS.override(0.5):
        yield


def run_spin_once(sched):
    """Drive one on_idle pass; the publish throttle fires every call."""
    sched._last_stall_publish_ts = float("-inf")
    sched_mod.Scheduler.on_idle(sched)


def test_same_head_past_budget_is_retired(spin_clock, spin_env):
    sched = make_stub()
    req = make_req("wedged-rid")
    sched.waiting_queue.append(req)
    run_spin_once(sched)
    # First tick: window opens, nothing retires yet.
    assert sched.retire_unadmitted_request.calls == []
    assert sched._stall_pool_watch_rid == "wedged-rid"
    spin_clock["t"] = 1.0  # past the 0.5s budget with the same head
    run_spin_once(sched)
    assert len(sched.retire_unadmitted_request.calls) == 1
    # The head was popped before retiring, and the retired req is the head.
    assert sched.waiting_queue == []
    assert sched.retire_unadmitted_request.calls[0][0][0] is req
    # Window reset for the next head.
    assert sched._stall_pool_watch_rid is None


def test_running_decode_never_fails_out(spin_clock, spin_env):
    sched = make_stub()
    sched.running_batch.reqs = [make_req("predecessor")]  # flavor B shape
    sched.waiting_queue.append(make_req("wedged-rid"))
    spin_clock["t"] = 100.0  # arbitrarily long
    run_spin_once(sched)
    assert sched.retire_unadmitted_request.calls == []
    assert sched._stall_pool_watch_rid is None


def test_in_flight_load_resets_window(spin_clock, spin_env):
    sched = make_stub()
    sched.waiting_queue.append(make_req("wedged-rid"))
    run_spin_once(sched)
    assert sched._stall_pool_watch_rid == "wedged-rid"
    # A load-back appears (progress): window must clear, no retire.
    sched.tree_cache.ongoing_load_back = {"batch-0": object()}
    spin_clock["t"] = 100.0
    run_spin_once(sched)
    assert sched.retire_unadmitted_request.calls == []
    assert sched._stall_pool_watch_rid is None


def test_head_change_restarts_budget(spin_clock, spin_env):
    sched = make_stub()
    sched.waiting_queue.append(make_req("first-rid"))
    run_spin_once(sched)
    spin_clock["t"] = 1.0  # past budget for the first head
    # Head changes (e.g. priority requeue) before the fail-out lands.
    sched.waiting_queue[0] = make_req("second-rid")
    run_spin_once(sched)
    assert sched.retire_unadmitted_request.calls == []
    assert sched._stall_pool_watch_rid == "second-rid"
    assert sched._stall_pool_watch_since == 1.0


def test_disabled_never_retires(spin_clock, monkeypatch):
    monkeypatch.setattr(
        sched_mod,
        "get_parallel",
        lambda: types.SimpleNamespace(pp_max_micro_batch_size=0),
        raising=False,
    )
    with envs.SGLANG_STALL_POOL_FAILOUT_SECS.override(-1.0):
        sched = make_stub()
        sched.waiting_queue.append(make_req("wedged-rid"))
        run_spin_once(sched)
        spin_clock["t"] = 10_000.0
        run_spin_once(sched)
        assert sched.retire_unadmitted_request.calls == []
        assert sched._stall_pool_watch_rid is None


def test_chunked_prefill_blocks_failout(spin_clock, spin_env):
    sched = make_stub()
    sched.chunked_req = make_req("chunking-rid")
    sched.waiting_queue.append(make_req("wedged-rid"))
    spin_clock["t"] = 100.0
    run_spin_once(sched)
    assert sched.retire_unadmitted_request.calls == []
    assert sched._stall_pool_watch_rid is None