"""Hermetic (no-GPU) tests for the scheduler OOM guard.

Covers the strike/plan logic (scheduler_components/oom_guard.py, pure), the
shed mechanism and the mid-forward backstop on Scheduler (stubbed self), and
the process_batch_result empty-batch contract the shed relies on.
"""

import sys
import types
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "python"))

import sglang.srt.managers.scheduler as sched_mod  # noqa: E402
from sglang.srt.managers.scheduler_components.oom_guard import (  # noqa: E402
    plan_oom_shed,
)
from sglang.srt.managers.utils import GenerationBatchResult  # noqa: E402
from sglang.srt.model_executor.forward_batch_info import ForwardMode  # noqa: E402


def make_req(rid, finished=False, beam_group=None, output_ids=None):
    return types.SimpleNamespace(
        rid=rid,
        finished=lambda: finished,
        beam_group=beam_group,
        to_finish=None,
        output_ids=output_ids or [],
    )


# ---------------------------------------------------------------------------
# plan_oom_shed (pure)
# ---------------------------------------------------------------------------


def test_plan_requeues_and_strikes_fresh_rids():
    plan = plan_oom_shed(
        rids=["a", "b"], finished_rids=set(), beam_rids=set(), strikes={}, cap=3
    )
    assert plan.requeue_rids == ["a", "b"]
    assert plan.abort_rids == []
    assert plan.post_stripes == {"a": 1, "b": 1}


def test_plan_hard_sheds_after_cap_and_pops_entry():
    plan = plan_oom_shed(
        rids=["a"], finished_rids=set(), beam_rids=set(), strikes={"a": 3}, cap=3
    )
    assert plan.requeue_rids == []
    assert plan.abort_rids == ["a"]
    assert "a" not in plan.post_stripes


def test_plan_increments_from_existing_strike():
    plan = plan_oom_shed(
        rids=["a"], finished_rids=set(), beam_rids=set(), strikes={"a": 2}, cap=3
    )
    assert plan.requeue_rids == ["a"]
    assert plan.post_stripes == {"a": 3}


def test_plan_skips_finished_requests_entirely():
    plan = plan_oom_shed(
        rids=["a", "b"],
        finished_rids={"a"},
        beam_rids=set(),
        strikes={},
        cap=3,
    )
    assert plan.requeue_rids == ["b"]
    assert plan.abort_rids == []
    assert plan.post_stripes == {"b": 1}


def test_plan_sheds_beam_groups_without_strike():
    plan = plan_oom_shed(
        rids=["a", "b"],
        finished_rids=set(),
        beam_rids={"a"},
        strikes={"a": 1},
        cap=3,
    )
    assert plan.requeue_rids == ["b"]
    assert plan.abort_rids == ["a"]
    assert "a" not in plan.post_stripes  # beam shed does not strike


def test_plan_never_hard_sheds_streamed_requests():
    # The 2026-10-20 incident: mid-stream aborts are SSE error frames inside
    # an already-committed HTTP 200 — no classifier can intercept them, so a
    # streamed request must never hit the strike-cap hard-shed.
    plan = plan_oom_shed(
        rids=["a", "b"],
        finished_rids=set(),
        beam_rids=set(),
        strikes={"a": 3, "b": 3},
        cap=3,
        streamed_rids={"a"},
    )
    # "a" streamed -> requeues forever under sustained exhaustion (latency,
    # not a client-facing crash surface). "b" never streamed -> hard-sheds.
    assert plan.requeue_rids == ["a"]
    assert plan.abort_rids == ["b"]
    assert plan.post_stripes == {"a": 4}


def test_plan_streamed_rids_not_in_batch_is_noop():
    plan = plan_oom_shed(
        rids=["a"],
        finished_rids=set(),
        beam_rids=set(),
        strikes={"a": 3},
        cap=3,
        streamed_rids={"z"},  # not in this batch
    )
    assert plan.abort_rids == ["a"]


# ---------------------------------------------------------------------------
# Scheduler._shed_batch_for_oom on a stubbed self
# ---------------------------------------------------------------------------


class Recorder:
    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))


def make_stub(tp_size=1):
    stub = types.SimpleNamespace()
    stub.ps = types.SimpleNamespace(tp_size=tp_size, tp_rank=0)
    stub.oom_shed_strikes = {}
    stub.req_to_token_pool = object()
    stub.token_to_kv_pool_allocator = object()
    stub.hisparse_coordinator = None
    stub.tree_cache = object()
    stub.disaggregation_mode = "null"
    stub.chunked_req = None
    stub._add_request_to_queue = Recorder()
    stub.beam_coordinator = types.SimpleNamespace(retire_group=Recorder())
    stub.ipc_channels = types.SimpleNamespace(
        send_to_tokenizer=types.SimpleNamespace(send_output=Recorder())
    )
    return stub


def make_batch(mode, reqs, decoding_reqs=None, chunked_req=None):
    b = types.SimpleNamespace(
        forward_mode=mode,
        reqs=reqs,
        decoding_reqs=decoding_reqs,
        chunked_req=chunked_req,
        batch_is_full=True,
    )
    return b


@pytest.fixture()
def scheduler_module(monkeypatch):
    monkeypatch.setattr(sched_mod, "retract_all", Recorder(), raising=True)
    monkeypatch.setattr(
        sched_mod,
        "_make_abort_req",
        lambda req, finished_reason=None: ("abort", req.rid),
        raising=True,
    )
    monkeypatch.setattr(sched_mod, "free_member_rows", Recorder(), raising=True)
    return sched_mod


def test_shed_releases_live_and_requeues(scheduler_module):
    r0, r1, r2 = (
        make_req("a"),
        make_req("b", finished=True),
        make_req("c"),
    )
    stub = make_stub()
    batch = make_batch(ForwardMode.EXTEND, [r0, r1, r2])
    out = scheduler_module.Scheduler._shed_batch_for_oom(
        stub, batch, free_mb=100.0, cause="gate"
    )
    # finished request excluded from the release (previous batch's result
    # processing owns its KV) and not requeued.
    released = scheduler_module.retract_all.calls[0][1]["reqs"]
    assert [r.rid for r in released] == ["a", "c"]
    requeued = [c[0][0].rid for c in stub._add_request_to_queue.calls]
    assert requeued == ["a", "c"]
    assert all(c[1].get("is_retracted") for c in stub._add_request_to_queue.calls)
    # batch emptied in place; safe default result for the empty batch
    assert batch.reqs == []
    assert batch.decoding_reqs == []
    assert batch.batch_is_full is False
    assert out.next_token_ids is None


def test_shed_hard_sheds_at_strike_cap(scheduler_module):
    stub = make_stub()
    stub.oom_shed_strikes = {"a": 3}
    req = make_req("a")
    batch = make_batch(ForwardMode.EXTEND, [req])
    scheduler_module.Scheduler._shed_batch_for_oom(
        stub, batch, free_mb=90.0, cause="gate"
    )
    # aborted: 503 finish sent, no requeue, ledger cleaned
    send_calls = stub.ipc_channels.send_to_tokenizer.send_output.calls
    assert len(send_calls) == 1
    assert "OOM-shed" in req.to_finish.message
    assert stub._add_request_to_queue.calls == []
    assert scheduler_module.retract_all.calls[0][1]["reqs"] == [req]
    assert stub.oom_shed_strikes == {}


def test_shed_carried_decode_reqs_and_clears_chunked_req(scheduler_module):
    stub = make_stub()
    stub.chunked_req = make_req("chunked")
    prefill_req = make_req("p")
    carried = make_req("d")
    batch = make_batch(
        ForwardMode.EXTEND, [prefill_req], decoding_reqs=[carried]
    )
    out = scheduler_module.Scheduler._shed_batch_for_oom(
        stub, batch, free_mb=80.0, cause="gate"
    )
    released = scheduler_module.retract_all.calls[0][1]["reqs"]
    assert {r.rid for r in released} == {"p", "d"}
    requeued = {c[0][0].rid for c in stub._add_request_to_queue.calls}
    assert requeued == {"p", "d"}
    # mid-chunk request was released like any other; the scheduler's
    # chunked_req pointer must drop with it.
    assert stub.chunked_req is None
    assert batch.reqs == []
    assert batch.decoding_reqs == []


def test_shed_aborts_beam_group_with_member_rows_freed(scheduler_module):
    group = object()
    req = make_req("beam", beam_group=group)
    stub = make_stub()
    batch = make_batch(ForwardMode.EXTEND, [req])
    out = scheduler_module.Scheduler._shed_batch_for_oom(
        stub, batch, free_mb=70.0, cause="gate"
    )
    # member rows are freed before the leader release (retract_decode order)
    free_calls = scheduler_module.free_member_rows.calls
    assert len(free_calls) == 1
    released = scheduler_module.retract_all.calls[0][1]["reqs"]
    assert [r.rid for r in released] == ["beam"]
    retire_calls = stub.beam_coordinator.retire_group.calls
    assert len(retire_calls) == 1 and retire_calls[0][0][0] is req
    send_calls = stub.ipc_channels.send_to_tokenizer.send_output.calls
    assert len(send_calls) == 1


# ---------------------------------------------------------------------------
# Scheduler._shed_batch_on_forward_oom (backstop)
# ---------------------------------------------------------------------------


@pytest.fixture()
def fake_cuda(monkeypatch):
    import torch

    monkeypatch.setattr(
        torch.cuda,
        "mem_get_info",
        lambda *a, **k: (268_435_456, 8_589_934_592),  # 256MB free, 8GB total
        raising=True,
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None, raising=True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda *a, **k: None, raising=True)


# ---------------------------------------------------------------------------
# device_free_mb_allocator_aware: the gate metric (pins the incident numbers)
# ---------------------------------------------------------------------------


def test_free_metric_counts_reserved_unused_segments(monkeypatch):
    # The 2026-10-20 incident shape: driver free 72MB while the allocator
    # holds ~2.78GB reserved-but-reusable. The gate must read ~2.85GB —
    # what an empty_cache() flush would release — not the raw 72MB.
    import torch

    from sglang.srt.managers.scheduler_components.oom_guard import (
        device_free_mb_allocator_aware,
    )

    driver_free = 72 * 1024 * 1024
    reserved = 2780 * 1024 * 1024
    allocated = 12 * 1024 * 1024
    monkeypatch.setattr(
        torch.cuda, "mem_get_info", lambda *a, **k: (driver_free, 24 * 1024**3)
    )
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda *a, **k: reserved)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *a, **k: allocated)

    got = device_free_mb_allocator_aware()
    assert int(got) == 2840  # 72 + (2780 - 12) = 2840MB, not 72MB

    # Inverse shape: no reusable segments -> metric == raw free exactly.
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda *a, **k: allocated)
    got = device_free_mb_allocator_aware()
    assert int(got) == 72


def test_free_metric_never_drops_below_raw_free(monkeypatch):
    # allocated > reserved (sub-allocator edge) must clamp at 0, never go
    # negative and subtract from real free.
    import torch

    from sglang.srt.managers.scheduler_components.oom_guard import (
        device_free_mb_allocator_aware,
    )

    monkeypatch.setattr(
        torch.cuda, "mem_get_info", lambda *a, **k: (512 * 1024**2, 8 * 1024**3)
    )
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda *a, **k: 0)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *a, **k: 2**30)
    got = device_free_mb_allocator_aware()
    assert int(got) == 512


def test_forward_catch_tp1_recovers_and_sheds(scheduler_module, fake_cuda):
    stub = make_stub(tp_size=1)
    stub._shed_batch_for_oom = (
        lambda batch, free_mb, cause: scheduler_module.Scheduler._shed_batch_for_oom(
            stub, batch, free_mb=free_mb, cause=cause
        )
    )
    batch = make_batch(ForwardMode.DECODE, [make_req("d1")])
    out = scheduler_module.Scheduler._shed_batch_on_forward_oom(stub, batch)
    assert isinstance(out, GenerationBatchResult)
    assert out.next_token_ids is None
    assert stub._add_request_to_queue.calls[-1][0][0].rid == "d1"
    assert batch.reqs == []


def test_forward_catch_tp2_reraises_without_shed(scheduler_module, fake_cuda):
    stub = make_stub(tp_size=2)
    stub._shed_batch_for_oom = (
        lambda batch, free_mb, cause: scheduler_module.Scheduler._shed_batch_for_oom(
            stub, batch, free_mb=free_mb, cause=cause
        )
    )
    batch = make_batch(ForwardMode.DECODE, [make_req("d1")])
    try:
        raise torch.OutOfMemoryError("simulated")
    except torch.OutOfMemoryError:
        with pytest.raises(RuntimeError):
            scheduler_module.Scheduler._shed_batch_on_forward_oom(stub, batch)
    # no recovery actions on a TP group — the process takes the crash path
    assert stub._add_request_to_queue.calls == []


# ---------------------------------------------------------------------------
# process_batch_result empty-batch contract
# ---------------------------------------------------------------------------


def test_process_batch_result_empty_batch_is_housekeeping_only(monkeypatch):
    stub = types.SimpleNamespace()
    stub.scheduler_stage_metrics = None  # decorated wrapper: metrics off
    stub.tree_cache = types.SimpleNamespace(flush_pending_backups=Recorder())
    stub.maybe_send_health_check_signal = Recorder()
    stub.publish_load_snapshot = Recorder()
    stub.load_publisher = types.SimpleNamespace(publish_load_stat=Recorder())
    stub.load_inquirer = types.SimpleNamespace(get_loads=object())
    stub.batch_result_processor = types.SimpleNamespace(
        process_batch_result_decode=Recorder(),
        process_batch_result_prefill=Recorder(),
    )
    monkeypatch.setattr(sched_mod, "flush_trace_batch", Recorder(), raising=True)

    batch = make_batch(ForwardMode.DECODE, [])
    batch.is_empty = lambda: True
    sched_mod.Scheduler.process_batch_result.__wrapped__(stub, batch, GenerationBatchResult())

    # no mode dispatch ran; housekeeping preserved
    assert stub.batch_result_processor.process_batch_result_decode.calls == []
    assert stub.batch_result_processor.process_batch_result_prefill.calls == []
    assert len(stub.tree_cache.flush_pending_backups.calls) == 1
    assert len(stub.maybe_send_health_check_signal.calls) == 1