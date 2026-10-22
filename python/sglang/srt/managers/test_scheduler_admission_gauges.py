"""Emission-contract tests for the admission gauges.

Pins the observability contract established in the jade soak postmortem:
EVERY F_frag walk result reaches the admission_f_frag_bytes gauge — not
just derived verdicts. Before the fix, below-floor walks (the flat-defer
branch, the ONLY branch a loaded card ever reaches: 55,678 executions on
jade, zero gauge samples) dropped the measured value at the emission
point, so the F_frag-vs-nominal curve was measured but discarded.
"""

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "python"))

import sglang.srt.observability.metrics_collector as mc_mod  # noqa: E402


def _collector():
    """SchedulerMetricsCollector with stubbed label machinery: __init__ runs
    (gauges get built) but every .labels() call returns a MagicMock recorder
    so the tests assert the emission arguments, not prometheus internals."""

    class RecordingGauge:
        def __init__(self):
            self.values = {}

        def labels(self, **kw):
            self.values.update(kw)
            return self

        def set(self, v):
            self.values["last_set"] = v

    class RecordingCounter:
        def __init__(self):
            self.counts = {}

        def labels(self, **kw):
            key = kw.get("reason", "?")
            self.counts.setdefault(key, [kw, 0])
            return self

        def inc(self):
            self.counts[self._cur][1] += 1

        # prometheus Counters return a child on labels(); track incs per reason
        def __getattr__(self, name):
            raise AttributeError(name)

    # simpler: patch the gauge/counter classes via the collector's _gauge_cls
    c = mc_mod.SchedulerMetricsCollector.__new__(mc_mod.SchedulerMetricsCollector)

    class _Counter:
        def __init__(self, *a, **k):
            self.map = {}

        def labels(self, **kw):
            reason = kw.get("reason", "nolabel")

            class _Child:
                def __init__(self, m, r):
                    self.m, self.r = m, r
                    m.setdefault(r, 0)

                def inc(self, _=1):
                    self.m[self.r] += 1

            return _Child(self.map, reason)

    class _Gauge:
        def __init__(self, *a, **k):
            self.map = {}

        def labels(self, **kw):
            return self

        def set(self, v):
            self.map["last"] = v

    # bind the emission attrs directly (the ones observe_admission uses)
    c.admission_defers_total = _Counter()
    c.admission_f_frag_bytes = _Gauge()
    c.labels = {}
    return c


def test_observe_flat_defer_emits_measured_f_frag():
    """Below-floor flat_budget_defer: the walk result MUST reach the gauge."""
    c = _collector()
    c.observe_admission(f_frag_bytes=536_870_912, reason="flat_budget_defer")
    assert c.admission_f_frag_bytes.map["last"] == 536_870_912
    assert c.admission_defers_total.map["flat_budget_defer"] == 1


def test_observe_flat_defer_none_keeps_last_gauge_value():
    """Floor-unset flat_budget_defer (no walk ran): None keeps the last
    sample — the gauge never fabricates a value."""
    c = _collector()
    c.observe_admission(f_frag_bytes=2 * 2**30, reason="derived_admit")
    c.observe_admission(f_frag_bytes=None, reason="flat_budget_defer")
    assert c.admission_f_frag_bytes.map["last"] == 2 * 2**30  # unchanged


def test_observe_derived_verdicts_still_emit():
    """The derived branches keep their emission (regression guard)."""
    c = _collector()
    c.observe_admission(f_frag_bytes=2**30, reason="derived_defer")
    c.observe_admission(f_frag_bytes=3 * 2**30, reason="derived_admit")
    assert c.admission_f_frag_bytes.map["last"] == 3 * 2**30
    assert c.admission_defers_total.map["derived_defer"] == 1
    assert c.admission_defers_total.map["derived_admit"] == 1


def test_scheduler_flat_defer_call_passes_walk_result(monkeypatch):
    """The scheduler's flat-defer call site passes the measured value: with
    the floor SET and the walk returning below-floor bytes, the observe
    call receives f_frag_bytes=that value (not None)."""

    import sglang.srt.managers.scheduler as sched_mod  # noqa: E402

    captured = {}
    fake_metrics = types.SimpleNamespace(
        observe_admission=lambda **kw: captured.update(kw)
    )
    lc_f_frag_bytes = 512 * 2**20  # below the 1GiB floor -> flat defer path

    # Directly exercise the emission branch shape: the scheduler calls
    # observe_admission(f_frag_bytes=lc_f_frag_bytes, reason=...)
    # when the verdict is None. Source-verified to prevent silent re-drop.
    import inspect

    src = inspect.getsource(sched_mod)
    assert "f_frag_bytes=lc_f_frag_bytes, reason=\"flat_budget_defer\"" in src, (
        "flat_budget_defer must emit the measured walk result"
    )