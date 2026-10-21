# SPDX-License-Identifier: Apache-2.0
"""Hermetic (no-GPU) tests for the derived long-context admission rule.

Covers the pure decision (T1 pool + T2 transient + solo guard,
scheduler_components/admission.py) and the fragmented-contiguous floor
walk (``largest_inactive_run_bytes``) against stubbed snapshot data — the
cross-check compositions from the infra derivation note (2026-10-21) are
pinned as tests, including the crash-#19 fatal composition the rule must
defer and the JT-invariant-2 composition (3x100k) the flat 250k floor
mis-defers but the derived rule admits.
"""

import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "python"))

from sglang.srt.managers.scheduler_components.admission import (  # noqa: E402
    DEFAULT_TRANSIENT_KIB_PER_TOKEN,
    AdmissionInputs,
    decide_admission,
    largest_inactive_run_bytes,
)

KIB = 1024
P = 356_096  # pool: max_total_num_tokens (both 27b workers)
T = 32_768  # long-context threshold = 2 x chunked_prefill_size
CTX = 262_144  # context cap
# F_frag at a healthy, unfragmented moment (~2.8 GiB largest contiguous run,
# matching the fleet's measured boot-time state).
F_FRAG_HEALTHY = int(2.8 * 2**30)


def inputs(**kw):
    base = dict(
        pool_tokens=P,
        candidate_len=100_000,
        inflight_lens=(),
        midchunk_len=0,
        decode_footprint_tokens=0,
        f_frag_bytes=F_FRAG_HEALTHY,
        transient_kib_per_token=DEFAULT_TRANSIENT_KIB_PER_TOKEN,
    )
    base.update(kw)
    return AdmissionInputs(**base)


# ---------------------------------------------------------------------------
# Solo guard
# ---------------------------------------------------------------------------


def test_solo_at_context_cap_admits_t1_pass():
    """JT invariant 1: a solo 262,144 candidate is legal — T2 skipped."""
    v = decide_admission(inputs(candidate_len=CTX, inflight_lens=()))
    assert v.admit and v.t1_pass and v.t2_pass
    assert v.reason == "admit_solo"


def test_solo_deferred_when_pool_full():
    """Even solo, T1 binds: nothing admits past the token pool."""
    v = decide_admission(
        inputs(candidate_len=100_000, inflight_lens=(300_000,), pool_tokens=P)
    )
    # Not solo (one long in flight) — but also check the true solo+overfull-pool
    # case: a single request longer than the pool itself.
    v2 = decide_admission(inputs(candidate_len=P + 1, inflight_lens=()))
    assert not v2.admit and not v2.t1_pass
    assert v2.reason == "defer_pool"


# ---------------------------------------------------------------------------
# T1 (pool)
# ---------------------------------------------------------------------------


def test_t1_admits_three_way_100k_overlap():
    """JT invariant 2 / cross-check: 2x100k in flight + 100k candidate.
    Flat 250k floor DEFERS this (300k sum); T1 admits (300k <= 356k) and
    T2 passes on a healthy F_frag (0.78 GiB demand)."""
    v = decide_admission(
        inputs(
            candidate_len=100_000,
            inflight_lens=(100_000, 100_000),
        )
    )
    assert v.admit and v.t1_pass and v.t2_pass
    assert v.reason == "admit_joint"


def test_t1_defers_pool_overflow():
    v = decide_admission(
        inputs(
            candidate_len=150_000,
            inflight_lens=(150_000, 150_000),  # 450k > 356,096
        )
    )
    assert not v.admit and not v.t1_pass
    assert v.reason == "defer_pool"


# ---------------------------------------------------------------------------
# T2 (transient) — the crash-#19 term
# ---------------------------------------------------------------------------


def test_t2_defers_the_crash19_fatal_composition():
    """The 2026-10-21 composition OOM: 338.5k in-flight sum was POOL-FITTING
    (T1 passes at 94.9%) but the 168,516-token candidate's 1.29 GiB
    staging demand exceeded the ~0.83 GiB fragmented headroom. The derived
    rule must defer EXACTLY here — this is the composition the flat
    budget could not see."""
    # Reconstructed at the fatal moment: two longs in flight, fragmented
    # F_frag collapsed to 0.83 GiB, decode footprint already resident.
    f_frag_fatal = int(0.83 * 2**30)
    decode_fp = 30_000  # sub-T running decode extend-len sum
    cand_len = 168_516
    # T2 demand: 8 KiB * 168,516 ~= 1.29 GiB > 0.83 GiB - 8KiB*30k ~= 0.60 GiB
    v = decide_admission(
        inputs(
            candidate_len=cand_len,
            inflight_lens=(170_000,),
            f_frag_bytes=f_frag_fatal,
            decode_footprint_tokens=decode_fp,
        )
    )
    assert v.t1_pass, "crash #19 was pool-fitting — T1 must pass"
    assert not v.t2_pass
    assert not v.admit
    assert v.reason == "defer_transient"


def test_t2_admits_200k_plus_60k_on_healthy_f_frag():
    """Cross-check from the derivation note: T1 260k pass, T2 1.53 GiB vs
    boot-healthy F_frag ~2.8 GiB -> ADMIT (flat 250k DEFERS this too)."""
    v = decide_admission(
        inputs(
            candidate_len=200_000,
            inflight_lens=(60_000,),
        )
    )
    assert v.admit and v.t2_pass


def test_t2_counts_midchunk_transient():
    """The mid-chunk request's transient materializes concurrently — its
    length must add to the candidate's demand. With F_frag tight, a
    candidate that fits alone must defer when a long mid-chunk is in
    flight."""
    cand = 120_000
    mid = 120_000
    # Headroom: exactly k*(cand)+1 with no midchunk -> admits without it.
    headroom = DEFAULT_TRANSIENT_KIB_PER_TOKEN * KIB * cand + 1
    v_alone = decide_admission(
        inputs(candidate_len=cand, inflight_lens=(1,), f_frag_bytes=headroom)
    )
    assert v_alone.admit, "sanity: fits without the midchunk term"
    v_mid = decide_admission(
        inputs(
            candidate_len=cand,
            inflight_lens=(1,),
            midchunk_len=mid,
            f_frag_bytes=headroom,
        )
    )
    assert not v_mid.t2_pass and not v_mid.admit
    assert v_mid.reason == "defer_transient"


def test_t2_subtracts_decode_footprint():
    """Resident sub-T decode requests consume the same dynamic region —
    their extend-len sum must shrink the T2 headroom."""
    cand = 100_000
    headroom = DEFAULT_TRANSIENT_KIB_PER_TOKEN * KIB * cand + 1
    v_clean = decide_admission(
        inputs(candidate_len=cand, inflight_lens=(1,), f_frag_bytes=headroom)
    )
    assert v_clean.admit, "sanity: admits with an empty decode footprint"
    v_loaded = decide_admission(
        inputs(
            candidate_len=cand,
            inflight_lens=(1,),
            decode_footprint_tokens=1,  # one token over the line
            f_frag_bytes=headroom,
        )
    )
    assert not v_loaded.t2_pass and v_loaded.reason == "defer_transient"


def test_t2_binds_per_card_f_frag_not_nominal_free():
    """Fragmented headroom, not nominal free, is what T2 trusts: a
    candidate whose transient fits nominal-free but not the fragmented
    floor must defer (ornith's ~3x-overstatement caveat)."""
    cand = 200_000
    # Nominal free 2.5 GiB would fit 1.53 GiB; fragmented floor 1.2 GiB does not.
    v = decide_admission(
        inputs(
            candidate_len=cand,
            inflight_lens=(60_000,),
            f_frag_bytes=int(1.2 * 2**30),
        )
    )
    assert not v.t2_pass and v.reason == "defer_transient"


# ---------------------------------------------------------------------------
# largest_inactive_run_bytes (F_frag walk)
# ---------------------------------------------------------------------------


def _seg(*blocks):
    return {"blocks": list(blocks)}


def _blk(state, size):
    return {"state": state, "size": size}


class FakeSnapshot:
    def __init__(self, segments):
        self.segments = segments

    def __call__(self, include_traces=False):
        return self.segments


def _fake_torch_with(snapshot_fn):
    """torch stub matching the walk's call: torch.cuda.memory_snapshot(...)."""
    return types.SimpleNamespace(
        cuda=types.SimpleNamespace(memory_snapshot=snapshot_fn)
    )


def test_walk_finds_largest_contiguous_inactive_run(monkeypatch):
    fake_torch = _fake_torch_with(
        FakeSnapshot(
            [
                _seg(_blk("active", 100), _blk("inactive", 300), _blk("inactive", 500)),
                _seg(_blk("inactive", 400), _blk("active", 1), _blk("inactive", 700)),
            ]
        )
    )
    monkeypatch.setattr(
        "sglang.srt.managers.scheduler_components.admission.torch", fake_torch
    )
    assert largest_inactive_run_bytes() == 800  # 300+500 beats 400 and 700


def test_walk_live_blocks_break_runs(monkeypatch):
    """A pending-free (split) block mid-run breaks contiguity — the walk
    must NOT bridge across it (a single huge allocation would fail there)."""
    fake_torch = _fake_torch_with(
        FakeSnapshot(
            [
                _seg(
                    _blk("inactive", 600),
                    _blk("pending-free_0", 16),
                    _blk("inactive", 600),
                )
            ]
        )
    )
    monkeypatch.setattr(
        "sglang.srt.managers.scheduler_components.admission.torch", fake_torch
    )
    assert largest_inactive_run_bytes() == 600


def test_walk_returns_zero_on_snapshot_failure(monkeypatch):
    def boom(include_traces=False):
        raise RuntimeError("device busy")

    fake_torch = _fake_torch_with(boom)
    monkeypatch.setattr(
        "sglang.srt.managers.scheduler_components.admission.torch", fake_torch
    )
    assert largest_inactive_run_bytes() == 0