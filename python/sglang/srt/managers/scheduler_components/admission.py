# SPDX-License-Identifier: Apache-2.0
"""Derived long-context admission rule: T1 (pool) + T2 (transient) + solo guard.

The flat long-context budget (``SGLANG_LONG_CONTEXT_MAX_INFLIGHT_TOKENS``,
the sum of in-flight long-context prompt tokens) is a proxy: it caps the sum
of prompt lengths, but the 2026-10-21 composition OOM (crash #19) was NOT a
sum problem — a pool-fitting 338.5k-token in-flight sum died because the
attention staging transient scales with each candidate's length and must fit
the FRAGMENTED-contiguous free headroom, not the pool. Conversely the flat
budget's 250k floor defers JT-mandated-capacity compositions (3x100k = 300k
admits under T1) that the dynamic region serves fine.

The derived rule (posted on infra 2026-10-21, adopted by the managers):

    Admit a long-context candidate iff BOTH:
      T1 (pool):    sum(in-flight seq lens) + candidate_len <= P
      T2 (transient): k * (candidate_len + midchunk_len)
                          <= F_frag - decode_footprint
    Solo guard: when NO other long-context request is in flight, skip T2 —
      a solo candidate at the context cap fits an unfragmented dynamic
      region by construction (262,144 x 8 KiB ~= 2.05 GiB < 2.85 GiB D).

Inputs (all read at scheduler init from the worker's OWN runtime state,
never manifest-const):
  P           = max_total_num_tokens (pool, logical tokens)
  T           = 2 x chunked_prefill_size (the long-context threshold; a
                sub-T candidate stages <= k*T = 256 MiB with chunk=16,384,
                i.e. <= 9% of D — below the noise floor of F_frag itself)
  k           = SGLANG_ADMISSION_TRANSIENT_KIB_PER_TOKEN (default 8 KiB/
                token; measured on both fleets independently — 27b crash
                #19's 168,516-token candidate vs 1.29-1.41 GiB demand, and
                ornith-9b's 104,192 x 8,192 B = 853,540,864 B exactly)
  F_frag      = fragmented-contiguous floor: the LARGEST contiguous run of
                inactive allocator blocks (``memory_snapshot`` walk), not
                nominal free — on a fragmented card nominal free overstates
                the headroom by ~3x (ornith witness: 22.2 GiB reserved /
                2.8 GiB live / ~0.6 GB contiguous)
  decode_footprint = k * sum(extend-len of RUNNING non-long-context reqs)

The flat budget stays as a documented SAFE floor: when the derived rule is
disabled (T=0 or P=0), the lc-budget envs behave exactly as today.
"""

from dataclasses import dataclass
from typing import Iterable, Optional

import torch

# Default transient constant (KiB per token of extending sequence length).
# Both fleets measured ~8 KiB/token in the attention staging path.
DEFAULT_TRANSIENT_KIB_PER_TOKEN = 8


@dataclass(frozen=True)
class AdmissionInputs:
    """One admission decision's inputs, all from worker runtime state."""

    pool_tokens: int  # P = max_total_num_tokens
    candidate_len: int  # extending sequence length of the candidate
    inflight_lens: tuple  # extending lens of OTHER in-flight long-context reqs
    midchunk_len: int  # extending len of the mid-chunk (chunked) req, 0 if none
    decode_footprint_tokens: int  # extend-len sum of RUNNING sub-T reqs
    f_frag_bytes: int  # fragmented-contiguous floor (largest inactive run)
    transient_kib_per_token: int = DEFAULT_TRANSIENT_KIB_PER_TOKEN


@dataclass(frozen=True)
class AdmissionVerdict:
    """Decision + the term that failed, for /metrics and the defer log line."""

    admit: bool
    t1_pass: bool
    t2_pass: bool  # solo guard counts as pass
    reason: str  # machine-stable short tag for logs/metrics


_ADMIT_SOLO = "admit_solo"
_ADMIT_JOINT = "admit_joint"
_DEFER_T1 = "defer_pool"
_DEFER_T2 = "defer_transient"


def largest_inactive_run_bytes() -> int:
    """F_frag: the largest contiguous run of inactive allocator blocks.

    Walks ``torch.cuda.memory_snapshot()`` across all segments and returns
    the largest contiguous inactive-run size in bytes. Live blocks and
    pending-free blocks break runs — this is the largest extent a single
    CUDA allocation can actually occupy, which is what the staging
    transient needs (a single huge allocation fails against fragmented
    free even when the SUM of free bytes is large).

    Mirrors ``find_free_graph_pool_runs``'s walk semantics for the default
    pool. Best-effort: any failure returns 0 (callers treat 0 as "cannot
    measure — fall back to the flat budget floor").
    """
    try:
        snapshot = torch.cuda.memory_snapshot(include_traces=False)
    except Exception:  # noqa: BLE001 — must never break the scheduler loop
        return 0
    largest = 0
    run = 0
    for segment in snapshot or []:
        run = 0
        for block in segment.get("blocks", []):
            if block.get("state") == "inactive":
                run += block.get("size", 0)
            else:
                if run > largest:
                    largest = run
                run = 0
        if run > largest:
            largest = run
    return largest


def decide_admission(inputs: AdmissionInputs) -> AdmissionVerdict:
    """Pure T1+T2+solo decision. No I/O, no clock, no env reads."""
    k = inputs.transient_kib_per_token
    cand = inputs.candidate_len

    # T1 (pool): the sum of in-flight extending lens + the candidate must
    # fit the token pool. Cached-prefix overlap does not help here: the
    # pool is mamba-state-backed (per-request state) and the crash #19
    # composition was pool-fitting at 94.9% — T1 is the allocator-truth sum.
    inflight_sum = sum(inputs.inflight_lens)
    t1_pass = inflight_sum + cand <= inputs.pool_tokens

    # T2 (transient): the candidate's staging (plus the mid-chunk's, which
    # materializes concurrently as it runs) must fit the fragmented-
    # contiguous floor minus the RUNNING sub-T decode footprint.
    if not inputs.inflight_lens:
        # Solo guard: no other long-context request in flight — the solo
        # transient fits an unfragmented D by construction; skip T2.
        t2_pass = True
        reason = _ADMIT_SOLO if t1_pass else _DEFER_T1
        return AdmissionVerdict(
            admit=t1_pass, t1_pass=t1_pass, t2_pass=t2_pass, reason=reason
        )

    transient_demand = k * 1024 * (cand + inputs.midchunk_len)
    headroom = inputs.f_frag_bytes - k * 1024 * inputs.decode_footprint_tokens
    t2_pass = transient_demand <= headroom

    admit = t1_pass and t2_pass
    if not t1_pass:
        reason = _DEFER_T1
    elif not t2_pass:
        reason = _DEFER_T2
    else:
        reason = _ADMIT_JOINT
    return AdmissionVerdict(
        admit=admit, t1_pass=t1_pass, t2_pass=t2_pass, reason=reason
    )