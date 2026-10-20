"""Plan an OOM shed: which requests get requeued and which are hard-shed.

The scheduler sheds a batch that cannot fit in real device memory (the
pre-forward gate, or the mid-forward catch). Before touching pools it needs a
rank-symmetric plan — every TP rank must release/requeue exactly the same
requests — and the plan must be a pure function of per-request state plus the
strike history, so it can be unit-tested without a GPU or a process group.

The strike ledger lives on the Scheduler (one entry per request rid): a shed
request that later gets admitted clears its strike; a request shed
``SGLANG_OOM_SHED_MAX_STRIKES`` times without fitting is hard-shed to the
client (503) instead of looping the waiting queue.
"""

from dataclasses import dataclass
from typing import Dict, Iterable

import torch

from sglang.srt.environ import envs


@dataclass(frozen=True)
class ShedPlan:
    """Outcome of planning an OOM shed for one batch.

    ``requeue_rids``: released requests requeued for (re)admission, in
    release order. ``abort_rids``: requests hard-shed to the client (strike
    cap) or that cannot be retracted at all (beam groups) — the caller sends
    the 503 finish + retires any beam group. ``post_stripes``: the strike
    ledger after this shed (only shed requests are touched; cleared entries
    are dropped, not zeroed).
    """

    requeue_rids: list
    abort_rids: list
    post_stripes: Dict[str, int]


def device_free_mb_allocator_aware() -> float:
    """Device free memory (MB) including the allocator's unused reserved
    segments — the hot-path equivalent of boot-time "avail mem", which
    sglang measures as raw free *after* an ``empty_cache()`` flush.

    Reading raw ``mem_get_info`` free alone is wrong in the hot path: the
    caching allocator holds freed blocks as reserved-but-reusable segments
    and does not return them to the driver. Raw free can sit at ~70MB while
    effective free is ~2.9GB (the segments a flush would release) — the
    2026-10-20 gate incident: a threshold read against raw free fired on
    every batch and hard-shed all traffic.

    Equivalent to post-``empty_cache()`` free without calling it (that
    synchronizes the device and adds jitter to every scheduler iteration):

        effective free = driver free + (reserved - allocated)

    where ``reserved - allocated`` is exactly what ``empty_cache()`` would
    return to the driver (PyTorch's own docs: empty_cache "releases all
    unoccupied cached memory").
    """

    free_bytes, _total = torch.cuda.mem_get_info()
    reserved = torch.cuda.memory_reserved()
    allocated = torch.cuda.memory_allocated()
    return (free_bytes + max(reserved - allocated, 0)) / (1024 * 1024)


def plan_oom_shed(
    *,
    rids: Iterable[str],
    finished_rids: Iterable[str] = (),
    beam_rids: Iterable[str] = (),
    strikes: Dict[str, int] = None,
    cap: int = None,
    streamed_rids: Iterable[str] = (),
) -> ShedPlan:
    """Plan the shed for a batch.

    ``rids``: the batch's requests in release order (carried decode requests
    included). Finished requests are skipped entirely: their KV is freed by
    the *previous* batch's result processing, and freeing them here too would
    double-free the pool.

    ``streamed_rids``: requests that already streamed tokens to the client.
    These are never hard-shed at the strike cap: a mid-stream abort is an
    SSE error frame inside an already-committed HTTP 200 — no upstream
    classifier can intercept it, so it always surfaces to the client as a
    retryable-looking error. They strike and requeue like everyone else;
    under sustained exhaustion they wait (visible as latency, not as a
    crash surface), which is the correct degradation.
    """
    finished = set(finished_rids)
    beam = set(beam_rids)
    streamed = set(streamed_rids)
    ledger = dict(strikes or {})
    if cap is None:
        cap = envs.SGLANG_OOM_SHED_MAX_STRIKES.get()

    requeue, abort = [], []
    for rid in rids:
        if rid in finished:
            continue
        if rid in beam:
            # Beam groups cannot be retracted (see ScheduleBatch.retract_decode:
            # members need the leader's lockstep length), so they are shed.
            ledger.pop(rid, None)
            abort.append(rid)
            continue
        n = ledger.get(rid, 0) + 1
        if n > cap and rid not in streamed:
            ledger.pop(rid, None)
            abort.append(rid)
        else:
            ledger[rid] = n
            requeue.append(rid)
    return ShedPlan(requeue, abort, ledger)