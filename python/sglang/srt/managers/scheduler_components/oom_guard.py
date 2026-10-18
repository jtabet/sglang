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


def plan_oom_shed(
    *,
    rids: Iterable[str],
    finished_rids: Iterable[str] = (),
    beam_rids: Iterable[str] = (),
    strikes: Dict[str, int] = None,
    cap: int = None,
) -> ShedPlan:
    """Plan the shed for a batch.

    ``rids``: the batch's requests in release order (carried decode requests
    included). Finished requests are skipped entirely: their KV is freed by
    the *previous* batch's result processing, and freeing them here too would
    double-free the pool.
    """
    finished = set(finished_rids)
    beam = set(beam_rids)
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
        if n > cap:
            ledger.pop(rid, None)
            abort.append(rid)
        else:
            ledger[rid] = n
            requeue.append(rid)
    return ShedPlan(requeue, abort, ledger)