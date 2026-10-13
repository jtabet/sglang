from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FuturesTimeoutError
from queue import Empty, Queue

import torch

try:
    from asyncio import CancelledError
except ImportError:  # pragma: no cover — py<3.8 style fallback
    class CancelledError(Exception):
        pass

from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorageConfig,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import (
    resolve_hybrid_device_pool_group,
)
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import UnifiedCacheLinker
from sglang.srt.runtime_context import (
    get_memory,
    get_model,
    get_parallel,
)
from sglang.srt.utils import freeze_gc, get_device_module

logger = logging.getLogger(__name__)
device_module = get_device_module()


class MooncakeLoadTimeout(RuntimeError):
    """A registered load batch blew its wall-clock deadline.

    Raised on the scheduler thread from ``wait_until`` so the in-forward wait
    becomes a diagnosable error instead of an unbounded scheduler wedge (the
    20:59:35Z celadon crash class).
    """


def _storage_suffix(
    *, rank_replicated: bool, tp_rank: int, attn_cp_rank: int, pp_rank: int
) -> str:
    parts = []
    if not rank_replicated:
        parts.append(f"tp{tp_rank}")
    parts.extend((f"cp{attn_cp_rank}", f"pp{pp_rank}"))
    return "_".join(parts)


class LayerWiseLoadCounter:
    """CPU completion counter compatible with KV pools' layer wait hook.

    Every registered batch carries a wall-clock deadline (``load_deadline_secs``,
    clamped below the scheduler watchdog timeout). ``wait_until`` — which runs on
    the scheduler thread in the middle of an attention forward — never blocks
    without one: an unbounded ``future.result()`` here is what wedged the whole
    scheduler (frozen ``forward_ct`` → watchdog SIGQUIT → pod restart) when the
    store stalled. On expiry the batch's futures are failed, which turns the
    wedge into a diagnosable RuntimeError on the waiting thread.
    """

    _CANCELLED = "__load_cancelled__"

    def __init__(self, num_layers: int, load_deadline_secs: float | None = None):
        self.num_layers = num_layers
        self.producer_index = -1
        self.consumer_index = -1
        self.futures: dict[int, list[Future]] = {}
        # index -> monotonic deadline for batches awaiting completion.
        self._pending_until: dict[int, float] = {}
        self._deadline_secs = max(1.0, float(load_deadline_secs or 540.0))

    def update_producer(self) -> int:
        self.producer_index += 1
        index = self.producer_index
        self.futures[index] = [Future() for _ in range(self.num_layers)]
        self._pending_until[index] = time.monotonic() + self._deadline_secs
        return index

    def set_consumer(self, index: int) -> None:
        self.consumer_index = index
        # The forward is about to wait on this batch: any registered batch
        # missing a deadline (e.g. queued long before) still gets one.
        self._pending_until.setdefault(
            index, time.monotonic() + self._deadline_secs
        )

    def complete(self, index: int, layer: int) -> None:
        self.futures[index][layer].set_result(None)

    def layer_ready(self, index: int, layer: int) -> bool:
        """Non-blocking readiness probe for the pre-forward gate."""
        futures = self.futures.get(index)
        return futures is None or futures[layer].done()

    def fail(self, index: int, error: BaseException) -> None:
        self._pending_until.pop(index, None)
        for future in self.futures.get(index, ()):
            if not future.done():
                future.set_exception(error)

    def _reap_expired(self) -> None:
        """Fail every registered batch whose deadline has passed.

        Called only from ``wait_until`` (scheduler thread): the failing futures
        release the in-forward waiters; batches with no waiter get their
        exception delivered whenever a later wait (or reset) touches them.
        """
        now = time.monotonic()
        expired = [
            index
            for index, until in self._pending_until.items()
            if until <= now
        ]
        for index in expired:
            timeout = self.futures.get(index)
            if timeout is None:
                self._pending_until.pop(index, None)
                continue
            waiter = [f for f in timeout if not f.done()]
            if not waiter:
                self._pending_until.pop(index, None)
                continue
            error = MooncakeLoadTimeout(
                f"Mooncake load batch {index} exceeded "
                f"{self._deadline_secs:.0f}s deadline; "
                f"layers pending: {len(waiter)}/{self.num_layers}"
            )
            logger.error(
                "Mooncake load deadline exceeded: batch=%d pending_layers=%d/%d "
                "deadline=%.0fs — failing in-forward waiters",
                index,
                len(waiter),
                self.num_layers,
                self._deadline_secs,
            )
            self.fail(index, error)

    def wait_until(self, threshold: int) -> None:
        index = self.consumer_index
        futures = self.futures.get(index)
        if futures is None:
            return
        self._reap_expired()
        future = futures[threshold]
        remaining = self._pending_until.get(index)
        timeout = (
            None
            if remaining is None
            else max(0.0, remaining - time.monotonic())
        )
        try:
            future.result(timeout=timeout)
        except FuturesTimeoutError:
            # The reaper has (or will) deliver the authoritative error; try to
            # self-fail here so the wait cannot be re-entered unbounded.
            raise MooncakeLoadTimeout(
                f"Mooncake load batch {index} exceeded "
                f"{self._deadline_secs:.0f}s deadline in wait_until"
            ) from None
        except MooncakeLoadTimeout:
            # The reaper already failed this batch's futures with the
            # authoritative deadline error — surface it verbatim.
            raise
        except BaseException as error:
            raise RuntimeError("Mooncake layer-wise KV load failed.") from error
        finally:
            if threshold == self.num_layers - 1:
                self.futures.pop(index, None)
                self._pending_until.pop(index, None)

    def drop(self, index: int) -> None:
        """Discard a registered batch that will never be waited on.

        Used when every rid in the batch was retracted/requeued before forward
        (e.g. the pre-forward load gate tripped and requeued them). Without this
        the per-batch futures would linger for the process lifetime, so a
        prolonged store-down (the gate tripping tick after tick) would leak
        ``num_layers`` futures per trip. Safe because no forward will wait on a
        batch whose rids were all retracted.
        """
        self.futures.pop(index, None)
        self._pending_until.pop(index, None)

    def reset(self) -> None:
        self.producer_index = -1
        self.consumer_index = -1
        self.futures.clear()
        self._pending_until.clear()


def _resolve_load_deadline_secs() -> float:
    """Wall-clock deadline for one registered mooncake load batch.

    Must stay comfortably below the scheduler watchdog timeout: the whole point
    is to turn a stuck store call into a diagnosable load failure BEFORE the
    watchdog SIGQUITs the process tree. Default: min(540, watchdog-60).
    """
    from sglang.srt.environ import envs

    raw = envs.SGLANG_MOONCAKE_LOAD_DEADLINE_SECS.get()
    watchdog = 600.0
    try:
        from sglang.srt.runtime_context import get_device

        watchdog = float(get_device().watchdog_timeout)
    except Exception:
        pass
    cap = max(30.0, watchdog - 60.0)
    value = float(raw) if raw else min(540.0, cap)
    return value


class MooncakeDirectLinker(UnifiedCacheLinker):
    def __init__(
        self,
        server_args,
        params: CacheInitParams,
        *,
        components,
        storage=None,
    ):
        self.page_size = params.page_size
        kvcache = params.token_to_kv_pool_allocator.get_kvcache()
        self.pool_group = resolve_hybrid_device_pool_group(
            kvcache=kvcache,
            page_size=self.page_size,
            params=params,
            components=components,
        )
        self.pools = self.pool_group.entry_map
        self.num_layers = self.pool_group.num_layers

        tp_rank = 0
        tp_size = get_parallel().tp_size
        tp_group = params.attn_tp_cache_group or params.tp_cache_group
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            tp_rank = torch.distributed.get_rank(group=tp_group)
            tp_size = torch.distributed.get_world_size(group=tp_group)
        rank_replicated = self.pool_group.rank_replicated
        self.offload_owner = not rank_replicated or tp_rank == 0
        extra_config, *_ = HybridCacheController.parse_storage_backend_extra_config(
            get_memory().hicache_storage_backend_extra_config
        )
        storage_config = HiCacheStorageConfig(
            tp_rank=tp_rank,
            tp_size=tp_size,
            pp_rank=params.pp_rank,
            pp_size=params.pp_size,
            attn_cp_rank=params.attn_cp_rank,
            attn_cp_size=params.attn_cp_size,
            is_mla_model=rank_replicated,
            enable_storage_metrics=False,
            is_page_first_layout=False,
            model_name=get_model().model_path,
            extra_config=extra_config,
        )
        if storage is None:
            from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
                MooncakeStore,
            )

            self.storage = MooncakeStore(storage_config, mem_pool=None)
        else:
            self.storage = storage
        self.storage.mem_pool_host = self.pool_group
        self.storage.registered_pools = self.pools
        storage_suffix = _storage_suffix(
            rank_replicated=rank_replicated,
            tp_rank=tp_rank,
            attn_cp_rank=params.attn_cp_rank,
            pp_rank=params.pp_rank,
        )
        self.storage.mla_suffix = storage_suffix
        self.storage.mha_suffix = storage_suffix
        logger.info(
            "Mooncake direct linker storage topology: "
            "rank_replicated=%s, tp_rank=%d/%d, offload_owner=%s, suffix=%s",
            rank_replicated,
            tp_rank,
            tp_size,
            self.offload_owner,
            storage_suffix,
        )

        self.register_buffers()
        self.load_deadline_secs = _resolve_load_deadline_secs()
        self.layer_done_counter = LayerWiseLoadCounter(
            self.num_layers, load_deadline_secs=self.load_deadline_secs
        )
        if PoolName.MAMBA in self.pools:
            params.req_to_token_pool.register_layer_transfer_counter(
                self.layer_done_counter
            )
        self.pending_loads: dict[str, list[PoolTransfer]] = {}
        # rids whose queued load was cancelled by the scheduler before the
        # load thread picked it up, and rid -> counter_index for started
        # batches so the thread can observe cancellation.
        self._cancelled_loads: set[str] = set()
        self._load_counter_index: dict[str, int] = {}
        self.gc_frozen = False
        self.load_queue: Queue[
            tuple[int, dict[str, list[PoolTransfer]], object] | None
        ] = Queue()
        self.completed_loads: Queue[list[str]] = Queue()
        # rid -> error for batches whose load failed (deadline, store error,
        # cancellation). Drained by the wrapper alongside completed_loads so
        # failed tails are quarantined instead of published as loaded.
        self.failed_loads: Queue[tuple[str, BaseException]] = Queue()
        self.offload_queue: Queue[tuple[list[PoolTransfer], int, object] | None] = (
            Queue()
        )
        self.offload_results: Queue[bool] = Queue()
        self.stats = {"lookup": 0, "load": 0, "offload": 0}
        self.load_thread = threading.Thread(
            target=self.load_thread_func,
            daemon=True,
            name=f"mooncake-load-tp{tp_rank}",
        )
        self.load_thread.start()
        self.offload_thread = threading.Thread(
            target=self.offload_thread_func,
            daemon=True,
            name=f"mooncake-offload-tp{tp_rank}",
        )
        self.offload_thread.start()

    def register_buffers(self) -> None:
        seen = set()
        for pool in self.pools.values():
            for buffer in pool.get_hybrid_pool_buffer():
                storage = buffer.untyped_storage()
                allocation = (int(storage.data_ptr()), int(storage.nbytes()))
                if allocation in seen:
                    continue
                seen.add(allocation)
                result = self.storage.store.register_buffer(*allocation)
                if result not in (0, None):
                    raise RuntimeError(
                        "Failed to register GPU KV buffer with Mooncake, "
                        f"error code: {result}."
                    )

    def lookup(self, rid: str, transfers: list[PoolTransfer]) -> list[int]:
        expanded = self.pool_group.resolve_transfers(
            transfers, allow_partial=True, allow_missing_kv=True
        )
        if not expanded:
            logger.info("mooncake.lookup: rid=%s — no expanded transfers", rid)
            return []
        kv = next(transfer for transfer in transfers if transfer.name == PoolName.KV)
        page_keys = list(kv.keys)
        if not page_keys:
            logger.info("mooncake.lookup: rid=%s — no page keys", rid)
            return []
        result = self.storage.batch_exists_v2(page_keys, expanded)
        restorable = result.restorable_prefix_pages or []
        self.stats["lookup"] += 1
        if restorable:
            logger.info(
                "Mooncake direct linker lookup hit: rid=%s pages=%d candidates=%d",
                rid,
                restorable[-1],
                len(restorable),
            )
        return restorable

    def load(self, rid: str, transfers: list[PoolTransfer]) -> bool:
        # Query establishes a boundary at which every component is restorable;
        # insert then removes pages already resident in L1. Loading is therefore
        # intentionally partial and may contain only a side pool such as SWA.
        if rid in self._cancelled_loads:
            return False
        expanded = self.pool_group.resolve_transfers(
            transfers, allow_partial=True, allow_missing_kv=True
        )
        if not expanded:
            return False
        if rid in self.pending_loads:
            raise RuntimeError(f"Mooncake load for rid={rid} is already queued.")
        self.pending_loads[rid] = expanded
        return True

    def cancel_queued_load(self, rid: str) -> bool:
        # Loads still in pending_loads are cancelled here; loads already
        # published to a batch are tracked by counter index so the load thread
        # skips their transfers. Either way the wrapper can release the tree
        # lock for this rid — the load that lands later is marked dead by the
        # counter's cancellation (no tree state is written for it).
        self._cancelled_loads.add(rid)
        self.pending_loads.pop(rid, None)
        return True

    def num_completed_loads(self) -> int:
        return self.completed_loads.qsize()

    def num_failed_loads(self) -> int:
        return self.failed_loads.qsize()

    def pop_completed_load(self) -> list[str]:
        return self.completed_loads.get_nowait()

    def pop_failed_loads(self) -> list[tuple[str, BaseException]]:
        """Drain every queued load failure (rid, error)."""
        failures = []
        while True:
            try:
                failures.append(self.failed_loads.get_nowait())
            except Empty:
                return failures

    def freeze_gc_once(self) -> None:
        if self.gc_frozen:
            return
        # Transfer metadata creates many short-lived lists. Keep the mature
        # model graph out of cyclic GC scans before load or offload traffic.
        freeze_gc("Mooncake direct linker")
        self.gc_frozen = True

    def start_layer_wise_loading(self) -> int:
        if not self.pending_loads:
            return -1
        self.freeze_gc_once()
        pending = self.pending_loads
        self.pending_loads = {}

        # Drop rids cancelled between load() and this publication so the load
        # thread never spends a batch slot (and a counter registration) on them.
        cancelled = self._cancelled_loads.intersection(pending)
        for rid in cancelled:
            del pending[rid]
        self._cancelled_loads.difference_update(pending)
        if not pending:
            return -1
        self.stats["load"] += len(pending)

        counter_index = self.layer_done_counter.update_producer()
        for rid in pending:
            self._load_counter_index[rid] = counter_index
        ready_event = device_module.Event()
        ready_event.record()
        self.load_queue.put((counter_index, pending, ready_event))
        return counter_index

    def load_thread_func(self) -> None:
        while True:
            task = self.load_queue.get()
            try:
                if task is None:
                    return
                counter_index, pending, ready_event = task
                # rids cancelled between publication and thread pickup: their
                # counter already exists, so the batch must still be closed —
                # but the cancelled rids are failures (no tree state published).
                live: dict[str, list[PoolTransfer]] = {}
                cancelled: list[str] = []
                for rid, transfers in pending.items():
                    if rid in self._cancelled_loads:
                        cancelled.append(rid)
                    else:
                        live[rid] = transfers
                self._cancelled_loads.difference_update(pending)
                load_error: BaseException | None = None
                if live:
                    try:
                        ready_event.synchronize()
                        self.load_layer_wise(
                            counter_index, list(live.values())
                        )
                    except BaseException as error:
                        load_error = error
                if load_error is not None:
                    # A store error (or a wedged store) mid-load: fail the
                    # whole batch's in-forward waiters and report EVERY rid as
                    # a failure so it is quarantined and recomputed. It must
                    # NEVER be reported as a success — a success would publish
                    # the batch's device slots as loaded when the store never
                    # filled them (silent KV corruption).
                    self.layer_done_counter.fail(counter_index, load_error)
                    logger.exception(
                        "Mooncake layer-wise load batch %d failed", counter_index
                    )
                    for rid in pending:
                        self.failed_loads.put((rid, load_error))
                    self.completed_loads.put([])
                elif cancelled:
                    # Only cancellations (no load error): the live rids loaded
                    # fine (their layer futures are already complete), so report
                    # them as successes. The cancelled rids are failures — fail
                    # the batch's (still-pending if fully cancelled) futures and
                    # report them so the wrapper quarantines + releases their
                    # tree locks; no tree state is published for them.
                    for rid in cancelled:
                        self.layer_done_counter.fail(
                            counter_index, CancelledError(rid)
                        )
                        self.failed_loads.put((rid, CancelledError(rid)))
                    self.completed_loads.put(list(live))
                else:
                    self.completed_loads.put(list(pending.keys()))
                # A batch with no live rids was fully retracted (e.g. the
                # pre-forward gate tripped) — no forward will wait on it, so
                # drop its futures to keep a long store-down from leaking them.
                if not live:
                    self.layer_done_counter.drop(counter_index)
            finally:
                self.load_queue.task_done()

    def _load_layer_wise_fallback(
        self,
        counter_index: int,
        batches: dict[PoolName, tuple[list[str], list[int]]],
        pool_indices: dict[PoolName, list[torch.Tensor]],
    ) -> None:
        """Non-layer-wise fallback: load all layers at once.

        Propagates errors to the caller (``load_thread_func``) instead of
        swallowing them — a partial/failed batch must be a *failure* (quarantined
        + recomputed), never a silent partial success that publishes unfilled
        device slots as loaded.
        """
        store = self.storage.store
        for name, (keys, _locations) in batches.items():
            if len(keys) == 0:
                continue
            # Gather all host_indices for this pool and compute full buffer
            # ptrs/sizes across all layers at once.
            all_indices = torch.cat(pool_indices[name])
            ptrs, sizes = self.pools[name].get_page_buffer_meta(all_indices)
            # Pack flat ptrs/sizes into nested lists (one list per key) as
            # expected by batch_get_into_multi_buffers.
            if len(ptrs) != len(keys):
                assert len(ptrs) % len(keys) == 0
                nbuf = len(ptrs) // len(keys)
                ptrs = [ptrs[i : i + nbuf] for i in range(0, len(ptrs), nbuf)]
                sizes = [sizes[i : i + nbuf] for i in range(0, len(sizes), nbuf)]
            result = store.batch_get_into_multi_buffers(keys, ptrs, sizes)
            if result is None or isinstance(result, int):
                raise RuntimeError(
                    f"Mooncake batch get returned no result for pool={name}: "
                    f"result={result}"
                )
            missing = [i for i, r in enumerate(result) if r < 0]
            if missing:
                # A key the store cannot serve means those device slots were
                # never filled — the load for the affected rids must fail (and
                # be recomputed), not be reported as a loaded prefix.
                raise RuntimeError(
                    f"Mooncake batch get: {len(missing)}/{len(result)} keys "
                    f"(pool={name}) not found in store — failing the batch"
                )
        # Every key served; signal all layers complete at once.
        for layer in range(self.num_layers):
            self.layer_done_counter.complete(counter_index, layer)

    def load_layer_wise(
        self, counter_index: int, request_transfers: list[list[PoolTransfer]]
    ) -> None:
        """Run the store->device transfer for one registered batch.

        Propagates any store error to ``load_thread_func`` (which fails the
        batch's in-forward futures and reports the rids as failures). The
        session path still runs its ``batch_get_session_end`` cleanup in a
        ``finally`` so a wedged store call cannot leak an open session.
        """
        # Check if the raw mooncake store supports the layer-wise session API.
        store = self.storage.store
        has_session_api = hasattr(store, "batch_get_session_start")

        batches: dict[PoolName, tuple[list[str], list[int]]] = {}
        # For the non-layer-wise fallback we also need the original device indices
        # to compute full buffer ptrs/sizes via get_page_buffer_meta.
        pool_indices: dict[PoolName, list[torch.Tensor]] = {}
        for transfers in request_transfers:
            for transfer in transfers:
                keys, locations = batches.setdefault(transfer.name, ([], []))
                component_keys, _ = self.storage._get_hybrid_page_component_keys(
                    list(transfer.keys), transfer
                )
                keys.extend(self.storage._tag_keys(component_keys))
                locations.extend(
                    self.pools[transfer.name].prepare_locations(
                        transfer.host_indices
                    )
                )
                idx_list = pool_indices.setdefault(transfer.name, [])
                idx_list.append(transfer.host_indices)

        if not has_session_api:
            self._load_layer_wise_fallback(counter_index, batches, pool_indices)
            return

        # Layer-wise path: session-based, per-layer transfer with overlap.
        started = []
        try:
            for keys, _ in batches.values():
                result = store.batch_get_session_start(keys)
                if list(result) != [0] * len(keys):
                    raise RuntimeError(
                        f"Mooncake get session start failed: keys={len(keys)}, "
                        f"results={result}"
                    )
                started.append(keys)

            for layer in range(self.num_layers):
                for name, (keys, locations) in batches.items():
                    meta = self.pools[name].get_prepared_layer_range_meta(
                        locations, layer
                    )
                    if meta is None:
                        continue
                    ptrs, sizes, offsets = meta
                    result = store.batch_get_into_multi_buffer_ranges(
                        keys,
                        ptrs,
                        sizes,
                        offsets,
                    )
                    expected = [sum(item) for item in sizes]
                    if (
                        result is None
                        or isinstance(result, int)
                        or list(result) != expected
                    ):
                        raise RuntimeError(
                            f"Mooncake range get failed for pool={name}, "
                            f"layer={layer}: transferred={result}, "
                            f"expected={expected}"
                        )
                self.layer_done_counter.complete(counter_index, layer)
        finally:
            for keys in started:
                try:
                    store.batch_get_session_end(keys)
                except BaseException:
                    logger.exception(
                        "Mooncake layer-wise load session cleanup failed"
                    )

    def offload(self, transfers: list[PoolTransfer]) -> bool:
        expanded = self.pool_group.resolve_transfers(transfers, allow_partial=True)
        if not expanded:
            return False
        self.freeze_gc_once()
        if not self.offload_owner:
            self.offload_results.put(True)
            return True
        kv = next(transfer for transfer in transfers if transfer.name == PoolName.KV)
        tokens = len(kv.keys) * self.page_size
        ready_event = device_module.Event()
        ready_event.record()
        self.offload_queue.put((expanded, tokens, ready_event))
        return True

    def offload_thread_func(self) -> None:
        while True:
            task = self.offload_queue.get()
            try:
                if task is None:
                    return
                expanded, tokens, ready_event = task
                ready_event.synchronize()
                results = self.storage.batch_set_v2(expanded)
                success = all(all(pool_results) for pool_results in results.values())
                if success:
                    self.stats["offload"] += 1
                    if self.stats["offload"] == 1:
                        logger.info("Mooncake direct linker offload: tokens=%d", tokens)
                self.offload_results.put(success)
            except BaseException:
                logger.exception("Mooncake offload failed")
                self.offload_results.put(False)
            finally:
                self.offload_queue.task_done()

    def num_completed_offloads(self) -> int:
        return self.offload_results.qsize()

    def pop_completed_offload(self) -> bool:
        return self.offload_results.get_nowait()

    def reset(self) -> None:
        self.pending_loads.clear()
        self.load_queue.join()
        self.offload_queue.join()
        while True:
            try:
                self.offload_results.get_nowait()
            except Empty:
                break
        while True:
            try:
                self.completed_loads.get_nowait()
            except Empty:
                break
        self.layer_done_counter.reset()

    def close(self) -> None:
        self.reset()
        self.load_queue.put(None)
        self.offload_queue.put(None)
        self.load_thread.join()
        self.offload_thread.join()
        logger.info("Mooncake direct linker stats: %s", self.stats)
        self.storage.close()
