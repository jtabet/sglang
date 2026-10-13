"""Direct L3 support for :class:`UnifiedRadixCache`.

Links the cache's device pools straight to an external KV store, with no host
tier in between. The transport contract and tree-side wrapper live here, while
each backend owns its device-pool layout and physical I/O.

* :class:`UnifiedCacheLinker` -- the transport interface a backend implements.
* :class:`UnifiedCacheLinkerWrapper` -- the tree-side flow that drives it. The
  cache owns one as a plain attribute, keeping the whole external-cache path out
  of the main tree file.

The tree only needs a handful of guarded hooks:

* ``match_prefix``      -> :meth:`UnifiedCacheLinkerWrapper.match`
* ``init_load_back``    -> :meth:`UnifiedCacheLinkerWrapper.load_back`
* ``BackupKV`` actions  -> :meth:`UnifiedCacheLinkerWrapper.offload_nodes`

"""

from __future__ import annotations

import logging

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, NamedTuple

import torch

from sglang.srt.mem_cache.allocator.swa import is_swa_req_ring

logger = logging.getLogger(__name__)
from sglang.srt.mem_cache.base_prefix_cache import (
    DecLockRefParams,
    InsertParams,
    MatchResult,
)
from sglang.srt.mem_cache.hicache_storage import (
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.components import (
    ComponentType,
    ExternalLinkerLoadPhase,
    LinkerTransferPhase,
    TreeComponent,
)
from sglang.srt.mem_cache.utils import get_storage_hash_str

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.unified_cache.unified_tree_core_interface import NodeId
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache


_EXTERNAL_LINKER_SUPPORTED_COMPONENTS = frozenset(
    {
        ComponentType.FULL,
        ComponentType.SWA,
        ComponentType.MAMBA,
    }
)


class LoadQueueRejected(RuntimeError):
    """The external store refused/dropped a queued load before it started.

    Raised from ``init_load_back`` so the scheduler requeues the request for
    the next tick — where the match re-probes the store — instead of crashing
    the scheduler (event loops have no per-batch catch). The store rejecting a
    load queue is recoverable; a whole-pod SIGQUIT is not.
    """


class UnifiedCacheLinker(ABC):
    """External KV store reached directly from the device pools."""

    layer_done_counter: object

    @abstractmethod
    def lookup(self, rid: str, transfers: list[PoolTransfer]) -> list[int]:
        """Return every prefix length (in pages) that is fully restorable.

        A length is included only when *all* pools satisfy their hit policy at
        that exact boundary (contiguous prefix pools, plus each trailing-window
        pool's window ending there). Trailing-window state (SWA / compress
        state) only exists at offloaded node boundaries, so the set is sparse
        and generally non-contiguous -- returning just the local maximum would
        let the tree pick a length that is invalid on another rank.

        Local to this rank; the tree intersects the sets across ranks.
        """

    @abstractmethod
    def load(self, rid: str, transfers: list[PoolTransfer]) -> bool:
        """Queue a load into the given device indices.

        The transfer is executed by the next ``start_layer_wise_loading`` call,
        not here.
        """

    @abstractmethod
    def start_layer_wise_loading(self) -> int:
        """Start queued loads and return the layer-counter consumer index."""

    @abstractmethod
    def cancel_queued_load(self, rid: str) -> bool:
        """Cancel a load that has not started yet."""

    @abstractmethod
    def num_completed_loads(self) -> int:
        """Return the number of completed load batches waiting to be consumed."""

    @abstractmethod
    def pop_completed_load(self) -> list[str]:
        """Consume the oldest completed load batch and return its request IDs."""

    @abstractmethod
    def offload(self, transfers: list[PoolTransfer]) -> bool:
        """Queue every transfer for atomic persistence."""

    @abstractmethod
    def num_completed_offloads(self) -> int:
        """Return the number of completed offloads waiting to be consumed."""

    @abstractmethod
    def pop_completed_offload(self) -> bool:
        """Consume the oldest completed offload and return its result."""

    @abstractmethod
    def reset(self) -> None:
        """Quiesce all transfers and reset backend state before returning."""

    @abstractmethod
    def close(self) -> None:
        """Quiesce all transfers and release backend resources."""


class ExternalCacheHitMarker(NamedTuple):
    """What ``match`` found in the external store, consumed by ``load_back``.

    ``prefix_key`` covers the device-cached prefix plus the restorable tail, so
    it is what gets inserted once the tail lands. ``tail_hashes`` are the
    per-page storage hashes of that tail alone, starting at ``device_hit_len``.
    """

    prefix_key: RadixKey
    tail_hashes: list[str]
    device_hit_len: int


class _PendingOffload(NamedTuple):
    lock_node_id: NodeId
    lock_params: DecLockRefParams
    publish_node_ids: list[NodeId]


class UnifiedCacheLinkerWrapper:
    """Drives an external KV store on behalf of one :class:`UnifiedRadixCache`."""

    def __init__(
        self,
        cache: UnifiedRadixCache,
        cache_linker: UnifiedCacheLinker,
    ):
        unsupported = set(cache.tree_components) - _EXTERNAL_LINKER_SUPPORTED_COMPONENTS
        if unsupported:
            names = ", ".join(
                component.name for component in sorted(unsupported, key=int)
            )
            raise ValueError(
                "External cache linker supports only Full and SWA tree "
                f"components; unsupported: {names}"
            )

        self.cache = cache
        self.cache_linker = cache_linker
        swa = cache.components.get(ComponentType.SWA)
        self._skip_swa = swa is not None and is_swa_req_ring(
            cache.token_to_kv_pool_allocator
        )
        self._components = tuple(
            component
            for component in cache._components_tuple
            if not (self._skip_swa and component is swa)
        )
        # rid -> what match found, consumed by the next init_load_back.
        self.hit_markers: dict[str, ExternalCacheHitMarker] = {}
        # Loads in flight, each pinning its inserted endpoint until DMA completes.
        self.pending_loads: dict[str, tuple[NodeId, DecLockRefParams]] = {}
        # Offloads in flight, each holding a lock on its node until it lands.
        self.pending_offloads: list[_PendingOffload] = []

        cache.tree_core.enable_external_cache_linker = True
        cache.write_through_threshold = 1

    @property
    def layer_done_counter(self) -> object:
        return self.cache_linker.layer_done_counter

    def has_hit(self, rid: str) -> bool:
        return rid in self.hit_markers

    # ---- match: probe the remote store and report host_hit_length ----

    def match(self, key: RadixKey, req: Req, result: MatchResult) -> MatchResult:
        cache = self.cache
        key, _ = key.maybe_to_bigram_view(cache.tree_core.is_eagle)
        page = cache.page_size
        device_hit_len = int(result.device_indices.numel())
        if device_hit_len >= len(key):
            return result

        tail_hashes = self._tail_hashes(key, result, device_hit_len)
        if not tail_hashes:
            logger.info(
                "linker.match: rid=%s device_hit=%d key_len=%d — no tail_hashes (last_hash missing?)",
                req.rid, device_hit_len, len(key),
            )
            return result

        lookup_transfers = []
        for component in self._components:
            transfer = component.build_external_linker_transfer(
                LinkerTransferPhase.LOOKUP, None, tail_hashes
            )
            if transfer is None:
                # Component declines external-linker participation (e.g.
                # Mamba SSM state is per-request, not page-based). Skip it
                # and continue with the remaining components — the KV/Full
                # component is the one that matters for cache lookups.
                logger.debug(
                    "linker.match: rid=%s — component %s returned None transfer for LOOKUP, skipping",
                    req.rid, component,
                )
                continue
            lookup_transfers.append(transfer)
        if not lookup_transfers:
            return result
        by_pool = {transfer.name: transfer for transfer in lookup_transfers}

        # Tail-relative: page 0 of `tail_hashes` is the first uncached page.
        restorable = self.cache_linker.lookup(req.rid, lookup_transfers)
        logger.debug(
            "linker.match: rid=%s device_hit=%d key_len=%d tail_pages=%d "
            "restorable_count=%d restorable_first10=%s stats=%s",
            req.rid, device_hit_len, len(key), len(tail_hashes),
            len(restorable), restorable[:10] if restorable else "[]",
            self.cache_linker.stats,
        )
        hit_pages = self._sync_restorable_prefix(
            restorable,
            num_pages=len(tail_hashes),
            device_hit_pages=0,
        )
        if hit_pages == 0:
            return result
        hit_tokens = hit_pages * page

        swa_transfer = by_pool.get(PoolName.SWA)
        swa_host_hit_length = (
            min(len(swa_transfer.keys), hit_pages) * page
            if swa_transfer is not None
            else 0
        )
        # Mamba keeps a single state slot per node, so a hit is worth one slot.
        mamba_host_hit_length = 1 if PoolName.MAMBA in by_pool else 0

        self.hit_markers[req.rid] = ExternalCacheHitMarker(
            prefix_key=key[: device_hit_len + hit_tokens],
            tail_hashes=list(tail_hashes[:hit_pages]),
            device_hit_len=device_hit_len,
        )
        return result._replace(
            last_host_node=result.best_match_node,
            host_hit_length=hit_tokens,
            swa_host_hit_length=max(result.swa_host_hit_length, swa_host_hit_length),
            mamba_host_hit_length=max(
                result.mamba_host_hit_length, mamba_host_hit_length
            ),
        )

    def _sync_restorable_prefix(
        self, restorable: list[int], *, num_pages: int, device_hit_pages: int
    ) -> int:
        """Intersect the per-rank sets of restorable prefix lengths and return the
        longest one, or 0 when the ranks share none beyond the device prefix.

        A rank's set is sparse, so reducing per-rank maxima could land on a
        length that only some ranks can restore. On a 0/1 mask MIN is AND, which
        makes the reduction an intersection.
        """
        mask = torch.zeros(num_pages + 1, dtype=torch.int)
        for pages in restorable:
            if device_hit_pages < pages <= num_pages:
                mask[pages] = 1
        self.cache._all_reduce_attn_groups(mask, torch.distributed.ReduceOp.MIN)
        common = mask.nonzero()
        if common.numel() == 0:
            return 0
        return int(common[-1].item())

    def _tail_hashes(
        self, key: RadixKey, result: MatchResult, device_hit_len: int
    ) -> list[str]:
        """Per-page storage hashes for the device-uncached tail of the prefix."""
        last_hash = None
        if device_hit_len > 0:
            last_hash = self.cache.get_last_hash_value(result.last_device_node)
            if last_hash is None:
                # Without the anchor the tail would hash as if it started at the
                # sequence head, yielding keys that can never match.
                return []
        page = self.cache.page_size
        tail_len = (len(key) - device_hit_len) // page * page
        if tail_len == 0:
            return []
        return get_storage_hash_str(
            key[device_hit_len : device_hit_len + tail_len],
            last_hash,
            page_size=page,
        )

    # ---- init_load_back: remote -> device, then insert ----

    def load_back(self, req: Req) -> tuple[torch.Tensor, NodeId]:
        cache = self.cache
        empty_indices = cache.tree_core.empty_match_result.device_indices
        hit = self.hit_markers.pop(req.rid, None)
        if hit is None:
            return empty_indices, req.last_node

        device_hit_len = hit.device_hit_len
        tail_hashes = hit.tail_hashes
        prefix_len = device_hit_len + len(tail_hashes) * cache.page_size

        # Build per-component linker transfers.
        component_transfers: list[tuple[TreeComponent, PoolTransfer]] = []
        for component in self._components:
            transfer = component.build_external_linker_transfer(
                LinkerTransferPhase.LOAD, None, tail_hashes
            )
            if transfer is None:
                # Component declines external-linker participation (e.g.
                # Mamba SSM state is per-request). Skip it — the KV/Full
                # component is the one that matters for load-back.
                logger.debug(
                    "linker.load_back: rid=%s — component %s returned None for LOAD, skipping",
                    req.rid, component,
                )
                continue
            component_transfers.append((component, transfer))
        if not component_transfers:
            return empty_indices, req.last_node

        full_transfer = component_transfers[0][1]
        assert full_transfer.name == PoolName.KV
        self._update_load(
            ExternalLinkerLoadPhase.PREPARE,
            req,
            component_transfers,
            prefix_len,
        )

        # Components omitted from the linker do not run their PREPARE hook.
        # Keep a non-restorable SWA range as tombstones instead of rebuilding
        # it from an uninitialized FULL-to-SWA mapping during cache.insert().
        if self._skip_swa:
            if req.kv is None:
                from sglang.srt.managers.schedule_batch import ReqKvInfo

                req.kv = ReqKvInfo(
                    kv_allocated_len=prefix_len,
                    swa_evicted_seqlen=prefix_len,
                )
            else:
                req.kv.swa_evicted_seqlen = max(req.kv.swa_evicted_seqlen, prefix_len)

        # Insert the newly loaded tail into the tree.
        prefix_indices = torch.cat(
            [req.prefix_indices.to(torch.int64), full_transfer.device_indices]
        )
        mamba_transfer = next(
            (
                transfer
                for _, transfer in component_transfers
                if transfer.name == PoolName.MAMBA
            ),
            None,
        )
        if mamba_transfer is not None:
            mamba_value = mamba_transfer.device_indices[:1]
        else:
            # Mamba component declined external-linker participation.
            # Allocate a fresh mamba slot so the tree insert succeeds —
            # the SSM state will be computed during prefill of the loaded tail.
            mamba_alloc = cache.req_to_token_pool.mamba_allocator.alloc(1)
            if mamba_alloc is None:
                # Not enough mamba slots. Abort the load-back gracefully:
                # free the KV device indices we allocated, then return the
                # "no hit" result so the request falls back to normal prefill.
                logger.warning(
                    "linker.load_back: rid=%s — mamba allocator exhausted, "
                    "aborting load-back (prefill will recompute the prefix)",
                    req.rid,
                )
                self._update_load(
                    ExternalLinkerLoadPhase.ABORT,
                    req,
                    component_transfers,
                    prefix_len,
                )
                return empty_indices, req.last_node
            mamba_value = torch.tensor(
                [mamba_alloc[0]], dtype=torch.int64, device=cache.device
            )
        insert_result = cache.insert(
            InsertParams(
                key=hit.prefix_key,
                value=prefix_indices,
                mamba_value=mamba_value,
                prev_prefix_len=device_hit_len,
                swa_evicted_seqlen=(
                    req.kv.swa_evicted_seqlen if req.kv is not None else 0
                ),
                chunked=True,
                priority=getattr(req, "priority", 0) or 0,
                track_adopted_ranges=True,
            )
        )
        if insert_result.mamba_exist:
            # The insert found an existing mamba value on the node.
            # Free whichever slot we won't be using.
            if mamba_transfer is not None:
                cache.req_to_token_pool.mamba_allocator.free(
                    mamba_transfer.device_indices[:1]
                )
            else:
                cache.req_to_token_pool.mamba_allocator.free(mamba_value)

        canonical_tail = cache.tree_core.collect_full_device_indices(
            insert_result.last_device_node, req.last_node
        )
        assert canonical_tail.numel() == len(tail_hashes) * cache.page_size
        load_transfers = self._update_load(
            ExternalLinkerLoadPhase.COMMIT,
            req,
            component_transfers,
            prefix_len,
            insert_result=insert_result,
            canonical_full=canonical_tail,
        )

        self._queue_load(req.rid, insert_result.last_device_node, load_transfers)

        cache.tree_core.mark_external_cache_stored_path(
            insert_result.last_device_node, req.last_node
        )
        return canonical_tail, insert_result.last_device_node

    def _queue_load(
        self, rid: str, node_id: NodeId, transfers: list[PoolTransfer]
    ) -> None:
        if not transfers:
            return
        assert rid not in self.pending_loads
        lock_params = self.cache.inc_lock_ref(node_id).to_dec_params()
        try:
            queued = self.cache_linker.load(rid, transfers)
        except BaseException:
            # The insert above already committed a device tail, but the store
            # raised before queueing a load into it — the tail's device slots
            # are uninitialised. Quarantine it so the match walk stops there
            # and L2 reclaims the slots instead of matching them as a valid
            # device prefix (silent-corruption guard), then release the lock.
            self.cache.dec_lock_ref(node_id, lock_params)
            self.cache.tree_core.mark_external_cache_load_quarantined(node_id)
            raise
        if not queued:
            # Store-side rejected the queue (or the rid was cancelled between
            # match and queue). Same post-insert state as the exception path:
            # the tail node is in the tree holding device slots the store will
            # never write, so quarantine it (match-stop + eviction-reclaimable)
            # before releasing the lock and falling back to recompute prefill.
            self.cache.dec_lock_ref(node_id, lock_params)
            self.cache.tree_core.mark_external_cache_load_quarantined(node_id)
            logger.warning(
                "linker._queue_load: rid=%s — store rejected the load queue; "
                "quarantining the loaded tail and falling back to recompute "
                "prefill",
                rid,
            )
            raise LoadQueueRejected(
                f"Linker load queue rejected for rid={rid!r}."
            )
        self.pending_loads[rid] = (node_id, lock_params)
        self.cache.tree_core.mark_external_cache_load_pending_path(node_id)

    def _update_load(
        self,
        phase: ExternalLinkerLoadPhase,
        req: Req,
        component_transfers: list[tuple[TreeComponent, PoolTransfer]],
        prefix_len: int,
        *,
        insert_result=None,
        canonical_full: torch.Tensor | None = None,
    ) -> list[PoolTransfer]:
        if not component_transfers:
            return []
        full = component_transfers[0][1]
        result = []
        transfers = (
            reversed(component_transfers)
            if phase == ExternalLinkerLoadPhase.ABORT
            else component_transfers
        )
        for component, transfer in transfers:
            component_canonical = canonical_full
            if phase == ExternalLinkerLoadPhase.COMMIT:
                assert insert_result.adopted_ranges is not None
                coverage_start = prefix_len - len(transfer.device_indices)
                ranges = [
                    (max(start, coverage_start), min(end, prefix_len))
                    for start, end in insert_result.adopted_ranges.get(
                        component.component_type, ()
                    )
                    if max(start, coverage_start) < min(end, prefix_len)
                ]
                indices, keys = self._select_adopted_pages(
                    transfer.device_indices,
                    ranges,
                    prefix_len,
                    transfer.keys,
                )
                if not keys:
                    continue
                transfer.device_indices = indices
                transfer.keys = keys
                component_canonical, _ = self._select_adopted_pages(
                    canonical_full, ranges, prefix_len
                )
            transfer = component.update_external_linker_load(
                phase,
                req,
                full,
                transfer,
                prefix_len,
                insert_result=insert_result,
                canonical_full=component_canonical,
            )
            if transfer is not None:
                result.append(transfer)
        return result

    def _select_adopted_pages(
        self,
        indices: torch.Tensor,
        ranges: Sequence[tuple[int, int]],
        prefix_len: int,
        keys: Sequence[str] | None = None,
    ) -> tuple[torch.Tensor, list[str]]:
        page = self.cache.page_size
        coverage_start = prefix_len - len(indices)
        pages = indices.reshape(-1, page)
        if keys is not None:
            assert len(keys) == len(pages)

        chunks = []
        selected_keys = []
        for start, end in ranges:
            start = max(start, coverage_start)
            end = min(end, prefix_len)
            if start >= end:
                continue
            assert (start - coverage_start) % page == 0
            assert (end - coverage_start) % page == 0
            first = (start - coverage_start) // page
            last = (end - coverage_start) // page
            chunks.append(pages[first:last].reshape(-1))
            if keys is not None:
                selected_keys.extend(keys[first:last])

        if not chunks:
            return indices[:0], selected_keys
        selected = chunks[0] if len(chunks) == 1 else torch.cat(chunks)
        return selected, selected_keys

    # ---- offload: device -> remote, driven by the write-through chain ----

    def offload_nodes(self, node_ids: Sequence[NodeId]) -> None:
        """Persist a write-through chain, skipping nodes already in the store."""
        for node_id in node_ids:
            transfers = self.cache.tree_core.build_external_linker_offload_transfers(
                node_id
            )
            if transfers is not None:
                if self._skip_swa:
                    transfers = [t for t in transfers if t.name != PoolName.SWA]
                self._offload_node(node_id, transfers)

    def _offload_node(self, node_id: NodeId, transfers: list[PoolTransfer]) -> None:
        cache = self.cache
        lock_params = cache.inc_lock_ref(node_id).to_dec_params()
        try:
            queued = self.cache_linker.offload(transfers)
        except BaseException:
            cache.dec_lock_ref(node_id, lock_params)
            raise
        if not queued:
            cache.dec_lock_ref(node_id, lock_params)
            return

        cache.tree_core.mark_external_linker_offload_pending(node_id)
        self.pending_offloads.append(_PendingOffload(node_id, lock_params, [node_id]))

    def replace_pending_offload_node(
        self, ack_id: NodeId, old_node_id: NodeId, new_node_ids: list[NodeId]
    ) -> None:
        for index, pending in enumerate(self.pending_offloads):
            if pending.lock_node_id != ack_id:
                continue
            publish_node_ids = []
            for node_id in pending.publish_node_ids:
                if node_id == old_node_id:
                    publish_node_ids.extend(new_node_ids)
                else:
                    publish_node_ids.append(node_id)
            self.pending_offloads[index] = pending._replace(
                publish_node_ids=publish_node_ids
            )
            return

    def num_completed_offloads(self) -> int:
        return min(
            self.cache_linker.num_completed_offloads(), len(self.pending_offloads)
        )

    def num_completed_loads(self) -> int:
        return self.cache_linker.num_completed_loads()

    def num_failed_loads(self) -> int:
        return self.cache_linker.num_failed_loads()

    def drain_failed_loads(self) -> None:
        """Drain load failures even when no batch completed (deadline expiry)."""
        for rid, error in self.cache_linker.pop_failed_loads():
            pending = self.pending_loads.pop(rid, None)
            if pending is not None:
                node_id, lock_params = pending
                self.cache.dec_lock_ref(node_id, lock_params)
                self.cache.tree_core.mark_external_cache_load_quarantined(node_id)
                logger.warning(
                    "linker.drain_failed_loads: rid=%s load FAILED (%s) — "
                    "tail %s quarantined until eviction",
                    rid,
                    error,
                    node_id,
                )

    def drain_loads(self, finish_count: int) -> None:
        # Failures first: quarantine tails BEFORE the success channel could
        # publish overlapping rids, so a partially-successful batch never
        # briefly exposes unverified device content.
        for rid, error in self.cache_linker.pop_failed_loads():
            pending = self.pending_loads.pop(rid, None)
            if pending is not None:
                node_id, lock_params = pending
                self.cache.dec_lock_ref(node_id, lock_params)
                self.cache.tree_core.mark_external_cache_load_quarantined(node_id)
                logger.warning(
                    "linker.drain_loads: rid=%s load FAILED (%s) — tail %s "
                    "quarantined until eviction",
                    rid,
                    error,
                    node_id,
                )
        for _ in range(finish_count):
            for rid in self.cache_linker.pop_completed_load():
                pending = self.pending_loads.pop(rid, None)
                if pending is None:
                    # Drained by release_request (abort) after publication —
                    # the load's tree lock was already released there.
                    continue
                node_id, lock_params = pending
                self.cache.dec_lock_ref(node_id, lock_params)
                self.cache.tree_core.clear_external_cache_load_pending(node_id)

    def take_completed_offloads(self, finish_count: int) -> list[bool]:
        assert finish_count <= len(self.pending_offloads)
        return [self.cache_linker.pop_completed_offload() for _ in range(finish_count)]

    def commit_completed_offloads(self, successes: Sequence[bool]) -> None:
        assert len(successes) <= len(self.pending_offloads)
        for success in successes:
            pending = self.pending_offloads.pop(0)
            self.cache.tree_core.finish_external_linker_offload(
                pending.publish_node_ids, pending.lock_node_id, success
            )
            self.cache.dec_lock_ref(pending.lock_node_id, pending.lock_params)

    def start_layer_wise_loading(self) -> int:
        return self.cache_linker.start_layer_wise_loading()

    # ---- lifecycle ----

    def reset(self) -> None:
        self.cache_linker.reset()
        self.hit_markers.clear()
        self._release_pending_locks()

    def _release_pending_locks(self) -> None:
        for node_id, lock_params in self.pending_loads.values():
            self.cache.dec_lock_ref(node_id, lock_params)
        self.pending_loads.clear()
        for pending in self.pending_offloads:
            self.cache.tree_core.finish_external_linker_offload(
                pending.publish_node_ids, pending.lock_node_id, False
            )
            self.cache.dec_lock_ref(pending.lock_node_id, pending.lock_params)
        self.pending_offloads.clear()

    def release_request(self, rid: str) -> None:
        self.hit_markers.pop(rid, None)
        self.cache.tree_core.clear_external_cache_load_pending_rid(rid)
        if self.cache_linker.cancel_queued_load(rid):
            # The load will never report completion through pop_completed_load
            # (cancelled pre-publication skips it; started batches fail their
            # counter and are drained without a completed-load entry because
            # the rid was already marked cancelled). Release the tree lock now
            # — but keep the node's quarantine flag: device slots the store is
            # still RDMA-writing must not be matched or evicted-as-loaded.
            pending = self.pending_loads.pop(rid, None)
            if pending is not None:
                node_id, lock_params = pending
                self.cache.dec_lock_ref(node_id, lock_params)
                self.cache.tree_core.mark_external_cache_load_quarantined(node_id)

    def close(self) -> None:
        self.cache_linker.close()
        self._release_pending_locks()
