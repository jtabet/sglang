//! Load balancing policies for SGLang router
//!
//! This module provides a unified abstraction for routing policies that work
//! across both regular and prefill-decode (PD) routing modes.

use std::{fmt::Debug, sync::Arc};

use async_trait::async_trait;
use smg_mesh::OptionalMeshSyncManager;

use crate::core::{HashRing, Worker};

mod bucket;
mod cache_aware;
mod consistent_hashing;
mod factory;
mod manual;
mod power_of_two;
mod prefix_hash;
mod random;
mod registry;
mod round_robin;
pub mod tree;
pub(crate) mod utils;
pub use bucket::BucketPolicy;
pub use cache_aware::CacheAwarePolicy;
pub use consistent_hashing::ConsistentHashingPolicy;
pub use factory::PolicyFactory;
pub use manual::{ManualConfig, ManualPolicy};
pub use power_of_two::PowerOfTwoPolicy;
pub use prefix_hash::{PrefixHashConfig, PrefixHashPolicy};
pub use random::RandomPolicy;
pub use registry::PolicyRegistry;
pub use round_robin::RoundRobinPolicy;
pub use tree::PrefixMatchResult;

/// Core trait for load balancing policies
///
/// This trait provides a unified interface for implementing routing algorithms
/// that can work with both regular single-worker selection and PD dual-worker selection.
#[async_trait]
pub trait LoadBalancingPolicy: Send + Sync + Debug {
    /// Select a single worker from the available workers
    ///
    /// This is used for regular routing mode where requests go to a single worker.
    /// Now uses Arc<dyn Worker> for better performance and to avoid unnecessary cloning.
    ///
    /// # Arguments
    /// * `workers` - Available workers to select from
    /// * `info` - Additional information for routing decisions
    async fn select_worker(
        &self,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo<'_>,
    ) -> Option<usize>;

    /// Select a worker and also report selection metadata (see
    /// [`SelectionInfo`]).
    ///
    /// Default: delegates to [`Self::select_worker`] and reports
    /// `match_rate: 0.0`. Cache-aware policies override this to surface the
    /// ratio their affinity tree already computed, so the router can estimate
    /// TTFT without re-walking the tree.
    async fn select_worker_with_info(
        &self,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo<'_>,
    ) -> Option<SelectionInfo> {
        self.select_worker(workers, info)
            .await
            .map(|index| SelectionInfo { index, match_rate: 0.0 })
    }

    /// Update policy state after request completion
    ///
    /// This is called when a request completes (successfully or not) to allow
    /// policies to update their internal state.
    fn on_request_complete(&self, _worker_url: &str, _success: bool) {
        // Default: no-op for stateless policies
    }

    /// Get policy name for metrics and debugging
    fn name(&self) -> &'static str;

    /// Check if this policy needs request text for routing decisions
    fn needs_request_text(&self) -> bool {
        false // Default: most policies don't need request text
    }

    /// Update worker load information
    ///
    /// This is called periodically with current load information for load-aware policies.
    fn update_loads(&self, _loads: &std::collections::HashMap<String, isize>) {
        // Default: no-op for policies that don't use load information
    }

    /// Set mesh sync manager
    fn set_mesh_sync(&mut self, _mesh_sync: OptionalMeshSyncManager) {
        // Default: no-op for policies that don't use mesh sync
    }

    /// Reset any internal state
    ///
    /// This is useful for policies that maintain state (e.g., round-robin counters).
    fn reset(&self) {
        // Default: no-op for stateless policies
    }

    /// Get as Any for downcasting
    fn as_any(&self) -> &dyn std::any::Any;
}

/// Configuration for cache-aware policy
#[derive(Debug, Clone)]
pub struct CacheAwareConfig {
    pub cache_threshold: f32,
    /// Request-count scale: rebalance when one worker has more than this many
    /// more in-flight *requests* than the least loaded worker.
    pub balance_abs_threshold: usize,
    /// Token scale: rebalance when one worker has more than this many more
    /// in-flight *estimated prompt tokens* than the least loaded worker.
    /// The request-count valve cannot see a 250k-token prefill sitting
    /// behind a single in-flight request, so a token valve guards the mix of
    /// giant agent prompts and small turns.
    pub balance_abs_threshold_tokens: usize,
    pub balance_rel_threshold: f32,
    pub eviction_interval_secs: u64,
    pub max_tree_size: usize,
    /// Per-worker capacity gate for the cache-hit path, in *estimated*
    /// tokens. `None` (default) disables the gate: routing behavior is
    /// exactly the pre-gate affinity behavior. `Some(n)` arms it: when the
    /// prefix-match tenant's projected token load — its in-flight
    /// `load_tokens()` plus this request's estimated tokens (same
    /// chars/token estimator the router charges at dispatch) — would exceed
    /// `n`, the request falls through to the load-based (min tokens)
    /// selection instead of pinning to the tenant. If no worker fits, the
    /// min-token worker is returned anyway (the request is never failed by
    /// the gate itself).
    ///
    /// Size `n` to the worker pool's KV capacity in tokens minus a safety
    /// margin. The estimate is chars-based (the policy path has no
    /// tokenizer), so err conservative: real tokenizers can expand text and
    /// decode tokens also consume the pool beyond this estimate. Fleet
    /// sizing example: ~355k-token pools serving 174k–224k-token agent
    /// requests → ~190k = at most one of those big requests admitted at a
    /// time per worker; stacking a second wedges the scheduler in admission
    /// or OOMs the forward.
    pub capacity_per_worker_tokens: Option<u64>,
}

impl Default for CacheAwareConfig {
    fn default() -> Self {
        Self {
            cache_threshold: 0.5,
            balance_abs_threshold: 32,
            balance_abs_threshold_tokens: 8192,
            balance_rel_threshold: 1.1,
            eviction_interval_secs: 30,
            max_tree_size: 10000,
            capacity_per_worker_tokens: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct BucketConfig {
    pub balance_abs_threshold: usize,
    pub balance_rel_threshold: f32,
    pub bucket_adjust_interval_secs: usize,
}

impl Default for BucketConfig {
    fn default() -> Self {
        Self {
            balance_abs_threshold: 32,
            balance_rel_threshold: 1.0001,
            bucket_adjust_interval_secs: 5,
        }
    }
}

/// Helper function to filter healthy workers and return their indices
pub(crate) fn get_healthy_worker_indices(workers: &[Arc<dyn Worker>]) -> Vec<usize> {
    workers
        .iter()
        .enumerate()
        .filter(|(_, w)| w.is_healthy() && w.circuit_breaker().can_execute())
        .map(|(idx, _)| idx)
        .collect()
}

/// Helper function to normalize model_id to a key for policy lookups.
///
/// Returns UNKNOWN_MODEL_ID for empty model_ids to ensure consistent behavior
/// across single-model and multi-model deployments.
#[inline]
pub(crate) fn normalize_model_key(model_id: &str) -> &str {
    if model_id.is_empty() {
        crate::core::UNKNOWN_MODEL_ID
    } else {
        model_id
    }
}

/// Information passed to policy for worker selection
#[derive(Debug, Clone, Default)]
pub struct SelectWorkerInfo<'a> {
    /// Request text for cache-aware routing
    pub request_text: Option<&'a str>,
    /// Tokenized request for prefix-hash routing
    /// Used by PrefixHashPolicy for token-based prefix hashing
    pub tokens: Option<&'a [u32]>,
    /// HTTP headers for header-based routing policies
    /// Policies can extract routing information from headers like:
    /// - X-SMG-Target-Worker: Direct routing to a specific worker by index
    /// - X-SMG-Routing-Key: Consistent hash routing for session affinity
    pub headers: Option<&'a http::HeaderMap>,
    /// Chars-per-token ratio used to estimate this request's token count
    /// from its text length. Consumed by capacity-aware routing decisions
    /// (cache_aware's per-worker capacity gate) so their token estimates
    /// match the units workers are charged in at dispatch
    /// (`WorkerLoadGuard::new_weighted`, which divides the same text length
    /// by this rate, falling back to 3.5). `0.0` means "unset" — consumers
    /// apply their own conservative fallback (3.5, the router default).
    pub chars_per_token: f32,
    /// Pre-computed hash ring for O(log n) consistent hashing
    /// Built and cached by WorkerRegistry, passed through to avoid per-request rebuilds
    pub hash_ring: Option<Arc<HashRing>>,
}

/// Extra data produced by worker selection, for callers that need more than
/// the chosen index.
///
/// `match_rate` is the raw prefix-match ratio (0.0..=1.0) computed by
/// cache-aware policies against the per-worker affinity tree. Policies that
/// don't track cache affinity report 0.0. The HTTP router consumes this for
/// time-to-first-token estimation when a request carries an
/// `X-TTFT-Budget-Ms` header (see `route_typed_request_once`).
#[derive(Debug, Clone, Copy, Default)]
pub struct SelectionInfo {
    pub index: usize,
    pub match_rate: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{BasicWorkerBuilder, WorkerType};

    #[tokio::test]
    async fn test_get_healthy_worker_indices() {
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(
                BasicWorkerBuilder::new("http://w1:8000")
                    .worker_type(WorkerType::Regular)
                    .api_key("test_api_key")
                    .build(),
            ),
            Arc::new(
                BasicWorkerBuilder::new("http://w2:8000")
                    .worker_type(WorkerType::Regular)
                    .api_key("test_api_key2")
                    .build(),
            ),
            Arc::new(
                BasicWorkerBuilder::new("http://w3:8000")
                    .worker_type(WorkerType::Regular)
                    .api_key("test_api_key")
                    .build(),
            ),
        ];

        // All healthy initially
        let indices = get_healthy_worker_indices(&workers);
        assert_eq!(indices, vec![0, 1, 2]);

        // Mark one unhealthy
        workers[1].set_healthy(false);
        let indices = get_healthy_worker_indices(&workers);
        assert_eq!(indices, vec![0, 2]);
    }
}
