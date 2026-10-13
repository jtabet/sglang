use std::{sync::Arc, time::Instant};

use axum::{
    body::{to_bytes, Body},
    extract::Request,
    http::{header::CONTENT_TYPE, HeaderMap, HeaderValue, Method, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use futures_util::{stream, StreamExt};
use reqwest::Client;
use tracing::{debug, error, warn};

use crate::{
    app_context::AppContext,
    config::types::RetryConfig,
    core::{
        is_retryable_status, AttachedBody, ConnectionMode, RetryExecutor, Worker, WorkerLoadGuard,
        WorkerRegistry, WorkerType, UNKNOWN_MODEL_ID,
    },
    observability::{
        events::{self, Event},
        metrics::{bool_to_static_str, metrics_labels, Metrics},
        otel_trace::inject_trace_context_http,
    },
    policies::{PolicyRegistry, SelectWorkerInfo},
    protocols::{
        chat::ChatCompletionRequest,
        classify::ClassifyRequest,
        common::GenerationRequest,
        completion::CompletionRequest,
        embedding::EmbeddingRequest,
        generate::GenerateRequest,
        rerank::{RerankRequest, RerankResponse, RerankResult},
        responses::{ResponsesGetParams, ResponsesRequest},
    },
    routers::{
        error::{self, extract_error_code_from_response},
        grpc::utils::{error_type_from_status, route_to_endpoint},
        header_utils,
        streaming_utils::{BreakerTrackedStream, TtftObserverStream},
        RouterTrait,
    },
};

/// Regular router that uses injected load balancing policies
pub struct Router {
    worker_registry: Arc<WorkerRegistry>,
    policy_registry: Arc<PolicyRegistry>,
    client: Client,
    dp_aware: bool,
    enable_igw: bool,
    retry_config: RetryConfig,
    /// Prefill throughput estimate (tokens/second/worker) for TTFT budget
    /// gating. 0 disables the gate (X-TTFT-Budget-Ms ignored).
    ttft_prefill_rate_tps: f32,
    /// Chars-per-token used to convert routing text into estimated tokens.
    ttft_chars_per_token: f32,
    /// Max concurrent background warmups (config).
    /// Worker URLs with a background warmup prefill currently running. A
    /// budget-bearing request routed to one of these is declined until the
    /// warmup lands — the affinity tree was already updated at decline time,
    /// so the estimate alone would wrongly claim the cache is warm.
    ttft_warming: Arc<TtftWarmingRegistry>,
    /// Max concurrent background warmups (from RouterConfig).
    ttft_max_warmups: usize,
    /// Per-worker EWMA of measured prefill throughput. Seeded from
    /// `ttft_prefill_rate_tps` and updated on each warmup completion.
    ttft_prefill_rates: Arc<TtftPrefillRateRegistry>,
    /// Receipts proving (worker, prefix) KV actually resides in the engine's
    /// L0 right now. The tree is a promise; these are verification.
    ttft_verified_warm: Arc<TtftVerifiedWarmRegistry>,
}

/// Tracks background warmup prefills per worker URL. Not clone-safe state —
/// the Router holds one Arc; both the request path and spawned warmup tasks
/// share it.
#[derive(Debug, Default)]
struct TtftWarmingRegistry {
    in_flight: std::sync::Mutex<std::collections::HashSet<String>>,
}

impl TtftWarmingRegistry {
    fn is_warming(&self, url: &str) -> bool {
        self.in_flight.lock().unwrap().contains(url)
    }

    /// Register a warmup for `url`. Fails when one is already running for the
    /// worker or the global cap is reached — the caller then declines without
    /// spawning.
    fn register(&self, url: &str, max_warmups: usize) -> bool {
        let mut guard = self.in_flight.lock().unwrap();
        if guard.contains(url) || guard.len() >= max_warmups {
            return false;
        }
        guard.insert(url.to_string());
        true
    }

    fn release(&self, url: &str) {
        self.in_flight.lock().unwrap().remove(url);
    }

    fn len(&self) -> usize {
        self.in_flight.lock().unwrap().len()
    }
}

/// Per-worker, per-cache-band EWMA of measured prefill throughput
/// (tokens/second).
///
/// With the L0/L1/L2 cache hierarchy, a "match_rate=0.94" prompt is NOT cheap
/// unless the KV currently lives in VRAM (L0): when the prefix was offloaded
/// to the central mooncake store (L1), the engine must load the pages back
/// before prefilling the tail, and load-back dominates — measured effective
/// rates of 75–300 tok/s vs 2k–30k tok/s for a true L0 hit. A single
/// per-worker EWMA is polluted by the fast L0-hot completions and keeps
/// opening the budget gate for L1-warm requests that then blow the caller's
/// TTFT budget (fetch timeout → escalation → cloud).
///
/// So the registry keeps a separate EWMA per (worker, band), where the band is
/// derived from the match_rate the selector observed for THIS request:
///   - Cold    (match_rate < 0.10): full prefill, no KV to load.
///   - Partial (0.10..=0.60):      mixed hit; load-back of some pages.
///   - Warm    (> 0.60):           KV exists but may live in L1/L2 —
///                                 load-back cost dominates when L0-cold.
/// Every measurement (live first-chunk and warmup) is recorded as
/// UNMATCHED tokens / latency, the same quantity the gate's estimate
/// divides by, so the band holds a true per-uncached-token rate.
///
/// The tree that supplies match_rate is a shadow model of the engine cache:
/// it is updated at selection time, but the engine's actual L0 state changes
/// with every request (notably the background warmups, which populate the
/// cache the tree already claims). A sample whose implied rate jumps far
/// above the band's current value is therefore a stale-tree artifact, not a
/// GPU speedup, and `record_guarded` rejects it (logged with the reason) —
/// the conservative direction for the gate is always the safe one.
#[derive(Debug, Default)]
struct TtftPrefillRateRegistry {
    rates: std::sync::Mutex<std::collections::HashMap<(String, u8), f32>>,
}

impl TtftPrefillRateRegistry {
    /// EWMA smoothing factor — 0.3 gives ~3-sample response time. Higher
    /// reacts faster but is noisier; lower is smoother but slower to adapt
    /// to a thermal throttle event.
    const ALPHA: f32 = 0.3;

    /// Band index for a match_rate. `band(match)` is stable so `get`/`record`
    /// always agree on the bucket.
    fn band(match_rate: f32) -> u8 {
        let mr = match_rate.clamp(0.0, 1.0);
        if mr < 0.10 {
            0 // cold
        } else if mr <= 0.60 {
            1 // partial
        } else {
            2 // warm (≥ 0.60)
        }
    }

    fn get(&self, url: &str, match_rate: f32, fallback: f32) -> f32 {
        let mut guard = self.rates.lock().unwrap();
        *guard
            .entry((url.to_string(), Self::band(match_rate)))
            .or_insert(fallback)
    }

    /// Update the band EWMA with a new measurement. Seeded with `fallback` if
    /// this is the first sample for the band.
    fn record(&self, url: &str, match_rate: f32, measured_tps: f32, fallback: f32) {
        let band = Self::band(match_rate);
        let mut guard = self.rates.lock().unwrap();
        let entry = guard
            .entry((url.to_string(), band))
            .or_insert(fallback);
        *entry = Self::ALPHA * measured_tps + (1.0 - Self::ALPHA) * *entry;
    }
}

/// Receipts: proof that worker W's engine L0 ACTUALLY holds prefix P right
/// now (within a TTL).
///
/// The cache tree is a promise registered at selection time — including for
/// 425'd requests. But a promise can be unfulfilled: its warmup may be
/// rejected (capacity full) or still in flight, in which case every
/// high-match request routed to that tenant means a full cold prefill of
/// 60–400s — the exact fetch-timeout→escalation leak. A receipt is only
/// issued when a forwarded request on (W, P) really did produce its first
/// chunk fast (or a warmup on (W, P) completed): for a large prompt that is
/// only possible on an L0 hit (a 48k-token cold prefill takes ~70s at the
/// measured ~700 tps), so a receipt is trustworthy evidence the KV is
/// resident. The gate forwards high-match requests only on receipts; without
/// one it re-estimates as if cold.
#[derive(Debug, Default)]
struct TtftVerifiedWarmRegistry {
    entries: std::sync::Mutex<std::collections::HashMap<(String, u64), Instant>>,
}

impl TtftVerifiedWarmRegistry {
    /// Receipt TTL — a backstop only, NOT the primary staleness defense.
    ///
    /// The engine can evict L0 entries under VRAM pressure and reports nothing
    /// to the router, and a pod restart wipes L0 entirely without the router
    /// ever seeing it. The PRIMARY defense against a stale receipt (which would
    /// forward the next same-prefix request into a cold prefill → 90s fetch
    /// timeout → patchbay cooldown storm) is the REACTIVE invalidation on the
    /// live path: the slow-first-chunk hook (>= PROOF_MAX_SECS on a high-match
    /// request) and the no-chunk-on-drop hook (pb abort before first chunk)
    /// both drop the (worker, prefix) receipt in a SINGLE request — the request
    /// that actually proves the KV is no longer L0-resident. They bound the
    /// stale-receipt window to one request, which is strictly better than any
    /// TTL can do, because they react to the actual loss rather than a timer.
    ///
    /// The TTL therefore only has to cover IDLE prefixes — a prefix whose
    /// receipt goes stale while nothing requests it — and that case is
    /// harmless: no request, no bad forward. So keep it GENEROUS (15 min): a
    /// short TTL is actively harmful, because craft traffic revisits a shared
    /// 16K prefix over idle gaps > 2 min, and a 2-min TTL expires every
    /// warmup-minted receipt before the next organic request arrives — the
    /// system never reaches steady state and collapses to constant 425 → cloud
    /// (the receipt-thrash regression). 15 min lets a re-warmed prefix stick
    /// long enough for organic traffic to ride it, while the reactive hooks
    /// still kill any receipt the moment a real request exposes it as stale.
    const TTL: std::time::Duration = std::time::Duration::from_secs(15 * 60);

    /// First-chunk latency under which a forwarded request proves the KV was
    /// resident in L0 — 2s is far above a true L0-hit first chunk (≤ ~0.5s
    /// tail prefill) and far below any cold prefill of a budget-relevant
    /// prompt (a 6k-token cold prefill is already ~9s at ~700 tps), so a
    /// small prompt's fast completion can never mint a false large-prompt
    /// receipt that matters: false receipts only exist for prompts whose
    /// ENTIRE cold prefill also fits the budget, which is harmless.
    const PROOF_MAX_SECS: f64 = 2.0;

    /// Length (chars) of the stable prefix hashed for receipt identity. Two
    /// turns of a conversation share at least this much (system prompt +
    /// early context); two different conversations do not. Content shared by
    /// two conversations colliding on this hash is a REAL shared engine
    /// prefix (the engine's radix keys on prefixes too), so a collision never
    /// mints a receipt for KV that does not exist.
    const PREFIX_HASH_CHARS: usize = 16_384;

    /// Stable hash of the request's shared prefix. Uses SipHash via
    /// std's DefaultHasher — a 64-bit collision on honest content is
    /// negligible, and content-derived collisions are real shared prefixes.
    fn hash_prefix(text: &str) -> u64 {
        use std::hash::{Hash, Hasher};
        let end = text
            .char_indices()
            .nth(Self::PREFIX_HASH_CHARS)
            .map(|(i, _)| i)
            .unwrap_or(text.len());
        let mut h = std::collections::hash_map::DefaultHasher::new();
        text[..end].hash(&mut h);
        h.finish()
    }

    /// Mark (worker, prefix) as verified-resident now. Also prunes expired
    /// receipts (cheap amortization while the lock is held).
    fn mark(&self, url: &str, prefix_hash: u64) {
        let mut guard = self.entries.lock().unwrap();
        guard.insert((url.to_string(), prefix_hash), Instant::now());
        guard.retain(|_, at| at.elapsed() < Self::TTL);
    }

    /// True if (worker, prefix) holds an unexpired receipt.
    fn is_verified(&self, url: &str, prefix_hash: u64) -> bool {
        let guard = self.entries.lock().unwrap();
        guard
            .get(&(url.to_string(), prefix_hash))
            .map(|at| at.elapsed() < Self::TTL)
            .unwrap_or(false)
    }

    /// Drop the (worker, prefix) receipt if present. Called when a forwarded
    /// high-match request's first chunk turns out SLOW — direct proof the KV
    /// is no longer L0-resident (it was evicted under VRAM pressure, which the
    /// engine never reports). A receipt left in place after that would send the
    /// next same-prefix request into the same cold prefill → 90s fetch timeout.
    /// The accompanying background warmup re-warms and re-mints a fresh receipt.
    fn invalidate(&self, url: &str, prefix_hash: u64) {
        let mut guard = self.entries.lock().unwrap();
        guard.remove(&(url.to_string(), prefix_hash));
    }
}

/// Estimated milliseconds to the first token for a request routed with the
/// given cache `match_rate` (0.0..=1.0). Uncached characters are converted
/// to tokens with `chars_per_token` and divided by `prefill_tps`.
pub(crate) fn estimate_ttft_ms(
    text: &str,
    match_rate: f32,
    chars_per_token: f32,
    prefill_tps: f32,
) -> u64 {
    let input_chars = text.chars().count();
    if input_chars == 0 || prefill_tps <= 0.0 {
        return 0;
    }
    let uncached_chars = input_chars as f32 * (1.0 - match_rate.clamp(0.0, 1.0));
    let est_tokens = uncached_chars / chars_per_token.max(0.1);
    (est_tokens / prefill_tps.max(0.1) * 1000.0) as u64
}

/// Parse the caller's time-to-first-token budget (`X-TTFT-Budget-Ms`).
/// Absent or malformed header → None (no budgeting for this request).
fn ttft_budget_ms(headers: Option<&HeaderMap>) -> Option<u64> {
    let value = headers?.get("x-ttft-budget-ms")?.to_str().ok()?;
    value.trim().parse::<u64>().ok()
}

/// Match-rate boundary above which the selector routes by the tree's tenant
/// (the worker where the prefix was last registered). Must equal the
/// deployment's `--cache-threshold`; below it, selection is load-based and
/// the band estimate is already near-cold, so no receipt is required.
const RECEIPT_MATCH_THRESHOLD: f32 = 0.3;

/// Context for recording live-request TTFT measurements into the per-worker,
/// per-band prefill-rate EWMA. Populated by `route_typed_request_once` only
/// when a TTFT budget header was present (i.e. the caller is patchbay and the
/// measurement is relevant to the gate); `None` otherwise so ordinary
/// requests pay zero overhead beyond one closure allocation.
#[derive(Clone, Copy)]
struct TtftRecordCtx {
    match_rate: f32,
    est_tokens: u64,
    seed_tps: f32,
    prefix_hash: u64,
}

impl std::fmt::Debug for Router {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Router")
            .field("worker_registry", &self.worker_registry)
            .field("policy_registry", &self.policy_registry)
            .field("client", &self.client)
            .field("dp_aware", &self.dp_aware)
            .field("enable_igw", &self.enable_igw)
            .field("retry_config", &self.retry_config)
            .finish()
    }
}

impl Router {
    /// Create a new router with injected policy and client
    pub async fn new(ctx: &Arc<AppContext>) -> Result<Self, String> {
        Ok(Router {
            worker_registry: ctx.worker_registry.clone(),
            policy_registry: ctx.policy_registry.clone(),
            client: ctx.client.clone(),
            dp_aware: ctx.router_config.dp_aware,
            enable_igw: ctx.router_config.enable_igw,
            retry_config: ctx.router_config.effective_retry_config(),
            ttft_prefill_rate_tps: ctx.router_config.ttft_prefill_rate_tps,
            ttft_chars_per_token: ctx.router_config.ttft_chars_per_token,
            ttft_max_warmups: ctx.router_config.ttft_max_warmups,
            ttft_warming: Arc::new(TtftWarmingRegistry::default()),
            ttft_prefill_rates: Arc::new(TtftPrefillRateRegistry::default()),
            ttft_verified_warm: Arc::new(TtftVerifiedWarmRegistry::default()),
        })
    }

    fn select_first_worker(&self) -> Result<String, String> {
        let workers = self.worker_registry.get_all();
        let healthy_workers: Vec<_> = workers.iter().filter(|w| w.is_healthy()).collect();
        if healthy_workers.is_empty() {
            Err("No workers are available".to_string())
        } else {
            Ok(healthy_workers[0].url().to_string())
        }
    }

    async fn proxy_get_request(&self, req: Request<Body>, endpoint: &str) -> Response {
        let headers = header_utils::copy_request_headers(&req);

        match self.select_first_worker() {
            Ok(worker_url) => {
                let mut request_builder = self.client.get(format!("{}/{}", worker_url, endpoint));
                for (name, value) in headers {
                    if header_utils::should_forward_request_header(&name) {
                        request_builder = request_builder.header(name, value);
                    }
                }

                match request_builder.send().await {
                    Ok(res) => {
                        let status = StatusCode::from_u16(res.status().as_u16())
                            .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);

                        // Preserve headers from backend
                        let response_headers =
                            header_utils::preserve_response_headers(res.headers());

                        match res.bytes().await {
                            Ok(body) => {
                                let mut response = Response::new(Body::from(body));
                                *response.status_mut() = status;
                                *response.headers_mut() = response_headers;
                                response
                            }
                            Err(e) => error::internal_error(
                                "read_response_failed",
                                format!("Failed to read response: {}", e),
                            ),
                        }
                    }
                    Err(e) => convert_reqwest_error(e),
                }
            }
            Err(e) => error::service_unavailable("no_workers", e),
        }
    }

    /// Select worker for a specific model considering circuit breaker state.
    /// Also reports the policy's match_rate for TTFT estimation.
    async fn select_worker_for_model(
        &self,
        model_id: Option<&str>,
        text: Option<&str>,
        headers: Option<&HeaderMap>,
    ) -> Option<(Arc<dyn Worker>, f32)> {
        let effective_model_id = if !self.enable_igw { None } else { model_id };

        // Get workers for the specified model O(1), filtered by connection mode
        let workers = self.worker_registry.get_workers_filtered(
            effective_model_id,
            Some(WorkerType::Regular),
            Some(ConnectionMode::Http),
            None,  // any runtime type
            false, // get all workers, we'll filter by is_available() next
        );

        // Filter out workers that have a TTFT warmup in flight — they are
        // busy doing a background prefill and won't be able to serve a new
        // request within the fetch timeout. If ALL workers are warming,
        // fall back to the full set (the request will likely time out, but
        // at least we don't return 503 when workers are technically alive).
        let available: Vec<Arc<dyn Worker>> = workers
            .iter()
            .filter(|w| w.is_available())
            .cloned()
            .collect();
        if available.is_empty() {
            return None;
        }
        let available: Vec<Arc<dyn Worker>> = available
            .iter()
            .filter(|w| !self.ttft_warming.is_warming(w.url()))
            .cloned()
            .collect();
        let available = if available.is_empty() {
            // All workers warming — fall back to all available workers
            workers
                .iter()
                .filter(|w| w.is_available())
                .cloned()
                .collect()
        } else {
            available
        };

        // Get the appropriate policy for this model
        let policy = match model_id {
            Some(model) => self.policy_registry.get_policy_or_default(model),
            None => self.policy_registry.get_default_policy(),
        };

        // Get cached hash ring for consistent hashing (O(log n) lookup)
        let hash_ring = self
            .worker_registry
            .get_hash_ring(effective_model_id.unwrap_or(UNKNOWN_MODEL_ID));

        let selection = policy
            .select_worker_with_info(
                &available,
                &SelectWorkerInfo {
                    request_text: text,
                    tokens: None, // HTTP doesn't have tokens, use gRPC for PrefixHash
                    headers,
                    hash_ring,
                },
            )
            .await?;

        // Record worker selection metric (Layer 3)
        Metrics::record_worker_selection(
            metrics_labels::WORKER_REGULAR,
            metrics_labels::CONNECTION_HTTP,
            model_id.unwrap_or(UNKNOWN_MODEL_ID),
            policy.name(),
        );

        Some((available[selection.index].clone(), selection.match_rate))
    }

    pub async fn route_typed_request<T: GenerationRequest + serde::Serialize + Clone + 'static>(
        &self,
        headers: Option<&HeaderMap>,
        typed_req: &T,
        route: &'static str,
        model_id: Option<&str>,
    ) -> Response {
        let start = Instant::now();
        let is_stream = typed_req.is_stream();
        let text = typed_req.extract_text_for_routing();
        let model = model_id.unwrap_or(UNKNOWN_MODEL_ID);
        let endpoint = route_to_endpoint(route);

        // Record request start (Layer 2)
        Metrics::record_router_request(
            metrics_labels::ROUTER_HTTP,
            metrics_labels::BACKEND_REGULAR,
            metrics_labels::CONNECTION_HTTP,
            model,
            endpoint,
            bool_to_static_str(is_stream),
        );

        let response = RetryExecutor::execute_response_with_retry(
            &self.retry_config,
            // operation per attempt
            |_: u32| async {
                let res = self
                    .route_typed_request_once(headers, typed_req, route, model_id, is_stream, &text)
                    .await;

                // Need to be outside `route_typed_request_once` because that function has multiple return paths
                Metrics::record_router_upstream_response(
                    metrics_labels::ROUTER_HTTP,
                    res.status().as_u16(),
                    extract_error_code_from_response(&res),
                );

                res
            },
            // should_retry predicate
            |res, _attempt| is_retryable_status(res.status()),
            // on_backoff hook
            |delay, attempt| {
                // Layer 3 worker metrics
                Metrics::record_worker_retry(metrics_labels::WORKER_REGULAR, endpoint);
                Metrics::record_worker_retry_backoff(attempt, delay);
            },
            // on_exhausted hook
            || {
                Metrics::record_worker_retries_exhausted(metrics_labels::WORKER_REGULAR, endpoint);
            },
        )
        .await;

        if response.status().is_success() {
            let duration = start.elapsed();
            Metrics::record_router_duration(
                metrics_labels::ROUTER_HTTP,
                metrics_labels::BACKEND_REGULAR,
                metrics_labels::CONNECTION_HTTP,
                model,
                endpoint,
                duration,
            );
        } else if !is_retryable_status(response.status()) {
            Metrics::record_router_error(
                metrics_labels::ROUTER_HTTP,
                metrics_labels::BACKEND_REGULAR,
                metrics_labels::CONNECTION_HTTP,
                model,
                endpoint,
                error_type_from_status(response.status()),
            );
        }

        response
    }

    async fn route_typed_request_once<T: GenerationRequest + serde::Serialize + Clone + 'static>(
        &self,
        headers: Option<&HeaderMap>,
        typed_req: &T,
        route: &'static str,
        model_id: Option<&str>,
        is_stream: bool,
        text: &str,
    ) -> Response {
        let req_start = std::time::Instant::now();
        let select_start = std::time::Instant::now();
        let text_len = text.chars().count();
        let (mut worker, match_rate) = match self
            .select_worker_for_model(model_id, Some(text), headers)
            .await
        {
            Some((w, mr)) => (w, mr),
            None => {
                return error::service_unavailable(
                    "no_available_workers",
                    "No available workers (all circuits open or unhealthy)",
                );
            }
        };

        let select_elapsed = select_start.elapsed();
        if select_elapsed.as_millis() > 100 {
            warn!(
                "route_typed_request_once: select_worker took {}ms for {}char text (match_rate={:.2})",
                select_elapsed.as_millis(),
                text_len,
                match_rate,
            );
        }

        // ── TTFT budget gate (orchestrel) ─────────────────────────────────
        // When the caller declares a time-to-first-token budget and the
        // cache-aware estimate says this worker cannot meet it, decline
        // immediately with 425 while keeping the prefill running in the
        // background (prefill-then-abort warmup). The affinity tree was
        // already updated by select_worker_with_info, so the NEXT request
        // with the same prefix sees a high match_rate — and the real engine
        // cache becomes warm as the warmup lands. While a warmup is in
        // flight for the selected worker, budgeted requests keep declining:
        // the tree already claims affinity but the engine cache is not
        // populated yet.
        let gate_active = self.ttft_prefill_rate_tps > 0.0
            && ttft_budget_ms(headers).is_some();
        if self.ttft_prefill_rate_tps > 0.0 {
            if let Some(budget_ms) = ttft_budget_ms(headers) {
                // Receipt check: the tree's tenant claim is only trustworthy
                // when the engine L0 state for (worker, prefix) is actually
                // verified. Without a receipt we estimate as cold — a promise
                // whose warmup was capacity-rejected or is still in flight
                // would otherwise forward into a 90s+ cold prefill.
                let prefix_hash = TtftVerifiedWarmRegistry::hash_prefix(text);
                // Estimate inputs. When the request matches the tree above
                // the cache threshold but the selected worker holds no
                // verified L0 receipt for this prefix, the match is an
                // UNFULFILLED PROMISE (the warmup that would have populated
                // it was capacity-rejected, is still in flight, or the KV was
                // evicted) — assume NOTHING is resident and use the cold
                // band's rate with match_rate 0.0, so every token is charged.
                //
                // FLEXIBLE ASSIGNMENT (mooncake store is shared): before
                // declining, check whether ANOTHER healthy worker holds a
                // verified L0 receipt for the same prefix — the KV bytes are
                // content-addressed in the shared store, so any worker can
                // serve them, and the one holding an L0 receipt gives the
                // sub-2s path regardless of which worker prefilled it. Route
                // there instead of 425 → cloud. If no receipt anywhere, fall
                // through to the cold estimate (the warmup/425 path heals it).
                let prefix_hash = TtftVerifiedWarmRegistry::hash_prefix(text);
                let mut verified = self
                    .ttft_verified_warm
                    .is_verified(worker.url(), prefix_hash);
                if !verified && match_rate > RECEIPT_MATCH_THRESHOLD {
                    let alternates = self.worker_registry.get_workers_filtered(
                        if !self.enable_igw { None } else { model_id },
                        Some(WorkerType::Regular),
                        Some(ConnectionMode::Http),
                        None,
                        false,
                    );
                    if let Some(alt) = alternates.iter().find(|w| {
                        w.url() != worker.url()
                            && w.is_available()
                            && !self.ttft_warming.is_warming(w.url())
                            && self.ttft_verified_warm.is_verified(w.url(), prefix_hash)
                    }) {
                        tracing::info!(
                            "TTFT cross-worker receipt hit: {} unverified for prefix but \
                             {} holds a verified L0 receipt — mooncake store is shared, \
                             rerouting to the receipt holder",
                            worker.url(),
                            alt.url()
                        );
                        worker = alt.clone();
                        verified = true;
                    }
                }
                let unverified = match_rate > RECEIPT_MATCH_THRESHOLD && !verified;
                let (est_match_rate, prefill_tps) = if unverified {
                    tracing::info!(
                        "TTFT receipt miss: prefix matches {:.0}% in tree but {} has no verified \
                         L0 receipt for it — estimating cold",
                        match_rate * 100.0,
                        worker.url()
                    );
                    (
                        0.0,
                        self.ttft_prefill_rates
                            .get(worker.url(), 0.0, self.ttft_prefill_rate_tps),
                    )
                } else {
                    // The band comes from this request's match_rate so
                    // L1-warm (load-back-dominated) requests are estimated
                    // with load-back-inclusive rates, not L0-hot ones.
                    (
                        match_rate,
                        self.ttft_prefill_rates
                            .get(worker.url(), match_rate, self.ttft_prefill_rate_tps),
                    )
                };
                let est_ms = estimate_ttft_ms(
                    text,
                    est_match_rate,
                    self.ttft_chars_per_token,
                    prefill_tps,
                );
                // Safety margin: only forward if the estimate is comfortably
                // under budget. Requests near the boundary have estimation
                // error (match_rate approximation, EWMA lag) that can push
                // actual prefill over the fetch timeout — which escalates the
                // worker into cooldown and sends everything to cloud. 0.5
                // keeps a 50% margin (5s est for a 10s budget) — large enough
                // to absorb estimation error and queue overhead without
                // hitting the fetch timeout.
                let gated_budget_ms = (budget_ms as f32 * 0.5) as u64;
                if est_ms > gated_budget_ms {
                    let warmup_start = std::time::Instant::now();
                    let reason = format!(
                        "estimated ttft {}ms exceeds budget {}ms (gated {}ms, match_rate {:.2})",
                        est_ms, budget_ms, gated_budget_ms, match_rate
                    );
                    debug!(
                        "TTFT budget gate: declining {} → {} ({})",
                        model_id.unwrap_or(UNKNOWN_MODEL_ID),
                        worker.url(),
                        reason
                    );
                    self.maybe_spawn_warmup(
                        headers, typed_req, route, &worker, is_stream,
                        est_ms, budget_ms, &text, match_rate,
                    );
                    let total_425_latency = req_start.elapsed();
                    let warmup_elapsed = warmup_start.elapsed();
                    warn!(
                        "TTFT 425: total={}ms (select={}ms, warmup={}ms, text={}chars, match_rate={:.2}, est={}ms, gated={}ms)",
                        total_425_latency.as_millis(),
                        select_elapsed.as_millis(),
                        warmup_elapsed.as_millis(),
                        text_len,
                        match_rate,
                        est_ms,
                        gated_budget_ms,
                    );
                    return error::too_early(
                        "prefill_budget_exceeded",
                        format!(
                            "Prompt cache cold for this request: {} — skipping this request; \
                             the prefill continues in the background and the cache will be \
                             warm for your next calls.",
                            reason
                        ),
                    );
                }
            }
        }

        let policy = match model_id {
            Some(model) => self.policy_registry.get_policy_or_default(model),
            None => self.policy_registry.get_default_policy(),
        };

        let load_guard = ["cache_aware", "manual"]
            .contains(&policy.name())
            .then(|| WorkerLoadGuard::new(worker.clone(), headers));

        // Note: Using borrowed reference avoids heap allocation
        events::RequestSentEvent { url: worker.url() }.emit();
        let mut headers_with_trace = headers.cloned().unwrap_or_default();
        inject_trace_context_http(&mut headers_with_trace);
        let headers = Some(&headers_with_trace);

        // Only budgeted (patchbay) requests feed the live TTFT recorder.
        // est_tokens mirrors the warmup recorder: full text / chars_per_token
        // — the band absorbs the cached fraction.
        let ttft_ctx = if gate_active {
            Some(TtftRecordCtx {
                match_rate,
                est_tokens: (text_len as f32 / self.ttft_chars_per_token.max(0.1))
                    .max(1.0) as u64,
                seed_tps: self.ttft_prefill_rate_tps,
                prefix_hash: TtftVerifiedWarmRegistry::hash_prefix(text),
            })
        } else {
            None
        };

        let response = self
            .send_typed_request(
                headers, typed_req, route, &worker, is_stream, load_guard, ttft_ctx,
            )
            .await;

        events::RequestReceivedEvent {}.emit();

        let status = response.status();
        // For streaming responses, the wrapped body (`BreakerTrackedStream`)
        // records the circuit-breaker outcome once the stream actually
        // terminates (success on clean end, failure on mid-stream error).
        // Recording it eagerly here based on the initial status code would
        // mask "200-then-broken" workers — every request would tick a
        // success before the stream had a chance to error out.
        if !is_stream {
            worker.record_outcome(status.is_success());
        }

        // Record worker errors for server errors (5xx)
        if status.is_server_error() {
            Metrics::record_worker_error(
                metrics_labels::WORKER_REGULAR,
                metrics_labels::CONNECTION_HTTP,
                error_type_from_status(status),
            );
        }

        response
    }

    /// Spawn a background warmup prefill for a TTFT-budget decline.
    ///
    /// Re-issues the request against the selected worker with
    /// `max_tokens: 1` and `stream: true` so the engine pays the prefill and
    /// the first streamed token tells us the prompt cache is populated, then
    /// aborts — no decode tokens are generated (the real answer for the
    /// declined request came from another route; these tokens would be
    /// discarded anyway).
    ///
    /// No-op when another warmup is already running for this worker or the
    /// global warmup cap is reached (decline without warming).
    ///
    /// The warming registry entry is released once the first body byte
    /// Arrives (prefill done) or on any error — either way the next budgeted
    /// request re-evaluates the (now accurate) match_rate estimate.
    fn maybe_spawn_warmup<T: GenerationRequest + serde::Serialize + Clone + 'static>(
        &self,
        headers: Option<&HeaderMap>,
        typed_req: &T,
        route: &'static str,
        worker: &Arc<dyn Worker>,
        _is_stream: bool,
        _est_ms: u64,
        _budget_ms: u64,
        text: &str,
        match_rate: f32,
    ) {
        // Large prompts are exactly the ones that cannot cold-prefill within
        // a fetch budget, so they are the ones that most need a background
        // warmup to populate the external cache. Serialization already runs
        // inside the tokio::spawn below, so a large body never blocks the 425
        // response. Do not skip them.

        if !self
            .ttft_warming
            .register(worker.url(), self.ttft_warming_capacity())
        {
            debug!(
                "TTFT warmup cap reached or warmup already running for {} — declining without warmup",
                worker.url()
            );
            return;
        }

        // Clone the typed request so we can move it into the spawned task.
        // The serde_json::to_value + request building happen inside the
        // spawn so the 425 response is not blocked by serialization.
        let typed_req_clone = typed_req.clone();
        let worker_url = self.worker_base_url(worker.url()).to_string();
        let endpoint = format!("{}{}", worker_url, route);
        let api_key = worker.api_key().clone();
        let client = self.client.clone();
        let header_pairs: Vec<(String, String)> = headers
            .map(|h| {
                h.iter()
                    .filter(|(name, _)| {
                        name.as_str() != "authorization" && name.as_str() != "content-length"
                    })
                    .filter_map(|(name, value)| {
                        Some((name.as_str().to_string(), value.to_str().ok()?.to_string()))
                    })
                    .collect()
            })
            .unwrap_or_default();

        let warming = self.ttft_warming.clone();
        let prefill_rates = self.ttft_prefill_rates.clone();
        let verified_warm = self.ttft_verified_warm.clone();
        let worker_url_str = worker.url().to_string();
        let chars_per_token = self.ttft_chars_per_token;
        let warmup_match_rate = match_rate;
        let warmup_seed_tps = self.ttft_prefill_rate_tps;
        let warmup_prefix_hash = TtftVerifiedWarmRegistry::hash_prefix(text);
        // Estimate the token count the engine will prefill. We use the same
        // chars_per_token conversion as the budget gate — the match_rate is
        // not known here (the tree was updated but we don't have it), so we
        // use the full text length. This slightly overestimates tokens for
        // partially-cached requests, which makes the recorded tps slightly
        // conservative — safe for the budget gate.
        let est_tokens =
            (text.chars().count() as f32 / chars_per_token.max(0.1)).max(1.0) as u64;
        tokio::spawn(async move {
            // Serialize + build request inside the spawn so it doesn't
            // block the 425 response.
            let mut warm_value = match serde_json::to_value(&typed_req_clone) {
                Ok(v) => v,
                Err(e) => {
                    warn!("TTFT warmup: failed to serialize request: {}", e);
                    warming.release(&worker_url_str);
                    return;
                }
            };
            if let Some(map) = warm_value.as_object_mut() {
                map.insert("max_tokens".to_string(), serde_json::json!(1));
                map.insert("stream".to_string(), serde_json::json!(true));
                map.insert("ignore_eos".to_string(), serde_json::json!(false));
            }

            let mut request = client.post(&endpoint).json(&warm_value);
            if let Some(key) = &api_key {
                request = request.header("Authorization", format!("Bearer {}", key));
            }
            for (name, value) in &header_pairs {
                request = request.header(name.as_str(), value.as_str());
            }

            let started = Instant::now();
            match request.send().await {
                Ok(resp) => {
                    // Drain the whole body. We set `max_tokens: 1` and
                    // `stream: true`, so the engine prefills the prompt and
                    // emits exactly one decode token before ending the stream
                    // itself with FINISH. Crucially we must NOT drop the body
                    // after the first byte: dropping closes the connection and
                    // sglang aborts the request, which calls
                    // `release_kv_cache(..., is_insert=False)` and FREES the KV
                    // we just spent the prefill building. That made the warmup
                    // a no-op (verified live: readback was a cold 16s re-prefill
                    // after an aborted warmup vs 0.27s after a completing one).
                    // Draining to the natural FINISH persists the cache so the
                    // next request rides the hit.
                    let mut body = resp.bytes_stream();
                    let mut got_first = false;
                    while let Some(chunk) = body.next().await {
                        if chunk.is_err() {
                            break;
                        }
                        if !got_first {
                            got_first = true;
                            let elapsed = started.elapsed();
                            tracing::info!(
                                "TTFT warmup: prefill completed in {:?} for {}",
                                elapsed,
                                worker_url_str
                            );
                          // Feed the per-worker, WARM-band EWMA: a warmup runs
                          // against a prefix the selector counts as matched
                          // but which lives in L1/L2 (or nowhere) — the
                          // measurement includes load-back cost, exactly the
                          // physics budgeted requests pay when the gate
                          // wrongly opens. Skip degenerate cases (< 50ms
                          // means the KV was already L0-hot; that sample
                          // belongs to the warm band too, but it is the
                          // *good* tail of it — feeding it keeps the warm
                          // band honest about the load-back-free case).
                           let elapsed_secs = elapsed.as_secs_f64().max(0.001);
                           if elapsed_secs > 0.05 && elapsed_secs < 600.0 {
                               // The warmup's time-to-first-token is spent
                               // prefilling the UNMATCHED portion (the matched
                               // prefix is already cached), so the honest
                               // per-uncached-token rate is
                               // uncached_tokens / warmup_latency — matching
                               // the gate's uncached-based estimate.
                               let uncached_tokens =
                                   ((1.0 - warmup_match_rate).max(0.0) * est_tokens as f32).max(1.0);
                               let measured_tps = uncached_tokens / elapsed_secs as f32;
                               if measured_tps > 0.0 && measured_tps.is_finite() {
                                   prefill_rates.record(
                                       &worker_url_str,
                                       warmup_match_rate,
                                       measured_tps,
                                       warmup_seed_tps,
                                   );
                                   tracing::info!(
                                       "TTFT prefill rate update (warmup, {} band): {} = \
                                        {:.0} tps ({:.0} uncached of {} tokens / {:.1}s, EWMA applied)",
                                       if TtftPrefillRateRegistry::band(warmup_match_rate) == 2 {
                                           "warm"
                                       } else if TtftPrefillRateRegistry::band(warmup_match_rate) == 1 {
                                           "partial"
                                       } else {
                                           "cold"
                                       },
                                       worker_url_str,
                                       prefill_rates.get(
                                           &worker_url_str,
                                           warmup_match_rate,
                                           warmup_seed_tps,
                                       ),
                                       uncached_tokens,
                                       est_tokens,
                                       elapsed_secs,
                                   );
                               }
                           }
                        }
                    }
                    // Fully drained → the engine PERSISTED this prompt's KV
                    // into its radix cache (the documented drop-body-midstream
                    // pitfall is avoided by the drain). The tree promised this
                    // prefix would be warm once the warmup landed; mint the
                    // receipt so budgeted requests may now trust that claim.
                    verified_warm.mark(&worker_url_str, warmup_prefix_hash);
                    tracing::debug!(
                        "TTFT warmup: drained full body for {} (cache persisted, \
                         receipt minted for prefix {:x})",
                        worker_url_str,
                        warmup_prefix_hash
                    );
                }
                Err(e) => {
                    warn!(
                        "TTFT warmup failed for {} after {:?}: {}",
                        worker_url_str,
                        started.elapsed(),
                        e
                    );
                }
            }
            warming.release(&worker_url_str);
        });
    }

    /// Warmup cap from config (kept as a method so the registry stays the
    /// only shared mutable state).
    fn ttft_warming_capacity(&self) -> usize {
        self.ttft_max_warmups
    }
    fn worker_base_url(&self, worker_url: &str) -> String {
        if self.dp_aware {
            if let Ok((prefix, _)) = Self::extract_dp_rank(worker_url) {
                return prefix.to_string();
            }
        }
        worker_url.to_string()
    }

    // Generic simple routing for GET/POST without JSON body
    async fn route_simple_request(
        &self,
        headers: Option<&HeaderMap>,
        endpoint: &str,
        method: Method,
    ) -> Response {
        // TODO: currently the sglang worker is using in-memory state management, so this implementation has to fan out to all workers.
        // Eventually, we need to have router to manage the chat history with a proper database, will update this implementation accordingly.
        let workers = self.worker_registry.get_all();
        if workers.is_empty() {
            return error::service_unavailable("no_workers", "No available workers");
        }

        let filtered_headers: Vec<_> = headers
            .map(|hdrs| {
                hdrs.iter()
                    .filter(|(name, _)| header_utils::should_forward_request_header(name.as_str()))
                    .collect()
            })
            .unwrap_or_default();

        let futures: Vec<_> = workers
            .into_iter()
            .map(|worker| {
                let worker_url = worker.url();
                let base = self.worker_base_url(worker_url);
                let url = format!("{}/{}", base, endpoint);
                let client = self.client.clone();
                let method = method.clone();

                let headers = filtered_headers.clone();

                let api_key = worker.api_key().clone();

                async move {
                    let mut request_builder = match method {
                        Method::GET => client.get(url),
                        Method::POST => client.post(url),
                        _ => {
                            return Err(error::method_not_allowed(
                                "unsupported_method",
                                "Unsupported method for simple routing",
                            ))
                        }
                    };

                    if let Some(key) = api_key {
                        let mut auth_header = String::with_capacity(7 + key.len());
                        auth_header.push_str("Bearer ");
                        auth_header.push_str(&key);
                        request_builder = request_builder.header("Authorization", auth_header);
                    }

                    for (name, value) in headers {
                        request_builder = request_builder.header(name.clone(), value.clone());
                    }

                    request_builder.send().await.map_err(convert_reqwest_error)
                }
            })
            .collect();

        // Now execute the collected futures concurrently
        let mut stream = stream::iter(futures).buffer_unordered(32);
        let mut last_response: Option<Response> = None;

        while let Some(result) = stream.next().await {
            match result {
                Ok(res) => {
                    let status = StatusCode::from_u16(res.status().as_u16())
                        .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);

                    let response_headers = header_utils::preserve_response_headers(res.headers());

                    match res.bytes().await {
                        Ok(body) => {
                            let mut response = Response::new(Body::from(body));
                            *response.status_mut() = status;
                            *response.headers_mut() = response_headers;

                            if status.is_success() {
                                return response;
                            }
                            last_response = Some(response);
                        }
                        Err(e) => {
                            last_response = Some(error::internal_error(
                                "read_response_failed",
                                format!("Failed to read response: {}", e),
                            ));
                        }
                    }
                }
                Err(e) => {
                    last_response = Some(e);
                }
            }
        }

        last_response
            .unwrap_or_else(|| error::bad_gateway("no_worker_response", "No worker response"))
    }

    // Route a GET request with provided headers to a specific endpoint
    async fn route_get_request(&self, headers: Option<&HeaderMap>, endpoint: &str) -> Response {
        self.route_simple_request(headers, endpoint, Method::GET)
            .await
    }

    // Route a POST request with empty body to a specific endpoint
    async fn route_post_empty_request(
        &self,
        headers: Option<&HeaderMap>,
        endpoint: &str,
    ) -> Response {
        self.route_simple_request(headers, endpoint, Method::POST)
            .await
    }

    // TODO (rui): Better accommodate to the Worker abstraction
    fn extract_dp_rank(worker_url: &str) -> Result<(&str, usize), String> {
        let parts: Vec<&str> = worker_url.split('@').collect();
        if parts.len() != 2 {
            return Err(format!("invalid worker_url format: {}", worker_url));
        }

        // Parse the second part (dp_rank) into an integer
        match parts[1].parse::<usize>() {
            Ok(dp_rank) => Ok((parts[0], dp_rank)),
            Err(_) => Err(format!(
                "failed to parse dp_rank from worker_url: {}",
                worker_url
            )),
        }
    }

    // Send typed request directly without conversion
    async fn send_typed_request<T: serde::Serialize>(
        &self,
        headers: Option<&HeaderMap>,
        typed_req: &T,
        route: &'static str,
        worker: &Arc<dyn Worker>,
        is_stream: bool,
        load_guard: Option<WorkerLoadGuard>,
        ttft_ctx: Option<TtftRecordCtx>,
    ) -> Response {
        let worker_url = worker.url();
        let api_key = worker.api_key().clone();

        // Static key string to avoid per-request allocations
        const DP_RANK_KEY: &str = "data_parallel_rank";

        let mut request_builder = if self.dp_aware {
            let (worker_url_prefix, dp_rank) = match Self::extract_dp_rank(worker_url) {
                Ok(tup) => tup,
                Err(e) => {
                    error!("Failed to extract dp_rank: {}", e);
                    return error::internal_error(
                        "dp_rank_extraction_failed",
                        format!("Failed to extract dp_rank: {}", e),
                    );
                }
            };

            let mut json_val = match serde_json::to_value(typed_req) {
                Ok(j) => j,
                Err(e) => {
                    return error::bad_request(
                        "serialization_failed",
                        format!("Convert into serde_json::Value failed: {}", e),
                    );
                }
            };

            if let Some(map) = json_val.as_object_mut() {
                // Use static key string to avoid allocation
                map.insert(DP_RANK_KEY.to_string(), serde_json::json!(dp_rank));
                // Only serialize if debug logging is enabled to avoid CPU overhead
                if tracing::enabled!(tracing::Level::DEBUG) {
                    debug!(
                        "Modified request body: {}",
                        serde_json::to_string(&json_val).unwrap_or_else(|_| String::from("ERR"))
                    );
                }
            } else {
                return error::bad_request(
                    "dp_rank_insertion_failed",
                    "Failed to insert the data_parallel_rank field into the request body",
                );
            }

            self.client
                .post(format!("{}{}", worker_url_prefix, route))
                .json(&json_val)
        } else {
            self.client
                .post(format!("{}{}", worker_url, route))
                .json(typed_req) // Use json() directly with typed request
        };

        if let Some(key) = api_key {
            // Pre-allocate string with capacity to avoid reallocation
            let mut auth_header = String::with_capacity(7 + key.len());
            auth_header.push_str("Bearer ");
            auth_header.push_str(&key);
            request_builder = request_builder.header("Authorization", auth_header);
        }

        if let Some(headers) = headers {
            for (name, value) in headers {
                if header_utils::should_forward_request_header(name.as_str()) {
                    request_builder = request_builder.header(name, value);
                }
            }
        }

        let res = match request_builder.send().await {
            Ok(res) => res,
            Err(e) => {
                error!(
                    "Failed to send typed request worker_url={} route={} error={}",
                    worker_url, route, e
                );

                // For streaming requests the caller skips the eager
                // `record_outcome` on the assumption that a
                // `BreakerTrackedStream` will tick the breaker on drop —
                // but no tracked stream is installed when send() fails
                // before any response stream exists. Record the failure
                // here so a worker flapping at the TCP layer doesn't
                // stay permanently selectable. Non-streaming requests
                // are already covered by the caller's
                // `worker.record_outcome(status.is_success())`, so
                // gating on `is_stream` avoids double-counting.
                if is_stream {
                    worker.record_outcome(false);
                }
                return convert_reqwest_error(e);
            }
        };

        let status = StatusCode::from_u16(res.status().as_u16())
            .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);

        if !is_stream {
            // For non-streaming requests, preserve headers
            let response_headers = header_utils::preserve_response_headers(res.headers());

            // Non-streaming TTFT observation. Patchbay's tool-rescue path
            // re-issues craft completions with stream:false (see
            // forward-openai.ts), so this is where the bulk of craft traffic
            // actually lands. The streaming branch's TtftObserverStream never
            // runs for these, which left the prefill-rate EWMA fed by idle
            // warmups only and the receipt registry unpopulated. The full-body
            // arrival time is the only TTFT signal a non-stream response gives
            // us: sub-2s on a high-match request proves the KV was L0-resident
            // (mint the receipt); >=2s proves it was not (drop a stale one).
            // We also feed the prefill-rate EWMA with uncached_tokens/elapsed,
            // exactly like the streaming observer — a non-stream completion's
            // body is prefill-dominated for the short tool-rescue completions
            // that make up this traffic, so it is a usable cold-band sample.
            let started = std::time::Instant::now();

            let response = match res.bytes().await {
                Ok(body) => {
                    let elapsed = started.elapsed().as_secs_f64();
                    if let Some(ctx) = ttft_ctx {
                        let worker_url_str = worker_url.to_string();
                        let verified_warm = self.ttft_verified_warm.clone();
                        let prefill_rates = self.ttft_prefill_rates.clone();
                        // Receipt mint/invalidation: proof of L0 residency,
                        // gated on the 2s proof threshold, NOT the 50ms rate
                        // floor — a sub-2s full body on a high-match request is
                        // proof the KV was resident.
                        if ctx.match_rate > RECEIPT_MATCH_THRESHOLD {
                            if elapsed < TtftVerifiedWarmRegistry::PROOF_MAX_SECS {
                                verified_warm.mark(&worker_url_str, ctx.prefix_hash);
                            } else {
                                verified_warm.invalidate(&worker_url_str, ctx.prefix_hash);
                                tracing::info!(
                                    "TTFT non-stream receipt invalidation: {:.1}s full body (>= {}s) \
                                     on a {:.0}%-match request to {} proves KV not L0-resident — \
                                     dropping (worker, prefix) receipt",
                                    elapsed,
                                    TtftVerifiedWarmRegistry::PROOF_MAX_SECS,
                                    ctx.match_rate * 100.0,
                                    worker_url_str
                                );
                            }
                        }
                        if elapsed > 0.05 && elapsed < 600.0 {
                            let uncached_tokens =
                                ((1.0 - ctx.match_rate).max(0.0) * ctx.est_tokens as f32).max(1.0);
                            let measured_tps = uncached_tokens / elapsed.max(0.001) as f32;
                            if measured_tps > 0.0 && measured_tps.is_finite() {
                                prefill_rates.record(
                                    &worker_url_str,
                                    ctx.match_rate,
                                    measured_tps,
                                    ctx.seed_tps,
                                );
                                tracing::info!(
                                    "TTFT non-stream prefill rate update: {} = {:.0} tps \
                                     ({:.0} uncached of {} tokens / {:.1}s, EWMA applied)",
                                    worker_url_str,
                                    measured_tps,
                                    uncached_tokens,
                                    ctx.est_tokens,
                                    elapsed,
                                );
                            }
                        }
                    }
                    let mut response = Response::new(Body::from(body));
                    *response.status_mut() = status;
                    *response.headers_mut() = response_headers;
                    response
                }
                Err(e) => {
                    let error_msg = format!("Failed to get response body: {}", e);
                    error::internal_error("read_response_body_failed", error_msg)
                }
            };

            // load_guard dropped here automatically after response body is read
            response
        } else {
            // Preserve headers for streaming response
            let mut response_headers = header_utils::preserve_response_headers(res.headers());
            // Ensure we set the correct content-type for SSE
            response_headers.insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));

            // Pass the reqwest byte stream straight through as the response body.
            // Dropping the response body drops this stream, which closes the
            // upstream HTTP connection and lets the engine abort generation —
            // no spawned task or channel needed. `BreakerTrackedStream`
            // updates the worker's circuit breaker exactly once on drop:
            // success on clean end, failure on stream error, neither on
            // client disconnect. For non-2xx responses we pre-mark the
            // wrapper as Errored — otherwise the small error body would
            // stream cleanly to `None` and Drop would record a spurious
            // success (and the streaming branch also skips the eager
            // `record_outcome` above).
            // Wrap the raw byte stream in a TTFT observer (when budgeted) so
            // the first chunk's latency — load-back + prefill included — is
            // recorded into the per-worker, per-band prefill-rate EWMA.
            // Streaming is exactly where live TTFT samples come from: warmups
            // only cover the 425 path, while live streams cover every
            // forwarded request including L1 load-back dominated ones.
            let raw_stream = if let Some(ctx) = ttft_ctx {
                // Token estimate must mirror what the warmup path uses: full
                // text (est_tokens covers the whole prompt; the band already
                // accounts for how much of it was cached).
                let prefill_rates = self.ttft_prefill_rates.clone();
                let worker_url_str = worker_url.to_string();
                let seed_tps = ctx.seed_tps;
                let match_rate = ctx.match_rate;
                let est_tokens = ctx.est_tokens;
                let prefix_hash = ctx.prefix_hash;
                let verified_warm = self.ttft_verified_warm.clone();
                // Clones for the abort hook (on_first's `move` closure moves
                // the originals out; these feed the no-chunk-drop invalidation).
                let verified_warm_drop = verified_warm.clone();
                let worker_url_drop = worker_url_str.clone();
                TtftObserverStream::new(res.bytes_stream(), move |elapsed_secs| {
                    // Receipt mint/invalidation is a proof of L0 residency and
                    // must run for EVERY first chunk — including sub-50ms
                    // L0-hot hits, the strongest possible proof. The 50ms floor
                    // applies ONLY to the prefill-RATE sample below (a sub-50ms
                    // "rate" would be a cache hit, not a prefill).
                    if elapsed_secs < TtftVerifiedWarmRegistry::PROOF_MAX_SECS
                        && match_rate > RECEIPT_MATCH_THRESHOLD
                    {
                        // Fast first chunk on a high-match request is proof the
                        // KV is L0-resident here — mint/refresh the receipt.
                        verified_warm.mark(&worker_url_str, prefix_hash);
                    } else if elapsed_secs >= TtftVerifiedWarmRegistry::PROOF_MAX_SECS
                        && match_rate > RECEIPT_MATCH_THRESHOLD
                    {
                        // SLOW first chunk on a high-match request proves the KV
                        // was NOT L0-resident (evicted since a prior receipt was
                        // minted, or never was). Left in place, the surviving
                        // receipt would forward the next same-prefix request into
                        // the same cold prefill → 90s fetch timeout. Drop it so
                        // that request re-estimates cold → 425 → clean cloud
                        // fallback, while the background warmup re-warms and
                        // re-mints a fresh receipt.
                        verified_warm.invalidate(&worker_url_str, prefix_hash);
                        tracing::info!(
                            "TTFT receipt invalidation: {:.1}s first chunk (>= {}s) on a \
                             {:.0}%-match request to {} proves KV not L0-resident — \
                             dropping (worker, prefix) receipt",
                            elapsed_secs,
                            TtftVerifiedWarmRegistry::PROOF_MAX_SECS,
                            match_rate * 100.0,
                            worker_url_str
                        );
                    }
                    if elapsed_secs > 0.05 && elapsed_secs < 600.0 {
                        // The engine's time-to-first-token is spent prefilling
                        // the UNMATCHED portion of the prompt (the matched
                        // prefix is already cached). So the honest per-uncached-
                        // token rate is uncached_tokens / first_chunk_latency —
                        // NOT full_tokens / latency, which overstates the rate
                        // for warm requests (the L0-hit inflation this banding
                        // exists to remove).
                        let uncached_tokens =
                            ((1.0 - match_rate).max(0.0) * est_tokens as f32).max(1.0);
                        let measured_tps = uncached_tokens / elapsed_secs.max(0.001) as f32;
                        if measured_tps > 0.0 && measured_tps.is_finite() {
                            prefill_rates.record(
                                &worker_url_str,
                                match_rate,
                                measured_tps,
                                seed_tps,
                            );
                            tracing::info!(
                                "TTFT live prefill rate update ({} band): {} = {:.0} tps \
                                 ({:.0} uncached of {} tokens / {:.1}s, EWMA applied)",
                                if TtftPrefillRateRegistry::band(match_rate) == 2 {
                                    "warm"
                                } else if TtftPrefillRateRegistry::band(match_rate) == 1 {
                                    "partial"
                                } else {
                                    "cold"
                                },
                                worker_url_str,
                                prefill_rates.get(&worker_url_str, match_rate, seed_tps),
                                uncached_tokens,
                                est_tokens,
                                elapsed_secs,
                            );
                        }
                    } else if elapsed_secs <= 0.05 {
                        tracing::debug!(
                            "TTFT live sample skipped: {}s < 50ms floor (L0-hot?)",
                            elapsed_secs
                        );
                    }
                })
                .on_drop_no_chunk(move || {
                    // Cancelled/aborted before any first chunk: the worker was
                    // still cold-prefilling, so the KV was NOT L0-resident. A
                    // receipt left in place would forward the next same-prefix
                    // request into the same prefill → another 90s fetch
                    // timeout. Drop it so that request re-estimates cold → 425
                    // → clean cloud fallback while the warmup re-mints one.
                    // (Deliberately NOT fed to the prefill-rate EWMA — a cancel
                    // point is not a prefill completion.)
                    if match_rate > RECEIPT_MATCH_THRESHOLD {
                        verified_warm_drop.invalidate(&worker_url_drop, prefix_hash);
                        tracing::info!(
                            "TTFT receipt invalidation (abort): no first chunk \
                             before cancel on a {:.0}%-match request to {} — \
                             dropping (worker, prefix) receipt",
                            match_rate * 100.0,
                            worker_url_drop
                        );
                    }
                })
                .boxed()
            } else {
                res.bytes_stream().boxed()
            };

            let mut tracked = BreakerTrackedStream::new(
                raw_stream,
                worker.clone(),
                worker_url.to_string(),
            );
            if !status.is_success() {
                tracked.mark_errored();
            }
            let body = Body::from_stream(tracked);

            let mut response = Response::new(body);
            *response.status_mut() = status;
            *response.headers_mut() = response_headers;

            // Attach load guard to response body for proper RAII lifecycle
            // Guard is dropped when response body is consumed or client disconnects
            if let Some(guard) = load_guard {
                response = AttachedBody::wrap_response(response, guard);
            }
            response
        }
    }

    async fn build_rerank_response(
        req: &RerankRequest,
        response: Response,
    ) -> anyhow::Result<Response> {
        let (_, response_body) = response.into_parts();
        let body_bytes = to_bytes(response_body, usize::MAX).await?;
        let rerank_results = serde_json::from_slice::<Vec<RerankResult>>(&body_bytes)?;
        let mut rerank_response =
            RerankResponse::new(rerank_results, req.model.clone(), req.rid.clone());
        // Sorting is handled by Python worker (serving_rerank.py)
        if let Some(top_k) = req.top_k {
            rerank_response.apply_top_k(top_k);
        }
        if !req.return_documents {
            rerank_response.drop_documents();
        }
        Ok(Json(rerank_response).into_response())
    }
}

fn convert_reqwest_error(e: reqwest::Error) -> Response {
    let url = e
        .url()
        .map(|u| u.to_string())
        .unwrap_or_else(|| "unknown".to_string());
    let message = format!("{}. URL: {}", e, url);

    // TODO improve error status code
    let (status, code) = if let Some(upstream_status) = e.status() {
        (upstream_status, "call_upstream_status_error")
    } else if e.is_builder() {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "call_upstream_builder_error",
        )
    } else if e.is_request() {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "call_upstream_request_error",
        )
    } else if e.is_redirect() {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "call_upstream_redirect_error",
        )
    } else if e.is_body() {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "call_upstream_body_error",
        )
    } else if e.is_decode() {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "call_upstream_decode_error",
        )
    } else if e.is_timeout() {
        (StatusCode::GATEWAY_TIMEOUT, "call_upstream_timeout")
    } else if e.is_connect() {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "call_upstream_connection_failed",
        )
    } else {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "call_upstream_request_failed",
        )
    };

    error::create_error(status, code, message)
}

use async_trait::async_trait;

#[async_trait]
impl RouterTrait for Router {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    async fn health_generate(&self, req: Request<Body>) -> Response {
        self.proxy_get_request(req, "health_generate").await
    }

    async fn get_server_info(&self, req: Request<Body>) -> Response {
        self.proxy_get_request(req, "server_info").await
    }

    async fn get_models(&self, req: Request<Body>) -> Response {
        self.proxy_get_request(req, "v1/models").await
    }

    async fn get_model_info(&self, req: Request<Body>) -> Response {
        self.proxy_get_request(req, "model_info").await
    }

    async fn route_generate(
        &self,
        headers: Option<&HeaderMap>,
        body: &GenerateRequest,
        model_id: Option<&str>,
    ) -> Response {
        self.route_typed_request(headers, body, "/generate", model_id)
            .await
    }

    async fn route_chat(
        &self,
        headers: Option<&HeaderMap>,
        body: &ChatCompletionRequest,
        model_id: Option<&str>,
    ) -> Response {
        self.route_typed_request(headers, body, "/v1/chat/completions", model_id)
            .await
    }

    async fn route_completion(
        &self,
        headers: Option<&HeaderMap>,
        body: &CompletionRequest,
        model_id: Option<&str>,
    ) -> Response {
        self.route_typed_request(headers, body, "/v1/completions", model_id)
            .await
    }

    async fn route_responses(
        &self,
        headers: Option<&HeaderMap>,
        body: &ResponsesRequest,
        model_id: Option<&str>,
    ) -> Response {
        self.route_typed_request(headers, body, "/v1/responses", model_id)
            .await
    }

    async fn get_response(
        &self,
        headers: Option<&HeaderMap>,
        response_id: &str,
        _params: &ResponsesGetParams,
    ) -> Response {
        let endpoint = format!("v1/responses/{}", response_id);
        self.route_get_request(headers, &endpoint).await
    }

    async fn cancel_response(&self, headers: Option<&HeaderMap>, response_id: &str) -> Response {
        let endpoint = format!("v1/responses/{}/cancel", response_id);
        self.route_post_empty_request(headers, &endpoint).await
    }

    async fn route_embeddings(
        &self,
        headers: Option<&HeaderMap>,
        body: &EmbeddingRequest,
        model_id: Option<&str>,
    ) -> Response {
        self.route_typed_request(headers, body, "/v1/embeddings", model_id)
            .await
    }

    async fn route_classify(
        &self,
        headers: Option<&HeaderMap>,
        body: &ClassifyRequest,
        model_id: Option<&str>,
    ) -> Response {
        self.route_typed_request(headers, body, "/v1/classify", model_id)
            .await
    }

    async fn route_rerank(
        &self,
        headers: Option<&HeaderMap>,
        body: &RerankRequest,
        model_id: Option<&str>,
    ) -> Response {
        let response = self
            .route_typed_request(headers, body, "/v1/rerank", model_id)
            .await;
        if response.status().is_success() {
            match Self::build_rerank_response(body, response).await {
                Ok(rerank_response) => rerank_response,
                Err(e) => {
                    error!("Failed to build rerank response: {}", e);
                    return error::internal_error(
                        "rerank_response_build_failed",
                        "Failed to build rerank response",
                    );
                }
            }
        } else {
            response
        }
    }

    fn router_type(&self) -> &'static str {
        "regular"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::BasicWorkerBuilder;

    fn create_test_regular_router() -> Router {
        // Create registries
        let worker_registry = Arc::new(WorkerRegistry::new());
        let policy_registry = Arc::new(PolicyRegistry::new(
            crate::config::types::PolicyConfig::RoundRobin,
        ));

        // Register test workers
        let worker1 = BasicWorkerBuilder::new("http://worker1:8080")
            .worker_type(WorkerType::Regular)
            .build();
        let worker2 = BasicWorkerBuilder::new("http://worker2:8080")
            .worker_type(WorkerType::Regular)
            .build();
        worker_registry.register(Arc::new(worker1));
        worker_registry.register(Arc::new(worker2));

        Router {
            worker_registry,
            policy_registry,
            dp_aware: false,
            client: Client::new(),
            retry_config: RetryConfig::default(),
            enable_igw: false,
            ttft_prefill_rate_tps: 0.0,
            ttft_chars_per_token: 3.5,
            ttft_max_warmups: 4,
            ttft_warming: Arc::new(TtftWarmingRegistry::default()),
            ttft_prefill_rates: Arc::new(TtftPrefillRateRegistry::default()),
            ttft_verified_warm: Arc::new(TtftVerifiedWarmRegistry::default()),
        }
    }

    fn create_test_unhealthy_router() -> Router {
        let router = create_test_regular_router();
        let workers = router.worker_registry.get_all();
        workers[0].set_healthy(false);
        router
    }

    #[test]
    fn test_router_get_worker_urls_regular() {
        let router = create_test_regular_router();
        let workers = router.worker_registry.get_all();
        let urls: Vec<String> = workers.iter().map(|w| w.url().to_string()).collect();

        assert_eq!(urls.len(), 2);
        assert!(urls.contains(&"http://worker1:8080".to_string()));
        assert!(urls.contains(&"http://worker2:8080".to_string()));
    }

    #[test]
    fn test_select_first_worker_regular() {
        let router = create_test_regular_router();
        let result = router.select_first_worker();

        assert!(result.is_ok());
        let url = result.unwrap();
        // DashMap doesn't guarantee order, so just check we get one of the workers
        assert!(url == "http://worker1:8080" || url == "http://worker2:8080");
    }

    #[test]
    fn test_select_first_worker_with_unhealthy_worker() {
        let router = create_test_unhealthy_router();
        let result = router.select_first_worker();

        assert!(result.is_ok());
        let url = result.unwrap();

        let worker = router.worker_registry.get_by_url(&url).unwrap();
        assert!(worker.is_healthy());
    }

    #[test]
    fn test_estimate_ttft_ms_shapes() {
        // zero text -> 0
        assert_eq!(estimate_ttft_ms("", 0.0, 3.5, 460.0), 0);
        // disabled prefill rate -> 0
        assert_eq!(estimate_ttft_ms("hello world", 0.0, 3.5, 0.0), 0);
        // fully cached -> 0
        assert_eq!(estimate_ttft_ms("some prompt", 1.0, 3.5, 460.0), 0);
        // 142k text: 97% matched ~2.6s; fully cold ~88s (matches the
        // observed 71s cold TTFT order of magnitude on crimson)
        let text = "x".repeat(142_000);
        let warmish = estimate_ttft_ms(&text, 0.97, 3.5, 460.0);
        assert!(warmish > 2000 && warmish < 3500, "warmish={}", warmish);
        let cold = estimate_ttft_ms(&text, 0.0, 3.5, 460.0);
        assert!(cold > 80_000 && cold < 95_000, "cold={}", cold);
    }

    #[test]
    fn test_ttft_budget_header_parsing() {
        let mut hm = HeaderMap::new();
        assert_eq!(ttft_budget_ms(None), None);
        assert_eq!(ttft_budget_ms(Some(&hm)), None);
        hm.insert(
            "x-ttft-budget-ms",
            HeaderValue::from_static("10000"),
        );
        assert_eq!(ttft_budget_ms(Some(&hm)), Some(10000));
        let mut bad = HeaderMap::new();
        bad.insert("x-ttft-budget-ms", HeaderValue::from_static("soon"));
        assert_eq!(ttft_budget_ms(Some(&bad)), None);
    }

    #[test]
    fn test_warming_registry() {
        let reg = TtftWarmingRegistry::default();
        assert!(reg.register("w1", 4));
        assert!(reg.is_warming("w1"));
        assert!(!reg.register("w1", 4), "same worker double-warmup rejected");
        assert!(reg.register("w2", 4));
        assert!(reg.register("w3", 4));
        assert!(reg.register("w4", 4));
        assert_eq!(reg.len(), 4);
        assert!(!reg.register("w5", 4), "cap enforced");
        reg.release("w1");
        assert!(!reg.is_warming("w1"));
        assert!(reg.register("w5", 4), "slot freed after release");
    }

    #[test]
    fn test_prefill_rate_ewma() {
        let reg = TtftPrefillRateRegistry::default();
        let seed = 2600.0;
        let cold = 0.0;
        let warm = 0.95;

        // First access seeds with the fallback, per band
        let r = reg.get("w1", cold, seed);
        assert_eq!(r, seed, "cold band seeds at fallback");
        let rw = reg.get("w1", warm, seed);
        assert_eq!(rw, seed, "warm band independently seeds at fallback");

        // Record a measurement of 2000 tps into the COLD band — the warm
        // band must stay at seed (bands are independent)
        reg.record("w1", cold, 2000.0, seed);
        let r1 = reg.get("w1", cold, seed);
        // alpha=0.3: 0.3*2000 + 0.7*2600 = 2420
        assert!((r1 - 2420.0).abs() < 1.0, "r1={}", r1);
        assert_eq!(reg.get("w1", warm, seed), seed, "warm band untouched");

        // Another measurement of 2000 — moves further toward 2000
        reg.record("w1", cold, 2000.0, seed);
        let r2 = reg.get("w1", cold, seed);
        // 0.3*2000 + 0.7*2420 = 2294
        assert!((r2 - 2294.0).abs() < 1.0, "r2={}", r2);

        // Different worker is independent even in the same band
        let r3 = reg.get("w2", cold, seed);
        assert_eq!(r3, seed, "w2 should still be at seed");

        // Band boundaries: 0.05 → cold(0), 0.10 → partial(1), 0.60 → partial(1),
        // 0.61 → warm(2)
        assert_eq!(TtftPrefillRateRegistry::band(0.05), 0);
        assert_eq!(TtftPrefillRateRegistry::band(0.10), 1);
        assert_eq!(TtftPrefillRateRegistry::band(0.60), 1);
        assert_eq!(TtftPrefillRateRegistry::band(0.61), 2);
        assert_eq!(TtftPrefillRateRegistry::band(0.95), 2);
    }

    #[test]
    fn test_verified_warm_registry() {
        let reg = TtftVerifiedWarmRegistry::default();
        let prefix_a = "conversation-a system prompt".repeat(200);
        let prefix_b = "conversation-b system prompt".repeat(200);
        let hash_a = TtftVerifiedWarmRegistry::hash_prefix(&prefix_a);
        let hash_b = TtftVerifiedWarmRegistry::hash_prefix(&prefix_b);
        assert_ne!(hash_a, hash_b, "distinct prefixes must not collide");
        // Stable across calls (two turns of the same conversation re-hash
        // identically, so the receipt survives a multi-turn session).
        assert_eq!(hash_a, TtftVerifiedWarmRegistry::hash_prefix(&prefix_a));

        assert!(!reg.is_verified("w1", hash_a), "fresh registry has no receipts");
        reg.mark("w1", hash_a);
        assert!(reg.is_verified("w1", hash_a), "marked receipt is visible");
        assert!(!reg.is_verified("w1", hash_b), "prefix B has no receipt");
        assert!(!reg.is_verified("w2", hash_a), "other worker has no receipt");

        // A slow first chunk on a high-match request invalidates the receipt
        // (KV evicted under VRAM pressure) so the next same-prefix request
        // re-estimates cold → 425 instead of re-forwarding into a timeout.
        reg.mark("w1", hash_b);
        reg.invalidate("w1", hash_a);
        assert!(!reg.is_verified("w1", hash_a), "invalidated receipt is gone");
        assert!(reg.is_verified("w1", hash_b), "sibling receipt survives");
    }
}
