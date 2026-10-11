//! Shared streaming helpers used by the HTTP / OpenAI / PD routers.
//!
//! The main type here is [`BreakerTrackedStream`], a `Stream` adapter that
//! records circuit-breaker outcomes based on how the upstream stream
//! terminates:
//!
//! - **Clean end** (`Poll::Ready(None)`) → `record_success`.
//! - **Upstream transport error** (`Poll::Ready(Some(Err(_)))`) → `record_failure`.
//! - **Caller drops the stream while still active** (client disconnect) → no
//!   breaker call. The HTTP response already shipped a 200 status; whether
//!   the worker is healthy is unknown from this signal alone.
//!
//! Callers can override the terminal state in two cases:
//!
//! - [`BreakerTrackedStream::mark_completed`] — for routers that detect
//!   end-of-stream via an in-band sentinel (e.g. PD's `data: [DONE]`)
//!   before the underlying byte stream returns `None`. Note that
//!   `Completed` is *not* absorbing — a subsequent `poll_next` returning
//!   `Err` will still escalate the terminal to `Errored`. Callers should
//!   stop polling after `mark_completed`.
//! - [`BreakerTrackedStream::mark_errored`] — for routers wrapping content
//!   that already represents a worker failure (e.g. a non-2xx response
//!   body or a synthetic SSE error envelope built from a 5xx). `Errored`
//!   is absorbing: once set, it persists regardless of later polls.

use std::{
    fmt::Display,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
    time::Instant,
};

use bytes::Bytes;
use futures_util::Stream;
use tracing::error;

use crate::core::Worker;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Terminal {
    /// Stream is still in flight, or was dropped before terminating.
    Active,
    /// Stream returned `None` or the caller marked it complete.
    Completed,
    /// Stream yielded an `Err` item.
    Errored,
}

/// One-shot TTFT observer for a live streamed response.
///
/// Times the first body chunk of the upstream SSE stream (≈ time to first
/// token, including any cache load-back the engine had to pay) and reports it
/// through the provided closure exactly once. This is what teaches the
/// per-worker prefill-rate EWMA the *real* cost of serving a prompt whose KV
/// was evicted to L1/L2 — warmup measurements alone under-sample that path.
pub struct TtftObserverStream<E = reqwest::Error> {
    inner: Pin<Box<dyn Stream<Item = Result<Bytes, E>> + Send + 'static>>,
    observed: bool,
    started: Instant,
    /// Called with (elapsed_secs, dropped_after_first_bytes) exactly once,
    /// either when the first chunk arrives or on stream error/EOF without
    /// any chunk (elapsed from start, so a failed send still records).
    on_first: Option<Box<dyn FnOnce(f64) + Send>>,
    /// Called ONCE when the body is discarded before any first chunk arrived
    /// while the request was pending for at least `ABORT_SIGNAL_MIN_SECS`.
    /// This is the cancelled/aborted case: `poll_next` never delivered a
    /// first chunk, an error, or EOF, so `on_first` never fires and the abort
    /// is otherwise invisible. A receipt for a (worker, prefix) whose KV was
    /// evicted then survives its TTL and every repeat same-prefix request
    /// re-forwards into the same cold prefill → 90s fetch timeout.
    ///
    /// Deliberately SEPARATE from `on_first`: an abort's elapsed time is the
    /// caller's cancel point, NOT a prefill completion, so it must NOT be fed
    /// to the prefill-rate EWMA (a 3s cancel of a 60s cold prefill would
    /// register an absurd 20k tps and inflate the band). Only the receipt
    /// invalidation happens on abort.
    on_drop_no_chunk: Option<Box<dyn FnOnce() + Send>>,
}

impl<E> TtftObserverStream<E> {
    /// Minimum pending time for a no-first-chunk drop to count as a stall.
    /// Below it the client cancelled a fast request early (or nothing was
    /// really in flight) — neither the receipt registry nor the rate EWMA
    /// should care.
    const ABORT_SIGNAL_MIN_SECS: f64 = 2.0;

    pub fn new<S, F>(inner: S, on_first: F) -> Self
    where
        S: Stream<Item = Result<Bytes, E>> + Send + 'static,
        F: FnOnce(f64) + Send + 'static,
    {
        Self {
            inner: Box::pin(inner),
            observed: false,
            started: Instant::now(),
            on_first: Some(Box::new(on_first)),
            on_drop_no_chunk: None,
        }
    }

    /// Register the abort hook (receipt invalidation, no rate sample).
    pub fn on_drop_no_chunk(mut self, f: impl FnOnce() + Send + 'static) -> Self {
        self.on_drop_no_chunk = Some(Box::new(f));
        self
    }
}

/// Report a cancellation-without-first-token to the hook registered via
/// `on_drop_no_chunk`. The elapsed gate keeps a very early cancel from being
/// treated as a stall signal.
impl<E> Drop for TtftObserverStream<E> {
    fn drop(&mut self) {
        if !self.observed && self.on_first.is_some() {
            if self.started.elapsed().as_secs_f64() >= Self::ABORT_SIGNAL_MIN_SECS {
                if let Some(f) = self.on_drop_no_chunk.take() {
                    f();
                }
            }
        }
    }
}

impl<E: Display> Stream for TtftObserverStream<E> {
    type Item = Result<Bytes, E>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.observed {
            return self.inner.as_mut().poll_next(cx);
        }
        match self.inner.as_mut().poll_next(cx) {
            Poll::Ready(Some(Ok(b))) => {
                self.observed = true;
                let elapsed = self.started.elapsed().as_secs_f64();
                if let Some(f) = self.on_first.take() {
                    f(elapsed);
                }
                Poll::Ready(Some(Ok(b)))
            }
            Poll::Ready(Some(Err(e))) => {
                self.observed = true;
                if let Some(f) = self.on_first.take() {
                    f(self.started.elapsed().as_secs_f64());
                }
                Poll::Ready(Some(Err(e)))
            }
            Poll::Ready(None) => {
                self.observed = true;
                if let Some(f) = self.on_first.take() {
                    f(self.started.elapsed().as_secs_f64());
                }
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

/// Wraps a `Stream<Item = Result<Bytes, E>>` so that the circuit breaker on
/// `worker` is updated exactly once on drop:
/// - completed → `record_success`
/// - errored → `record_failure`
/// - dropped while still active → neither (client disconnected; the worker
///   is innocent from our point of view).
///
/// `E` defaults to `reqwest::Error` to match the common producer
/// (`response.bytes_stream()`), but can be any `Display + Send + 'static`
/// error type — useful for tests that construct streams with simpler
/// error types.
#[must_use = "BreakerTrackedStream must be polled to completion (or pre-marked) \
              and then dropped for the circuit breaker to record an outcome; \
              discarding it immediately records nothing"]
pub struct BreakerTrackedStream<E = reqwest::Error> {
    inner: Pin<Box<dyn Stream<Item = Result<Bytes, E>> + Send + 'static>>,
    worker: Arc<dyn Worker>,
    log_url: String,
    terminal: Terminal,
}

impl<E> BreakerTrackedStream<E> {
    pub fn new<S>(inner: S, worker: Arc<dyn Worker>, log_url: String) -> Self
    where
        S: Stream<Item = Result<Bytes, E>> + Send + 'static,
    {
        Self {
            inner: Box::pin(inner),
            worker,
            log_url,
            terminal: Terminal::Active,
        }
    }

    /// Mark the stream as cleanly completed. Use this from callers that
    /// detect end-of-stream via an in-band sentinel (e.g. `data: [DONE]`)
    /// before the underlying byte stream returns `None`.
    ///
    /// Has no effect once the wrapper is in any non-Active state. `Completed`
    /// is *not* absorbing — a later `poll_next` returning `Err` will still
    /// escalate the terminal to `Errored`. Callers should stop polling
    /// after calling this.
    pub fn mark_completed(&mut self) {
        if self.terminal == Terminal::Active {
            self.terminal = Terminal::Completed;
        }
    }

    /// Pre-tag the stream as terminally errored. Use this from callers
    /// constructing a wrapper around content that already represents a
    /// failed worker outcome (e.g. a non-2xx response body or a
    /// synthetic error envelope) so Drop records `record_failure` even
    /// if the underlying stream terminates cleanly. `Errored` is
    /// absorbing — once set it stays set regardless of later events.
    pub fn mark_errored(&mut self) {
        self.terminal = Terminal::Errored;
    }
}

impl<E: Display> Stream for BreakerTrackedStream<E> {
    type Item = Result<Bytes, E>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match self.inner.as_mut().poll_next(cx) {
            Poll::Ready(Some(Ok(b))) => Poll::Ready(Some(Ok(b))),
            Poll::Ready(Some(Err(e))) => {
                error!("Upstream stream error from worker {}: {}", self.log_url, e);
                self.terminal = Terminal::Errored;
                Poll::Ready(Some(Err(e)))
            }
            Poll::Ready(None) => {
                if self.terminal == Terminal::Active {
                    self.terminal = Terminal::Completed;
                }
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

impl<E> Drop for BreakerTrackedStream<E> {
    fn drop(&mut self) {
        match self.terminal {
            Terminal::Completed => self.worker.circuit_breaker().record_success(),
            Terminal::Errored => self.worker.circuit_breaker().record_failure(),
            // Client disconnected before we knew the worker's verdict.
            // Leaving the breaker untouched is the correct default — we
            // got a 200 header and some bytes; nothing said the worker
            // is unhealthy.
            Terminal::Active => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{fmt, sync::Arc};

    use bytes::Bytes;
    use futures_util::StreamExt;

    use super::{BreakerTrackedStream, TtftObserverStream};
    use crate::core::{BasicWorkerBuilder, Worker};

    /// Lightweight error type for tests — keeps the wrapper generic so we
    /// don't need to fabricate `reqwest::Error` instances.
    #[derive(Debug)]
    struct TestErr(&'static str);

    impl fmt::Display for TestErr {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str(self.0)
        }
    }

    fn worker() -> Arc<dyn Worker> {
        Arc::new(BasicWorkerBuilder::new("http://test-worker").build())
    }

    fn breaker_counters(w: &Arc<dyn Worker>) -> (u64, u64) {
        let cb = w.circuit_breaker();
        (cb.total_successes(), cb.total_failures())
    }

    #[tokio::test]
    async fn drop_while_active_records_nothing() {
        let w = worker();
        let inner = futures_util::stream::pending::<Result<Bytes, TestErr>>();
        let tracked = BreakerTrackedStream::new(inner, Arc::clone(&w), "u".into());
        drop(tracked);
        assert_eq!(breaker_counters(&w), (0, 0));
    }

    #[tokio::test]
    async fn clean_stream_records_one_success() {
        let w = worker();
        let inner = futures_util::stream::iter(vec![
            Ok::<_, TestErr>(Bytes::from_static(b"a")),
            Ok(Bytes::from_static(b"b")),
        ]);
        let mut tracked = BreakerTrackedStream::new(inner, Arc::clone(&w), "u".into());
        while tracked.next().await.is_some() {}
        drop(tracked);
        assert_eq!(breaker_counters(&w), (1, 0));
    }

    #[tokio::test]
    async fn stream_error_records_one_failure() {
        let w = worker();
        let inner =
            futures_util::stream::iter(vec![Ok(Bytes::from_static(b"a")), Err(TestErr("boom"))]);
        let mut tracked = BreakerTrackedStream::new(inner, Arc::clone(&w), "u".into());
        while tracked.next().await.is_some() {}
        drop(tracked);
        assert_eq!(breaker_counters(&w), (0, 1));
    }

    #[tokio::test]
    async fn errored_is_absorbing_across_subsequent_polls() {
        let w = worker();
        let inner = futures_util::stream::iter(vec![
            Err::<Bytes, _>(TestErr("boom")),
            Ok(Bytes::from_static(b"after")),
        ]);
        let mut tracked = BreakerTrackedStream::new(inner, Arc::clone(&w), "u".into());
        while tracked.next().await.is_some() {}
        drop(tracked);
        assert_eq!(breaker_counters(&w), (0, 1));
    }

    #[tokio::test]
    async fn mark_completed_then_drop_records_success() {
        let w = worker();
        let inner = futures_util::stream::pending::<Result<Bytes, TestErr>>();
        let mut tracked = BreakerTrackedStream::new(inner, Arc::clone(&w), "u".into());
        tracked.mark_completed();
        drop(tracked);
        assert_eq!(breaker_counters(&w), (1, 0));
    }

    #[tokio::test]
    async fn mark_errored_then_clean_end_still_records_failure() {
        let w = worker();
        let inner = futures_util::stream::iter(vec![Ok::<_, TestErr>(Bytes::from_static(b"a"))]);
        let mut tracked = BreakerTrackedStream::new(inner, Arc::clone(&w), "u".into());
        tracked.mark_errored();
        while tracked.next().await.is_some() {}
        drop(tracked);
        assert_eq!(breaker_counters(&w), (0, 1));
    }

    #[tokio::test]
    async fn mark_completed_does_not_overwrite_errored() {
        let w = worker();
        let inner = futures_util::stream::iter(vec![Err::<Bytes, _>(TestErr("boom"))]);
        let mut tracked = BreakerTrackedStream::new(inner, Arc::clone(&w), "u".into());
        while tracked.next().await.is_some() {}
        tracked.mark_completed();
        drop(tracked);
        assert_eq!(breaker_counters(&w), (0, 1));
    }

    // PD's [DONE] handler calls mark_completed before the underlying byte
    // stream finishes; a trailing transport error must still flip the
    // terminal to Errored so the breaker records failure.
    #[tokio::test]
    async fn mark_completed_then_later_err_escalates_to_failure() {
        let w = worker();
        let inner = futures_util::stream::iter(vec![
            Ok::<_, TestErr>(Bytes::from_static(b"data: [DONE]\n\n")),
            Err(TestErr("trailing transport error")),
        ]);
        let mut tracked = BreakerTrackedStream::new(inner, Arc::clone(&w), "u".into());
        // Caller observes the [DONE] chunk and pre-marks completed...
        assert!(tracked.next().await.is_some());
        tracked.mark_completed();
        // ...but a trailing poll surfaces a transport error.
        assert!(matches!(tracked.next().await, Some(Err(_))));
        drop(tracked);
        assert_eq!(breaker_counters(&w), (0, 1));
    }
    // ── TtftObserverStream ────────────────────────────────────────────────

    #[tokio::test]
    async fn observer_reports_first_chunk_elapsed() {
        let (tx, got) = std::sync::mpsc::channel::<f64>();
        let inner = futures_util::stream::iter(vec![
            Ok::<_, TestErr>(Bytes::from_static(b"first")),
            Ok(Bytes::from_static(b"second")),
            Ok(Bytes::from_static(b"third")),
        ]);
        let mut obs = TtftObserverStream::new(inner, move |e| tx.send(e).unwrap());
        assert!(obs.next().await.is_some(), "first chunk");
        assert!(obs.next().await.is_some(), "second chunk");
        assert!(obs.next().await.is_some(), "third chunk");
        assert!(obs.next().await.is_none(), "eof");
        // exactly one sample, tiny but ≥ 0
        let e = got.try_recv().expect("one sample recorded");
        assert!(e >= 0.0);
        assert!(got.try_recv().is_err(), "no second sample");
    }

    #[tokio::test]
    async fn observer_reports_on_error_without_chunk() {
        let (tx, got) = std::sync::mpsc::channel::<f64>();
        let inner = futures_util::stream::iter(vec![Err::<Bytes, TestErr>(TestErr("boom"))]);
        let mut obs = TtftObserverStream::new(inner, move |e| tx.send(e).unwrap());
        assert!(matches!(obs.next().await, Some(Err(_))));
        assert!(got.try_recv().is_ok(), "error before first chunk still records");
        // error AFTER first chunk must not re-fire — verified by the exact-one
        // guarantee in the channel assertion above.
    }

    #[tokio::test]
    async fn observer_reports_on_eof_without_chunk() {
        let (tx, got) = std::sync::mpsc::channel::<f64>();
        let inner: futures_util::stream::Iter<
            std::vec::IntoIter<Result<Bytes, TestErr>>,
        > = futures_util::stream::iter(vec![]);
        let mut obs = TtftObserverStream::new(inner, move |e| tx.send(e).unwrap());
        assert!(obs.next().await.is_none(), "eof immediately");
        assert!(got.try_recv().is_ok(), "eof without chunk records elapsed");
    }
}
