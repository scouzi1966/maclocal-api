import Foundation
import os

/// Global thread-safe aggregator for AFM server metrics, exposed via
/// `GET /metrics` in Prometheus text exposition format.
///
/// Modelled after vLLM's `vllm/engine/metrics.py`:
///
///   - counters (monotonic)     — total tokens, total requests, cache events,
///                                request_success per finished_reason
///   - gauges (instantaneous)   — running / waiting / peak batch size,
///                                gpu_cache_usage_perc
///   - histograms (cumulative)  — per-request latency / size / params
///
/// Bucket boundaries match vLLM's defaults exactly so Grafana dashboards
/// authored against `vllm:*` work against `afm:*` after a search-and-replace
/// of the namespace prefix.
///
/// Both the batched code path (`BatchScheduler`) and the serial
/// single-sequence path (`MLXModelService.generate*`) call into the
/// same singleton so `/metrics` always reflects the complete server
/// activity regardless of which path served the request.
///
/// Live-state gauges (inflight, queue depth, gpu cache usage) are read
/// through closures registered at startup by whichever component owns
/// the state. This lets the metrics endpoint pull live values from
/// existing nonisolated locks without hopping through any actor.
public final class StatsAggregator: @unchecked Sendable {

    /// Process-wide singleton. All increment calls are O(1) under a
    /// single unfair lock; contention is negligible vs the Metal
    /// decode step.
    public static let shared = StatsAggregator()

    public typealias GaugeReader = @Sendable () -> Int
    public typealias FractionReader = @Sendable () -> Double

    // MARK: - vLLM bucket boundaries

    /// Bucket boundaries match upstream vLLM exactly so Grafana
    /// dashboards from the vLLM ecosystem are drop-in compatible.
    ///
    /// Provenance (port-time snapshot, **not a live dependency** — values
    /// are literals; nothing in afm's build or runtime reaches out to
    /// vllm-project/vllm):
    ///
    ///   - source repo: vllm-project/vllm
    ///   - source file: vllm/v1/metrics/loggers.py
    ///   - source blob: 6855efd9f54c6f8ac5b95704455f64d6e456b4c8
    ///   - repo HEAD:   2ee8c2a56e41fbd00b4fb52f29464fb7fca48dba
    ///   - port date:   2026-05-09
    ///
    /// See `Scripts/grafana/UPSTREAM.md` for the full provenance trail
    /// and the procedure to re-port if upstream evolves.
    public enum Buckets {
        /// Used for e2e latency, queue time, inference time, prefill time,
        /// decode time. Seconds.
        public static let requestLatency: [Double] = [
            0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0,
            10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0,
        ]

        /// Time to first token. Seconds.
        public static let timeToFirstToken: [Double] = [
            0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1,
            0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0,
        ]

        /// Time per output token (inter-token latency). Seconds.
        public static let timePerOutputToken: [Double] = [
            0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2,
            0.3, 0.4, 0.5, 0.75, 1.0, 2.5,
        ]

        /// Token-count histograms (prompt + generation).
        public static let tokenCount: [Double] = [
            1, 2, 5, 10, 20, 50, 100, 200, 500, 1000,
            2000, 5000, 10000, 20000, 50000, 100000,
        ]

        /// Sampling-parameter histograms (n, best_of).
        public static let samplingParam: [Double] = [1, 2, 5, 10, 20]
    }

    // MARK: - Histogram primitive

    /// Cumulative histogram with `+Inf` bucket. `bucketCounts[i]` is the
    /// number of observations `<= buckets[i]`; the final entry is `+Inf`
    /// (i.e. the total count, equal to `count`).
    public struct Histogram: Sendable {
        public let buckets: [Double]
        public var bucketCounts: [UInt64]
        public var sum: Double
        public var count: UInt64

        public init(buckets: [Double]) {
            self.buckets = buckets
            self.bucketCounts = Array(repeating: 0, count: buckets.count + 1)
            self.sum = 0
            self.count = 0
        }

        public mutating func observe(_ value: Double) {
            // Guard against NaN / negative noise — clamp to 0 so the
            // sum is meaningful.
            let v = (value.isFinite && value >= 0) ? value : 0
            sum += v
            count &+= 1
            for i in 0..<buckets.count where v <= buckets[i] {
                bucketCounts[i] &+= 1
            }
            // +Inf bucket always increments
            bucketCounts[buckets.count] &+= 1
        }
    }

    // MARK: - Storage

    private struct Counters {
        var genTokensTotal: UInt64 = 0
        var promptTokensTotal: UInt64 = 0
        var requestsStartedTotal: UInt64 = 0
        var requestsCompletedTotal: UInt64 = 0
        var cacheHitsTotal: UInt64 = 0
        var cacheMissesTotal: UInt64 = 0
        /// vLLM's `request_success_total{finished_reason=...}`. Keyed by
        /// the finished_reason string ("stop", "length", "abort", "error").
        var requestSuccessByReason: [String: UInt64] = [:]
    }

    private struct Histograms {
        var e2eLatency = Histogram(buckets: Buckets.requestLatency)
        var queueTime = Histogram(buckets: Buckets.requestLatency)
        var inferenceTime = Histogram(buckets: Buckets.requestLatency)
        var prefillTime = Histogram(buckets: Buckets.requestLatency)
        var decodeTime = Histogram(buckets: Buckets.requestLatency)
        var timeToFirstToken = Histogram(buckets: Buckets.timeToFirstToken)
        var timePerOutputToken = Histogram(buckets: Buckets.timePerOutputToken)
        var promptTokens = Histogram(buckets: Buckets.tokenCount)
        var generationTokens = Histogram(buckets: Buckets.tokenCount)
        var paramsN = Histogram(buckets: Buckets.samplingParam)
        var paramsBestOf = Histogram(buckets: Buckets.samplingParam)
    }

    private struct Meta {
        var modelName: String = ""
        var maxConcurrent: Int = 0
        var processStartEpoch: Double
    }

    private struct GaugeState {
        var running: GaugeReader?
        var waiting: GaugeReader?
        var gpuCacheUsage: FractionReader?
        var batchSizePeak: Int = 0
    }

    private let counters = OSAllocatedUnfairLock(initialState: Counters())
    private let histograms = OSAllocatedUnfairLock(initialState: Histograms())
    private let meta: OSAllocatedUnfairLock<Meta>
    private let gauges = OSAllocatedUnfairLock(initialState: GaugeState())

    private init() {
        self.meta = OSAllocatedUnfairLock(
            initialState: Meta(processStartEpoch: Date().timeIntervalSince1970)
        )
    }

    // MARK: - Configuration

    /// Called once at server startup by `MLXModelService` (or the serial
    /// path equivalent) with the resolved model id and the configured
    /// `--concurrent` capacity. Appears in every response as a label.
    public func setModel(_ name: String, maxConcurrent: Int) {
        meta.withLock { m in
            m.modelName = name
            m.maxConcurrent = maxConcurrent
        }
    }

    /// Register live-gauge providers. `running` returns the number of
    /// requests currently generating on the GPU (active batch size).
    /// `waiting` returns the number of requests queued behind the
    /// `--concurrent` cap. Both are polled once per `/metrics` request.
    public func registerGaugeReaders(
        running: @escaping GaugeReader,
        waiting: @escaping GaugeReader
    ) {
        gauges.withLock { g in
            g.running = running
            g.waiting = waiting
        }
    }

    /// Register a reader for `gpu_cache_usage_perc` (a value in [0, 1]).
    /// Polled once per `/metrics` request. Optional — if not registered,
    /// the gauge is omitted from the exposition.
    public func registerGpuCacheUsageReader(_ reader: @escaping FractionReader) {
        gauges.withLock { $0.gpuCacheUsage = reader }
    }

    /// Reset counters and histograms (for long-running processes that
    /// want to rebaseline). Gauge readers and metadata are preserved.
    public func reset() {
        counters.withLock { $0 = Counters() }
        histograms.withLock { $0 = Histograms() }
        gauges.withLock { $0.batchSizePeak = 0 }
    }

    // MARK: - Counter increments

    public func addGenTokens(_ n: Int = 1) {
        guard n > 0 else { return }
        counters.withLock { $0.genTokensTotal &+= UInt64(n) }
    }

    public func addPromptTokens(_ n: Int) {
        guard n > 0 else { return }
        counters.withLock { $0.promptTokensTotal &+= UInt64(n) }
    }

    public func requestStarted() {
        counters.withLock { $0.requestsStartedTotal &+= 1 }
    }

    public func requestCompleted() {
        counters.withLock { $0.requestsCompletedTotal &+= 1 }
    }

    public func cacheHit() {
        counters.withLock { $0.cacheHitsTotal &+= 1 }
    }

    public func cacheMiss() {
        counters.withLock { $0.cacheMissesTotal &+= 1 }
    }

    /// vLLM's `request_success_total{finished_reason=...}`. `reason` is
    /// the OpenAI-style finish reason — typically one of:
    /// `"stop"`, `"length"`, `"tool_calls"`, `"abort"`, `"error"`.
    /// Sanitized to lowercase with non-alphanumerics replaced by `_`
    /// before being used as a Prometheus label value.
    public func requestSucceeded(reason: String) {
        let key = Self.sanitizeReason(reason)
        counters.withLock { $0.requestSuccessByReason[key, default: 0] &+= 1 }
    }

    private static func sanitizeReason(_ s: String) -> String {
        let trimmed = s.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if trimmed.isEmpty { return "unknown" }
        var out = ""
        out.reserveCapacity(trimmed.count)
        for ch in trimmed.unicodeScalars {
            if (ch >= "a" && ch <= "z") || (ch >= "0" && ch <= "9") {
                out.unicodeScalars.append(ch)
            } else {
                out.append("_")
            }
        }
        return out
    }

    // MARK: - Histogram observations

    /// Per-request observation captured at completion. Pass `nil` for any
    /// timestamp the caller wasn't able to capture (e.g. `firstTokenAt`
    /// if the request produced zero tokens).
    public struct RequestObservation: Sendable {
        public var queuedAt: Double
        public var startedAt: Double?
        public var firstTokenAt: Double?
        public var completedAt: Double
        public var promptTokens: Int
        public var generationTokens: Int
        public var paramsN: Int
        public var paramsBestOf: Int

        public init(
            queuedAt: Double,
            startedAt: Double?,
            firstTokenAt: Double?,
            completedAt: Double,
            promptTokens: Int,
            generationTokens: Int,
            paramsN: Int = 1,
            paramsBestOf: Int = 1
        ) {
            self.queuedAt = queuedAt
            self.startedAt = startedAt
            self.firstTokenAt = firstTokenAt
            self.completedAt = completedAt
            self.promptTokens = promptTokens
            self.generationTokens = generationTokens
            self.paramsN = paramsN
            self.paramsBestOf = paramsBestOf
        }
    }

    /// Observe every histogram derivable from a single completed request.
    /// Safe to call from any thread — takes the histogram lock once.
    public func observeRequest(_ obs: RequestObservation) {
        let e2e = max(0, obs.completedAt - obs.queuedAt)
        let queue: Double = obs.startedAt.map { max(0, $0 - obs.queuedAt) } ?? 0
        let inference: Double = obs.startedAt.map { max(0, obs.completedAt - $0) } ?? e2e
        let prefill: Double? = obs.startedAt.flatMap { s in
            obs.firstTokenAt.map { ft in max(0, ft - s) }
        }
        let decode: Double? = obs.firstTokenAt.map { ft in
            max(0, obs.completedAt - ft)
        }
        let ttft: Double? = obs.firstTokenAt.map { ft in
            max(0, ft - obs.queuedAt)
        }
        let tpot: Double? = {
            guard let d = decode, obs.generationTokens > 1 else { return nil }
            return d / Double(obs.generationTokens - 1)
        }()

        histograms.withLock { h in
            h.e2eLatency.observe(e2e)
            h.queueTime.observe(queue)
            h.inferenceTime.observe(inference)
            if let p = prefill { h.prefillTime.observe(p) }
            if let d = decode { h.decodeTime.observe(d) }
            if let t = ttft { h.timeToFirstToken.observe(t) }
            if let p = tpot { h.timePerOutputToken.observe(p) }
            if obs.promptTokens > 0 {
                h.promptTokens.observe(Double(obs.promptTokens))
            }
            if obs.generationTokens > 0 {
                h.generationTokens.observe(Double(obs.generationTokens))
            }
            h.paramsN.observe(Double(max(1, obs.paramsN)))
            h.paramsBestOf.observe(Double(max(1, obs.paramsBestOf)))
        }
    }

    /// Lower-level observation helpers (use these when only one
    /// dimension is available, e.g. abort path).
    public func observeE2eLatency(_ seconds: Double) {
        histograms.withLock { $0.e2eLatency.observe(seconds) }
    }
    public func observeTimeToFirstToken(_ seconds: Double) {
        histograms.withLock { $0.timeToFirstToken.observe(seconds) }
    }
    public func observeTimePerOutputToken(_ seconds: Double) {
        histograms.withLock { $0.timePerOutputToken.observe(seconds) }
    }
    public func observePromptTokens(_ n: Int) {
        guard n > 0 else { return }
        histograms.withLock { $0.promptTokens.observe(Double(n)) }
    }
    public func observeGenerationTokens(_ n: Int) {
        guard n >= 0 else { return }
        histograms.withLock { $0.generationTokens.observe(Double(n)) }
    }

    // MARK: - Snapshot

    public struct Snapshot: Sendable {
        public let timestampMs: Int64
        public let processStartEpoch: Double
        public let modelName: String
        public let maxConcurrent: Int
        public let numRunning: Int
        public let numWaiting: Int
        public let batchSizePeak: Int
        public let gpuCacheUsage: Double?
        public let genTokensTotal: UInt64
        public let promptTokensTotal: UInt64
        public let requestsStartedTotal: UInt64
        public let requestsCompletedTotal: UInt64
        public let cacheHitsTotal: UInt64
        public let cacheMissesTotal: UInt64
        public let requestSuccessByReason: [String: UInt64]
        public let e2eLatency: Histogram
        public let queueTime: Histogram
        public let inferenceTime: Histogram
        public let prefillTime: Histogram
        public let decodeTime: Histogram
        public let timeToFirstToken: Histogram
        public let timePerOutputToken: Histogram
        public let promptTokens: Histogram
        public let generationTokens: Histogram
        public let paramsN: Histogram
        public let paramsBestOf: Histogram
    }

    /// Build a single-point-in-time snapshot of every metric. Cheap —
    /// four lock acquisitions, one call to each gauge reader.
    public func snapshot() -> Snapshot {
        let c = counters.withLock { $0 }
        let h = histograms.withLock { $0 }
        let m = meta.withLock { $0 }
        let (running, waiting, peak, gpuCache) = gauges.withLock {
            g -> (Int, Int, Int, Double?) in
            let r = g.running?() ?? 0
            let w = g.waiting?() ?? 0
            let cache = g.gpuCacheUsage?()
            if r > g.batchSizePeak { g.batchSizePeak = r }
            return (r, w, g.batchSizePeak, cache)
        }
        return Snapshot(
            timestampMs: Int64(Date().timeIntervalSince1970 * 1000),
            processStartEpoch: m.processStartEpoch,
            modelName: m.modelName,
            maxConcurrent: m.maxConcurrent,
            numRunning: running,
            numWaiting: waiting,
            batchSizePeak: peak,
            gpuCacheUsage: gpuCache,
            genTokensTotal: c.genTokensTotal,
            promptTokensTotal: c.promptTokensTotal,
            requestsStartedTotal: c.requestsStartedTotal,
            requestsCompletedTotal: c.requestsCompletedTotal,
            cacheHitsTotal: c.cacheHitsTotal,
            cacheMissesTotal: c.cacheMissesTotal,
            requestSuccessByReason: c.requestSuccessByReason,
            e2eLatency: h.e2eLatency,
            queueTime: h.queueTime,
            inferenceTime: h.inferenceTime,
            prefillTime: h.prefillTime,
            decodeTime: h.decodeTime,
            timeToFirstToken: h.timeToFirstToken,
            timePerOutputToken: h.timePerOutputToken,
            promptTokens: h.promptTokens,
            generationTokens: h.generationTokens,
            paramsN: h.paramsN,
            paramsBestOf: h.paramsBestOf
        )
    }
}
