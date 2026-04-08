import Vapor
import Foundation

/// `GET /metrics` — Prometheus text exposition format.
///
/// Mirrors vLLM's metric taxonomy (`vllm:` → `afm:` prefix, same
/// gauge/counter semantics) so Grafana dashboards written for vLLM
/// work against AFM with a search-and-replace of the namespace.
///
/// Reads from `StatsAggregator.shared`, which is populated by
/// `BatchScheduler` (batch path) and `MLXModelService` (serial path).
struct MetricsController: RouteCollection {

    let aggregator: StatsAggregator

    init(aggregator: StatsAggregator = .shared) {
        self.aggregator = aggregator
    }

    func boot(routes: RoutesBuilder) throws {
        routes.get("metrics", use: metrics)
    }

    func metrics(req: Request) async throws -> Response {
        let body = Self.renderPrometheus(aggregator.snapshot())
        let response = Response(status: .ok)
        response.headers.replaceOrAdd(
            name: .contentType,
            value: "text/plain; version=0.0.4; charset=utf-8"
        )
        response.headers.add(name: .accessControlAllowOrigin, value: "*")
        response.body = .init(string: body)
        return response
    }

    // MARK: - Prometheus text format

    /// Escape a string for inclusion in a Prometheus label value.
    /// Spec: backslash, double-quote and newline must be escaped.
    private static func labelEscape(_ s: String) -> String {
        var out = ""
        out.reserveCapacity(s.count)
        for ch in s {
            switch ch {
            case "\\": out += "\\\\"
            case "\"": out += "\\\""
            case "\n": out += "\\n"
            default:   out.append(ch)
            }
        }
        return out
    }

    static func renderPrometheus(_ s: StatsAggregator.Snapshot) -> String {
        let modelLabel = "{model_name=\"\(labelEscape(s.modelName))\"}"

        var out = ""
        out.reserveCapacity(2048)

        func gauge(_ name: String, _ help: String, _ value: String) {
            out += "# HELP \(name) \(help)\n"
            out += "# TYPE \(name) gauge\n"
            out += "\(name)\(modelLabel) \(value)\n"
        }
        func counter(_ name: String, _ help: String, _ value: UInt64) {
            out += "# HELP \(name) \(help)\n"
            out += "# TYPE \(name) counter\n"
            out += "\(name)\(modelLabel) \(value)\n"
        }

        // --- Gauges ---------------------------------------------------------
        gauge(
            "afm:num_requests_running",
            "Number of requests currently generating on the GPU (active batch size).",
            String(s.numRunning)
        )
        gauge(
            "afm:num_requests_waiting",
            "Number of requests queued behind the --concurrent capacity.",
            String(s.numWaiting)
        )
        gauge(
            "afm:batch_size_peak",
            "Highest num_requests_running observed since server start.",
            String(s.batchSizePeak)
        )
        gauge(
            "afm:max_concurrent_slots",
            "Configured --concurrent capacity of the server.",
            String(s.maxConcurrent)
        )

        // --- Counters -------------------------------------------------------
        counter(
            "afm:generation_tokens_total",
            "Total number of output tokens generated since server start.",
            s.genTokensTotal
        )
        counter(
            "afm:prompt_tokens_total",
            "Total number of prompt tokens processed by prefill since server start.",
            s.promptTokensTotal
        )
        counter(
            "afm:requests_started_total",
            "Total number of requests accepted since server start.",
            s.requestsStartedTotal
        )
        counter(
            "afm:requests_completed_total",
            "Total number of requests fully completed since server start.",
            s.requestsCompletedTotal
        )
        counter(
            "afm:radix_cache_hits_total",
            "Total number of prefix cache hits (radix tree) since server start.",
            s.cacheHitsTotal
        )
        counter(
            "afm:radix_cache_misses_total",
            "Total number of prefix cache misses (radix tree) since server start.",
            s.cacheMissesTotal
        )

        // --- Process info (unlabeled, scraper-friendly) ---------------------
        out += "# HELP afm:process_start_time_seconds Unix epoch time the afm process started.\n"
        out += "# TYPE afm:process_start_time_seconds gauge\n"
        out += "afm:process_start_time_seconds \(s.processStartEpoch)\n"

        out += "# HELP afm:snapshot_timestamp_ms Unix epoch time (ms) this snapshot was taken.\n"
        out += "# TYPE afm:snapshot_timestamp_ms gauge\n"
        out += "afm:snapshot_timestamp_ms \(s.timestampMs)\n"

        return out
    }
}
