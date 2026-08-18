@_spi(AFMKitTelemetry) import AFMKitCore
import Foundation
import os

public final class InferenceTelemetryCollector: @unchecked Sendable {
    private struct MutableHistogram: Sendable {
        let buckets: [Double]
        var bucketCounts: [UInt64]
        var sum: Double = 0
        var count: UInt64 = 0

        init(buckets: [Double]) {
            self.buckets = buckets
            self.bucketCounts = Array(repeating: 0, count: buckets.count + 1)
        }

        mutating func observe(_ value: Double) {
            guard value.isFinite, value >= 0 else { return }
            sum += value
            count &+= 1
            for index in buckets.indices where value <= buckets[index] {
                bucketCounts[index] &+= 1
            }
            bucketCounts[buckets.count] &+= 1
        }

        var snapshot: AFMHistogramSnapshot {
            AFMHistogramSnapshot(
                buckets: buckets,
                bucketCounts: bucketCounts,
                sum: sum,
                count: count
            )
        }
    }

    private struct RequestState: Sendable {
        let acceptedAt: Double
        var startedAt: Double?
        var firstTokenAt: Double?
        var previousTokenAt: Double?
        var outputTokenTimestamps: [Double]
    }

    private struct TimedAmount: Sendable {
        let timestamp: Double
        let amount: UInt64
    }

    private struct State: Sendable {
        let processStartEpochSeconds: Double
        var modelName = ""
        var maximumConcurrentRequests = 0

        var runningRequests = 0
        var waitingRequests = 0
        var peakRunningRequests = 0
        var logicalCacheUsage = 0.0
        var memoryCacheUsage: Double?
        var prefixCacheFill: Double?

        var generatedTokensTotal: UInt64 = 0
        var fullPromptTokensTotal: UInt64 = 0
        var computedPromptTokensTotal: UInt64 = 0
        var acceptedRequestsTotal: UInt64 = 0
        var terminalRequestsTotal: UInt64 = 0
        var preemptionsTotal: UInt64 = 0
        var prefixCacheQueriesTotal: UInt64 = 0
        var prefixCacheHitsTotal: UInt64 = 0
        var speculativeDraftRoundsTotal: UInt64 = 0
        var speculativeDraftTokensTotal: UInt64 = 0
        var speculativeAcceptedTokensTotal: UInt64 = 0
        var terminalCounts = Dictionary(
            uniqueKeysWithValues: AFMInferenceFinishReason.allCases.map { ($0.rawValue, UInt64(0)) }
        )
        var legacyTerminalCounts: [String: UInt64] = [:]
        var failureCounts = Dictionary(
            uniqueKeysWithValues: AFMInferenceFailureReason.allCases.map { ($0.rawValue, UInt64(0)) }
        )
        var ingressRejections = Dictionary(
            uniqueKeysWithValues: AFMIngressRejectionReason.allCases.map { ($0.rawValue, UInt64(0)) }
        )

        var activeConnections = Set<AFMIngressConnectionToken>()
        var activeConnectionsPeak = 0
        var legacyCacheHitsTotal: UInt64 = 0
        var legacyCacheMissesTotal: UInt64 = 0

        var requests: [AFMInferenceRequestToken: RequestState] = [:]
        var promptWindow: [TimedAmount] = []
        var generationWindow: [TimedAmount] = []
        var terminalWindow: [TimedAmount] = []

        var endToEndLatency = MutableHistogram(buckets: Buckets.requestLatency)
        var queueLatency = MutableHistogram(buckets: Buckets.requestLatency)
        var inferenceLatency = MutableHistogram(buckets: Buckets.requestLatency)
        var prefillLatency = MutableHistogram(buckets: Buckets.requestLatency)
        var decodeLatency = MutableHistogram(buckets: Buckets.requestLatency)
        var timeToFirstToken = MutableHistogram(buckets: Buckets.timeToFirstToken)
        var timePerOutputToken = MutableHistogram(buckets: Buckets.timePerOutputToken)
        var interTokenLatency = MutableHistogram(buckets: Buckets.timePerOutputToken)
        var fullPromptTokens = MutableHistogram(buckets: Buckets.tokenCount)
        var computedPromptTokens = MutableHistogram(buckets: Buckets.tokenCount)
        var generatedTokens = MutableHistogram(buckets: Buckets.tokenCount)
        var samplingN = MutableHistogram(buckets: Buckets.samplingParam)
        var samplingBestOf = MutableHistogram(buckets: Buckets.samplingParam)

        init(processStartEpochSeconds: Double) {
            self.processStartEpochSeconds = processStartEpochSeconds
        }
    }

    private enum Buckets {
        static let requestLatency: [Double] = [
            0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0,
            10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0,
        ]
        static let timeToFirstToken: [Double] = [
            0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1,
            0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0,
        ]
        static let timePerOutputToken: [Double] = [
            0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2,
            0.3, 0.4, 0.5, 0.75, 1.0, 2.5,
        ]
        static let tokenCount: [Double] = [
            1, 2, 5, 10, 20, 50, 100, 200, 500, 1000,
            2000, 5000, 10000, 20000, 50000, 100000,
        ]
        static let samplingParam: [Double] = [1, 2, 5, 10, 20]
    }

    private static let rollingWindowSeconds = 10.0

    private let now: @Sendable () -> Double
    private let wallTime: @Sendable () -> Double
    private let state: OSAllocatedUnfairLock<State>

    public init(
        now: @escaping @Sendable () -> Double = { ProcessInfo.processInfo.systemUptime },
        wallTime: @escaping @Sendable () -> Double = { Date().timeIntervalSince1970 }
    ) {
        self.now = now
        self.wallTime = wallTime
        self.state = OSAllocatedUnfairLock(
            initialState: State(processStartEpochSeconds: wallTime())
        )
    }

    public func configure(modelName: String, maximumConcurrentRequests: Int) {
        state.withLock { state in
            state.modelName = modelName
            state.maximumConcurrentRequests = max(0, maximumConcurrentRequests)
        }
    }

    public func reset() {
        state.withLock { current in
            let start = current.processStartEpochSeconds
            let modelName = current.modelName
            let capacity = current.maximumConcurrentRequests
            let connections = current.activeConnections
            let connectionPeak = current.activeConnectionsPeak
            current = State(processStartEpochSeconds: start)
            current.modelName = modelName
            current.maximumConcurrentRequests = capacity
            current.activeConnections = connections
            current.activeConnectionsPeak = connectionPeak
        }
    }

    // MARK: Legacy forwarding

    public func legacyAddGeneratedTokens(_ count: Int) {
        guard count > 0 else { return }
        let timestamp = now()
        state.withLock { state in
            let amount = UInt64(count)
            state.generatedTokensTotal &+= amount
            state.generationWindow.append(TimedAmount(timestamp: timestamp, amount: amount))
        }
    }

    public func legacyAddComputedPromptTokens(_ count: Int) {
        guard count > 0 else { return }
        let timestamp = now()
        state.withLock { state in
            let amount = UInt64(count)
            state.computedPromptTokensTotal &+= amount
            state.promptWindow.append(TimedAmount(timestamp: timestamp, amount: amount))
        }
    }

    public func legacyRequestStarted() {
        state.withLock { $0.acceptedRequestsTotal &+= 1 }
    }

    public func legacyRequestCompleted() {
        let timestamp = now()
        state.withLock { state in
            state.terminalRequestsTotal &+= 1
            state.terminalWindow.append(TimedAmount(timestamp: timestamp, amount: 1))
        }
    }

    public func legacyCacheHit() {
        state.withLock { $0.legacyCacheHitsTotal &+= 1 }
    }

    public func legacyCacheMiss() {
        state.withLock { $0.legacyCacheMissesTotal &+= 1 }
    }

    public func legacyRequestSucceeded(reason: String) {
        let key = Self.legacySanitizedReason(reason)
        state.withLock { $0.legacyTerminalCounts[key, default: 0] &+= 1 }
    }

    public func legacyObserveRequest(
        queuedAt: Double,
        startedAt: Double?,
        firstTokenAt: Double?,
        completedAt: Double,
        promptTokens: Int,
        generationTokens: Int,
        samplingN: Int,
        samplingBestOf: Int
    ) {
        state.withLock { state in
            Self.observeLatency(
                state: &state,
                acceptedAt: queuedAt,
                startedAt: startedAt,
                firstTokenAt: firstTokenAt,
                previousTokenTimes: [],
                completedAt: completedAt,
                fullPromptTokens: 0,
                computedPromptTokens: promptTokens,
                generatedTokens: generationTokens,
                samplingN: samplingN,
                samplingBestOf: samplingBestOf
            )
        }
    }

    public func legacyObserveEndToEndLatency(_ seconds: Double) {
        state.withLock { $0.endToEndLatency.observe(seconds) }
    }

    public func legacyObserveTimeToFirstToken(_ seconds: Double) {
        state.withLock { $0.timeToFirstToken.observe(seconds) }
    }

    public func legacyObserveTimePerOutputToken(_ seconds: Double) {
        state.withLock { $0.timePerOutputToken.observe(seconds) }
    }

    public func legacyObserveComputedPromptTokens(_ count: Int) {
        guard count > 0 else { return }
        state.withLock { $0.computedPromptTokens.observe(Double(count)) }
    }

    public func legacyObserveGeneratedTokens(_ count: Int) {
        guard count >= 0 else { return }
        state.withLock { $0.generatedTokens.observe(Double(count)) }
    }

    private static func legacySanitizedReason(_ reason: String) -> String {
        let trimmed = reason.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard !trimmed.isEmpty else { return "unknown" }
        var result = ""
        result.reserveCapacity(trimmed.count)
        for scalar in trimmed.unicodeScalars {
            if (scalar >= "a" && scalar <= "z") || (scalar >= "0" && scalar <= "9") {
                result.unicodeScalars.append(scalar)
            } else {
                result.append("_")
            }
        }
        return result
    }

    private static func observeLatency(
        state: inout State,
        acceptedAt: Double,
        startedAt: Double?,
        firstTokenAt: Double?,
        previousTokenTimes: [Double],
        completedAt: Double,
        fullPromptTokens: Int,
        computedPromptTokens: Int,
        generatedTokens: Int,
        samplingN: Int,
        samplingBestOf: Int
    ) {
        let endToEnd = max(0, completedAt - acceptedAt)
        let queue = startedAt.map { max(0, $0 - acceptedAt) } ?? 0
        let inference = startedAt.map { max(0, completedAt - $0) } ?? endToEnd
        let prefill = startedAt.flatMap { start in firstTokenAt.map { max(0, $0 - start) } }
        let decode = firstTokenAt.map { max(0, completedAt - $0) }
        let ttft = firstTokenAt.map { max(0, $0 - acceptedAt) }

        state.endToEndLatency.observe(endToEnd)
        state.queueLatency.observe(queue)
        state.inferenceLatency.observe(inference)
        if let prefill { state.prefillLatency.observe(prefill) }
        if let decode {
            state.decodeLatency.observe(decode)
            if generatedTokens > 1 {
                state.timePerOutputToken.observe(decode / Double(generatedTokens - 1))
            }
        }
        if let ttft { state.timeToFirstToken.observe(ttft) }
        for interval in zip(previousTokenTimes, previousTokenTimes.dropFirst()).map({ $1 - $0 }) {
            state.interTokenLatency.observe(max(0, interval))
        }
        if fullPromptTokens > 0 { state.fullPromptTokens.observe(Double(fullPromptTokens)) }
        if computedPromptTokens > 0 {
            state.computedPromptTokens.observe(Double(computedPromptTokens))
        }
        if generatedTokens >= 0 { state.generatedTokens.observe(Double(generatedTokens)) }
        state.samplingN.observe(Double(max(1, samplingN)))
        state.samplingBestOf.observe(Double(max(1, samplingBestOf)))
    }

    private static func clampFraction(_ value: Double) -> Double {
        guard value.isFinite else { return 0 }
        return min(1, max(0, value))
    }

    private static func namedCounts(_ values: [String: UInt64]) -> [AFMNamedCount] {
        values.keys.sorted().map { AFMNamedCount(name: $0, count: values[$0, default: 0]) }
    }

    private static func windowRate(
        _ values: inout [TimedAmount],
        at timestamp: Double
    ) -> Double {
        let cutoff = timestamp - rollingWindowSeconds
        values.removeAll { $0.timestamp < cutoff }
        let sum = values.reduce(UInt64(0)) { $0 &+ $1.amount }
        return Double(sum) / rollingWindowSeconds
    }
}

extension InferenceTelemetryCollector: AFMInferenceTelemetryObserving {
    public func requestAccepted(at timestamp: Double) -> AFMInferenceRequestToken {
        let token = AFMInferenceRequestToken()
        state.withLock { state in
            state.acceptedRequestsTotal &+= 1
            state.requests[token] = RequestState(
                acceptedAt: timestamp,
                outputTokenTimestamps: []
            )
        }
        return token
    }

    public func requestStarted(_ token: AFMInferenceRequestToken, at timestamp: Double) {
        state.withLock { state in
            guard var request = state.requests[token], request.startedAt == nil else { return }
            request.startedAt = timestamp
            state.requests[token] = request
        }
    }

    public func outputToken(_ token: AFMInferenceRequestToken, at timestamp: Double) {
        state.withLock { state in
            guard var request = state.requests[token] else { return }
            if request.firstTokenAt == nil { request.firstTokenAt = timestamp }
            request.previousTokenAt = timestamp
            request.outputTokenTimestamps.append(timestamp)
            state.requests[token] = request
        }
    }

    public func prefixCacheObserved(queriedTokens: Int, hitTokens: Int) {
        guard queriedTokens > 0 else { return }
        state.withLock { state in
            state.prefixCacheQueriesTotal &+= UInt64(queriedTokens)
            state.prefixCacheHitsTotal &+= UInt64(min(max(0, hitTokens), queriedTokens))
        }
    }

    public func speculativeRound(draftTokens: Int, acceptedTokens: Int) {
        guard draftTokens > 0 else { return }
        state.withLock { state in
            state.speculativeDraftRoundsTotal &+= 1
            state.speculativeDraftTokensTotal &+= UInt64(draftTokens)
            state.speculativeAcceptedTokensTotal &+= UInt64(
                min(max(0, acceptedTokens), draftTokens)
            )
        }
    }

    public func preemptionObserved() {
        state.withLock { $0.preemptionsTotal &+= 1 }
    }

    public func updateProviderState(_ providerState: AFMInferenceProviderState) {
        state.withLock { state in
            state.runningRequests = max(0, providerState.runningRequests)
            state.waitingRequests = max(0, providerState.waitingRequests)
            state.peakRunningRequests = max(state.peakRunningRequests, state.runningRequests)
            if providerState.logicalCacheCapacity > 0 {
                state.logicalCacheUsage = Self.clampFraction(
                    Double(providerState.activeLogicalCachePositions)
                        / Double(providerState.logicalCacheCapacity)
                )
            } else {
                state.logicalCacheUsage = 0
            }
            state.memoryCacheUsage = providerState.memoryCacheUsage.map(Self.clampFraction)
            state.prefixCacheFill = providerState.prefixCacheFill.map(Self.clampFraction)
        }
    }

    public func requestFinished(
        _ token: AFMInferenceRequestToken,
        observation: AFMInferenceRequestFinishObservation
    ) -> Bool {
        let timestamp = now()
        return state.withLock { state in
            guard let request = state.requests.removeValue(forKey: token) else { return false }
            let full = UInt64(max(0, observation.fullPromptTokens))
            let computed = UInt64(max(0, observation.computedPromptTokens))
            let generated = UInt64(max(0, observation.generatedTokens))
            state.fullPromptTokensTotal &+= full
            state.computedPromptTokensTotal &+= computed
            state.generatedTokensTotal &+= generated
            state.terminalRequestsTotal &+= 1
            state.terminalCounts[observation.reason.rawValue, default: 0] &+= 1
            state.promptWindow.append(TimedAmount(timestamp: timestamp, amount: computed))
            state.generationWindow.append(TimedAmount(timestamp: timestamp, amount: generated))
            state.terminalWindow.append(TimedAmount(timestamp: timestamp, amount: 1))
            Self.observeLatency(
                state: &state,
                acceptedAt: request.acceptedAt,
                startedAt: request.startedAt,
                firstTokenAt: request.firstTokenAt,
                previousTokenTimes: request.outputTokenTimestamps,
                completedAt: observation.completedAt,
                fullPromptTokens: observation.fullPromptTokens,
                computedPromptTokens: observation.computedPromptTokens,
                generatedTokens: observation.generatedTokens,
                samplingN: observation.samplingN,
                samplingBestOf: observation.samplingBestOf
            )
            return true
        }
    }

    public func requestFailed(
        _ token: AFMInferenceRequestToken,
        reason: AFMInferenceFailureReason,
        at timestamp: Double
    ) -> Bool {
        state.withLock { state in
            guard let request = state.requests.removeValue(forKey: token) else { return false }
            state.terminalRequestsTotal &+= 1
            let finishReason: AFMInferenceFinishReason = reason == .cancelled ? .abort : .error
            state.terminalCounts[finishReason.rawValue, default: 0] &+= 1
            state.failureCounts[reason.rawValue, default: 0] &+= 1
            state.terminalWindow.append(TimedAmount(timestamp: timestamp, amount: 1))
            state.endToEndLatency.observe(max(0, timestamp - request.acceptedAt))
            if let startedAt = request.startedAt {
                state.queueLatency.observe(max(0, startedAt - request.acceptedAt))
                state.inferenceLatency.observe(max(0, timestamp - startedAt))
            }
            return true
        }
    }
}

extension InferenceTelemetryCollector: AFMIngressTelemetryRecording {
    public func recordRejection(_ reason: AFMIngressRejectionReason) {
        state.withLock { $0.ingressRejections[reason.rawValue, default: 0] &+= 1 }
    }

    public func connectionOpened() -> AFMIngressConnectionToken {
        let token = AFMIngressConnectionToken()
        state.withLock { state in
            state.activeConnections.insert(token)
            state.activeConnectionsPeak = max(
                state.activeConnectionsPeak,
                state.activeConnections.count
            )
        }
        return token
    }

    public func connectionClosed(_ token: AFMIngressConnectionToken) {
        state.withLock { state in
            _ = state.activeConnections.remove(token)
        }
    }
}

extension InferenceTelemetryCollector: AFMInferenceMetricsSnapshotSource {
    public func metricsSnapshot() -> AFMInferenceMetricsSnapshot {
        let monotonicNow = now()
        let wallNow = wallTime()
        return state.withLock { state in
            let promptRate = Self.windowRate(&state.promptWindow, at: monotonicNow)
            let generationRate = Self.windowRate(&state.generationWindow, at: monotonicNow)
            let requestRate = Self.windowRate(&state.terminalWindow, at: monotonicNow)
            let failures = state.failureCounts.merging(state.ingressRejections) { $0 &+ $1 }
            return AFMInferenceMetricsSnapshot(
                timestampMilliseconds: Int64(wallNow * 1_000),
                processStartEpochSeconds: state.processStartEpochSeconds,
                modelName: state.modelName,
                maximumConcurrentRequests: state.maximumConcurrentRequests,
                runningRequests: state.runningRequests,
                waitingRequests: state.waitingRequests,
                peakRunningRequests: state.peakRunningRequests,
                logicalCacheUsage: state.logicalCacheUsage,
                memoryCacheUsage: state.memoryCacheUsage,
                prefixCacheFill: state.prefixCacheFill,
                generatedTokensTotal: state.generatedTokensTotal,
                fullPromptTokensTotal: state.fullPromptTokensTotal,
                computedPromptTokensTotal: state.computedPromptTokensTotal,
                acceptedRequestsTotal: state.acceptedRequestsTotal,
                terminalRequestsTotal: state.terminalRequestsTotal,
                preemptionsTotal: state.preemptionsTotal,
                prefixCacheQueriesTotal: state.prefixCacheQueriesTotal,
                prefixCacheHitsTotal: state.prefixCacheHitsTotal,
                speculativeDraftRoundsTotal: state.speculativeDraftRoundsTotal,
                speculativeDraftTokensTotal: state.speculativeDraftTokensTotal,
                speculativeAcceptedTokensTotal: state.speculativeAcceptedTokensTotal,
                terminalCounts: Self.namedCounts(state.terminalCounts),
                failureCounts: Self.namedCounts(failures),
                supplementalCounts: [
                    AFMNamedCount(name: "legacy_cache_hits", count: state.legacyCacheHitsTotal),
                    AFMNamedCount(name: "legacy_cache_misses", count: state.legacyCacheMissesTotal),
                ] + state.legacyTerminalCounts.keys.sorted().map {
                    AFMNamedCount(
                        name: "legacy_finish:\($0)",
                        count: state.legacyTerminalCounts[$0, default: 0]
                    )
                },
                supplementalIntegerGauges: [
                    AFMNamedIntegerGauge(
                        name: "active_connections",
                        value: state.activeConnections.count
                    ),
                    AFMNamedIntegerGauge(
                        name: "active_connections_peak",
                        value: state.activeConnectionsPeak
                    ),
                ],
                supplementalDoubleGauges: [
                    AFMNamedDoubleGauge(name: "computed_prompt_throughput", value: promptRate),
                    AFMNamedDoubleGauge(name: "generation_throughput", value: generationRate),
                    AFMNamedDoubleGauge(name: "request_throughput", value: requestRate),
                ],
                endToEndLatency: state.endToEndLatency.snapshot,
                queueLatency: state.queueLatency.snapshot,
                inferenceLatency: state.inferenceLatency.snapshot,
                prefillLatency: state.prefillLatency.snapshot,
                decodeLatency: state.decodeLatency.snapshot,
                timeToFirstToken: state.timeToFirstToken.snapshot,
                timePerOutputToken: state.timePerOutputToken.snapshot,
                interTokenLatency: state.interTokenLatency.snapshot,
                fullPromptTokens: state.fullPromptTokens.snapshot,
                computedPromptTokens: state.computedPromptTokens.snapshot,
                generatedTokens: state.generatedTokens.snapshot,
                samplingN: state.samplingN.snapshot,
                samplingBestOf: state.samplingBestOf.snapshot
            )
        }
    }
}
