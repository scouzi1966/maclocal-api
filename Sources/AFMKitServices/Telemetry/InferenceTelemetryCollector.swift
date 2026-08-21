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
        var recordedFullPromptTokens: UInt64
        var recordedComputedPromptTokens: UInt64
        var recordedGeneratedTokens: UInt64
    }

    private struct TimedAmount: Sendable {
        let timestamp: Double
        var amount: UInt64
    }

    private struct RollingWindow: Sendable {
        private static let bucketWidth = 0.1
        private static let maximumBucketCount = 102
        private var buckets: [TimedAmount] = []

        mutating func record(_ amount: UInt64, at timestamp: Double, window: Double) {
            guard amount > 0, timestamp.isFinite else { return }
            prune(at: timestamp, window: window)
            let bucketTimestamp = floor(timestamp / Self.bucketWidth) * Self.bucketWidth
            if let lastIndex = buckets.indices.last,
               buckets[lastIndex].timestamp == bucketTimestamp {
                buckets[lastIndex].amount &+= amount
            } else if buckets.last.map({ bucketTimestamp > $0.timestamp }) ?? true {
                buckets.append(TimedAmount(timestamp: bucketTimestamp, amount: amount))
            } else if let existingIndex = buckets.firstIndex(where: {
                $0.timestamp == bucketTimestamp
            }) {
                buckets[existingIndex].amount &+= amount
            } else {
                let insertionIndex = buckets.firstIndex(where: {
                    $0.timestamp > bucketTimestamp
                }) ?? buckets.endIndex
                buckets.insert(
                    TimedAmount(timestamp: bucketTimestamp, amount: amount),
                    at: insertionIndex
                )
            }
            if buckets.count > Self.maximumBucketCount {
                buckets.removeFirst(buckets.count - Self.maximumBucketCount)
            }
        }

        mutating func rate(at timestamp: Double, window: Double) -> Double {
            prune(at: timestamp, window: window)
            let sum = buckets.reduce(UInt64(0)) { $0 &+ $1.amount }
            return Double(sum) / window
        }

        private mutating func prune(at timestamp: Double, window: Double) {
            let cutoff = timestamp - window
            buckets.removeAll { $0.timestamp <= cutoff }
        }
    }

    private struct State: Sendable {
        let processStartEpochSeconds: Double
        var modelName = ""
        var maximumConcurrentRequests = 0
        var maximumContextTokens: Int

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
        var promptWindow = RollingWindow()
        var generationWindow = RollingWindow()
        var terminalWindow = RollingWindow()

        var endToEndLatency = MutableHistogram(buckets: Buckets.requestLatency)
        var queueLatency = MutableHistogram(buckets: Buckets.requestLatency)
        var inferenceLatency = MutableHistogram(buckets: Buckets.requestLatency)
        var prefillLatency = MutableHistogram(buckets: Buckets.requestLatency)
        var decodeLatency = MutableHistogram(buckets: Buckets.requestLatency)
        var timeToFirstToken = MutableHistogram(buckets: Buckets.timeToFirstToken)
        var timePerOutputToken = MutableHistogram(buckets: Buckets.timePerOutputToken)
        var interTokenLatency = MutableHistogram(buckets: Buckets.timePerOutputToken)
        var fullPromptTokens: MutableHistogram
        var computedPromptTokens: MutableHistogram
        var generatedTokens: MutableHistogram
        var maximumGeneratedTokens: MutableHistogram
        var maximumOutputTokens: MutableHistogram
        var samplingN = MutableHistogram(buckets: Buckets.samplingParam)
        var samplingBestOf = MutableHistogram(buckets: Buckets.samplingParam)

        init(
            processStartEpochSeconds: Double,
            maximumContextTokens: Int = Buckets.defaultMaximumContextTokens
        ) {
            self.processStartEpochSeconds = processStartEpochSeconds
            self.maximumContextTokens = max(1, maximumContextTokens)
            let tokenBuckets = Buckets.tokenCount(maximum: self.maximumContextTokens)
            self.fullPromptTokens = MutableHistogram(buckets: tokenBuckets)
            self.computedPromptTokens = MutableHistogram(buckets: tokenBuckets)
            self.generatedTokens = MutableHistogram(buckets: tokenBuckets)
            self.maximumGeneratedTokens = MutableHistogram(buckets: tokenBuckets)
            self.maximumOutputTokens = MutableHistogram(buckets: tokenBuckets)
        }

        mutating func configureMaximumContextTokens(_ maximum: Int) {
            guard maximum > 0, maximum != maximumContextTokens else { return }
            guard fullPromptTokens.count == 0,
                  computedPromptTokens.count == 0,
                  generatedTokens.count == 0,
                  maximumGeneratedTokens.count == 0,
                  maximumOutputTokens.count == 0 else {
                return
            }
            maximumContextTokens = maximum
            let tokenBuckets = Buckets.tokenCount(maximum: maximum)
            fullPromptTokens = MutableHistogram(buckets: tokenBuckets)
            computedPromptTokens = MutableHistogram(buckets: tokenBuckets)
            generatedTokens = MutableHistogram(buckets: tokenBuckets)
            maximumGeneratedTokens = MutableHistogram(buckets: tokenBuckets)
            maximumOutputTokens = MutableHistogram(buckets: tokenBuckets)
        }
    }

    private enum Buckets {
        static let requestLatency: [Double] = [
            0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0,
            10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0,
            120.0, 240.0, 480.0, 960.0, 1920.0, 7680.0,
        ]
        static let timeToFirstToken: [Double] = [
            0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1,
            0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0,
            20.0, 40.0, 80.0, 160.0, 640.0, 2560.0,
        ]
        static let timePerOutputToken: [Double] = [
            0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2,
            0.3, 0.4, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5,
            10.0, 20.0, 40.0, 80.0,
        ]
        static let defaultMaximumContextTokens = 100_000

        static func tokenCount(maximum: Int) -> [Double] {
            guard maximum > 0 else { return [] }
            var buckets: [Double] = []
            var magnitude = 1
            while true {
                for mantissa in [1, 2, 5] {
                    let value = mantissa * magnitude
                    if value > maximum { return buckets }
                    buckets.append(Double(value))
                }
                guard magnitude <= Int.max / 10 else { return buckets }
                magnitude *= 10
            }
        }
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

    public func configure(
        modelName: String,
        maximumConcurrentRequests: Int,
        maximumContextTokens: Int = 0
    ) {
        state.withLock { state in
            state.modelName = modelName
            state.maximumConcurrentRequests = max(0, maximumConcurrentRequests)
            state.configureMaximumContextTokens(maximumContextTokens)
        }
    }

    public func reset() {
        state.withLock { current in
            let start = current.processStartEpochSeconds
            let modelName = current.modelName
            let capacity = current.maximumConcurrentRequests
            let maximumContextTokens = current.maximumContextTokens
            let connections = current.activeConnections
            let connectionPeak = current.activeConnectionsPeak
            let requests = current.requests
            let runningRequests = current.runningRequests
            let waitingRequests = current.waitingRequests
            let logicalCacheUsage = current.logicalCacheUsage
            let memoryCacheUsage = current.memoryCacheUsage
            let prefixCacheFill = current.prefixCacheFill
            current = State(
                processStartEpochSeconds: start,
                maximumContextTokens: maximumContextTokens
            )
            current.modelName = modelName
            current.maximumConcurrentRequests = capacity
            current.activeConnections = connections
            current.activeConnectionsPeak = connectionPeak
            current.requests = requests
            current.runningRequests = runningRequests
            current.waitingRequests = waitingRequests
            current.peakRunningRequests = runningRequests
            current.logicalCacheUsage = logicalCacheUsage
            current.memoryCacheUsage = memoryCacheUsage
            current.prefixCacheFill = prefixCacheFill
        }
    }

    // MARK: Legacy forwarding

    public func legacyAddGeneratedTokens(_ count: Int) {
        guard count > 0 else { return }
        let timestamp = now()
        state.withLock { state in
            let amount = UInt64(count)
            state.generatedTokensTotal &+= amount
            state.generationWindow.record(
                amount,
                at: timestamp,
                window: Self.rollingWindowSeconds
            )
        }
    }

    public func legacyAddComputedPromptTokens(_ count: Int) {
        guard count > 0 else { return }
        let timestamp = now()
        state.withLock { state in
            let amount = UInt64(count)
            state.computedPromptTokensTotal &+= amount
            state.promptWindow.record(
                amount,
                at: timestamp,
                window: Self.rollingWindowSeconds
            )
        }
    }

    public func legacyRequestStarted() {
        state.withLock { $0.acceptedRequestsTotal &+= 1 }
    }

    public func legacyRequestCompleted() {
        let timestamp = now()
        state.withLock { state in
            state.terminalRequestsTotal &+= 1
            state.terminalWindow.record(1, at: timestamp, window: Self.rollingWindowSeconds)
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
                completedAt: completedAt,
                fullPromptTokens: 0,
                computedPromptTokens: promptTokens,
                generatedTokens: generationTokens,
                maximumOutputTokens: nil,
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
        completedAt: Double,
        fullPromptTokens: Int,
        computedPromptTokens: Int,
        generatedTokens: Int,
        maximumOutputTokens: Int?,
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
            } else if generatedTokens == 1 {
                state.timePerOutputToken.observe(0)
            }
        }
        if let ttft { state.timeToFirstToken.observe(ttft) }
        if fullPromptTokens > 0 { state.fullPromptTokens.observe(Double(fullPromptTokens)) }
        if computedPromptTokens > 0 {
            state.computedPromptTokens.observe(Double(computedPromptTokens))
        }
        if generatedTokens >= 0 { state.generatedTokens.observe(Double(generatedTokens)) }
        if generatedTokens >= 0 {
            state.maximumGeneratedTokens.observe(Double(generatedTokens))
        }
        if let maximumOutputTokens, maximumOutputTokens >= 0 {
            state.maximumOutputTokens.observe(Double(maximumOutputTokens))
        }
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

}

extension InferenceTelemetryCollector: AFMInferenceTelemetryObserving {
    public func requestAccepted(at timestamp: Double) -> AFMInferenceRequestToken {
        let token = AFMInferenceRequestToken()
        state.withLock { state in
            state.acceptedRequestsTotal &+= 1
            state.requests[token] = RequestState(
                acceptedAt: timestamp,
                recordedFullPromptTokens: 0,
                recordedComputedPromptTokens: 0,
                recordedGeneratedTokens: 0
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

    public func promptTokensProcessed(
        _ token: AFMInferenceRequestToken,
        fullPromptTokens: Int,
        computedPromptTokens: Int,
        at timestamp: Double
    ) {
        let receivedAt = now()
        state.withLock { state in
            guard var request = state.requests[token] else { return }
            let full = UInt64(max(0, fullPromptTokens))
            let computed = UInt64(min(max(0, computedPromptTokens), max(0, fullPromptTokens)))
            let fullDelta = full > request.recordedFullPromptTokens
                ? full - request.recordedFullPromptTokens
                : 0
            let computedDelta = computed > request.recordedComputedPromptTokens
                ? computed - request.recordedComputedPromptTokens
                : 0
            state.fullPromptTokensTotal &+= fullDelta
            state.computedPromptTokensTotal &+= computedDelta
            if computedDelta > 0 {
                state.promptWindow.record(
                    computedDelta,
                    at: receivedAt,
                    window: Self.rollingWindowSeconds
                )
            }
            request.recordedFullPromptTokens = max(request.recordedFullPromptTokens, full)
            request.recordedComputedPromptTokens = max(request.recordedComputedPromptTokens, computed)
            state.requests[token] = request
        }
    }

    public func outputToken(_ token: AFMInferenceRequestToken, at timestamp: Double) {
        let receivedAt = now()
        state.withLock { state in
            guard var request = state.requests[token] else { return }
            if request.firstTokenAt == nil { request.firstTokenAt = timestamp }
            if let previousTokenAt = request.previousTokenAt {
                state.interTokenLatency.observe(max(0, timestamp - previousTokenAt))
            }
            request.previousTokenAt = timestamp
            request.recordedGeneratedTokens &+= 1
            state.generatedTokensTotal &+= 1
            state.generationWindow.record(1, at: receivedAt, window: Self.rollingWindowSeconds)
            state.requests[token] = request
        }
    }

    public func prefixCacheObserved(queriedTokens: Int, hitTokens: Int) {
        guard queriedTokens > 0 else { return }
        state.withLock { state in
            if hitTokens > 0 {
                state.legacyCacheHitsTotal &+= 1
            } else {
                state.legacyCacheMissesTotal &+= 1
            }
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
            let computed = UInt64(
                min(max(0, observation.computedPromptTokens), max(0, observation.fullPromptTokens))
            )
            let generated = UInt64(max(0, observation.generatedTokens))
            let remainingFull = full > request.recordedFullPromptTokens
                ? full - request.recordedFullPromptTokens
                : 0
            let remainingComputed = computed > request.recordedComputedPromptTokens
                ? computed - request.recordedComputedPromptTokens
                : 0
            let remainingGenerated = generated > request.recordedGeneratedTokens
                ? generated - request.recordedGeneratedTokens
                : 0
            state.fullPromptTokensTotal &+= remainingFull
            state.computedPromptTokensTotal &+= remainingComputed
            state.generatedTokensTotal &+= remainingGenerated
            state.terminalRequestsTotal &+= 1
            state.terminalCounts[observation.reason.rawValue, default: 0] &+= 1
            if remainingComputed > 0 {
                state.promptWindow.record(
                    remainingComputed,
                    at: timestamp,
                    window: Self.rollingWindowSeconds
                )
            }
            if remainingGenerated > 0 {
                state.generationWindow.record(
                    remainingGenerated,
                    at: timestamp,
                    window: Self.rollingWindowSeconds
                )
            }
            state.terminalWindow.record(
                1,
                at: timestamp,
                window: Self.rollingWindowSeconds
            )
            Self.observeLatency(
                state: &state,
                acceptedAt: request.acceptedAt,
                startedAt: request.startedAt,
                firstTokenAt: request.firstTokenAt,
                completedAt: observation.completedAt,
                fullPromptTokens: observation.fullPromptTokens,
                computedPromptTokens: observation.computedPromptTokens,
                generatedTokens: observation.generatedTokens,
                maximumOutputTokens: observation.maximumOutputTokens,
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
        let receivedAt = now()
        return state.withLock { state in
            guard let request = state.requests.removeValue(forKey: token) else { return false }
            state.terminalRequestsTotal &+= 1
            let finishReason: AFMInferenceFinishReason = reason == .cancelled ? .abort : .error
            state.terminalCounts[finishReason.rawValue, default: 0] &+= 1
            state.failureCounts[reason.rawValue, default: 0] &+= 1
            state.terminalWindow.record(
                1,
                at: receivedAt,
                window: Self.rollingWindowSeconds
            )
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
            let promptRate = state.promptWindow.rate(
                at: monotonicNow,
                window: Self.rollingWindowSeconds
            )
            let generationRate = state.generationWindow.rate(
                at: monotonicNow,
                window: Self.rollingWindowSeconds
            )
            let requestRate = state.terminalWindow.rate(
                at: monotonicNow,
                window: Self.rollingWindowSeconds
            )
            let failures = state.failureCounts.merging(state.ingressRejections) { $0 &+ $1 }
            return AFMInferenceMetricsSnapshot(
                timestampMilliseconds: Int64(wallNow * 1_000),
                processStartEpochSeconds: state.processStartEpochSeconds,
                modelName: state.modelName,
                maximumConcurrentRequests: state.maximumConcurrentRequests,
                maximumContextTokens: state.maximumContextTokens,
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
                maximumGeneratedTokens: state.maximumGeneratedTokens.snapshot,
                maximumOutputTokens: state.maximumOutputTokens.snapshot,
                samplingN: state.samplingN.snapshot,
                samplingBestOf: state.samplingBestOf.snapshot
            )
        }
    }
}
