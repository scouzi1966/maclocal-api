import Foundation

/// One immutable cumulative histogram captured at a metrics snapshot boundary.
public struct AFMHistogramSnapshot: Hashable, Sendable {
    public let buckets: [Double]
    public let bucketCounts: [UInt64]
    public let sum: Double
    public let count: UInt64

    public init(
        buckets: [Double],
        bucketCounts: [UInt64],
        sum: Double,
        count: UInt64
    ) {
        self.buckets = buckets
        self.bucketCounts = bucketCounts
        self.sum = sum
        self.count = count
    }

    public init(buckets: [Double]) {
        self.init(
            buckets: buckets,
            bucketCounts: Array(repeating: 0, count: buckets.count + 1),
            sum: 0,
            count: 0
        )
    }
}

/// A renderer-neutral named cumulative count with bounded producer-defined keys.
public struct AFMNamedCount: Hashable, Sendable {
    public let name: String
    public let count: UInt64

    public init(name: String, count: UInt64) {
        self.name = name
        self.count = count
    }
}

/// A renderer-neutral named integer gauge.
public struct AFMNamedIntegerGauge: Hashable, Sendable {
    public let name: String
    public let value: Int

    public init(name: String, value: Int) {
        self.name = name
        self.value = value
    }
}

/// A renderer-neutral named floating-point gauge.
public struct AFMNamedDoubleGauge: Hashable, Sendable {
    public let name: String
    public let value: Double

    public init(name: String, value: Double) {
        self.name = name
        self.value = value
    }
}

/// One immutable process-level inference telemetry snapshot.
public struct AFMInferenceMetricsSnapshot: Hashable, Sendable {
    public let timestampMilliseconds: Int64
    public let processStartEpochSeconds: Double
    public let modelName: String
    public let maximumConcurrentRequests: Int
    public let maximumContextTokens: Int

    public let runningRequests: Int
    public let waitingRequests: Int
    public let peakRunningRequests: Int
    public let logicalCacheUsage: Double
    public let memoryCacheUsage: Double?
    public let prefixCacheFill: Double?

    public let generatedTokensTotal: UInt64
    public let fullPromptTokensTotal: UInt64
    public let computedPromptTokensTotal: UInt64
    public let acceptedRequestsTotal: UInt64
    public let terminalRequestsTotal: UInt64
    public let preemptionsTotal: UInt64
    public let prefixCacheQueriesTotal: UInt64
    public let prefixCacheHitsTotal: UInt64
    public let speculativeDraftRoundsTotal: UInt64
    public let speculativeDraftTokensTotal: UInt64
    public let speculativeAcceptedTokensTotal: UInt64

    public let terminalCounts: [AFMNamedCount]
    public let failureCounts: [AFMNamedCount]
    public let supplementalCounts: [AFMNamedCount]
    public let supplementalIntegerGauges: [AFMNamedIntegerGauge]
    public let supplementalDoubleGauges: [AFMNamedDoubleGauge]

    public let endToEndLatency: AFMHistogramSnapshot
    public let queueLatency: AFMHistogramSnapshot
    public let inferenceLatency: AFMHistogramSnapshot
    public let prefillLatency: AFMHistogramSnapshot
    public let decodeLatency: AFMHistogramSnapshot
    public let timeToFirstToken: AFMHistogramSnapshot
    public let timePerOutputToken: AFMHistogramSnapshot
    public let interTokenLatency: AFMHistogramSnapshot
    public let fullPromptTokens: AFMHistogramSnapshot
    public let computedPromptTokens: AFMHistogramSnapshot
    public let generatedTokens: AFMHistogramSnapshot
    public let maximumGeneratedTokens: AFMHistogramSnapshot
    public let maximumOutputTokens: AFMHistogramSnapshot
    public let samplingN: AFMHistogramSnapshot
    public let samplingBestOf: AFMHistogramSnapshot

    public init(
        timestampMilliseconds: Int64,
        processStartEpochSeconds: Double,
        modelName: String = "",
        maximumConcurrentRequests: Int = 0,
        maximumContextTokens: Int = 0,
        runningRequests: Int = 0,
        waitingRequests: Int = 0,
        peakRunningRequests: Int = 0,
        logicalCacheUsage: Double = 0,
        memoryCacheUsage: Double? = nil,
        prefixCacheFill: Double? = nil,
        generatedTokensTotal: UInt64 = 0,
        fullPromptTokensTotal: UInt64 = 0,
        computedPromptTokensTotal: UInt64 = 0,
        acceptedRequestsTotal: UInt64 = 0,
        terminalRequestsTotal: UInt64 = 0,
        preemptionsTotal: UInt64 = 0,
        prefixCacheQueriesTotal: UInt64 = 0,
        prefixCacheHitsTotal: UInt64 = 0,
        speculativeDraftRoundsTotal: UInt64 = 0,
        speculativeDraftTokensTotal: UInt64 = 0,
        speculativeAcceptedTokensTotal: UInt64 = 0,
        terminalCounts: [AFMNamedCount] = [],
        failureCounts: [AFMNamedCount] = [],
        supplementalCounts: [AFMNamedCount] = [],
        supplementalIntegerGauges: [AFMNamedIntegerGauge] = [],
        supplementalDoubleGauges: [AFMNamedDoubleGauge] = [],
        endToEndLatency: AFMHistogramSnapshot,
        queueLatency: AFMHistogramSnapshot,
        inferenceLatency: AFMHistogramSnapshot,
        prefillLatency: AFMHistogramSnapshot,
        decodeLatency: AFMHistogramSnapshot,
        timeToFirstToken: AFMHistogramSnapshot,
        timePerOutputToken: AFMHistogramSnapshot,
        interTokenLatency: AFMHistogramSnapshot,
        fullPromptTokens: AFMHistogramSnapshot,
        computedPromptTokens: AFMHistogramSnapshot,
        generatedTokens: AFMHistogramSnapshot,
        maximumGeneratedTokens: AFMHistogramSnapshot? = nil,
        maximumOutputTokens: AFMHistogramSnapshot? = nil,
        samplingN: AFMHistogramSnapshot,
        samplingBestOf: AFMHistogramSnapshot
    ) {
        self.timestampMilliseconds = timestampMilliseconds
        self.processStartEpochSeconds = processStartEpochSeconds
        self.modelName = modelName
        self.maximumConcurrentRequests = maximumConcurrentRequests
        self.maximumContextTokens = maximumContextTokens
        self.runningRequests = runningRequests
        self.waitingRequests = waitingRequests
        self.peakRunningRequests = peakRunningRequests
        self.logicalCacheUsage = logicalCacheUsage
        self.memoryCacheUsage = memoryCacheUsage
        self.prefixCacheFill = prefixCacheFill
        self.generatedTokensTotal = generatedTokensTotal
        self.fullPromptTokensTotal = fullPromptTokensTotal
        self.computedPromptTokensTotal = computedPromptTokensTotal
        self.acceptedRequestsTotal = acceptedRequestsTotal
        self.terminalRequestsTotal = terminalRequestsTotal
        self.preemptionsTotal = preemptionsTotal
        self.prefixCacheQueriesTotal = prefixCacheQueriesTotal
        self.prefixCacheHitsTotal = prefixCacheHitsTotal
        self.speculativeDraftRoundsTotal = speculativeDraftRoundsTotal
        self.speculativeDraftTokensTotal = speculativeDraftTokensTotal
        self.speculativeAcceptedTokensTotal = speculativeAcceptedTokensTotal
        self.terminalCounts = terminalCounts
        self.failureCounts = failureCounts
        self.supplementalCounts = supplementalCounts
        self.supplementalIntegerGauges = supplementalIntegerGauges
        self.supplementalDoubleGauges = supplementalDoubleGauges
        self.endToEndLatency = endToEndLatency
        self.queueLatency = queueLatency
        self.inferenceLatency = inferenceLatency
        self.prefillLatency = prefillLatency
        self.decodeLatency = decodeLatency
        self.timeToFirstToken = timeToFirstToken
        self.timePerOutputToken = timePerOutputToken
        self.interTokenLatency = interTokenLatency
        self.fullPromptTokens = fullPromptTokens
        self.computedPromptTokens = computedPromptTokens
        self.generatedTokens = generatedTokens
        self.maximumGeneratedTokens = maximumGeneratedTokens ?? generatedTokens
        self.maximumOutputTokens = maximumOutputTokens ?? AFMHistogramSnapshot(
            buckets: generatedTokens.buckets
        )
        self.samplingN = samplingN
        self.samplingBestOf = samplingBestOf
    }
}

public protocol AFMInferenceMetricsSnapshotSource: Sendable {
    func metricsSnapshot() -> AFMInferenceMetricsSnapshot
}
