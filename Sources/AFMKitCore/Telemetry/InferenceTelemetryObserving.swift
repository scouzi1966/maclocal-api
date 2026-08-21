import Foundation

public struct AFMInferenceRequestToken: Hashable, Sendable {
    public let rawValue: UUID

    @_spi(AFMKitTelemetry)
    public init(rawValue: UUID = UUID()) {
        self.rawValue = rawValue
    }
}

public enum AFMInferenceFinishReason: String, CaseIterable, Codable, Hashable, Sendable {
    case stop
    case length
    case abort
    case error
    case repetition
}

public enum AFMInferenceFailureReason: String, CaseIterable, Codable, Hashable, Sendable {
    case cancelled
    case inference
    case `internal`
}

public struct AFMInferenceProviderState: Hashable, Sendable {
    public var runningRequests: Int
    public var waitingRequests: Int
    public var activeLogicalCachePositions: Int
    public var logicalCacheCapacity: Int
    public var memoryCacheUsage: Double?
    public var prefixCacheFill: Double?

    public init(
        runningRequests: Int,
        waitingRequests: Int,
        activeLogicalCachePositions: Int = 0,
        logicalCacheCapacity: Int = 0,
        memoryCacheUsage: Double? = nil,
        prefixCacheFill: Double? = nil
    ) {
        self.runningRequests = runningRequests
        self.waitingRequests = waitingRequests
        self.activeLogicalCachePositions = activeLogicalCachePositions
        self.logicalCacheCapacity = logicalCacheCapacity
        self.memoryCacheUsage = memoryCacheUsage
        self.prefixCacheFill = prefixCacheFill
    }
}

public struct AFMInferenceRequestFinishObservation: Hashable, Sendable {
    public var reason: AFMInferenceFinishReason
    public var completedAt: Double
    public var fullPromptTokens: Int
    public var computedPromptTokens: Int
    public var generatedTokens: Int
    public var maximumOutputTokens: Int?
    public var samplingN: Int
    public var samplingBestOf: Int

    public init(
        reason: AFMInferenceFinishReason,
        completedAt: Double,
        fullPromptTokens: Int,
        computedPromptTokens: Int,
        generatedTokens: Int,
        maximumOutputTokens: Int? = nil,
        samplingN: Int = 1,
        samplingBestOf: Int = 1
    ) {
        self.reason = reason
        self.completedAt = completedAt
        self.fullPromptTokens = fullPromptTokens
        self.computedPromptTokens = computedPromptTokens
        self.generatedTokens = generatedTokens
        self.maximumOutputTokens = maximumOutputTokens
        self.samplingN = samplingN
        self.samplingBestOf = samplingBestOf
    }
}

/// Synchronous provider telemetry. Implementations must not call provider code or suspend.
public protocol AFMInferenceTelemetryObserving: Sendable {
    func requestAccepted(at timestamp: Double) -> AFMInferenceRequestToken
    func requestStarted(_ token: AFMInferenceRequestToken, at timestamp: Double)
    /// Records cumulative prompt progress for a request. Repeated calls must
    /// provide totals observed so far; observers account only for positive deltas.
    func promptTokensProcessed(
        _ token: AFMInferenceRequestToken,
        fullPromptTokens: Int,
        computedPromptTokens: Int,
        at timestamp: Double
    )
    func outputToken(_ token: AFMInferenceRequestToken, at timestamp: Double)
    func prefixCacheObserved(queriedTokens: Int, hitTokens: Int)
    func speculativeRound(draftTokens: Int, acceptedTokens: Int)
    func preemptionObserved()
    func updateProviderState(_ state: AFMInferenceProviderState)

    @discardableResult
    func requestFinished(
        _ token: AFMInferenceRequestToken,
        observation: AFMInferenceRequestFinishObservation
    ) -> Bool

    @discardableResult
    func requestFailed(
        _ token: AFMInferenceRequestToken,
        reason: AFMInferenceFailureReason,
        at timestamp: Double
    ) -> Bool
}

public extension AFMInferenceTelemetryObserving {
    func promptTokensProcessed(
        _ token: AFMInferenceRequestToken,
        fullPromptTokens: Int,
        computedPromptTokens: Int,
        at timestamp: Double
    ) {}
}

public struct AFMNoopInferenceTelemetryObserver: AFMInferenceTelemetryObserving {
    public init() {}

    public func requestAccepted(at timestamp: Double) -> AFMInferenceRequestToken {
        AFMInferenceRequestToken()
    }

    public func requestStarted(_ token: AFMInferenceRequestToken, at timestamp: Double) {}
    public func promptTokensProcessed(
        _ token: AFMInferenceRequestToken,
        fullPromptTokens: Int,
        computedPromptTokens: Int,
        at timestamp: Double
    ) {}
    public func outputToken(_ token: AFMInferenceRequestToken, at timestamp: Double) {}
    public func prefixCacheObserved(queriedTokens: Int, hitTokens: Int) {}
    public func speculativeRound(draftTokens: Int, acceptedTokens: Int) {}
    public func preemptionObserved() {}
    public func updateProviderState(_ state: AFMInferenceProviderState) {}

    public func requestFinished(
        _ token: AFMInferenceRequestToken,
        observation: AFMInferenceRequestFinishObservation
    ) -> Bool { true }

    public func requestFailed(
        _ token: AFMInferenceRequestToken,
        reason: AFMInferenceFailureReason,
        at timestamp: Double
    ) -> Bool { true }
}

/// A default-provider telemetry endpoint that can be attached by a host after
/// provider construction without changing explicitly injected observers.
public final class AFMInferenceTelemetryRelay:
    AFMInferenceTelemetryObserving,
    @unchecked Sendable
{
    private let lock = NSLock()
    private var target: any AFMInferenceTelemetryObserving

    public init(
        target: any AFMInferenceTelemetryObserving = AFMNoopInferenceTelemetryObserver()
    ) {
        self.target = target
    }

    public func connect(to target: any AFMInferenceTelemetryObserving) {
        lock.withLock { self.target = target }
    }

    private func observer() -> any AFMInferenceTelemetryObserving {
        lock.withLock { target }
    }

    public func requestAccepted(at timestamp: Double) -> AFMInferenceRequestToken {
        observer().requestAccepted(at: timestamp)
    }

    public func requestStarted(_ token: AFMInferenceRequestToken, at timestamp: Double) {
        observer().requestStarted(token, at: timestamp)
    }

    public func promptTokensProcessed(
        _ token: AFMInferenceRequestToken,
        fullPromptTokens: Int,
        computedPromptTokens: Int,
        at timestamp: Double
    ) {
        observer().promptTokensProcessed(
            token,
            fullPromptTokens: fullPromptTokens,
            computedPromptTokens: computedPromptTokens,
            at: timestamp
        )
    }

    public func outputToken(_ token: AFMInferenceRequestToken, at timestamp: Double) {
        observer().outputToken(token, at: timestamp)
    }

    public func prefixCacheObserved(queriedTokens: Int, hitTokens: Int) {
        observer().prefixCacheObserved(queriedTokens: queriedTokens, hitTokens: hitTokens)
    }

    public func speculativeRound(draftTokens: Int, acceptedTokens: Int) {
        observer().speculativeRound(draftTokens: draftTokens, acceptedTokens: acceptedTokens)
    }

    public func preemptionObserved() {
        observer().preemptionObserved()
    }

    public func updateProviderState(_ state: AFMInferenceProviderState) {
        observer().updateProviderState(state)
    }

    public func requestFinished(
        _ token: AFMInferenceRequestToken,
        observation: AFMInferenceRequestFinishObservation
    ) -> Bool {
        observer().requestFinished(token, observation: observation)
    }

    public func requestFailed(
        _ token: AFMInferenceRequestToken,
        reason: AFMInferenceFailureReason,
        at timestamp: Double
    ) -> Bool {
        observer().requestFailed(token, reason: reason, at: timestamp)
    }
}

/// Optional capability used by hosts to connect default provider telemetry.
public protocol AFMInferenceTelemetryConnecting: Sendable {
    func connectInferenceTelemetry(to observer: any AFMInferenceTelemetryObserving)
}
