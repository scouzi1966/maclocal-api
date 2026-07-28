import Foundation
import Metal
import MLX
import Cmlx

public struct AFMMLXRuntimeMemoryUsage: Equatable, Sendable {
    public let activeMemoryGB: Double

    public init(activeMemoryGB: Double) {
        self.activeMemoryGB = activeMemoryGB
    }

    public static var current: AFMMLXRuntimeMemoryUsage {
        AFMMLXRuntimeMemoryUsage(
            activeMemoryGB: Double(MLX.Memory.activeMemory) / Double(AFMMLXRuntimeMemoryController.bytesPerGB)
        )
    }
}

public struct AFMMLXRuntimeMemoryDefaults: Equatable, Sendable {
    public let compileEnabled: Bool?
    public let cacheLimitBytes: Int
    public let wiredLimitBytes: Int
    public let previousWiredLimitBytes: Int
    public let totalMemoryGB: UInt64
    public let maxRecommendedWorkingSetBytes: Int

    public init(
        compileEnabled: Bool?,
        cacheLimitBytes: Int,
        wiredLimitBytes: Int,
        previousWiredLimitBytes: Int,
        totalMemoryGB: UInt64,
        maxRecommendedWorkingSetBytes: Int
    ) {
        self.compileEnabled = compileEnabled
        self.cacheLimitBytes = cacheLimitBytes
        self.wiredLimitBytes = wiredLimitBytes
        self.previousWiredLimitBytes = previousWiredLimitBytes
        self.totalMemoryGB = totalMemoryGB
        self.maxRecommendedWorkingSetBytes = maxRecommendedWorkingSetBytes
    }
}

public enum AFMMLXRuntimeMemoryController {
    public nonisolated static let bytesPerMB = 1024 * 1024
    public nonisolated static let bytesPerGB = 1024 * 1024 * 1024
    public nonisolated static let defaultWiredLimitPercent = 90

    public static var hasMetalDevice: Bool {
        !MTLCopyAllDevices().isEmpty
    }

    /// Apple Silicon shares memory between CPU and GPU, so MLX can use a
    /// larger cache on machines with more unified memory.
    public static func optimalGPUCacheLimitBytes(
        physicalMemoryBytes: UInt64 = ProcessInfo.processInfo.physicalMemory
    ) -> Int {
        optimalGPUCacheLimitMB(physicalMemoryBytes: physicalMemoryBytes) * bytesPerMB
    }

    public static func optimalGPUCacheLimitMB(
        physicalMemoryBytes: UInt64 = ProcessInfo.processInfo.physicalMemory
    ) -> Int {
        let totalMemoryGB = physicalMemoryBytes / UInt64(bytesPerGB)
        switch totalMemoryGB {
        case 0..<12:
            return 128
        case 12..<24:
            return 256
        case 24..<48:
            return 512
        default:
            return 1024
        }
    }

    public static func wiredLimitBytes(
        maxRecommendedWorkingSetSize: Int,
        percent: Int = defaultWiredLimitPercent
    ) -> Int {
        Int(Double(maxRecommendedWorkingSetSize) * Double(percent) / 100.0)
    }

    public static func applyDefaults(
        compileEnabled: Bool?,
        cacheLimitBytes: Int? = nil,
        wiredLimitPercent: Int = defaultWiredLimitPercent
    ) -> AFMMLXRuntimeMemoryDefaults {
        if let compileEnabled {
            MLX.compile(enable: compileEnabled)
        }

        let resolvedCacheLimitBytes = cacheLimitBytes ?? optimalGPUCacheLimitBytes()
        MLX.Memory.cacheLimit = resolvedCacheLimitBytes

        let maxWorkingSet = Int(GPU.deviceInfo().maxRecommendedWorkingSetSize)
        let resolvedWiredLimitBytes = wiredLimitBytes(
            maxRecommendedWorkingSetSize: maxWorkingSet,
            percent: wiredLimitPercent
        )
        var previousLimit: size_t = 0
        mlx_set_wired_limit(&previousLimit, size_t(resolvedWiredLimitBytes))

        return AFMMLXRuntimeMemoryDefaults(
            compileEnabled: compileEnabled,
            cacheLimitBytes: resolvedCacheLimitBytes,
            wiredLimitBytes: resolvedWiredLimitBytes,
            previousWiredLimitBytes: Int(previousLimit),
            totalMemoryGB: ProcessInfo.processInfo.physicalMemory / UInt64(bytesPerGB),
            maxRecommendedWorkingSetBytes: maxWorkingSet
        )
    }

    public static func clearCache() {
        MLX.Memory.clearCache()
    }

    public static func applyBenchmarkSettings(
        enableCompile: Bool,
        setCacheLimit: Bool,
        cacheLimitMB: Int,
        setWiredLimit: Bool,
        wiredLimitPercent: Int,
        logger: (String) -> Void
    ) {
        MLX.compile(enable: enableCompile)
        logger("  MLX.compile: \(enableCompile)")

        if setCacheLimit {
            MLX.Memory.cacheLimit = cacheLimitMB * bytesPerMB
            logger("  cacheLimit: \(cacheLimitMB) MB")
        } else {
            logger("  cacheLimit: NOT SET (MLX default)")
        }

        if setWiredLimit {
            let maxWorkingSet = Int(GPU.deviceInfo().maxRecommendedWorkingSetSize)
            let wiredLimit = wiredLimitBytes(
                maxRecommendedWorkingSetSize: maxWorkingSet,
                percent: wiredLimitPercent
            )
            var previousLimit: size_t = 0
            mlx_set_wired_limit(&previousLimit, size_t(wiredLimit))
            logger("  wiredLimit: \(wiredLimit / bytesPerMB) MB (\(wiredLimitPercent)% of \(maxWorkingSet / bytesPerMB) MB)")
        } else {
            logger("  wiredLimit: NOT SET (MLX default)")
        }
    }
}
