import Darwin
import Foundation

public struct AFMMLXMetalSchedulingLimits: Equatable, Sendable {
    public let maxOperationsPerBuffer: Int
    public let maxMegabytesPerBuffer: Int

    public init(maxOperationsPerBuffer: Int, maxMegabytesPerBuffer: Int) {
        self.maxOperationsPerBuffer = maxOperationsPerBuffer
        self.maxMegabytesPerBuffer = maxMegabytesPerBuffer
    }
}

public enum AFMMLXMetalSchedulingPolicy {
    public static let operationsEnvironmentKey = "MLX_MAX_OPS_PER_BUFFER"
    public static let megabytesEnvironmentKey = "MLX_MAX_MB_PER_BUFFER"

    /// DeepSeek V4 repeatedly references its large expert banks while building
    /// each decode graph. MLX's conservative Ultra defaults split that work
    /// into more command buffers than this architecture needs.
    public static func recommendedLimits(
        canonicalModelType: String,
        processorBrand: String,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> AFMMLXMetalSchedulingLimits? {
        guard environment[operationsEnvironmentKey] == nil,
              environment[megabytesEnvironmentKey] == nil,
              canonicalModelType == "deepseek_v4",
              processorBrand.localizedCaseInsensitiveContains("Ultra")
        else {
            return nil
        }

        return AFMMLXMetalSchedulingLimits(
            maxOperationsPerBuffer: 200,
            maxMegabytesPerBuffer: 400)
    }

    @discardableResult
    public static func applyIfRecommended(
        canonicalModelType: String,
        processorBrand: String = currentProcessorBrand(),
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> AFMMLXMetalSchedulingLimits? {
        guard let limits = recommendedLimits(
            canonicalModelType: canonicalModelType,
            processorBrand: processorBrand,
            environment: environment)
        else {
            return nil
        }

        setenv(operationsEnvironmentKey, String(limits.maxOperationsPerBuffer), 0)
        setenv(megabytesEnvironmentKey, String(limits.maxMegabytesPerBuffer), 0)
        return limits
    }

    public static func currentProcessorBrand() -> String {
        var size = 0
        guard sysctlbyname("machdep.cpu.brand_string", nil, &size, nil, 0) == 0,
              size > 1
        else {
            return ""
        }

        var bytes = [CChar](repeating: 0, count: size)
        guard sysctlbyname("machdep.cpu.brand_string", &bytes, &size, nil, 0) == 0 else {
            return ""
        }
        let content = bytes.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }
        return String(decoding: content, as: UTF8.self)
    }
}
