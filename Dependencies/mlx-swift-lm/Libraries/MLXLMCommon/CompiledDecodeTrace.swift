// Marks when the current thread is building an MLX `compile` trace for
// compiled decode. Model forward paths that use mid-graph `eval` as a
// scheduling/materialization aid (e.g. Gemma4 per-layer-input projection)
// must skip those evals while tracing: `eval` during a compile transform is
// a fatal error in MLX, and the traced graph materializes shared
// subexpressions once anyway.
//
// A plain static is sufficient: the compile trace executes the closure
// synchronously on the calling thread, and compiled replays do not re-run
// the Swift closure body.

import Foundation

public enum CompiledDecodeTrace {
    @TaskLocal private static var taskLocalActive = false

    nonisolated(unsafe) private static var threadActive: Bool {
        get { (Thread.current.threadDictionary["vmlx.compiledDecodeTrace"] as? Bool) ?? false }
        set { Thread.current.threadDictionary["vmlx.compiledDecodeTrace"] = newValue }
    }

    public static var isActive: Bool { threadActive }

    public static func withActive<T>(_ body: () throws -> T) rethrows -> T {
        let previous = threadActive
        threadActive = true
        defer { threadActive = previous }
        return try body()
    }
}

/// Process-level tied-head quantization policy, set by the host (Osaurus)
/// from `VMLXServerPerformanceSettings.tiedHeadCodec` before model load.
/// Same host-set-static pattern as `NativeMTPActivation` /
/// `JangPressActivation`. The loader applies it only to bundles that are
/// themselves quantized and whose tied head ships without quantization
/// sidecars; `VMLX_QUANT_TIED_HEAD_BITS` env remains a bench override.
public enum TiedHeadQuantizationPolicy {
    public struct Quantization: Sendable, Equatable {
        public var bits: Int
        public var groupSize: Int
        public init(bits: Int, groupSize: Int = 64) {
            self.bits = bits
            self.groupSize = groupSize
        }
    }

    private static let lock = NSLock()
    nonisolated(unsafe) private static var _current: Quantization?

    public static var current: Quantization? {
        get { lock.withLock { _current } }
        set { lock.withLock { _current = newValue } }
    }
}
