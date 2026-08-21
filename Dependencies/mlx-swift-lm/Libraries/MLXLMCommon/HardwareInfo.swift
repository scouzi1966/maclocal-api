// Copyright © 2026 Osaurus AI

import Foundation

/// Hardware capability detection for Apple Silicon chip generations.
///
/// This module provides runtime detection of Apple Silicon GPU families to gate
/// Metal JIT features that behave differently across chip generations.
///
/// ## Background
///
/// `compile(shapeless: true)` in MLX wraps closures in a `CompiledFunction` that
/// calls the C++ `Compiled::eval_gpu`. On certain macOS Tahoe GPU drivers (particularly
/// M1/M2 with A14/A15 GPU), this compiled kernel path returns zero results instead
/// of the expected array. This causes an `Index out of range` crash when Swift code
/// accesses `compileState.call([a])[0]` on an empty array.
///
/// - M1/M2: A14/A15 GPU (g7x family) — **Metal JIT bug present**
/// - M3+: A16/A17/A18 GPU (g8x family) — Metal JIT works correctly
///
/// See: MLX issues #3329, #3201, #3256
public enum HardwareInfo {
    /// Returns `true` when `compile(shapeless: true)` is safe to use.
    ///
    /// Currently returns `false` unconditionally. The macOS Tahoe Metal JIT bug
    /// (MLX #3329, #3201, #3256) causes `Compiled::eval_gpu` to return zero results,
    /// crashing at `compiledState.callsToFill[0]` (Index out of range).
    ///
    /// Originally gated by chip generation (M1/M2 = false, M3+ = true), but the
    /// crash was confirmed on M4 Pro (Mac16,x) as well — the bug is a macOS Tahoe
    /// Metal shader compiler issue, not hardware-specific.
    ///
    /// Performance impact of disabling: SIGNIFICANT for real decode, not the
    /// negligible micro-fusion cost this comment previously claimed.
    /// `compile(shapeless:)` fuses the whole single-slot decode step (not just
    /// GELU/SwiGLU/softcap), so gating it off leaves the per-token graph
    /// unfused. Local decode-throughput benchmarks put the compile-ON gain in
    /// the ~+45% to +70% range across gemma-4-e2b and qwen (the exact magnitude
    /// varies with build config and measurement path — RunBench direct decode
    /// vs. the full server path differ substantially in absolute tok/s, so
    /// treat the *ratio*, not any single absolute, as the takeaway). It is
    /// disabled only because it is not yet proven model-switch-safe (see #1173
    /// below), NOT because it is cheap — enabling it (Settings -> Decode
    /// Performance, or VMLX_ENABLE_UNSAFE_COMPILE=1) is the single biggest
    /// local-decode speedup available.
    ///
    /// Re-enable by default once the Metal JIT is fixed AND the #1173 model-switch
    /// corruption is resolved (e.g. clearing the MLX compile cache on model swap).
    public static var isCompiledDecodeSupported: Bool {
        // 2026-05-20: keep MLX compile off by default for host apps. A live
        // Osaurus PR #1173 switch test reproduced process-local decode
        // corruption after loading Qwen3.6 JANG_2K and switching back to the
        // MXFP4 MTP sibling; the same sequence passed when launched with
        // MLX_DISABLE_COMPILE=1. The failure also reproduced on an explicit
        // sampler request that bypassed native MTP, so this gate must cover
        // both single-slot compiled decode and the small shapeless micro-fusion
        // helpers until MLX compile is proven model-switch-safe.
        //
        // Keep an explicit opt-in for controlled benchmarks and diagnostics.
        let env = ProcessInfo.processInfo.environment
        let raw = env["VMLX_ENABLE_UNSAFE_COMPILE"]
            ?? env["MLXPRESS_ENABLE_UNSAFE_COMPILE"]
            ?? ""
        return raw == "1" || raw.lowercased() == "true"
    }

    /// Returns `true` if running on Apple Silicon hardware.
    private static var isAppleSilicon: Bool {
        #if os(macOS) && arch(arm64)
        return true
        #else
        return false
        #endif
    }

    /// Returns the hardware machine identifier string (e.g., "Mac15,10", "Mac14,7").
    ///
    /// Queried via `sysctl hw.machine` at runtime.
    private static var machineIdentifier: String {
        var size: Int = 0
        sysctlbyname("hw.machine", nil, &size, nil, 0)
        guard size > 0 else { return "" }
        var machine = [CChar](repeating: 0, count: size)
        sysctlbyname("hw.machine", &machine, &size, nil, 0)
        let count = machine.firstIndex(of: 0) ?? machine.count
        let bytes = machine.prefix(count).map { UInt8(bitPattern: $0) }
        return String(bytes: bytes, encoding: .utf8) ?? ""
    }
}
