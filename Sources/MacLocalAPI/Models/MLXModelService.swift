import Foundation
import CoreImage
import os
import MLX
import Cmlx
import MLXLLM
import MLXVLM
import MLXLMCommon
import Tokenizers
import Hub
import HuggingFace

/// Resolved log probability entry with token strings (ready for API response).
struct ResolvedLogprob: Sendable {
    let token: String
    let tokenId: Int
    let logprob: Float
    let topTokens: [(token: String, tokenId: Int, logprob: Float)]
}

/// A chunk of streaming output, optionally carrying per-token log probabilities or tool calls.
struct StreamChunk: Sendable {
    let text: String
    let logprobs: [ResolvedLogprob]?
    let toolCalls: [ResponseToolCall]?
    let toolCallDeltas: [StreamDeltaToolCall]?
    let promptTokens: Int?
    let completionTokens: Int?
    let cachedTokens: Int?
    let promptTime: Double?
    let generateTime: Double?
    let stoppedBySequence: Bool?

    init(text: String, logprobs: [ResolvedLogprob]? = nil, toolCalls: [ResponseToolCall]? = nil, toolCallDeltas: [StreamDeltaToolCall]? = nil, promptTokens: Int? = nil, completionTokens: Int? = nil, cachedTokens: Int? = nil, promptTime: Double? = nil, generateTime: Double? = nil, stoppedBySequence: Bool? = nil) {
        self.text = text
        self.logprobs = logprobs
        self.toolCalls = toolCalls
        self.toolCallDeltas = toolCallDeltas
        self.promptTokens = promptTokens
        self.completionTokens = completionTokens
        self.cachedTokens = cachedTokens
        self.promptTime = promptTime
        self.generateTime = generateTime
        self.stoppedBySequence = stoppedBySequence
    }
}

enum MLXLoadStage: String {
    case checkingCache = "checking cache"
    case downloading = "downloading"
    case resuming = "resuming download"
    case loadingModel = "loading model"
    case ready = "ready"
}

enum MLXServiceError: Error, LocalizedError {
    case invalidModel(String)
    case modelNotFoundInCache(String)
    case downloadFailed(String)
    case loadFailed(String)
    case noModelLoaded
    case serviceShuttingDown
    case serverBusy(Int)

    var errorDescription: String? {
        switch self {
        case .invalidModel(let value):
            return "Invalid model identifier: \(value)"
        case .modelNotFoundInCache(let value):
            return "Model not found in cache: \(value)"
        case .downloadFailed(let value):
            return "Failed to download model: \(value)"
        case .loadFailed(let value):
            return "Failed to load model: \(value)"
        case .noModelLoaded:
            return "No MLX model loaded"
        case .serviceShuttingDown:
            return "MLX service is shutting down"
        case .serverBusy(let max):
            return "Server at capacity (\(max) concurrent requests). Please retry shortly."
        }
    }
}

private let debugLogging = ProcessInfo.processInfo.environment["AFM_DEBUG"].map { $0 == "1" } ?? false
private let clearGPUCachePerRequest = ProcessInfo.processInfo.environment["AFM_CLEAR_GPU_CACHE"].map { $0 == "1" } ?? false

private let _tsFormatter: DateFormatter = {
    let f = DateFormatter()
    f.dateFormat = "yyyy-MM-dd HH:mm:ss.SSS"
    f.locale = Locale(identifier: "en_US_POSIX")
    return f
}()
private func ts() -> String { _tsFormatter.string(from: Date()) }
private let vvCyan = "\u{1B}[38;5;87m"
private let vvReset = "\u{1B}[0m"
/// Module-level trace flag, set by MLXModelService.trace when the service is configured.
/// Used by static parsing/conversion methods that can't access instance properties.
private var traceLogging = false
/// Module-level grammar constraint flag — enables cross-parameter dedup in the XML parser.
private var grammarConstraintsActive = false

final class MLXModelService: @unchecked Sendable {
    private struct ConstrainedDecodingSetup {
        let processor: GrammarLogitProcessor
        let mode: String
        let matcherHandle: GrammarMatcherHandle?
    }

    private static let registerModelFactoriesOnce: Void = {
        ModelFactoryRegistry.shared.addTrampoline { LLMModelFactory.shared }
        ModelFactoryRegistry.shared.addTrampoline { VLMModelFactory.shared }
    }()

    private let resolver: MLXCacheResolver
    private let registry = MLXModelRegistry()
    private let stateLock = NSLock()
    private var currentModelID: String?
    private var currentContainer: ModelContainer?
    private var activeOperations: Int = 0
    private var isShuttingDown = false
    private var gpuInitialized = false
    private var radixCache: RadixTreeCache?
    private var currentToolCallFormat: ToolCallFormat?
    var prefillStepSize: Int = 1024
    var toolCallParser: String?
    var fixToolArgs: Bool = false
    var forceVLM: Bool = false
    var kvBits: Int?
    var kvEvictionPolicy: String = "none"  // "none" or "streaming"
    var enablePrefixCaching: Bool = false
    var enableGrammarConstraints: Bool = false { didSet { grammarConstraintsActive = enableGrammarConstraints } }
    var trace: Bool = false { didSet { traceLogging = trace } }
    /// Path to write a Metal GPU trace (.gputrace) — captures the first request only, then resets to nil.
    /// Auto-limits max tokens to 5 to keep trace size manageable.
    var gpuCapturePath: String?
    /// Duration in seconds for xctrace Metal System Trace recording. nil = disabled.
    var gpuTraceDuration: Int?
    /// Print per-request GPU profiling stats (device info, memory snapshots, bandwidth estimates).
    var gpuProfile: Bool = false
    /// Also sample DRAM bandwidth via mactop (adds ~5s). Requires `brew install mactop`.
    var gpuProfileBandwidth: Bool = false
    var defaultChatTemplateKwargs: [String: Any]?
    var cacheProfilePath: String?
    /// Detected think start/end tags from the tokenizer vocabulary (e.g., "<think>"/"</think>").
    /// Set after model load. nil if the model doesn't have think tokens.
    private(set) var thinkStartTag: String?
    private(set) var thinkEndTag: String?
    private var xgrammarService: XGrammarService?
    /// Concurrent generation scheduler (nil = serial mode via container.perform).
    private var scheduler: BatchScheduler?
    /// Maximum concurrent generations (0 = serial mode, 2+ = batch mode).
    var maxConcurrent: Int = 0

    /// Whether the server was started with --concurrent (persistent batch mode).
    private var startedInBatchMode = false

    /// Number of in-flight batch operations (for auto-teardown).
    private let _activeBatchCount = OSAllocatedUnfairLock(initialState: 0)

    /// Whether a promotion is currently in progress (prevents races).
    private var promotionInProgress = false

    /// Scheduled teardown work item (cancelled if new batch arrives).
    private var teardownWorkItem: DispatchWorkItem?

    /// Atomically reserve a concurrent slot. Returns true if reserved (or serial mode).
    func tryReserveSlot() -> Bool { scheduler?.tryReserve() ?? true }
    /// Wait for a concurrent slot with timeout. Returns true if reserved (or serial mode).
    func waitForSlot(timeout: TimeInterval = 30) async -> Bool {
        guard let sched = scheduler else { return true }
        return await sched.waitForSlot(timeout: timeout)
    }
    /// Release a reserved slot (call if request fails before generation starts).
    func releaseSlot() { scheduler?.releaseReservation() }
    init(resolver: MLXCacheResolver) {
        _ = Self.registerModelFactoriesOnce
        self.resolver = resolver
        self.resolver.applyEnvironment()
    }

    /// Configure MLX GPU settings once, before first model load.
    /// Must be called after Metal is available (not during early init).
    private func ensureGPUConfigured() {
        guard !gpuInitialized else { return }
        gpuInitialized = true

        let totalMemoryGB = ProcessInfo.processInfo.physicalMemory / (1024 * 1024 * 1024)
        let cacheMB: Int
        switch totalMemoryGB {
        case 0..<12:  cacheMB = 128
        case 12..<24: cacheMB = 256
        case 24..<48: cacheMB = 512
        default:      cacheMB = 1024
        }
        Memory.cacheLimit = cacheMB * 1024 * 1024

        let maxWorkingSet = GPU.deviceInfo().maxRecommendedWorkingSetSize
        let wiredLimitBytes = Int(Double(maxWorkingSet) * 0.9)
        var previousWired: size_t = 0
        mlx_set_wired_limit(&previousWired, size_t(wiredLimitBytes))

        print("[\(ts())] MLX GPU: cache=\(cacheMB)MB wired=\(wiredLimitBytes / (1024*1024))MB (system \(totalMemoryGB)GB)")
    }

    // MARK: - GPU Capture & Profiling

    /// Begin GPU capture if a capture path is set. Returns true if capture was started.
    /// Consumes the capture path (sets to nil) so only the first request is captured.
    private func beginGPUCaptureIfNeeded() -> Bool {
        guard let path = gpuCapturePath else { return false }
        gpuCapturePath = nil  // capture first request only
        let url = URL(fileURLWithPath: path)
        GPU.startCapture(url: url)
        print("[\(ts())] [GPU-CAPTURE] Started → \(path)")
        return true
    }

    /// Stop GPU capture.
    private func endGPUCapture(path: String) {
        let url = URL(fileURLWithPath: path)
        GPU.stopCapture(url: url)
        print("[\(ts())] [GPU-CAPTURE] Stopped → \(path)")
        print("[\(ts())] [GPU-CAPTURE] Open in Xcode: open \(path)")
    }

    /// Max tokens override when GPU capture is active (keeps trace files small).
    private static let gpuCaptureMaxTokens = 5

    /// Apply token cap for GPU capture mode. Returns capped value.
    private func capMaxTokensForCapture(_ maxTokens: Int) -> Int {
        if gpuCapturePath != nil {
            let capped = Swift.min(maxTokens, Self.gpuCaptureMaxTokens)
            if capped < maxTokens {
                print("[\(ts())] [GPU-CAPTURE] Capping max_tokens: \(maxTokens) → \(capped) (full trace overhead)")
            }
            return capped
        }
        return maxTokens
    }

    /// Active xctrace Process (launched by beginGPUTrace, stopped by endGPUTrace).
    private var xctraceProcess: Process?
    private var xctraceOutputPath: String?

    /// Launch xctrace Metal System Trace in background, attaching to our PID.
    /// Returns true if xctrace was launched.
    private func beginGPUTraceIfNeeded() -> Bool {
        guard let duration = gpuTraceDuration else { return false }
        gpuTraceDuration = nil  // trace first request only
        let pid = ProcessInfo.processInfo.processIdentifier
        let outputPath = "/tmp/afm-metal.trace"
        // Remove existing trace
        try? FileManager.default.removeItem(atPath: outputPath)
        // Prefer custom shader-enabled template if available (has per-kernel names)
        let shaderTemplate = NSString(string: "~/Library/Developer/Xcode/UserData/Instruments/Templates/Metal Shader Profile.tracetemplate").expandingTildeInPath
        let templateArg: String
        if FileManager.default.fileExists(atPath: shaderTemplate) {
            templateArg = shaderTemplate
            print("[\(ts())] [GPU-TRACE] Using Metal Shader Profile template (per-kernel shader names enabled)")
        } else {
            templateArg = "Metal System Trace"
            print("[\(ts())] [GPU-TRACE] Using default Metal System Trace (no per-kernel names)")
            print("[\(ts())] [GPU-TRACE]   For shader names: python3 Scripts/create-shader-template.py")
        }
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/xcrun")
        process.arguments = [
            "xctrace", "record",
            "--template", templateArg,
            "--attach", "\(pid)",
            "--time-limit", "\(duration)s",
            "--output", outputPath
        ]
        // Suppress xctrace stdout/stderr noise
        process.standardOutput = FileHandle.nullDevice
        process.standardError = FileHandle.nullDevice
        do {
            try process.run()
            xctraceProcess = process
            xctraceOutputPath = outputPath
            print("[\(ts())] [GPU-TRACE] Recording for \(duration)s (PID \(pid))")
            print("[\(ts())] [GPU-TRACE] Output: \(outputPath)")
            // Give xctrace time to attach before we start inference
            Thread.sleep(forTimeInterval: 1.5)
            return true
        } catch {
            print("[\(ts())] [GPU-TRACE] Failed to launch xctrace: \(error.localizedDescription)")
            print("[\(ts())] [GPU-TRACE] Make sure Xcode Command Line Tools are installed")
            return false
        }
    }

    /// Stop xctrace recording (sends SIGINT for graceful stop, then waits).
    private func endGPUTrace() {
        guard let process = xctraceProcess else { return }
        let outputPath = xctraceOutputPath ?? "/tmp/afm-metal.trace"
        if process.isRunning {
            process.interrupt()  // SIGINT = graceful stop
            process.waitUntilExit()
        }
        xctraceProcess = nil
        xctraceOutputPath = nil
        let fm = FileManager.default
        if fm.fileExists(atPath: outputPath) {
            print("[\(ts())] [GPU-TRACE] Trace saved: \(outputPath)")
            print("[\(ts())] [GPU-TRACE] Open in Instruments: open \(outputPath)")
            print("[\(ts())] [GPU-TRACE] Look at: Shader Timeline, GPU Counters, Performance Limiters")
        } else {
            print("[\(ts())] [GPU-TRACE] Warning: trace file not found at \(outputPath)")
            print("[\(ts())] [GPU-TRACE] xctrace may require running without sandbox or with elevated privileges")
        }
    }

    // MARK: - Native IOReport GPU Monitor (no mactop, no sudo)

    // IOReport C function declarations (linked via -lIOReport)
    @_silgen_name("IOReportCopyChannelsInGroup")
    private static func IOReportCopyChannelsInGroup(_ group: CFString?, _ subgroup: CFString?, _ a: UInt64, _ b: UInt64, _ c: UInt64) -> CFDictionary?
    @_silgen_name("IOReportMergeChannels")
    private static func IOReportMergeChannels(_ a: CFDictionary, _ b: CFDictionary, _ c: UnsafeMutableRawPointer?)
    @_silgen_name("IOReportCreateSubscription")
    private static func IOReportCreateSubscription(_ a: UnsafeMutableRawPointer?, _ channels: CFMutableDictionary, _ buf: UnsafeMutablePointer<Unmanaged<CFMutableDictionary>?>, _ c: UInt64, _ d: UnsafeMutableRawPointer?) -> UnsafeMutableRawPointer?
    @_silgen_name("IOReportCreateSamples")
    private static func IOReportCreateSamples(_ sub: UnsafeMutableRawPointer, _ channels: CFMutableDictionary?, _ b: UnsafeMutableRawPointer?) -> CFDictionary?
    @_silgen_name("IOReportCreateSamplesDelta")
    private static func IOReportCreateSamplesDelta(_ prev: CFDictionary, _ curr: CFDictionary, _ a: UnsafeMutableRawPointer?) -> CFDictionary?
    @_silgen_name("IOReportChannelGetChannelName")
    private static func IOReportChannelGetChannelName(_ ch: CFDictionary) -> Unmanaged<CFString>?
    @_silgen_name("IOReportSimpleGetIntegerValue")
    private static func IOReportSimpleGetIntegerValue(_ ch: CFDictionary, _ idx: Int32) -> Int64
    @_silgen_name("IOReportStateGetCount")
    private static func IOReportStateGetCount(_ ch: CFDictionary) -> Int32
    @_silgen_name("IOReportStateGetResidency")
    private static func IOReportStateGetResidency(_ ch: CFDictionary, _ idx: Int32) -> Int64
    @_silgen_name("IOReportChannelGetUnitLabel")
    private static func IOReportChannelGetUnitLabel(_ ch: CFDictionary) -> Unmanaged<CFString>?

    /// Convert an IOReport energy value to watts using the channel's unit label.
    /// Channels report in mJ, uJ, nJ, or raw (NULL → assume /1e6).
    private static func energyToWatts(_ value: Int64, unit: String?, dt: Double) -> Double {
        guard dt > 0 else { return 0 }
        let rate = Double(value) / dt
        switch unit {
        case "mJ":  return rate / 1e3
        case "uJ":  return rate / 1e6
        case "nJ":  return rate / 1e9
        default:    return rate / 1e6  // mactop default for NULL unit
        }
    }

    /// Snapshot of GPU metrics from IOReport.
    struct IOReportGPUSample {
        let t: Double  // seconds since profile start
        let gpuPowerW: Double
        let gpuActivePercent: Double
        let gpuFreqMHz: Int
        let dramPowerW: Double
    }

    /// IOReport subscription handle for GPU monitoring.
    private var ioReportSub: UnsafeMutableRawPointer?
    private var ioReportSubChannels: Unmanaged<CFMutableDictionary>?
    private var ioReportPrevSample: CFDictionary?
    private var ioReportPrevTime: CFAbsoluteTime = 0
    private var ioReportGPUSamples: [IOReportGPUSample] = []
    private let ioReportLock = NSLock()
    private var ioReportTimer: DispatchSourceTimer?
    private var ioReportStartTime: CFAbsoluteTime = 0
    private var ioReportActive = false

    /// Initialize IOReport subscription for GPU power + utilization.
    /// Call once before inference; then call sampleGPU() repeatedly.
    private func initIOReportGPU() -> Bool {
        // Cleanup any leftover subscription from a previous profile
        if ioReportSub != nil { _ = stopIOReportGPU() }
        guard let energyCh = Self.IOReportCopyChannelsInGroup("Energy Model" as CFString, nil, 0, 0, 0) else { return false }
        // Merge GPU Stats for frequency/active residency
        if let gpuCh = Self.IOReportCopyChannelsInGroup("GPU Stats" as CFString, nil, 0, 0, 0) {
            Self.IOReportMergeChannels(energyCh, gpuCh, nil)
        }
        let mc = NSMutableDictionary(dictionary: energyCh as NSDictionary) as CFMutableDictionary
        var subRef: Unmanaged<CFMutableDictionary>? = nil
        guard let sub = Self.IOReportCreateSubscription(nil, mc, &subRef, 0, nil) else { return false }
        ioReportSub = sub
        ioReportSubChannels = subRef
        ioReportGPUSamples = []
        // Take initial sample
        ioReportStartTime = CFAbsoluteTimeGetCurrent()
        ioReportPrevSample = Self.IOReportCreateSamples(sub, subRef?.takeUnretainedValue(), nil)
        ioReportPrevTime = CFAbsoluteTimeGetCurrent()
        return ioReportPrevSample != nil
    }

    /// Take a GPU sample (call at ~300ms intervals). Returns nil on error.
    private func sampleIOReportGPU() -> IOReportGPUSample? {
        ioReportLock.lock()
        defer { ioReportLock.unlock() }
        guard let sub = ioReportSub, let prev = ioReportPrevSample else { return nil }
        guard let curr = Self.IOReportCreateSamples(sub, ioReportSubChannels?.takeUnretainedValue(), nil) else { return nil }
        let now = CFAbsoluteTimeGetCurrent()
        let dt = now - ioReportPrevTime
        guard dt > 0.01 else { return nil }
        guard let delta = Self.IOReportCreateSamplesDelta(prev, curr, nil),
              let arr = (delta as NSDictionary)["IOReportChannels"] as? [NSDictionary] else { return nil }

        var gpuPowerW: Double = 0
        var dramPowerW: Double = 0
        var gpuActiveResidency: Int64 = 0
        var gpuTotalResidency: Int64 = 0

        for ch in arr {
            let cfCh = ch as CFDictionary
            let name = Self.IOReportChannelGetChannelName(cfCh)?.takeUnretainedValue() as String? ?? ""
            let unit = Self.IOReportChannelGetUnitLabel(cfCh)?.takeUnretainedValue() as String?
            let val = Self.IOReportSimpleGetIntegerValue(cfCh, 0)
            if name == "GPU Energy" {
                gpuPowerW = Self.energyToWatts(val, unit: unit, dt: dt)
            } else if name.hasPrefix("DRAM") {
                dramPowerW += Self.energyToWatts(val, unit: unit, dt: dt)
            } else if name.hasPrefix("GPU ") {
                let stateCount = Self.IOReportStateGetCount(cfCh)
                if stateCount > 1 {
                    for s in 0..<stateCount {
                        let res = Self.IOReportStateGetResidency(cfCh, s)
                        gpuTotalResidency += res
                        if s > 0 { gpuActiveResidency += res }
                    }
                }
            }
        }
        let gpuActivePct = gpuTotalResidency > 0 ? Double(gpuActiveResidency) / Double(gpuTotalResidency) * 100.0 : 0

        ioReportPrevSample = curr
        ioReportPrevTime = now

        let elapsed = now - ioReportStartTime
        let sample = IOReportGPUSample(t: (elapsed * 10).rounded() / 10, gpuPowerW: gpuPowerW, gpuActivePercent: gpuActivePct, gpuFreqMHz: 0, dramPowerW: dramPowerW)
        ioReportGPUSamples.append(sample)
        return sample
    }

    /// Stop IOReport monitoring and return all collected samples.
    private func stopIOReportGPU() -> [IOReportGPUSample] {
        ioReportSub = nil
        ioReportSubChannels = nil
        ioReportPrevSample = nil
        ioReportLock.lock()
        let samples = ioReportGPUSamples
        ioReportGPUSamples = []
        ioReportLock.unlock()
        return samples
    }

    // MARK: - API Profile (X-AFM-Profile header)

    /// Start GPU profiling for an API request. Call before inference.
    func startAPIProfile() {
        ioReportLock.lock()
        if ioReportActive {
            ioReportLock.unlock()
            return  // Another request is already profiling — skip this one
        }
        ioReportActive = true
        ioReportLock.unlock()

        _ = Self.calibrationOnce
        guard initIOReportGPU() else {
            ioReportLock.lock()
            ioReportActive = false
            ioReportLock.unlock()
            return
        }
        // First sample after 100ms settling, then every 300ms — no blocking sleep
        ioReportTimer = DispatchSource.makeTimerSource(queue: DispatchQueue.global(qos: .utility))
        ioReportTimer?.schedule(deadline: .now() + .milliseconds(100), repeating: .milliseconds(300))
        ioReportTimer?.setEventHandler { [weak self] in
            _ = self?.sampleIOReportGPU()
        }
        ioReportTimer?.resume()
    }

    // DRAM power → bandwidth calibration via GPU memory stress (MLX).
    // Runs a known GPU workload (x + 1 on 1 GiB array) for 1s, measures DRAM power
    // via IOReport, derives GB/s-per-watt from actual GPU→DRAM traffic.
    private static var dramIdlePowerW = 0.3
    private static var dramGBsPerWatt = 10.5  // fallback if calibration fails
    private static var dramCalibrated = false

    /// Thread-safe dispatch_once calibration — runs async on background queue.
    /// Default dramGBsPerWatt (10.5) is used until calibration completes.
    private static let calibrationOnce: Void = {
        DispatchQueue.global(qos: .utility).async {
            calibrateDRAMBandwidthImpl()
        }
    }()

    /// Public entry point — schedules calibration exactly once, returns immediately.
    static func calibrateDRAMBandwidth() {
        _ = calibrationOnce
    }

    /// Calibrate DRAM bandwidth using GPU memory stress via MLX.
    /// Runs ~2s total (0.5s idle + 0.5s warmup + 1s measurement). Call once at startup.
    private static func calibrateDRAMBandwidthImpl() {
        guard let energyCh = IOReportCopyChannelsInGroup("Energy Model" as CFString, nil, 0, 0, 0) else {
            print("[\(ts())] [CALIBRATE] IOReport unavailable — using default \(dramGBsPerWatt) GB/s/W")
            dramCalibrated = true
            return
        }
        let mc = NSMutableDictionary(dictionary: energyCh as NSDictionary) as CFMutableDictionary
        var subRef: Unmanaged<CFMutableDictionary>? = nil
        guard let sub = IOReportCreateSubscription(nil, mc, &subRef, 0, nil) else {
            dramCalibrated = true; return
        }
        let subCh = subRef?.takeUnretainedValue()

        // Step 1: Measure idle DRAM power (0.5s)
        guard let idleS1 = IOReportCreateSamples(sub, subCh, nil) else { dramCalibrated = true; return }
        Thread.sleep(forTimeInterval: 0.5)
        guard let idleS2 = IOReportCreateSamples(sub, subCh, nil) else { dramCalibrated = true; return }
        let idlePower = extractDRAMPower(from: idleS1, to: idleS2, dt: 0.5)
        if idlePower > 0.01 { dramIdlePowerW = idlePower }

        // Step 2: Allocate 1 GiB MLX array on GPU and warm up
        let elemCount = 256 * 1024 * 1024  // 256M float32 = 1 GiB
        let x = MLXArray.ones([elemCount], dtype: .float32)
        MLX.eval(x)
        Stream.gpu.synchronize()
        // Warm up: one full read+write pass
        let w = x + 1
        MLX.eval(w)
        Stream.gpu.synchronize()

        // Step 3: GPU memory stress for 1s — measure DRAM power
        // Each iteration: y = x + 1 → reads 1 GiB, writes 1 GiB = 2 GiB
        guard let loadS1 = IOReportCreateSamples(sub, subCh, nil) else { dramCalibrated = true; return }
        let t0 = CFAbsoluteTimeGetCurrent()
        var totalBytes: Int64 = 0
        let bytesPerIter = Int64(elemCount) * 4 * 2  // float32=4B, read+write
        while CFAbsoluteTimeGetCurrent() - t0 < 1.0 {
            let y = x + 1
            MLX.eval(y)
            Stream.gpu.synchronize()
            totalBytes += bytesPerIter
        }
        let loadDt = CFAbsoluteTimeGetCurrent() - t0
        guard let loadS2 = IOReportCreateSamples(sub, subCh, nil) else { dramCalibrated = true; return }
        let loadPower = extractDRAMPower(from: loadS1, to: loadS2, dt: loadDt)

        // Step 4: Derive calibration constant
        let activePower = loadPower - dramIdlePowerW
        let measuredGBs = Double(totalBytes) / loadDt / 1e9

        if activePower > 0.1 && measuredGBs > 1.0 {
            let candidate = measuredGBs / activePower
            if candidate >= 2.0 && candidate <= 200.0 {
                dramGBsPerWatt = candidate
            } else {
                print("[\(ts())] [CALIBRATE] Out of range (\(String(format: "%.1f", candidate)) GB/s/W) — keeping default \(dramGBsPerWatt)")
            }
            print("[\(ts())] [CALIBRATE] DRAM BW calibrated: \(String(format: "%.1f", dramGBsPerWatt)) GB/s/W (idle: \(String(format: "%.2f", dramIdlePowerW))W, load: \(String(format: "%.2f", loadPower))W, GPU: \(String(format: "%.1f", measuredGBs)) GB/s, \(Int(totalBytes / 1_073_741_824)) GiB in \(String(format: "%.1f", loadDt))s)")
        } else {
            print("[\(ts())] [CALIBRATE] Inconclusive (active=\(String(format: "%.2f", activePower))W) — using default \(dramGBsPerWatt) GB/s/W")
        }
        dramCalibrated = true
    }

    /// Extract DRAM power (watts) from an IOReport sample delta.
    private static func extractDRAMPower(from s1: CFDictionary, to s2: CFDictionary, dt: Double) -> Double {
        guard let delta = IOReportCreateSamplesDelta(s1, s2, nil),
              let arr = (delta as NSDictionary)["IOReportChannels"] as? [NSDictionary] else { return 0 }
        var watts: Double = 0
        for ch in arr {
            let cfCh = ch as CFDictionary
            let name = IOReportChannelGetChannelName(cfCh)?.takeUnretainedValue() as String? ?? ""
            if name.hasPrefix("DRAM") {
                let unit = IOReportChannelGetUnitLabel(cfCh)?.takeUnretainedValue() as String?
                let val = IOReportSimpleGetIntegerValue(cfCh, 0)
                watts += energyToWatts(val, unit: unit, dt: dt)
            }
        }
        return watts
    }

    /// Stop GPU profiling and build an AFMProfile for the API response.
    func stopAPIProfile(promptTokens: Int, completionTokens: Int, promptTime: Double, generateTime: Double) -> AFMProfile {
        ioReportTimer?.cancel()
        ioReportTimer = nil
        _ = sampleIOReportGPU()
        let gpuSamples = stopIOReportGPU()

        ioReportLock.lock()
        ioReportActive = false
        ioReportLock.unlock()

        let profile = buildProfileSummary(gpuSamples: gpuSamples, promptTokens: promptTokens, completionTokens: completionTokens, promptTime: promptTime, generateTime: generateTime)

        // Log to stderr
        if let avg = profile.gpuPowerAvgW, let peak = profile.gpuPowerPeakW {
            var line = "[\(ts())] [AFM-PROFILE] GPU: peak \(String(format: "%.1f", peak))W avg \(String(format: "%.1f", avg))W (\(gpuSamples.count) samples) | \(profile.chip ?? "unknown")"
            if let bw = profile.estBandwidthGbs {
                line += " | est BW: \(String(format: "%.1f", bw)) GB/s"
            }
            print(line)
        }

        return profile
    }

    private func round1(_ v: Double) -> Double { (v * 10).rounded() / 10 }

    /// Shared helper: build an AFMProfile summary from collected GPU samples.
    private func buildProfileSummary(gpuSamples: [IOReportGPUSample], promptTokens: Int, completionTokens: Int, promptTime: Double, generateTime: Double) -> AFMProfile {
        let snap = Memory.snapshot()
        let info = GPU.deviceInfo()
        let memGB = info.memorySize / (1024 * 1024 * 1024)
        let (theoreticalBW, chipName) = Self.estimateTheoreticalBandwidth(architecture: info.architecture, memoryGB: memGB)
        let gib = 1024.0 * 1024.0 * 1024.0
        let weightsBytes = gpuProfileModelWeightBytes > 0 ? Double(gpuProfileModelWeightBytes) : Double(snap.activeMemory)
        let weightsGiB = weightsBytes / gib
        let kvGiB = Double(Swift.max(0, Int(snap.activeMemory) - gpuProfileModelWeightBytes)) / gib
        let peakGiB = Double(snap.peakMemory) / gib

        let peakPower = gpuSamples.map(\.gpuPowerW).max()
        let avgPower = gpuSamples.isEmpty ? nil : gpuSamples.map(\.gpuPowerW).reduce(0, +) / Double(gpuSamples.count)
        let prefillTokS = promptTime > 0 ? Double(promptTokens) / promptTime : nil
        let decodeTokS = generateTime > 0 ? Double(completionTokens) / generateTime : nil

        // Estimate DRAM bandwidth from DRAM power samples (calibrated)
        let dramPowers = gpuSamples.map(\.dramPowerW).filter { $0 > 0.01 }
        let estBandwidth: Double?
        if !dramPowers.isEmpty && Self.dramCalibrated {
            let avgDramPower = dramPowers.reduce(0, +) / Double(dramPowers.count)
            let activePower = Swift.max(0, avgDramPower - Self.dramIdlePowerW)
            estBandwidth = activePower * Self.dramGBsPerWatt
        } else {
            estBandwidth = nil
        }

        return AFMProfile(
            gpuPowerAvgW: avgPower.map { round1($0) },
            gpuPowerPeakW: peakPower.map { round1($0) },
            gpuSamples: gpuSamples.isEmpty ? nil : gpuSamples.count,
            memoryWeightsGiB: round1(weightsGiB),
            memoryKvGiB: round1(kvGiB),
            memoryPeakGiB: round1(peakGiB),
            prefillTokS: prefillTokS.map { round1($0) },
            decodeTokS: decodeTokS.map { round1($0) },
            chip: chipName,
            theoreticalBwGbs: theoreticalBW,
            estBandwidthGbs: estBandwidth.map { round1($0) }
        )
    }

    /// Stop GPU profiling and build an AFMProfileExtended with time-series samples.
    func stopAPIProfileExtended(promptTokens: Int, completionTokens: Int, promptTime: Double, generateTime: Double) -> AFMProfileExtended {
        // Collect raw samples before clearing state
        ioReportTimer?.cancel()
        ioReportTimer = nil
        _ = sampleIOReportGPU()
        let rawSamples = stopIOReportGPU()

        ioReportLock.lock()
        ioReportActive = false
        ioReportLock.unlock()

        let summary = buildProfileSummary(gpuSamples: rawSamples, promptTokens: promptTokens, completionTokens: completionTokens, promptTime: promptTime, generateTime: generateTime)

        // Log to stderr
        if let avg = summary.gpuPowerAvgW, let peak = summary.gpuPowerPeakW {
            var line = "[\(ts())] [AFM-PROFILE] GPU: peak \(String(format: "%.1f", peak))W avg \(String(format: "%.1f", avg))W (\(rawSamples.count) samples) | \(summary.chip ?? "unknown")"
            if let bw = summary.estBandwidthGbs { line += " | est BW: \(String(format: "%.1f", bw)) GB/s" }
            print(line)
        }

        // Convert to API samples
        let apiSamples: [AFMProfileSample] = rawSamples.map { s in
            let activePower = Swift.max(0, s.dramPowerW - Self.dramIdlePowerW)
            let bw = Self.dramCalibrated ? activePower * Self.dramGBsPerWatt : nil
            return AFMProfileSample(
                t: s.t,
                bwGbs: bw.map { round1($0) },
                gpuPct: round1(s.gpuActivePercent),
                gpuPowerW: round1(s.gpuPowerW),
                dramPowerW: round1(s.dramPowerW)
            )
        }

        return AFMProfileExtended(summary: summary, samples: apiSamples)
    }

    // MARK: - mactop Bandwidth Monitor

    /// Collected bandwidth samples from mactop during inference.
    private struct BandwidthSample {
        let readGBs: Double
        let writeGBs: Double
        let combinedGBs: Double
    }

    /// Collect bandwidth samples from mactop synchronously (blocks for ~1s).
    /// mactop doesn't flush pipe/file output on kill, so we must let it run to completion.
    /// Uses --count 3 at 300ms = ~0.9s total wait.
    private func collectBandwidthViaMactop() -> [BandwidthSample] {
        let mactopPath = "/opt/homebrew/bin/mactop"
        guard FileManager.default.fileExists(atPath: mactopPath) else { return [] }
        let pipe = Pipe()
        let process = Process()
        process.executableURL = URL(fileURLWithPath: mactopPath)
        process.arguments = ["--headless", "--format", "json", "-i", "300", "--count", "3"]
        process.standardOutput = pipe
        process.standardError = FileHandle.nullDevice
        do {
            try process.run()
            process.waitUntilExit()  // blocks ~0.9s
            let data = pipe.fileHandleForReading.readDataToEndOfFile()
            guard let text = String(data: data, encoding: .utf8), !text.isEmpty else { return [] }
            return parseMactopOutput(text)
        } catch {
            return []
        }
    }

    /// Parse mactop JSON output (array or line-delimited) into bandwidth samples.
    private func parseMactopOutput(_ text: String) -> [BandwidthSample] {
        var samples: [BandwidthSample] = []
        // Try parsing as a complete JSON array
        if let arrayData = text.data(using: .utf8),
           let array = try? JSONSerialization.jsonObject(with: arrayData) as? [[String: Any]] {
            for obj in array {
                if let sample = extractBandwidth(from: obj) { samples.append(sample) }
            }
            return samples
        }
        // Fallback: line-by-line parsing (strip JSON array delimiters)
        for line in text.split(separator: "\n") {
            var cleaned = line.trimmingCharacters(in: .whitespaces)
            if cleaned.hasPrefix("[") { cleaned = String(cleaned.dropFirst()) }
            if cleaned.hasPrefix(",") { cleaned = String(cleaned.dropFirst()) }
            if cleaned.hasSuffix(",") { cleaned = String(cleaned.dropLast()) }
            if cleaned == "]" || cleaned.isEmpty { continue }
            if let d = cleaned.data(using: .utf8),
               let obj = try? JSONSerialization.jsonObject(with: d) as? [String: Any],
               let sample = extractBandwidth(from: obj) {
                samples.append(sample)
            }
        }
        return samples
    }

    /// Extract bandwidth fields from a single mactop JSON object.
    private func extractBandwidth(from json: [String: Any]) -> BandwidthSample? {
        guard let soc = json["soc_metrics"] as? [String: Any] else { return nil }
        let read = soc["dram_read_bw_gbs"] as? Double ?? 0
        let write = soc["dram_write_bw_gbs"] as? Double ?? 0
        let combined = soc["dram_bw_combined_gbs"] as? Double ?? 0
        guard combined > 0 else { return nil }
        return BandwidthSample(readGBs: read, writeGBs: write, combinedGBs: combined)
    }

    // MARK: - GPU Profile Reporting

    /// Pre-inference active memory snapshot — set by printGPUProfileHeader, read by footer.
    /// This approximates model weight size (before KV cache allocation).
    private var gpuProfileModelWeightBytes: Int = 0

    /// Print GPU profiling header: device info, current memory state.
    /// Records pre-inference active memory as the model weight size estimate.
    private func printGPUProfileHeader() {
        let info = GPU.deviceInfo()
        let snap = Memory.snapshot()
        gpuProfileModelWeightBytes = snap.activeMemory
        GPU.resetPeakMemory()
        // Emit the exact command line used (for reproducibility / HTML report header)
        let args = ProcessInfo.processInfo.arguments
        let cmd = args.joined(separator: " ")
        let envPrefix = ProcessInfo.processInfo.environment["MACAFM_MLX_MODEL_CACHE"].map { "MACAFM_MLX_MODEL_CACHE=\($0) " } ?? ""
        print("[\(ts())] [GPU-PROFILE] ─── Command ───")
        print("[\(ts())] [GPU-PROFILE]   \(envPrefix)\(cmd)")
        print("[\(ts())] [GPU-PROFILE] ─── Device ───")
        print("[\(ts())] [GPU-PROFILE]   Architecture: \(info.architecture)")
        print("[\(ts())] [GPU-PROFILE]   Memory: \(info.memorySize / (1024*1024*1024)) GB")
        print("[\(ts())] [GPU-PROFILE]   Max buffer: \(info.maxBufferSize / (1024*1024)) MB")
        print("[\(ts())] [GPU-PROFILE]   Max working set: \(info.maxRecommendedWorkingSetSize / (1024*1024*1024)) GB")
        print("[\(ts())] [GPU-PROFILE] ─── Pre-inference Memory ───")
        print("[\(ts())] [GPU-PROFILE]   Active (model weights): \(snap.activeMemory / (1024*1024)) MB")
        print("[\(ts())] [GPU-PROFILE]   Cache: \(snap.cacheMemory / (1024*1024)) MB")
        print("[\(ts())] [GPU-PROFILE]   Peak (reset): 0 MB")
        // Calibrate DRAM bandwidth on first use
        _ = Self.calibrationOnce
        // Start native IOReport GPU monitoring (power + utilization, no mactop needed)
        if initIOReportGPU() {
            print("[\(ts())] [GPU-PROFILE]   IOReport GPU monitor: active")
            // Sample GPU every 300ms in background during inference
            ioReportTimer = DispatchSource.makeTimerSource(queue: DispatchQueue.global(qos: .utility))
            ioReportTimer?.schedule(deadline: .now() + .milliseconds(300), repeating: .milliseconds(300))
            ioReportTimer?.setEventHandler { [weak self] in
                _ = self?.sampleIOReportGPU()
            }
            ioReportTimer?.resume()
        }
    }

    /// Estimate theoretical memory bandwidth (GB/s) from Apple Silicon architecture string.
    ///
    /// Architecture strings: applegpu_g13x (M1), g14x (M2), g15x (M3), g16x (M4).
    /// Suffix: p=Pro, g=Max, d=Ultra (dual-die), s=standard.
    private static func estimateTheoreticalBandwidth(architecture: String, memoryGB: Int) -> (bandwidth: Double, chipName: String) {
        let arch = architecture.lowercased()
        // Parse generation from "applegpu_gNNx" pattern
        let generation: Int
        if arch.contains("g16") { generation = 4 }       // M4
        else if arch.contains("g15") { generation = 3 }   // M3
        else if arch.contains("g14") { generation = 2 }   // M2
        else if arch.contains("g13") { generation = 1 }   // M1
        else { generation = 0 }
        // Parse die variant
        let isUltra = arch.hasSuffix("d")
        let isMax = arch.hasSuffix("g")
        let isPro = arch.hasSuffix("p")
        // Also infer from memory: >192GB is almost certainly Ultra
        let likelyUltra = isUltra || memoryGB > 192
        let likelyMax = isMax || (!likelyUltra && memoryGB > 36)

        switch (generation, likelyUltra, likelyMax, isPro) {
        case (4, true, _, _):  return (819.2, "M4 Ultra")   // 2x M4 Max
        case (4, _, true, _):  return (546.0, "M4 Max")
        case (4, _, _, true):  return (273.0, "M4 Pro")
        case (4, _, _, _):     return (120.0, "M4")
        case (3, true, _, _):  return (800.0, "M3 Ultra")
        case (3, _, true, _):  return (400.0, "M3 Max")
        case (3, _, _, true):  return (150.0, "M3 Pro")
        case (3, _, _, _):     return (100.0, "M3")
        case (2, true, _, _):  return (800.0, "M2 Ultra")
        case (2, _, true, _):  return (400.0, "M2 Max")
        case (2, _, _, true):  return (200.0, "M2 Pro")
        case (2, _, _, _):     return (100.0, "M2")
        case (1, true, _, _):  return (800.0, "M1 Ultra")
        case (1, _, true, _):  return (400.0, "M1 Max")
        case (1, _, _, true):  return (200.0, "M1 Pro")
        case (1, _, _, _):     return (68.25, "M1")
        default:               return (400.0, "Unknown (\(architecture))")
        }
    }

    /// Print GPU profiling footer: post-inference memory, timing, bandwidth (measured + estimated).
    private func printGPUProfileFooter(promptTokens: Int, completionTokens: Int, promptTime: Double, generateTime: Double) {
        // Stop IOReport timer and collect all GPU metrics
        ioReportTimer?.cancel()
        ioReportTimer = nil
        _ = sampleIOReportGPU()  // one final sample
        let gpuSamples = stopIOReportGPU()

        // Collect bandwidth from mactop if --gpu-profile-bw was specified.
        // mactop has ~5s startup overhead, so bandwidth sampling is opt-in.
        let bwSamples: [BandwidthSample]
        if gpuProfileBandwidth {
            print("[\(ts())] [GPU-PROFILE] Sampling DRAM bandwidth via mactop (~5s)...")
            bwSamples = collectBandwidthViaMactop()
        } else {
            bwSamples = []
        }

        let snap = Memory.snapshot()
        let info = GPU.deviceInfo()
        let modelWeightsMB = gpuProfileModelWeightBytes / (1024 * 1024)
        let modelWeightsGB = Double(gpuProfileModelWeightBytes) / (1024.0 * 1024.0 * 1024.0)
        let kvCacheMB = (snap.activeMemory - gpuProfileModelWeightBytes) / (1024 * 1024)
        print("[\(ts())] [GPU-PROFILE] ─── Post-inference Memory ───")
        print("[\(ts())] [GPU-PROFILE]   Active: \(snap.activeMemory / (1024*1024)) MB (weights: \(modelWeightsMB) MB + KV/runtime: \(Swift.max(0, kvCacheMB)) MB)")
        print("[\(ts())] [GPU-PROFILE]   Cache: \(snap.cacheMemory / (1024*1024)) MB")
        print("[\(ts())] [GPU-PROFILE]   Peak: \(snap.peakMemory / (1024*1024)) MB")
        print("[\(ts())] [GPU-PROFILE] ─── Timing ───")
        let prefillTokS = promptTime > 0 ? String(format: "%.1f", Double(promptTokens) / promptTime) : "n/a"
        let decodeTokS = generateTime > 0 ? String(format: "%.1f", Double(completionTokens) / generateTime) : "n/a"
        print("[\(ts())] [GPU-PROFILE]   Prefill: \(String(format: "%.3f", promptTime))s (\(promptTokens) tokens, \(prefillTokS) tok/s)")
        print("[\(ts())] [GPU-PROFILE]   Decode: \(String(format: "%.3f", generateTime))s (\(completionTokens) tokens, \(decodeTokS) tok/s)")

        let memGB = info.memorySize / (1024 * 1024 * 1024)
        let (theoreticalBW, chipName) = Self.estimateTheoreticalBandwidth(architecture: info.architecture, memoryGB: memGB)

        // Measured bandwidth from mactop (if available)
        if !bwSamples.isEmpty {
            let peakBW = bwSamples.map(\.combinedGBs).max() ?? 0
            let avgBW = bwSamples.map(\.combinedGBs).reduce(0, +) / Double(bwSamples.count)
            let peakRead = bwSamples.map(\.readGBs).max() ?? 0
            let peakWrite = bwSamples.map(\.writeGBs).max() ?? 0
            let measuredUtil = (peakBW / theoreticalBW) * 100
            print("[\(ts())] [GPU-PROFILE] ─── DRAM Bandwidth (mactop, \(bwSamples.count) samples, post-inference) ───")
            print("[\(ts())] [GPU-PROFILE]   Peak:  \(String(format: "%.1f", peakBW)) GB/s (read: \(String(format: "%.1f", peakRead)) + write: \(String(format: "%.1f", peakWrite)))")
            print("[\(ts())] [GPU-PROFILE]   Avg:   \(String(format: "%.1f", avgBW)) GB/s")
            print("[\(ts())] [GPU-PROFILE]   Chip:  \(chipName) (~\(Int(theoreticalBW)) GB/s theoretical)")
            print("[\(ts())] [GPU-PROFILE]   Utilization: \(String(format: "%.1f", measuredUtil))% of theoretical")
            if measuredUtil > 80 {
                print("[\(ts())] [GPU-PROFILE]   Bandwidth-saturated — kernel optimization won't help. Consider speculative decoding or smaller model.")
            } else if measuredUtil < 30 {
                print("[\(ts())] [GPU-PROFILE]   Low utilization — check for CPU-GPU bubbles (--gpu-trace)")
            }
        } else {
            // Fallback: calculated estimate (no mactop)
            if generateTime > 0 && completionTokens > 0 && modelWeightsGB > 0 {
                let tokPerSec = Double(completionTokens) / generateTime
                let bwGBs = modelWeightsGB * tokPerSec
                let utilPct = (bwGBs / theoreticalBW) * 100
                print("[\(ts())] [GPU-PROFILE] ─── Bandwidth Estimate (no mactop — calculated) ───")
                print("[\(ts())] [GPU-PROFILE]   Model weights: \(modelWeightsMB) MB")
                print("[\(ts())] [GPU-PROFILE]   Est. bandwidth: \(String(format: "%.1f", bwGBs)) GB/s (weights × tok/s)")
                print("[\(ts())] [GPU-PROFILE]   Chip: \(chipName) (~\(Int(theoreticalBW)) GB/s theoretical)")
                print("[\(ts())] [GPU-PROFILE]   Est. utilization: \(String(format: "%.1f", utilPct))%")
                if utilPct > 100 {
                    print("[\(ts())] [GPU-PROFILE]   Note: >100% likely means MoE model (only active expert weights read per token)")
                }
                print("[\(ts())] [GPU-PROFILE]   Measured bandwidth: --gpu-profile-bw (requires: brew install mactop)")
            }
        }
        // Native IOReport GPU power + utilization (always available, no mactop)
        if !gpuSamples.isEmpty {
            let peakPower = gpuSamples.map(\.gpuPowerW).max() ?? 0
            let avgPower = gpuSamples.map(\.gpuPowerW).reduce(0, +) / Double(gpuSamples.count)
            let peakActive = gpuSamples.map(\.gpuActivePercent).max() ?? 0
            let avgActive = gpuSamples.map(\.gpuActivePercent).reduce(0, +) / Double(gpuSamples.count)
            print("[\(ts())] [GPU-PROFILE] ─── GPU Power & Utilization (IOReport, \(gpuSamples.count) samples) ───")
            print("[\(ts())] [GPU-PROFILE]   Peak power: \(String(format: "%.1f", peakPower))W | Avg: \(String(format: "%.1f", avgPower))W")
            print("[\(ts())] [GPU-PROFILE]   Peak active: \(String(format: "%.1f", peakActive))% | Avg: \(String(format: "%.1f", avgActive))%")
            // DRAM bandwidth from IOReport power (calibrated at startup)
            let dramPowers = gpuSamples.map(\.dramPowerW).filter { $0 > 0.01 }
            if !dramPowers.isEmpty && Self.dramCalibrated {
                let avgDram = dramPowers.reduce(0, +) / Double(dramPowers.count)
                let peakDram = dramPowers.max() ?? 0
                let avgBW = Swift.max(0, avgDram - Self.dramIdlePowerW) * Self.dramGBsPerWatt
                let peakBW = Swift.max(0, peakDram - Self.dramIdlePowerW) * Self.dramGBsPerWatt
                print("[\(ts())] [GPU-PROFILE]   DRAM power: peak \(String(format: "%.1f", peakDram))W avg \(String(format: "%.1f", avgDram))W")
                print("[\(ts())] [GPU-PROFILE]   Est. DRAM BW: peak \(String(format: "%.1f", peakBW)) GB/s avg \(String(format: "%.1f", avgBW)) GB/s (calibrated: \(String(format: "%.1f", Self.dramGBsPerWatt)) GB/s/W)")
            }
        }

        print("[\(ts())] [GPU-PROFILE] ─── For deeper analysis ───")
        print("[\(ts())] [GPU-PROFILE]   Metal trace: afm mlx -m <model> --gpu-trace 10 -s \"prompt\"")
        print("[\(ts())] [GPU-PROFILE]   GPU capture: afm mlx -m <small-model> --gpu-capture /tmp/afm-trace.gputrace")
    }

    /// Apply StreamingLLM eviction to the given KV cache layers.
    /// Keeps `sinkCount` initial tokens + a sliding window of recent tokens.
    ///
    /// **TODO:** Wire into the context-length check in generate()/generateStreaming()
    /// once the `feature/max-model-len` branch is merged. The integration point is
    /// where `MLXContextLengthError` is thrown — replace the throw with eviction when
    /// `kvEvictionPolicy == "streaming"`. Until then, `--kv-eviction` is accepted but
    /// has no effect.
    func applyStreamingLLMEviction(cache: [KVCache], maxLen: Int) {
        let sinkCount = 4
        let windowSize = Swift.min(maxLen - sinkCount, maxLen * 3 / 4)
        for layer in cache {
            if let simple = layer as? KVCacheSimple {
                simple.evictStreamingLLM(sinkCount: sinkCount, windowSize: windowSize)
            }
        }
        if debugLogging {
            print("[\(ts())] [KVCache] StreamingLLM eviction applied: kept \(sinkCount) sinks + \(windowSize) recent tokens")
        }
    }

    func normalizeModel(_ raw: String) -> String {
        resolver.normalizedModelID(raw)
    }

    func revalidateRegistry() throws -> [String] {
        try registry.revalidate(using: resolver)
    }

    func ensureLoaded(
        model rawModel: String,
        progress: (@Sendable (Progress) -> Void)? = nil,
        stage: (@Sendable (MLXLoadStage) -> Void)? = nil,
        countOperation: Bool = true
    ) async throws -> String {
        var didBeginOperation = false
        if countOperation {
            try beginOperation()
            didBeginOperation = true
        }
        defer {
            if didBeginOperation {
                endOperation()
            }
        }

        let modelID = normalizeModel(rawModel)
        guard !modelID.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw MLXServiceError.invalidModel(rawModel)
        }
        stage?(.checkingCache)

        if let cached = withStateLock({ () -> (String, ModelContainer)? in
            guard currentModelID == modelID, let container = currentContainer else { return nil }
            return (modelID, container)
        }) {
            stage?(.ready)
            return cached.0
        }

        ensureGPUConfigured()

        // Loading priority:
        // 1. Absolute/relative path — use directly (no cache or download)
        // 2. AFM cache (MACAFM_MLX_MODEL_CACHE) — always wins if model is there
        // 3. HF hub cache — validates/resumes/downloads via HubClient
        let directory: URL
        if modelID.hasPrefix("/") || modelID.hasPrefix("./") || modelID.hasPrefix("..") {
            // Absolute or relative path — resolve directly
            guard let resolved = resolver.localModelDirectory(repoId: modelID) else {
                throw MLXServiceError.modelNotFoundInCache(modelID)
            }
            directory = resolved
        } else if let root = resolver.cacheRoot, let cached = resolver.localModelDirectory(repoId: modelID),
                  cached.path.hasPrefix(root.path + "/") || cached.path == root.path {
            // Model found in AFM cache — use directly, no HubClient
            directory = cached
        } else {
            // Not in AFM cache — go through HubClient (validates, resumes, downloads)
            let parts = modelID.split(separator: "/", maxSplits: 1).map(String.init)
            let hfStyleName = "models--\(parts.count > 1 ? parts[0] : "mlx-community")--\(parts.count > 1 ? parts[1] : modelID)"
            let hfDir = Self.resolveHFHubCache().appendingPathComponent(hfStyleName)
            let isResume = FileManager.default.fileExists(atPath: hfDir.appendingPathComponent("blobs").path)
            stage?(isResume ? .resuming : .downloading)
            try await downloadModel(modelID: modelID, progress: progress)
            guard let resolved = resolver.localModelDirectory(repoId: modelID) else {
                throw MLXServiceError.modelNotFoundInCache(modelID)
            }
            directory = resolved
        }
        print("Model path: \(directory.path)")

        var config = ModelConfiguration(directory: directory)
        // Auto-detect tool call format from model type (vendor LLMModelFactory lost this code)
        var detectedFormat = inferToolCallFormat(directory: directory)
        if let fmt = detectedFormat {
            print("[\(ts())] [ToolCallParser] Auto-detected format: \(fmt) (from model_type)")
        } else {
            print("[\(ts())] [ToolCallParser] No tool call format detected for this model")
        }
        // --tool-call-parser override: force format for the specified parser
        if let parser = toolCallParser {
            switch parser {
            case "qwen3_xml", "afm_adaptive_xml":
                detectedFormat = .xmlFunction
            case "hermes", "llama3_json", "mistral":
                detectedFormat = .json
            case "gemma":
                detectedFormat = .gemma
            default:
                break
            }
            print("[\(ts())] [ToolCallParser] --tool-call-parser override: \(parser) → \(String(describing: detectedFormat))")
        }
        config.toolCallFormat = detectedFormat
        stage?(.loadingModel)

        // Loading strategy:
        // --vlm flag: force VLM factory (for models that need vision processing)
        // Otherwise: try LLM first (faster for dual-capable models like Qwen3.5-35B-A3B),
        //            fall back to VLM for vision-only model types (qwen3_vl, lfm2_vl, etc.)
        //
        // VLM-only guard: if text_config exists but lacks key architecture fields
        // (num_attention_heads, head_dim), the LLM model will use wrong defaults
        // and crash (e.g. gemma-3 VLM). Skip LLM and go straight to VLM.
        let vlmOnly = isVLMOnlyConfig(directory: directory)
        do {
            let loaded: ModelContainer
            if forceVLM || vlmOnly {
                loaded = try await VLMModelFactory.shared.loadContainer(configuration: config)
            } else {
                do {
                    loaded = try await LLMModelFactory.shared.loadContainer(configuration: config)
                } catch {
                    if debugLogging {
                        print("[\(ts())] [MLX] LLM factory load failed for \(modelID): \(error)")
                    }
                    // LLM factory failed — try VLM factory as fallback
                    loaded = try await VLMModelFactory.shared.loadContainer(configuration: config)
                }
            }
            withStateLock {
                currentContainer = loaded
                currentModelID = modelID
                currentToolCallFormat = detectedFormat
            }
            // Detect think start/end tags from tokenizer vocabulary
            do {
                let ctx = try await loaded.perform { context in context }
                let knownThinkPairs: [(start: String, end: String)] = [
                    ("<|channel>", "<channel|>"),  // Gemma 4 channel-based thinking
                    ("<think>", "</think>"),
                    ("<|think|>", "<|/think|>"),
                    ("<reasoning>", "</reasoning>"),
                ]
                // Check if a string is a single token in the vocabulary (not decomposed into subwords)
                func isSingleToken(_ s: String, _ tokenizer: any Tokenizer) -> Bool {
                    let ids = tokenizer.encode(text: s)
                    // A real single-vocab-entry token encodes to exactly 1 token
                    // (some tokenizers add BOS, so allow 1 or 2 with the token being the last)
                    return ids.count == 1 || (ids.count == 2 && tokenizer.decode(tokens: [ids.last!]) == s)
                }
                for pair in knownThinkPairs {
                    if isSingleToken(pair.start, ctx.tokenizer)
                        && isSingleToken(pair.end, ctx.tokenizer)
                    {
                        self.thinkStartTag = pair.start
                        self.thinkEndTag = pair.end
                        if debugLogging {
                            print("[\(ts())] [Think] Detected think tags: \(pair.start) / \(pair.end)")
                        }
                        break
                    }
                }
                // Gemma 4 channel-based thinking: auto-enable so the template
                // wraps reasoning in <|channel>...<channel|> tags (extractable).
                // Without this, the model still thinks but as plain text with no markers.
                if self.thinkStartTag == "<|channel>" {
                    var kwargs = self.defaultChatTemplateKwargs ?? [:]
                    if kwargs["enable_thinking"] == nil {
                        kwargs["enable_thinking"] = true
                        self.defaultChatTemplateKwargs = kwargs
                        if debugLogging {
                            print("[\(ts())] [Think] Auto-enabled Gemma 4 channel thinking")
                        }
                    }
                }
                if self.thinkStartTag == nil && debugLogging {
                    print("[\(ts())] [Think] No think tokens found in vocabulary")
                }
            }
            self.radixCache?.invalidateAll()
            if self.enablePrefixCaching {
                self.radixCache = RadixTreeCache(
                    modelID: modelID,
                    maxEntries: 64,
                    debugLogging: debugLogging
                )
                print("[\(ts())] [PrefixCache] Radix tree prefix caching active (64 entries max)")
            } else {
                self.radixCache = nil
                print("[\(ts())] [PrefixCache] Prefix caching disabled")
            }
            try registry.registerModel(modelID)
            stage?(.ready)
            return modelID
        } catch {
            throw MLXServiceError.loadFailed("\(modelID): \(error.localizedDescription)")
        }
    }

    /// Initialize the concurrent BatchScheduler by extracting model/tokenizer/processor
    /// from the container. Must be called after ensureLoaded() and only when maxConcurrent >= 2.
    func initScheduler() async throws {
        guard maxConcurrent >= 2 else { return }
        startedInBatchMode = true
        guard let container = withStateLock({ currentContainer }) else {
            throw MLXServiceError.noModelLoaded
        }
        let prefixCaching = self.enablePrefixCaching
        let limit = self.maxConcurrent
        let sched = await container.perform { context -> BatchScheduler in
            BatchScheduler(
                model: context.model,
                tokenizer: context.tokenizer,
                processor: context.processor,
                configuration: context.configuration,
                maxConcurrent: limit,
                enablePrefixCaching: prefixCaching,
                cacheProfilePath: self.cacheProfilePath
            )
        }
        self.scheduler = sched
        print("[\(ts())] Concurrent mode: up to \(limit) parallel generations\(prefixCaching ? " (prefix caching enabled)" : "")")
    }

    /// Auto-promote from serial to batch mode for batch requests.
    /// Thread-safe: uses stateLock + promotionInProgress to prevent races.
    func ensureBatchMode(concurrency: Int) async throws {
        // Fast path: scheduler already exists
        if withStateLock({ scheduler != nil }) {
            _activeBatchCount.withLock { $0 += 1 }
            // Cancel any pending teardown
            teardownWorkItem?.cancel()
            teardownWorkItem = nil
            return
        }

        // Check if another caller is already promoting
        let shouldPromote = withStateLock { () -> Bool in
            if scheduler != nil { return false }
            if promotionInProgress { return false }
            promotionInProgress = true
            return true
        }

        if !shouldPromote {
            // Wait for the other caller to finish promotion
            while withStateLock({ promotionInProgress && scheduler == nil }) {
                try await Task.sleep(nanoseconds: 10_000_000) // 10ms
            }
            // Verify scheduler was actually installed — promotion may have failed
            guard withStateLock({ scheduler != nil }) else {
                throw MLXServiceError.noModelLoaded
            }
            _activeBatchCount.withLock { $0 += 1 }
            return
        }

        // Promote: create scheduler
        let limit = max(concurrency, 8)
        self.maxConcurrent = limit
        self.enablePrefixCaching = true

        guard let container = withStateLock({ currentContainer }) else {
            withStateLock { promotionInProgress = false }
            throw MLXServiceError.noModelLoaded
        }

        let sched = await container.perform { context -> BatchScheduler in
            BatchScheduler(
                model: context.model,
                tokenizer: context.tokenizer,
                processor: context.processor,
                configuration: context.configuration,
                maxConcurrent: limit,
                enablePrefixCaching: true,
                cacheProfilePath: self.cacheProfilePath
            )
        }

        withStateLock {
            self.scheduler = sched
            self.promotionInProgress = false
        }
        _activeBatchCount.withLock { $0 += 1 }
        print("[\(ts())] Auto-promoted to batch mode: \(limit) concurrent slots (prefix caching enabled)")
    }

    /// Decrement batch reference count and schedule teardown if appropriate.
    func releaseBatchReference() {
        let remaining = _activeBatchCount.withLock { count -> Int in
            count = max(0, count - 1)
            return count
        }

        // Only teardown if auto-promoted (not started with --concurrent)
        guard !startedInBatchMode, remaining == 0 else { return }

        // Schedule teardown after grace period
        teardownWorkItem?.cancel()
        let workItem = DispatchWorkItem { [weak self] in
            guard let self else { return }
            Task {
                await self.performTeardownIfIdle()
            }
        }
        teardownWorkItem = workItem
        DispatchQueue.global().asyncAfter(deadline: .now() + 5.0, execute: workItem)
    }

    /// Tear down auto-promoted scheduler if no active slots or batches.
    private func performTeardownIfIdle() async {
        let shouldTeardown = _activeBatchCount.withLock { $0 == 0 }
        guard shouldTeardown else { return }

        // Check scheduler has no active slots
        if let sched = withStateLock({ scheduler }) {
            guard sched.activeSlotCount == 0 else { return }
        }

        withStateLock {
            self.scheduler = nil
            self.maxConcurrent = 0
            self.enablePrefixCaching = false
        }
        print("[\(ts())] Auto-teardown: returned to serial mode")
    }

    /// Forward cancellation to the scheduler for in-flight batch slots.
    func cancelBatchSlots(ids: Set<UUID>) async {
        guard let sched = withStateLock({ scheduler }) else { return }
        await sched.cancelSlots(ids: ids)
    }

    func generate(
        model: String,
        messages: [Message],
        temperature: Double?,
        maxTokens: Int?,
        topP: Double?,
        repetitionPenalty: Double?,
        topK: Int? = nil,
        minP: Double? = nil,
        presencePenalty: Double? = nil,
        seed: Int? = nil,
        logprobs: Bool? = nil,
        topLogprobs: Int? = nil,
        tools: [RequestTool]? = nil,
        stop: [String]? = nil,
        responseFormat: ResponseFormat? = nil,
        chatTemplateKwargs: [String: AnyCodable]? = nil
    ) async throws -> (modelID: String, content: String, promptTokens: Int, completionTokens: Int, tokenLogprobs: [ResolvedLogprob]?, toolCalls: [ResponseToolCall]?, cachedTokens: Int, promptTime: Double, generateTime: Double, stoppedBySequence: Bool) {
        try beginOperation()
        defer { endOperation() }

        let modelID = try await ensureLoaded(model: model, countOperation: false)
        guard let container = withStateLock({ currentContainer }) else { throw MLXServiceError.noModelLoaded }

        let promptText = buildPrompt(from: messages)
        let toolSpecs = convertToToolSpecs(tools)
        let (userInput, mediaTempFiles) = try buildUserInput(from: messages, tools: toolSpecs, responseFormat: responseFormat, chatTemplateKwargs: chatTemplateKwargs)
        defer { cleanupTempFiles(mediaTempFiles) }
        let wantLogprobs = logprobs == true
        let effectiveMaxTokens = capMaxTokensForCapture(maxTokens ?? 2000)

        var params = GenerateParameters(
            maxTokens: effectiveMaxTokens,
            kvBits: self.kvBits,
            kvGroupSize: 64,
            quantizedKVStart: 0,
            temperature: normalizedTemperature(temperature),
            topP: normalizedTopP(topP),
            repetitionPenalty: normalizedRepetitionPenalty(repetitionPenalty),
            repetitionContextSize: 64,
            topK: normalizedTopK(topK),
            minP: normalizedMinP(minP),
            presencePenalty: normalizedPresencePenalty(presencePenalty),
            seed: normalizedSeed(seed),
            computeLogprobs: wantLogprobs,
            topLogprobsCount: wantLogprobs ? min(max(topLogprobs ?? 0, 0), 20) : 0,
            prefillStepSize: self.prefillStepSize
        )

        var collectedLogprobs = [TokenLogprobData]()
        var resolvedLogprobs: [ResolvedLogprob]? = nil
        var collectedToolCalls = [ToolCall]()
        var completionInfo: GenerateCompletionInfo? = nil
        var cachedTokenCount = 0
        var stoppedBySequence = false
        var cacheOutcome = "disabled"
        var cacheLookupTime: Double? = nil
        var cacheRestoreTime: Double? = nil
        var cacheTrimTime: Double? = nil
        var cacheTruncateTime: Double? = nil
        var cacheInputTokenCount = 0
        var saveTrimTime: Double? = nil
        var saveTruncateTime: Double? = nil
        var saveInsertTime: Double? = nil
        // GPU capture/trace/profile: start before inference
        let capturePath = gpuCapturePath
        let capturing = beginGPUCaptureIfNeeded()
        let tracing = beginGPUTraceIfNeeded()
        if gpuProfile { printGPUProfileHeader() }
        let generated: String = try await container.perform { context in
            // Grammar constraint setup (needs tokenizer from context)
            let constrainedDecoding = self.setupConstrainedDecodingProcessor(
                modelID: modelID,
                responseFormat: responseFormat,
                tokenizer: context.tokenizer,
                tools: tools
            )
            defer {
                constrainedDecoding?.matcherHandle?.release()
            }
            if let constrainedDecoding {
                params.extraProcessor = constrainedDecoding.processor
            }

            let input = try await context.processor.prepare(input: userInput)

            // DEBUG/VV: decode and print the full prompt to see what the template produced
            if debugLogging || self.trace {
                let allTokens = input.text.tokens.reshaped(-1).asArray(Int.self)
                let decoded = context.tokenizer.decode(tokens: allTokens)
                if self.trace {
                    print("\(vvCyan)[\(ts())] [VV] SEND→MODEL full prompt (\(allTokens.count) tokens):\n\(decoded)\(vvReset)")
                    fflush(stdout)
                } else {
                    print("[\(ts())] [DEBUG] Full tokenized prompt (\(allTokens.count) tokens):\n\(decoded)\n[/DEBUG]")
                }
                // Hash the full token array to detect non-deterministic tokenization
                let tokenHash = allTokens.withUnsafeBufferPointer { buf -> UInt64 in
                    var h: UInt64 = 0xcbf29ce484222325  // FNV-1a
                    for t in buf {
                        h ^= UInt64(bitPattern: Int64(t))
                        h &*= 0x100000001b3
                    }
                    return h
                }
                print("[\(ts())] [PrefixCache] Token hash: \(String(tokenHash, radix: 16)) (\(allTokens.count) tokens)")
            }

            // If the chat template appended a think start tag, prepend it so extractors can detect it
            let thinkStart = self.thinkStartTag
            let tokens = input.text.tokens
            let ndim = tokens.ndim
            let seqLen = tokens.dim(ndim - 1)
            var out = ""
            var templateInjectedThink = false
            if let thinkStart, seqLen >= 2 {
                let flat = tokens.reshaped(-1)
                let lastTwo = flat[seqLen - 2 ..< seqLen].asArray(Int.self)
                let decoded = context.tokenizer.decode(tokens: lastTwo)
                if decoded.contains(thinkStart) {
                    out = thinkStart
                    templateInjectedThink = true
                }
            }

            // Prompt caching: determine cache hit/miss
            let useCache = !self.isMultimodalInput(input)
            let inputTokens = useCache ? self.extractTokenArray(input) : []
            if debugLogging {
                let cacheState = self.radixCache != nil ? "active(\(self.radixCache!.count) entries)" : "nil"
                print("[\(ts())] [PrefixCache] Path: non-streaming | useCache=\(useCache) | radixCache=\(cacheState)")
                if useCache && inputTokens.count > 0 {
                    print("[\(ts())] [PrefixCache] Input: \(inputTokens.count) tokens, first20=\(Array(inputTokens.prefix(20)))")
                }
            }
            var generationCache = context.model.newCache(parameters: params)
            var generateInput: LMInput

            cacheOutcome = useCache ? "disabled" : "multimodal-skip"
            cacheLookupTime = nil
            cacheRestoreTime = nil
            cacheTrimTime = nil
            cacheTruncateTime = nil
            cacheInputTokenCount = inputTokens.count

            if useCache, let radix = self.radixCache {
                let tLookup0 = Date.timeIntervalSinceReferenceDate
                let (prefixLen, layerStates, layerMetaStates) = radix.findPrefix(inputTokens)
                let tLookup1 = Date.timeIntervalSinceReferenceDate
                cacheLookupTime = tLookup1 - tLookup0
                let effectivePrefix = self.effectiveCachedPrefix(
                    prefixLen: prefixLen,
                    inputTokenCount: inputTokens.count,
                    cache: generationCache
                )
                let bypassExactReplay = prefixLen == inputTokens.count && effectivePrefix == 0 && prefixLen > 0

                if effectivePrefix > 0, let states = layerStates {
                    let tRestore0 = Date.timeIntervalSinceReferenceDate
                    // Restore KV cache from radix tree state
                    if debugLogging {
                        print("[\(ts())] [PrefixCache] RESTORE-BEGIN: prefixLen=\(prefixLen), effectivePrefix=\(effectivePrefix), inputTokens=\(inputTokens.count)")
                        for i in [0, 1, 3, 7] where i < states.count {
                            let shapes = states[i].map { "\($0.shape)" }.joined(separator: ", ")
                            print("[\(ts())] [PrefixCache] STORED layer[\(i)]: \(states[i].count) arrays, shapes=[\(shapes)]")
                        }
                    }
                    for i in 0..<generationCache.count where i < states.count {
                        generationCache[i].state = states[i]
                        let savedMetaState = layerMetaStates.flatMap { i < $0.count ? $0[i] : nil }
                        if let adjustedMetaState = self.restoredMetaState(
                            for: generationCache[i],
                            savedMetaState: savedMetaState
                        ) {
                            generationCache[i].metaState = adjustedMetaState
                        }
                    }
                    let tRestore1 = Date.timeIntervalSinceReferenceDate
                    if debugLogging {
                        let diagLayers = [0, 1, 3, 7]
                        for i in diagLayers where i < generationCache.count {
                            let layer = generationCache[i]
                            let cacheType = type(of: layer)
                            let stateShapes = layer.state.map { "\($0.shape)" }.joined(separator: ", ")
                            print("[\(ts())] [PrefixCache] POST-ASSIGN layer[\(i)] (\(cacheType)): offset=\(layer.offset), shapes=[\(stateShapes)]")
                        }
                    }
                    // Trim to effective prefix length
                    for i in 0..<generationCache.count {
                        let excess = generationCache[i].offset - effectivePrefix
                        if debugLogging && (i == 0 || i == 3 || i == 7) {
                            print("[\(ts())] [PrefixCache] TRIM layer[\(i)]: offset=\(generationCache[i].offset), effectivePrefix=\(effectivePrefix), excess=\(excess)")
                        }
                        if excess > 0 { generationCache[i].trim(excess) }
                    }
                    let tTrim = Date.timeIntervalSinceReferenceDate
                    // Physically truncate trimmed cache arrays to eliminate stale data. (#47)
                    for i in 0..<generationCache.count {
                        if generationCache[i].isTrimmable && generationCache[i].offset > 0
                            && self.supportsPhysicalTruncation(generationCache[i])
                        {
                            generationCache[i].truncateToOffset()
                        }
                    }
                    let tRoundtrip = Date.timeIntervalSinceReferenceDate
                    let suffixTokens = Array(inputTokens[effectivePrefix...])
                    generateInput = LMInput(text: .init(tokens: MLXArray(suffixTokens)))
                    cachedTokenCount = effectivePrefix
                    cacheOutcome = "hit"
                    cacheRestoreTime = tRestore1 - tRestore0
                    cacheTrimTime = tTrim - tRestore1
                    cacheTruncateTime = tRoundtrip - tTrim
                    self.logCachePrefill(
                        mode: "non-streaming",
                        outcome: "hit",
                        inputTokenCount: inputTokens.count,
                        cachedTokenCount: effectivePrefix,
                        suffixTokenCount: suffixTokens.count,
                        radixEntryCount: radix.count,
                        cache: generationCache,
                        lookupTime: cacheLookupTime,
                        restoreTime: tRestore1 - tRestore0,
                        trimTime: tTrim - tRestore1,
                        truncateTime: tRoundtrip - tTrim
                    )
                    self.logReplayBoundary(
                        tokenizer: context.tokenizer,
                        mode: "non-streaming",
                        inputTokens: inputTokens,
                        effectivePrefix: effectivePrefix
                    )
                    if debugLogging {
                        print("[\(ts())] [KVCache] Radix hit: \(effectivePrefix)/\(inputTokens.count) tokens cached, processing \(suffixTokens.count) suffix")
                        print("[\(ts())] [PrefixCache] Timing: restore=\(String(format: "%.3f", tRestore1 - tRestore0))s trim=\(String(format: "%.3f", tTrim - tRestore1))s roundtrip=\(String(format: "%.3f", tRoundtrip - tTrim))s total=\(String(format: "%.3f", tRoundtrip - tRestore0))s")
                    }
                } else {
                    generateInput = input
                    cacheOutcome = bypassExactReplay ? "exact-replay-bypass" : "miss"
                    self.logCachePrefill(
                        mode: "non-streaming",
                        outcome: cacheOutcome,
                        inputTokenCount: inputTokens.count,
                        cachedTokenCount: 0,
                        suffixTokenCount: inputTokens.count,
                        radixEntryCount: radix.count,
                        cache: generationCache,
                        lookupTime: cacheLookupTime
                    )
                    if debugLogging {
                        print("[\(ts())] [KVCache] Cache miss, full prefill for \(inputTokens.count) tokens")
                    }
                }
            } else {
                generateInput = input
                self.logCachePrefill(
                    mode: "non-streaming",
                    outcome: cacheOutcome,
                    inputTokenCount: inputTokens.count,
                    cachedTokenCount: 0,
                    suffixTokenCount: inputTokens.count,
                    radixEntryCount: self.radixCache?.count,
                    cache: generationCache
                )
                if debugLogging {
                    print("[\(ts())] [KVCache] Multimodal input, skipping cache")
                }
            }

            let activeStops = stop?.filter { !$0.isEmpty } ?? []
            var insideThink = templateInjectedThink
            var visibleContentStart: String.Index? = nil  // Index where content after </think> begins
            let genStart = Date()
            var firstTokenTime: Date?
            // INSTRUMENT: Dump cache state right before generation starts
            if debugLogging || self.trace {
                print("[\(ts())] [PREFLIGHT] About to generate. Cache layers: \(generationCache.count), input shape: \(generateInput.text.tokens.shape)")
                for i in 0..<min(generationCache.count, 40) {
                    let layer = generationCache[i]
                    if layer.offset > 0 || (i < 8 && (i % 4 == 3 || i < 2)) {
                        let shapes = layer.state.map { "\($0.shape)" }.joined(separator: ", ")
                        print("[\(ts())] [PREFLIGHT] cache[\(i)] (\(type(of: layer))): offset=\(layer.offset), shapes=[\(shapes)]")
                    }
                }
                fflush(stdout)
            }
            do {
                for await piece in try MLXLMCommon.generate(input: generateInput, cache: generationCache, parameters: params, context: context) {
                    if debugLogging {
                        print("[\(ts())] [DEBUG] Generation piece: \(piece)")
                    }
                    if case .chunk(let text) = piece {
                        if firstTokenTime == nil { firstTokenTime = Date() }
                        // Track think boundaries — stop sequences only apply outside
                        if let ts = thinkStart, text.contains(ts) { insideThink = true }
                        if let te = self.thinkEndTag, text.contains(te) { insideThink = false }
                        out += text
                        // Record where visible content starts (after think end tag)
                        if !insideThink && visibleContentStart == nil {
                            if let te = self.thinkEndTag, let thinkEnd = out.range(of: te) {
                                visibleContentStart = thinkEnd.upperBound
                            } else {
                                visibleContentStart = out.startIndex
                            }
                        }
                        // Only check stop sequences against visible content (after </think>)
                        if !activeStops.isEmpty && !insideThink, let vcStart = visibleContentStart {
                            let visibleContent = String(out[vcStart...])
                            if let match = activeStops.first(where: { visibleContent.contains($0) }) {
                                if let range = visibleContent.range(of: match) {
                                    let keepEnd = out.index(vcStart, offsetBy: visibleContent.distance(from: visibleContent.startIndex, to: range.lowerBound))
                                    out = String(out[..<keepEnd])
                                }
                                stoppedBySequence = true
                                break
                            }
                        }
                    } else if case .tokenLogprobs(let lps) = piece {
                        collectedLogprobs.append(contentsOf: lps)
                    } else if case .toolCall(let tc) = piece {
                        if debugLogging {
                            print("[\(ts())] [DEBUG] Tool call detected: \(tc.function.name)(\(tc.function.arguments))")
                        }
                        collectedToolCalls.append(tc)
                    } else if case .info(let info) = piece {
                        completionInfo = info
                    }
                }
            } catch {
                // On generation error, invalidate prompt cache inside the
                // serialized block so no stale state leaks to the next request.
                if debugLogging {
                    print("[\(ts())] [PrefixCache] Invalidate: generation error")
                }
                self.radixCache?.invalidateAll()
                throw error
            }

            Stream.gpu.synchronize()
            // Optional per-request GPU memory cleanup (gated to avoid throughput hit).
            // Enable with AFM_CLEAR_GPU_CACHE=1 if you see memory-related crashes.
            if clearGPUCachePerRequest {
                Memory.clearCache()
            }
            if debugLogging {
                let ttft = firstTokenTime.map { $0.timeIntervalSince(genStart) } ?? 0
                let total = Date().timeIntervalSince(genStart)
                let promptTok = completionInfo?.promptTokenCount ?? 0
                let genTok = completionInfo?.generationTokenCount ?? 0
                print("[\(ts())] [KVCache] Timing: TTFT=\(String(format: "%.3f", ttft))s total=\(String(format: "%.3f", total))s prompt_tokens=\(promptTok) gen_tokens=\(genTok)")
            }

            // Save prompt cache state into radix tree
            if useCache, let radix = self.radixCache, !inputTokens.isEmpty {
                let promptLen = inputTokens.count
                let tSave0 = Date.timeIntervalSinceReferenceDate
                if debugLogging {
                    // Log KVCacheSimple layers (full attn at interval 4: indices 3,7,11,...) and MambaCache layers (0,1)
                    let diagLayers = [0, 1, 3, 7]  // 0,1=Mamba, 3,7=KVCacheSimple
                    for i in diagLayers where i < generationCache.count {
                        let layer = generationCache[i]
                        let cacheType = type(of: layer)
                        let stateShapes = layer.state.map { "\($0.shape)" }.joined(separator: ", ")
                        print("[\(ts())] [PrefixCache] PRE-TRIM layer[\(i)] (\(cacheType)): offset=\(layer.offset), isTrimmable=\(layer.isTrimmable), shapes=[\(stateShapes)]")
                    }
                    print("[\(ts())] [PrefixCache] PRE-TRIM: promptLen=\(promptLen)")
                }
                for layer in generationCache {
                    let excess = layer.offset - promptLen
                    if excess > 0 { layer.trim(excess) }
                }
                let tSaveTrim = Date.timeIntervalSinceReferenceDate
                // Physically truncate trimmed cache arrays to eliminate stale data. (#47)
                for i in 0..<generationCache.count {
                    if generationCache[i].isTrimmable && generationCache[i].offset > 0
                        && self.supportsPhysicalTruncation(generationCache[i])
                    {
                        generationCache[i].truncateToOffset()
                    }
                }
                let tSaveTruncate = Date.timeIntervalSinceReferenceDate
                if debugLogging {
                    let diagLayers = [0, 1, 3, 7]
                    for i in diagLayers where i < generationCache.count {
                        let layer = generationCache[i]
                        let cacheType = type(of: layer)
                        let stateShapes = layer.state.map { "\($0.shape)" }.joined(separator: ", ")
                        print("[\(ts())] [PrefixCache] POST-TRIM layer[\(i)] (\(cacheType)): offset=\(layer.offset), shapes=[\(stateShapes)]")
                    }
                }
                let layerStates = generationCache.map { $0.state }
                let layerMetaStates = generationCache.map { $0.metaState }
                radix.insert(
                    tokens: inputTokens,
                    layerStates: layerStates,
                    layerMetaStates: layerMetaStates
                )
                let tSaveInsert = Date.timeIntervalSinceReferenceDate
                saveTrimTime = tSaveTrim - tSave0
                saveTruncateTime = tSaveTruncate - tSaveTrim
                saveInsertTime = tSaveInsert - tSaveTruncate
                self.logCacheSave(
                    mode: "non-streaming",
                    inputTokenCount: inputTokens.count,
                    radixEntryCount: radix.count,
                    cache: generationCache,
                    trimTime: saveTrimTime,
                    truncateTime: saveTruncateTime,
                    insertTime: saveInsertTime
                )
                if debugLogging {
                    print("[\(ts())] [PrefixCache] Insert: \(inputTokens.count) tokens, \(generationCache.count) layers")
                    // Log stored state for KVCacheSimple layers
                    for i in [3, 7] where i < layerStates.count {
                        let shapes = layerStates[i].map { "\($0.shape)" }.joined(separator: ", ")
                        print("[\(ts())] [PrefixCache] Stored layer[\(i)] shapes: [\(shapes)]")
                    }
                }
            }

            if wantLogprobs && !collectedLogprobs.isEmpty {
                resolvedLogprobs = self.resolveLogprobs(collectedLogprobs, tokenizer: context.tokenizer)
            }

            return out
        }

        // GPU capture/trace/profile: end after GPU sync completes inside container.perform
        if capturing, let path = capturePath {
            endGPUCapture(path: path)
        }
        if tracing {
            endGPUTrace()
        }
        if gpuProfile {
            let pTok = completionInfo?.promptTokenCount ?? estimateTokens(promptText)
            let cTok = completionInfo?.generationTokenCount ?? estimateTokens(generated)
            let pTime = completionInfo?.promptTime ?? 0
            let gTime = completionInfo?.generateTime ?? 0
            printGPUProfileFooter(promptTokens: pTok, completionTokens: cTok, promptTime: pTime, generateTime: gTime)
        }

        // If the vendor ToolCallProcessor didn't detect tool calls, try fallback parsing.
        // Qwen3-Coder outputs <tool_call><function=name>...</function></tool_call> which
        // the vendor's XMLFunctionParser misses (regex doesn't match multiline content).
        var finalToolCalls = collectedToolCalls
        var finalContent = generated
        if finalToolCalls.isEmpty && tools != nil {
            if debugLogging {
                print("[\(ts())] [ToolCallParser] Vendor parser found 0 tool calls, trying fallback on \(generated.count) chars")
            }
            let (parsed, remaining) = ToolCallStreamingRuntime.parseCompletedToolCalls(
                from: generated,
                toolCallParser: self.toolCallParser,
                tools: tools
            )
            if !parsed.isEmpty {
                finalToolCalls = parsed
                finalContent = remaining
            }
        }

        let responseToolCalls: [ResponseToolCall]? = finalToolCalls.isEmpty
            ? nil
            : Self.normalizeToolCalls(
                finalToolCalls,
                tools: tools,
                fixToolArgs: self.fixToolArgs
            )

            let promptTokens = cacheInputTokenCount > 0
                ? cacheInputTokenCount
                : (completionInfo?.promptTokenCount ?? estimateTokens(promptText))
            let completionTokens = completionInfo?.generationTokenCount ?? estimateTokens(generated)
            let promptTime = completionInfo?.promptTime ?? 0
            let generateTime = completionInfo?.generateTime ?? 0
            self.logCacheProfile(
                phase: "restore",
                mode: "non-streaming",
                outcome: cacheOutcome,
                inputTokenCount: cacheInputTokenCount,
                cachedTokenCount: cachedTokenCount,
                promptTime: promptTime,
                lookupTime: cacheLookupTime,
                restoreTime: cacheRestoreTime,
                trimTime: cacheTrimTime,
                truncateTime: cacheTruncateTime
            )
            self.logCacheProfile(
                phase: "save",
                mode: "non-streaming",
                outcome: saveInsertTime != nil ? "save" : "skip",
                inputTokenCount: cacheInputTokenCount,
                cachedTokenCount: cachedTokenCount,
                promptTime: promptTime,
                trimTime: saveTrimTime,
                truncateTime: saveTruncateTime,
                insertTime: saveInsertTime
            )
            return (modelID, finalContent, promptTokens, completionTokens, resolvedLogprobs, responseToolCalls, cachedTokenCount, promptTime, generateTime, stoppedBySequence)
        }

    func generateStreaming(
        model: String,
        messages: [Message],
        temperature: Double?,
        maxTokens: Int?,
        topP: Double?,
        repetitionPenalty: Double?,
        topK: Int? = nil,
        minP: Double? = nil,
        presencePenalty: Double? = nil,
        seed: Int? = nil,
        logprobs: Bool? = nil,
        topLogprobs: Int? = nil,
        tools: [RequestTool]? = nil,
        stop: [String]? = nil,
        responseFormat: ResponseFormat? = nil,
        chatTemplateKwargs: [String: AnyCodable]? = nil
    ) async throws -> (modelID: String, stream: AsyncThrowingStream<StreamChunk, Error>, promptTokens: Int, toolCallStartTag: String?, toolCallEndTag: String?, thinkStartTag: String?, thinkEndTag: String?) {
        try beginOperation()
        var endOperationOnExit = true
        defer {
            if endOperationOnExit {
                endOperation()
            }
        }
        // Streaming keeps the operation open until the background Task finishes.
        // The fallback defer above only handles setup failures before the Task
        // is created; the Task itself owns the normal endOperation() call.

        let modelID = try await ensureLoaded(model: model, countOperation: false)
        guard let container = withStateLock({ currentContainer }) else { throw MLXServiceError.noModelLoaded }

        let promptText = buildPrompt(from: messages)
        let toolSpecs = convertToToolSpecs(tools)
        // -VV: Log tool schemas as sent to model's Jinja template
        if trace, let toolSpecs {
            for spec in toolSpecs {
                if let data = try? JSONSerialization.data(withJSONObject: spec, options: [.prettyPrinted, .sortedKeys]),
                   let str = String(data: data, encoding: .utf8) {
                    print("\(vvCyan)[\(ts())] [VV] SEND→MODEL tool spec (Jinja):\n\(str)\(vvReset)")
                }
            }
            fflush(stdout)
        }
        let (userInput, mediaTempFiles) = try buildUserInput(from: messages, tools: toolSpecs, responseFormat: responseFormat, chatTemplateKwargs: chatTemplateKwargs)
        let promptTokens = estimateTokens(promptText)
        let wantLogprobs = logprobs == true
        let effectiveMaxTokens = capMaxTokensForCapture(maxTokens ?? 2000)

        var params = GenerateParameters(
            maxTokens: effectiveMaxTokens,
            kvBits: self.kvBits,
            kvGroupSize: 64,
            quantizedKVStart: 0,
            temperature: normalizedTemperature(temperature),
            topP: normalizedTopP(topP),
            repetitionPenalty: normalizedRepetitionPenalty(repetitionPenalty),
            repetitionContextSize: 64,
            topK: normalizedTopK(topK),
            minP: normalizedMinP(minP),
            presencePenalty: normalizedPresencePenalty(presencePenalty),
            seed: normalizedSeed(seed),
            computeLogprobs: wantLogprobs,
            topLogprobsCount: wantLogprobs ? min(max(topLogprobs ?? 0, 0), 20) : 0,
            prefillStepSize: self.prefillStepSize
        )

        // --- Concurrent path: bypass container.perform lock, route through BatchScheduler ---
        if let scheduler = self.scheduler {
            let constrainedDecoding = try await container.perform { context in
                self.setupConstrainedDecodingProcessor(
                    modelID: modelID,
                    responseFormat: responseFormat,
                    tokenizer: context.tokenizer,
                    tools: tools
                )
            }
            if let constrainedDecoding {
                params.extraProcessor = constrainedDecoding.processor
            }

            let toolRuntimeConfig: BatchScheduler.ToolCallRuntimeConfiguration?
            if let tools, !tools.isEmpty {
                let format = withStateLock({ currentToolCallFormat })
                if let format {
                    switch format {
                    case .xmlFunction:
                        toolRuntimeConfig = .init(
                            startTag: "<tool_call>",
                            endTag: "</tool_call>",
                            parser: self.toolCallParser,
                            tools: tools
                        )
                    default:
                        let parser = format.createParser()
                        toolRuntimeConfig = .init(
                            startTag: parser.startTag ?? "<tool_call>",
                            endTag: parser.endTag ?? "</tool_call>",
                            parser: self.toolCallParser,
                            tools: tools
                        )
                    }
                } else {
                    toolRuntimeConfig = .init(
                        startTag: "<tool_call>",
                        endTag: "</tool_call>",
                        parser: self.toolCallParser,
                        tools: tools
                    )
                }
            } else {
                toolRuntimeConfig = nil
            }

            let input = try await scheduler.prepareInput(userInput)
            let preparedPromptTokens = input.text.tokens.reshaped(-1).asArray(Int.self).count
            let schedulerStream = scheduler.submit(
                input: input,
                parameters: params,
                promptTokens: preparedPromptTokens,
                toolCallRuntimeConfig: toolRuntimeConfig,
                constraintRuntimeConfig: constrainedDecoding.map {
                    BatchScheduler.ConstraintRuntimeConfiguration(
                        mode: $0.mode,
                        matcherHandle: $0.matcherHandle
                    )
                },
                stopSequences: stop ?? [],
                thinkStartTag: self.thinkStartTag,
                thinkEndTag: self.thinkEndTag
            )
            self.cleanupTempFiles(mediaTempFiles)

            // Derive tool call tags (same logic as serial path, below)
            let toolTags = toolRuntimeConfig.map { ($0.startTag, $0.endTag) }

            endOperationOnExit = false
            return (modelID, schedulerStream, preparedPromptTokens, toolTags?.0, toolTags?.1, self.thinkStartTag, self.thinkEndTag)
        }

        // --- Serial path: full-featured generation via container.perform lock ---
        // GPU capture/trace/profile: snapshot state before the async Task
        let streamCapturePath = gpuCapturePath
        let streamCapturing = beginGPUCaptureIfNeeded()
        let streamTracing = beginGPUTraceIfNeeded()
        let streamGpuProfile = gpuProfile
        if streamGpuProfile { printGPUProfileHeader() }
        let stream = AsyncThrowingStream<StreamChunk, Error> { continuation in
            let task = Task {
                defer { self.endOperation() }
                do {
                    try await container.perform { context in
                        // Grammar constraint setup — see non-streaming path for details.
                        let constrainedDecoding = self.setupConstrainedDecodingProcessor(
                            modelID: modelID,
                            responseFormat: responseFormat,
                            tokenizer: context.tokenizer,
                            tools: tools
                        )
                        defer {
                            constrainedDecoding?.matcherHandle?.release()
                        }
                        if let constrainedDecoding {
                            params.extraProcessor = constrainedDecoding.processor
                        }

                        let input = try await context.processor.prepare(input: userInput)

                        // -VV: decode and print the full prompt (streaming path)
                        if self.trace {
                            let allTokens = input.text.tokens.reshaped(-1).asArray(Int.self)
                            let decoded = context.tokenizer.decode(tokens: allTokens)
                            print("\(vvCyan)[\(ts())] [VV] SEND→MODEL full prompt (\(allTokens.count) tokens):\n\(decoded)\(vvReset)")
                            fflush(stdout)
                        }

                        // If the chat template appended a think tag, inject it
                        // into the stream so the reasoning extractor can detect it.
                        let thinkStart = self.thinkStartTag
                        var templateInjectedThink = false
                        let tokens = input.text.tokens
                        let ndim = tokens.ndim
                        let seqLen = tokens.dim(ndim - 1)
                        if let thinkStart, seqLen >= 2 {
                            let flat = tokens.reshaped(-1)
                            let lastTwo = flat[seqLen - 2 ..< seqLen].asArray(Int.self)
                            let decoded = context.tokenizer.decode(tokens: lastTwo)
                            if decoded.contains(thinkStart) {
                                continuation.yield(StreamChunk(text: thinkStart))
                                templateInjectedThink = true
                            }
                        }

                        // Prompt caching: determine cache hit/miss
                        let fullPromptTokenCount = input.text.tokens.reshaped(-1).asArray(Int.self).count
                        let useCache = !self.isMultimodalInput(input)
                        let inputTokens = useCache ? self.extractTokenArray(input) : []
                        if debugLogging {
                            let cacheState = self.radixCache != nil ? "active(\(self.radixCache!.count) entries)" : "nil"
                            print("[\(ts())] [PrefixCache] Path: streaming | useCache=\(useCache) | radixCache=\(cacheState)")
                            if useCache && inputTokens.count > 0 {
                                print("[\(ts())] [PrefixCache] Input: \(inputTokens.count) tokens, first20=\(Array(inputTokens.prefix(20)))")
                            }
                        }
                        var generationCache = context.model.newCache(parameters: params)
                        var generateInput: LMInput
                        var streamCachedTokens = 0

                        var cacheOutcome = useCache ? "disabled" : "multimodal-skip"
                        var cacheLookupTime: Double? = nil
                        var cacheRestoreTime: Double? = nil
                        var cacheTrimTime: Double? = nil
                        var cacheTruncateTime: Double? = nil

                        if useCache, let radix = self.radixCache {
                            let tLookup0 = Date.timeIntervalSinceReferenceDate
                            let (prefixLen, layerStates, layerMetaStates) = radix.findPrefix(inputTokens)
                            let tLookup1 = Date.timeIntervalSinceReferenceDate
                            cacheLookupTime = tLookup1 - tLookup0
                            let effectivePrefix = self.effectiveCachedPrefix(
                                prefixLen: prefixLen,
                                inputTokenCount: inputTokens.count,
                                cache: generationCache
                            )
                            let bypassExactReplay = prefixLen == inputTokens.count && effectivePrefix == 0 && prefixLen > 0

                            if effectivePrefix > 0, let states = layerStates {
                                let tRestore0 = Date.timeIntervalSinceReferenceDate
                                // Restore KV cache from radix tree state
                                for i in 0..<generationCache.count where i < states.count {
                                    generationCache[i].state = states[i]
                                    let savedMetaState = layerMetaStates.flatMap {
                                        i < $0.count ? $0[i] : nil
                                    }
                                    if let adjustedMetaState = self.restoredMetaState(
                                        for: generationCache[i],
                                        savedMetaState: savedMetaState
                                    ) {
                                        generationCache[i].metaState = adjustedMetaState
                                    }
                                }
                                let tRestore1 = Date.timeIntervalSinceReferenceDate
                                // Trim to effective prefix length
                                for i in 0..<generationCache.count {
                                    let excess = generationCache[i].offset - effectivePrefix
                                    if excess > 0 { generationCache[i].trim(excess) }
                                }
                                let tTrim = Date.timeIntervalSinceReferenceDate
                                // Physically truncate trimmed cache arrays to eliminate stale data. (#47)
                                for i in 0..<generationCache.count {
                                    if generationCache[i].isTrimmable && generationCache[i].offset > 0
                                        && self.supportsPhysicalTruncation(generationCache[i])
                                    {
                                        generationCache[i].truncateToOffset()
                                    }
                                }
                                let tRoundtrip = Date.timeIntervalSinceReferenceDate
                                let suffixTokens = Array(inputTokens[effectivePrefix...])
                                generateInput = LMInput(text: .init(tokens: MLXArray(suffixTokens)))
                                streamCachedTokens = effectivePrefix
                                cacheOutcome = "hit"
                                cacheRestoreTime = tRestore1 - tRestore0
                                cacheTrimTime = tTrim - tRestore1
                                cacheTruncateTime = tRoundtrip - tTrim
                                self.logCachePrefill(
                                    mode: "streaming",
                                    outcome: "hit",
                                    inputTokenCount: inputTokens.count,
                                    cachedTokenCount: effectivePrefix,
                                    suffixTokenCount: suffixTokens.count,
                                    radixEntryCount: radix.count,
                                    cache: generationCache,
                                    lookupTime: cacheLookupTime,
                                    restoreTime: tRestore1 - tRestore0,
                                    trimTime: tTrim - tRestore1,
                                    truncateTime: tRoundtrip - tTrim
                                )
                                self.logReplayBoundary(
                                    tokenizer: context.tokenizer,
                                    mode: "streaming",
                                    inputTokens: inputTokens,
                                    effectivePrefix: effectivePrefix
                                )
                                if debugLogging {
                                    print("[\(ts())] [KVCache] Radix hit: \(effectivePrefix)/\(inputTokens.count) tokens cached, processing \(suffixTokens.count) suffix")
                                    print("[\(ts())] [PrefixCache] Timing: restore=\(String(format: "%.3f", tRestore1 - tRestore0))s trim=\(String(format: "%.3f", tTrim - tRestore1))s roundtrip=\(String(format: "%.3f", tRoundtrip - tTrim))s total=\(String(format: "%.3f", tRoundtrip - tRestore0))s")
                                }
                            } else {
                                generateInput = input
                                cacheOutcome = bypassExactReplay ? "exact-replay-bypass" : "miss"
                                self.logCachePrefill(
                                    mode: "streaming",
                                    outcome: cacheOutcome,
                                    inputTokenCount: inputTokens.count,
                                    cachedTokenCount: 0,
                                    suffixTokenCount: inputTokens.count,
                                    radixEntryCount: radix.count,
                                    cache: generationCache,
                                    lookupTime: cacheLookupTime
                                )
                                if debugLogging {
                                    print("[\(ts())] [KVCache] Cache miss, full prefill for \(inputTokens.count) tokens")
                                }
                            }
                        } else {
                            generateInput = input
                            self.logCachePrefill(
                                mode: "streaming",
                                outcome: useCache ? "disabled" : "multimodal-skip",
                                inputTokenCount: inputTokens.count,
                                cachedTokenCount: 0,
                                suffixTokenCount: inputTokens.count,
                                radixEntryCount: self.radixCache?.count,
                                cache: generationCache
                            )
                            if debugLogging {
                                print("[\(ts())] [KVCache] Multimodal input, skipping cache")
                            }
                        }

                        // Emit cached token count so the controller can include it in usage
                        continuation.yield(StreamChunk(text: "", cachedTokens: streamCachedTokens))
                        self.logUsageChunk(stage: "preliminary", streaming: true, cachedTokens: streamCachedTokens)

                        let activeStops = stop?.filter { !$0.isEmpty } ?? []
                        // Buffer to handle stop strings that span chunk boundaries.
                        // Stop sequences only apply to content OUTSIDE <think> blocks —
                        // thinking models emit reasoning inside <think>...</think> tags
                        // and stop strings like "3." or "\n" commonly appear in reasoning.
                        let maxStopLen = activeStops.map(\.count).max() ?? 0
                        var stopBuffer = ""
                        var insideThink = templateInjectedThink
                        let genStart = Date()
                        var firstTokenTime: Date?

                        var pendingLogprobs: [TokenLogprobData]? = nil
                        // INSTRUMENT: Dump cache state right before generation starts (streaming)
                        if debugLogging || self.trace {
                            print("[\(ts())] [PREFLIGHT-STREAM] About to generate. Cache layers: \(generationCache.count), input shape: \(generateInput.text.tokens.shape)")
                            for i in 0..<min(generationCache.count, 40) {
                                let layer = generationCache[i]
                                if layer.offset > 0 || (i < 8 && (i % 4 == 3 || i < 2)) {
                                    let shapes = layer.state.map { "\($0.shape)" }.joined(separator: ", ")
                                    print("[\(ts())] [PREFLIGHT-STREAM] cache[\(i)] (\(type(of: layer))): offset=\(layer.offset), shapes=[\(shapes)]")
                                }
                            }
                            fflush(stdout)
                        }
                        do {
                            for await piece in try MLXLMCommon.generate(input: generateInput, cache: generationCache, parameters: params, context: context) {
                                if Task.isCancelled {
                                    print("[\(ts())] [MLX] Generation cancelled by client")
                                    break
                                }
                                if case .tokenLogprobs(let lps) = piece {
                                    pendingLogprobs = lps
                                } else if case .chunk(let text) = piece {
                                    if firstTokenTime == nil { firstTokenTime = Date() }
                                    let resolved: [ResolvedLogprob]?
                                    if let lps = pendingLogprobs {
                                        resolved = self.resolveLogprobs(lps, tokenizer: context.tokenizer)
                                    } else {
                                        resolved = nil
                                    }

                                    // Track think boundaries for stop sequence scoping
                                    let wasInsideThink = insideThink
                                    if let ts = thinkStart, text.contains(ts) { insideThink = true }
                                    if let te = self.thinkEndTag, text.contains(te) { insideThink = false }

                                    if !activeStops.isEmpty && !insideThink {
                                        // If we just transitioned out of think, only buffer text after end tag
                                        if wasInsideThink, let te = self.thinkEndTag, let thinkEndRange = text.range(of: te) {
                                            let afterThink = String(text[thinkEndRange.upperBound...])
                                            if !afterThink.isEmpty {
                                                stopBuffer += afterThink
                                            }
                                        } else {
                                            stopBuffer += text
                                        }
                                        // Check for a complete stop string match
                                        if let match = activeStops.first(where: { stopBuffer.contains($0) }) {
                                            // Emit text up to the stop string
                                            if let range = stopBuffer.range(of: match) {
                                                let before = String(stopBuffer[..<range.lowerBound])
                                                if !before.isEmpty {
                                                    continuation.yield(StreamChunk(text: before, logprobs: resolved, stoppedBySequence: true))
                                                } else {
                                                    continuation.yield(StreamChunk(text: "", logprobs: resolved, stoppedBySequence: true))
                                                }
                                            }
                                            break
                                        }
                                        // Flush safe portion of the buffer (keep tail that could be partial stop match)
                                        if stopBuffer.count > maxStopLen {
                                            let flushEnd = stopBuffer.index(stopBuffer.endIndex, offsetBy: -maxStopLen)
                                            let flushText = String(stopBuffer[..<flushEnd])
                                            stopBuffer = String(stopBuffer[flushEnd...])
                                            continuation.yield(StreamChunk(text: flushText, logprobs: resolved))
                                        }
                                    } else {
                                        // Inside <think> or no stop sequences — pass through
                                        continuation.yield(StreamChunk(text: text, logprobs: resolved))
                                    }
                                    pendingLogprobs = nil
                                } else if case .toolCall(let tc) = piece {
                                    // Emit tool call as a stream chunk with empty text
                                    let responseTC = Self.convertToolCall(tc, index: 0)
                                    continuation.yield(StreamChunk(text: "", toolCalls: [responseTC]))
                                } else if case .info(let info) = piece {
                                    // Emit real token counts and timing as a final info chunk
                                    let finalPromptTokens = fullPromptTokenCount
                                    self.logUsageChunk(
                                        stage: "final",
                                        streaming: true,
                                        cachedTokens: streamCachedTokens,
                                        promptTokens: finalPromptTokens,
                                        completionTokens: info.generationTokenCount,
                                        promptTime: info.promptTime,
                                        generateTime: info.generateTime
                                    )
                                    self.logCacheProfile(
                                        phase: "restore",
                                        mode: "streaming",
                                        outcome: cacheOutcome,
                                        inputTokenCount: inputTokens.count,
                                        cachedTokenCount: streamCachedTokens,
                                        promptTime: info.promptTime,
                                        lookupTime: cacheLookupTime,
                                        restoreTime: cacheRestoreTime,
                                        trimTime: cacheTrimTime,
                                        truncateTime: cacheTruncateTime
                                    )
                                    continuation.yield(StreamChunk(text: "", promptTokens: finalPromptTokens, completionTokens: info.generationTokenCount, promptTime: info.promptTime, generateTime: info.generateTime))
                                    // GPU profile: emit footer with real token counts
                                    if streamGpuProfile {
                                        self.printGPUProfileFooter(promptTokens: finalPromptTokens, completionTokens: info.generationTokenCount, promptTime: info.promptTime, generateTime: info.generateTime)
                                    }
                                }
                            }
                        } catch {
                            // On generation error, invalidate prompt cache inside the
                            // serialized block so no stale state leaks to the next request.
                            if debugLogging {
                                print("[\(ts())] [PrefixCache] Invalidate: generation error (streaming)")
                            }
                            self.radixCache?.invalidateAll()
                            throw error
                        }
                        // Flush any remaining buffered text (no stop match found)
                        if !activeStops.isEmpty && !stopBuffer.isEmpty {
                            continuation.yield(StreamChunk(text: stopBuffer))
                        }
                        // Synchronize GPU after generation completes (or breaks early).
                        Stream.gpu.synchronize()
                        // GPU capture/trace: stop after GPU sync
                        if streamCapturing, let path = streamCapturePath {
                            self.endGPUCapture(path: path)
                        }
                        if streamTracing {
                            self.endGPUTrace()
                        }
                        // Optional per-request GPU memory cleanup (gated to avoid throughput hit).
                        if clearGPUCachePerRequest {
                            Memory.clearCache()
                        }
                        // Invalidate prompt cache on cancellation to prevent stale state
                        if Task.isCancelled {
                            if debugLogging {
                                print("[\(ts())] [PrefixCache] Invalidate: task cancelled")
                            }
                            self.radixCache?.invalidateAll()
                        }
                        if debugLogging {
                            let ttft = firstTokenTime.map { $0.timeIntervalSince(genStart) } ?? 0
                            let total = Date().timeIntervalSince(genStart)
                            print("[\(ts())] [KVCache] Timing: TTFT=\(String(format: "%.3f", ttft))s total=\(String(format: "%.3f", total))s (streaming)")
                        }

                        var saveTrimTime: Double? = nil
                        var saveTruncateTime: Double? = nil
                        var saveInsertTime: Double? = nil

                        // Save prompt cache state into radix tree
                        if useCache, let radix = self.radixCache, !inputTokens.isEmpty, !Task.isCancelled {
                            let promptLen = inputTokens.count
                            let tSave0 = Date.timeIntervalSinceReferenceDate
                            for layer in generationCache {
                                let excess = layer.offset - promptLen
                                if excess > 0 { layer.trim(excess) }
                            }
                            let tSaveTrim = Date.timeIntervalSinceReferenceDate
                            // Physically truncate trimmed cache arrays to eliminate stale data. (#47)
                            for i in 0..<generationCache.count {
                                if generationCache[i].isTrimmable && generationCache[i].offset > 0
                                    && self.supportsPhysicalTruncation(generationCache[i])
                                {
                                    generationCache[i].truncateToOffset()
                                }
                            }
                            let tSaveTruncate = Date.timeIntervalSinceReferenceDate
                            let layerStates = generationCache.map { $0.state }
                            let layerMetaStates = generationCache.map { $0.metaState }
                            radix.insert(
                                tokens: inputTokens,
                                layerStates: layerStates,
                                layerMetaStates: layerMetaStates
                            )
                            let tSaveInsert = Date.timeIntervalSinceReferenceDate
                            saveTrimTime = tSaveTrim - tSave0
                            saveTruncateTime = tSaveTruncate - tSaveTrim
                            saveInsertTime = tSaveInsert - tSaveTruncate
                            self.logCacheSave(
                                mode: "streaming",
                                inputTokenCount: inputTokens.count,
                                radixEntryCount: radix.count,
                                cache: generationCache,
                                trimTime: saveTrimTime,
                                truncateTime: saveTruncateTime,
                                insertTime: saveInsertTime
                            )
                            if debugLogging {
                                print("[\(ts())] [PrefixCache] Insert: \(inputTokens.count) tokens, \(generationCache.count) layers")
                            }
                        }
                        self.logCacheProfile(
                            phase: "save",
                            mode: "streaming",
                            outcome: useCache && !Task.isCancelled ? "save" : "skip",
                            inputTokenCount: inputTokens.count,
                            cachedTokenCount: streamCachedTokens,
                            promptTime: 0,
                            trimTime: saveTrimTime,
                            truncateTime: saveTruncateTime,
                            insertTime: saveInsertTime
                        )
                    }
                    self.cleanupTempFiles(mediaTempFiles)
                    continuation.finish()
                } catch {
                    if debugLogging {
                        print("[\(ts())] [PrefixCache] Invalidate: stream error")
                    }
                    self.radixCache?.invalidateAll()
                    self.cleanupTempFiles(mediaTempFiles)
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in
                task.cancel()
            }
        }

        // Derive tool call start/end tags for streaming detection
        let toolTags: (start: String, end: String)?
        if let tools, !tools.isEmpty {
            let format = withStateLock({ currentToolCallFormat })
            if let format {
                switch format {
                case .xmlFunction:
                    // XMLFunctionParser has nil tags; chat template wraps in <tool_call>
                    toolTags = ("<tool_call>", "</tool_call>")
                default:
                    let parser = format.createParser()
                    toolTags = (parser.startTag ?? "<tool_call>", parser.endTag ?? "</tool_call>")
                }
            } else {
                toolTags = ("<tool_call>", "</tool_call>")
            }
        } else {
            toolTags = nil
        }

        endOperationOnExit = false
        return (modelID, stream, promptTokens, toolTags?.start, toolTags?.end, self.thinkStartTag, self.thinkEndTag)
    }

    func shutdownAndReleaseResources(verbose: Bool = false, timeoutSeconds: TimeInterval = 30) async {
        // Shut down concurrent scheduler first (cancels pending + active)
        if let scheduler = self.scheduler {
            await scheduler.shutdown()
            self.scheduler = nil
        }

        let start = Date()
        withStateLock { isShuttingDown = true }

        while Date().timeIntervalSince(start) < timeoutSeconds {
            if withStateLock({ activeOperations == 0 }) {
                break
            }
            try? await Task.sleep(nanoseconds: 100_000_000)
        }

        if debugLogging {
            print("[\(ts())] [PrefixCache] Invalidate: cleanup")
        }
        self.radixCache?.invalidateAll()
        autoreleasepool {
            withStateLock {
                currentContainer = nil
                currentModelID = nil
            }
        }

        // Ensure queued GPU work is complete before clearing recycled buffers.
        // The BatchScheduler's decode loop synchronizes the GPU stream before
        // exiting on cancellation, so this should be safe.
        Stream.gpu.synchronize()
        Stream.cpu.synchronize()
        Memory.clearCache()
        Stream.gpu.synchronize()
        Memory.clearCache()

        if verbose {
            let snapshot = Memory.snapshot()
            print("MLX memory after shutdown - active: \(formatBytes(snapshot.activeMemory)), cache: \(formatBytes(snapshot.cacheMemory)), peak: \(formatBytes(snapshot.peakMemory))")
        }
    }

    // MARK: - Tool conversion helpers

    /// Convert OpenAI-format RequestTool array to vendor ToolSpec array.
    private func convertToToolSpecs(_ tools: [RequestTool]?) -> [ToolSpec]? {
        guard let tools, !tools.isEmpty else { return nil }
        return tools.map { tool -> ToolSpec in
            var funcDict: [String: any Sendable] = [
                "name": tool.function.name
            ]
            if let desc = tool.function.description {
                funcDict["description"] = desc
            }
            if let params = tool.function.parameters {
                funcDict["parameters"] = params.toJinjaCompatible()
            }
            return [
                "type": tool.type,
                "function": funcDict
            ]
        }
    }

    /// Convert a vendor ToolCall to an OpenAI-compatible ResponseToolCall.
    static func convertToolCall(_ tc: ToolCall, index: Int, paramNameMapping: [String: String] = [:]) -> ResponseToolCall {
        // Apply parameter name mapping (e.g. snake_case → camelCase) if provided.
        // Qwen3-Coder converts camelCase param names to snake_case in XML output.
        let argsDict: [String: Any]
        if paramNameMapping.isEmpty {
            argsDict = tc.function.arguments.mapValues { $0.anyValue }
        } else {
            var mapped = [String: Any]()
            for (key, value) in tc.function.arguments {
                let mappedKey = paramNameMapping[key] ?? key
                mapped[mappedKey] = value.anyValue
            }
            argsDict = mapped
        }
        let argsJSON: String
        if let data = try? JSONSerialization.data(withJSONObject: argsDict, options: [.sortedKeys]),
           let str = String(data: data, encoding: .utf8) {
            argsJSON = str
        } else {
            argsJSON = "{}"
        }
        // -VV: Log the serialized JSON that will be sent to the client
        if traceLogging {
            let preview = argsJSON.count > 500 ? "\(argsJSON.prefix(250))...\(argsJSON.suffix(250))" : argsJSON
            print("\(vvCyan)[\(ts())] [VV] SEND→CLIENT tool_call \(tc.function.name) args JSON (\(argsJSON.count) chars):\n\(preview)\(vvReset)")
            fflush(stdout)
        }
        return ResponseToolCall(
            index: index,
            id: "call_\(generateCallID())",
            type: "function",
            function: ResponseToolCallFunction(
                name: tc.function.name,
                arguments: argsJSON
            )
        )
    }

    static func normalizeToolCall(
        _ tc: ToolCall,
        index: Int,
        paramNameMapping: [String: String] = [:],
        tools: [RequestTool]? = nil,
        fixToolArgs: Bool = false
    ) -> ResponseToolCall {
        var converted = convertToolCall(tc, index: index, paramNameMapping: paramNameMapping)
        if fixToolArgs, let tools, !tools.isEmpty {
            converted = remapResponseToolCallArguments(converted, tools: tools)
        }
        return coerceArgumentTypes(converted, tools: tools)
    }

    static func normalizeToolCalls(
        _ toolCalls: [ToolCall],
        startIndex: Int = 0,
        paramNameMapping: [String: String] = [:],
        tools: [RequestTool]? = nil,
        fixToolArgs: Bool = false
    ) -> [ResponseToolCall] {
        toolCalls.enumerated().map { offset, toolCall in
            normalizeToolCall(
                toolCall,
                index: startIndex + offset,
                paramNameMapping: paramNameMapping,
                tools: tools,
                fixToolArgs: fixToolArgs
            )
        }
    }

    static func remapResponseToolCallArguments(_ rtc: ResponseToolCall, tools: [RequestTool]?) -> ResponseToolCall {
        guard let tools, !tools.isEmpty else { return rtc }
        guard let data = rtc.function.arguments.data(using: .utf8),
              let argsDict = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return rtc }
        var sendableArgs = [String: any Sendable]()
        for (key, value) in argsDict { sendableArgs[key] = value }
        let remapped = remapArgumentKeys(sendableArgs, toolName: rtc.function.name, tools: tools)
        let remappedAny = remapped.mapValues { $0 as Any }
        guard let newData = try? JSONSerialization.data(withJSONObject: remappedAny, options: [.sortedKeys]),
              let newString = String(data: newData, encoding: .utf8) else { return rtc }
        return ResponseToolCall(
            index: rtc.index,
            id: rtc.id,
            type: rtc.type,
            function: ResponseToolCallFunction(name: rtc.function.name, arguments: newString)
        )
    }

    /// Remap tool call argument keys to match the original tool schema.
    /// Heuristics (in priority order):
    /// 1. Exact match — key exists in schema → keep as-is
    /// 2. Case-insensitive match — e.g. "filepath" → "filePath"
    /// 3. Snake↔Camel match — e.g. "file_path" → "filePath" or "filePath" → "file_path"
    /// 4. Suffix match — e.g. "path" matches "filePath" (only if exactly one candidate)
    static func remapArgumentKeys(_ arguments: [String: any Sendable], toolName: String, tools: [RequestTool]) -> [String: any Sendable] {
        // Find the matching tool schema
        guard let tool = tools.first(where: { $0.function.name == toolName }),
              let paramsAny = tool.function.parameters?.toSendable() as? [String: Any],
              let props = paramsAny["properties"] as? [String: Any] else {
            return arguments
        }
        let schemaKeys = Array(props.keys)
        let schemaKeysLower = schemaKeys.map { $0.lowercased() }

        var remapped = [String: any Sendable]()
        for (key, value) in arguments {
            // 1. Exact match
            if props[key] != nil {
                remapped[key] = value
                continue
            }

            // 2. Case-insensitive match
            let keyLower = key.lowercased()
            if let idx = schemaKeysLower.firstIndex(of: keyLower) {
                let mapped = schemaKeys[idx]
                if debugLogging { print("[\(ts())] [ToolCallRemap] \(key) → \(mapped) (case-insensitive)") }
                remapped[mapped] = value
                continue
            }

            // 3. Snake↔Camel conversion
            // Try converting key from snake_case to camelCase
            let camelized = snakeToCamel(key)
            if camelized != key, props[camelized] != nil {
                if debugLogging { print("[\(ts())] [ToolCallRemap] \(key) → \(camelized) (snake→camel)") }
                remapped[camelized] = value
                continue
            }
            // Try converting key from camelCase to snake_case
            let snaked = camelToSnake(key)
            if snaked != key, props[snaked] != nil {
                if debugLogging { print("[\(ts())] [ToolCallRemap] \(key) → \(snaked) (camel→snake)") }
                remapped[snaked] = value
                continue
            }

            // 4. Suffix match — model's key is a suffix of exactly one schema key
            let suffixCandidates = schemaKeys.filter {
                $0.lowercased().hasSuffix(keyLower) && $0.count > key.count
            }
            if suffixCandidates.count == 1 {
                let mapped = suffixCandidates[0]
                if debugLogging { print("[\(ts())] [ToolCallRemap] \(key) → \(mapped) (suffix)") }
                remapped[mapped] = value
                continue
            }

            // No match — keep original key
            remapped[key] = value
        }
        return remapped
    }

    /// Convert snake_case to camelCase: "file_path" → "filePath"
    private static func snakeToCamel(_ s: String) -> String {
        let parts = s.split(separator: "_", omittingEmptySubsequences: false)
        guard parts.count > 1 else { return s }
        return String(parts[0]) + parts.dropFirst().map { $0.prefix(1).uppercased() + $0.dropFirst() }.joined()
    }

    /// Convert camelCase to snake_case: "filePath" → "file_path"
    private static func camelToSnake(_ s: String) -> String {
        var result = ""
        for (i, char) in s.enumerated() {
            if char.isUppercase {
                if i > 0 { result += "_" }
                result += char.lowercased()
            } else {
                result += String(char)
            }
        }
        return result
    }

    /// Generate a random alphanumeric ID for tool call IDs.
    private static func generateCallID() -> String {
        let chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        return String((0..<24).map { _ in chars.randomElement()! })
    }

    /// Coerce string argument values to match the tool's declared schema types.
    /// XML tool call parsers emit all values as strings; this converts "true" → true, "5" → 5, etc.
    /// Also fills in default values for missing required parameters (model omission fix).
    static func coerceArgumentTypes(_ rtc: ResponseToolCall, tools: [RequestTool]?) -> ResponseToolCall {
        guard let tools, !tools.isEmpty else { return rtc }
        guard let tool = tools.first(where: { $0.function.name == rtc.function.name }),
              let paramsAny = tool.function.parameters?.toSendable() as? [String: Any],
              let props = paramsAny["properties"] as? [String: Any] else { return rtc }
        guard let data = rtc.function.arguments.data(using: .utf8),
              var argsDict = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return rtc }

        var changed = false

        // Type coercion: "5" → 5, "true" → true, etc.
        for (key, value) in argsDict {
            guard let stringValue = value as? String,
                  let propSchema = props[key] as? [String: Any],
                  let schemaType = propSchema["type"] as? String else { continue }
            if let coerced = coerceStringValue(stringValue, schemaType: schemaType) {
                if debugLogging {
                    print("[\(ts())] [ToolCallParser] coerce \(rtc.function.name).\(key): \"\(stringValue)\" → \(coerced) (schema: \(schemaType))")
                }
                argsDict[key] = coerced
                changed = true
            }
        }

        // Fill missing required parameters with type-appropriate defaults (recursive).
        // Models (especially Qwen3-Coder) often omit required params like "description"
        // which causes downstream validation errors ("expected string, received undefined").
        // This walks the full schema tree: top-level object → array items → nested objects.
        if fillMissingRequired(&argsDict, schema: paramsAny, toolName: rtc.function.name) {
            changed = true
        }

        guard changed,
              let newData = try? JSONSerialization.data(withJSONObject: argsDict, options: [.sortedKeys]),
              let newStr = String(data: newData, encoding: .utf8) else { return rtc }
        // -VV: Log coercion result
        if traceLogging {
            print("\(vvCyan)[\(ts())] [VV] COERCED \(rtc.function.name):\n  before: \(rtc.function.arguments.prefix(300))\n  after:  \(newStr.prefix(300))\(vvReset)")
            fflush(stdout)
        }
        return ResponseToolCall(
            index: rtc.index,
            id: rtc.id, type: rtc.type,
            function: ResponseToolCallFunction(name: rtc.function.name, arguments: newStr)
        )
    }

    /// Coerce a single string value to the schema-declared type.
    static func coerceStringValue(_ stringValue: String, schemaType: String) -> Any? {
        switch schemaType {
        case "integer":
            return Int(stringValue)
        case "number":
            if let d = Double(stringValue) {
                let i = Int(d)
                return d == Double(i) ? i : d
            }
            return nil
        case "boolean":
            switch stringValue.lowercased() {
            case "true": return true
            case "false": return false
            default: return nil
            }
        case "array", "object":
            if let jsonData = stringValue.data(using: .utf8),
               let parsed = try? JSONSerialization.jsonObject(with: jsonData) {
                return parsed
            }
            return nil
        default:
            return nil
        }
    }

    /// Recursively fill missing required fields in a JSON object according to its schema.
    /// Walks: object → check required keys; array → iterate items; nested objects → recurse.
    /// Returns true if any fields were filled.
    @discardableResult
    private static func fillMissingRequired(_ dict: inout [String: Any], schema: [String: Any], toolName: String, path: String = "") -> Bool {
        guard let props = schema["properties"] as? [String: Any] else { return false }
        var filled = false

        // 1. Fill missing required keys at this level
        if let required = schema["required"] as? [String] {
            for key in required where dict[key] == nil {
                if let propSchema = props[key] as? [String: Any],
                   let schemaType = propSchema["type"] as? String {
                    let fullPath = path.isEmpty ? key : "\(path).\(key)"
                    switch schemaType {
                    case "string":  // Don't fill strings
                        print("[\(ts())] [ToolCallParser] Missing required string param '\(fullPath)' for \(toolName) — not filling")
                    case "boolean": dict[key] = false; filled = true
                        print("[\(ts())] [ToolCallParser] Filled missing required param '\(fullPath)' (\(schemaType)) with default for \(toolName)")
                    case "integer", "number": dict[key] = 0; filled = true
                        print("[\(ts())] [ToolCallParser] Filled missing required param '\(fullPath)' (\(schemaType)) with default for \(toolName)")
                    case "array":   dict[key] = [Any](); filled = true
                        print("[\(ts())] [ToolCallParser] Filled missing required param '\(fullPath)' (\(schemaType)) with default for \(toolName)")
                    case "object":  dict[key] = [String: Any](); filled = true
                        print("[\(ts())] [ToolCallParser] Filled missing required param '\(fullPath)' (\(schemaType)) with default for \(toolName)")
                    default: break
                    }
                }
            }
        }

        // 2. Recurse into existing values that have nested schemas
        for (key, propSchema) in props {
            guard let propDict = propSchema as? [String: Any],
                  let propType = propDict["type"] as? String else { continue }
            let childPath = path.isEmpty ? key : "\(path).\(key)"

            if propType == "object", var nested = dict[key] as? [String: Any] {
                // Recurse into nested object
                if fillMissingRequired(&nested, schema: propDict, toolName: toolName, path: childPath) {
                    dict[key] = nested
                    filled = true
                }
            } else if propType == "array", let itemSchema = propDict["items"] as? [String: Any],
                      itemSchema["type"] as? String == "object",
                      var arr = dict[key] as? [[String: Any]] {
                // Recurse into each array item
                for i in 0..<arr.count {
                    let itemPath = "\(childPath)[\(i)]"
                    if fillMissingRequired(&arr[i], schema: itemSchema, toolName: toolName, path: itemPath) {
                        filled = true
                    }
                }
                if filled { dict[key] = arr }
            }
        }

        return filled
    }

    /// Fallback tool call extraction for formats the vendor parser misses.
    /// Handles <tool_call><function=name><parameter=key>value</parameter></function></tool_call>
    /// and <tool_call>{"name":"func","arguments":{...}}</tool_call> patterns.
    /// Returns extracted ToolCalls and remaining non-tool-call content.
    static func extractToolCallsFallback(from text: String) -> ([ToolCall], String) {
        var toolCalls = [ToolCall]()
        var remaining = text

        // Match <tool_call>...</tool_call> blocks (dotMatchesLineSeparators for multiline)
        let toolCallRegex = try! NSRegularExpression(
            pattern: #"<tool_call>\s*(.*?)\s*</tool_call>"#,
            options: [.dotMatchesLineSeparators]
        )
        let matches = toolCallRegex.matches(in: text, range: NSRange(text.startIndex..., in: text))

        if debugLogging && !matches.isEmpty {
            print("[\(ts())] [ToolCallParser] extractToolCallsFallback: found \(matches.count) <tool_call> block(s)")
        }

        for match in matches.reversed() {
            guard let innerRange = Range(match.range(at: 1), in: text) else { continue }
            let inner = String(text[innerRange])

            // Try XML function format: <function=name><parameter=key>value</parameter></function>
            if let tc = parseXMLFunction(inner) {
                toolCalls.insert(tc, at: 0)
                if let fullRange = Range(match.range, in: remaining) {
                    remaining.removeSubrange(fullRange)
                }
                continue
            }

            // Try hybrid format: <function=NAME, "arguments": {JSON}}
            // Some models mix XML function tags with inline JSON arguments.
            if let tc = parseXMLFunctionWithEmbeddedJSON(inner) {
                if debugLogging {
                    print("[\(ts())] [ToolCallParser] XML+embedded-JSON: \(tc.function.name)(\(tc.function.arguments.keys.joined(separator: ", ")))")
                }
                toolCalls.insert(tc, at: 0)
                if let fullRange = Range(match.range, in: remaining) {
                    remaining.removeSubrange(fullRange)
                }
                continue
            }

            // Try JSON format: {"name":"func","arguments":{...}}
            if let tc = parseJSONToolCall(inner) {
                if debugLogging {
                    print("[\(ts())] [ToolCallParser] JSON-in-XML: \(tc.function.name)(\(tc.function.arguments.keys.joined(separator: ", ")))")
                }
                toolCalls.insert(tc, at: 0)
                if let fullRange = Range(match.range, in: remaining) {
                    remaining.removeSubrange(fullRange)
                }
            }
        }

        // Fallback: Mistral models may emit [TOOL_CALLS] in various formats
        if toolCalls.isEmpty {
            let trimmed = remaining.trimmingCharacters(in: .whitespacesAndNewlines)
            if trimmed.hasPrefix("[TOOL_CALLS]") {
                if debugLogging {
                    print("[\(ts())] [ToolCallParser] Trying Mistral [TOOL_CALLS] format")
                }
                let afterPrefix = String(trimmed.dropFirst("[TOOL_CALLS]".count)).trimmingCharacters(in: .whitespacesAndNewlines)

                // Format 1: [TOOL_CALLS][{"name":"func","arguments":{...}}]
                if let data = afterPrefix.data(using: .utf8),
                   let arr = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]] {
                    for item in arr {
                        if let tc = parseJSONToolCall(String(data: (try? JSONSerialization.data(withJSONObject: item)) ?? Data(), encoding: .utf8) ?? "{}") {
                            toolCalls.append(tc)
                        }
                    }
                    if !toolCalls.isEmpty {
                        remaining = ""
                    }
                }

                // Format 2: [TOOL_CALLS]func_name[ARGS]{"key":"value"}
                if toolCalls.isEmpty {
                    let argsPattern = try! NSRegularExpression(
                        pattern: #"([a-zA-Z_][a-zA-Z0-9_]*)\[ARGS\](\{[\s\S]*?\})(?:\s|$)"#,
                        options: [])
                    let matches = argsPattern.matches(in: afterPrefix, range: NSRange(afterPrefix.startIndex..., in: afterPrefix))
                    for match in matches {
                        if let nameRange = Range(match.range(at: 1), in: afterPrefix),
                           let argsRange = Range(match.range(at: 2), in: afterPrefix) {
                            let name = String(afterPrefix[nameRange])
                            let argsStr = String(afterPrefix[argsRange])
                            if let argsData = argsStr.data(using: .utf8),
                               let argsDict = try? JSONSerialization.jsonObject(with: argsData) as? [String: Any] {
                                var args: [String: String] = [:]
                                for (k, v) in argsDict {
                                    args[k] = "\(v)"
                                }
                                toolCalls.append(ToolCall(function: .init(name: name, arguments: args)))
                            }
                        }
                    }
                    if !toolCalls.isEmpty {
                        remaining = ""
                    }
                }
            }
        }

        // Fallback: bare NAME[ARGS]{JSON} without [TOOL_CALLS] prefix
        // Some models (e.g. Ministral) emit the Mistral format without the prefix.
        if toolCalls.isEmpty {
            let trimmed = remaining.trimmingCharacters(in: .whitespacesAndNewlines)
            if trimmed.contains("[ARGS]") {
                let argsPattern = try! NSRegularExpression(
                    pattern: #"([a-zA-Z_][a-zA-Z0-9_]*)\[ARGS\](\{[\s\S]*?\})(?:\s|$|\[)"#,
                    options: [])
                let matches = argsPattern.matches(in: trimmed, range: NSRange(trimmed.startIndex..., in: trimmed))
                for match in matches {
                    if let nameRange = Range(match.range(at: 1), in: trimmed),
                       let argsRange = Range(match.range(at: 2), in: trimmed) {
                        let name = String(trimmed[nameRange])
                        let argsStr = String(trimmed[argsRange])
                        if let argsData = argsStr.data(using: .utf8),
                           let argsDict = try? JSONSerialization.jsonObject(with: argsData) as? [String: Any] {
                            var args: [String: String] = [:]
                            for (k, v) in argsDict {
                                args[k] = "\(v)"
                            }
                            toolCalls.append(ToolCall(function: .init(name: name, arguments: args)))
                        }
                    }
                }
                if !toolCalls.isEmpty {
                    remaining = ""
                }
            }
        }

        // Fallback: bare <function=...></function> without <tool_call> wrapper
        // Some models (e.g. Qwen3-Coder-Next) emit the XML function block directly,
        // sometimes with a trailing </tool_call> but no opening <tool_call>.
        if toolCalls.isEmpty {
            let funcRegex = try! NSRegularExpression(
                pattern: #"<function=([^>]+)>(.*?)</function>"#,
                options: [.dotMatchesLineSeparators]
            )
            let funcMatches = funcRegex.matches(in: remaining, range: NSRange(remaining.startIndex..., in: remaining))
            for match in funcMatches.reversed() {
                let fullContent = String(remaining[Range(match.range, in: remaining)!])
                if let tc = parseXMLFunction(fullContent) {
                    if debugLogging {
                        print("[\(ts())] [ToolCallParser] Bare XML function: \(tc.function.name)(\(tc.function.arguments.keys.joined(separator: ", ")))")
                    }
                    toolCalls.insert(tc, at: 0)
                    if let fullRange = Range(match.range, in: remaining) {
                        remaining.removeSubrange(fullRange)
                    }
                }
            }
            // Clean up orphaned </tool_call> tags
            if !toolCalls.isEmpty {
                remaining = remaining.replacingOccurrences(of: "</tool_call>", with: "")
            }
        }

        // Fallback: bare JSON tool call (no wrapper tags)
        // e.g. {"name":"get_weather","arguments":{"city":"Tokyo"}} or with "parameters"
        if toolCalls.isEmpty {
            let trimmed = remaining.trimmingCharacters(in: .whitespacesAndNewlines)
            if trimmed.hasPrefix("{") && trimmed.hasSuffix("}") {
                if let tc = parseJSONToolCall(trimmed) {
                    if debugLogging {
                        print("[\(ts())] [ToolCallParser] Bare JSON: \(tc.function.name)(\(tc.function.arguments.keys.joined(separator: ", ")))")
                    }
                    toolCalls.append(tc)
                    remaining = ""
                }
            }
        }

        // Trim leftover whitespace/think tags from remaining
        remaining = remaining
            .replacingOccurrences(of: #"<think>\s*</think>"#, with: "", options: .regularExpression)
            .trimmingCharacters(in: .whitespacesAndNewlines)

        if debugLogging && !toolCalls.isEmpty {
            let names = toolCalls.map { "\($0.function.name)(\($0.function.arguments.count) args)" }.joined(separator: ", ")
            print("[\(ts())] [ToolCallParser] extractToolCallsFallback result: \(toolCalls.count) tool call(s) → \(names)")
        }

        return (toolCalls, remaining)
    }

    /// Parse <function=name><parameter=key>value</parameter></function> via regex.
    /// NSXMLParser silently drops parameters when values contain bare < or & (common in code),
    /// so we go straight to regex which handles arbitrary content correctly.
    private static func parseXMLFunction(_ content: String) -> ToolCall? {
        var funcName: String?
        var normalized = content

        // Standard: <function=NAME>
        let funcNameRegex = try! NSRegularExpression(
            pattern: #"<function=([^>]+)>"#, options: []
        )
        if let nameMatch = funcNameRegex.firstMatch(in: content, range: NSRange(content.startIndex..., in: content)),
           let nameRange = Range(nameMatch.range(at: 1), in: content) {
            funcName = String(content[nameRange])
        }

        // Hybrid JSON/XML: {"name": "NAME"> or {"name="NAME">
        // Qwen3.5-27B generates this format without grammar constraints.
        // Rewrite to standard <function=NAME> so parseXMLFunctionRegex handles it.
        if funcName == nil {
            let hybridRegex = try! NSRegularExpression(
                pattern: #"\{["\s]*name["\s]*[:=]\s*"?([a-zA-Z_][a-zA-Z0-9_]*)"?\s*>"#, options: []
            )
            if let match = hybridRegex.firstMatch(in: content, range: NSRange(content.startIndex..., in: content)),
               let nameRange = Range(match.range(at: 1), in: content),
               let fullRange = Range(match.range, in: content) {
                funcName = String(content[nameRange])
                normalized = content.replacingCharacters(in: fullRange, with: "<function=\(funcName!)>")
                if debugLogging {
                    print("[\(ts())] [ToolCallParser] hybrid JSON/XML opener rewritten for \(funcName!)")
                }
            }
        }

        guard let funcName else { return nil }

        if debugLogging {
            print("[\(ts())] [ToolCallParser] parseXMLFunction: \(funcName) (\(content.count) chars)")
        }
        return parseXMLFunctionRegex(normalized, funcName: funcName)
    }

    /// Parse hybrid format: `<function=NAME", "arguments": {JSON}}` or `<function=NAME, "arguments": {JSON}>`
    /// Models sometimes mix XML function tags with inline JSON arguments instead of `<parameter>` tags.
    private static func parseXMLFunctionWithEmbeddedJSON(_ content: String) -> ToolCall? {
        // Match <function=NAME followed by optional quote, comma, then "arguments": {JSON}
        let regex = try! NSRegularExpression(
            pattern: #"<function=([a-zA-Z_][a-zA-Z0-9_]*)["']?[,\s]*"arguments"\s*:\s*(\{.*\})"#,
            options: [.dotMatchesLineSeparators]
        )
        guard let match = regex.firstMatch(in: content, range: NSRange(content.startIndex..., in: content)),
              let nameRange = Range(match.range(at: 1), in: content),
              let argsRange = Range(match.range(at: 2), in: content) else {
            return nil
        }
        let name = String(content[nameRange])
        var argsStr = String(content[argsRange])
        // Greedy regex may capture trailing }} — trim until valid JSON.
        var parsed: [String: Any]?
        for _ in 0..<3 {
            if let data = argsStr.data(using: .utf8),
               let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                parsed = obj
                break
            }
            if argsStr.hasSuffix("}") {
                argsStr = String(argsStr.dropLast())
            } else {
                break
            }
        }
        guard let parsed else { return nil }
        var arguments: [String: String] = [:]
        for (k, v) in parsed {
            if let s = v as? String {
                arguments[k] = s
            } else if let data = try? JSONSerialization.data(withJSONObject: v),
                      let s = String(data: data, encoding: .utf8) {
                arguments[k] = s
            } else {
                arguments[k] = "\(v)"
            }
        }
        return ToolCall(function: .init(name: name, arguments: arguments))
    }

    /// Regex-based XML function parser with entity decoding.
    private static func parseXMLFunctionRegex(_ content: String, funcName: String) -> ToolCall? {
        let funcRegex = try! NSRegularExpression(
            pattern: #"<function=([^>]+)>(.*?)</function>"#,
            options: [.dotMatchesLineSeparators]
        )
        guard let funcMatch = funcRegex.firstMatch(in: content, range: NSRange(content.startIndex..., in: content)),
              let bodyRange = Range(funcMatch.range(at: 2), in: content) else {
            return nil
        }
        let body = String(content[bodyRange])
        // -VV: Log raw XML body exactly as model generated it
        if traceLogging {
            print("\(vvCyan)[\(ts())] [VV] RECV←MODEL raw XML body for \(funcName) (\(body.count) chars):\n\(body)\(vvReset)")
            fflush(stdout)
        }

        var arguments: [String: any Sendable] = [:]
        let paramRegex = try! NSRegularExpression(
            pattern: #"<parameter=([^>]+)>(.*?)</parameter>"#,
            options: [.dotMatchesLineSeparators]
        )
        let paramMatches = paramRegex.matches(in: body, range: NSRange(body.startIndex..., in: body))
        for pm in paramMatches {
            guard let keyRange = Range(pm.range(at: 1), in: body),
                  let valRange = Range(pm.range(at: 2), in: body) else { continue }
            let key = String(body[keyRange])
            let val = decodeJSONEscapes(decodeXMLEntities(String(body[valRange]).trimmingCharacters(in: .whitespacesAndNewlines)))
            if !val.isEmpty, arguments[key] == nil {
                // Preserve JSON arrays/objects as structured data
                if let data = val.data(using: .utf8),
                   let parsed = try? JSONSerialization.jsonObject(with: data),
                   (parsed is [Any] || parsed is [String: Any]) {
                    arguments[key] = parsed
                } else {
                    arguments[key] = val
                }
            }
        }

        // Salvage unclosed parameters (e.g. model hit max_tokens mid-content)
        let unclosedRegex = try! NSRegularExpression(
            pattern: #"<parameter=([^>]+)>([\s\S]+)$"#,
            options: []
        )
        if let unclosedMatch = unclosedRegex.firstMatch(in: body, range: NSRange(body.startIndex..., in: body)),
           let keyRange = Range(unclosedMatch.range(at: 1), in: body),
           let valRange = Range(unclosedMatch.range(at: 2), in: body) {
            let key = String(body[keyRange])
            if arguments[key] == nil {
                var val = String(body[valRange]).trimmingCharacters(in: .whitespacesAndNewlines)
                if let funcEnd = val.range(of: "</function>") {
                    val = String(val[..<funcEnd.lowerBound]).trimmingCharacters(in: .whitespacesAndNewlines)
                }
                let decoded = decodeJSONEscapes(decodeXMLEntities(val))
                if !decoded.isEmpty {
                    if let data = decoded.data(using: .utf8),
                       let parsed = try? JSONSerialization.jsonObject(with: data),
                       (parsed is [Any] || parsed is [String: Any]) {
                        arguments[key] = parsed
                    } else {
                        arguments[key] = decoded
                    }
                    if debugLogging {
                        print("[\(ts())] [ToolCallParser] Salvaged unclosed parameter '\(key)' (\(decoded.count) chars)")
                    }
                }
            }
        }

        // Cross-parameter deduplication: strip JSON fragments leaked into parameter
        // values when the same key+value exists as a separate <parameter> tag.
        // Only when grammar constraints are active (XML structure is reliable).
        if grammarConstraintsActive && arguments.count > 1 {
            let otherKeys = arguments.keys
            for key in otherKeys {
                guard var val = arguments[key] as? String else { continue }
                var cleaned = false
                // Check if this value contains JSON fragments referencing other parsed params
                for otherKey in otherKeys where otherKey != key {
                    guard let otherVal = arguments[otherKey] as? String else { continue }
                    // Look for patterns like: ", "otherKey": "otherVal"\n} or ","otherKey":"otherVal"}}
                    // at the end of the value (with flexible whitespace/punctuation)
                    let escaped = NSRegularExpression.escapedPattern(for: otherKey)
                    let escapedVal = NSRegularExpression.escapedPattern(for: otherVal)
                    let trailPattern = #"\n?"?\s*,?\s*"?"# + escaped + #""?\s*:\s*"?"# + escapedVal + #""?\s*\n?\}*\s*$"#
                    if let trailRegex = try? NSRegularExpression(pattern: trailPattern, options: []),
                       let match = trailRegex.firstMatch(in: val, range: NSRange(val.startIndex..., in: val)) {
                        let cleanEnd = val.index(val.startIndex, offsetBy: match.range.location)
                        val = String(val[..<cleanEnd]).trimmingCharacters(in: .whitespacesAndNewlines)
                        cleaned = true
                    }
                }
                if cleaned {
                    arguments[key] = val
                    if debugLogging {
                        let preview = val.count > 100 ? "\(val.prefix(50))...\(val.suffix(50))" : val
                        print("[\(ts())] [ToolCallParser] Cross-param dedup cleaned '\(key)' → \(val.count) chars: \(preview)")
                    }
                }
            }
        }

        // -VV: Log each parsed parameter value (post-decode)
        if traceLogging {
            for (key, value) in arguments {
                let valStr = (value as? String) ?? String(describing: value)
                let preview = valStr.count > 200 ? "\(valStr.prefix(100))...\(valStr.suffix(100))" : valStr
                print("\(vvCyan)[\(ts())] [VV] PARSED param \(funcName).\(key) = \(preview) (\(valStr.count) chars)\(vvReset)")
            }
            fflush(stdout)
        }

        return ToolCall(function: .init(name: funcName, arguments: arguments))
    }

    /// Decode the five standard XML entities in a string value.
    static func decodeXMLEntities(_ s: String) -> String {
        guard s.contains("&") else { return s }
        return s
            .replacingOccurrences(of: "&lt;", with: "<")
            .replacingOccurrences(of: "&gt;", with: ">")
            .replacingOccurrences(of: "&amp;", with: "&")
            .replacingOccurrences(of: "&quot;", with: "\"")
            .replacingOccurrences(of: "&apos;", with: "'")
    }

    /// Undo JSON-style escape sequences in XML parameter values.
    /// Some models (e.g. Qwen3-Coder) emit edit oldString/newString with literal `\n`, `\"`, `\t`
    /// instead of real newlines and quotes — the model pre-escapes content as if generating JSON.
    /// Detect this: if the value has NO real newlines but contains literal `\n`, unescape.
    static func decodeJSONEscapes(_ s: String) -> String {
        // Only activate when the value looks like it was JSON-escaped:
        // no real newlines but has literal \n sequences
        guard !s.contains("\n"), s.contains("\\n") || s.contains("\\\"") else { return s }
        if traceLogging {
            print("\(vvCyan)[\(ts())] [VV] decodeJSONEscapes: unescaping \\n/\\\" in \(s.count)-char value (no real newlines detected)\(vvReset)")
            fflush(stdout)
        }
        return s
            .replacingOccurrences(of: "\\\\", with: "\u{0000}")  // protect real backslashes first
            .replacingOccurrences(of: "\\\"", with: "\"")
            .replacingOccurrences(of: "\\n", with: "\n")
            .replacingOccurrences(of: "\\t", with: "\t")
            .replacingOccurrences(of: "\\r", with: "\r")
            .replacingOccurrences(of: "\u{00000}", with: "\\")   // restore real backslashes
    }

    /// Parse {"name":"func","arguments":{...}} JSON tool call
    private static func parseJSONToolCall(_ content: String) -> ToolCall? {
        guard let data = content.trimmingCharacters(in: .whitespacesAndNewlines).data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let name = json["name"] as? String else {
            return nil
        }
        var arguments: [String: any Sendable] = [:]
        if let args = (json["arguments"] as? [String: Any]) ?? (json["parameters"] as? [String: Any]) {
            for (k, v) in args {
                arguments[k] = v
            }
        }
        return ToolCall(function: .init(name: name, arguments: arguments))
    }

    // MARK: - Private helpers

    private func beginOperation() throws {
        try withStateLock {
            if isShuttingDown {
                throw MLXServiceError.serviceShuttingDown
            }
            activeOperations += 1
        }
    }

    private func endOperation() {
        withStateLock {
            activeOperations = max(0, activeOperations - 1)
        }
    }

    private func withStateLock<T>(_ body: () throws -> T) rethrows -> T {
        stateLock.lock()
        defer { stateLock.unlock() }
        return try body()
    }

    private func formatBytes(_ bytes: Int) -> String {
        let gb = Double(bytes) / 1_073_741_824.0
        return String(format: "%.2f GB", gb)
    }

    /// Resolve the HF hub cache directory, respecting env vars.
    /// Used by both downloadModel() and the resolver to ensure they agree on the path.
    static func resolveHFHubCache() -> URL {
        let env = ProcessInfo.processInfo.environment
        if let val = env["HF_HUB_CACHE"] ?? env["HUGGINGFACE_HUB_CACHE"],
           !val.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return URL(fileURLWithPath: NSString(string: val).expandingTildeInPath)
        }
        if let val = env["HF_HOME"], !val.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return URL(fileURLWithPath: NSString(string: val).expandingTildeInPath)
                .appendingPathComponent("hub")
        }
        return FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache/huggingface/hub")
    }

    private func downloadModel(modelID: String, progress: (@Sendable (Progress) -> Void)?) async throws {
        guard let repoID = HuggingFace.Repo.ID(rawValue: modelID) else {
            throw MLXServiceError.invalidModel(modelID)
        }
        do {
            // Download to HF hub cache (HF-style layout, shared with Python mlx_lm).
            let cacheDir = Self.resolveHFHubCache()
            let cache = HubCache(cacheDirectory: cacheDir)
            print("Download destination: \(cacheDir.path)")
            let client = HuggingFace.HubClient(cache: cache)
            // No @MainActor progress handler — it deadlocks in single-prompt mode
            // because MainActor is suspended waiting for downloadSnapshot to return.
            // The spinner is driven by MLXLoadReporter independently.
            _ = try await client.downloadSnapshot(
                of: repoID,
                matching: ["*.json", "*.jinja", "*.safetensors", "*.txt", "*.model", "*.tiktoken", "tokenizer*", "*.bpe", "*.bin"]
            )
        } catch {
            throw MLXServiceError.downloadFailed("\(modelID): \(error.localizedDescription)")
        }
    }

    private func inferToolCallFormat(directory: URL) -> ToolCallFormat? {
        let configURL = directory.appendingPathComponent("config.json")
        guard let data = try? Data(contentsOf: configURL),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let modelType = json["model_type"] as? String else {
            return nil
        }
        return ToolCallFormat.infer(from: modelType)
    }

    /// Read `vocab_size` from a model's config.json.
    private func readVocabSize(directory: URL) -> Int? {
        let configURL = directory.appendingPathComponent("config.json")
        guard let data = try? Data(contentsOf: configURL),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return nil
        }
        // Try top-level first, then text_config (VLM models)
        if let vs = json["vocab_size"] as? Int { return vs }
        if let textConfig = json["text_config"] as? [String: Any],
           let vs = textConfig["vocab_size"] as? Int { return vs }
        return nil
    }

    /// Check whether any tool in the request has `strict: true`.
    static func hasStrictTools(_ tools: [RequestTool]?) -> Bool {
        tools?.contains { $0.function.strict == true } ?? false
    }

    /// Check whether a response_format has json_schema with strict: true.
    static func hasStrictSchema(_ responseFormat: ResponseFormat?) -> Bool {
        responseFormat?.type == "json_schema" && responseFormat?.jsonSchema?.strict == true
    }

    /// Set up grammar-constrained decoding based on strict × CLI policy.
    /// Returns a ConstrainedDecodingSetup on success, nil on failure (falls back to prompt injection).
    /// Policy: strict: true is the per-request opt-in, --enable-grammar-constraints is the admin opt-in.
    private func setupConstrainedDecodingProcessor(
        modelID: String,
        responseFormat: ResponseFormat?,
        tokenizer: any Tokenizer,
        tools: [RequestTool]?
    ) -> ConstrainedDecodingSetup? {
        let wantStrictSchema = Self.hasStrictSchema(responseFormat) && self.enableGrammarConstraints
        let wantStrictTools = Self.hasStrictTools(tools) && self.enableGrammarConstraints

        if wantStrictSchema {
            guard let processor = setupGrammarConstraint(
                modelID: modelID,
                responseFormat: responseFormat,
                tokenizer: tokenizer
            ) else {
                return nil
            }
            return ConstrainedDecodingSetup(
                processor: processor,
                mode: "json_schema",
                matcherHandle: processor.matcherHandle as? GrammarMatcherHandle
            )
        }
        if wantStrictTools {
            guard let processor = setupToolCallGrammarConstraint(
                modelID: modelID,
                tokenizer: tokenizer,
                tools: tools
            ) else {
                return nil
            }
            return ConstrainedDecodingSetup(
                processor: processor,
                mode: "tool_call_grammar",
                matcherHandle: processor.matcherHandle as? GrammarMatcherHandle
            )
        }

        // Observability: warn when strict requested but engine not enabled
        if !self.enableGrammarConstraints {
            if Self.hasStrictSchema(responseFormat) {
                print("[\(ts())] [XGrammar] strict: true on json_schema but --enable-grammar-constraints not set; best-effort")
            }
            if Self.hasStrictTools(tools) {
                print("[\(ts())] [XGrammar] strict: true on tools but --enable-grammar-constraints not set; best-effort")
            }
        }

        return nil
    }

    /// Set up grammar-constrained decoding for a json_schema response format.
    /// Returns a GrammarLogitProcessor on success, nil on failure (falls back to prompt injection).
    private func setupGrammarConstraint(
        modelID: String,
        responseFormat: ResponseFormat?,
        tokenizer: any Tokenizer
    ) -> GrammarLogitProcessor? {
        guard let responseFormat, responseFormat.type == "json_schema",
              let schema = responseFormat.jsonSchema?.schema else {
            return nil
        }

        do {
            // Initialize service if needed
            if xgrammarService == nil {
                guard let directory = resolver.localModelDirectory(repoId: modelID) else {
                    if debugLogging { print("[\(ts())] [XGrammar] Model directory not found") }
                    return nil
                }
                let vocabSize = readVocabSize(directory: directory) ?? 151936
                let service = XGrammarService(vocabSize: vocabSize, debugLogging: debugLogging)
                let eosId = tokenizer.eosTokenId
                try service.setupTokenizer(tokenizer: tokenizer, eosTokenId: eosId)
                self.xgrammarService = service
            }

            guard let service = xgrammarService else { return nil }

            // Convert schema to JSON string
            let schemaValue = schema.toSendable()
            let schemaData = try JSONSerialization.data(withJSONObject: schemaValue)
            let schemaJSON = String(data: schemaData, encoding: .utf8) ?? "{}"

            // Compile and create matcher
            let matcher = try service.compileAndCreateMatcher(schemaJSON: schemaJSON)

            // Create processor
            let proc = GrammarLogitProcessor()
            proc.matcherHandle = matcher
            proc.tokenMask = matcher.nextTokenMask()
            proc.onTokenSampled = { [weak matcher] tokenID in
                guard let matcher, !matcher.isTerminated() else {
                    proc.tokenMask = nil
                    return
                }
                matcher.acceptToken(tokenID)
                proc.tokenMask = matcher.nextTokenMask()
            }

            if debugLogging {
                print("[\(ts())] [XGrammar] Grammar constraint active for json_schema (native C++, vocab_size=\(service.vocabSize))")
            }

            return proc
        } catch {
            if debugLogging {
                print("[\(ts())] [XGrammar] Failed to set up grammar: \(error). Falling back to prompt injection.")
            }
            return nil
        }
    }

    /// Set up grammar-constrained decoding for XML tool call format (afm_adaptive_xml).
    /// Uses xgrammar EBNF to force valid <tool_call><function=...> structure.
    /// Grammar enforces minimum required parameter count per tool from the schema.
    private func setupToolCallGrammarConstraint(
        modelID: String,
        tokenizer: any Tokenizer,
        tools: [RequestTool]?
    ) -> GrammarLogitProcessor? {
        guard let tools, !tools.isEmpty else { return nil }

        do {
            // Initialize xgrammar service if needed (same pattern as json_schema path)
            if xgrammarService == nil {
                guard let directory = resolver.localModelDirectory(repoId: modelID) else {
                    if debugLogging { print("[\(ts())] [XGrammar] Model directory not found") }
                    return nil
                }
                let vocabSize = readVocabSize(directory: directory) ?? 151936
                let service = XGrammarService(vocabSize: vocabSize, debugLogging: debugLogging)
                let eosId = tokenizer.eosTokenId
                try service.setupTokenizer(tokenizer: tokenizer, eosTokenId: eosId)
                self.xgrammarService = service
            }

            guard let service = xgrammarService else { return nil }

            // Build EBNF grammar with literal tool names (llama.cpp approach).
            // Reasoner gating suspends the grammar during <think>...</think>,
            // so the free_text rule doesn't need to handle think tags.
            let ebnfGrammar = Self.buildToolCallEBNF(tools: tools)
            if debugLogging || trace {
                print("\(vvCyan)[\(ts())] [VV] SEND→MODEL grammar (EBNF):\n\(ebnfGrammar)\(vvReset)")
            }

            let matcher: GrammarMatcherHandle
            do {
                matcher = try service.compileAndCreateMatcherFromEBNF(grammar: ebnfGrammar)
            } catch {
                // Fallback to structural tag if EBNF compilation fails
                if debugLogging {
                    print("[\(ts())] [XGrammar] EBNF failed (\(error)), falling back to structural tag")
                }
                let structuralTagJSON = Self.buildToolCallStructuralTag(tools: tools)
                matcher = try service.compileAndCreateMatcherFromStructuralTag(json: structuralTagJSON)
            }

            let proc = GrammarLogitProcessor()
            proc.matcherHandle = matcher

            // vLLM-style reasoner gating: disable grammar during <think>...</think>
            // The grammar only applies AFTER reasoning ends (</think> detected).
            let thinkTokenId = tokenizer.convertTokenToId("<think>")
            let endThinkTokenId = tokenizer.convertTokenToId("</think>")
            var inReasoning = false
            let dbg = debugLogging
            let vvTrace = self.trace
            let vvTokenizer = tokenizer

            // Initial mask: nil (model may start with <think>, so don't constrain yet)
            // If model's first token is NOT <think>, we enable the grammar in onTokenSampled.
            proc.tokenMask = nil

            var firstToken = true
            var grammarTokenCount = 0
            proc.onTokenSampled = { [weak matcher] tokenID in
                guard let matcher else {
                    proc.tokenMask = nil
                    return
                }
                grammarTokenCount += 1

                // Track reasoning state
                if firstToken {
                    firstToken = false
                    if tokenID == thinkTokenId {
                        inReasoning = true
                        if dbg { print("[\(ts())] [XGrammar] Reasoning started, grammar suspended") }
                        proc.tokenMask = nil
                        return  // Don't advance grammar during thinking
                    }
                    // First token is not <think> — enable grammar immediately
                    if dbg { print("[\(ts())] [XGrammar] No reasoning, grammar active from token 1") }
                }

                if inReasoning {
                    if tokenID == endThinkTokenId {
                        inReasoning = false
                        if dbg { print("[\(ts())] [XGrammar] Reasoning ended, grammar active") }
                        // Start applying grammar from here
                        proc.tokenMask = matcher.nextTokenMask()
                    } else {
                        // Still in reasoning — don't advance grammar, don't apply mask
                        proc.tokenMask = nil
                    }
                    return
                }

                // If grammar reached accepting state, reset it so the Kleene star
                // `root ::= free_text (tool_call free_text)*` can match more tool calls.
                // Without reset, subsequent tool calls are unconstrained.
                if matcher.isTerminated() {
                    matcher.reset()
                    if dbg {
                        print("[\(ts())] [XGrammar] Grammar terminated at token \(grammarTokenCount) — reset to constrain next tool call")
                    }
                }

                // Outside reasoning: advance grammar normally
                let accepted = matcher.acceptToken(tokenID)
                if !accepted && dbg {
                    print("[\(ts())] [XGrammar] WARNING: acceptToken(\(tokenID)) returned false — grammar state may be corrupted")
                    if let err = XGrammarService.consumeLastError() {
                        print("[\(ts())] [XGrammar] C++ error: \(err)")
                    }
                }
                let mask = matcher.nextTokenMask()
                proc.tokenMask = mask
                // -VV: Log each grammar-constrained token
                if vvTrace {
                    let tokenStr = vvTokenizer.decode(tokens: [tokenID])
                    let maskStatus = mask != nil ? "constrained" : "unconstrained"
                    print("\(vvCyan)[\(ts())] [VV] GRAMMAR token[\(grammarTokenCount)] id=\(tokenID) \"\(tokenStr.replacingOccurrences(of: "\n", with: "\\n"))\" accepted=\(accepted) \(maskStatus)\(vvReset)")
                }
                if mask == nil && !matcher.isTerminated() && dbg {
                    // nil mask on non-terminated grammar means all tokens are allowed.
                    print("[\(ts())] [XGrammar] INFO: nextTokenMask() returned nil (all tokens allowed) at token \(grammarTokenCount)")
                }
            }

            if debugLogging {
                print("[\(ts())] [XGrammar] Grammar constraint active for tool_call (EBNF, \(tools.count) tools, vocab_size=\(service.vocabSize), reasoner gating=\(thinkTokenId != nil ? "enabled" : "disabled"))")
            }

            return proc
        } catch {
            if debugLogging {
                print("[\(ts())] [XGrammar] Failed to set up tool call grammar: \(error). Falling back to parsing-only mode.")
            }
            return nil
        }
    }

    /// Build a structural tag JSON spec for xgrammar's TagDispatch.
    /// Uses TriggeredTagsFormat: allows ALL text until "<tool_call>\n<function=" trigger,
    /// then constrains to valid XML parameter format per tool schema.
    /// This is the same approach vLLM/SGLang use — no character-class issues with <think> etc.
    static func buildToolCallStructuralTag(tools: [RequestTool]) -> String {
        var tags: [[String: Any]] = []
        for tool in tools {
            let name = tool.function.name
            // Build JSON schema for the tool's parameters
            var paramSchema: Any = true  // true = any JSON
            if let params = tool.function.parameters?.toSendable() as? [String: Any] {
                paramSchema = params
            }
            tags.append([
                "type": "tag",
                "begin": "<tool_call>\n<function=\(name)>\n",
                "content": [
                    "type": "qwen_xml_parameter",
                    "json_schema": paramSchema
                ] as [String: Any],
                "end": "\n</function>\n</tool_call>"
            ])
        }

        let triggeredTags: [String: Any] = [
            "type": "triggered_tags",
            "triggers": ["<tool_call>\n<function="],
            "tags": tags,
            "at_least_one": false,
            "stop_after_first": false,
            "excludes": [] as [String]  // Think tokens handled by reasoner gating, not excludes
        ]

        let structuralTag: [String: Any] = [
            "type": "structural_tag",
            "format": triggeredTags
        ]

        let data = try! JSONSerialization.data(withJSONObject: structuralTag, options: [.sortedKeys])
        return String(data: data, encoding: .utf8)!
    }

    /// Build an EBNF grammar that allows free text OR valid XML tool calls.
    /// For tools with `type: "array"` or `type: "object"` parameters, enumerates ALL
    /// parameters explicitly with typed value rules — no generic `param` fallback, so
    /// the JSON constraint is actually enforced (CFG union of alternatives can't escape).
    static func buildToolCallEBNF(tools: [RequestTool]) -> String {
        var toolRules: [String] = []
        var toolVariantNames: [String] = []
        var typedRules: [String] = []
        var needsJsonGrammar = false

        for tool in tools {
            let name = tool.function.name
            let safeName = name.unicodeScalars.map { CharacterSet.alphanumerics.contains($0) ? String($0) : "_" }.joined()
            let ruleName = "call_\(safeName)"
            let requiredNames = Self.requiredParamNames(for: tool)
            let requiredSet = Set(requiredNames)

            // Check for array/object parameters that need JSON grammar enforcement
            let structured = Self.structuredParams(for: tool) // [(name, "array"|"object")]
            let structuredNames = Set(structured.map { $0.name })
            let allParams = Self.allParams(for: tool) // [(name, type)]

            if !structuredNames.isEmpty { needsJsonGrammar = true }

            // Value rule based on parameter type (JSON grammar for array/object, permissive otherwise)
            func valueRule(for paramName: String) -> String {
                if structuredNames.contains(paramName) {
                    let pType = allParams.first(where: { $0.name == paramName })?.type ?? "object"
                    return pType == "array" ? "json_array" : "json_object"
                }
                return "param_value"
            }

            // Generate named rules for required params — enforces each must appear, in order
            var requiredRuleRefs: [String] = []
            for paramName in requiredNames {
                let rRule = "\(safeName)_rp_\(paramName)"
                typedRules.append("\(rRule) ::= \"<parameter=\(paramName)>\\n\" \(valueRule(for: paramName)) \"\\n</parameter>\\n\"")
                requiredRuleRefs.append(rRule)
            }

            // Build the extras section (optional params after required)
            let optionalParams = allParams.filter { !requiredSet.contains($0.name) }

            if allParams.isEmpty && requiredNames.isEmpty {
                // No schema info at all — fall back to at least 1 generic param
                toolRules.append("\(ruleName) ::= \"<function=\(name)>\\n\" param extra_params \"</function>\\n\"")
            } else {
                // Named required params + typed union for optional params.
                // Only schema-defined parameter names are allowed — prevents
                // the model from hallucinating extra parameters like "description".
                var optAlts: [String] = []
                for (pName, _) in optionalParams {
                    let oRule = "\(safeName)_op_\(pName)"
                    typedRules.append("\(oRule) ::= \"<parameter=\(pName)>\\n\" \(valueRule(for: pName)) \"\\n</parameter>\\n\"")
                    optAlts.append(oRule)
                }

                let requiredPart = requiredRuleRefs.joined(separator: " ")
                if optAlts.isEmpty {
                    toolRules.append("\(ruleName) ::= \"<function=\(name)>\\n\" \(requiredPart) \"</function>\\n\"")
                } else {
                    let optUnion = "\(safeName)_opt"
                    typedRules.append("\(optUnion) ::= \(optAlts.joined(separator: " | "))")
                    typedRules.append("\(safeName)_extra ::= \(optUnion)*")
                    let sep = requiredPart.isEmpty ? "" : " "
                    toolRules.append("\(ruleName) ::= \"<function=\(name)>\\n\" \(requiredPart)\(sep)\(safeName)_extra \"</function>\\n\"")
                }
            }

            toolVariantNames.append(ruleName)
        }

        let toolAlternation = toolVariantNames.joined(separator: " | ")

        var grammar = """
        root ::= free_text (tool_call free_text)*
        free_text ::= free_char*
        free_char ::= [^<] | "<" [^t] | "<t" [^o] | "<to" [^o] | "<too" [^l] | "<tool" [^_] | "<tool_" [^c] | "<tool_c" [^a] | "<tool_ca" [^l] | "<tool_cal" [^l] | "<tool_call" [^>]
        tool_call ::= "<tool_call>\\n" tool_variant "</tool_call>"
        tool_variant ::= \(toolAlternation)
        \(toolRules.joined(separator: "\n        "))
        extra_params ::= param*
        param ::= "<parameter=" param_name ">\\n" param_value "\\n</parameter>\\n"
        param_name ::= [a-zA-Z_] [a-zA-Z0-9_]*
        pv_char ::= [^\\n] | "\\n" [^<] | "\\n<" [^/]
        param_value ::= pv_char+
        \(typedRules.joined(separator: "\n        "))
        """

        if needsJsonGrammar {
            grammar += "\n"
            grammar += "        json_array ::= \"[\" json_ws (json_value (json_ws \",\" json_ws json_value)*)? json_ws \"]\"\n"
            grammar += "        json_object ::= \"{\" json_ws (json_pair (json_ws \",\" json_ws json_pair)*)? json_ws \"}\"\n"
            grammar += "        json_pair ::= json_string json_ws \":\" json_ws json_value\n"
            grammar += "        json_value ::= json_string | json_number | json_object | json_array | \"true\" | \"false\" | \"null\"\n"
            grammar += "        json_string ::= \"\\\"\" json_char* \"\\\"\"\n"
            grammar += "        json_char ::= [^\"\\\\] | \"\\\\\" [\"\\\\bfnrt/] | \"\\\\u\" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]\n"
            grammar += "        json_number ::= \"-\"? (\"0\" | [1-9] [0-9]*) (\".\" [0-9]+)? ([eE] [+-]? [0-9]+)?\n"
            grammar += "        json_ws ::= [ \\t\\n]*\n"
        }

        return grammar
    }

    /// Extract parameters with type "array" or "object" from a tool's JSON Schema.
    private static func structuredParams(for tool: RequestTool) -> [(name: String, type: String)] {
        guard let params = tool.function.parameters,
              let dict = params.value.toAny() as? [String: Any],
              let props = dict["properties"] as? [String: Any] else { return [] }

        var result: [(String, String)] = []
        for (key, value) in props {
            if let propDict = value as? [String: Any],
               let propType = propDict["type"] as? String,
               propType == "array" || propType == "object" {
                result.append((key, propType))
            }
        }
        return result
    }

    /// Extract ALL parameters (name, type) from a tool's JSON Schema.
    private static func allParams(for tool: RequestTool) -> [(name: String, type: String)] {
        guard let params = tool.function.parameters,
              let dict = params.value.toAny() as? [String: Any],
              let props = dict["properties"] as? [String: Any] else { return [] }

        var result: [(String, String)] = []
        for (key, value) in props {
            let propType: String
            if let propDict = value as? [String: Any],
               let t = propDict["type"] as? String {
                propType = t
            } else {
                propType = "string" // default
            }
            result.append((key, propType))
        }
        return result.sorted(by: { $0.0 < $1.0 }) // deterministic order
    }

    /// Extract the names of required parameters from a tool's JSON Schema.
    /// Preserves the order from the client's "required" array — this matters because
    /// the grammar forces params in this order, and the natural schema order
    /// (e.g. filePath → oldString → newString) helps the model generate correct values.
    static func requiredParamNames(for tool: RequestTool) -> [String] {
        guard let params = tool.function.parameters,
              let dict = params.value.toAny() as? [String: Any],
              let required = dict["required"] as? [String],
              !required.isEmpty else { return [] }
        return required
    }

    /// Returns true when the model has a VLM config layout that can't be loaded
    /// correctly by the LLM factory.  VLM models like gemma-3 store architecture
    /// fields (num_attention_heads, head_dim, etc.) only inside text_config, not at
    /// the top level.  The LLM factory's Codable config fills in wrong defaults for
    /// these missing fields, causing inference crashes.
    ///
    /// Models like Qwen3.5-35B-A3B also have text_config but their LLM model class
    /// (Qwen3_5MoE) properly reads from text_config with full field coverage.
    /// We detect the "sparse text_config" case by checking for missing key fields.
    private func isVLMOnlyConfig(directory: URL) -> Bool {
        let configURL = directory.appendingPathComponent("config.json")
        guard let data = try? Data(contentsOf: configURL),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let textConfig = json["text_config"] as? [String: Any],
              json["vision_config"] != nil else {
            return false
        }
        // If text_config lacks num_attention_heads AND the top-level config also lacks it,
        // the LLM factory will use wrong defaults. Prefer VLM factory.
        let hasTopLevelHeads = json["num_attention_heads"] != nil
        let hasNestedHeads = textConfig["num_attention_heads"] != nil
        if !hasTopLevelHeads && !hasNestedHeads {
            return true
        }
        return false
    }

    private func isVisionModel(directory: URL) throws -> Bool {
        let config = directory.appendingPathComponent("config.json")
        guard let data = try? Data(contentsOf: config),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return false
        }
        let modelType = (json["model_type"] as? String ?? "").lowercased()
        if modelType.contains("vl") || modelType.contains("vision") {
            return true
        }
        // Multimodal models (e.g. gemma3) have both text_config and vision_config
        if json["text_config"] != nil && json["vision_config"] != nil {
            return true
        }
        // Vision token IDs only count as VLM if vision_config is also present.
        // Text-only variants (e.g. gemma-3n-*-lm) may inherit image_token_id
        // from the base multimodal config without actually being VLMs.
        if json["vision_config"] != nil
            && (json["image_token_id"] != nil || json["vision_start_token_id"] != nil)
        {
            return true
        }
        return false
    }

    private func buildPrompt(from messages: [Message]) -> String {
        messages.map { "\($0.role): \($0.textContent)" }.joined(separator: "\n")
    }

    private func buildUserInput(from messages: [Message], tools: [ToolSpec]? = nil, responseFormat: ResponseFormat? = nil, chatTemplateKwargs: [String: AnyCodable]? = nil) throws -> (UserInput, tempFiles: [URL]) {
        var chatMessages: [Chat.Message] = []
        var hasSystemMessage = false
        var allTempFiles: [URL] = []
        // Merge consecutive system/developer messages into one so that Jinja
        // templates requiring a single system message at position 0 don't fail
        // (e.g. Qwen3.5). The OpenAI API allows multiple system messages, so
        // we consolidate them here for broader compatibility.
        var pendingSystemParts: [String] = []
        func flushSystemParts() {
            guard !pendingSystemParts.isEmpty else { return }
            hasSystemMessage = true
            chatMessages.append(.system(pendingSystemParts.joined(separator: "\n\n")))
            pendingSystemParts.removeAll()
        }
        for m in messages {
            let text = m.textContent
            let media = try extractMedia(from: m)
            allTempFiles.append(contentsOf: media.tempFiles)
            switch m.role {
            case "system", "developer":
                pendingSystemParts.append(text)
            case "assistant":
                flushSystemParts()
                if let toolCalls = m.toolCalls, !toolCalls.isEmpty {
                    // Reconstruct assistant tool-call message as text for the chat template.
                    // Models expect tool calls in a specific format that the template handles.
                    var parts: [String] = []
                    if !text.isEmpty {
                        parts.append(text)
                    }
                    for tc in toolCalls {
                        parts.append("<tool_call>\n{\"name\": \"\(tc.function.name)\", \"arguments\": \(tc.function.arguments)}\n</tool_call>")
                    }
                    chatMessages.append(.assistant(parts.joined(separator: "\n")))
                } else {
                    chatMessages.append(.assistant(text))
                }
            case "tool":
                flushSystemParts()
                // Tool result messages — use the vendor's .tool() factory.
                // Resolve function name from tool_call_id if not provided (OpenAI spec
                // makes name optional, but some templates like Gemma require it).
                var resolvedName = m.name
                if resolvedName == nil, let callId = m.toolCallId {
                    for prev in messages {
                        if let tcs = prev.toolCalls {
                            if let match = tcs.first(where: { $0.id == callId }) {
                                resolvedName = match.function.name
                                break
                            }
                        }
                    }
                }
                let toolContent: String
                if let name = resolvedName {
                    toolContent = "<tool_response>\n{\"name\": \"\(name)\", \"content\": \(text)}\n</tool_response>"
                } else {
                    toolContent = text
                }
                chatMessages.append(.tool(toolContent, name: resolvedName))
            default:
                flushSystemParts()
                chatMessages.append(.user(text, images: media.images, videos: media.videos))
            }
        }
        flushSystemParts()

        // Align with Vesta behavior: always include a base system instruction
        // when callers don't explicitly provide one.
        if !hasSystemMessage {
            chatMessages.insert(.system("You are a helpful assistant!"), at: 0)
        }

        // Inject JSON format instructions when response_format is requested
        if let format = responseFormat {
            let jsonInstruction: String?
            switch format.type {
            case "json_object":
                jsonInstruction = "Respond with valid JSON only. Do not include any text outside the JSON object."
            case "json_schema":
                if let schema = format.jsonSchema {
                    var parts = ["Respond with valid JSON only. Do not include any text outside the JSON object."]
                    if let schemaValue = schema.schema {
                        let encoder = JSONEncoder()
                        encoder.outputFormatting = [.sortedKeys]
                        if let data = try? encoder.encode(schemaValue),
                           let schemaStr = String(data: data, encoding: .utf8) {
                            parts.append("Your response must conform to this JSON schema: \(schemaStr)")
                        }
                    }
                    if let name = schema.name {
                        parts.append("The response object is: \(name)")
                    }
                    jsonInstruction = parts.joined(separator: "\n")
                } else {
                    jsonInstruction = "Respond with valid JSON only. Do not include any text outside the JSON object."
                }
            default:
                jsonInstruction = nil
            }
            if let instruction = jsonInstruction {
                // Append to existing system message rather than adding a second one,
                // because some chat templates (e.g. Qwen3.5) don't support multiple
                // system messages and throw Jinja.TemplateException.
                if let sysIdx = chatMessages.firstIndex(where: { $0.role == .system }) {
                    chatMessages[sysIdx] = .system(chatMessages[sysIdx].content + "\n\n" + instruction)
                } else {
                    chatMessages.insert(.system(instruction), at: 0)
                }
            }
        }

        if chatMessages.isEmpty {
            return (UserInput(prompt: ""), tempFiles: allTempFiles)
        }

        var input = UserInput(chat: chatMessages, processing: .init(resize: .init(width: 1024, height: 1024)), tools: tools)

        // When --tool-call-parser is set and tools are present, override the chat template
        if let parser = toolCallParser, tools != nil, !tools!.isEmpty {
            let templateOverride: String?
            switch parser {
            case "qwen3_xml", "afm_adaptive_xml":
                templateOverride = Self.qwen3XMLTemplate
            case "hermes":
                templateOverride = Self.hermesTemplate
            case "llama3_json":
                templateOverride = Self.llama3JSONTemplate
            case "mistral":
                templateOverride = Self.mistralTemplate
            case "gemma":
                // Gemma uses the model's built-in template; no override needed
                templateOverride = nil
            default:
                print("Warning: unknown tool-call-parser '\(parser)', using default chat template")
                templateOverride = nil
            }
            if let tpl = templateOverride {
                input.additionalContext = (input.additionalContext ?? [:])
                input.additionalContext?["chatTemplateOverride"] = tpl
            }
            if debugLogging {
                print("[\(ts())] [ToolCallParser] Using \(parser) chat template override")
            }
        }

        // Merge chat template kwargs: server defaults first, then request-level overrides
        var resolvedKwargs: [String: Any] = self.defaultChatTemplateKwargs ?? [:]
        if let requestKwargs = chatTemplateKwargs {
            for (key, value) in requestKwargs {
                resolvedKwargs[key] = value.value.toAny()
            }
        }
        if responseFormat?.type == "json_schema",
           thinkStartTag != nil, thinkEndTag != nil,
           (resolvedKwargs["enable_thinking"] as? Bool) != false {
            resolvedKwargs["enable_thinking"] = false
            print("[\(ts())] [StructuredOutput] Disabling thinking for guided JSON on reasoning-capable model")
        }
        if !resolvedKwargs.isEmpty {
            if input.additionalContext == nil { input.additionalContext = [:] }
            for (key, value) in resolvedKwargs {
                input.additionalContext?[key] = value
            }
            if debugLogging {
                print("[\(ts())] [ChatTemplateKwargs] Merged into additionalContext: \(resolvedKwargs.keys.sorted().joined(separator: ", "))")
            }
        }

        return (input, tempFiles: allTempFiles)
    }

    /// Remove temp files created during media extraction.
    private func cleanupTempFiles(_ files: [URL]) {
        for file in files {
            try? FileManager.default.removeItem(at: file)
        }
    }

    private static let videoExtensions: Set<String> = ["mp4", "mov", "avi", "mkv", "webm", "m4v"]

    private func extractMedia(from message: Message) throws -> (images: [UserInput.Image], videos: [UserInput.Video], tempFiles: [URL]) {
        guard let content = message.content, case .parts(let parts) = content else { return ([], [], []) }
        var images: [UserInput.Image] = []
        var videos: [UserInput.Video] = []
        var tempFiles: [URL] = []
        for part in parts where part.type == "image_url" {
            guard let raw = part.image_url?.url else { continue }
            if raw.hasPrefix("data:") {
                // Parse "data:<mime>;base64,..." data URLs
                guard let commaIndex = raw.firstIndex(of: ",") else { continue }
                let header = raw[raw.startIndex..<commaIndex]
                let payload = String(raw[raw.index(after: commaIndex)...])
                guard let decoded = Data(base64Encoded: payload, options: .ignoreUnknownCharacters),
                      !decoded.isEmpty else { continue }
                // Check MIME type to route image vs video
                if header.hasPrefix("data:video/") {
                    // Video: extract extension from MIME (e.g. "video/mp4" → "mp4"), write temp file
                    let ext: String
                    if let slash = header.firstIndex(of: "/") {
                        let sub = header[header.index(after: slash)...].prefix(while: { $0 != ";" && $0 != "," })
                        ext = sub.isEmpty ? "mp4" : String(sub)
                    } else { ext = "mp4" }
                    let temp = FileManager.default.temporaryDirectory
                        .appendingPathComponent("afm_mlx_video_\(UUID().uuidString).\(ext)")
                    try decoded.write(to: temp)
                    videos.append(.url(temp))
                    tempFiles.append(temp)
                } else {
                    // Image: decode to CIImage directly (no temp file)
                    guard let ciImage = CIImage(data: decoded) else { continue }
                    images.append(.ciImage(ciImage))
                }
            } else if let url = URL(string: raw),
                      let scheme = url.scheme, scheme == "http" || scheme == "https" {
                let (data, _) = try awaitURL(url: url)
                let temp = FileManager.default.temporaryDirectory
                    .appendingPathComponent("afm_mlx_image_\(UUID().uuidString).\(url.pathExtension.isEmpty ? "jpg" : url.pathExtension)")
                try data.write(to: temp)
                images.append(.url(temp))
                tempFiles.append(temp)
            } else if let url = URL(string: raw) {
                let ext = url.pathExtension.lowercased()
                if Self.videoExtensions.contains(ext) {
                    videos.append(.url(url))
                } else {
                    images.append(.url(url))
                }
            }
        }
        return (images, videos, tempFiles)
    }

    private func awaitURL(url: URL) throws -> (Data, URLResponse) {
        let sem = DispatchSemaphore(value: 0)
        var result: Result<(Data, URLResponse), Error>?
        let task = URLSession.shared.dataTask(with: url) { data, response, error in
            if let error {
                result = .failure(error)
            } else if let data, let response {
                result = .success((data, response))
            } else {
                result = .failure(MLXServiceError.downloadFailed("image download failed"))
            }
            sem.signal()
        }
        task.resume()
        sem.wait()
        switch result {
        case .success(let pair):
            return pair
        case .failure(let error):
            throw error
        case .none:
            throw MLXServiceError.downloadFailed("image download failed")
        }
    }

    private func estimateTokens(_ text: String) -> Int {
        let words = text.split(whereSeparator: \.isWhitespace).count
        let charBased = Double(text.count) / 4.0
        let wordBased = Double(words) / 0.75
        return Int(max(charBased, wordBased))
    }

    private func resolveLogprobs(_ data: [TokenLogprobData], tokenizer: any Tokenizer) -> [ResolvedLogprob] {
        data.map { entry in
            let token = tokenizer.decode(tokens: [entry.tokenId])
            let topTokens = zip(entry.topTokenIds, entry.topLogprobs).map { (id, lp) in
                (token: tokenizer.decode(tokens: [id]), tokenId: id, logprob: lp)
            }
            return ResolvedLogprob(
                token: token,
                tokenId: entry.tokenId,
                logprob: entry.logprob,
                topTokens: topTokens
            )
        }
    }

    private func summarizeCacheState(_ cache: [KVCache], sampleLimit: Int = 4) -> String {
        guard !cache.isEmpty else { return "layers=0" }

        let activeLayers = cache.reduce(into: 0) { count, layer in
            if layer.offset > 0 { count += 1 }
        }
        let maxOffset = cache.map(\.offset).max() ?? 0
        let preferredSamples = [0, 1, 3, 7]
        var sampleIndices = preferredSamples.filter { $0 < cache.count && cache[$0].offset > 0 }

        if sampleIndices.count < sampleLimit {
            for index in cache.indices where cache[index].offset > 0 && !sampleIndices.contains(index) {
                sampleIndices.append(index)
                if sampleIndices.count == sampleLimit { break }
            }
        }

        if sampleIndices.isEmpty {
            sampleIndices = Array(cache.indices.prefix(min(sampleLimit, cache.count)))
        }

        let sample = sampleIndices.map { index in
            "\(index):\(type(of: cache[index]))@\(cache[index].offset)"
        }.joined(separator: ", ")

        return "layers=\(cache.count) active_layers=\(activeLayers) max_offset=\(maxOffset) sample=[\(sample)]"
    }

    private func formatCacheSeconds(_ value: Double?) -> String {
        String(format: "%.6f", value ?? 0)
    }

    private func exportCacheProfile(_ record: [String: Any]) {
        let environment = ProcessInfo.processInfo.environment
        guard let path = cacheProfilePath ?? environment["MACAFM_CACHE_PROFILE_PATH"] ?? environment["AFM_CACHE_PROFILE_PATH"] else {
            return
        }

        CacheProfileExporter.append(record: record, to: path)
    }

    private func logCachePrefill(
        mode: String,
        outcome: String,
        inputTokenCount: Int,
        cachedTokenCount: Int,
        suffixTokenCount: Int,
        radixEntryCount: Int?,
        cache: [KVCache],
        lookupTime: Double? = nil,
        restoreTime: Double? = nil,
        trimTime: Double? = nil,
        truncateTime: Double? = nil
    ) {
        var parts = [
            "mode=\(mode)",
            "outcome=\(outcome)",
            "input_tokens=\(inputTokenCount)",
            "cached_tokens=\(cachedTokenCount)",
            "suffix_tokens=\(suffixTokenCount)"
        ]
        if let radixEntryCount {
            parts.append("radix_entries=\(radixEntryCount)")
        }
        if let lookupTime {
            parts.append("lookup=\(formatCacheSeconds(lookupTime))s")
        }
        if let restoreTime {
            parts.append("restore=\(formatCacheSeconds(restoreTime))s")
        }
        if let trimTime {
            parts.append("trim=\(formatCacheSeconds(trimTime))s")
        }
        if let truncateTime {
            parts.append("truncate=\(formatCacheSeconds(truncateTime))s")
        }
        parts.append(summarizeCacheState(cache))

        print("[\(ts())] [PrefixCache] Prefill: \(parts.joined(separator: " | "))")
    }

    private func logCacheSave(
        mode: String,
        inputTokenCount: Int,
        radixEntryCount: Int?,
        cache: [KVCache],
        trimTime: Double? = nil,
        truncateTime: Double? = nil,
        insertTime: Double? = nil
    ) {
        var parts = [
            "mode=\(mode)",
            "stored_tokens=\(inputTokenCount)"
        ]
        if let radixEntryCount {
            parts.append("radix_entries=\(radixEntryCount)")
        }
        if let trimTime {
            parts.append("trim=\(formatCacheSeconds(trimTime))s")
        }
        if let truncateTime {
            parts.append("truncate=\(formatCacheSeconds(truncateTime))s")
        }
        if let insertTime {
            parts.append("insert=\(formatCacheSeconds(insertTime))s")
        }
        parts.append(summarizeCacheState(cache))
        print("[\(ts())] [PrefixCache] Save complete: \(parts.joined(separator: " | "))")
    }

    private func logCacheProfile(
        phase: String,
        mode: String,
        outcome: String,
        inputTokenCount: Int,
        cachedTokenCount: Int,
        promptTime: Double,
        lookupTime: Double? = nil,
        restoreTime: Double? = nil,
        trimTime: Double? = nil,
        truncateTime: Double? = nil,
        insertTime: Double? = nil
    ) {
        let lookup = lookupTime ?? 0
        let restore = restoreTime ?? 0
        let trim = trimTime ?? 0
        let truncate = truncateTime ?? 0
        let insert = insertTime ?? 0
        let cacheOverhead = lookup + restore + trim + truncate + insert
        let reuseRatio = inputTokenCount > 0 ? Double(cachedTokenCount) / Double(inputTokenCount) : 0
        let overheadShare = promptTime > 0 ? cacheOverhead / promptTime : 0

        print(
            "[\(ts())] [CacheProfile] phase=\(phase) | mode=\(mode) | outcome=\(outcome) | " +
                "input_tokens=\(inputTokenCount) | cached_tokens=\(cachedTokenCount) | " +
                "reuse_ratio=\(String(format: "%.3f", reuseRatio)) | " +
                "lookup=\(formatCacheSeconds(lookupTime))s | " +
                "restore=\(formatCacheSeconds(restoreTime))s | " +
                "trim=\(formatCacheSeconds(trimTime))s | " +
                "truncate=\(formatCacheSeconds(truncateTime))s | " +
                "insert=\(formatCacheSeconds(insertTime))s | " +
                "cache_overhead=\(formatCacheSeconds(cacheOverhead))s | " +
                "prompt_time=\(String(format: "%.3f", promptTime))s | " +
                "overhead_share=\(String(format: "%.6f", overheadShare))"
        )

        exportCacheProfile([
            "timestamp": ts(),
            "phase": phase,
            "mode": mode,
            "outcome": outcome,
            "input_tokens": inputTokenCount,
            "cached_tokens": cachedTokenCount,
            "reuse_ratio": reuseRatio,
            "lookup_s": lookup,
            "restore_s": restore,
            "trim_s": trim,
            "truncate_s": truncate,
            "insert_s": insert,
            "cache_overhead_s": cacheOverhead,
            "prompt_time_s": promptTime,
            "overhead_share": overheadShare,
        ])
    }

    private func logUsageChunk(
        stage: String,
        streaming: Bool,
        cachedTokens: Int? = nil,
        promptTokens: Int? = nil,
        completionTokens: Int? = nil,
        promptTime: Double? = nil,
        generateTime: Double? = nil
    ) {
        func describe<T>(_ value: T?) -> String {
            value.map { "\($0)" } ?? "pending"
        }

        let promptTimeValue = promptTime.map { String(format: "%.3f", $0) } ?? "pending"
        let generateTimeValue = generateTime.map { String(format: "%.3f", $0) } ?? "pending"

        print(
            "[\(ts())] [ChunkStats] stage=\(stage) | stream=\(streaming) | " +
                "cached_tokens=\(describe(cachedTokens)) | " +
                "prompt_tokens=\(describe(promptTokens)) | " +
                "completion_tokens=\(describe(completionTokens)) | " +
                "prompt_time=\(promptTimeValue)s | " +
                "generate_time=\(generateTimeValue)s"
        )
    }

    private func shouldTraceReplayBoundary() -> Bool {
        let env = ProcessInfo.processInfo.environment
        let value = env["AFM_PREFIX_CACHE_TRACE_BOUNDARY"]?
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        return value == "1" || value == "true" || value == "yes"
    }

    private func logReplayBoundary(
        tokenizer: any Tokenizer,
        mode: String,
        inputTokens: [Int],
        effectivePrefix: Int
    ) {
        guard shouldTraceReplayBoundary() else { return }

        let split = max(0, min(effectivePrefix, inputTokens.count))
        let prefixTail = Array(inputTokens[max(0, split - 8)..<split])
        let suffixHead = Array(inputTokens[split..<min(inputTokens.count, split + 16)])
        let prefixTailDecoded = prefixTail.isEmpty ? "" : tokenizer.decode(tokens: prefixTail)
        let suffixHeadDecoded = suffixHead.isEmpty ? "" : tokenizer.decode(tokens: suffixHead)

        print(
            "[\(ts())] [PrefixCache] Boundary: mode=\(mode) | prefix_tokens=\(split) | " +
                "suffix_tokens=\(inputTokens.count - split) | prefix_tail_ids=\(prefixTail) | " +
                "suffix_head_ids=\(suffixHead) | prefix_tail_decoded=\(String(reflecting: prefixTailDecoded)) | " +
                "suffix_head_decoded=\(String(reflecting: suffixHeadDecoded))"
        )
    }

    private func hasRecurrentLayers(_ cache: [KVCache]) -> Bool {
        cache.contains { $0 is ArraysCache || $0 is CacheList }
    }

    private func unsafeExactReplaySuffix() -> Int? {
        let env = ProcessInfo.processInfo.environment
        guard let rawValue = env["AFM_PREFIX_CACHE_ALLOW_UNSAFE_EXACT_REPLAY"]?
            .trimmingCharacters(in: .whitespacesAndNewlines), !rawValue.isEmpty else {
            return nil
        }
        let value = rawValue.lowercased()
        if value == "true" || value == "yes" {
            return 1
        }
        if let suffix = Int(value), suffix >= 1 {
            return suffix
        }
        return nil
    }

    private func supportsPhysicalTruncation(_ cache: KVCache) -> Bool {
        !(cache is RotatingKVCache)
    }

    private func restoredMetaState(for cache: KVCache, savedMetaState: [String]?) -> [String]? {
        guard let savedMetaState else { return nil }
        if cache is RotatingKVCache, savedMetaState.count == 5 {
            // Use saved offset/idx directly — they were captured at save time and
            // reflect the correct rotation state. The state setter already set
            // offset = keys.dim(2); metaState setter will override with saved values.
            return savedMetaState
        }
        return savedMetaState
    }

    private func effectiveCachedPrefix(prefixLen: Int, inputTokenCount: Int, cache: [KVCache]) -> Int {
        // Hybrid recurrent layers (for example Mamba/GatedDeltaNet state in ArraysCache)
        // are not replay-safe for exact full-prefix restores on Qwen3.5-35B-A3B. Falling
        // back to a cold prefill preserves correctness for deterministic replays.
        let forcedSuffix = unsafeExactReplaySuffix()

        if prefixLen == inputTokenCount && hasRecurrentLayers(cache) && forcedSuffix == nil {
            return 0
        }

        if prefixLen == inputTokenCount, let forcedSuffix {
            return max(0, inputTokenCount - forcedSuffix)
        }

        let minSuffix = 16
        return min(prefixLen, max(0, inputTokenCount - minSuffix))
    }

    // MARK: - Prompt cache helpers

    /// Extract a flat array of token IDs from prepared LMInput.
    private func extractTokenArray(_ input: LMInput) -> [Int] {
        input.text.tokens.reshaped(-1).asArray(Int.self)
    }

    /// Check if the LMInput contains multimodal content (images/video) which we don't cache.
    private func isMultimodalInput(_ input: LMInput) -> Bool {
        input.image != nil || input.video != nil
    }

    private func normalizedRepetitionPenalty(_ value: Double?) -> Float? {
        guard let value else { return nil }
        if abs(value - 1.0) < 0.000_001 {
            return nil
        }
        return Float(value)
    }

    private func normalizedTopP(_ value: Double?) -> Float {
        guard let value else { return 1.0 }  // MLX library default
        return Float(min(max(value, 0.0), 1.0))
    }

    private func normalizedTemperature(_ value: Double?) -> Float {
        guard let value else { return 0.6 }  // MLX library default
        return Float(min(max(value, 0.0), 1.0))
    }

    private func normalizedTopK(_ value: Int?) -> Int {
        guard let value else { return 0 }  // 0 = disabled
        return max(0, value)
    }

    private func normalizedMinP(_ value: Double?) -> Float {
        guard let value else { return 0.0 }  // 0.0 = disabled
        return Float(min(max(value, 0.0), 1.0))
    }

    private func normalizedPresencePenalty(_ value: Double?) -> Float {
        guard let value else { return 0.0 }  // 0.0 = disabled
        return Float(value)
    }

    private func normalizedSeed(_ value: Int?) -> UInt64? {
        guard let value else { return nil }
        return UInt64(max(0, value))
    }

    // MARK: - Tool call parser templates

    /// vLLM's tool_chat_template_qwen3coder.jinja — teaches the model the exact XML tool call format.
    /// Source: https://github.com/vllm-project/vllm/blob/main/examples/tool_chat_template_qwen3coder.jinja
    static let qwen3XMLTemplate = """
    {%- macro sorted_json(obj) -%}
        {%- if obj is mapping -%}
            {%- set ns = namespace(first=true) -%}
            {{- "{" -}}
            {%- for k, v in obj|dictsort -%}
                {%- if not ns.first -%},{%- endif -%}
                {%- set ns.first = false -%}
                {{- '"' ~ k ~ '":' -}}
                {{- sorted_json(v) -}}
            {%- endfor -%}
            {{- "}" -}}
        {%- elif obj is sequence and obj is not string -%}
            {{- "[" -}}
            {%- for item in obj -%}
                {%- if not loop.first -%},{%- endif -%}
                {{- sorted_json(item) -}}
            {%- endfor -%}
            {{- "]" -}}
        {%- elif obj is string -%}
            {{- '"' ~ obj ~ '"' -}}
        {%- elif obj is boolean -%}
            {{- "true" if obj else "false" -}}
        {%- elif obj is none -%}
            {{- "null" -}}
        {%- else -%}
            {{- obj -}}
        {%- endif -%}
    {%- endmacro -%}
    {% macro render_extra_keys(json_dict, handled_keys) %}
        {%- if json_dict is mapping %}
            {%- for json_key, _ in json_dict|dictsort if json_key not in handled_keys %}
                {%- if json_dict[json_key] is mapping or (json_dict[json_key] is sequence and json_dict[json_key] is not string) %}
                    {{- '\\n<' ~ json_key ~ '>' }}{{ sorted_json(json_dict[json_key]) }}{{- '</' ~ json_key ~ '>' }}
                {%- else %}
                    {{-'\\n<' ~ json_key ~ '>' ~ (json_dict[json_key] | string) ~ '</' ~ json_key ~ '>' }}
                {%- endif %}
            {%- endfor %}
        {%- endif %}
    {% endmacro %}

    {%- if messages[0]["role"] == "system" %}
        {%- set system_message = messages[0]["content"] %}
        {%- set loop_messages = messages[1:] %}
    {%- else %}
        {%- set loop_messages = messages %}
    {%- endif %}

    {%- if not tools is defined %}
        {%- set tools = [] %}
    {%- endif %}

    {%- if system_message is defined %}
        {{- "<|im_start|>system\\n" + system_message }}
    {%- else %}
        {%- if tools is iterable and tools | length > 0 %}
            {{- "<|im_start|>system\\nYou are Qwen, a helpful AI assistant that can interact with a computer to solve tasks." }}
        {%- endif %}
    {%- endif %}
    {%- if tools is iterable and tools | length > 0 %}
        {{- "\\n\\n# Tools\\n\\nYou have access to the following functions:\\n\\n" }}
        {{- "<tools>" }}
        {%- for tool in tools %}
            {%- if tool.function is defined %}
                {%- set tool = tool.function %}
            {%- endif %}
            {{- "\\n<function>\\n<name>" ~ tool.name ~ "</name>" }}
            {%- if tool.description is defined %}
                {{- '\\n<description>' ~ (tool.description | trim) ~ '</description>' }}
            {%- endif %}
            {{- '\\n<parameters>' }}
            {%- if tool.parameters is defined and tool.parameters is mapping and tool.parameters.properties is defined and tool.parameters.properties is mapping %}
                {%- for param_name, param_fields in tool.parameters.properties|dictsort %}
                    {{- '\\n<parameter>' }}
                    {{- '\\n<name>' ~ param_name ~ '</name>' }}
                    {%- if param_fields.type is defined %}
                        {{- '\\n<type>' ~ (param_fields.type | string) ~ '</type>' }}
                    {%- endif %}
                    {%- if param_fields.description is defined %}
                        {{- '\\n<description>' ~ (param_fields.description | trim) ~ '</description>' }}
                    {%- endif %}
                    {%- set handled_keys = ['name', 'type', 'description'] %}
                    {{- render_extra_keys(param_fields, handled_keys) }}
                    {{- '\\n</parameter>' }}
                {%- endfor %}
            {%- endif %}
            {% set handled_keys = ['type', 'properties'] %}
            {{- render_extra_keys(tool.parameters, handled_keys) }}
            {{- '\\n</parameters>' }}
            {%- set handled_keys = ['type', 'name', 'description', 'parameters'] %}
            {{- render_extra_keys(tool, handled_keys) }}
            {{- '\\n</function>' }}
        {%- endfor %}
        {{- "\\n</tools>" }}
        {{- '\\nIf you choose to call a function ONLY reply in the following format with NO suffix:\\n\\n<tool_call>\\n<function=example_function_name>\\n<parameter=example_parameter_1>\\nvalue_1\\n</parameter>\\n<parameter=example_parameter_2>\\nThis is the value for the second parameter\\nthat can span\\nmultiple lines\\n</parameter>\\n</function>\\n</tool_call>\\n\\n<IMPORTANT>\\nReminder:\\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\\n- Required parameters MUST be specified\\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\\n</IMPORTANT>' }}
    {%- endif %}
    {%- if system_message is defined %}
        {{- '<|im_end|>\\n' }}
    {%- else %}
        {%- if tools is iterable and tools | length > 0 %}
            {{- '<|im_end|>\\n' }}
        {%- endif %}
    {%- endif %}
    {%- for message in loop_messages %}
        {%- if message.role == "assistant" and message.tool_calls is defined and message.tool_calls is iterable and message.tool_calls | length > 0 %}
            {{- '<|im_start|>' + message.role }}
            {%- if message.content is defined and message.content is string and message.content | trim | length > 0 %}
                {{- '\\n' + message.content | trim + '\\n' }}
            {%- endif %}
            {%- for tool_call in message.tool_calls %}
                {%- if tool_call.function is defined %}
                    {%- set tool_call = tool_call.function %}
                {%- endif %}
                {{- '\\n<tool_call>\\n<function=' + tool_call.name + '>\\n' }}
                {%- if tool_call.arguments is defined %}
                    {%- for args_name, args_value in tool_call.arguments|dictsort %}
                        {{- '<parameter=' + args_name + '>\\n' }}
                        {%- set args_value = args_value | tojson | safe if args_value is mapping or (args_value is sequence and args_value is not string) else args_value | string %}
                        {{- args_value }}
                        {{- '\\n</parameter>\\n' }}
                    {%- endfor %}
                {%- endif %}
                {{- '</function>\\n</tool_call>' }}
            {%- endfor %}
            {{- '<|im_end|>\\n' }}
        {%- elif message.role == "user" or message.role == "system" or message.role == "assistant" %}
            {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}
        {%- elif message.role == "tool" %}
            {%- if loop.previtem and loop.previtem.role != "tool" %}
                {{- '<|im_start|>user\\n' }}
            {%- endif %}
            {{- '<tool_response>\\n' }}
            {{- message.content }}
            {{- '\\n</tool_response>\\n' }}
            {%- if not loop.last and loop.nextitem.role != "tool" %}
                {{- '<|im_end|>\\n' }}
            {%- elif loop.last %}
                {{- '<|im_end|>\\n' }}
            {%- endif %}
        {%- else %}
            {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>\\n' }}
        {%- endif %}
    {%- endfor %}
    {%- if add_generation_prompt %}
        {{- '<|im_start|>assistant\\n' }}
    {%- endif %}
    """

    /// vLLM's tool_chat_template_hermes.jinja — ChatML format with <tool_call> JSON wrapping.
    /// Source: https://github.com/vllm-project/vllm/blob/main/examples/tool_chat_template_hermes.jinja
    static let hermesTemplate = """
    {%- macro json_to_python_type(json_spec) %}
        {%- set basic_type_map = {
        "string": "str",
        "number": "float",
        "integer": "int",
        "boolean": "bool"
    } %}
        {%- if basic_type_map[json_spec.type] is defined %}
            {{- basic_type_map[json_spec.type] }}
        {%- elif json_spec.type == "array" %}
            {{- "list[" +  json_to_python_type(json_spec|items) + "]" }}
        {%- elif json_spec.type == "object" %}
            {%- if json_spec.additionalProperties is defined %}
                {{- "dict[str, " + json_to_python_type(json_spec.additionalProperties) + ']' }}
            {%- else %}
                {{- "dict" }}
            {%- endif %}
        {%- elif json_spec.type is iterable %}
            {{- "Union[" }}
            {%- for t in json_spec.type %}
                {{- json_to_python_type({"type": t}) }}
                {%- if not loop.last %}
                    {{- "," }}
                {%- endif %}
            {%- endfor %}
            {{- "]" }}
        {%- else %}
            {{- "Any" }}
        {%- endif %}
    {%- endmacro %}

    {{- bos_token }}
    {{- "<|im_start|>system\\nYou are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags. You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions. Here are the available tools: <tools> " }}
    {%- if tools is iterable and tools | length > 0 %}
        {%- for tool in tools %}
            {%- if tool.function is defined %}
                {%- set tool = tool.function %}
            {%- endif %}
            {{- '{"type": "function", "function": ' }}
            {{- '{"name": "' + tool.name + '", ' }}
            {{- '"description": "' + tool.name + '(' }}
            {%- for param_name, param_fields in tool.parameters.properties|dictsort %}
                {{- param_name + ": " + json_to_python_type(param_fields) }}
                {%- if not loop.last %}
                    {{- ", " }}
                {%- endif %}
            {%- endfor %}
            {{- ")" }}
            {%- if tool.return is defined %}
                {{- " -> " + json_to_python_type(tool.return) }}
            {%- endif %}
            {{- " - " + tool.description + "\\n\\n" }}
            {%- for param_name, param_fields in tool.parameters.properties|dictsort %}
                {%- if loop.first %}
                    {{- "    Args:\\n" }}
                {%- endif %}
                {{- "        " + param_name + "(" + json_to_python_type(param_fields) + "): " + param_fields.description|trim }}
            {%- endfor %}
            {%- if tool.return is defined and tool.return.description is defined %}
                {{- "\\n    Returns:\\n        " + tool.return.description }}
            {%- endif %}
            {{- '"' }}
            {{- ', "parameters": ' }}
            {%- if tool.parameters.properties | length == 0 %}
                {{- "{}" }}
            {%- else %}
                {{- tool.parameters|tojson }}
            {%- endif %}
            {{- "}" }}
            {%- if not loop.last %}
                {{- "\\n" }}
            {%- endif %}
        {%- endfor %}
    {%- endif %}
    {{- " </tools>" }}
    {{- 'Use the following pydantic model json schema for each tool call you will make: {"properties": {"name": {"title": "Name", "type": "string"}, "arguments": {"title": "Arguments", "type": "object"}}, "required": ["name", "arguments"], "title": "FunctionCall", "type": "object"}\\n' }}
    {{- "For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows:\\n" }}
    {{- "<tool_call>\\n" }}
    {{- '{"name": <function-name>, "arguments": <args-dict>}\\n' }}
    {{- '</tool_call><|im_end|>' }}
    {%- for message in messages %}
        {%- if message.role == "user" or message.role == "system" or (message.role == "assistant" and message.tool_calls is not defined) %}
            {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}
        {%- elif message.role == "assistant" and message.tool_calls is defined %}
            {{- '<|im_start|>' + message.role }}
            {%- for tool_call in message.tool_calls %}
                {{- '\\n<tool_call>\\n' }}
                {%- if tool_call.function is defined %}
                    {%- set tool_call = tool_call.function %}
                {%- endif %}
                {{- '{' }}
                {{- '"name": "' }}
                {{- tool_call.name }}
                {{- '"' }}
                {%- if tool_call.arguments is defined %}
                    {{- ', ' }}
                    {{- '"arguments": ' }}
                    {{- tool_call.arguments|tojson }}
                {%- endif %}
                {{- '}' }}
                {{- '\\n</tool_call>' }}
            {%- endfor %}
            {{- '<|im_end|>\\n' }}
        {%- elif message.role == "tool" %}
            {%- if loop.previtem and loop.previtem.role != "tool" %}
                {{- '<|im_start|>tool\\n' }}
            {%- endif %}
            {{- '<tool_response>\\n' }}
            {{- message.content }}
            {%- if not loop.last %}
                {{- '\\n</tool_response>\\n' }}
            {%- else %}
                {{- '\\n</tool_response>' }}
            {%- endif %}
            {%- if not loop.last and loop.nextitem.role != "tool" %}
                {{- '<|im_end|>' }}
            {%- elif loop.last %}
                {{- '<|im_end|>' }}
            {%- endif %}
        {%- endif %}
    {%- endfor %}
    {%- if add_generation_prompt %}
        {{- '<|im_start|>assistant\\n' }}
    {%- endif %}
    """

    /// Adapted from vLLM's tool_chat_template_llama3.1_json.jinja — Llama 3.1/3.3 format.
    /// Modifications: wraps tool calls in <tool_call>/<\/tool_call> for streaming detection,
    /// uses "arguments" key, removes raise_exception, hardcodes date fallback.
    /// Source: https://github.com/vllm-project/vllm/blob/main/examples/tool_chat_template_llama3.1_json.jinja
    static let llama3JSONTemplate = """
    {{- bos_token }}
    {%- if custom_tools is defined %}
        {%- set tools = custom_tools %}
    {%- endif %}
    {%- if not tools_in_user_message is defined %}
        {%- set tools_in_user_message = true %}
    {%- endif %}
    {%- if not date_string is defined %}
        {%- set date_string = "23 Feb 2026" %}
    {%- endif %}
    {%- if not tools is defined %}
        {%- set tools = none %}
    {%- endif %}

    {%- if messages[0]['role'] == 'system' %}
        {%- if messages[0]['content'] is string %}
            {%- set system_message = messages[0]['content']|trim %}
        {%- else %}
            {%- set system_message = messages[0]['content'][0]['text']|trim %}
        {%- endif %}
        {%- set messages = messages[1:] %}
    {%- else %}
        {%- if tools is not none %}
            {%- set system_message = "You are a helpful assistant with tool calling capabilities. Only reply with a tool call if the function exists in the library provided by the user. If it doesn't exist, just reply directly in natural language. When you receive a tool call response, use the output to format an answer to the original user question." %}
        {%- else %}
            {%- set system_message = "" %}
        {%- endif %}
    {%- endif %}

    {{- "<|start_header_id|>system<|end_header_id|>\\n\\n" }}
    {%- if tools is not none %}
        {{- "Environment: ipython\\n" }}
    {%- endif %}
    {{- "Cutting Knowledge Date: December 2023\\n" }}
    {{- "Today Date: " + date_string + "\\n\\n" }}
    {%- if tools is not none and not tools_in_user_message %}
        {{- "You have access to the following functions. To call a function, please respond with JSON for a function call " }}
        {{- 'wrapped in <tool_call></tool_call> tags with the keys "name" and "arguments".\\n\\n' }}
        {%- for t in tools %}
            {{- t | tojson(indent=4) }}
            {{- "\\n\\n" }}
        {%- endfor %}
    {%- endif %}
    {{- system_message }}
    {{- "<|eot_id|>" }}

    {%- if tools_in_user_message and not tools is none %}
        {%- if messages | length != 0 %}
            {%- if messages[0]['content'] is string %}
                {%- set first_user_message = messages[0]['content']|trim %}
            {%- else %}
                {%- set first_user_message = messages[0]['content'] | selectattr('type', 'equalto', 'text') | map(attribute='text') | map('trim') | join('\\n') %}
            {%- endif %}
            {%- set messages = messages[1:] %}
        {%- else %}
            {%- set first_user_message = "" %}
        {%- endif %}
        {{- '<|start_header_id|>user<|end_header_id|>\\n\\n' -}}
        {{- "Given the following functions, please respond with a JSON for a function call " }}
        {{- 'wrapped in <tool_call></tool_call> tags with the keys "name" and "arguments".\\n\\n' }}
        {%- for t in tools %}
            {{- t | tojson(indent=4) }}
            {{- "\\n\\n" }}
        {%- endfor %}
        {{- first_user_message + "<|eot_id|>"}}
    {%- endif %}

    {%- for message in messages %}
        {%- if not (message.role == 'ipython' or message.role == 'tool' or 'tool_calls' in message) %}
            {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n' }}
            {%- if message['content'] is string %}
                {{- message['content'] | trim}}
            {%- else %}
                {%- for content in message['content'] %}
                    {%- if content['type'] == 'text' %}
                        {{- content['text'] | trim }}
                    {%- endif %}
                {%- endfor %}
            {%- endif %}
            {{- '<|eot_id|>' }}
        {%- elif 'tool_calls' in message %}
            {%- set tool_call = message.tool_calls[0].function %}
            {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' -}}
            {{- '<tool_call>\\n' }}
            {{- '{"name": "' + tool_call.name + '", ' }}
            {{- '"arguments": ' }}
            {{- tool_call.arguments | tojson }}
            {{- '}\\n' }}
            {{- '</tool_call>' }}
            {{- "<|eot_id|>" }}
        {%- elif message.role == "tool" or message.role == "ipython" %}
            {{- "<|start_header_id|>ipython<|end_header_id|>\\n\\n" }}
            {%- if message.content is string %}
                {{- message.content }}
            {%- else %}
                {%- for content in message['content'] %}
                    {%- if content['type'] == 'text' %}
                        {{- content['text'] }}
                    {%- endif %}
                {%- endfor %}
            {%- endif %}
            {{- "<|eot_id|>" }}
        {%- endif %}
    {%- endfor %}
    {%- if add_generation_prompt %}
        {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}
    {%- endif %}
    """

    /// Adapted from vLLM's tool_chat_template_mistral.jinja — Mistral v7 format.
    /// Modifications: wraps tool calls in <tool_call>/<\/tool_call> instead of [TOOL_CALLS],
    /// removes raise_exception calls, simplified tool_call_id handling.
    /// Source: https://github.com/vllm-project/vllm/blob/main/examples/tool_chat_template_mistral.jinja
    static let mistralTemplate = """
    {%- if messages[0]["role"] == "system" %}
        {%- set system_message = messages[0]["content"] %}
        {%- set loop_messages = messages[1:] %}
    {%- else %}
        {%- set loop_messages = messages %}
    {%- endif %}
    {%- if not tools is defined %}
        {%- set tools = none %}
    {%- endif %}
    {%- set user_messages = loop_messages | selectattr("role", "equalto", "user") | list %}

    {{- bos_token }}
    {%- for message in loop_messages %}
        {%- if message["role"] == "user" %}
            {%- if tools is not none and (message == user_messages[-1]) %}
                {{- "[AVAILABLE_TOOLS] [" }}
                {%- for tool in tools %}
                    {%- set tool = tool.function %}
                    {{- '{"type": "function", "function": {' }}
                    {%- for key, val in tool.items() if key != "return" %}
                        {%- if val is string %}
                            {{- '"' + key + '": "' + val + '"' }}
                        {%- else %}
                            {{- '"' + key + '": ' + val|tojson }}
                        {%- endif %}
                        {%- if not loop.last %}
                            {{- ", " }}
                        {%- endif %}
                    {%- endfor %}
                    {{- "}}" }}
                    {%- if not loop.last %}
                        {{- ", " }}
                    {%- else %}
                        {{- "]" }}
                    {%- endif %}
                {%- endfor %}
                {{- "[/AVAILABLE_TOOLS]" }}
            {%- endif %}
            {%- if loop.last and system_message is defined %}
                {{- "[INST] " + system_message + "\\n\\n" + message["content"] + "[/INST]" }}
            {%- else %}
                {{- "[INST] " + message["content"] + "[/INST]" }}
            {%- endif %}
        {%- elif message["role"] == "tool_calls" or message.tool_calls is defined %}
            {%- if message.tool_calls is defined %}
                {%- set tool_calls = message.tool_calls %}
            {%- else %}
                {%- set tool_calls = message.content %}
            {%- endif %}
            {%- for tool_call in tool_calls %}
                {{- "\\n<tool_call>\\n" }}
                {%- if tool_call.function is defined %}
                    {{- tool_call.function|tojson }}
                {%- else %}
                    {{- tool_call|tojson }}
                {%- endif %}
                {{- "\\n</tool_call>" }}
            {%- endfor %}
            {{- eos_token }}
        {%- elif message["role"] == "assistant" %}
            {{- " " + message["content"] + eos_token }}
        {%- elif message["role"] == "tool_results" or message["role"] == "tool" %}
            {%- if message.content is defined and message.content.content is defined %}
                {%- set content = message.content.content %}
            {%- else %}
                {%- set content = message.content %}
            {%- endif %}
            {{- '[TOOL_RESULTS] {"content": ' + content|string + '}[/TOOL_RESULTS]' }}
        {%- endif %}
    {%- endfor %}
    """

}
