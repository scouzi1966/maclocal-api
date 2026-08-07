import AFMKitCore
import AFMKitDwarfStar
import ArgumentParser
import CryptoKit
import Foundation

private let dwarfStarCanonicalPrompt =
    "Count upward from 1, separated only by commas. Continue until stopped."

struct DwarfStarBenchmarkCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "dwarfstar-bench",
        abstract: "Benchmark AFMKit's in-process fixed-schedule DwarfStar runtime"
    )

    static let canonicalPrompt = dwarfStarCanonicalPrompt

    @Option(name: [.customShort("m"), .long], help: "Native DwarfStar GGUF checkpoint")
    var model: String

    @Option(name: .long, help: "Prompt sent as one user message")
    var prompt = dwarfStarCanonicalPrompt

    @Option(name: .long, help: "Generated tokens per measured run")
    var tokens = 256

    @Option(name: .long, help: "Number of measured runs")
    var runs = 3

    @Option(name: .long, help: "Generated tokens in the unmeasured warmup")
    var warmupTokens = 16

    @Option(name: .long, help: "DwarfStar context window")
    var context = 32_768

    @Option(name: .long, help: "DwarfStar prefill chunk size (0 uses runtime default)")
    var prefillChunk = 0

    @Option(name: .long, help: "GPU power percentage")
    var powerPercent = 100

    @Option(name: .long, help: "Expected full response SHA-256 for every measured run")
    var expectedSHA256: String?

    @Option(name: .long, help: "Text that every measured response must contain")
    var expectedSubstring: String?

    @Option(name: .long, help: "Text that no measured response may contain")
    var forbiddenSubstring: String?

    @Option(name: .long, help: "Optional JSON result path on persistent storage")
    var output: String?

    mutating func run() async throws {
        guard tokens > 0, runs > 0, warmupTokens >= 0, context > 0 else {
            throw ValidationError("tokens, runs, and context must be positive; warmup-tokens cannot be negative")
        }
        guard FileManager.default.fileExists(atPath: model) else {
            throw ValidationError("GGUF model does not exist at \(model)")
        }
        let runtimeModel = AFMDwarfStarModel(
            modelID: AFMModelID(rawValue: URL(fileURLWithPath: model).lastPathComponent),
            modelPath: model,
            contextWindow: context,
            prefillChunk: prefillChunk,
            powerPercent: powerPercent,
            runtime: AFMDwarfStarRuntimeCoordinator()
        )

        print("runtime: in-process DwarfStar (\(AFMDwarfStarRuntime.backendName))")
        print("model: \(model)")
        print("mapping: vanilla DwarfStar GGUF mmap")
        let loadStart = ContinuousClock.now
        _ = try await runtimeModel.load { progress in
            if progress >= 1 { print("model loaded") }
        }
        let loadSeconds = Self.seconds(since: loadStart)

        if warmupTokens > 0 {
            print("warmup: \(warmupTokens) tokens")
            _ = try await runtimeModel.respond(
                to: Self.request(prompt: prompt, tokens: warmupTokens)
            )
        }

        var measuredRuns: [RunResult] = []
        for index in 1...runs {
            let start = ContinuousClock.now
            let response = try await runtimeModel.respond(
                to: Self.request(prompt: prompt, tokens: tokens)
            )
            let elapsed = Self.seconds(since: start)
            let digest = Self.sha256(response.text)
            let wallTPS = elapsed > 0 ? Double(response.usage.outputTokens) / elapsed : 0
            let runtimeTPS: Double?
            if case .number(let value) = response.metadata["tokensPerSecond"] {
                runtimeTPS = value
            } else {
                runtimeTPS = nil
            }

            let result = RunResult(
                run: index,
                outputTokens: response.usage.outputTokens,
                elapsedSeconds: elapsed,
                wallTokensPerSecond: wallTPS,
                runtimeTokensPerSecond: runtimeTPS,
                finishReason: response.finishReason.rawValue,
                contentSHA256: digest
            )
            measuredRuns.append(result)
            print(
                String(
                    format: "run %d: %.2f wall tok/s, %.3fs, %d tokens, sha256=%@",
                    index, wallTPS, elapsed, response.usage.outputTokens, digest
                )
            )

            if let expectedSHA256,
               digest.caseInsensitiveCompare(expectedSHA256) != .orderedSame {
                throw ValidationError(
                    "run \(index) hash mismatch: expected \(expectedSHA256), got \(digest)"
                )
            }
            if let expectedSubstring, !response.text.contains(expectedSubstring) {
                throw ValidationError(
                    "run \(index) semantic mismatch: response does not contain \(String(reflecting: expectedSubstring)); preview=\(String(reflecting: String(response.text.prefix(160))))"
                )
            }
            if let forbiddenSubstring, response.text.contains(forbiddenSubstring) {
                throw ValidationError(
                    "run \(index) semantic mismatch: response contains forbidden text \(String(reflecting: forbiddenSubstring)); preview=\(String(reflecting: String(response.text.prefix(160))))"
                )
            }
        }

        await runtimeModel.unload()
        let average = measuredRuns.map(\.wallTokensPerSecond).reduce(0, +)
            / Double(measuredRuns.count)
        let summary = Summary(
            runtime: "in-process-dwarfstar",
            backend: AFMDwarfStarRuntime.backendName,
            model: model,
            prompt: prompt,
            tokens: tokens,
            warmupTokens: warmupTokens,
            context: context,
            loadSeconds: loadSeconds,
            averageWallTokensPerSecond: average,
            expectedSHA256: expectedSHA256,
            expectedSubstring: expectedSubstring,
            forbiddenSubstring: forbiddenSubstring,
            runs: measuredRuns
        )
        print(String(format: "average: %.2f wall tok/s", average))

        if let output {
            let url = URL(fileURLWithPath: output)
            try FileManager.default.createDirectory(
                at: url.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
            try encoder.encode(summary).write(to: url, options: .atomic)
            print("results: \(url.path)")
        }
    }

    private static func request(prompt: String, tokens: Int) -> AFMRequest {
        AFMRequest(
            messages: [.init(role: .user, text: prompt)],
            options: .init(temperature: 0, maximumResponseTokens: tokens)
        )
    }

    private static func sha256(_ text: String) -> String {
        SHA256.hash(data: Data(text.utf8))
            .map { String(format: "%02x", $0) }
            .joined()
    }

    private static func seconds(since start: ContinuousClock.Instant) -> Double {
        let duration = start.duration(to: .now)
        return Double(duration.components.seconds)
            + Double(duration.components.attoseconds) / 1_000_000_000_000_000_000
    }
}

private struct RunResult: Codable {
    let run: Int
    let outputTokens: Int
    let elapsedSeconds: Double
    let wallTokensPerSecond: Double
    let runtimeTokensPerSecond: Double?
    let finishReason: String
    let contentSHA256: String
}

private struct Summary: Codable {
    let runtime: String
    let backend: String
    let model: String
    let prompt: String
    let tokens: Int
    let warmupTokens: Int
    let context: Int
    let loadSeconds: Double
    let averageWallTokensPerSecond: Double
    let expectedSHA256: String?
    let expectedSubstring: String?
    let forbiddenSubstring: String?
    let runs: [RunResult]
}
