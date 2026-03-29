import Testing
import Foundation
@testable import MacLocalAPI

// Integration test — requires a running AFM server on localhost:9999
// Swift port of Scripts/feature-mlx-concurrent-batch/validate_responses.py
//
// Run with: swift test --filter ValidateResponsesTests
// Start server first:
//   MACAFM_MLX_MODEL_CACHE=/Volumes/edata/models/vesta-test-cache \
//   afm mlx -m mlx-community/Qwen3.5-35B-A3B-4bit --port 9999 --concurrent 8

// MARK: - Validation cases

private struct ValidationCase {
    let prompt: String
    let expected: [String]   // At least one must appear (case-insensitive)
    let description: String
}

private let VALIDATIONS: [ValidationCase] = [
    ValidationCase(
        prompt: "What is 2+2? Answer with just the number.",
        expected: ["4"],
        description: "basic arithmetic"
    ),
    ValidationCase(
        prompt: "What is the capital of France? Answer in one word.",
        expected: ["paris"],
        description: "capital of France"
    ),
    ValidationCase(
        prompt: "What is the chemical symbol for water? Answer in one word.",
        expected: ["h2o"],
        description: "chemical formula"
    ),
    ValidationCase(
        prompt: "Name the largest planet in our solar system. Answer in one word.",
        expected: ["jupiter"],
        description: "largest planet"
    ),
    ValidationCase(
        prompt: "What color do you get when you mix red and blue? Answer in one word.",
        expected: ["purple", "violet"],
        description: "color mixing"
    ),
    ValidationCase(
        prompt: "In what year did World War II end? Answer with just the year.",
        expected: ["1945"],
        description: "WWII end year"
    ),
    ValidationCase(
        prompt: "What is the square root of 144? Answer with just the number.",
        expected: ["12"],
        description: "square root"
    ),
    ValidationCase(
        prompt: "Who wrote Romeo and Juliet? Answer with just the name.",
        expected: ["shakespeare"],
        description: "Shakespeare authorship"
    ),
]

// MARK: - Result types

private struct RequestResult {
    let text: String
    let elapsed: TimeInterval
}

private enum CheckResult {
    case pass(matched: String)
    case garbage(preview: String)
    case missing(expected: [String], preview: String)
}

// MARK: - Helpers

private func sendRequest(client: AFMClient, prompt: String, maxTokens: Int = 200) async throws -> RequestResult {
    var req = AFMClient.ChatRequest()
    req.messages = [AFMClient.userMessage(prompt)]
    req.maxTokens = maxTokens
    req.temperature = 0.3
    req.stream = true

    let start = Date()
    let stream = try await client.chatCompletionStream(req)
    var text = ""

    for try await chunk in stream {
        if let content = chunk.choices.first?.delta.content, !content.isEmpty {
            text += content
        }
        if let reasoning = chunk.choices.first?.delta.reasoningContent, !reasoning.isEmpty {
            text += reasoning
        }
    }

    return RequestResult(text: text, elapsed: Date().timeIntervalSince(start))
}

private func checkResponse(_ text: String, expected: [String]) -> CheckResult {
    let lower = text.lowercased()
    let isGarbage = text.trimmingCharacters(in: .whitespaces).count < 2
        || text.unicodeScalars.filter({ $0.value == 0xFFFD }).count > 5

    if isGarbage {
        let preview = String(text.prefix(80)).replacingOccurrences(of: "\n", with: " ")
        return .garbage(preview: preview)
    }
    for sub in expected {
        if lower.contains(sub.lowercased()) {
            return .pass(matched: sub)
        }
    }
    let preview = String(text.prefix(100)).replacingOccurrences(of: "\n", with: " ")
    return .missing(expected: expected, preview: preview)
}

// MARK: - Batch runner

private struct BatchStats {
    var passed: Int = 0
    var failed: Int = 0
}

private func runValidation(client: AFMClient, batchSize: Int) async -> BatchStats {
    print("\n" + String(repeating: "=", count: 70))
    print("  Validating B=\(batchSize)")
    print(String(repeating: "=", count: 70))

    var stats = BatchStats()

    // Send validations in batches of `batchSize`, cycling through cases
    var batchStart = 0
    while batchStart < VALIDATIONS.count {
        let batch = Array(VALIDATIONS[batchStart..<min(batchStart + batchSize, VALIDATIONS.count)])
        batchStart += batchSize

        // Fire batch concurrently
        let batchResults: [(ValidationCase, Result<RequestResult, Error>)] =
            await withTaskGroup(of: (Int, Result<RequestResult, Error>).self) { group in
                for (i, v) in batch.enumerated() {
                    group.addTask {
                        do {
                            let r = try await sendRequest(client: client, prompt: v.prompt)
                            return (i, .success(r))
                        } catch {
                            return (i, .failure(error))
                        }
                    }
                }
                var indexed = [(Int, Result<RequestResult, Error>)]()
                for await item in group { indexed.append(item) }
                // Restore original order
                return indexed.sorted(by: { $0.0 < $1.0 }).map { (batch[$0.0], $0.1) }
            }

        for (v, result) in batchResults {
            switch result {
            case .failure(let error):
                stats.failed += 1
                print("  FAIL  \(v.description): exception \(error.localizedDescription)")
            case .success(let r):
                let check = checkResponse(r.text, expected: v.expected)
                switch check {
                case .pass(let matched):
                    stats.passed += 1
                    let preview = String(r.text.prefix(60)).replacingOccurrences(of: "\n", with: " ")
                    print("  OK    \(v.description): found '\(matched)' (\(String(format: "%.1f", r.elapsed))s) | \(preview)...")
                case .garbage(let preview):
                    stats.failed += 1
                    print("  FAIL  \(v.description): garbage output '\(preview)...'")
                case .missing(let expected, let preview):
                    stats.failed += 1
                    print("  FAIL  \(v.description): expected \(expected) in '\(preview)...'")
                }
            }
        }
    }

    print(String(repeating: "=", count: 70))
    print("  B=\(batchSize): \(stats.passed)/\(stats.passed + stats.failed) passed")
    print(String(repeating: "=", count: 70))
    return stats
}

// MARK: - Test suite

@Suite("Validate Responses — Batch Correctness (port 9999)")
struct ValidateResponsesTests {

    let client = AFMClient(baseURL: "http://localhost:9999")

    /// Quick smoke test — single request, B=1.
    @Test("B=1 single-request coherence check")
    func batchSize1() async throws {
        guard await serverReachable() else { return }
        let stats = await runValidation(client: client, batchSize: 1)
        #expect(stats.failed == 0, "B=1: \(stats.failed) failures — output may be corrupt")
    }

    /// B=2: two concurrent requests.
    @Test("B=2 concurrent coherence check")
    func batchSize2() async throws {
        guard await serverReachable() else { return }
        let stats = await runValidation(client: client, batchSize: 2)
        #expect(stats.failed == 0, "B=2: \(stats.failed) failures — output may be corrupt")
    }

    /// B=4: four concurrent requests.
    @Test("B=4 concurrent coherence check")
    func batchSize4() async throws {
        guard await serverReachable() else { return }
        let stats = await runValidation(client: client, batchSize: 4)
        #expect(stats.failed == 0, "B=4: \(stats.failed) failures — output may be corrupt")
    }

    /// B=8: eight concurrent requests.
    @Test("B=8 concurrent coherence check")
    func batchSize8() async throws {
        guard await serverReachable() else { return }
        let stats = await runValidation(client: client, batchSize: 8)
        #expect(stats.failed == 0, "B=8: \(stats.failed) failures — output may be corrupt")
    }

    /// Full sweep B=1,2,4,8 in one test — matches Python script default behaviour.
    @Test("Full sweep B=1,2,4,8")
    func fullSweep() async throws {
        guard await serverReachable() else { return }
        var totalPassed = 0
        var totalFailed = 0
        for bs in [1, 2, 4, 8] {
            let stats = await runValidation(client: client, batchSize: bs)
            totalPassed += stats.passed
            totalFailed += stats.failed
            try await Task.sleep(nanoseconds: 500_000_000)
        }
        print("\n" + String(repeating: "=", count: 70))
        print("  TOTAL: \(totalPassed)/\(totalPassed + totalFailed) passed across 4 batch sizes")
        if totalFailed > 0 {
            print("  *** \(totalFailed) FAILURES — output may be corrupt ***")
        } else {
            print("  All responses coherent and correct.")
        }
        print(String(repeating: "=", count: 70))
        #expect(totalFailed == 0, "\(totalFailed) total failures across B=1,2,4,8")
    }

    // MARK: - Private

    private func serverReachable() async -> Bool {
        do {
            _ = try await client.health()
            return true
        } catch {
            print("  ⚠️  Server not reachable at localhost:9999 — skipping test")
            return false
        }
    }
}
