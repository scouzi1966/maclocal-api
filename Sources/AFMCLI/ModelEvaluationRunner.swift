import AFMKit
import ArgumentParser
import Darwin
import Foundation

nonisolated(unsafe) private var afmEvaluationInterrupted = false

private func handleEvaluationInterrupt(_ signal: Int32) {
    afmEvaluationInterrupted = true
    FileHandle.standardError.write(Data("\nEvaluation interruption requested; preserving partial results…\n".utf8))
}

private struct EvaluationGeneration {
    let content: String
    let reasoning: String?
    let toolCalls: [AFMEvaluationToolCall]
    let promptTokens: Int
    let cachedPromptTokens: Int
    let completionTokens: Int
    let finishReason: String
    let promptTime: Double?
    let generationTime: Double?
    let timeToFirstToken: Double?
}

extension MlxCommand {
    func handleEvaluationManagement(_ action: AFMEvaluationCLIAction) throws -> Bool {
        let store = AFMEvaluationSuiteStore()
        switch action {
        case .none, .run:
            return false
        case .list:
            let suites = try store.discover()
            print("Available evaluation suites:")
            for suite in suites {
                print("  \(suite.name) [\(suite.origin.rawValue), \(suite.caseCount) cases]")
                print("    \(suite.description)")
            }
            print("Custom suites: \(store.rootDirectory.path)/*.json")
            return true
        case .scaffold(let name):
            let url = try store.scaffold(named: name)
            print("Created custom evaluation suite: \(url.path)")
            print("Validate it with: afm mlx --eval-validate \(shellQuote(url.path))")
            return true
        case .validate(let reference):
            let suite = try store.load(reference: reference)
            print("Valid evaluation suite '\(suite.name)' (\(suite.cases.count) cases).")
            return true
        }
    }

    func runEvaluation(modelID: String, suites suiteNames: [String], openReport: Bool) throws {
        let store = AFMEvaluationSuiteStore()
        let suites = try suiteNames.map { try store.load(named: $0) }
        let resultDirectory = try store.makeRunDirectory(model: modelID, suites: suiteNames)
        let resultsURL = resultDirectory.appendingPathComponent("results.jsonl")
        let runURL = resultDirectory.appendingPathComponent("run.json")
        let reportURL = resultDirectory.appendingPathComponent("report.html")
        let logURL = resultDirectory.appendingPathComponent("eval.log")
        let suiteSnapshotURL = resultDirectory.appendingPathComponent("suites.json")
        FileManager.default.createFile(atPath: resultsURL.path, contents: nil)
        FileManager.default.createFile(atPath: logURL.path, contents: nil)
        try AFMEvaluationReportWriter.jsonEncoder().encode(suites)
            .write(to: suiteSnapshotURL, options: [.atomic])

        let baseParameters = AFMEvaluationParameters(
            temperature: temperature ?? 0,
            maxTokens: maxTokens ?? 256,
            topP: topP,
            topK: topK,
            minP: minP,
            repetitionPenalty: repetitionPenalty,
            presencePenalty: presencePenalty,
            seed: seed ?? 42,
            logprobs: maxLogprobs != nil,
            topLogprobs: maxLogprobs,
            stop: stop.map { $0.split(separator: ",").map { String($0).trimmingCharacters(in: .whitespaces) } }
        )
        let engine = AFMEngine(
            backend: .mlx(modelID: modelID),
            config: EngineConfig(
                instructions: instructions,
                kvBits: kvBits,
                enablePrefixCaching: enablePrefixCaching,
                mlxKernels: mlxKernels,
                mtpEnabled: mtp,
                mtpDepth: mtpDepth,
                mtpModelID: mtpModel,
                eagle3DrafterPath: eagle3,
                enableGrammarConstraints: enableGrammarConstraints,
                toolCallParser: toolCallParser,
                maxConcurrent: 0,
                prefillStepSize: prefillStepSize,
                kvEvictionPolicy: kvEviction ?? "none",
                fixToolArguments: fixToolArgs,
                forceVLM: false,
                cacheProfilePath: cacheProfilePath,
                trace: vv,
                gpuCapturePath: gpuCapture,
                gpuTraceDuration: gpuTrace,
                gpuProfile: gpuProfile || gpuProfileBw,
                gpuProfileBandwidth: gpuProfileBw))

        let startedAt = Date()
        let reproducibilityCommand = makeEvaluationReproducibilityCommand(
            modelID: modelID,
            suites: suiteNames,
            openReport: openReport)
        let systemInfo = Self.evaluationSystemInfo()
        let output = SendableBox<Result<[AFMEvaluationCaseResult], Error>?>(nil)
        let group = DispatchGroup()
        group.enter()

        afmEvaluationInterrupted = false
        signal(SIGINT, handleEvaluationInterrupt)
        signal(SIGTERM, handleEvaluationInterrupt)
        print("AFM evaluation → \(resultDirectory.path)")
        print("Loading \(modelID) once for \(suites.reduce(0) { $0 + $1.cases.count }) cases…")

        Task {
            var results: [AFMEvaluationCaseResult] = []
            do {
                let reporter = MLXLoadReporter(modelID: modelID)
                reporter.start()
                _ = try await engine.load(progress: { fraction in
                    let progress = Progress(totalUnitCount: 1_000)
                    progress.completedUnitCount = Int64(fraction * 1_000)
                    reporter.updateDownload(progress)
                })
                reporter.finish(success: true)

                let total = suites.reduce(0) { $0 + $1.cases.count }
                var index = 0
                for suite in suites {
                    for testCase in suite.cases {
                        if afmEvaluationInterrupted { break }
                        index += 1
                        print("[\(index)/\(total)] \(suite.name)/\(testCase.id)…", terminator: " ")
                        fflush(stdout)
                        let parameters = baseParameters
                            .merging(suite.defaults)
                            .merging(testCase.parameters)
                        let caseStart = Date()
                        do {
                            var messages: [Message] = []
                            let systemPrompt = testCase.system ?? instructions
                            if !systemPrompt.isEmpty {
                                messages.append(Message(role: "system", content: systemPrompt))
                            }
                            if let developer = testCase.developer, !developer.isEmpty {
                                messages.append(Message(role: "developer", content: developer))
                            }
                            messages.append(Message(role: "user", content: testCase.prompt))
                            let generationConfig = GenerationConfig(
                                temperature: parameters.temperature,
                                maxTokens: parameters.maxTokens,
                                topP: parameters.topP,
                                topK: parameters.topK,
                                minP: parameters.minP,
                                repetitionPenalty: parameters.repetitionPenalty,
                                presencePenalty: parameters.presencePenalty,
                                seed: parameters.seed,
                                logprobs: parameters.logprobs,
                                topLogprobs: parameters.topLogprobs,
                                stop: parameters.stop,
                                tools: parameters.tools,
                                responseFormat: parameters.responseFormat)
                            let generated = try await Self.generateEvaluationResponse(
                                engine: engine,
                                messages: messages,
                                config: generationConfig,
                                streaming: parameters.streaming == true,
                                startedAt: caseStart)
                            let duration = Date().timeIntervalSince(caseStart)
                            var scoring = AFMEvaluationScorer.score(
                                output: generated.content,
                                toolCallNames: generated.toolCalls.map(\.name),
                                expectations: testCase.expectations)
                            if let matchID = testCase.expectations?.matchesCase {
                                let matchingOutput = results.last { $0.caseID == matchID }?.output
                                let passed = matchingOutput == generated.content
                                let check = AFMEvaluationCheckResult(
                                    name: "matchesCase",
                                    passed: passed,
                                    detail: "Output matches case '\(matchID)'")
                                scoring.1.append(check)
                                scoring.0 = scoring.1.allSatisfy(\.passed) ? .passed : .missed
                            }
                            let tokensPerSecond = generated.completionTokens > 0
                                ? Double(generated.completionTokens) / max(generated.generationTime ?? duration, 0.000_001)
                                : nil
                            let result = AFMEvaluationCaseResult(
                                suite: suite.name,
                                caseID: testCase.id,
                                prompt: testCase.prompt,
                                system: testCase.system,
                                output: generated.content,
                                reasoning: generated.reasoning,
                                toolCalls: generated.toolCalls,
                                outcome: scoring.0,
                                checks: scoring.1,
                                error: nil,
                                startedAt: caseStart,
                                durationSeconds: duration,
                                timeToFirstTokenSeconds: generated.timeToFirstToken,
                                promptTimeSeconds: generated.promptTime,
                                generationTimeSeconds: generated.generationTime,
                                promptTokens: generated.promptTokens,
                                cachedPromptTokens: generated.cachedPromptTokens,
                                completionTokens: generated.completionTokens,
                                tokensPerSecond: tokensPerSecond,
                                finishReason: generated.finishReason,
                                parameters: parameters)
                            results.append(result)
                            try Self.appendJSONLine(result, to: resultsURL)
                            try Self.persistEvaluationReport(
                                results: results,
                                modelID: modelID,
                                suites: suiteNames,
                                startedAt: startedAt,
                                interrupted: afmEvaluationInterrupted,
                                reproducibilityCommand: reproducibilityCommand,
                                systemInfo: systemInfo,
                                runURL: runURL,
                                reportURL: reportURL)
                            print("\(scoring.0.rawValue) · \(generated.completionTokens) tok · \(String(format: "%.1f", tokensPerSecond ?? 0)) tok/s")
                        } catch {
                            let duration = Date().timeIntervalSince(caseStart)
                            let result = AFMEvaluationCaseResult(
                                suite: suite.name,
                                caseID: testCase.id,
                                prompt: testCase.prompt,
                                system: testCase.system,
                                output: "",
                                reasoning: nil,
                                toolCalls: [],
                                outcome: .error,
                                checks: [],
                                error: error.localizedDescription,
                                startedAt: caseStart,
                                durationSeconds: duration,
                                timeToFirstTokenSeconds: nil,
                                promptTimeSeconds: nil,
                                generationTimeSeconds: nil,
                                promptTokens: 0,
                                cachedPromptTokens: 0,
                                completionTokens: 0,
                                tokensPerSecond: nil,
                                finishReason: "error",
                                parameters: parameters)
                            results.append(result)
                            try Self.appendJSONLine(result, to: resultsURL)
                            try Self.appendLog("\(suite.name)/\(testCase.id): \(error.localizedDescription)", to: logURL)
                            try Self.persistEvaluationReport(
                                results: results,
                                modelID: modelID,
                                suites: suiteNames,
                                startedAt: startedAt,
                                interrupted: afmEvaluationInterrupted,
                                reproducibilityCommand: reproducibilityCommand,
                                systemInfo: systemInfo,
                                runURL: runURL,
                                reportURL: reportURL)
                            print("error")
                        }
                    }
                    if afmEvaluationInterrupted { break }
                }
                try Self.persistEvaluationReport(
                    results: results,
                    modelID: modelID,
                    suites: suiteNames,
                    startedAt: startedAt,
                    interrupted: afmEvaluationInterrupted,
                    reproducibilityCommand: reproducibilityCommand,
                    systemInfo: systemInfo,
                    runURL: runURL,
                    reportURL: reportURL)
                output.value = .success(results)
            } catch {
                try? Self.appendLog("Infrastructure failure: \(error.localizedDescription)", to: logURL)
                try? Self.persistEvaluationReport(
                    results: results,
                    modelID: modelID,
                    suites: suiteNames,
                    startedAt: startedAt,
                    interrupted: afmEvaluationInterrupted,
                    reproducibilityCommand: reproducibilityCommand,
                    systemInfo: systemInfo,
                    runURL: runURL,
                    reportURL: reportURL)
                output.value = .failure(error)
            }
            await engine.unload()
            group.leave()
        }
        group.wait()

        switch output.value {
        case .success(let results):
            let errors = results.filter { $0.outcome == .error }.count
            let misses = results.filter { $0.outcome == .missed }.count
            print("Report: \(reportURL.path)")
            print("Completed \(results.count) cases: \(misses) quality miss(es), \(errors) infrastructure error(s).")
            if openReport && !afmEvaluationInterrupted && errors == 0 {
                try Self.openReport(reportURL)
            }
            if afmEvaluationInterrupted || errors > 0 {
                throw ExitCode.failure
            }
        case .failure(let error):
            FileHandle.standardError.write(Data("Evaluation failed: \(error.localizedDescription)\nPartial artifacts: \(resultDirectory.path)\n".utf8))
            throw ExitCode.failure
        case .none:
            throw ExitCode.failure
        }
    }

    private func makeEvaluationReproducibilityCommand(
        modelID: String,
        suites: [String],
        openReport: Bool
    ) -> String {
        var args = ["afm", "mlx", "-m", shellQuote(modelID), "--eval"]
        if suites != ["comprehensive"] {
            for suite in suites { args += ["--eval-suite", shellQuote(suite)] }
        }
        if !openReport { args.append("--no-open") }
        if let kvBits { args += ["--kv-bits", String(kvBits)] }
        if enablePrefixCaching { args.append("--enable-prefix-caching") }
        if mtp { args.append("--mtp") }
        if let mtpModel { args += ["--mtp-model", shellQuote(mtpModel)] }
        if let eagle3 { args += ["--eagle3", shellQuote(eagle3)] }
        if let parser = toolCallParser { args += ["--tool-call-parser", shellQuote(parser)] }
        if enableGrammarConstraints { args.append("--enable-grammar-constraints") }
        if noThink { args.append("--no-think") }
        return args.joined(separator: " ")
    }

    private static func persistEvaluationReport(
        results: [AFMEvaluationCaseResult],
        modelID: String,
        suites: [String],
        startedAt: Date,
        interrupted: Bool,
        reproducibilityCommand: String,
        systemInfo: AFMEvaluationSystemInfo,
        runURL: URL,
        reportURL: URL
    ) throws {
        let report = AFMEvaluationRunReport(
            afmVersion: BuildInfo.fullVersion,
            model: modelID,
            suites: suites,
            startedAt: startedAt,
            finishedAt: Date(),
            interrupted: interrupted,
            reproducibilityCommand: reproducibilityCommand,
            system: systemInfo,
            results: results)
        try AFMEvaluationReportWriter.jsonEncoder().encode(report).write(to: runURL, options: [.atomic])
        try Data(AFMEvaluationReportWriter.html(for: report).utf8).write(to: reportURL, options: [.atomic])
    }

    private static func generateEvaluationResponse(
        engine: AFMEngine,
        messages: [Message],
        config: GenerationConfig,
        streaming: Bool,
        startedAt: Date
    ) async throws -> EvaluationGeneration {
        if !streaming {
            let response = try await engine.respond(to: messages, config)
            return EvaluationGeneration(
                content: response.content,
                reasoning: response.reasoningContent,
                toolCalls: (response.toolCalls ?? []).map {
                    AFMEvaluationToolCall(name: $0.function.name, arguments: $0.function.arguments)
                },
                promptTokens: response.promptTokens,
                cachedPromptTokens: response.cachedPromptTokens,
                completionTokens: response.completionTokens,
                finishReason: response.finishReason.rawValue,
                promptTime: numberMetadata(response.metadata["promptTime"]),
                generationTime: numberMetadata(response.metadata["generateTime"]),
                timeToFirstToken: nil)
        }

        var content = ""
        var reasoning = ""
        var toolCalls: [AFMEvaluationToolCall] = []
        var promptTokens = 0
        var cachedTokens = 0
        var completionTokens = 0
        var finishReason = "unknown"
        var promptTime: Double?
        var generationTime: Double?
        var timeToFirstToken: Double?
        for try await event in engine.streamEvents(to: messages, config) {
            switch event {
            case .text(let text, _):
                if timeToFirstToken == nil, !text.isEmpty {
                    timeToFirstToken = Date().timeIntervalSince(startedAt)
                }
                content += text
            case .reasoning(let text, _):
                if timeToFirstToken == nil, !text.isEmpty {
                    timeToFirstToken = Date().timeIntervalSince(startedAt)
                }
                reasoning += text
            case .toolCall(let call, let stage):
                if case .completed = stage {
                    toolCalls.append(.init(name: call.name, arguments: call.arguments))
                }
            case .usage(let prompt, let completion, let cached):
                promptTokens = prompt
                completionTokens = completion
                cachedTokens = cached
            case .metadata(let metadata):
                promptTime = numberMetadata(metadata["promptTime"]) ?? promptTime
                generationTime = numberMetadata(metadata["generateTime"]) ?? generationTime
            case .completed(let reason):
                finishReason = reason.rawValue
            case .tokenLogprobs, .custom:
                break
            }
        }
        return EvaluationGeneration(
            content: content,
            reasoning: reasoning.isEmpty ? nil : reasoning,
            toolCalls: toolCalls,
            promptTokens: promptTokens,
            cachedPromptTokens: cachedTokens,
            completionTokens: completionTokens,
            finishReason: finishReason,
            promptTime: promptTime,
            generationTime: generationTime,
            timeToFirstToken: timeToFirstToken)
    }

    private static func appendJSONLine<T: Encodable>(_ value: T, to url: URL) throws {
        var data = try AFMEvaluationReportWriter.jsonEncoder(pretty: false).encode(value)
        data.append(0x0A)
        let handle = try FileHandle(forWritingTo: url)
        defer { try? handle.close() }
        try handle.seekToEnd()
        try handle.write(contentsOf: data)
    }

    private static func appendLog(_ value: String, to url: URL) throws {
        let handle = try FileHandle(forWritingTo: url)
        defer { try? handle.close() }
        try handle.seekToEnd()
        try handle.write(contentsOf: Data("[\(ISO8601DateFormatter().string(from: Date()))] \(value)\n".utf8))
    }

    private static func numberMetadata(_ value: AFMJSONValue?) -> Double? {
        switch value {
        case .number(let number): return number
        case .integer(let number): return Double(number)
        default: return nil
        }
    }

    private static func evaluationSystemInfo() -> AFMEvaluationSystemInfo {
        var system = utsname()
        uname(&system)
        let machine = withUnsafePointer(to: &system.machine) {
            $0.withMemoryRebound(to: CChar.self, capacity: 1) { String(cString: $0) }
        }
        return AFMEvaluationSystemInfo(
            operatingSystem: ProcessInfo.processInfo.operatingSystemVersionString,
            architecture: machine,
            processorCount: ProcessInfo.processInfo.processorCount,
            physicalMemoryBytes: ProcessInfo.processInfo.physicalMemory)
    }

    private static func openReport(_ url: URL) throws {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/open")
        process.arguments = [url.path]
        try process.run()
    }
}

private func shellQuote(_ value: String) -> String {
    if value.range(of: "^[A-Za-z0-9_./:+-]+$", options: .regularExpression) != nil {
        return value
    }
    return "'" + value.replacingOccurrences(of: "'", with: "'\\''") + "'"
}
