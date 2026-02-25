import ArgumentParser
import Foundation
import Darwin

// Global references for signal handling
private var globalServer: Server?
private var shouldKeepRunning = true

// Signal handler function
func handleShutdown(_ signal: Int32) {
    print("\n🛑 Received shutdown signal, shutting down...")
    globalServer?.shutdown()
    shouldKeepRunning = false
}

struct ServeCommand: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "serve",
        abstract: "Start the AFM server (default command)",
        discussion: "Starts the macOS server that exposes Apple's Foundation Models through OpenAI-compatible API"
    )
    
    @Option(name: .shortAndLong, help: "Port to run the server on")
    var port: Int = 9999
    
    @Option(name: [.customShort("H"), .long], help: "Hostname to bind server to")
    var hostname: String = "127.0.0.1"
    
    @Flag(name: .shortAndLong, help: "Enable verbose logging")
    var verbose: Bool = false

    @Flag(name: [.customShort("V"), .long], help: "Enable very verbose logging (full requests/responses and all parameters)")
    var veryVerbose: Bool = false
    
    @Flag(name: .long, help: "Disable streaming responses (streaming is enabled by default)")
    var noStreaming: Bool = false
    
    @Option(name: [.short, .long], help: "Custom instructions for the AI assistant")
    var instructions: String = "You are a helpful assistant"
    
    @Option(name: [.customShort("a"), .long], help: "Path to a .fmadapter file for LoRA adapter fine-tuning")
    var adapter: String?

    @Option(name: [.short, .long], help: "Temperature for response generation (0.0-1.0)")
    var temperature: Double?

    @Option(name: [.short, .long], help: "Sampling mode: 'greedy', 'random', 'random:top-p=<0.0-1.0>', 'random:top-k=<int>', with optional ':seed=<int>'")
    var randomness: String?

    @Flag(name: [.customShort("P"), .long], help: "Permissive guardrails for unsafe or inappropriate responses")
    var permissiveGuardrails: Bool = false

    @Option(name: .long, help: "Stop sequences - comma-separated strings where generation should stop (e.g., '###,END')")
    var stop: String?

    @Flag(name: [.customShort("w"), .long], help: "Enable webui and open in default browser")
    var webui: Bool = false

    @Flag(name: [.customShort("g"), .long], help: "Enable API gateway mode: discover and proxy to local LLM backends (Ollama, LM Studio, Jan, etc.)")
    var gateway: Bool = false

    @Option(name: .long, help: "Pre-warm the model on server startup for faster first response (y/n, default: y)")
    var prewarm: String = "y"

    func run() throws {
        // Validate temperature parameter
        if let temp = temperature {
            guard temp >= 0.0 && temp <= 1.0 else {
                throw ValidationError("Temperature must be between 0.0 and 1.0")
            }
        }

        // Validate randomness parameter
        if let rand = randomness {
            do {
                _ = try RandomnessConfig.parse(rand)
            } catch let error as FoundationModelError {
                throw ValidationError(error.localizedDescription)
            } catch {
                throw ValidationError("Invalid randomness parameter format")
            }
        }

        // Parse prewarm flag
        let prewarmEnabled = prewarm.lowercased() != "n" && prewarm.lowercased() != "no" && prewarm != "0"

        if verbose {
            print("Starting afm server with verbose logging enabled...")
        }

        // Use RunLoop to handle the server lifecycle properly
        let runLoop = RunLoop.current

        // Set up signal handling for graceful shutdown
        signal(SIGINT, handleShutdown)
        signal(SIGTERM, handleShutdown)

        // Start server in async context
        _ = Task {
            do {
                let server = try await Server(port: port, hostname: hostname, verbose: verbose, veryVerbose: veryVerbose, streamingEnabled: !noStreaming, instructions: instructions, adapter: adapter, temperature: temperature, randomness: randomness, permissiveGuardrails: permissiveGuardrails, stop: stop, webuiEnabled: webui, gatewayEnabled: gateway, prewarmEnabled: prewarmEnabled)
                globalServer = server
                try await server.start()
            } catch {
                print("Error starting server. CTRL-C to stop: \(error)")
                shouldKeepRunning = false
            }
        }

        // Keep the main thread alive until shutdown
        while shouldKeepRunning && runLoop.run(mode: .default, before: Date(timeIntervalSinceNow: 0.1)) {
            // Keep running until shutdown signal
        }

        print("Server shutdown complete.")
    }
}

struct MlxCommand: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "mlx",
        abstract: "Run local MLX LLM/VLM models via AFM",
        discussion: "Uses MLX Swift libraries + HuggingFace Hub. Model cache root can be overridden with MACAFM_MLX_MODEL_CACHE. Metallib path can be overridden with MACAFM_MLX_METALLIB."
    )

    @Option(name: [.customShort("m"), .long], help: "Model id (org/model or model). If org omitted, defaults to mlx-community.")
    var model: String?

    @Option(name: [.customShort("s"), .long], help: "Run a single prompt without starting the server")
    var singlePrompt: String?

    @Option(name: [.short, .long], help: "Custom instructions for the AI assistant")
    var instructions: String = "You are a helpful assistant"

    @Option(name: .shortAndLong, help: "Port to run server on (default: 9999, falls back to ephemeral if busy)")
    var port: Int?

    @Option(name: [.customShort("H"), .long], help: "Hostname to bind server to")
    var hostname: String = "127.0.0.1"

    @Flag(name: .shortAndLong, help: "Enable verbose logging")
    var verbose: Bool = false

    @Flag(name: [.customShort("V"), .long], help: "Enable very verbose logging (full requests/responses and all parameters)")
    var veryVerbose: Bool = false

    @Flag(name: .long, help: "Disable streaming responses (streaming is enabled by default)")
    var noStreaming: Bool = false

    @Flag(name: .long, help: "Output raw model content without extracting <think> tags into reasoning_content")
    var raw: Bool = false

    @Option(name: [.short, .long], help: "Temperature for response generation (0.0-2.0)")
    var temperature: Double?

    @Flag(name: [.customShort("w"), .long], help: "Enable webui and open in default browser")
    var webui: Bool = false

    @Flag(name: [.customShort("g"), .long], help: "Gateway mode is not supported in afm mlx")
    var gateway: Bool = false

    // Sampling parameters
    @Option(name: .long, help: "Top-p (nucleus) sampling threshold (0.0-1.0, default: 1.0)")
    var topP: Double?
    @Option(name: .long, help: "Top-k sampling: keep only the k most likely tokens (0 = disabled)")
    var topK: Int?
    @Option(name: .long, help: "Min-p sampling: filter tokens with probability < min_p * max_prob (0.0 = disabled)")
    var minP: Double?
    @Option(name: .long, help: "Presence penalty: flat additive penalty for tokens already generated (0.0 = disabled)")
    var presencePenalty: Double?
    @Option(name: .long, help: "Maximum tokens to generate per response (default: 8192)")
    var maxTokens: Int?
    @Option(name: .long, help: "Random seed for reproducible sampling (nil = non-deterministic)")
    var seed: Int?
    @Option(name: .long, help: "Maximum number of top log probabilities returned per token (default: 20)")
    var maxLogprobs: Int?
    @Option(name: .long, help: "Repetition penalty (compatibility)")
    var repetitionPenalty: Double?
    @Option(name: .long, help: "KV cache size (compatibility)")
    var maxKVSize: Int?
    @Option(name: .long, help: "KV bits (compatibility)")
    var kvBits: Int?
    @Option(name: .long, help: "Prefill step size — number of prompt tokens processed per GPU pass (default: 2048)")
    var prefillStepSize: Int?
    @Flag(name: .long, help: "Trust remote code (compatibility)")
    var trustRemoteCode: Bool = false
    @Flag(name: .long, inversion: .prefixedNo, help: "Enable automatic prefix caching (KV cache reuse across requests)")
    var enablePrefixCaching: Bool = false
    @Option(name: .long, help: "Chat template (compatibility)")
    var chatTemplate: String?
    @Option(name: .long, help: "Dtype (compatibility)")
    var dtype: String?
    @Flag(name: .long, help: "VLM hint (compatibility)")
    var vlm: Bool = false

    @Option(name: .long, help: "Stop sequences - comma-separated strings where generation should stop (e.g., '###,END')")
    var stop: String?

    @Option(name: .long, help: "Constrain output to match a JSON schema (vLLM-compatible)")
    var guidedJson: String?

    @Option(name: .long, help: "Tool call parser override: hermes, llama3_json, gemma, mistral, qwen3_xml. Forces a custom chat template and tool call format for reliable tool calling.")
    var toolCallParser: String?

    @Flag(name: .long, help: "Post-process tool call argument names to match the original tool schema (fixes model renaming e.g. path→filePath)")
    var fixToolArgs: Bool = false

    @Flag(name: .long, help: "Print OpenClaw provider config JSON and exit")
    var openclawConfig: Bool = false

    func run() throws {
        if gateway {
            print("Error: -g/--gateway is not supported in 'afm mlx' mode.")
            throw ExitCode.failure
        }

        emitCompatibilityWarnings()

        let resolver = MLXCacheResolver()
        let service = MLXModelService(resolver: resolver)
        service.enablePrefixCaching = enablePrefixCaching
        service.toolCallParser = toolCallParser
        service.fixToolArgs = fixToolArgs
        if let prefillStepSize { service.prefillStepSize = prefillStepSize }
        _ = try service.revalidateRegistry()

        let rawModel: String
        if let m = model, !m.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            rawModel = m
        } else if isatty(STDIN_FILENO) != 0 {
            let discovered = discoverAllModels(resolver: resolver)
            guard !discovered.isEmpty else {
                print("No models found locally. Use: afm mlx -m <org/model>")
                throw ExitCode.failure
            }
            guard let selected = runInteractiveModelPicker(models: discovered) else {
                throw ExitCode.failure
            }
            rawModel = selected
        } else {
            let registered = try service.revalidateRegistry()
            if !registered.isEmpty {
                print("No model provided. Available models in registry:")
                for m in registered {
                    print("  - \(m)")
                }
            } else {
                print("No model provided and registry is empty.")
                print("Use: afm mlx -m <org/model> ...")
            }
            throw ExitCode.failure
        }

        let selectedModel = service.normalizeModel(rawModel)

        if openclawConfig {
            let chosenPort = port ?? 9999
            printOpenClawConfig(model: selectedModel, hostname: hostname, port: chosenPort, resolver: resolver)
            return
        }

        print("MLX model: \(selectedModel)")
        try ensureMLXMetalLibraryAvailable(verbose: verbose)

        // Backward compatibility: support piped input in mlx mode too
        if let stdinContent = try readFromStdin() {
            try runSinglePrompt(stdinContent, service: service, modelID: selectedModel)
            return
        }

        if let prompt = singlePrompt {
            try runSinglePrompt(prompt, service: service, modelID: selectedModel)
            return
        }

        let explicitPort = port != nil
        let chosenPort: Int
        if let requested = port {
            chosenPort = requested
        } else if isPortAvailable(9999) {
            chosenPort = 9999
        } else {
            chosenPort = try findEphemeralPort()
            print("Port 9999 is busy, using ephemeral port \(chosenPort)")
        }

        if verbose {
            print("Loading MLX model (download if needed): \(selectedModel)")
        }

        _ = Task {
            do {
                let loadReporter = MLXLoadReporter(modelID: selectedModel)
                loadReporter.start()
                _ = try await service.ensureLoaded(
                    model: selectedModel,
                    progress: { p in loadReporter.updateDownload(p) },
                    stage: { s in loadReporter.updateStage(s) }
                )
                loadReporter.finish(success: true)
                let server = try await Server(
                    port: chosenPort,
                    hostname: hostname,
                    verbose: verbose,
                    veryVerbose: veryVerbose,
                    streamingEnabled: !noStreaming,
                    instructions: instructions,
                    adapter: nil,
                    temperature: temperature,
                    randomness: nil,
                    permissiveGuardrails: false,
                    stop: stop,
                    webuiEnabled: webui,
                    gatewayEnabled: false,
                    prewarmEnabled: false,
                    mlxModelID: selectedModel,
                    mlxModelService: service,
                    mlxRepetitionPenalty: repetitionPenalty,
                    mlxTopP: topP,
                    mlxMaxTokens: maxTokens,
                    mlxRawOutput: raw,
                    mlxTopK: topK,
                    mlxMinP: minP,
                    mlxPresencePenalty: presencePenalty,
                    mlxSeed: seed,
                    mlxMaxLogprobs: maxLogprobs
                )
                globalServer = server
                if !explicitPort && chosenPort != 9999 {
                    print("MLX API URL: http://\(hostname):\(chosenPort)")
                }
                try await server.start()
            } catch {
                MLXLoadReporter.finishActiveWithError(error.localizedDescription)
                print("Error starting MLX server. CTRL-C to stop: \(error)")
                shouldKeepRunning = false
            }
        }

        let runLoop = RunLoop.current
        signal(SIGINT, handleShutdown)
        signal(SIGTERM, handleShutdown)
        while shouldKeepRunning && runLoop.run(mode: .default, before: Date(timeIntervalSinceNow: 0.1)) {}
        print("Server shutdown complete.")
    }

    private func runSinglePrompt(_ prompt: String, service: MLXModelService, modelID: String) throws {
        defer {
            let cleanup = DispatchGroup()
            cleanup.enter()
            Task {
                await service.shutdownAndReleaseResources(verbose: verbose)
                cleanup.leave()
            }
            cleanup.wait()
        }

        let group = DispatchGroup()
        var output: Result<String, Error>?
        group.enter()
        Task {
            do {
                // Pre-load with progress bar (downloads if needed)
                let loadReporter = MLXLoadReporter(modelID: modelID)
                loadReporter.start()
                _ = try await service.ensureLoaded(
                    model: modelID,
                    progress: { p in loadReporter.updateDownload(p) },
                    stage: { s in loadReporter.updateStage(s) }
                )
                loadReporter.finish(success: true)

                var messages = [Message]()
                messages.append(Message(role: "system", content: self.instructions))
                messages.append(Message(role: "user", content: prompt))
                var responseFormat: ResponseFormat? = nil
                if let guidedJson = self.guidedJson {
                    let schema = try parseGuidedJsonSchema(guidedJson)
                    responseFormat = ResponseFormat(type: "json_schema", jsonSchema: schema)
                }
                let stopSequences: [String]? = stop.map { $0.split(separator: ",").map { String($0.trimmingCharacters(in: .whitespaces)) } }
                let res = try await service.generate(
                    model: modelID,
                    messages: messages,
                    temperature: temperature,
                    maxTokens: maxTokens,
                    topP: topP,
                    repetitionPenalty: repetitionPenalty,
                    stop: stopSequences,
                    responseFormat: responseFormat
                )
                output = .success(res.content)
            } catch {
                MLXLoadReporter.finishActiveWithError(error.localizedDescription)
                output = .failure(error)
            }
            group.leave()
        }
        group.wait()

        switch output {
        case .success(let text):
            print(text)
        case .failure(let error):
            print("Error: \(error.localizedDescription)")
            throw ExitCode.failure
        case .none:
            throw ExitCode.failure
        }
    }

    private func readFromStdin() throws -> String? {
        guard isatty(STDIN_FILENO) == 0 else { return nil }
        let data = FileHandle.standardInput.readDataToEndOfFile()
        guard !data.isEmpty else { return nil }
        guard let content = String(data: data, encoding: .utf8)?
            .trimmingCharacters(in: .whitespacesAndNewlines), !content.isEmpty else {
            throw ExitCode.failure
        }
        return content
    }

    private func printOpenClawConfig(model: String, hostname: String, port: Int, resolver: MLXCacheResolver) {
        // Auto-detect capabilities from cached config.json
        var supportsVision = false
        var supportsReasoning = false
        var contextWindow = 131072
        var defaultMaxTokens = 8192

        if let modelDir = resolver.localModelDirectory(repoId: model) {
            let configURL = modelDir.appendingPathComponent("config.json")
            if let data = try? Data(contentsOf: configURL),
               let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                // Vision: check for vision_config or visual key
                if json["vision_config"] != nil || json["visual"] != nil {
                    supportsVision = true
                }
                // Context window from max_position_embeddings
                if let maxPos = json["max_position_embeddings"] as? Int {
                    contextWindow = maxPos
                }
            }
            // Reasoning: check chat_template for <think> tags
            let templateURL = modelDir.appendingPathComponent("tokenizer_config.json")
            if let data = try? Data(contentsOf: templateURL),
               let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                if let chatTemplate = json["chat_template"] as? String, chatTemplate.contains("<think>") {
                    supportsReasoning = true
                }
            }
            // Reasoning: check generation_config.json for enable_thinking
            let genConfigURL = modelDir.appendingPathComponent("generation_config.json")
            if let data = try? Data(contentsOf: genConfigURL),
               let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
               let enableThinking = json["enable_thinking"] as? Bool, enableThinking {
                supportsReasoning = true
            }
        }

        // Fallback: detect reasoning from known model family name patterns
        if !supportsReasoning {
            let lower = model.lowercased()
            let reasoningPatterns = [
                "qwen3", "deepseek-r", "glm-4", "glm-5", "kimi",
                "qwq", "marco-o1", "skywork-o1", "ling-",
                "nemotron", "minimax", "gpt-oss"
            ]
            supportsReasoning = reasoningPatterns.contains(where: { lower.contains($0) })
        }

        // Short model name (strip org prefix for display)
        let shortName = model.contains("/") ? String(model.split(separator: "/", maxSplits: 1).last!) : model

        var input: [String] = ["text"]
        if supportsVision { input.append("image") }

        let config: [String: Any] = [
            "models": [
                "providers": [
                    "afm": [
                        "baseUrl": "http://\(hostname):\(port)/v1",
                        "apiKey": "not-needed",
                        "api": "openai-completions",
                        "models": [[
                            "id": shortName,
                            "name": "\(shortName) (afm)",
                            "reasoning": supportsReasoning,
                            "input": input,
                            "cost": ["input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0],
                            "contextWindow": contextWindow,
                            "maxTokens": defaultMaxTokens
                        ] as [String: Any]]
                    ] as [String: Any]
                ]
            ],
            "agents": [
                "defaults": [
                    "model": ["primary": "afm/\(shortName)"]
                ]
            ]
        ]

        if let jsonData = try? JSONSerialization.data(withJSONObject: config, options: [.prettyPrinted, .sortedKeys]),
           let jsonString = String(data: jsonData, encoding: .utf8) {
            print(jsonString)
        }
    }

    private func emitCompatibilityWarnings() {
        var ignored: [String] = []
        if maxKVSize != nil { ignored.append("--max-kv-size") }
        if kvBits != nil { ignored.append("--kv-bits") }
        if trustRemoteCode { ignored.append("--trust-remote-code") }
        if chatTemplate != nil { ignored.append("--chat-template") }
        if dtype != nil { ignored.append("--dtype") }
        if vlm { ignored.append("--vlm") }
        if !ignored.isEmpty {
            print("Warning: accepted compatibility switches currently ignored: \(ignored.joined(separator: ", "))")
        }
    }
}

struct MacLocalAPI: ParsableCommand {
    static let buildVersion: String = BuildInfo.fullVersion
    
    static let configuration = CommandConfiguration(
        commandName: "afm",
        abstract: "macOS server that exposes Apple's Foundation Models through OpenAI-compatible API",
        discussion: "Use -w to enable the WebUI, -g to enable API gateway mode, or `afm mlx` for local MLX models.\n\nGitHub: https://github.com/scouzi1966/maclocal-api",
        version: buildVersion
    )
}

struct RootCommand: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "afm",
        abstract: "macOS server that exposes Apple's Foundation Models through OpenAI-compatible API",
        discussion: "Use -w to enable the WebUI, -g to enable API gateway mode, or `afm mlx` for local MLX models.\n\nGitHub: https://github.com/scouzi1966/maclocal-api",
        version: MacLocalAPI.buildVersion,
        subcommands: [MlxCommand.self, VisionCommand.self]
    )

    @Option(name: [.customShort("s"), .long], help: "Run a single prompt without starting the server")
    var singlePrompt: String?

    @Option(name: [.short, .long], help: "Custom instructions for the AI assistant")
    var instructions: String = "You are a helpful assistant"

    @Flag(name: .shortAndLong, help: "Enable verbose logging")
    var verbose: Bool = false

    @Flag(name: [.customShort("V"), .long], help: "Enable very verbose logging (full requests/responses and all parameters)")
    var veryVerbose: Bool = false

    @Flag(name: .long, help: "Disable streaming responses (streaming is enabled by default)")
    var noStreaming: Bool = false

    @Option(name: [.customShort("a"), .long], help: "Path to a .fmadapter file for LoRA adapter fine-tuning")
    var adapter: String?

    @Option(name: .shortAndLong, help: "Port to run the server on")
    var port: Int = 9999

    @Option(name: [.customShort("H"), .long], help: "Hostname to bind server to")
    var hostname: String = "127.0.0.1"

    @Option(name: [.short, .long], help: "Temperature for response generation (0.0-1.0)")
    var temperature: Double?

    @Option(name: [.short, .long], help: "Sampling mode: 'greedy', 'random', 'random:top-p=<0.0-1.0>', 'random:top-k=<int>', with optional ':seed=<int>'")
    var randomness: String?

    @Flag(name: [.customShort("P"), .long], help: "Permissive guardrails for unsafe or inappropriate responses")
    var permissiveGuardrails: Bool = false

    @Flag(name: [.customShort("w"), .long], help: "Enable webui and open in default browser")
    var webui: Bool = false

    @Flag(name: [.customShort("g"), .long], help: "Enable API gateway mode: discover and proxy to local LLM backends (Ollama, LM Studio, Jan, etc.)")
    var gateway: Bool = false

    @Option(name: .long, help: "Constrain output to match a JSON schema (vLLM-compatible)")
    var guidedJson: String?

    @Option(name: .long, help: "Stop sequences - comma-separated strings where generation should stop (e.g., '###,END')")
    var stop: String?

    @Option(name: .long, help: "Pre-warm the model on server startup for faster first response (y/n, default: y)")
    var prewarm: String = "y"

    func run() throws {
        // Validate temperature parameter
        if let temp = temperature {
            guard temp >= 0.0 && temp <= 1.0 else {
                throw ValidationError("Temperature must be between 0.0 and 1.0")
            }
        }

        // Validate randomness parameter
        if let rand = randomness {
            do {
                _ = try RandomnessConfig.parse(rand)
            } catch let error as FoundationModelError {
                throw ValidationError(error.localizedDescription)
            } catch {
                throw ValidationError("Invalid randomness parameter format")
            }
        }

        // Handle single-prompt mode for backward compatibility
        if let prompt = singlePrompt {
            return try runSinglePrompt(prompt, adapter: adapter)
        }

        // Check for piped input for backward compatibility
        if let stdinContent = try readFromStdin() {
            return try runSinglePrompt(stdinContent, adapter: adapter)
        }

        // If no subcommand specified and no single prompt, run server
        var serveCommand = ServeCommand()
        serveCommand.port = port
        serveCommand.hostname = hostname
        serveCommand.verbose = verbose
        serveCommand.veryVerbose = veryVerbose
        serveCommand.noStreaming = noStreaming
        serveCommand.instructions = instructions
        serveCommand.adapter = adapter
        serveCommand.temperature = temperature
        serveCommand.randomness = randomness
        serveCommand.permissiveGuardrails = permissiveGuardrails
        serveCommand.stop = stop
        serveCommand.webui = webui
        serveCommand.gateway = gateway
        serveCommand.prewarm = prewarm
        try serveCommand.run()
    }
}

// Manual dispatch for subcommands to avoid flag conflicts between root and subcommands.
// Subcommands are still registered in RootCommand.configuration so they appear in -h.
if CommandLine.arguments.count > 1 && CommandLine.arguments[1] == "mlx" {
    let args = Array(CommandLine.arguments.dropFirst(2))
    do {
        var cmd = try MlxCommand.parseAsRoot(args)
        try cmd.run()
    } catch {
        MlxCommand.exit(withError: error)
    }
} else if CommandLine.arguments.count > 1 && CommandLine.arguments[1] == "vision" {
    let args = Array(CommandLine.arguments.dropFirst(2))
    do {
        let cmd = try VisionCommand.parse(args)
        let group = DispatchGroup()
        var caughtError: Error?
        group.enter()
        Task {
            do {
                try await cmd.run()
            } catch {
                caughtError = error
            }
            group.leave()
        }
        group.wait()
        if let error = caughtError {
            throw error
        }
    } catch {
        VisionCommand.exit(withError: error)
    }
} else {
    RootCommand.main()
}

private func ensureMLXMetalLibraryAvailable(verbose: Bool) throws {
    let fileManager = FileManager.default
    let executableURL = URL(fileURLWithPath: CommandLine.arguments[0]).resolvingSymlinksInPath()
    let executableDir = executableURL.deletingLastPathComponent()

    // Resolution order:
    // 1. MACAFM_MLX_METALLIB env var (explicit override)
    // 2. Bundle.module resource (standard Swift PM — works from build dir and when bundle is co-located)
    // 3. MacLocalAPI_MacLocalAPI.bundle/ next to executable (relocated binary with bundle)
    // MLX loads kernels from default.metallib in the cwd, so we always chdir.

    let env = ProcessInfo.processInfo.environment
    let explicit = env["MACAFM_MLX_METALLIB"].flatMap { raw -> URL? in
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : URL(fileURLWithPath: trimmed)
    }
    let packaged = Bundle.module.url(forResource: "default", withExtension: "metallib")
    let bundleRelative: URL? = {
        let candidate = executableDir
            .appendingPathComponent("MacLocalAPI_MacLocalAPI.bundle")
            .appendingPathComponent("default.metallib")
        return fileManager.fileExists(atPath: candidate.path) ? candidate : nil
    }()

    guard let source = explicit ?? packaged ?? bundleRelative,
          fileManager.fileExists(atPath: source.path) else {
        throw ValidationError(
            "MLX metallib not found. Ensure MacLocalAPI_MacLocalAPI.bundle is next to the afm binary."
        )
    }

    let metalDir = source.deletingLastPathComponent().path
    guard fileManager.changeCurrentDirectoryPath(metalDir) else {
        throw ValidationError("Failed to switch to metallib directory: \(metalDir)")
    }
    if verbose {
        print("Using MLX metallib: \(source.path)")
    }
}

private final class MLXLoadReporter {
    private static let reporterLock = NSLock()
    private static weak var activeReporter: MLXLoadReporter?

    private let modelID: String
    private let lock = NSLock()
    private var stage: MLXLoadStage = .checkingCache
    private var downloadFraction: Double?
    private var timer: DispatchSourceTimer?
    private var spinnerIndex: Int = 0
    private var startedAt = Date()
    private var finished = false

    private let spinnerFrames = ["|", "/", "-", "\\"]

    init(modelID: String) {
        self.modelID = modelID
    }

    func start() {
        Self.reporterLock.lock()
        Self.activeReporter = self
        Self.reporterLock.unlock()

        startedAt = Date()
        print("Loading MLX model: \(modelID)")

        let timer = DispatchSource.makeTimerSource(queue: DispatchQueue.global(qos: .utility))
        timer.schedule(deadline: .now(), repeating: .milliseconds(200))
        timer.setEventHandler { [weak self] in
            self?.renderTick()
        }
        self.timer = timer
        timer.resume()
    }

    func updateDownload(_ progress: Progress) {
        lock.lock()
        stage = .downloading
        downloadFraction = progress.totalUnitCount > 0 ? progress.fractionCompleted : nil
        lock.unlock()
    }

    func updateStage(_ stage: MLXLoadStage) {
        lock.lock()
        self.stage = stage
        if stage == .loadingModel || stage == .ready {
            downloadFraction = nil
        }
        lock.unlock()
    }

    func finish(success: Bool, errorMessage: String? = nil) {
        lock.lock()
        guard !finished else {
            lock.unlock()
            return
        }
        finished = true
        let elapsed = Date().timeIntervalSince(startedAt)
        timer?.cancel()
        timer = nil
        let memory = Self.currentResidentMemoryGB()
        lock.unlock()

        Self.reporterLock.lock()
        if Self.activeReporter === self {
            Self.activeReporter = nil
        }
        Self.reporterLock.unlock()

        let status = success ? "ready" : "failed"
        var line = String(
            format: "\r[%@] %@ | mem %.2f GB | %.1fs",
            success ? "done" : "fail",
            status,
            memory,
            elapsed
        )
        if let errorMessage, !errorMessage.isEmpty {
            line += " | \(errorMessage)"
        }
        print(line)
    }

    static func finishActiveWithError(_ message: String) {
        reporterLock.lock()
        let active = activeReporter
        reporterLock.unlock()
        active?.finish(success: false, errorMessage: message)
    }

    private func renderTick() {
        lock.lock()
        if finished {
            lock.unlock()
            return
        }
        let stage = self.stage
        let downloadFraction = self.downloadFraction
        spinnerIndex = (spinnerIndex + 1) % spinnerFrames.count
        let spinner = spinnerFrames[spinnerIndex]
        let elapsed = Date().timeIntervalSince(startedAt)
        lock.unlock()

        let memory = Self.currentResidentMemoryGB()
        let barText: String
        if stage == .downloading, let fraction = downloadFraction {
            barText = Self.progressBar(fraction: fraction, width: 24)
        } else {
            barText = "[\(spinner)\(String(repeating: " ", count: 23))]"
        }

        let percentText: String
        if stage == .downloading, let fraction = downloadFraction {
            percentText = String(format: "%5.1f%%", max(0, min(100, fraction * 100)))
        } else {
            percentText = "  --.-%"
        }

        let line = String(
            format: "\r%@ %@ %@ | mem %.2f GB | %.1fs",
            barText,
            percentText,
            stage.rawValue,
            memory,
            elapsed
        )
        fputs(line, stdout)
        fflush(stdout)
    }

    private static func progressBar(fraction: Double, width: Int) -> String {
        let clamped = max(0.0, min(1.0, fraction))
        let filled = Int((clamped * Double(width)).rounded(.down))
        let bar = String(repeating: "#", count: filled) + String(repeating: "-", count: max(0, width - filled))
        return "[\(bar)]"
    }

    private static func currentResidentMemoryGB() -> Double {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size / MemoryLayout<natural_t>.size)
        let result = withUnsafeMutablePointer(to: &info) { ptr in
            ptr.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { rebound in
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), rebound, &count)
            }
        }
        guard result == KERN_SUCCESS else { return 0 }
        return Double(info.resident_size) / 1_073_741_824.0
    }
}

private func isPortAvailable(_ port: Int) -> Bool {
    let fd = socket(AF_INET, SOCK_STREAM, 0)
    guard fd >= 0 else { return false }
    defer { close(fd) }

    var value: Int32 = 1
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &value, socklen_t(MemoryLayout<Int32>.size))

    var addr = sockaddr_in()
    addr.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
    addr.sin_family = sa_family_t(AF_INET)
    addr.sin_port = in_port_t(UInt16(port).bigEndian)
    addr.sin_addr = in_addr(s_addr: inet_addr("127.0.0.1"))

    let result = withUnsafePointer(to: &addr) {
        $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
            bind(fd, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
        }
    }
    return result == 0
}

private func findEphemeralPort() throws -> Int {
    let fd = socket(AF_INET, SOCK_STREAM, 0)
    guard fd >= 0 else { throw ExitCode.failure }
    defer { close(fd) }

    var addr = sockaddr_in()
    addr.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
    addr.sin_family = sa_family_t(AF_INET)
    addr.sin_port = 0
    addr.sin_addr = in_addr(s_addr: inet_addr("127.0.0.1"))

    let bindResult = withUnsafePointer(to: &addr) {
        $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
            bind(fd, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
        }
    }
    guard bindResult == 0 else { throw ExitCode.failure }

    var sockAddr = sockaddr_in()
    var len = socklen_t(MemoryLayout<sockaddr_in>.size)
    let nameResult = withUnsafeMutablePointer(to: &sockAddr) {
        $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
            getsockname(fd, $0, &len)
        }
    }
    guard nameResult == 0 else { throw ExitCode.failure }
    return Int(UInt16(bigEndian: sockAddr.sin_port))
}

extension RootCommand {
    private func readFromStdin() throws -> String? {
        // Check if stdin is connected to a terminal (not piped)
        guard isatty(STDIN_FILENO) == 0 else {
            return nil
        }

        let stdin = FileHandle.standardInput
        let maxInputSize = 1024 * 1024 // 1MB limit
        var inputData = Data()

        // Read all available data from stdin
        while true {
            let chunk = stdin.availableData
            if chunk.isEmpty {
                break
            }

            inputData.append(chunk)

            // Prevent excessive memory usage
            if inputData.count > maxInputSize {
                print("Error: Input too large (max 1MB)")
                throw ExitCode.failure
            }
        }

        // If no data was read, stdin was likely /dev/null or similar, not a real pipe
        // Return nil to proceed to server mode
        guard !inputData.isEmpty else {
            return nil
        }

        // Convert to string and validate
        guard let content = String(data: inputData, encoding: .utf8) else {
            print("Error: Invalid UTF-8 input. Binary data not supported.")
            throw ExitCode.failure
        }

        let trimmedContent = content.trimmingCharacters(in: .whitespacesAndNewlines)

        // Check for empty input
        guard !trimmedContent.isEmpty else {
            print("Error: Empty input received from pipe")
            throw ExitCode.failure
        }

        return trimmedContent
    }
    
    private func runSinglePrompt(_ prompt: String, adapter: String?) throws {
        DebugLogger.log("Starting single prompt mode with prompt: '\(prompt)'")
        DebugLogger.log("Temperature: \(temperature?.description ?? "nil"), Randomness: \(randomness ?? "nil")")

        let group = DispatchGroup()
        var result: Result<String, Error>?

        group.enter()
        Task {
            do {
                if #available(macOS 26.0, *) {
                    DebugLogger.log("macOS 26+ detected, initializing FoundationModelService...")
                    let foundationService = try await FoundationModelService(instructions: instructions, adapter: adapter, temperature: temperature, randomness: randomness, permissiveGuardrails: permissiveGuardrails)
                    DebugLogger.log("FoundationModelService initialized successfully")
                    let message = Message(role: "user", content: prompt)
                    DebugLogger.log("Generating response...")
                    let response: String
                    if let guidedJson = self.guidedJson {
                        let schema = try parseGuidedJsonSchema(guidedJson)
                        response = try await foundationService.generateGuidedResponse(for: [message], jsonSchema: schema, temperature: temperature, randomness: randomness)
                    } else {
                        response = try await foundationService.generateResponse(for: [message], temperature: temperature, randomness: randomness)
                    }
                    DebugLogger.log("Response generated successfully")
                    result = .success(response)
                } else {
                    DebugLogger.log("macOS 26+ not available")
                    result = .failure(FoundationModelError.notAvailable)
                }
            } catch {
                DebugLogger.log("Error occurred: \(error)")
                result = .failure(error)
            }
            group.leave()
        }
        
        group.wait()
        
        switch result {
        case .success(let response):
            print(response)
        case .failure(let error):
            if let foundationError = error as? FoundationModelError {
                print("Error: \(foundationError.localizedDescription)")
            } else {
                print("Error: \(error.localizedDescription)")
            }
            throw ExitCode.failure
        case .none:
            print("Error: Unexpected error occurred")
            throw ExitCode.failure
        }
    }
}

// MARK: - Guided JSON schema parsing

func parseGuidedJsonSchema(_ jsonString: String) throws -> ResponseJsonSchema {
    guard let data = jsonString.data(using: .utf8),
          let jsonObj = try? JSONSerialization.jsonObject(with: data),
          jsonObj is [String: Any] else {
        throw ValidationError("Invalid JSON schema: must be a valid JSON object")
    }
    return ResponseJsonSchema(
        name: "guided",
        description: nil,
        schema: AnyCodable(jsonObj),
        strict: true
    )
}
