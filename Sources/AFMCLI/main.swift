import AFMKit
import AFMKitCore
import AFMKitDwarfStar
import AFMKitMLX
import AFMServer
import AFMTerminalUI
import ArgumentParser
import Foundation
import Darwin

// CLI-only conformance: AFMServer's TelegramReplyFormat stays free of ArgumentParser; the
// `@Option` flag parsing it needs is supplied here. It's a String-RawRepresentable enum, so
// ExpressibleByArgument's default rawValue-based init applies — an empty conformance suffices.
extension TelegramReplyFormat: ExpressibleByArgument {}

// Global references for signal handling. Accessed from the C signal handler
// (a nonisolated context), so these opt out of the main-actor isolation that
// Swift 6 infers for top-level globals. Signal-handler access is inherently
// single-threaded with respect to the run loop, so the unsafety is contained.
nonisolated(unsafe) private var globalServer: Server?
nonisolated(unsafe) private var shouldKeepRunning = true

private func runTerminalChat(_ configuration: TerminalChatConfiguration) throws {
    let outputIsolation = try TerminalOutputIsolation()
    defer { outputIsolation.restore() }
    let terminal = TerminalIO(
        inputFD: STDIN_FILENO,
        outputFD: outputIsolation.terminalOutputFD
    )
    let capabilities = TerminalCapabilities.detect(
        inputFD: STDIN_FILENO,
        outputFD: outputIsolation.terminalOutputFD
    )
    let group = DispatchGroup()
    let errorBox = SendableBox<Error?>(nil)
    group.enter()
    Task.detached {
        do {
            try await AFMTerminalChat(
                configuration: configuration,
                terminal: terminal,
                capabilities: capabilities
            ).run()
        } catch {
            errorBox.value = error
        }
        group.leave()
    }
    group.wait()
    if let error = errorBox.value { throw error }
}

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
    
    @Option(name: .shortAndLong, help: "Port to run server on (default: 9999, falls back to ephemeral if busy)")
    var port: Int?

    @Option(name: [.customShort("H"), .long], help: "Hostname to bind server to")
    var hostname: String = "127.0.0.1"

    @Flag(name: .shortAndLong, help: "Enable verbose logging")
    var verbose: Bool = false

    @Flag(name: [.customShort("V"), .long], help: "Enable very verbose logging (full requests/responses and all parameters)")
    var veryVerbose: Bool = false

    @Flag(name: .long, help: "Trace logging: raw model output, parsed/coerced client output, grammar constraints sent to model — all data at every boundary")
    var vv: Bool = false

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

    @Option(name: .long, help: "Telegram bot token for remote AFM access")
    var telegramBotToken: String?

    @Option(name: .long, help: "Enable Telegram bridge with a comma-separated allowlist of Telegram numeric user IDs")
    var telegramAllow: String?

    @Option(name: .long, help: "Telegram reply format: markdown, plain, or html (default: markdown)")
    var telegramFormat: TelegramReplyFormat = .markdown

    @Option(name: .long, help: "Require a specific prefix for Telegram messages, for example '/afm' (default: no prefix required)")
    var telegramRequirePrefix: String?

    @Flag(name: [.customShort("g"), .long], help: "Enable API gateway mode: discover and proxy to local LLM backends (Ollama, LM Studio, Jan, etc.)")
    var gateway: Bool = false

    @Option(name: .long, help: "Pre-warm the model on server startup for faster first response (y/n, default: y)")
    var prewarm: String = "y"

    @Option(name: .long, help: "Constrain output to match a JSON schema (vLLM-compatible). Applied to chat completions that omit their own response_format.")
    var guidedJson: String?

    func run() throws {
        // Validate temperature parameter
        if let temp = temperature {
            guard temp >= 0.0 && temp <= 1.0 else {
                throw ValidationError("Temperature must be between 0.0 and 1.0")
            }
        }

        // Validate randomness parameter
        if let rand = randomness {
#if compiler(>=6.4)
            do {
                _ = try RandomnessConfig.parse(rand)
            } catch let error as FoundationModelError {
                throw ValidationError(error.localizedDescription)
            } catch {
                throw ValidationError("Invalid randomness parameter format")
            }
#else
            throw ValidationError(
                "--randomness requires the Swift 6.4 toolchain or newer")
#endif
        }

        let defaultGuidedJsonSchema: ResponseFormat?
        if let guidedJson {
            let schema = try parseGuidedJsonSchema(guidedJson)
            defaultGuidedJsonSchema = ResponseFormat(type: "json_schema", jsonSchema: schema)
        } else {
            defaultGuidedJsonSchema = nil
        }

        // Port selection: use requested port, default 9999, or fall back to ephemeral
        let chosenPort: Int
        if let requested = port {
            chosenPort = requested
        } else if isPortAvailable(9999) {
            chosenPort = 9999
        } else {
            chosenPort = try findEphemeralPort()
            print("Port 9999 is busy, using ephemeral port \(chosenPort)")
        }

        // Parse prewarm flag
        let prewarmEnabled = prewarm.lowercased() != "n" && prewarm.lowercased() != "no" && prewarm != "0"
        let telegramConfiguration = try makeTelegramConfiguration(
            rawBotToken: telegramBotToken,
            rawAllowlist: telegramAllow,
            hostname: hostname,
            port: chosenPort,
            modelID: "foundation",
            instructions: instructions,
            verbose: verbose || veryVerbose || vv,
            replyFormat: telegramFormat,
            requirePrefix: telegramRequirePrefix
        )

        if gateway && telegramConfiguration != nil {
            throw ValidationError("--telegram-bot-token/--telegram-allow are not supported with --gateway")
        }

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
                let server = try await Server(port: chosenPort, hostname: hostname, verbose: verbose, veryVerbose: veryVerbose || vv, trace: vv, streamingEnabled: !noStreaming, instructions: instructions, adapter: adapter, temperature: temperature, randomness: randomness, permissiveGuardrails: permissiveGuardrails, stop: stop, webuiEnabled: webui, gatewayEnabled: gateway, prewarmEnabled: prewarmEnabled, telegramConfiguration: telegramConfiguration, defaultGuidedJsonSchema: defaultGuidedJsonSchema)
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

private enum MLXRuntimeBackend: String, CaseIterable {
    case auto
    case mlx
    case dwarfstar
}

struct MlxCommand: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "mlx",
        abstract: "Run local MLX LLM/VLM models via AFM",
        discussion: """
        ---
        name: afm-mlx
        description: Run MLX-format LLM/VLM models from Hugging Face on Apple Silicon with OpenAI-compatible API. Supports streaming, tool calling, logprobs, thinking/reasoning extraction, prompt caching, quantized KV cache, and all OpenAI sampling parameters.
        tags: [mlx, huggingface, llm, vlm, inference, streaming, tool-calling, logprobs, thinking, sampling, quantization, kv-cache, prompt-caching]
        api_endpoints: [/v1/chat/completions, /v1/models]
        env_vars:
          MACAFM_MLX_MODEL_CACHE: Override model download/cache directory (avoids re-downloading)
          MACAFM_MLX_METALLIB: Override Metal library path
          AFM_DEBUG: Enable debug logging (KVCache stats, tool call detection, timing)
          AFM_PERF: Enable per-token performance breakdown (model, eval, sync, overhead)
        cli_flags:
          -m, --model: Model id (org/model or just model name, defaults to mlx-community org)
          -s, --single-prompt: Run a single prompt and exit (no server)
          -i, --instructions: System prompt / custom instructions (default: "You are a helpful assistant")
          -p, --port: Server port (default: 9999, auto-fallback to ephemeral if busy)
          -H, --hostname: Bind address (default: 127.0.0.1)
          -v, --verbose: Enable verbose logging
          -V, --very-verbose: Log full requests/responses and all parameters
          -w, --webui: Enable WebUI and open in browser
          --tui: Run the native interactive terminal chat UI
          --no-alt-screen: Disable alternate-screen overlays and keep the TUI inline
          --telegram-bot-token: Telegram bot token for remote AFM access
          --telegram-allow: Comma-separated allowlist of Telegram numeric user IDs
          --telegram-format: Telegram reply format: markdown, plain, or html
          --telegram-require-prefix: Require a specific prefix for Telegram messages, for example '/afm'
          -t, --temperature: Sampling temperature (0.0-2.0)
          --top-p: Nucleus sampling threshold (0.0-1.0)
          --top-k: Keep only k most likely tokens (0 = disabled)
          --min-p: Filter tokens below min_p * max_prob (0.0 = disabled)
          --presence-penalty: Additive penalty for tokens already generated
          --repetition-penalty: Penalize repeated tokens
          --max-tokens: Maximum tokens per response (default: 8192)
          --seed: Random seed for reproducible output
          --max-logprobs: Max top logprobs per token (default: 20)
          --stop: Stop sequences, comma-separated (e.g. "###,END")
          --guided-json: Constrain output to JSON schema (vLLM-compatible; auto-disables thinking on reasoning models)
          --no-streaming: Disable streaming (streaming enabled by default)
          --raw: Output raw model text without extracting <think> tags
          --vlm: Force load as vision model (VLM) instead of text-only LLM
          --media: Image/video paths for VLM single-prompt mode (implies --vlm)
          --kv-bits: Quantize KV cache (4 or 8 bits) to reduce memory
          --prefill-step-size: Prompt tokens per GPU pass (default: 1024)
          --mlx-runtime: Runtime backend: auto, mlx, or dwarfstar (default: auto)
          --gguf-file: Exact GGUF path inside a Hugging Face repository; otherwise AFM selects the largest model artifact that fits memory
          --enable-prefix-caching / --no-enable-prefix-caching: KV cache reuse across requests
          --mtp: Enable serial MTP self-speculative decoding for compatible Qwen models
          --mtp-depth: MTP draft depth compatibility setting
          --mtp-model: Override the automatic MTP head with a Hugging Face repo, local directory, or .safetensors file
          --dspark-support: DwarfStar DSpark support GGUF for speculative decoding
          --dspark-draft-tokens: Maximum DSpark speculative tokens per cycle (default: 5)
          --dspark-confidence: DSpark confidence-pruning threshold (default: 0.7)
          --dspark-strict: Load DSpark support but use target-only decoding
          --eagle3: EAGLE3 drafter directory for compatible dense Gemma4 models
          --tool-call-parser: Override tool call format (none, afm_adaptive_xml, deepseek_dsml, hermes, llama3_json, gemma, mistral, qwen3_xml). Omit for default native mode and MLX Python-style parity; use "none" for raw output; use "afm_adaptive_xml" for opt-in repair mode.
          --fix-tool-args: Opt-in repair-mode helper that post-processes tool call arg names to match original tool schema
          --enable-grammar-constraints: Enable grammar-constrained decoding engine. When active, API requests with strict: true on tools or response_format.json_schema use xgrammar for token-level enforcement. Without this flag, strict: true is silently downgraded to best-effort.
          --no-think: Disable thinking/reasoning when supported; for Muse, requests the lowest reasoning strength
          --reasoning-effort: Default reasoning effort for compatible models: low, high, or max
          --concurrent: Maximum concurrent requests; values greater than one enable batch mode
          --default-chat-template-kwargs: JSON object merged into chat template context
          --cache-profile-path: Write cache timing profile records as JSONL
          --gpu-capture <path>: Capture Metal GPU trace to .gputrace file for Xcode analysis (auto-limits to 5 tokens)
          --gpu-trace <seconds>: Record Metal System Trace via xctrace for N seconds (lightweight per-kernel timing)
          --gpu-profile: Print per-request GPU profiling stats (device info, memory, bandwidth estimates)
          --gpu-profile-bw: Also sample DRAM bandwidth with mactop
          --openclaw-config: Print OpenClaw provider config JSON and exit
          --eval: Run the bundled comprehensive local evaluation and open its HTML report
          --bench: Alias for --eval
          --eval-suite: Select a bundled/custom suite (repeatable; implies --eval)
          --eval-list: List bundled and ~/.afm/evals custom suites
          --eval-init: Scaffold a custom JSON suite under ~/.afm/evals
          --eval-validate: Validate a suite name or JSON file without loading a model
          --no-open: Do not open the evaluation report in a browser
          --help-json: Print machine-readable JSON capability card for AI agents and exit
        sampling_parameters: [temperature, top_p, top_k, min_p, presence_penalty, repetition_penalty, seed, max_tokens, logprobs, top_logprobs]
        features: [streaming-sse, tool-calling, think-reasoning-extraction, stop-sequences, json-mode, json-schema, prompt-caching, vlm-image-input, kv-cache-quantization, grammar-constrained-decoding, huggingface-gguf-resolution, openclaw-integration]
        api_compatibility: OpenAI Chat Completions API (https://platform.openai.com/docs/api-reference/chat/create)
        extra_request_fields:
          top_k: int (not in OpenAI spec)
          min_p: float (not in OpenAI spec)
          repetition_penalty: float (also accepts repeat_penalty, not in OpenAI spec)
          reasoning_effort: Reasoning effort for compatible models: low, high, or max
          chat_template_kwargs: object e.g. {"enable_thinking": false} (AFM-specific)
        extra_response_fields:
          choices[].message.reasoning_content: Extracted <think> reasoning (AFM-specific)
          usage.prompt_tokens_details.cached_tokens: Prefix cache hit count (AFM-specific)
        notes:
          - frequency_penalty is parsed but silently ignored
          - developer role is mapped to system
          - max_completion_tokens is accepted alongside max_tokens
        supported_model_types: [llama, qwen2, qwen3, qwen3_moe, qwen3_5, qwen3_5_moe, gemma, gemma2, phi3, starcoder2, openelm, cohere2, deepseek_v3, deepseek_v4, glm4, glm4_moe, lfm2, lfm2_moe, nemotron_h, minimax_m2, kimi_k2]
        tool_calling:
          auto_detection: Tool call format is auto-detected from model_type in config.json. Qwen XML models use the narrow qwen3_xml parser by default. Omit --tool-call-parser for default native mode and benchmark parity checks.
          parser_overrides:
            none: Raw mode. Disables AFM/server-side tool-call extraction and fallback repair. Generated tool text is returned as ordinary content.
            afm_adaptive_xml: Repair mode. Opt-in adaptive XML parser with JSON-in-XML fallback, type coercion, and EBNF grammar-constrained decoding (with --enable-grammar-constraints). Use only when you want AFM repair behavior beyond default native parsing.
            hermes: JSON format with Hermes chat template (Llama, Qwen, most models)
            llama3_json: JSON format with Llama-3 chat template
            mistral: JSON format with Mistral chat template
            deepseek_dsml: Native DeepSeek V4 DSML tool-call format
            qwen3_xml: XML function format with Qwen3-Coder chat template
            gemma: Gemma function call format (uses model's built-in template)
          benchmark_guidance: Use default native mode for MLX Python parity and VulcanBench comparisons. Treat afm_adaptive_xml / --fix-tool-args results as AFM repair-layer results, not plain-vanilla parity.
          auto_detected_formats:
            json: Default for Llama, Qwen, most models (<tool_call>...</tool_call> tags)
            xml_function: Qwen3 Coder, Qwen3.5 MoE (<function=name><parameter=key>value</parameter></function>)
            lfm2: LFM2 models (<|tool_call_start|>...<|tool_call_end|>)
            glm4: GLM4 models (<arg_key>k</arg_key><arg_value>v</arg_value>)
            gemma: Gemma models (call:name{key:value})
            kimi_k2: Kimi K2 (functions.name:0<|tool_call_argument_begin|>{...})
            minimax_m2: MiniMax M2 (<invoke name="f"><parameter name="k">v</parameter></invoke>)
          fix_tool_args: --fix-tool-args post-processes argument names to match original tool schema (fixes model renaming e.g. path to filePath)
        skill:
          what_it_does: Runs open-source LLM/VLM models locally on Apple Silicon via MLX, exposing an OpenAI-compatible API. No cloud, no API keys, no Python runtime.
          repository: https://github.com/scouzi1966/maclocal-api
          when_to_use:
            - You need a local OpenAI-compatible LLM endpoint for coding agents, chat apps, or tool-calling workflows
            - You want to run Hugging Face MLX models (Qwen, Llama, Gemma, DeepSeek, Phi, etc.) on Mac
            - You need streaming, tool calling, logprobs, or thinking/reasoning extraction locally
            - You want vision/multimodal inference with local VLM models
          when_not_to_use:
            - You need cloud-scale throughput or multi-GPU inference (use vLLM, TGI, or cloud APIs)
            - You need a GGUF architecture other than the supported DeepSeek V4 DwarfStar format
            - You are not on Apple Silicon (MLX is Apple-only)
          integration_pattern: Start server with `afm mlx -m <model>`, then point any OpenAI SDK client at http://127.0.0.1:9999/v1. Drop-in replacement for OpenAI API.
          limitations:
            - Single-sequence inference only (one request at a time, queued)
            - Safetensors models use MLX; compatible DeepSeek V4 GGUF files auto-select DwarfStar
            - JSON mode uses prompt injection, not grammar-constrained decoding
            - Apple Silicon Mac required (M1/M2/M3/M4)
          typical_workflow:
            - 1. Download model — afm mlx -m mlx-community/Qwen3.5-35B-A3B-4bit (auto-downloads on first use)
            - 2. Start server — afm mlx -m mlx-community/Qwen3.5-35B-A3B-4bit --port 9999
            - 3. Send requests — curl http://127.0.0.1:9999/v1/chat/completions -d '{...}'
            - 4. Or use WebUI — afm mlx -m <model> -w (opens browser chat interface)
        triggers:
          - run MLX model
          - local Hugging Face model inference
          - MLX tool calling
          - MLX streaming server
          - quantized model inference
          - run local LLM with tool calling
          - vision model inference with images
        examples:
          - afm mlx -m Qwen/Qwen3-Coder-Next-4bit --port 9999
          - afm mlx -m mlx-community/Llama-3.1-8B-Instruct-4bit --top-k 40 --min-p 0.05
          - afm mlx -m org/model -s "Explain quicksort" --temperature 0.7
          - afm mlx -m org/model --vlm --media photo.jpg -s "Describe this image"
          - afm mlx -m org/model --no-think --tool-call-parser qwen3_xml
          - afm mlx -m org/model --kv-bits 4 --enable-prefix-caching
          - 'curl http://127.0.0.1:9999/v1/chat/completions -d ''{"model":"m","messages":[{"role":"user","content":"Hi"}],"stream":true}'''
          - MACAFM_MLX_MODEL_CACHE=/path/to/cache afm mlx -m org/model
        ---

        Uses MLX Swift libraries + HuggingFace Hub.
        Model cache root can be overridden with MACAFM_MLX_MODEL_CACHE.
        Metallib path can be overridden with MACAFM_MLX_METALLIB.
        """
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

    @Flag(name: .long, help: "Trace logging: raw model output, parsed/coerced client output, grammar constraints sent to model — all data at every boundary")
    var vv: Bool = false

    @Flag(name: .long, help: "Disable streaming responses (streaming is enabled by default)")
    var noStreaming: Bool = false

    @Flag(name: .long, help: "Output raw model content without extracting <think> tags into reasoning_content")
    var raw: Bool = false

    @Flag(name: .long, help: "Emit single-prompt result as OpenAI-compatible JSON")
    var json: Bool = false

    @Option(name: [.short, .long], help: "Temperature for response generation (0.0-2.0)")
    var temperature: Double?

    @Flag(name: [.customShort("w"), .long], help: "Enable webui and open in default browser")
    var webui: Bool = false

    @Flag(name: .long, help: "Run the advanced native terminal chat UI")
    var tui: Bool = false

    @Flag(name: .long, help: "Disable alternate-screen overlays and keep the TUI inline")
    var noAltScreen: Bool = false

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
    var maxTokens: Int = MLXModelService.defaultMaximumResponseTokens
    @Option(name: .long, help: "Random seed for reproducible sampling (nil = non-deterministic)")
    var seed: Int?
    @Option(name: .long, help: "Maximum number of top log probabilities returned per token (default: 20)")
    var maxLogprobs: Int?
    @Option(name: .long, help: "Repetition penalty (compatibility)")
    var repetitionPenalty: Double?
    @Option(name: .long, help: "KV cache size (compatibility)")
    var maxKVSize: Int?
    @Option(name: .long, help: "Quantize KV cache to this many bits (4 or 8) to reduce memory usage")
    var kvBits: Int?
    @Option(name: .long, help: "Prefill step size — number of prompt tokens processed per GPU pass (default: 2048)")
    var prefillStepSize: Int?
    @Option(name: .long, help: .hidden)
    var mlxKernels: String = "native"
    @Option(name: .long, help: "Runtime backend: auto, mlx, or dwarfstar. auto selects vanilla DwarfStar for compatible DeepSeek V4 GGUF metadata; directory checkpoints use MLX.")
    var mlxRuntime: String = "auto"
    @Option(name: .customLong("gguf-file"), help: "Exact GGUF path inside a Hugging Face repository. By default AFM selects the largest model GGUF that fits memory and excludes speculative support files.")
    var ggufFile: String?
    @Option(name: .long, help: "Pre-warm MLX kernels on startup for faster first response/TTFT (y/n, default: y)")
    var prewarm: String = "y"
    @Flag(name: .long, help: "Trust remote code (compatibility)")
    var trustRemoteCode: Bool = false
    @Option(name: .long, help: "Chat template (compatibility)")
    var chatTemplate: String?
    @Option(name: .long, help: "Dtype (compatibility)")
    var dtype: String?
    @Flag(name: .long, help: "Load as vision model (VLM). Default: text-only LLM for better performance")
    var vlm: Bool = false

    @Option(name: .long, parsing: .upToNextOption, help: "Media file paths (images/videos) for single-prompt VLM mode. Implies --vlm.")
    var media: [String] = []

    @Option(name: .long, help: "Stop sequences - comma-separated strings where generation should stop (e.g., '###,END')")
    var stop: String?

    @Option(name: .long, help: "Constrain output to match a JSON schema (vLLM-compatible). Auto-disables thinking on reasoning models for deterministic output.")
    var guidedJson: String?

    @Option(name: .long, help: "Telegram bot token for remote AFM access")
    var telegramBotToken: String?

    @Option(name: .long, help: "Enable Telegram bridge with a comma-separated allowlist of Telegram numeric user IDs")
    var telegramAllow: String?

    @Option(name: .long, help: "Telegram reply format: markdown, plain, or html (default: markdown)")
    var telegramFormat: TelegramReplyFormat?

    @Option(name: .long, help: "Require a specific prefix for Telegram messages, for example '/afm' (default: no prefix required)")
    var telegramRequirePrefix: String?

    @Option(name: .long, help: "Tool call parser override: none, afm_adaptive_xml, deepseek_dsml, hermes, llama3_json, gemma, mistral, qwen3_xml. Omit for default native mode and MLX Python-style parity; Qwen XML models default to qwen3_xml and DeepSeek V4 models use deepseek_dsml. Use none for raw output with no AFM extraction. Use afm_adaptive_xml for opt-in repair behavior with JSON-in-XML fallback, type coercion, and optional xgrammar EBNF constrained decoding.")
    var toolCallParser: String?

    @Option(name: .long, help: "OpenAI-compatible tools array JSON for single-prompt mode")
    var toolsJson: String?

    @Flag(name: .long, help: "Opt-in repair helper: post-process tool call argument names to match the original tool schema (fixes model renaming e.g. path→filePath). Leave off for plain native/parity mode.")
    var fixToolArgs: Bool = false

    @Option(name: .customLong("kv-eviction"), help: "KV cache eviction policy: streaming (StreamingLLM) or none (default)")
    var kvEviction: String?

    @Option(name: .long, help: "Default chat template kwargs as JSON (e.g. '{\"enable_thinking\": false}')")
    var defaultChatTemplateKwargs: String?

    @Flag(name: .long, help: "Enable radix tree prefix caching for KV cache reuse across requests")
    var enablePrefixCaching: Bool = false

    @Flag(name: .long, help: "Enable MTP self-speculative decoding. Qwen 3.8 automatically downloads and uses the matching quantized MTP head; concurrent and batch requests safely use autoregressive decoding.")
    var mtp: Bool = false

    @Option(name: .long, help: "MTP draft depth (accepted for compatibility; the loop currently uses the fixed depth-2-bonus structure from mlx-lm PR #990 — ~+50% decode vs AR on M4 Pro — so this value is not used).")
    var mtpDepth: Int = 1

    @Option(name: .customLong("mtp-model"), help: "Override the automatically selected MTP head with a Hugging Face repo, local directory, or .safetensors file.")
    var mtpModel: String?

    @Option(name: .customLong("dspark-support"), help: "DwarfStar DSpark support GGUF. Supplying it enables greedy speculative decoding.")
    var dsparkSupportPath: String?

    @Option(name: .customLong("dspark-draft-tokens"), help: "Maximum DSpark speculative tokens per cycle (1...16, default: 5).")
    var dsparkDraftTokens: Int = 5

    @Option(name: .customLong("dspark-confidence"), help: "DSpark confidence-pruning threshold (0...1, default: 0.7).")
    var dsparkConfidenceThreshold: Double = 0.7

    @Flag(name: .customLong("dspark-strict"), help: "Load DSpark support but keep target-only decoding for correctness comparisons.")
    var dsparkStrict: Bool = false

    @Option(name: .long, help: "Enable EAGLE3 speculative decoding for a dense Gemma4 verifier. Pass the drafter directory (config.json + safetensors). Faster decode, quality-preserving (near-greedy output). No-op if the verifier is not a dense Gemma4 text model.")
    var eagle3: String?

    @Option(name: .long, help: "Write cache timing profile records as JSONL to this file")
    var cacheProfilePath: String?

    @Flag(name: .long, help: "Enable grammar-constrained decoding engine. When active, API requests with strict: true on tools or response_format.json_schema use xgrammar for token-level enforcement. Without this flag, strict: true is silently downgraded to best-effort.")
    var enableGrammarConstraints: Bool = false

    @Flag(name: [.customLong("no-think"), .customLong("no-thinking")], help: "Disable thinking/reasoning when supported. Overrides --reasoning-effort and chat-template kwargs; for Muse, requests the lowest reasoning strength.")
    var noThink: Bool = false

    @Option(
        name: .customLong("reasoning-effort"),
        help: "Reasoning effort for compatible models: low, high, or max."
    )
    var reasoningEffort: String?

    @Option(name: .long, help: "Max concurrent requests (enables batch mode; 0 or 1 reverts to serial)")
    var concurrent: Int?

    @Option(name: .long, help: "Capture a Metal GPU trace to the given path (e.g. /tmp/afm-trace.gputrace). Opens in Xcode for per-kernel analysis. Auto-limits to 5 tokens to keep trace small.")
    var gpuCapture: String?

    @Option(name: .long, help: "Record a Metal System Trace for N seconds using Instruments xctrace (e.g. --gpu-trace 10). Lightweight per-kernel GPU timing without massive trace files. Output: /tmp/afm-metal.trace")
    var gpuTrace: Int?

    @Flag(name: .long, help: "Print per-request GPU profiling stats: device info, memory snapshots, bandwidth estimates, peak memory")
    var gpuProfile: Bool = false

    @Flag(name: .long, help: "Also sample DRAM bandwidth via mactop after inference (adds ~5s). Implies --gpu-profile. Requires: brew install mactop")
    var gpuProfileBw: Bool = false

    @Flag(name: .long, help: "Print OpenClaw provider config JSON and exit")
    var openclawConfig: Bool = false

    @Flag(name: .long, help: "Run the bundled comprehensive local model evaluation and open its HTML report")
    var eval: Bool = false

    @Flag(name: .long, help: "Alias for --eval")
    var bench: Bool = false

    @Option(name: .customLong("eval-suite"), help: "Bundled/custom evaluation suite name; repeat to run multiple suites (implies --eval)")
    var evalSuites: [String] = []

    @Flag(name: .customLong("eval-list"), help: "List bundled and ~/.afm/evals custom suites, then exit")
    var evalList: Bool = false

    @Option(name: .customLong("eval-init"), help: "Create a safe example suite at ~/.afm/evals/<name>.json, then exit")
    var evalInit: String?

    @Option(name: .customLong("eval-validate"), help: "Validate a suite name or JSON file, then exit")
    var evalValidate: String?

    @Flag(name: .customLong("no-open"), help: "Do not open the generated evaluation HTML report")
    var noOpen: Bool = false

    @Flag(name: .long, help: "Print machine-readable JSON capability card for AI agents and exit")
    var helpJson: Bool = false

    func run() throws {
        if helpJson {
            printHelpJson(command: "afm mlx")
            return
        }

        let evaluationAction = try AFMEvaluationCLIPlan.resolve(
            evaluate: eval,
            bench: bench,
            suites: evalSuites,
            list: evalList,
            scaffold: evalInit,
            validate: evalValidate,
            noOpen: noOpen)
        if try handleEvaluationManagement(evaluationAction) {
            return
        }

        if gateway {
            print("Error: -g/--gateway is not supported in 'afm mlx' mode.")
            throw ExitCode.failure
        }

        let hasTelegramOptions = TUIInvocationPolicy.hasTelegramOptions(
            botToken: telegramBotToken,
            allowlist: telegramAllow,
            replyFormat: telegramFormat?.rawValue,
            requirePrefix: telegramRequirePrefix
        )
        do {
            try TUIInvocationPolicy.validate(
                tui: tui,
                webUI: webui,
                singlePrompt: singlePrompt != nil,
                telegramOptions: hasTelegramOptions,
                inputIsTTY: isatty(STDIN_FILENO) != 0,
                outputIsTTY: isatty(STDOUT_FILENO) != 0
            )
        } catch {
            throw ValidationError(error.localizedDescription)
        }
        if tui && (raw || json || openclawConfig) {
            throw ValidationError("--tui cannot be combined with --raw, --json, or --openclaw-config")
        }
        if let toolCallParser {
            let normalizedParser = toolCallParser.trimmingCharacters(in: .whitespacesAndNewlines)
            let supportedParsers: Set<String> = [
                "none", "afm_adaptive_xml", "deepseek_dsml", "hermes", "llama3_json",
                "gemma", "mistral", "qwen3_xml",
            ]
            guard supportedParsers.contains(normalizedParser) else {
                throw ValidationError(
                    "--tool-call-parser must be one of: \(supportedParsers.sorted().joined(separator: ", "))"
                )
            }
        }

        // GPU capture: set MTL_CAPTURE_ENABLED before Metal device is created
        if let capturePath = gpuCapture {
            setenv("MTL_CAPTURE_ENABLED", "1", 1)
            // Remove existing .gputrace at path (Metal requires it not to exist)
            let fm = FileManager.default
            if fm.fileExists(atPath: capturePath) {
                try? fm.removeItem(atPath: capturePath)
            }
            print("GPU capture enabled → \(capturePath)")
            print("  Auto-limiting to 5 tokens (full capture records every Metal dispatch)")
            print("  Open in Xcode after completion: open \(capturePath)")
        }

        // GPU trace: validate duration
        if let traceSec = gpuTrace, traceSec < 1 {
            print("Error: --gpu-trace duration must be >= 1 second")
            throw ExitCode.failure
        }

        if hasTelegramOptions && (singlePrompt != nil || isatty(STDIN_FILENO) == 0) {
            print("Error: --telegram requires server mode and cannot be used with -s or piped single-prompt input")
            throw ExitCode.failure
        }

        let shellCWD = URL(
            fileURLWithPath: ProcessInfo.processInfo.environment["PWD"]
                ?? FileManager.default.currentDirectoryPath,
            isDirectory: true
        )
        let resolvedMediaURLs: [URL]
        do {
            resolvedMediaURLs = try TUIMediaAttachmentPolicy.resolveAndValidate(media, cwd: shellCWD)
        } catch {
            throw ValidationError(error.localizedDescription)
        }
        let resolvedMedia = resolvedMediaURLs.map(\.path)

        emitCompatibilityWarnings()

        let kernelEngine = AFMMLXKernelEngine(configuredValue: mlxKernels)
        if kernelEngine.rawValue != mlxKernels.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
            let valid = AFMMLXKernelEngine.allCases.map(\.rawValue).joined(separator: ", ")
            fputs("Error: --mlx-kernels must be one of: \(valid)\n", stderr)
            throw ExitCode.failure
        }
        // Publish the engine before any MLX object can initialize. The runtime
        // also applies this typed value to the AFMKit MLX runtime; the environment is
        // the stable boundary consumed by the patched mlx-swift-lm package.
        setenv("AFM_MLX_KERNELS", kernelEngine.rawValue, 1)
        if verbose {
            print("Selected MLX kernel engine: \(kernelEngine.rawValue)")
        }

        let resolver = MLXCacheResolver()
        let modelStore = AFMMLXModelStore(resolver: resolver)

        // Parse template controls first, then apply typed CLI controls. The explicit
        // no-thinking flag is deliberately last so it cannot be undone by JSON.
        var parsedKwargs: [String: Any] = [:]
        if let jsonStr = defaultChatTemplateKwargs {
            guard let data = jsonStr.data(using: .utf8) else {
                fputs("Error: --default-chat-template-kwargs must be valid UTF-8\n", stderr)
                throw ExitCode.failure
            }
            do {
                let jsonObject = try JSONSerialization.jsonObject(with: data)
                guard let dict = jsonObject as? [String: Any] else {
                    fputs("Error: --default-chat-template-kwargs must be a JSON object (e.g. '{\"enable_thinking\": false}')\n", stderr)
                    throw ExitCode.failure
                }
                for (key, value) in dict {
                    parsedKwargs[key] = value
                }
            } catch let error where !(error is ExitCode) {
                fputs("Error: Failed to parse --default-chat-template-kwargs as JSON: \(error)\n", stderr)
                throw ExitCode.failure
            }
        }
        let normalizedReasoningEffort = reasoningEffort?
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        if let normalizedReasoningEffort {
            guard ["low", "high", "max"].contains(normalizedReasoningEffort) else {
                fputs("Error: --reasoning-effort must be low, high, or max\n", stderr)
                throw ExitCode.failure
            }
            parsedKwargs["reasoning_effort"] = normalizedReasoningEffort
            parsedKwargs["enable_thinking"] = true
        }
        if noThink {
            let configuredEffort = normalizedReasoningEffort != nil
                || parsedKwargs["reasoning_effort"] != nil
                || (parsedKwargs["enable_thinking"] as? Bool) == true
            if configuredEffort {
                fputs("Note: --no-thinking overrides the configured reasoning effort.\n", stderr)
            }
            parsedKwargs["enable_thinking"] = false
            parsedKwargs.removeValue(forKey: "reasoning_effort")
        }

        var defaultGuidedJsonSchema: ResponseFormat?
        if let guidedJson {
            let schema = try parseGuidedJsonSchema(guidedJson)
            defaultGuidedJsonSchema = ResponseFormat(type: "json_schema", jsonSchema: schema)
        }

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
            let discovered = modelStore.discoverLocalModels()
            if !discovered.isEmpty {
                print("No model provided. Available local models:")
                for model in discovered {
                    print("  - \(model.id)")
                }
            } else {
                print("No model provided and no complete local model was found.")
                print("Use: afm mlx -m <org/model> ...")
            }
            throw ExitCode.failure
        }

        let resolvedModel = try resolveRemoteDwarfStarModelIfNeeded(rawModel)
        let runtimeBackend = try resolveRuntimeBackend(model: resolvedModel)
        if case .run(let suites, let openReport) = evaluationAction {
            guard runtimeBackend == .mlx else {
                throw ValidationError("--eval currently supports MLX directory checkpoints; DwarfStar GGUF evaluation is not yet available")
            }
            guard singlePrompt == nil, media.isEmpty, !webui, !openclawConfig,
                  telegramBotToken == nil, telegramAllow == nil,
                  toolsJson == nil, !raw, !json, concurrent == nil,
                  !tui, !noAltScreen, maxKVSize == nil else {
                throw ValidationError(
                    "--eval cannot be combined with -s, --media, --webui, Telegram, " +
                    "--openclaw-config, --tools-json, --raw, --json, --concurrent, " +
                    "--tui, --no-alt-screen, or --max-kv-size")
            }
            try ensureMLXMetalLibraryAvailable(verbose: verbose)
            let evaluationChatTemplateKwargs = parsedKwargs.isEmpty
                ? nil
                : try parsedKwargs.mapValues { try Self.afmJSONValue(from: $0) }
            try runEvaluation(
                modelID: resolvedModel,
                suites: suites,
                openReport: openReport,
                chatTemplateKwargs: evaluationChatTemplateKwargs,
                defaultResponseFormat: defaultGuidedJsonSchema)
            return
        }
        if runtimeBackend != .dwarfstar, dsparkSupportPath != nil {
            throw ValidationError("--dspark-support requires --mlx-runtime dwarfstar or a DwarfStar executor checkpoint")
        }
        if runtimeBackend == .dwarfstar {
            if tui {
                throw ValidationError("--tui currently supports MLX directory checkpoints; DwarfStar GGUF terminal chat is not yet available")
            }
            try runDwarfStar(
                checkpointPath: localModelPath(resolvedModel),
                advertisedModelID: AFMDwarfStarModelIdentity.advertisedModelID(
                    requestedModel: rawModel,
                    checkpointPath: localModelPath(resolvedModel)
                ),
                modelStore: modelStore,
                chatTemplateKwargs: parsedKwargs,
                forceDisableThinking: noThink,
                defaultGuidedJsonSchema: defaultGuidedJsonSchema)
            return
        }

        let runtimeConfiguration = AFMMLXRuntimeConfiguration(
            kvBits: kvBits,
            enablePrefixCaching: enablePrefixCaching,
            kernelEngine: kernelEngine,
            mtpEnabled: mtp,
            mtpDepth: mtpDepth,
            mtpModelID: mtpModel,
            eagle3DrafterPath: eagle3,
            maxConcurrent: concurrent ?? 0,
            toolCallParser: toolCallParser,
            enableGrammarConstraints: enableGrammarConstraints,
            prefillStepSize: prefillStepSize,
            kvEvictionPolicy: kvEviction ?? "none",
            fixToolArguments: fixToolArgs,
            forceVLM: vlm || !media.isEmpty,
            cacheProfilePath: cacheProfilePath,
            trace: vv,
            gpuCapturePath: gpuCapture,
            gpuTraceDuration: gpuTrace,
            gpuProfile: gpuProfile || gpuProfileBw,
            gpuProfileBandwidth: gpuProfileBw,
            defaultChatTemplateKwargs: parsedKwargs.isEmpty
                ? nil
                : try parsedKwargs.mapValues { try Self.afmJSONValue(from: $0) },
            forceDisableThinking: noThink
        )
        let mlxModel = AFMMLXModel(
            modelID: AFMModelID(rawValue: resolvedModel),
            runtimeConfiguration: runtimeConfiguration,
            resolver: resolver
        )
        let selectedModel = mlxModel.normalizeModel(resolvedModel)

        if tui {
            var metadata: [String: AFMJSONValue] = [:]
            if !parsedKwargs.isEmpty {
                metadata["chatTemplateKwargs"] = .object(
                    try parsedKwargs.mapValues { try Self.afmJSONValue(from: $0) }
                )
            }
            let tuiLogprobs = TUILogprobConfiguration(maximum: maxLogprobs)
            let engineConfig = EngineConfig(
                instructions: instructions,
                kvBits: kvBits,
                enablePrefixCaching: enablePrefixCaching,
                mlxKernels: kernelEngine.rawValue,
                mtpEnabled: mtp,
                mtpDepth: mtpDepth,
                mtpModelID: mtpModel,
                eagle3DrafterPath: eagle3,
                enableGrammarConstraints: enableGrammarConstraints,
                toolCallParser: toolCallParser,
                maxConcurrent: concurrent ?? 0,
                prefillStepSize: prefillStepSize,
                kvEvictionPolicy: kvEviction ?? "none",
                fixToolArguments: fixToolArgs,
                forceVLM: vlm || !media.isEmpty,
                cacheProfilePath: cacheProfilePath,
                trace: vv,
                gpuCapturePath: gpuCapture,
                gpuTraceDuration: gpuTrace,
                gpuProfile: gpuProfile || gpuProfileBw,
                gpuProfileBandwidth: gpuProfileBw
            )
            let generation = GenerationConfig(
                temperature: temperature,
                maxTokens: maxTokens,
                topP: topP,
                topK: topK,
                minP: minP,
                repetitionPenalty: repetitionPenalty,
                presencePenalty: presencePenalty,
                seed: seed,
                logprobs: tuiLogprobs.enabled,
                topLogprobs: tuiLogprobs.maximum,
                stop: stop?.split(separator: ",").map(String.init),
                tools: try Self.parseToolsJSON(toolsJson),
                responseFormat: defaultGuidedJsonSchema,
                metadata: metadata
            )
            try ensureMLXMetalLibraryAvailable(verbose: verbose)
            try runTerminalChat(TerminalChatConfiguration(
                backend: .mlx(modelID: selectedModel),
                backendName: "MLX",
                modelName: selectedModel,
                engine: engineConfig,
                generation: generation,
                streaming: !noStreaming,
                useAlternateScreen: !noAltScreen,
                initialAttachments: resolvedMediaURLs
            ))
            return
        }

        if openclawConfig {
            let chosenPort = port ?? 9999
            printOpenClawConfig(model: selectedModel, hostname: hostname, port: chosenPort, modelStore: modelStore)
            return
        }

        let isSinglePromptMode = (singlePrompt != nil || isatty(STDIN_FILENO) == 0)
        if !isSinglePromptMode {
            print("MLX model: \(selectedModel)")
        }

        let contextWindow = modelStore.descriptor(for: selectedModel).contextWindow

        try ensureMLXMetalLibraryAvailable(verbose: verbose)

        // An explicit prompt must win over redirected stdin. Profilers and
        // automation runners commonly attach a pipe that remains open.
        if let prompt = singlePrompt {
            try runSinglePrompt(
                prompt,
                modelID: selectedModel,
                mediaPaths: resolvedMedia,
                chatTemplateKwargs: parsedKwargs
            )
            return
        }

        // Backward compatibility: support piped input in mlx mode too.
        if let stdinContent = try readFromStdin() {
            try runSinglePrompt(
                stdinContent,
                modelID: selectedModel,
                mediaPaths: resolvedMedia,
                chatTemplateKwargs: parsedKwargs
            )
            return
        }

        if !media.isEmpty {
            print("Error: --media requires -s (single prompt mode)")
            throw ExitCode.failure
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
        let telegramConfiguration = try makeTelegramConfiguration(
            rawBotToken: telegramBotToken,
            rawAllowlist: telegramAllow,
            hostname: hostname,
            port: chosenPort,
            modelID: selectedModel,
            instructions: instructions,
            verbose: verbose || veryVerbose || vv,
            replyFormat: telegramFormat ?? .markdown,
            requirePrefix: telegramRequirePrefix
        )

        if verbose {
            print("Loading MLX model (download if needed): \(selectedModel)")
        }

        let prewarmEnabled = prewarm.lowercased() != "n" && prewarm.lowercased() != "no" && prewarm != "0"

        _ = Task {
            do {
                let loadReporter = MLXLoadReporter(modelID: selectedModel)
                loadReporter.start()
                _ = try await mlxModel.load(progress: { fraction in
                    let progress = Progress(totalUnitCount: 1_000)
                    progress.completedUnitCount = Int64(fraction * 1_000)
                    loadReporter.updateDownload(progress)
                })
                loadReporter.finish(success: true)
                // Prewarm MLX Metal kernels (prefill + decode + gated-delta step) so the FIRST
                // real request doesn't pay the one-time ~0.35s graph/kernel compilation that
                // otherwise inflates time-to-first-token. Best-effort; never blocks serving.
                if prewarmEnabled {
                    let prewarmStart = Date()
                    do {
                        try await mlxModel.prewarm()
                        if verbose {
                            print("MLX prewarm complete in \(String(format: "%.2f", Date().timeIntervalSince(prewarmStart)))s")
                        }
                    } catch {
                        if verbose { print("MLX prewarm skipped: \(error)") }
                    }
                }
                let server = try await Server(
                    port: chosenPort,
                    hostname: hostname,
                    verbose: verbose,
                    veryVerbose: veryVerbose || vv,
                    trace: vv,
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
                    telegramConfiguration: telegramConfiguration,
                    defaultGuidedJsonSchema: defaultGuidedJsonSchema,
                    defaultChatTemplateKwargs: parsedKwargs.isEmpty
                        ? nil
                        : parsedKwargs.mapValues { AnyCodable($0) },
                    forceDisableThinking: noThink,
                    mlxModelID: selectedModel,
                    mlxModel: mlxModel,
                    mlxRepetitionPenalty: repetitionPenalty,
                    mlxTopP: topP,
                    mlxMaxTokens: maxTokens,
                    mlxRawOutput: raw,
                    mlxTopK: topK,
                    mlxMinP: minP,
                    mlxPresencePenalty: presencePenalty,
                    mlxSeed: seed,
                    mlxMaxLogprobs: maxLogprobs,
                    contextWindow: contextWindow
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

    private func resolveRuntimeBackend(model: String) throws -> MLXRuntimeBackend {
        let normalized = mlxRuntime.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard let requested = MLXRuntimeBackend(rawValue: normalized) else {
            throw ValidationError(
                "--mlx-runtime must be one of: \(MLXRuntimeBackend.allCases.map(\.rawValue).joined(separator: ", "))")
        }
        guard requested != .mlx else { return .mlx }

        let path = localModelPath(model)
        // Hugging Face snapshots expose model files as symlinks into blobs.
        // Classify the resolved target so a cached GGUF is not mistaken for MLX.
        let modelURL = URL(fileURLWithPath: path).resolvingSymlinksInPath()
        let resourceValues = try? modelURL.resourceValues(
            forKeys: [.isDirectoryKey, .isRegularFileKey])
        guard resourceValues?.isDirectory != true else {
            if requested == .dwarfstar {
                throw ValidationError(
                    "--mlx-runtime dwarfstar requires a native DwarfStar GGUF file; AFM checkpoint directories use MLX")
            }
            return .mlx
        }
        guard resourceValues?.isRegularFile == true else {
            if requested == .dwarfstar {
                throw ValidationError(
                    "--mlx-runtime dwarfstar requires a local compatible GGUF file")
            }
            return .mlx
        }
        guard AFMDwarfStarCheckpointCatalog.isDwarfStarCompatibleGGUF(at: modelURL) else {
            if requested == .dwarfstar {
                let architecture = AFMDwarfStarCheckpointCatalog.ggufArchitecture(at: modelURL)
                    ?? "unreadable"
                throw ValidationError(
                    "DwarfStar does not support GGUF architecture \(architecture)")
            }
            return .mlx
        }
        if requested == .auto {
            print(
                "[Runtime] Auto-selected DwarfStar over MLX: "
                    + "GGUF metadata declares general.architecture=deepseek4."
            )
        } else {
            print("[Runtime] Selected DwarfStar for compatible raw GGUF.")
        }
        return .dwarfstar
    }

    private func resolveRemoteDwarfStarModelIfNeeded(_ model: String) throws -> String {
        let requested = mlxRuntime.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard requested != MLXRuntimeBackend.mlx.rawValue else { return model }
        let localPath = localModelPath(model)
        let obviousLocalPrefixes = ["/", "~/", "./", "../"]
        let looksLocal = obviousLocalPrefixes.contains { model.hasPrefix($0) }
        let repositoryComponents = model.split(separator: "/", omittingEmptySubsequences: false)
        let looksLikeRepositoryID = !looksLocal
            && repositoryComponents.count == 2
            && repositoryComponents.allSatisfy { !$0.isEmpty }
        guard !FileManager.default.fileExists(atPath: localPath), looksLikeRepositoryID else {
            if ggufFile != nil {
                throw ValidationError("--gguf-file requires a Hugging Face repository ID")
            }
            return model
        }

        let group = DispatchGroup()
        let result = SendableBox<Result<URL, Error>?>(nil)
        let loadReporter = MLXLoadReporter(
            modelID: model,
            loadingLabel: "Resolving remote DwarfStar model")
        loadReporter.start()
        group.enter()
        Task.detached {
            do {
                let resolver = AFMDwarfStarHubResolver()
                result.value = .success(try await resolver.resolve(
                    repositoryID: model,
                    requestedPath: ggufFile,
                    progress: { loadReporter.updateDownload($0) }))
            } catch {
                result.value = .failure(error)
            }
            group.leave()
        }
        group.wait()

        switch result.value {
        case .success(let url):
            loadReporter.finish(success: true)
            print("[Runtime] Hugging Face GGUF resolved: \(model) -> \(url.lastPathComponent)")
            return url.path
        case .failure(let error as AFMDwarfStarHubSelectionError):
            if requested == MLXRuntimeBackend.auto.rawValue,
               case .noModelGGUF = error,
               ggufFile == nil {
                loadReporter.finish(success: true)
                return model
            }
            loadReporter.finish(success: false, errorMessage: error.localizedDescription)
            throw error
        case .failure(let error):
            loadReporter.finish(success: false, errorMessage: error.localizedDescription)
            throw error
        case .none:
            loadReporter.finish(success: false, errorMessage: "resolver returned no result")
            throw ValidationError("Hugging Face GGUF resolution ended without a result")
        }
    }

    private func localModelPath(_ model: String) -> String {
        let expanded = NSString(string: model).expandingTildeInPath
        if expanded.hasPrefix("/") {
            return URL(fileURLWithPath: expanded).standardizedFileURL.path
        }
        let cwd = ProcessInfo.processInfo.environment["PWD"]
            ?? FileManager.default.currentDirectoryPath
        return URL(fileURLWithPath: cwd, isDirectory: true)
            .appendingPathComponent(expanded)
            .standardizedFileURL.path
    }

    private func runDwarfStar(
        checkpointPath: String,
        advertisedModelID: String,
        modelStore: AFMMLXModelStore,
        chatTemplateKwargs: [String: Any],
        forceDisableThinking: Bool,
        defaultGuidedJsonSchema: ResponseFormat?
    ) throws {
        if !media.isEmpty || vlm {
            throw ValidationError("The DwarfStar runtime currently supports text input only")
        }
        if kvBits != nil || mtp || eagle3 != nil {
            throw ValidationError(
                "KV quantization and speculative decoding are unavailable in the DwarfStar runtime")
        }
        if repetitionPenalty != nil || presencePenalty != nil || guidedJson != nil {
            throw ValidationError(
                "Repetition/presence penalties and guided JSON are unavailable in the DwarfStar runtime")
        }
        guard (1...16).contains(dsparkDraftTokens) else {
            throw ValidationError("--dspark-draft-tokens must be between 1 and 16")
        }
        guard (0...1).contains(dsparkConfidenceThreshold) else {
            throw ValidationError("--dspark-confidence must be between 0 and 1")
        }
        let residentSessions = max(1, concurrent ?? 1)
        let resolvedDSparkPath = dsparkSupportPath.map(localModelPath)

        let modelID = advertisedModelID
        if openclawConfig {
            printOpenClawConfig(
                model: modelID,
                hostname: hostname,
                port: port ?? 9999,
                modelStore: modelStore)
            return
        }
        if verbose {
            print("Selected inference runtime: dwarfstar (fixed Metal schedule)")
            print("DwarfStar GGUF: \(checkpointPath)")
            if let resolvedDSparkPath {
                print("DSpark support: \(resolvedDSparkPath) (draft=\(dsparkDraftTokens), confidence=\(dsparkConfidenceThreshold), strict=\(dsparkStrict))")
            }
        }

        if let prompt = singlePrompt {
            try runSinglePrompt(
                prompt,
                modelID: modelID,
                chatTemplateKwargs: chatTemplateKwargs,
                runtimeBackend: .dwarfstar,
                modelPath: checkpointPath)
            return
        }
        if let stdinContent = try readFromStdin() {
            try runSinglePrompt(
                stdinContent,
                modelID: modelID,
                chatTemplateKwargs: chatTemplateKwargs,
                runtimeBackend: .dwarfstar,
                modelPath: checkpointPath)
            return
        }

        let explicitPort = port != nil
        let chosenPort: Int
        if let port {
            chosenPort = port
        } else if isPortAvailable(9999) {
            chosenPort = 9999
        } else {
            chosenPort = try findEphemeralPort()
            print("Port 9999 is busy, using ephemeral port \(chosenPort)")
        }
        let telegramConfiguration = try makeTelegramConfiguration(
            rawBotToken: telegramBotToken,
            rawAllowlist: telegramAllow,
            hostname: hostname,
            port: chosenPort,
            modelID: modelID,
            instructions: instructions,
            verbose: verbose || veryVerbose || vv,
            replyFormat: telegramFormat ?? .markdown,
            requirePrefix: telegramRequirePrefix)

        let model = AnyAFMModel(AFMDwarfStarModel(
            modelID: AFMModelID(rawValue: modelID),
            modelPath: checkpointPath,
            configuration: AFMDwarfStarRuntimeConfiguration(
                contextWindow: 32_768,
                dsparkSupportPath: resolvedDSparkPath,
                dsparkDraftTokens: dsparkDraftTokens,
                dsparkConfidenceThreshold: dsparkConfidenceThreshold,
                dsparkStrict: dsparkStrict,
                enablePrefixCaching: enablePrefixCaching,
                maxConcurrent: residentSessions
            )))
        let defaultChatTemplateKwargs = chatTemplateKwargs.isEmpty
            ? nil
            : chatTemplateKwargs.mapValues { AnyCodable($0) }
        _ = Task {
            do {
                _ = try await model.load(progress: nil)
                let server = try await Server(
                    port: chosenPort,
                    hostname: hostname,
                    verbose: verbose,
                    veryVerbose: veryVerbose || vv,
                    trace: vv,
                    streamingEnabled: !noStreaming,
                    instructions: instructions,
                    temperature: temperature,
                    stop: stop,
                    webuiEnabled: webui,
                    gatewayEnabled: false,
                    prewarmEnabled: false,
                    telegramConfiguration: telegramConfiguration,
                    defaultGuidedJsonSchema: defaultGuidedJsonSchema,
                    defaultChatTemplateKwargs: defaultChatTemplateKwargs,
                    forceDisableThinking: forceDisableThinking,
                    mlxModelID: modelID,
                    afmModel: model,
                    mlxTopP: topP,
                    mlxMaxTokens: maxTokens,
                    mlxRawOutput: raw,
                    mlxTopK: topK,
                    mlxMinP: minP,
                    mlxSeed: seed,
                    mlxMaxLogprobs: maxLogprobs,
                    contextWindow: 32_768)
                globalServer = server
                if !explicitPort && chosenPort != 9999 {
                    print("DwarfStar API URL: http://\(hostname):\(chosenPort)")
                }
                try await server.start()
            } catch {
                print("Error starting DwarfStar server. CTRL-C to stop: \(error)")
                shouldKeepRunning = false
            }
        }

        let runLoop = RunLoop.current
        signal(SIGINT, handleShutdown)
        signal(SIGTERM, handleShutdown)
        while shouldKeepRunning && runLoop.run(mode: .default, before: Date(timeIntervalSinceNow: 0.1)) {}
        print("Server shutdown complete.")
    }

    private func runSinglePrompt(
        _ prompt: String,
        modelID: String,
        mediaPaths: [String] = [],
        chatTemplateKwargs: [String: Any] = [:],
        runtimeBackend: MLXRuntimeBackend = .mlx,
        modelPath: String? = nil
    ) throws {
        let group = DispatchGroup()
        let output = SendableBox<Result<AFMResponse, Error>?>(nil)
        // In single-prompt mode, suppress ALL output (stdout + stderr) during model loading
        // and generation. Only the final response goes to stdout. --verbose overrides this.
        let stdoutFD = dup(STDOUT_FILENO)
        let stderrFD = dup(STDERR_FILENO)
        let quietMode = !verbose && !veryVerbose && !vv
        if stdoutFD == -1 {
            throw ValidationError("Failed to save stdout for single-prompt mode")
        }
        if quietMode {
            let devNull = open("/dev/null", O_WRONLY)
            if devNull != -1 {
                dup2(devNull, STDOUT_FILENO)
                dup2(devNull, STDERR_FILENO)
                close(devNull)
            }
        } else {
            // Verbose: redirect stdout to stderr so only the response goes to stdout
            dup2(STDERR_FILENO, STDOUT_FILENO)
        }
        let encodedChatTemplateKwargs = chatTemplateKwargs.isEmpty
            ? nil
            : try Self.afmJSONValue(from: chatTemplateKwargs)
        group.enter()
        Task {
            let engine: AFMEngine
            do {
                if runtimeBackend == .dwarfstar {
                    let registry = AFMProviderRegistry()
                    try registry.register(AFMDwarfStarProviderFactory())
                    engine = try AFMEngine(
                        providerID: AFMDwarfStarProviderFactory.providerID,
                        modelID: AFMModelID(rawValue: modelID),
                        configuration: AFMProviderConfiguration(values: [
                            "modelPath": .string(modelPath ?? modelID),
                            "contextWindow": .integer(32_768),
                            "dsparkSupportPath": self.dsparkSupportPath
                                .map { .string(self.localModelPath($0)) } ?? .null,
                            "dsparkDraftTokens": .integer(self.dsparkDraftTokens),
                            "dsparkConfidenceThreshold": .number(self.dsparkConfidenceThreshold),
                            "dsparkStrict": .bool(self.dsparkStrict),
                            "enablePrefixCaching": .bool(self.enablePrefixCaching),
                            "maxConcurrent": .integer(max(1, self.concurrent ?? 1))
                        ]),
                        registry: registry)
                } else {
                    engine = AFMEngine(
                        backend: .mlx(modelID: modelID),
                        config: EngineConfig(
                    instructions: self.instructions,
                    kvBits: self.kvBits,
                    enablePrefixCaching: self.enablePrefixCaching,
                    mlxKernels: self.mlxKernels,
                    mtpEnabled: self.mtp,
                    mtpDepth: self.mtpDepth,
                    mtpModelID: self.mtpModel,
                    eagle3DrafterPath: self.eagle3,
                    enableGrammarConstraints: self.enableGrammarConstraints,
                    toolCallParser: self.toolCallParser,
                    prefillStepSize: self.prefillStepSize,
                    kvEvictionPolicy: self.kvEviction ?? "none",
                    fixToolArguments: self.fixToolArgs,
                    forceVLM: self.vlm || !mediaPaths.isEmpty,
                    cacheProfilePath: self.cacheProfilePath,
                    trace: self.vv,
                    gpuCapturePath: self.gpuCapture,
                    gpuTraceDuration: self.gpuTrace,
                    gpuProfile: self.gpuProfile || self.gpuProfileBw,
                    gpuProfileBandwidth: self.gpuProfileBw
                        ))
                }
            } catch {
                output.value = .failure(error)
                group.leave()
                return
            }
            do {
                // Pre-load with progress bar (downloads if needed)
                let loadReporter = MLXLoadReporter(modelID: modelID)
                loadReporter.start()
                _ = try await engine.load(
                    progress: { fraction in
                        let progress = Progress(totalUnitCount: 1_000)
                        progress.completedUnitCount = Int64(fraction * 1_000)
                        loadReporter.updateDownload(progress)
                    }
                )
                loadReporter.finish(success: true)

                var messages = [Message]()
                if !self.instructions.isEmpty {
                    messages.append(Message(role: "system", content: self.instructions))
                }

                if mediaPaths.isEmpty {
                    messages.append(Message(role: "user", content: prompt))
                } else {
                    // Build multipart message with text + media references
                    var parts: [ContentPart] = [ContentPart(type: "text", text: prompt, image_url: nil)]
                    for path in mediaPaths {
                        let fileURL = URL(fileURLWithPath: path)
                        parts.append(ContentPart(
                            type: "image_url",
                            text: nil,
                            image_url: ImageURL(
                                url: try AFMMLXMediaSecurityPolicy.trustedLocalMediaDataURL(fileURL),
                                detail: nil
                            )
                        ))
                    }
                    messages.append(Message(role: "user", content: .parts(parts)))
                }
                var responseFormat: ResponseFormat? = nil
                if let guidedJson = self.guidedJson {
                    let schema = try parseGuidedJsonSchema(guidedJson)
                    responseFormat = ResponseFormat(type: "json_schema", jsonSchema: schema)
                }
                let requestTools = try Self.parseToolsJSON(self.toolsJson)
                let stopSequences: [String]? = stop.map { $0.split(separator: ",").map { String($0.trimmingCharacters(in: .whitespaces)) } }
                var metadata: [String: AFMJSONValue] = [:]
                if let encodedChatTemplateKwargs {
                    metadata["chatTemplateKwargs"] = encodedChatTemplateKwargs
                }
                let res = try await engine.respond(
                    to: messages,
                    GenerationConfig(
                    temperature: temperature,
                    maxTokens: maxTokens,
                    topP: topP,
                    topK: topK,
                    minP: minP,
                    repetitionPenalty: repetitionPenalty,
                    presencePenalty: presencePenalty,
                    seed: seed,
                    logprobs: maxLogprobs != nil,
                    topLogprobs: maxLogprobs,
                    stop: stopSequences,
                    tools: requestTools,
                    responseFormat: responseFormat,
                    metadata: metadata
                    )
                )
                output.value = .success(res)
            } catch {
                MLXLoadReporter.finishActiveWithError(error.localizedDescription)
                output.value = .failure(error)
            }
            await engine.unload()
            group.leave()
        }
        group.wait()
        fflush(stdout)
        fflush(stderr)
        // Restore stdout (and stderr if we suppressed it)
        dup2(stdoutFD, STDOUT_FILENO)
        close(stdoutFD)
        if stderrFD != -1 {
            dup2(stderrFD, STDERR_FILENO)
            close(stderrFD)
        }

        switch output.value {
        case .success(let response):
            if json {
                try printJSONResponse(response, modelID: modelID)
                return
            }
            if raw {
                if let reasoning = response.reasoningContent, !reasoning.isEmpty {
                    print("<think>\(reasoning)</think>\(response.content)")
                } else {
                    print(response.content)
                }
            } else {
                if response.content.isEmpty,
                   response.reasoningContent?.isEmpty == false {
                    print("(no visible response — model used all tokens for reasoning. Try increasing --max-tokens)")
                    return
                }
                print(response.content)
            }
        case .failure(let error):
            FileHandle.standardError.write(Data("Error: \(error.localizedDescription)\n".utf8))
            throw ExitCode.failure
        case .none:
            throw ExitCode.failure
        }
    }

    private func printJSONResponse(_ response: AFMResponse, modelID: String) throws {
        let choiceLogprobs = Self.buildChoiceLogprobs(response.logprobs)
        let encoded: ChatCompletionResponse
        if let toolCalls = response.toolCalls, !toolCalls.isEmpty {
            encoded = ChatCompletionResponse(
                model: modelID,
                toolCalls: toolCalls,
                logprobs: choiceLogprobs,
                promptTokens: response.promptTokens,
                completionTokens: response.completionTokens,
                cachedTokens: response.cachedPromptTokens
            )
        } else {
            encoded = ChatCompletionResponse(
                model: modelID,
                content: response.content,
                reasoningContent: response.reasoningContent,
                logprobs: choiceLogprobs,
                finishReason: Self.openAIFinishReason(response.finishReason),
                promptTokens: response.promptTokens,
                completionTokens: response.completionTokens,
                cachedTokens: response.cachedPromptTokens
            )
        }

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let data = try encoder.encode(encoded)
        guard let text = String(data: data, encoding: .utf8) else {
            throw ValidationError("Failed to encode JSON response as UTF-8")
        }
        print(text)
    }

    private static func openAIFinishReason(_ reason: AFMFinishReason) -> String {
        switch reason {
        case .stop: return "stop"
        case .length: return "length"
        case .toolCalls: return "tool_calls"
        case .cancelled: return "cancelled"
        case .contentFilter: return "content_filter"
        case .error: return "error"
        case .unknown: return "unknown"
        }
    }

    private static func buildChoiceLogprobs(_ resolved: [AFMTokenLogProbability]?) -> ChoiceLogprobs? {
        guard let resolved, !resolved.isEmpty else { return nil }
        let content = resolved.map { entry in
            let topLogprobs = entry.topTokens.map { top in
                TopLogprobEntry(
                    token: top.token,
                    logprob: Double(top.logprob),
                    bytes: Array(top.token.utf8).map { Int($0) }
                )
            }
            return TokenLogprobContent(
                token: entry.token,
                logprob: Double(entry.logprob),
                bytes: Array(entry.token.utf8).map { Int($0) },
                topLogprobs: topLogprobs
            )
        }
        return ChoiceLogprobs(content: content)
    }

    private static func parseToolsJSON(_ value: String?) throws -> [RequestTool]? {
        guard let value, !value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return nil
        }
        guard let data = value.data(using: .utf8) else {
            throw ValidationError("--tools-json must be valid UTF-8")
        }
        do {
            return try JSONDecoder().decode([RequestTool].self, from: data)
        } catch {
            throw ValidationError("--tools-json must be an OpenAI-compatible tools array: \(error.localizedDescription)")
        }
    }

    private static func afmJSONValue(from value: Any) throws -> AFMJSONValue {
        switch value {
        case is NSNull:
            return .null
        case let value as Bool:
            return .bool(value)
        case let value as Int:
            return .integer(value)
        case let value as NSNumber:
            return .number(value.doubleValue)
        case let value as String:
            return .string(value)
        case let values as [Any]:
            return .array(try values.map { try afmJSONValue(from: $0) })
        case let values as [String: Any]:
            return .object(
                try values.mapValues { try afmJSONValue(from: $0) }
            )
        default:
            throw ValidationError(
                "Unsupported chat-template value: \(type(of: value))"
            )
        }
    }

    private static func stripThinkContent(from text: String, startTag: String, endTag: String) -> String {
        var output = text
        while let start = output.range(of: startTag) {
            if let end = output.range(of: endTag, range: start.upperBound..<output.endIndex) {
                output.removeSubrange(start.lowerBound..<end.upperBound)
            } else {
                // Unclosed think tag — truncated output or grammar-constrained
                // generation. Strip everything from the opening tag onwards
                // since it's all thinking content without a visible response.
                output.removeSubrange(start.lowerBound..<output.endIndex)
            }
        }
        return output.trimmingCharacters(in: .whitespacesAndNewlines)
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

    private func printOpenClawConfig(model: String, hostname: String, port: Int, modelStore: AFMMLXModelStore) {
        let descriptor = modelStore.descriptor(for: model)
        let capabilities = descriptor.capabilities
        let supportsVision = capabilities.contains(.vision)
        let supportsReasoning = capabilities.contains(.reasoning)
        let contextWindow = descriptor.contextWindow ?? 131_072
        let defaultMaxTokens = min(8_192, contextWindow)
        let shortName = descriptor.displayName.isEmpty
            ? (model.split(separator: "/", maxSplits: 1).last.map(String.init) ?? model)
            : descriptor.displayName

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
        if trustRemoteCode { ignored.append("--trust-remote-code") }
        if chatTemplate != nil { ignored.append("--chat-template") }
        if dtype != nil { ignored.append("--dtype") }
        // --vlm is now functional, not ignored
        if !ignored.isEmpty {
            print("Warning: accepted compatibility switches currently ignored: \(ignored.joined(separator: ", "))")
        }
    }
}

// MARK: - Help JSON

/// Extract the YAML capability card from a command's discussion text and emit it as JSON.
/// Parses the YAML between `---` delimiters in the ArgumentParser discussion field.
func printHelpJson(command: String) {
    let config: CommandConfiguration
    switch command {
    case "afm mlx":
        config = MlxCommand.configuration
    case "afm vision":
        config = VisionCommand.configuration
    case "afm speech":
        config = SpeechCommand.configuration
    default:
        config = RootCommand.configuration
    }

    let discussion = config.discussion ?? ""
    guard !discussion.isEmpty else {
        print("{}")
        return
    }

    // Extract YAML between --- delimiters
    let lines = discussion.components(separatedBy: "\n")
    var yamlLines: [String] = []
    var inYaml = false
    for line in lines {
        let trimmed = line.trimmingCharacters(in: .whitespaces)
        if trimmed == "---" {
            if inYaml { break }
            inYaml = true
            continue
        }
        if inYaml { yamlLines.append(line) }
    }

    // Determine baseline indentation from first non-empty YAML line
    let baseIndent = yamlLines.first(where: { !$0.trimmingCharacters(in: .whitespaces).isEmpty })?
        .prefix(while: { $0 == " " }).count ?? 0

    // YAML-to-dict parser: supports 3 levels of nesting with scalars, inline arrays, dicts, and lists
    var root: [String: Any] = [:]
    root["version"] = BuildInfo.fullVersion

    // Level 1: top-level key being accumulated
    var l1Key: String?
    var l1List: [Any]?
    var l1Dict: [String: Any]?

    // Level 2: sub-key within an l1 dict
    var l2Key: String?
    var l2List: [String]?
    var l2Dict: [String: String]?

    func flushL2() {
        guard let k2 = l2Key else { return }
        if let list = l2List { l1Dict?[k2] = list }
        else if let dict = l2Dict { l1Dict?[k2] = dict }
        l2Key = nil
        l2List = nil
        l2Dict = nil
    }

    func flushL1() {
        flushL2()
        guard let k1 = l1Key else { return }
        if let dict = l1Dict { root[k1] = dict }
        else if let list = l1List { root[k1] = list }
        l1Key = nil
        l1List = nil
        l1Dict = nil
    }

    for line in yamlLines {
        let stripped = line.trimmingCharacters(in: .whitespaces)
        if stripped.isEmpty { continue }
        let indent = line.prefix(while: { $0 == " " }).count
        let rel = indent - baseIndent

        if rel == 0 && stripped.contains(":") && !stripped.hasPrefix("- ") {
            // Top-level key (indent 0)
            flushL1()
            let parts = stripped.split(separator: ":", maxSplits: 1)
            let key = String(parts[0]).trimmingCharacters(in: .whitespaces)
            let value = parts.count > 1 ? String(parts[1]).trimmingCharacters(in: .whitespaces) : ""
            if !value.isEmpty {
                if value.hasPrefix("[") && value.hasSuffix("]") {
                    let inner = String(value.dropFirst().dropLast())
                    root[key] = inner.components(separatedBy: ",").map { $0.trimmingCharacters(in: .whitespaces) }
                } else {
                    root[key] = value
                }
            } else {
                l1Key = key
            }
        } else if rel == 2 && stripped.contains(":") && !stripped.hasPrefix("- ") {
            // Level-2 key (child of current l1Key)
            flushL2()
            let parts = stripped.split(separator: ":", maxSplits: 1)
            let key = String(parts[0]).trimmingCharacters(in: .whitespaces)
            let value = parts.count > 1 ? String(parts[1]).trimmingCharacters(in: .whitespaces) : ""
            if l1Dict == nil && l1List == nil { l1Dict = [:] }
            if let _ = l1Dict {
                if !value.isEmpty {
                    if value.hasPrefix("[") && value.hasSuffix("]") {
                        let inner = String(value.dropFirst().dropLast())
                        l1Dict?[key] = inner.components(separatedBy: ",").map { $0.trimmingCharacters(in: .whitespaces) }
                    } else {
                        l1Dict?[key] = value
                    }
                } else {
                    l2Key = key
                }
            }
        } else if stripped.hasPrefix("- ") {
            let item = String(stripped.dropFirst(2)).trimmingCharacters(in: .whitespaces)
            if l2Key != nil {
                if l2List == nil { l2List = [] }
                l2List?.append(item)
            } else {
                if l1List == nil { l1List = [] }
                if item.contains(": ") && !item.hasPrefix("afm") && !item.hasPrefix("curl") && !item.hasPrefix("'") && !item.hasPrefix("MACAFM") && !item.hasPrefix("\"") {
                    let kv = item.split(separator: ":", maxSplits: 1)
                    if kv.count == 2 {
                        l1List?.append([String(kv[0]).trimmingCharacters(in: .whitespaces): String(kv[1]).trimmingCharacters(in: .whitespaces)])
                        continue
                    }
                }
                l1List?.append(item)
            }
        } else if rel == 4 && l2Key != nil && stripped.contains(":") && !stripped.hasPrefix("- ") {
            // Level-3 key: child of l2 sub-dict
            let parts = stripped.split(separator: ":", maxSplits: 1)
            if parts.count == 2 {
                if l2Dict == nil { l2Dict = [:] }
                l2Dict?[String(parts[0]).trimmingCharacters(in: .whitespaces)] = String(parts[1]).trimmingCharacters(in: .whitespaces)
            }
        } else if rel > 0 && stripped.contains(":") {
            let parts = stripped.split(separator: ":", maxSplits: 1)
            if parts.count == 2 {
                if l1Dict == nil && l1List == nil { l1Dict = [:] }
                l1Dict?[String(parts[0]).trimmingCharacters(in: .whitespaces)] = String(parts[1]).trimmingCharacters(in: .whitespaces)
            }
        }
    }
    flushL1()

    if let jsonData = try? JSONSerialization.data(withJSONObject: root, options: [.prettyPrinted, .sortedKeys]),
       let jsonString = String(data: jsonData, encoding: .utf8) {
        print(jsonString)
    }
}

struct MacLocalAPI: ParsableCommand {
    static let buildVersion: String = BuildInfo.fullVersion

    static let configuration = CommandConfiguration(
        commandName: "afm",
        abstract: "macOS server that exposes Apple's Foundation Models through OpenAI-compatible API",
        discussion: """
        ---
        name: afm
        description: OpenAI-compatible local LLM inference server for Apple Silicon. Supports Apple Foundation Models (on-device, macOS 26+), MLX models from Hugging Face, API gateway proxying to local backends (Ollama, LM Studio, Jan), and Vision OCR. Exposes /v1/chat/completions and /v1/models endpoints.
        tags: [llm, inference, apple-silicon, openai-compatible, mlx, foundation-models, local, server, api, streaming, tool-calling, vision, ocr, gateway]
        subcommands:
          mlx:
            description: Run MLX-format LLM/VLM models from Hugging Face on Apple Silicon
            usage: afm mlx -m <model> [options]
            full_details: afm mlx --help-json
          vision:
            description: Extract text and tables from images/PDFs using Apple Vision OCR
            usage: afm vision -f <file> [--table]
            full_details: afm vision --help-json
          speech:
            description: Transcribe audio to text and synthesize text to speech using Apple Speech/AVFoundation
            usage: afm speech transcribe -f <file> | afm speech synthesize <text> -o <file>
            full_details: afm speech --help-json
          embed:
            description: Serve OpenAI-compatible embeddings using Apple NaturalLanguage contextual embeddings
            usage: afm embed -m <model> [--port 9998]
            full_details: afm embed --list-models
        api_endpoints: [/v1/chat/completions, /v1/models, /v1/vision/ocr, /v1/embeddings, /health]
        env_vars:
          MACAFM_MLX_MODEL_CACHE: Override model cache directory
          MACAFM_MLX_METALLIB: Override metallib path
          AFM_DEBUG: Enable debug logging (KVCache, tool calls, timing)
          AFM_PERF: Enable per-token performance instrumentation
        cli_flags:
          -s, --single-prompt: Run a single prompt and exit (no server)
          -i, --instructions: System prompt / custom instructions
          -p, --port: Server port (default: 9999)
          -H, --hostname: Bind address (default: 127.0.0.1)
          -v, --verbose: Enable verbose logging
          -V, --very-verbose: Log full requests/responses
          -w, --webui: Enable WebUI and open in browser
          --tui: Run the native interactive terminal chat UI
          --no-alt-screen: Disable alternate-screen overlays and keep the TUI inline
          --telegram-bot-token: Telegram bot token for remote AFM access
          --telegram-allow: Comma-separated allowlist of Telegram numeric user IDs
          --telegram-format: Telegram reply format: markdown, plain, or html
          --telegram-require-prefix: Require a specific prefix for Telegram messages, for example '/afm'
          -g, --gateway: Enable API gateway (discover/proxy Ollama, LM Studio, Jan, etc.)
          -t, --temperature: Sampling temperature (0.0-1.0)
          -r, --randomness: "greedy", "random", "random:top-p=0.9", "random:top-k=40", ":seed=42"
          -P, --permissive-guardrails: Disable safety guardrails
          -a, --adapter: Path to .fmadapter LoRA adapter file
          --stop: Stop sequences, comma-separated
          --guided-json: Constrain output to JSON schema (auto-disables thinking on reasoning models)
          --no-streaming: Disable streaming
          --prewarm: Pre-warm model on startup (y/n, default: y)
          --help-json: Print machine-readable JSON capability card for AI agents and exit
        skill:
          what_it_does: Provides local OpenAI-compatible LLM inference on Apple Silicon. Two modes — Apple Foundation Models (on-device, macOS 26+) and MLX (open-source HuggingFace models).
          repository: https://github.com/scouzi1966/maclocal-api
          modes:
            - "afm" — Apple Foundation Models (on-device, requires macOS 26+)
            - "afm mlx -m <model>" — MLX open-source models from Hugging Face
            - "afm vision -f <file>" — Vision OCR text/table extraction
            - "afm speech transcribe -f <file>" — Speech transcription and synthesis
            - "afm embed -m <model>" — OpenAI-compatible embeddings (Apple NaturalLanguage)
            - "afm -g" — API gateway proxying to local backends
        triggers:
          - start local LLM server
          - run MLX model locally
          - OpenAI-compatible local inference
          - Apple Foundation Models API
          - local tool calling server
          - vision OCR text extraction
          - speech transcription and synthesis
          - local text embeddings for RAG / semantic search
          - API gateway for local LLM backends
        examples:
          - afm --port 9999
          - afm mlx -m Qwen/Qwen3-Coder-Next-4bit --port 9999
          - afm mlx -m mlx-community/Meta-Llama-3.1-8B-Instruct-4bit -s "Hello"
          - afm vision -f image.png
          - afm speech transcribe -f recording.wav
          - afm embed -m apple-nl-contextual-en --port 9998
          - afm -g --port 9999
        ---

        Use -w to enable the WebUI, -g to enable API gateway mode, or `afm mlx` for local MLX models.

        GitHub: https://github.com/scouzi1966/maclocal-api
        """,
        version: buildVersion
    )
}

struct RootCommand: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "afm",
        abstract: "macOS server that exposes Apple's Foundation Models through OpenAI-compatible API",
        discussion: """
        ---
        name: afm
        description: OpenAI-compatible local LLM inference server for Apple Silicon. Supports Apple Foundation Models (on-device, macOS 26+), MLX models from Hugging Face, API gateway proxying to local backends (Ollama, LM Studio, Jan), and Vision OCR. Exposes /v1/chat/completions and /v1/models endpoints.
        tags: [llm, inference, apple-silicon, openai-compatible, mlx, foundation-models, local, server, api, streaming, tool-calling, vision, ocr, gateway]
        subcommands:
          mlx:
            description: Run MLX-format LLM/VLM models from Hugging Face on Apple Silicon
            usage: afm mlx -m <model> [options]
            full_details: afm mlx --help-json
          vision:
            description: Extract text and tables from images/PDFs using Apple Vision OCR
            usage: afm vision -f <file> [--table]
            full_details: afm vision --help-json
          speech:
            description: Transcribe audio to text and synthesize text to speech using Apple Speech/AVFoundation
            usage: afm speech transcribe -f <file> | afm speech synthesize <text> -o <file>
            full_details: afm speech --help-json
          embed:
            description: Serve OpenAI-compatible embeddings using Apple NaturalLanguage contextual embeddings
            usage: afm embed -m <model> [--port 9998]
            full_details: afm embed --list-models
        api_endpoints: [/v1/chat/completions, /v1/models, /v1/vision/ocr, /v1/embeddings, /health]
        env_vars:
          MACAFM_MLX_MODEL_CACHE: Override model cache directory
          MACAFM_MLX_METALLIB: Override metallib path
          AFM_DEBUG: Enable debug logging (KVCache, tool calls, timing)
          AFM_PERF: Enable per-token performance instrumentation
        cli_flags:
          -s, --single-prompt: Run a single prompt and exit (no server)
          -i, --instructions: System prompt / custom instructions
          -p, --port: Server port (default: 9999)
          -H, --hostname: Bind address (default: 127.0.0.1)
          -v, --verbose: Enable verbose logging
          -V, --very-verbose: Log full requests/responses
          -w, --webui: Enable WebUI and open in browser
          --telegram-bot-token: Telegram bot token for remote AFM access
          --telegram-allow: Comma-separated allowlist of Telegram numeric user IDs
          --telegram-format: Telegram reply format: markdown, plain, or html
          --telegram-require-prefix: Require a specific prefix for Telegram messages, for example '/afm'
          -g, --gateway: Enable API gateway (discover/proxy Ollama, LM Studio, Jan, etc.)
          -t, --temperature: Sampling temperature (0.0-1.0)
          -r, --randomness: "greedy", "random", "random:top-p=0.9", "random:top-k=40", ":seed=42"
          -P, --permissive-guardrails: Disable safety guardrails
          -a, --adapter: Path to .fmadapter LoRA adapter file
          --stop: Stop sequences, comma-separated
          --guided-json: Constrain output to JSON schema (auto-disables thinking on reasoning models)
          --no-streaming: Disable streaming
          --prewarm: Pre-warm model on startup (y/n, default: y)
          --help-json: Print machine-readable JSON capability card for AI agents and exit
        skill:
          what_it_does: Provides local OpenAI-compatible LLM inference on Apple Silicon. Two modes — Apple Foundation Models (on-device, macOS 26+) and MLX (open-source HuggingFace models).
          repository: https://github.com/scouzi1966/maclocal-api
          modes:
            - "afm" — Apple Foundation Models (on-device, requires macOS 26+)
            - "afm mlx -m <model>" — MLX open-source models from Hugging Face
            - "afm vision -f <file>" — Vision OCR text/table extraction
            - "afm speech transcribe -f <file>" — Speech transcription and synthesis
            - "afm embed -m <model>" — OpenAI-compatible embeddings (Apple NaturalLanguage)
            - "afm -g" — API gateway proxying to local backends
        triggers:
          - start local LLM server
          - run MLX model locally
          - OpenAI-compatible local inference
          - Apple Foundation Models API
          - local tool calling server
          - vision OCR text extraction
          - speech transcription and synthesis
          - local text embeddings for RAG / semantic search
          - API gateway for local LLM backends
        examples:
          - afm --port 9999
          - afm mlx -m Qwen/Qwen3-Coder-Next-4bit --port 9999
          - afm mlx -m mlx-community/Meta-Llama-3.1-8B-Instruct-4bit -s "Hello"
          - afm vision -f image.png
          - afm speech transcribe -f recording.wav
          - afm embed -m apple-nl-contextual-en --port 9998
          - afm -g --port 9999
        ---

        Use -w to enable the WebUI, -g to enable API gateway mode, or `afm mlx` for local MLX models.

        GitHub: https://github.com/scouzi1966/maclocal-api
        """,
        version: MacLocalAPI.buildVersion,
        subcommands: [
            MlxCommand.self, MLXConvertCommand.self, MLXAlignExecutorCommand.self,
            DwarfStarBenchmarkCommand.self,
            VisionCommand.self,
            SpeechCommand.self, EmbeddingsCommand.self,
        ]
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

    @Option(name: .shortAndLong, help: "Port to run server on (default: 9999, falls back to ephemeral if busy)")
    var port: Int?

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

    @Flag(name: .long, help: "Run the advanced native terminal chat UI")
    var tui: Bool = false

    @Flag(name: .long, help: "Disable alternate-screen overlays and keep the TUI inline")
    var noAltScreen: Bool = false

    @Option(name: .long, help: "Telegram bot token for remote AFM access")
    var telegramBotToken: String?

    @Option(name: .long, help: "Enable Telegram bridge with a comma-separated allowlist of Telegram numeric user IDs")
    var telegramAllow: String?

    @Option(name: .long, help: "Telegram reply format: markdown, plain, or html (default: markdown)")
    var telegramFormat: TelegramReplyFormat?

    @Option(name: .long, help: "Require a specific prefix for Telegram messages, for example '/afm' (default: no prefix required)")
    var telegramRequirePrefix: String?

    @Flag(name: [.customShort("g"), .long], help: "Enable API gateway mode: discover and proxy to local LLM backends (Ollama, LM Studio, Jan, etc.)")
    var gateway: Bool = false

    @Option(name: .long, help: "Constrain output to match a JSON schema (vLLM-compatible). Auto-disables thinking on reasoning models for deterministic output.")
    var guidedJson: String?

    @Option(name: .long, help: "Stop sequences - comma-separated strings where generation should stop (e.g., '###,END')")
    var stop: String?

    @Option(name: .long, help: "Pre-warm the model on server startup for faster first response (y/n, default: y)")
    var prewarm: String = "y"

    @Flag(name: .long, help: "Print machine-readable JSON capability card for AI agents and exit")
    var helpJson: Bool = false

    func run() throws {
        if helpJson {
            printHelpJson(command: "afm")
            return
        }

        // Validate temperature parameter
        if let temp = temperature {
            guard temp >= 0.0 && temp <= 1.0 else {
                throw ValidationError("Temperature must be between 0.0 and 1.0")
            }
        }

        // Validate randomness parameter
        if let rand = randomness {
#if compiler(>=6.4)
            do {
                _ = try RandomnessConfig.parse(rand)
            } catch let error as FoundationModelError {
                throw ValidationError(error.localizedDescription)
            } catch {
                throw ValidationError("Invalid randomness parameter format")
            }
#else
            throw ValidationError(
                "--randomness requires the Swift 6.4 toolchain or newer")
#endif
        }

        let hasTelegramOptions = TUIInvocationPolicy.hasTelegramOptions(
            botToken: telegramBotToken,
            allowlist: telegramAllow,
            replyFormat: telegramFormat?.rawValue,
            requirePrefix: telegramRequirePrefix
        )
        if hasTelegramOptions && (singlePrompt != nil || isatty(STDIN_FILENO) == 0) {
            throw ValidationError("--telegram requires server mode and cannot be used with -s or piped single-prompt input")
        }

        do {
            try TUIInvocationPolicy.validate(
                tui: tui,
                webUI: webui,
                singlePrompt: singlePrompt != nil,
                telegramOptions: hasTelegramOptions,
                inputIsTTY: isatty(STDIN_FILENO) != 0,
                outputIsTTY: isatty(STDOUT_FILENO) != 0
            )
        } catch {
            throw ValidationError(error.localizedDescription)
        }

        if tui {
            if gateway { throw ValidationError("--tui cannot be combined with --gateway") }
            let responseFormat: ResponseFormat?
            if let guidedJson {
                responseFormat = ResponseFormat(type: "json_schema", jsonSchema: try parseGuidedJsonSchema(guidedJson))
            } else {
                responseFormat = nil
            }
            try runTerminalChat(TerminalChatConfiguration(
                backend: .foundationModels,
                backendName: "Foundation Models",
                modelName: "apple-foundation-model",
                engine: EngineConfig(
                    instructions: instructions,
                    adapter: adapter,
                    permissiveGuardrails: permissiveGuardrails,
                    foundationRandomness: randomness
                ),
                generation: GenerationConfig(
                    temperature: temperature,
                    stop: stop?.split(separator: ",").map(String.init),
                    responseFormat: responseFormat
                ),
                streaming: !noStreaming,
                useAlternateScreen: !noAltScreen
            ))
            return
        }

        // Handle single-prompt mode for backward compatibility
        if let prompt = singlePrompt {
            return try runSinglePrompt(prompt, adapter: adapter)
        }

        // Check for piped input for backward compatibility
        if let stdinContent = try readFromStdin() {
            return try runSinglePrompt(stdinContent, adapter: adapter)
        }

        // If no subcommand specified and no single prompt, run server.
        // Build argument array and parse — direct struct init doesn't work
        // with ArgumentParser property wrappers (they need parse() to initialize).
        var args: [String] = ["--hostname", hostname, "--instructions", instructions, "--prewarm", prewarm]
        if let port { args += ["--port", "\(port)"] }
        if verbose { args.append("--verbose") }
        if veryVerbose { args.append("--very-verbose") }
        if noStreaming { args.append("--no-streaming") }
        if permissiveGuardrails { args.append("--permissive-guardrails") }
        if webui { args.append("--webui") }
        if gateway { args.append("--gateway") }
        if let telegramBotToken { args += ["--telegram-bot-token", telegramBotToken] }
        if let telegramAllow { args += ["--telegram-allow", telegramAllow] }
        if let telegramFormat { args += ["--telegram-format", telegramFormat.rawValue] }
        if let telegramRequirePrefix { args += ["--telegram-require-prefix", telegramRequirePrefix] }
        if let adapter { args += ["--adapter", adapter] }
        if let temperature { args += ["--temperature", "\(temperature)"] }
        if let randomness { args += ["--randomness", randomness] }
        if let stop { args += ["--stop", stop] }
        if let guidedJson { args += ["--guided-json", guidedJson] }
        var serveCommand = try ServeCommand.parse(args)
        try serveCommand.run()
    }
}

// Manual dispatch for subcommands to avoid flag conflicts between root and subcommands.
// Subcommands are still registered in RootCommand.configuration so they appear in -h.
if CommandLine.arguments.count > 1 && CommandLine.arguments[1] == "__tui-preview" {
    if CommandLine.arguments.count != 3 {
        fputs("Invalid TUI preview invocation.\n", stderr)
        exit(EXIT_FAILURE)
    }
    do {
        let artifactURL = URL(fileURLWithPath: CommandLine.arguments[2]).standardizedFileURL
        try MainActor.assumeIsolated {
            try TUIBrowserPreview.run(artifactURL: artifactURL)
        }
    } catch {
        fputs("Unable to open TUI preview: \(error.localizedDescription)\n", stderr)
        exit(EXIT_FAILURE)
    }
} else if CommandLine.arguments.count > 1 && CommandLine.arguments[1] == "dwarfstar-bench" {
    let args = Array(CommandLine.arguments.dropFirst(2))
    do {
        var cmd = try DwarfStarBenchmarkCommand.parse(args)
        let group = DispatchGroup()
        let errorBox = SendableBox<Error?>(nil)
        group.enter()
        Task.detached {
            do {
                try await cmd.run()
            } catch {
                errorBox.value = error
            }
            group.leave()
        }
        group.wait()
        if let error = errorBox.value {
            throw error
        }
    } catch {
        DwarfStarBenchmarkCommand.exit(withError: error)
    }
} else if CommandLine.arguments.count > 1 && CommandLine.arguments[1] == "mlx" {
    let args = Array(CommandLine.arguments.dropFirst(2))
    do {
        var cmd = try MlxCommand.parseAsRoot(args)
        try cmd.run()
    } catch {
        MlxCommand.exit(withError: error)
    }
} else if CommandLine.arguments.count > 1 && CommandLine.arguments[1] == "speech" {
    var args = Array(CommandLine.arguments.dropFirst(2))
    // Legacy shim: pre-subcommand CLI accepted `afm speech file.wav`,
    // `afm speech -f file.wav`, and `afm speech --list-voices`. Route each
    // legacy form to the matching subcommand so existing muscle memory keeps
    // working. Root options would otherwise swallow subcommand options like
    // `--locale`, so the root stays deliberately flagless (except --help-json).
    let subcommands: Set<String> = ["synthesize", "transcribe", "voices", "help"]
    let transcribeFlags: Set<String> = ["-f", "--file", "--format", "--language", "--timestamps"]
    if let firstIdx = args.firstIndex(of: "--list-voices") {
        // Drop the flag and prepend `voices` so any remaining flags (e.g. --locale)
        // bind to SpeechVoicesCommand.
        args.remove(at: firstIdx)
        args.insert("voices", at: 0)
    } else if let first = args.first, !subcommands.contains(first) {
        if !first.hasPrefix("-") {
            // Bare positional — treat as transcribe file path
            args.insert(contentsOf: ["transcribe", "-f"], at: 0)
        } else if transcribeFlags.contains(first) {
            args.insert("transcribe", at: 0)
        }
        // Other flags (e.g. --help, --help-json) fall through to SpeechCommand.
    }
    do {
        let cmd = try SpeechCommand.parseAsRoot(args)
        // Use CFRunLoop so AVSpeechSynthesizer callbacks can fire on the main thread.
        // The non-Sendable command and the caught error cross into the Task; the
        // CFRunLoopStop happens-before the reads below, so these boxes are sound.
        let cmdBox = UncheckedSendable(cmd)
        let errorBox = SendableBox<Error?>(nil)
        Task {
            do {
                if var asyncCmd = cmdBox.value as? AsyncParsableCommand {
                    try await asyncCmd.run()
                } else {
                    var syncCmd = cmdBox.value
                    try syncCmd.run()
                }
            } catch {
                errorBox.value = error
            }
            CFRunLoopStop(CFRunLoopGetMain())
        }
        CFRunLoopRun()
        if let error = errorBox.value {
            throw error
        }
    } catch {
        SpeechCommand.exit(withError: error)
    }
} else if CommandLine.arguments.count > 1 && CommandLine.arguments[1] == "vision" {
    let args = Array(CommandLine.arguments.dropFirst(2))
    do {
        let cmd = try VisionCommand.parse(args)
        let group = DispatchGroup()
        var caughtError: Error?
        group.enter()
        Task.detached {
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
} else if CommandLine.arguments.count > 1 && CommandLine.arguments[1] == "embed" {
    let args = Array(CommandLine.arguments.dropFirst(2))
    do {
        let cmd = try EmbeddingsCommand.parse(args)
        let group = DispatchGroup()
        var caughtError: Error?
        group.enter()
        Task.detached {
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
        EmbeddingsCommand.exit(withError: error)
    }
} else {
    RootCommand.main()
}

private func ensureMLXMetalLibraryAvailable(verbose: Bool) throws {
    try MLXMetalLibrary.ensureAvailable(verbose: verbose)
}

private func debugLog(_ message: @autoclosure () -> String) {
    guard ProcessInfo.processInfo.environment["AFM_DEBUG"] == "1" else { return }
    print("DEBUG: \(message())")
}

private enum MLXLoadReporterStage: String {
    case checkingCache = "checking cache"
    case resuming
    case downloading
    case loadingModel = "loading model"
    case ready
}

final class MLXLoadReporter: @unchecked Sendable {
    private static let reporterLock = NSLock()
    nonisolated(unsafe) private static weak var activeReporter: MLXLoadReporter?

    private let modelID: String
    private let loadingLabel: String
    private let lock = NSLock()
    private var stage: MLXLoadReporterStage = .checkingCache
    private var downloadFraction: Double?
    private var downloadCompletedBytes: Int64?
    private var downloadTotalBytes: Int64?
    private var downloadBytesPerSecond: Double?
    private var downloadETA: TimeInterval?
    private var downloadCurrentFiles: [String] = []
    private var downloadCompletedFiles: Int?
    private var downloadTotalFiles: Int?
    private var downloadCurrentTransports: [String] = []
    private var lastDownloadSample: (date: Date, completed: Int64)?
    private var timer: DispatchSourceTimer?
    private var spinnerIndex: Int = 0
    private var startedAt = Date()
    private var finished = false

    private let spinnerFrames = ["|", "/", "-", "\\"]

    init(modelID: String, loadingLabel: String = "Loading MLX model") {
        self.modelID = modelID
        self.loadingLabel = loadingLabel
    }

    func start() {
        Self.reporterLock.lock()
        Self.activeReporter = self
        Self.reporterLock.unlock()

        startedAt = Date()
        Self.writeDiagnostic("\(loadingLabel): \(modelID)\n")

        let timer = DispatchSource.makeTimerSource(queue: DispatchQueue.global(qos: .utility))
        timer.schedule(deadline: .now(), repeating: .milliseconds(200))
        timer.setEventHandler { [weak self] in
            self?.renderTick()
        }
        self.timer = timer
        timer.resume()
    }

    func updateDownload(_ progress: Progress) {
        let now = Date()
        let completed = max(0, progress.completedUnitCount)
        let total = max(0, progress.totalUnitCount)
        lock.lock()
        if stage != .resuming { stage = .downloading }
        downloadFraction = total > 0 ? progress.fractionCompleted : nil
        downloadCompletedBytes = completed
        downloadTotalBytes = total > 1 ? total : nil
        downloadCurrentFiles = progress.userInfo[AFMDownloadProgressUserInfo.currentFiles] as? [String] ?? []
        downloadCompletedFiles = progress.userInfo[AFMDownloadProgressUserInfo.completedFiles] as? Int
        downloadTotalFiles = progress.userInfo[AFMDownloadProgressUserInfo.totalFiles] as? Int
        downloadCurrentTransports = progress.userInfo[AFMDownloadProgressUserInfo.currentTransports] as? [String] ?? []
        if let previous = lastDownloadSample {
            let elapsed = now.timeIntervalSince(previous.date)
            let delta = completed - previous.completed
            if elapsed >= 0.1, delta >= 0 {
                let instantaneous = Double(delta) / elapsed
                if instantaneous > 0 {
                    downloadBytesPerSecond = downloadBytesPerSecond.map {
                        ($0 * 0.7) + (instantaneous * 0.3)
                    } ?? instantaneous
                    if total > completed, let speed = downloadBytesPerSecond, speed > 0 {
                        downloadETA = Double(total - completed) / speed
                    } else {
                        downloadETA = nil
                    }
                }
            }
        }
        lastDownloadSample = (now, completed)
        lock.unlock()
    }

    private func updateStage(_ stage: MLXLoadReporterStage) {
        lock.lock()
        self.stage = stage
        if stage == .loadingModel || stage == .ready {
            downloadFraction = nil
            downloadCompletedBytes = nil
            downloadTotalBytes = nil
            downloadBytesPerSecond = nil
            downloadETA = nil
            downloadCurrentFiles = []
            downloadCompletedFiles = nil
            downloadTotalFiles = nil
            downloadCurrentTransports = []
            lastDownloadSample = nil
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
            format: "[%@] %@ | mem %.2f GB | %.1fs",
            success ? "done" : "fail",
            status,
            memory,
            elapsed
        )
        if let errorMessage, !errorMessage.isEmpty {
            line += " | \(errorMessage)"
        }
        Self.writeDiagnostic(Self.terminalSafeLine(line, clearExisting: true) + "\n")
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
        let completedBytes = self.downloadCompletedBytes
        let totalBytes = self.downloadTotalBytes
        let bytesPerSecond = self.downloadBytesPerSecond
        let eta = self.downloadETA
        let currentFiles = self.downloadCurrentFiles
        let completedFiles = self.downloadCompletedFiles
        let totalFiles = self.downloadTotalFiles
        let currentTransports = self.downloadCurrentTransports
        spinnerIndex = (spinnerIndex + 1) % spinnerFrames.count
        let spinner = spinnerFrames[spinnerIndex]
        let elapsed = Date().timeIntervalSince(startedAt)
        lock.unlock()

        let memory = Self.currentResidentMemoryGB()

        var line = String(
            format: "[%@] %@ | mem %.2f GB | %.1fs",
            spinner,
            stage.rawValue,
            memory,
            elapsed
        )
        if stage == .downloading {
            if let fraction = downloadFraction {
                line += " | \(Self.progressBar(fraction: fraction, width: 18))"
                line += String(format: " %5.1f%%", fraction * 100)
            }
            if let completedBytes, let totalBytes {
                line += " | \(Self.formatBytes(completedBytes))/\(Self.formatBytes(totalBytes))"
            }
            if let bytesPerSecond, bytesPerSecond > 0 {
                line += " | \(Self.formatBytes(Int64(bytesPerSecond)))/s"
            }
            if let eta, eta.isFinite, eta >= 0 {
                line += " | ETA \(Self.formatDuration(eta))"
            }
            if let completedFiles, let totalFiles {
                line += " | files \(completedFiles)/\(totalFiles)"
            }
            if let first = currentFiles.first {
                line += " | \(first)"
                if currentFiles.count > 1 { line += " (+\(currentFiles.count - 1))" }
            }
            let transports = Array(Set(currentTransports)).sorted()
            line += " | transport \(transports.isEmpty ? "auto" : transports.joined(separator: "+"))"
        }
        Self.writeDiagnostic(Self.terminalSafeLine(line, clearExisting: true))
    }

    private static func terminalSafeLine(_ line: String, clearExisting: Bool) -> String {
        guard isatty(STDERR_FILENO) != 0 else {
            return "\r\(line)"
        }

        var size = winsize()
        let width: Int
        if ioctl(STDERR_FILENO, UInt(TIOCGWINSZ), &size) == 0, size.ws_col > 1 {
            width = Int(size.ws_col) - 1
        } else {
            width = 119
        }
        let clipped: String
        if line.count <= width {
            clipped = line
        } else if width > 3 {
            clipped = String(line.prefix(width - 3)) + "..."
        } else {
            clipped = String(line.prefix(width))
        }
        let erase = clearExisting ? "\u{1B}[2K" : ""
        return "\r\(erase)\(clipped)"
    }

    private static func writeDiagnostic(_ text: String) {
        fputs(text, stderr)
        fflush(stderr)
    }

    private static func progressBar(fraction: Double, width: Int) -> String {
        let clamped = max(0.0, min(1.0, fraction))
        let filled = Int((clamped * Double(width)).rounded(.down))
        let bar = String(repeating: "#", count: filled) + String(repeating: "-", count: max(0, width - filled))
        return "[\(bar)]"
    }

    private static func formatBytes(_ bytes: Int64) -> String {
        let formatter = ByteCountFormatter()
        formatter.allowedUnits = [.useKB, .useMB, .useGB, .useTB]
        formatter.countStyle = .file
        formatter.includesUnit = true
        formatter.isAdaptive = true
        return formatter.string(fromByteCount: bytes)
    }

    private static func formatDuration(_ seconds: TimeInterval) -> String {
        let rounded = max(0, Int(seconds.rounded()))
        if rounded >= 3_600 {
            return String(format: "%dh%02dm", rounded / 3_600, (rounded % 3_600) / 60)
        }
        if rounded >= 60 {
            return String(format: "%dm%02ds", rounded / 60, rounded % 60)
        }
        return "\(rounded)s"
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
        debugLog("Starting single prompt mode with prompt: '\(prompt)'")
        debugLog("Temperature: \(temperature?.description ?? "nil"), Randomness: \(randomness ?? "nil")")

        let group = DispatchGroup()
        let result = SendableBox<Result<String, Error>?>(nil)

        group.enter()
        Task {
            do {
#if compiler(>=6.4)
                if #available(macOS 26.0, *) {
                    debugLog("macOS 26+ detected, initializing FoundationModelService...")
                    let foundationService = try await FoundationModelService(instructions: instructions, adapter: adapter, temperature: temperature, randomness: randomness, permissiveGuardrails: permissiveGuardrails)
                    debugLog("FoundationModelService initialized successfully")
                    let message = Message(role: "user", content: prompt)
                    debugLog("Generating response...")
                    let response: String
                    if let guidedJson = self.guidedJson {
                        let schema = try parseGuidedJsonSchema(guidedJson)
                        response = try await foundationService.generateGuidedResponse(for: [message], jsonSchema: schema, temperature: temperature, randomness: randomness)
                    } else {
                        response = try await foundationService.generateResponse(for: [message], temperature: temperature, randomness: randomness)
                    }
                    debugLog("Response generated successfully")
                    result.value = .success(response)
                } else {
                    debugLog("macOS 26+ not available")
                    result.value = .failure(FoundationModelError.notAvailable)
                }
#else
                result.value = .failure(AFMEngineError.foundationModelsUnavailable)
#endif
            } catch {
                debugLog("Error occurred: \(error)")
                result.value = .failure(error)
            }
            group.leave()
        }
        
        group.wait()
        
        switch result.value {
        case .success(let response):
            print(response)
        case .failure(let error):
#if compiler(>=6.4)
            if let foundationError = error as? FoundationModelError {
                print("Error: \(foundationError.localizedDescription)")
            } else {
                print("Error: \(error.localizedDescription)")
            }
#else
            print("Error: \(error.localizedDescription)")
#endif
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

private func makeTelegramConfiguration(
    rawBotToken: String?,
    rawAllowlist: String?,
    hostname: String,
    port: Int,
    modelID: String,
    instructions: String,
    verbose: Bool,
    replyFormat: TelegramReplyFormat,
    requirePrefix: String?
) throws -> TelegramConfiguration? {
    let token = rawBotToken?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
    let allowlist = rawAllowlist?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
    if token.isEmpty && allowlist.isEmpty {
        return nil
    }
    guard !token.isEmpty else {
        throw ValidationError("--telegram-bot-token is required when --telegram-allow is set")
    }
    guard !allowlist.isEmpty else {
        throw ValidationError("--telegram-allow is required when --telegram-bot-token is set")
    }

    let host: String
    switch hostname {
    case "0.0.0.0", "::", "[::]":
        host = "127.0.0.1"
    default:
        host = hostname
    }

    let normalizedPrefix = requirePrefix?.trimmingCharacters(in: .whitespacesAndNewlines)
    let effectivePrefix = normalizedPrefix.flatMap { $0.isEmpty ? nil : $0 }

    return TelegramConfiguration(
        botToken: token,
        allowedUserIDs: try TelegramConfiguration.parseAllowedUserIDs(allowlist),
        localBaseURL: "http://\(host):\(port)",
        modelID: modelID,
        instructions: instructions,
        verbose: verbose,
        pollIntervalSeconds: 2,
        replyFormat: replyFormat,
        requiredPrefix: effectivePrefix
    )
}
