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
        let telegramConfiguration = try makeTelegramConfiguration(
            rawBotToken: telegramBotToken,
            rawAllowlist: telegramAllow,
            hostname: hostname,
            port: port,
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
                let server = try await Server(port: port, hostname: hostname, verbose: verbose, veryVerbose: veryVerbose || vv, trace: vv, streamingEnabled: !noStreaming, instructions: instructions, adapter: adapter, temperature: temperature, randomness: randomness, permissiveGuardrails: permissiveGuardrails, stop: stop, webuiEnabled: webui, gatewayEnabled: gateway, prewarmEnabled: prewarmEnabled, telegramConfiguration: telegramConfiguration)
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
          --enable-prefix-caching / --no-enable-prefix-caching: KV cache reuse across requests
          --tool-call-parser: Override tool call format (afm_adaptive_xml, hermes, llama3_json, gemma, mistral, qwen3_xml)
          --fix-tool-args: Post-process tool call arg names to match original tool schema
          --enable-grammar-constraints: EBNF grammar-constrained decoding for tool calls (requires --tool-call-parser afm_adaptive_xml). Forces valid XML structure at generation time — prevents JSON-inside-XML and missing required parameters. Improves tool call success from ~60% to 100% on realistic workloads.
          --no-think: Disable thinking/reasoning (sets enable_thinking=false)
          --default-chat-template-kwargs: JSON object merged into chat template context
          --gpu-capture <path>: Capture Metal GPU trace to .gputrace file for Xcode analysis (auto-limits to 5 tokens)
          --gpu-trace <seconds>: Record Metal System Trace via xctrace for N seconds (lightweight per-kernel timing)
          --gpu-profile: Print per-request GPU profiling stats (device info, memory, bandwidth estimates)
          --openclaw-config: Print OpenClaw provider config JSON and exit
          --help-json: Print machine-readable JSON capability card for AI agents and exit
        sampling_parameters: [temperature, top_p, top_k, min_p, presence_penalty, repetition_penalty, seed, max_tokens, logprobs, top_logprobs]
        features: [streaming-sse, tool-calling, think-reasoning-extraction, stop-sequences, json-mode, json-schema, prompt-caching, vlm-image-input, kv-cache-quantization, grammar-constrained-decoding, openclaw-integration]
        api_compatibility: OpenAI Chat Completions API (https://platform.openai.com/docs/api-reference/chat/create)
        extra_request_fields:
          top_k: int (not in OpenAI spec)
          min_p: float (not in OpenAI spec)
          repetition_penalty: float (also accepts repeat_penalty, not in OpenAI spec)
          chat_template_kwargs: object e.g. {"enable_thinking": false} (AFM-specific)
        extra_response_fields:
          choices[].message.reasoning_content: Extracted <think> reasoning (AFM-specific)
          usage.prompt_tokens_details.cached_tokens: Prefix cache hit count (AFM-specific)
        notes:
          - frequency_penalty is parsed but silently ignored
          - developer role is mapped to system
          - max_completion_tokens is accepted alongside max_tokens
        supported_model_types: [llama, qwen2, qwen3, qwen3_moe, qwen3_5, qwen3_5_moe, gemma, gemma2, phi3, starcoder2, openelm, cohere2, deepseek_v3, glm4, glm4_moe, lfm2, lfm2_moe, nemotron_h, minimax_m2, kimi_k2]
        tool_calling:
          auto_detection: Tool call format is auto-detected from model_type in config.json. Most models work without --tool-call-parser.
          parser_overrides:
            afm_adaptive_xml: Adaptive XML parser with JSON-in-XML fallback, type coercion, and EBNF grammar-constrained decoding (with --enable-grammar-constraints). Best for Qwen3+ models. Recommended combo: --tool-call-parser afm_adaptive_xml --enable-grammar-constraints
            hermes: JSON format with Hermes chat template (Llama, Qwen, most models)
            llama3_json: JSON format with Llama-3 chat template
            mistral: JSON format with Mistral chat template
            qwen3_xml: XML function format with Qwen3-Coder chat template
            gemma: Gemma function call format (uses model's built-in template)
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
            - You need GGUF/llama.cpp format models (use Ollama or llama-server instead)
            - You are not on Apple Silicon (MLX is Apple-only)
          integration_pattern: Start server with `afm mlx -m <model>`, then point any OpenAI SDK client at http://127.0.0.1:9999/v1. Drop-in replacement for OpenAI API.
          limitations:
            - Single-sequence inference only (one request at a time, queued)
            - MLX-format models only (safetensors from Hugging Face, not GGUF)
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
    @Option(name: .long, help: "Quantize KV cache to this many bits (4 or 8) to reduce memory usage")
    var kvBits: Int?
    @Option(name: .long, help: "Prefill step size — number of prompt tokens processed per GPU pass (default: 2048)")
    var prefillStepSize: Int?
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
    var telegramFormat: TelegramReplyFormat = .markdown

    @Option(name: .long, help: "Require a specific prefix for Telegram messages, for example '/afm' (default: no prefix required)")
    var telegramRequirePrefix: String?

    @Option(name: .long, help: "Tool call parser override: afm_adaptive_xml, hermes, llama3_json, gemma, mistral, qwen3_xml. afm_adaptive_xml adds JSON-in-XML fallback, type coercion, and optional xgrammar EBNF constrained decoding for models that switch between XML and JSON formats. Recommended for Qwen3+ models.")
    var toolCallParser: String?

    @Flag(name: .long, help: "Post-process tool call argument names to match the original tool schema (fixes model renaming e.g. path→filePath)")
    var fixToolArgs: Bool = false

    @Option(name: .customLong("kv-eviction"), help: "KV cache eviction policy: streaming (StreamingLLM) or none (default)")
    var kvEviction: String?

    @Option(name: .long, help: "Default chat template kwargs as JSON (e.g. '{\"enable_thinking\": false}')")
    var defaultChatTemplateKwargs: String?

    @Flag(name: .long, help: "Enable radix tree prefix caching for KV cache reuse across requests")
    var enablePrefixCaching: Bool = false

    @Option(name: .long, help: "Write cache timing profile records as JSONL to this file")
    var cacheProfilePath: String?

    @Flag(name: .long, help: "Enable EBNF grammar-constrained decoding for tool calls (requires --tool-call-parser afm_adaptive_xml). Forces valid XML tool call structure at generation time, preventing JSON-inside-XML and missing parameters.")
    var enableGrammarConstraints: Bool = false

    @Flag(name: .long, help: "Disable thinking/reasoning (sets enable_thinking=false in chat template)")
    var noThink: Bool = false

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

    @Flag(name: .long, help: "Print machine-readable JSON capability card for AI agents and exit")
    var helpJson: Bool = false

    func run() throws {
        if helpJson {
            printHelpJson(command: "afm mlx")
            return
        }

        if gateway {
            print("Error: -g/--gateway is not supported in 'afm mlx' mode.")
            throw ExitCode.failure
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

        if (telegramBotToken != nil || telegramAllow != nil) && (singlePrompt != nil || isatty(STDIN_FILENO) == 0) {
            print("Error: --telegram requires server mode and cannot be used with -s or piped single-prompt input")
            throw ExitCode.failure
        }

        emitCompatibilityWarnings()

        let resolver = MLXCacheResolver()
        let service = MLXModelService(resolver: resolver)
        service.toolCallParser = toolCallParser
        service.fixToolArgs = fixToolArgs
        service.forceVLM = vlm || !media.isEmpty
        service.kvBits = kvBits
        if let prefillStepSize { service.prefillStepSize = prefillStepSize }
        service.kvEvictionPolicy = kvEviction ?? "none"
        service.enablePrefixCaching = enablePrefixCaching
        service.cacheProfilePath = cacheProfilePath
        service.enableGrammarConstraints = enableGrammarConstraints
        // --concurrent N: 0 or 1 silently falls back to serial; nil = serial; 2+ = batch mode
        let maxConcurrent = concurrent ?? 0
        service.maxConcurrent = (maxConcurrent >= 2) ? maxConcurrent : 0
        service.trace = vv
        service.gpuCapturePath = gpuCapture
        service.gpuTraceDuration = gpuTrace
        service.gpuProfile = gpuProfile || gpuProfileBw
        service.gpuProfileBandwidth = gpuProfileBw

        // Parse --default-chat-template-kwargs and --no-think into defaultChatTemplateKwargs
        var parsedKwargs: [String: Any] = [:]
        if noThink {
            parsedKwargs["enable_thinking"] = false
        }
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
                    parsedKwargs[key] = value  // explicit kwargs override --no-think
                }
            } catch let error where !(error is ExitCode) {
                fputs("Error: Failed to parse --default-chat-template-kwargs as JSON: \(error)\n", stderr)
                throw ExitCode.failure
            }
        }
        if !parsedKwargs.isEmpty {
            service.defaultChatTemplateKwargs = parsedKwargs
        }

        if let guidedJson {
            _ = try parseGuidedJsonSchema(guidedJson)
        }

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

        // Read context window from model config
        var contextWindow: Int? = nil
        if let modelDir = resolver.localModelDirectory(repoId: selectedModel) {
            let configURL = modelDir.appendingPathComponent("config.json")
            if let data = try? Data(contentsOf: configURL),
               let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                // Check top-level first, then text_config (VLM models nest it)
                if let maxPos = json["max_position_embeddings"] as? Int {
                    contextWindow = maxPos
                } else if let textConfig = json["text_config"] as? [String: Any],
                          let maxPos = textConfig["max_position_embeddings"] as? Int {
                    contextWindow = maxPos
                }
            }
        }

        try ensureMLXMetalLibraryAvailable(verbose: verbose)

        // Resolve and validate --media paths early (before model load)
        var resolvedMedia: [String] = []
        let shellCWD = ProcessInfo.processInfo.environment["PWD"] ?? FileManager.default.currentDirectoryPath
        for path in media {
            let expanded = NSString(string: path).expandingTildeInPath
            let absPath = expanded.hasPrefix("/") ? expanded : shellCWD + "/" + expanded
            let resolved = URL(fileURLWithPath: absPath).standardized.path
            guard FileManager.default.fileExists(atPath: resolved) else {
                print("Error: Media file not found: \(path)")
                throw ExitCode.failure
            }
            resolvedMedia.append(resolved)
        }

        // Backward compatibility: support piped input in mlx mode too
        if let stdinContent = try readFromStdin() {
            try runSinglePrompt(stdinContent, service: service, modelID: selectedModel, mediaPaths: resolvedMedia)
            return
        }

        if let prompt = singlePrompt {
            try runSinglePrompt(prompt, service: service, modelID: selectedModel, mediaPaths: resolvedMedia)
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
            replyFormat: telegramFormat,
            requirePrefix: telegramRequirePrefix
        )

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
                // Initialize concurrent scheduler after model is loaded
                try await service.initScheduler()
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

    private func runSinglePrompt(_ prompt: String, service: MLXModelService, modelID: String, mediaPaths: [String] = []) throws {
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
        let stdoutFD = dup(STDOUT_FILENO)
        if stdoutFD == -1 || dup2(STDERR_FILENO, STDOUT_FILENO) == -1 {
            if stdoutFD != -1 { close(stdoutFD) }
            throw ValidationError("Failed to redirect single-prompt operational logs to stderr")
        }
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

                if mediaPaths.isEmpty {
                    messages.append(Message(role: "user", content: prompt))
                } else {
                    // Build multipart message with text + media references
                    var parts: [ContentPart] = [ContentPart(type: "text", text: prompt, image_url: nil)]
                    for path in mediaPaths {
                        let fileURL = URL(fileURLWithPath: path)
                        parts.append(ContentPart(type: "image_url", text: nil, image_url: ImageURL(url: fileURL.absoluteString, detail: nil)))
                    }
                    messages.append(Message(role: "user", content: .parts(parts)))
                }
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
        fflush(stdout)
        if dup2(stdoutFD, STDOUT_FILENO) == -1 {
            close(stdoutFD)
            throw ValidationError("Failed to restore stdout after single-prompt execution")
        }
        close(stdoutFD)

        switch output {
        case .success(let text):
            let rendered: String
            if raw {
                rendered = text
            } else {
                rendered = Self.stripThinkContent(
                    from: text,
                    startTag: service.thinkStartTag ?? "<think>",
                    endTag: service.thinkEndTag ?? "</think>"
                )
            }
            print(rendered)
        case .failure(let error):
            print("Error: \(error.localizedDescription)")
            throw ExitCode.failure
        case .none:
            throw ExitCode.failure
        }
    }

    private static func stripThinkContent(from text: String, startTag: String, endTag: String) -> String {
        var output = text
        while let start = output.range(of: startTag) {
            if let end = output.range(of: endTag, range: start.upperBound..<output.endIndex) {
                output.removeSubrange(start.lowerBound..<end.upperBound)
            } else {
                // Some guided/grammar-constrained generations can emit a stray
                // opening think tag without ever closing it. In that case keep
                // the payload and drop only the unmatched tag.
                output.removeSubrange(start)
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
        api_endpoints: [/v1/chat/completions, /v1/models, /health]
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
            - "afm -g" — API gateway proxying to local backends
        triggers:
          - start local LLM server
          - run MLX model locally
          - OpenAI-compatible local inference
          - Apple Foundation Models API
          - local tool calling server
          - vision OCR text extraction
          - API gateway for local LLM backends
        examples:
          - afm --port 9999
          - afm mlx -m Qwen/Qwen3-Coder-Next-4bit --port 9999
          - afm mlx -m mlx-community/Meta-Llama-3.1-8B-Instruct-4bit -s "Hello"
          - afm vision -f image.png
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
        api_endpoints: [/v1/chat/completions, /v1/models, /health]
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
            - "afm -g" — API gateway proxying to local backends
        triggers:
          - start local LLM server
          - run MLX model locally
          - OpenAI-compatible local inference
          - Apple Foundation Models API
          - local tool calling server
          - vision OCR text extraction
          - API gateway for local LLM backends
        examples:
          - afm --port 9999
          - afm mlx -m Qwen/Qwen3-Coder-Next-4bit --port 9999
          - afm mlx -m mlx-community/Meta-Llama-3.1-8B-Instruct-4bit -s "Hello"
          - afm vision -f image.png
          - afm -g --port 9999
        ---

        Use -w to enable the WebUI, -g to enable API gateway mode, or `afm mlx` for local MLX models.

        GitHub: https://github.com/scouzi1966/maclocal-api
        """,
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
            do {
                _ = try RandomnessConfig.parse(rand)
            } catch let error as FoundationModelError {
                throw ValidationError(error.localizedDescription)
            } catch {
                throw ValidationError("Invalid randomness parameter format")
            }
        }

        if (telegramBotToken != nil || telegramAllow != nil) && (singlePrompt != nil || isatty(STDIN_FILENO) == 0) {
            throw ValidationError("--telegram requires server mode and cannot be used with -s or piped single-prompt input")
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
        var args: [String] = ["--port", "\(port)", "--hostname", hostname, "--instructions", instructions, "--prewarm", prewarm]
        if verbose { args.append("--verbose") }
        if veryVerbose { args.append("--very-verbose") }
        if noStreaming { args.append("--no-streaming") }
        if permissiveGuardrails { args.append("--permissive-guardrails") }
        if webui { args.append("--webui") }
        if gateway { args.append("--gateway") }
        if let telegramBotToken { args += ["--telegram-bot-token", telegramBotToken] }
        if let telegramAllow { args += ["--telegram-allow", telegramAllow] }
        args += ["--telegram-format", telegramFormat.rawValue]
        if let telegramRequirePrefix { args += ["--telegram-require-prefix", telegramRequirePrefix] }
        if let adapter { args += ["--adapter", adapter] }
        if let temperature { args += ["--temperature", "\(temperature)"] }
        if let randomness { args += ["--randomness", randomness] }
        if let stop { args += ["--stop", stop] }
        var serveCommand = try ServeCommand.parse(args)
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
