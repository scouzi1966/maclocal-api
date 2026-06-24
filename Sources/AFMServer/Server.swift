import Vapor
import AFMKit
import Foundation
import Compression
import Darwin
#if canImport(AppKit)
import AppKit
#endif
import Logging
// Storage key for the continuation
public struct ContinuationKey: StorageKey {
    public typealias Value = CheckedContinuation<Void, Error>
}

/// Storage key for the per-request correlation ID (T1.1).
/// Set by `RequestIDMiddleware`, read by controllers and error emitters so the
/// same ID appears on the `X-Request-ID` / `OpenAI-Request-ID` response header,
/// inside `error.request_id`, and in server logs.
enum RequestIDKey: StorageKey {
    typealias Value = String
}

extension Request {
    /// The agent-correlatable request ID (T1.1). Always present after the
    /// `RequestIDMiddleware` has run.
    var afmRequestID: String {
        storage[RequestIDKey.self] ?? ""
    }
}

/// Mints or echoes a stable request ID for every HTTP request and copies it
/// to the response headers. Honors inbound `X-Request-ID` (most common) and
/// `OpenAI-Request-ID` (OpenAI SDK convention); otherwise mints `req_<uuid12>`
/// matching the format used by `BatchAPIController`. (T1.1)
struct RequestIDMiddleware: AsyncMiddleware {
    static let inboundHeaders = ["X-Request-ID", "OpenAI-Request-ID"]
    static let outboundHeaders = ["X-Request-ID", "OpenAI-Request-ID"]

    static func mint() -> String {
        "req_" + UUID().uuidString.lowercased().replacingOccurrences(of: "-", with: "").prefix(12)
    }

    func respond(to request: Request, chainingTo next: any AsyncResponder) async throws -> Response {
        let inbound = Self.inboundHeaders
            .compactMap { request.headers.first(name: $0)?.trimmingCharacters(in: .whitespaces) }
            .first(where: { !$0.isEmpty })
        let id = inbound ?? Self.mint()
        await request.storage.setWithAsyncShutdown(RequestIDKey.self, to: id)

        let response = try await next.respond(to: request)
        for name in Self.outboundHeaders {
            response.headers.replaceOrAdd(name: name, value: id)
        }
        return response
    }
}

/// Renders our typed errors (`TokenizeUnsupportedError`, `TokenizeBadRequestError`)
/// in OpenAI's `{"error": {"message", "type", "code", "request_id"}}` shape so
/// agents using the OpenAI SDK get a parsable error body. Falls through for
/// everything else so Vapor's default error middleware still handles it. (T1.6)
///
/// Both `type` and `code` are populated with the same machine-readable
/// identifier (`tokenize_unsupported`, `invalid_request_error`, …). OpenAI's
/// SDKs commonly switch on `code`, so leaving it nil makes those clients fall
/// through to a generic-error path even though `type` is meaningful.
struct OpenAIErrorRenderingMiddleware: AsyncMiddleware {
    func respond(to request: Request, chainingTo next: any AsyncResponder) async throws -> Response {
        do {
            return try await next.respond(to: request)
        } catch let err as TokenizeUnsupportedError {
            return try Self.render(
                request: request,
                status: err.status,
                message: err.reason,
                type: TokenizeUnsupportedError.errorType,
                code: TokenizeUnsupportedError.errorType,
                requestId: err.requestId
            )
        } catch let err as TokenizeBadRequestError {
            return try Self.render(
                request: request,
                status: err.status,
                message: err.reason,
                type: TokenizeBadRequestError.errorType,
                code: TokenizeBadRequestError.errorType,
                requestId: err.requestId
            )
        }
    }

    private static func render(request: Request, status: HTTPResponseStatus, message: String, type: String, code: String?, requestId: String) throws -> Response {
        let id = requestId.isEmpty ? request.afmRequestID : requestId
        let payload = OpenAIError(
            message: message,
            type: type,
            code: code,
            requestId: id.isEmpty ? nil : id
        )
        let response = Response(status: status)
        response.headers.add(name: .contentType, value: "application/json")
        response.headers.add(name: .accessControlAllowOrigin, value: "*")
        try response.content.encode(payload)
        return response
    }
}

// Middleware to handle payload too large errors with a user-friendly message
struct PayloadTooLargeMiddleware: AsyncMiddleware {
    func respond(to request: Request, chainingTo next: any AsyncResponder) async throws -> Response {
        do {
            return try await next.respond(to: request)
        } catch let abort as Abort where abort.status == .payloadTooLarge {
            // Return a JSON error response compatible with OpenAI format
            let reqId = request.afmRequestID
            let errorResponse = OpenAIError(
                message: "Your conversation is too long. Please start a new conversation.",
                type: "payload_too_large",
                requestId: reqId.isEmpty ? nil : reqId
            )
            let response = Response(status: .payloadTooLarge)
            response.headers.add(name: .contentType, value: "application/json")
            response.headers.add(name: .accessControlAllowOrigin, value: "*")
            try response.content.encode(errorResponse)
            return response
        }
    }
}

/// Counts active HTTP requests so `/metrics` can expose
/// `afm:num_active_connections`. Increments on entry, decrements on
/// exit (via defer-style task-local cleanup so the gauge always
/// returns to zero even on early throws). Filters out the /metrics
/// endpoint itself so a Prometheus scrape doesn't show up as a
/// connection — that would be self-referential noise on every poll.
///
/// Streaming endpoints (chat completions when `stream:true`) account
/// for themselves: their handler returns the Response object
/// immediately while the SSE body keeps writing for the duration of
/// the generation. If we counted them here, the gauge would
/// undercount — defer fires when `next.respond` returns, not when the
/// body finishes. Streaming controllers wrap the asyncStream body in
/// their own connectionStarted/connectionEnded bracket.
struct ActiveConnectionsMiddleware: AsyncMiddleware {
    static let nonStreamingExcluded: Set<String> = ["/metrics", "/health", "/healthz", "/openapi.json", "/docs"]
    static let streamingPaths: Set<String> = [
        "/v1/chat/completions",
        "/v1/batch/completions"
    ]

    static func shouldTrackInMiddleware(path: String) -> Bool {
        if nonStreamingExcluded.contains(path) { return false }
        // Filter the streaming chat path — its controller handles its own counting.
        if streamingPaths.contains(path) { return false }
        return true
    }

    func respond(to request: Request, chainingTo next: any AsyncResponder) async throws -> Response {
        let track = Self.shouldTrackInMiddleware(path: request.url.path)
        if track { StatsAggregator.shared.connectionStarted() }
        defer { if track { StatsAggregator.shared.connectionEnded() } }
        return try await next.respond(to: request)
    }
}

// @unchecked Sendable: the server owns a Vapor Application and assorted service
// references that aren't Sendable-audited. Lifecycle (start/shutdown) is driven
// from a single controlling flow and the closures it spawns only read immutable
// configuration or hop to @MainActor, so cross-task sharing is safe in practice.
public class Server: @unchecked Sendable {
    private let app: Application
    private let port: Int
    private let hostname: String
    private let verbose: Bool
    private let veryVerbose: Bool
    private let trace: Bool
    private let streamingEnabled: Bool
    private let instructions: String
    private let adapter: String?
    private let temperature: Double?
    private let randomness: String?
    private let permissiveGuardrails: Bool
    private let stop: String?
    private let webuiEnabled: Bool
    private let webuiPath: String?
    private let gatewayEnabled: Bool
    private let prewarmEnabled: Bool
    private let telegramConfiguration: TelegramConfiguration?
    private let defaultGuidedJsonSchema: ResponseFormat?
    private let mlxModelID: String?
    private let mlxModelService: MLXModelService?
    private let mlxRepetitionPenalty: Double?
    private let mlxTopP: Double?
    private let mlxMaxTokens: Int?
    private let mlxRawOutput: Bool
    private let mlxTopK: Int?
    private let mlxMinP: Double?
    private let mlxPresencePenalty: Double?
    private let mlxSeed: Int?
    private let mlxMaxLogprobs: Int
    private let contextWindow: Int?
    private var telegramBridge: TelegramBridge?

    private static let audioAvailable: Bool = {
        if #available(macOS 13.0, *) { return true }
        return false
    }()

    public init(port: Int, hostname: String, verbose: Bool, veryVerbose: Bool = false, trace: Bool = false, streamingEnabled: Bool, instructions: String, adapter: String? = nil, temperature: Double? = nil, randomness: String? = nil, permissiveGuardrails: Bool = false, stop: String? = nil, webuiEnabled: Bool = false, gatewayEnabled: Bool = false, prewarmEnabled: Bool = true, telegramConfiguration: TelegramConfiguration? = nil, defaultGuidedJsonSchema: ResponseFormat? = nil, mlxModelID: String? = nil, mlxModelService: MLXModelService? = nil, mlxRepetitionPenalty: Double? = nil, mlxTopP: Double? = nil, mlxMaxTokens: Int? = nil, mlxRawOutput: Bool = false, mlxTopK: Int? = nil, mlxMinP: Double? = nil, mlxPresencePenalty: Double? = nil, mlxSeed: Int? = nil, mlxMaxLogprobs: Int? = nil, contextWindow: Int? = nil) async throws {
        self.port = port
        self.hostname = hostname
        self.verbose = verbose
        self.veryVerbose = veryVerbose
        self.trace = trace
        self.streamingEnabled = streamingEnabled
        self.instructions = instructions
        self.adapter = adapter
        self.temperature = temperature
        self.randomness = randomness
        self.permissiveGuardrails = permissiveGuardrails
        self.stop = stop
        self.webuiEnabled = webuiEnabled
        self.webuiPath = Server.findWebuiPath()
        self.gatewayEnabled = gatewayEnabled
        self.prewarmEnabled = prewarmEnabled
        self.telegramConfiguration = telegramConfiguration
        self.defaultGuidedJsonSchema = defaultGuidedJsonSchema
        self.mlxModelID = mlxModelID
        self.mlxModelService = mlxModelService
        self.mlxRepetitionPenalty = mlxRepetitionPenalty
        self.mlxTopP = mlxTopP
        self.mlxMaxTokens = mlxMaxTokens
        self.mlxRawOutput = mlxRawOutput
        self.mlxTopK = mlxTopK
        self.mlxMinP = mlxMinP
        self.mlxPresencePenalty = mlxPresencePenalty
        self.mlxSeed = mlxSeed
        self.mlxMaxLogprobs = mlxMaxLogprobs ?? 20
        self.contextWindow = contextWindow

        // Create environment without command line arguments to prevent Vapor from parsing them
        var env = Environment(name: "development", arguments: ["afm"])
        LoggingSystem.bootstrap { label in
            CompactLogHandler(label: label)
        }

        self.app = try await Application.make(env)

        if veryVerbose {
            app.logger.logLevel = .trace
        } else if verbose {
            app.logger.logLevel = .debug
        }

        // Initialize backend discovery and proxy services (only in gateway mode)
        if gatewayEnabled {
            let discovery = BackendDiscoveryService(logger: app.logger, selfPort: port)
            let proxy = BackendProxyService(logger: app.logger)
            app.backendDiscovery = discovery
            app.backendProxy = proxy
        }

        try configure()
    }
    
    private func configure() throws {
        app.http.server.configuration.port = port
        app.http.server.configuration.hostname = hostname

        // Increase max body size for long conversations (default is 16KB)
        // 100MB should handle very long conversation histories
        app.routes.defaultMaxBodySize = "100mb"

        // Mint/echo X-Request-ID for every request — must run before other
        // middleware so ID is visible in error paths too. (T1.1)
        app.middleware.use(RequestIDMiddleware())

        // Render typed errors (TokenizeUnsupportedError, etc.) in OpenAI shape. (T1.6)
        app.middleware.use(OpenAIErrorRenderingMiddleware())

        // Add custom error middleware to handle payload too large errors
        app.middleware.use(PayloadTooLargeMiddleware())
        // Track concurrent client connections for /metrics' afm:num_active_connections gauge.
        app.middleware.use(ActiveConnectionsMiddleware())

        try routes()
    }
    
    private func routes() throws {
        app.get("health") { req async -> HealthResponse in
            return HealthResponse(
                status: "healthy",
                timestamp: Date().timeIntervalSince1970,
                version: "1.0.0"
            )
        }

        app.get("v1", "models") { req async -> ModelsResponse in
            // Apple NL embedding models are served on the unified endpoint (lazily
            // loaded on first /v1/embeddings). Advertise them so clients discover
            // embedding capability on the main server, not just on `afm embed`. (#132/#133)
            let embeddingCatalog = EmbeddingModelRegistry().shippedModels()
            let embeddingModelInfos = embeddingCatalog.map { m in
                ModelInfo(id: m.id, object: "model", created: m.createdEpoch, owned_by: "apple", loaded: false)
            }
            let embeddingDetails = embeddingCatalog.map { m in
                ModelDetails(name: "\(m.id) (Apple NL)", model: m.id, capabilities: ["embeddings"])
            }

            if let mlxModelID = self.mlxModelID {
                return ModelsResponse(
                    object: "list",
                    data: [
                        ModelInfo(
                            id: mlxModelID,
                            object: "model",
                            created: Int(Date().timeIntervalSince1970),
                            owned_by: "mlx",
                            loaded: true,
                            max_context_length: self.contextWindow
                        )
                    ] + embeddingModelInfos,
                    models: [
                        ModelDetails(
                            name: mlxModelID,
                            model: mlxModelID,
                            capabilities: ["chat", "completion", "vision"]
                        )
                    ] + embeddingDetails
                )
            }

            var models: [ModelInfo] = [
                ModelInfo(
                    id: "foundation",
                    object: "model",
                    created: Int(Date().timeIntervalSince1970),
                    owned_by: "apple",
                    loaded: true
                )
            ]
            var details: [ModelDetails] = [
                ModelDetails(name: "foundation (Apple)", model: "foundation", capabilities: ModelCapabilities.foundation.capabilities)
            ]

            if let discovery = req.application.backendDiscovery {
                // Rescan backends if stale so new models/backends appear quickly
                await discovery.refreshIfStale()
                let discovered = await discovery.allDiscoveredModels()
                for dm in discovered {
                    models.append(ModelInfo(
                        id: dm.id,
                        object: "model",
                        created: dm.created,
                        owned_by: dm.ownedBy,
                        loaded: dm.loaded
                    ))
                    // Use cached capabilities if available, otherwise nil (probed lazily via /props)
                    let caps = await discovery.capabilitiesForModel(dm.id)
                    details.append(ModelDetails(
                        name: "\(dm.id) (\(dm.backendName))",
                        model: dm.id,
                        capabilities: caps.capabilities
                    ))
                }
            }

            models += embeddingModelInfos
            details += embeddingDetails
            return ModelsResponse(object: "list", data: models, models: details)
        }

        // Stub /models/load and /models/unload for router mode compatibility
        // The webui calls these when switching models; we just acknowledge success
        // since our backends manage their own model loading
        app.on(.POST, "models", "load", body: .collect(maxSize: "1mb")) { req -> Response in
            // Parse the requested model from the body
            var modelName = "unknown"
            if let body = req.body.data {
                let bodyData = Data(buffer: body)
                if let json = try? JSONSerialization.jsonObject(with: bodyData) as? [String: Any],
                   let model = json["model"] as? String {
                    modelName = model
                }
            }
            req.logger.info("WebUI model load request: '\(modelName)'")

            let response = Response(status: .ok)
            response.headers.add(name: .contentType, value: "application/json")
            response.headers.add(name: .accessControlAllowOrigin, value: "*")
            // Echo back the model info so the webui confirms the switch
            let responseBody: [String: Any] = [
                "success": true,
                "model": modelName
            ]
            if let data = try? JSONSerialization.data(withJSONObject: responseBody) {
                response.body = .init(data: data)
            }
            return response
        }
        app.on(.POST, "models", "unload", body: .collect(maxSize: "1mb")) { req -> Response in
            let response = Response(status: .ok)
            response.headers.add(name: .contentType, value: "application/json")
            response.headers.add(name: .accessControlAllowOrigin, value: "*")
            try response.content.encode(["success": true])
            return response
        }

        try app.register(collection: VisionAPIController())
        try app.register(collection: SpeechAPIController())
        // POST /v1/embeddings on the main server (#132). The Apple NL embedding
        // model is loaded lazily on first request (a chat-only server pays
        // nothing until used) and this path never triggers MLX init. It does NOT
        // register /v1/models — the main server owns that route. `afm embed`
        // remains a standalone option.
        try app.register(collection: EmbeddingsController(
            resolver: LazyAppleEmbeddingResolver(),
            registersModelsRoute: false
        ))
        // POST /v1/chat/completions/{id}/cancel — agent cancel endpoint (T1.5).
        try app.register(collection: CancelController())
        // POST /v1/tokenize, /v1/count_tokens — agent token-budgeting endpoints (T1.6).
        try app.register(collection: TokenizeController(
            mlxModelID: mlxModelID,
            mlxModelService: mlxModelService,
            contextWindow: contextWindow
        ))
        // GET /openapi.json + /docs — schema discovery for self-configuring agents (T1.7).
        try app.register(collection: OpenAPIController())

        if let mlxModelID = mlxModelID, let mlxModelService = mlxModelService {
            let mlxController = MLXChatCompletionsController(
                streamingEnabled: streamingEnabled,
                modelID: mlxModelID,
                service: mlxModelService,
                temperature: temperature,
                topP: mlxTopP,
                maxTokens: mlxMaxTokens,
                repetitionPenalty: mlxRepetitionPenalty,
                topK: mlxTopK,
                minP: mlxMinP,
                presencePenalty: mlxPresencePenalty,
                seed: mlxSeed,
                maxLogprobs: mlxMaxLogprobs,
                veryVerbose: veryVerbose,
                trace: trace,
                rawOutput: mlxRawOutput,
                stop: stop
            )
            try app.register(collection: mlxController)

            // Batch API endpoints
            let batchStore = BatchStore()

            let batchAPIController = BatchAPIController(
                service: mlxModelService,
                store: batchStore,
                modelID: mlxModelID,
                temperature: temperature,
                topP: mlxTopP,
                maxTokens: mlxMaxTokens,
                repetitionPenalty: mlxRepetitionPenalty,
                topK: mlxTopK,
                minP: mlxMinP,
                presencePenalty: mlxPresencePenalty,
                seed: mlxSeed,
                maxLogprobs: mlxMaxLogprobs
            )
            try app.register(collection: batchAPIController)

            let batchCompletionsController = BatchCompletionsController(
                service: mlxModelService,
                modelID: mlxModelID,
                temperature: temperature,
                topP: mlxTopP,
                maxTokens: mlxMaxTokens,
                repetitionPenalty: mlxRepetitionPenalty,
                topK: mlxTopK,
                minP: mlxMinP,
                presencePenalty: mlxPresencePenalty,
                seed: mlxSeed,
                maxLogprobs: mlxMaxLogprobs
            )
            try app.register(collection: batchCompletionsController)

            // Seed the metrics aggregator with the live model id and the
            // configured concurrency so /metrics labels are correct from
            // the first scrape.
            StatsAggregator.shared.setModel(
                mlxModelID,
                maxConcurrent: mlxModelService.maxConcurrent
            )
        } else {
            let chatController = ChatCompletionsController(
                streamingEnabled: streamingEnabled,
                instructions: instructions,
                adapter: adapter,
                temperature: temperature,
                randomness: randomness,
                permissiveGuardrails: permissiveGuardrails,
                veryVerbose: veryVerbose,
                stop: stop,
                defaultGuidedJsonSchema: defaultGuidedJsonSchema
            )
            try app.register(collection: chatController)
        }

        // Prometheus metrics — always on, regardless of backend.
        // GET /metrics returns afm:* counters/gauges modelled after vLLM.
        try app.register(collection: MetricsController())

        // Props endpoint for llama.cpp webui compatibility (per-model capabilities)
        app.get("props") { [self] req async -> PropsResponse in
            if let mlxModelID = self.mlxModelID {
                return PropsResponse(
                    default_generation_settings: DefaultGenerationSettings(
                        n_ctx: 8192,
                        params: GenerationParams(
                            n_predict: -1,
                            temperature: self.temperature ?? 0.8,
                            top_k: 40,
                            top_p: 0.95,
                            min_p: 0.05,
                            stream: self.streamingEnabled,
                            max_tokens: 2000
                        )
                    ),
                    total_slots: 1,
                    model_path: mlxModelID,
                    role: "mlx",
                    modalities: Modalities(vision: true, audio: Self.audioAvailable),
                    chat_template: "",
                    bos_token: "",
                    eos_token: "",
                    build_info: "AFM \(BuildInfo.version ?? "dev")",
                    default_model: mlxModelID
                )
            }

            let modelParam = req.query[String.self, at: "model"]
            let isFoundation = modelParam == nil || modelParam == "foundation"

            var nCtx = 4096
            var hasVision = isFoundation
            var modelPath = "foundation"

            if !isFoundation, let modelName = modelParam {
                modelPath = modelName
                if let discovery = req.application.backendDiscovery {
                    let caps = await discovery.capabilitiesForModel(modelName)
                    hasVision = caps.vision
                    nCtx = caps.contextLength ?? 4096
                } else {
                    hasVision = false
                }
            }

            return PropsResponse(
                default_generation_settings: DefaultGenerationSettings(
                    n_ctx: nCtx,
                    params: GenerationParams(
                        n_predict: -1,
                        temperature: self.temperature ?? 0.8,
                        top_k: 40,
                        top_p: 0.95,
                        min_p: 0.05,
                        stream: self.streamingEnabled,
                        max_tokens: 2000
                    )
                ),
                total_slots: 1,
                model_path: modelPath,
                role: self.gatewayEnabled ? "router" : "model",
                modalities: Modalities(vision: hasVision, audio: Self.audioAvailable),
                chat_template: "",
                bos_token: "",
                eos_token: "",
                build_info: "AFM \(BuildInfo.version ?? "dev")",
                default_model: "foundation"
            )
        }

        // WebUI routes (if enabled and webui files exist)
        if webuiEnabled, let webuiFilePath = webuiPath {
            // Serve index.html with injected CSS for root path
            app.get { req -> Response in
                return try await self.serveWebuiWithCustomCSS(webuiFilePath: webuiFilePath, req: req)
            }

            // SPA fallback for non-API routes
            app.get("**") { req -> Response in
                let path = req.url.path

                // Don't intercept API routes
                if path.hasPrefix("/v1/") || path == "/health" || path == "/props" {
                    throw Abort(.notFound)
                }

                return try await self.serveWebuiWithCustomCSS(webuiFilePath: webuiFilePath, req: req)
            }
        }
    }

    /// Custom CSS/JS to inject into webui (branding + auto-select default model + /metrics dashboard)
    private var customCSS: String {
        Self.customCSSTemplate.replacingOccurrences(of: "/*_IS_MLX_PLACEHOLDER*/false", with: mlxModelID != nil ? "true" : "false")
        + Self.dashboardTemplate
    }
    private static let customCSSTemplate = """
    <style>
    /* Hide page until branding + model selection complete */
    body { opacity: 0 !important; }
    body.afm-ready { opacity: 1 !important; transition: opacity 0.15s ease-in; }
    /* Make model labels on response bubbles static (non-clickable) */
    .info [data-slot="popover-trigger"] { pointer-events: none; }
    .info [data-slot="popover-trigger"] svg { display: none; }
    </style>
    <script>
    (function(){
        var _isMLX = /*_IS_MLX_PLACEHOLDER*/false;
        console.log('[AFM] _isMLX =', _isMLX);
        var _aiGradient = 'linear-gradient(to right, #3b82f6, #a855f7, #ec4899, #f97316)';

        function rebrand(){
            document.querySelectorAll('h1,h2,h3,p,span').forEach(function(el){
                if(el.textContent==='llama.cpp' || el.textContent==='AFM'){
                    el.textContent='AFM';
                    if(!el.nextElementSibling?.classList?.contains('afm-sub')){
                        var sub = document.createElement('div');
                        sub.className = 'afm-sub';
                        sub.style.cssText = 'font-size:11px;color:#888;font-weight:normal;margin-top:4px;';
                        sub.textContent = 'llama.cpp webui';
                        el.parentElement.insertBefore(sub, el.nextSibling);
                    }
                    var existingBadge = el.parentElement.querySelector('.afm-ai-badge');
                    if(_isMultiModel && existingBadge){ existingBadge.remove(); existingBadge=null; }
                    if(!_isMultiModel && !existingBadge){
                        var badge = document.createElement('div');
                        badge.className = 'afm-ai-badge';
                        badge.style.cssText = 'display:inline-flex;align-items:center;gap:5px;margin-top:8px;font-size:13px;font-weight:600;';
                        if(_isMLX){
                            badge.innerHTML = '<span style="font-size:15px;flex-shrink:0">⚡</span>'
                                + '<span style="background:linear-gradient(to right, #f97316, #eab308);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">Swift MLX</span>';
                        } else {
                        badge.innerHTML = '<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADQAAAAyCAYAAAATIfj2AAAAAXNSR0IArs4c6QAAADhlWElmTU0AKgAAAAgAAYdpAAQAAAABAAAAGgAAAAAAAqACAAQAAAABAAAANKADAAQAAAABAAAAMgAAAADxNzqZAAAGSElEQVRoBe2aa4hVVRTHHU0nxTRNoZjRbCSnmDEle1l+iDJCKypIsyYCNchJLCoI6kNEQmhmhFBh5JcKi8KBHiSBNCMVZA/LsqnJGVLJsKxMranxNf3+t7su6+w5595zH8pILvi511577bXOPnefvfc544ABJ+XkHTiud6DqWGbr7e29nPiz4Ww4CO2wrqqqajvliSMMpAY2QJwcwvgUDD4hRsSFToCdUEjexqF/D4oLHApbgpFspL4UVkJX0PZMv/6VuNhl7oKPos/3F0x9CLzkfKRe6X36jc6FTYSDusKsrIi7ONo0qC/MifJrGBjnW4ot9SpH0rEkuBmmwRmgVWs3dMBmeADmgWQH1LOa9WRqwT/EugjTJ2D5m9E74QKogzHQC7vgA3iXWIcoyxeSV8HD0A1p5Vkca+KyYx8E50H4rGFKlB20zIyLF9rsDoX2XJ1Az1NZlDMUp+zD/RfYC1rRRsFZUA3FyhE6zOWXasnXMe+AGMxtdF7rAuxBfw26YCicC9PhfKiEvE+QLbAT9FxdDLfAKSA5AI0MSu3FCYPRVNsGJptQdIcjgu05c6D8Cd6EH50tTtXm2g6vB406VUSE9umw3/mtijikrRCgwQU5gj4x7IttEujiTBabD4aRcAnMhiaYC9fAZBji/N6jbqLnSr9MRLDdbw6U2yONaSt0vMEF2Rr2o02/YKvz0aaZu9DQP6lOnwtBe5aJVsuI0FBnjdmyz6AjHeIqdNSdNYkb0IPWmC3nxMVJY6P/yy7W3+gNvh/1c1z7YfSSBlTvgmhajbAk6DeBApvkXXmsX1JJkDGw24JRapmuNX90TVeTbWYvuiSCT7JAAbDNAt1FEy0A2gjLEmJcB37qfUt9nIJStoDJmpITEUEbpMnnKPPBLwJ7qWt3r4gQ6z7wopulgfrZMKvkZASa6qMH+j7qM0oOntCRmI8GeXxVU3FQQtd0ZgK0+ohZXfvNlHQRivcidjNoqwilzwpYVHSiTQHdFS+bqWTmdlHBinQmh6aaZoEX7Vkjiwz1nzsdrwK/Q1vgsla0tBdDsuHwuyV15VZ0nQfTCx2ugG4XxK8+mgp16aOV5kmOJS5/qGoFHJ0qMo5j4WcX4U/066HD2ZanClaGE7m+cfmeRl/h6lLXpwqP4xrX8S90naa1F9zr7Bpw8bt1qivI5NIZ0CQ3IzA8YsZsmf90gtMo6HGdFto1YBsNfj/QN7djIuR53F1Dm0+CvcW16W02WXDUymLS51egoc0aKZcmRyqvhdibXJ7IUo39UtemTT5yIA6njb4VmPzAi9RRq2TLVlePHCCdvSyVC9RLZ6ML0uZ0qXq5NNGLX+6MKWM4IL2RmugrTrgrd1gjZb3TK6lqjxvmAn7vdKk+rz6c6PU+XhiAnhN/Vot8S6DNP6x+8PEBS7AGOfQ9Iie0VcFbYLIx15ik4PmqeVNqcA9B5o5R6muNSU9SjHLsBJ9pCSj1aSsj6OPgDdcm9XZrTyxxqgW/D6mjTtUvwGJVnIRTMjFu2gZi3+jid6EvAn0H9yusXNZD3o88uZw46r1/OxSShThU5NRAnIGgKb0aCokGODx3wWkUdYAn4A8oJDpFrITJaWKbD/4ahLaKV+BXKCQ68jRZ/5JKAgyDebAW/Bss1Vj5CGver5y0Dwa9zIUneUwR0RlSh1EdfWZAuimWdqQE1IPppd1XAl2rUZ/XDGxXgz8X+m46pXQ6w5dpr60kPxLpY4aX06nUgB7cT31DVtdikvmAT1kNutP+5E41s5Kuo9RioGl+B5h8XNKFpu1EFk0Vf0ETfF/adCz5EEJ5EcNngVGHTi0AkXcb6n4lTXei9hdRrE7CXWBybVx/Gu+E8G3T+qjU34SmJvRd5RxXx/lU1Eayd1zCZUnB8amDuJVrA/ZT8/T7ysW/O8mvYnaSNbuE+9EnhcGx1YIuPE40ZZ+E6ph+2tdM5Dc+9Kl4nSSngb/ze6gvgQa4DJbDAfCiTfofb0DvhHugEfTsaaodBpPj8u0ic4PIOMeypii1f42AafBdCn+56Cb1WfIr/uv4gCRsAk25JNGmeWvQR5u0plv4a/kYGnSl/oDm0xfWSXwm6HVZu3k3aGXTs7MA8j3442l/DLTaaXC/gd6G74L+/Z8xCt+Wkx7/rzvwL0grYyNpW+UdAAAAAElFTkSuQmCC" width="18" height="18" style="flex-shrink:0" />'
                            + '<span style="background:'+_aiGradient+';-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">Apple Intelligence</span>';
                        }
                        el.parentElement.insertBefore(badge, el.nextElementSibling?.nextElementSibling || null);
                    }
                }
            });
            document.title=document.title.replace('llama.cpp','AFM');
        }

        var _autoSelectDone = false;
        var _userClickedModel = false;
        var _isMultiModel = false; // detected from /v1/models count

        // Auto-select "foundation" model in router mode if no model is selected
        function autoSelectFoundation(){
            var trigger = getModelTrigger();
            if(!trigger) { _autoSelectDone = true; return; }
            var txt = (trigger.textContent || '').trim();
            // Already selected
            if(txt && !txt.includes('Select model')){ _autoSelectDone = true; return; }
            // Open the dropdown
            _selectingModel = true;
            trigger.click();
            setTimeout(function(){
                // Find the "foundation" option in the listbox
                var options = document.querySelectorAll('[role="option"]');
                var found = false;
                for(var i=0;i<options.length;i++){
                    var label = (options[i].textContent || '').trim().toLowerCase();
                    if(label.indexOf('foundation') !== -1 || label.indexOf('apple') !== -1){
                        options[i].click();
                        found = true;
                        break;
                    }
                }
                // If only one option and not found by name, click the first one
                if(!found && options.length === 1){
                    options[0].click();
                }
                // Close dropdown if still open
                setTimeout(function(){
                    var trigger2 = getModelTrigger();
                    if(trigger2){
                        var listbox = document.querySelector('[role="listbox"]');
                        if(listbox){ trigger2.click(); }
                    }
                    _selectingModel = false;
                    _autoSelectDone = true;
                }, 150);
            }, 300);
        }

        function getModelTrigger(){
            var form = document.querySelector('[data-slot="chat-form"]');
            if(!form) return null;

            // Prefer the actual model picker button (popover/listbox trigger)
            var popoverButtons = form.querySelectorAll('button[aria-haspopup="listbox"],button[data-slot="popover-trigger"]');
            if(popoverButtons.length > 0) return popoverButtons[0];

            // Fallback: find a button that looks like model text, not action buttons
            var buttons = form.querySelectorAll('button');
            for(var i=0;i<buttons.length;i++){
                var txt = (buttons[i].textContent || '').trim();
                if(!txt) continue;
                if(txt.includes('Select model')) return buttons[i];
                if(txt !== 'Send' && txt !== 'Stop' && txt !== '+') return buttons[i];
            }
            return null;
        }

        // Listen for user clicks on model dropdown options to track user intent
        document.addEventListener('click', function(e){
            if(!_isMultiModel) return;
            var el = e.target;
            while(el && el !== document.body){
                if(el.getAttribute && el.getAttribute('role') === 'option'){
                    _userClickedModel = true;
                    setTimeout(function(){
                        var trigger = getModelTrigger();
                        if(trigger){
                            var model = trigger.textContent.trim();
                            if(model && !model.includes('Select model')){
                                localStorage.setItem('afm-preferred-model', model);
                            }
                        }
                        _userClickedModel = false;
                    }, 300);
                    return;
                }
                el = el.parentElement;
            }
        }, true);

        // Model info strip
        var _lastModel = '';
        var _selectingModel = false; // Prevent repeated auto-select during dropdown animation
        var _modelsCache = null;

        function fmtCtx(n){
            if(!n) return '';
            if(n>=1000) return Math.round(n/1024)+'K ctx';
            return n+' ctx';
        }

        function getOrCreateStrip(){
            var el = document.getElementById('afm-model-info');
            if(el) return el;
            var trigger = getModelTrigger();
            if(!trigger) return null;
            var parent = trigger.parentElement;
            if(!parent) return null;
            el = document.createElement('div');
            el.id = 'afm-model-info';
            el.style.cssText = 'font-size:11px;color:#888;padding:2px 8px 0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:320px;';
            parent.insertBefore(el, trigger.nextSibling);
            return el;
        }

        function updateInfoStrip(){
            if(_selectingModel) return; // Don't interfere while selecting
            var trigger = getModelTrigger();
            if(!trigger) return;
            var model = trigger.textContent.trim();
            if(!model || model.includes('Select model')) {
                var strip = document.getElementById('afm-model-info');
                if(strip) strip.textContent = '';
                _lastModel = '';
                return;
            }
            // Re-render if strip was destroyed by SPA re-render even if model unchanged
            var stripExists = document.getElementById('afm-model-info');
            if(model === _lastModel && stripExists) return;
            _lastModel = model;

            // Fetch model details from /v1/models (cached) and /props
            var p1 = _modelsCache ? Promise.resolve(_modelsCache) : fetch('/v1/models').then(function(r){return r.json()}).then(function(d){_modelsCache=d;return d});
            var p2 = fetch('/props?model='+encodeURIComponent(model)).then(function(r){return r.json()});

            Promise.all([p1,p2]).then(function(res){
                var modelsData = res[0];
                var props = res[1];
                if(_lastModel !== model) return; // stale

                var backend = '';
                var hasTools = false;
                if(modelsData && modelsData.models){
                    for(var i=0;i<modelsData.models.length;i++){
                        var m = modelsData.models[i];
                        if(m.model === model){
                            // Extract backend from name like "model (Backend)"
                            var match = m.name && m.name.match(new RegExp('\\\\(([^)]+)\\\\)$'));
                            if(match) backend = match[1];
                            if(m.capabilities && m.capabilities.indexOf('tools')!==-1) hasTools=true;
                            break;
                        }
                    }
                }

                var hasVision = props.modalities && props.modalities.vision;
                var nCtx = props.default_generation_settings && props.default_generation_settings.n_ctx;

                var parts = [];
                if(backend) parts.push(backend);
                if(hasVision) parts.push('Vision');
                if(hasTools) parts.push('Tools');
                var ctx = fmtCtx(nCtx);
                if(ctx) parts.push(ctx);

                var strip = getOrCreateStrip();
                if(strip) strip.textContent = parts.join(' \\u00b7 ');
            }).catch(function(){});
        }

        function refreshModelList(){
            // Invalidate models cache so info strip picks up new backends
            _modelsCache = null;
        }

        function waitForSpaAndReveal(){
            // Wait for the SPA to render AND model auto-select to finish before revealing
            var attempts = 0;
            var check = setInterval(function(){
                attempts++;
                var h1 = document.querySelector('h1');
                var spaReady = h1 || attempts > 50;
                var selectReady = _autoSelectDone || attempts > 100; // max ~5s wait for select
                if(spaReady && selectReady){
                    clearInterval(check);
                    rebrand();
                    document.body.classList.add('afm-ready');
                }
            }, 50);
        }

        function init(){
            waitForSpaAndReveal();
            // Discover if gateway mode has multiple models, then auto-select foundation.
            fetch('/v1/models').then(function(r){return r.json()}).then(function(d){
                var count = d && d.data ? d.data.length : 0;
                _isMultiModel = count > 1;
                // In router mode, auto-select foundation after SPA renders
                if(_isMultiModel){
                    // Wait for the SPA model list to populate, then auto-select
                    var selectAttempts = 0;
                    var selectInterval = setInterval(function(){
                        selectAttempts++;
                        var trigger = getModelTrigger();
                        if(trigger || selectAttempts > 40){
                            clearInterval(selectInterval);
                            autoSelectFoundation();
                        }
                    }, 100);
                } else {
                    _autoSelectDone = true;
                }
            }).catch(function(){ _autoSelectDone = true; });

            // Update branding/info on real DOM changes rather than polling.
            var refreshTimer = null;
            function scheduleRefresh(){
                if(refreshTimer) clearTimeout(refreshTimer);
                refreshTimer = setTimeout(function(){
                    rebrand();
                    updateInfoStrip();
                }, 120);
            }
            scheduleRefresh();
            var observer = new MutationObserver(scheduleRefresh);
            observer.observe(document.documentElement, { childList: true, subtree: true });

            // Periodically check for new models from background port scanning
            setInterval(refreshModelList, 15000);
            setTimeout(refreshModelList, 5000);
        }

        if(document.readyState==='loading'){
            document.addEventListener('DOMContentLoaded', init);
        } else {
            init();
        }
    })();
    </script>
    """

    /// Live `/metrics` dashboard injected alongside the webui customCSS.
    /// Renders a slide-out panel from the right edge of the page that polls
    /// `GET /metrics` every 1s, parses the Prometheus text exposition output,
    /// and renders every `afm:*` series with p50/p95/p99 derived from
    /// cumulative bucket counts. Toggle button is fixed to the right edge.
    private static let dashboardTemplate = """
    <style>
      .afm-dash-toggle {
        position: fixed; right: 0; top: 50%; transform: translateY(-50%);
        z-index: 999998;
        width: 32px; height: 80px;
        border: none; background: rgba(15, 23, 42, 0.85); color: #e2e8f0;
        border-radius: 8px 0 0 8px;
        cursor: pointer; font-size: 18px;
        display: flex; align-items: center; justify-content: center;
        box-shadow: -2px 0 8px rgba(0,0,0,0.2);
        transition: background 0.15s;
      }
      .afm-dash-toggle:hover { background: rgba(15, 23, 42, 0.95); }
      /* Non-modal: no backdrop so the rest of the page stays interactive
         (chat input, model picker, etc.) while the dashboard is open. */
      .afm-dash {
        position: fixed; top: 0; right: 0; bottom: 0;
        width: min(560px, 50vw);
        z-index: 999999;
        background: #0f172a; color: #e2e8f0;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        font-size: 13px; line-height: 1.5;
        transform: translateX(100%); transition: transform 0.25s ease-out;
        overflow-y: auto;
        box-shadow: -8px 0 32px rgba(0,0,0,0.4);
      }
      .afm-dash.open { transform: translateX(0); }
      .afm-dash header {
        position: sticky; top: 0; z-index: 1;
        background: linear-gradient(to right, #1e293b, #0f172a);
        padding: 14px 20px; border-bottom: 1px solid #334155;
        display: flex; align-items: baseline; gap: 14px;
      }
      .afm-dash header h1 {
        margin: 0; font-size: 16px; font-weight: 600;
        background: linear-gradient(to right, #3b82f6, #a855f7, #ec4899);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text;
      }
      .afm-dash header .afm-dash-meta { color: #94a3b8; font-size: 11px; }
      .afm-dash header .afm-dash-meta b { color: #cbd5e1; font-weight: 600; }
      .afm-dash header .afm-dash-close {
        margin-left: auto; background: none; border: 1px solid #334155;
        color: #cbd5e1; padding: 4px 10px; border-radius: 4px; cursor: pointer;
        font-size: 12px;
      }
      .afm-dash header .afm-dash-close:hover { background: #1e293b; }
      .afm-dash section { padding: 16px 20px; border-bottom: 1px solid #1e293b; }
      .afm-dash section h2 {
        margin: 0 0 10px; font-size: 11px; font-weight: 700;
        color: #64748b; text-transform: uppercase; letter-spacing: 1.2px;
      }
      .afm-dash .grid {
        display: grid; gap: 8px;
        grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      }
      .afm-tile {
        background: #1e293b; border: 1px solid #334155; border-radius: 6px;
        padding: 10px 12px;
      }
      .afm-tile .lbl { color: #64748b; font-size: 10px; text-transform: uppercase; letter-spacing: 1px; }
      .afm-tile .val { color: #f1f5f9; font-size: 22px; font-weight: 600; font-variant-numeric: tabular-nums; margin-top: 4px; }
      .afm-tile .sub { color: #64748b; font-size: 10px; margin-top: 2px; font-variant-numeric: tabular-nums; }
      .afm-tile.accent .val { color: #60a5fa; }
      .afm-tile.live .val { color: #34d399; }
      .afm-bar {
        height: 4px; background: #334155; border-radius: 2px;
        margin-top: 6px; overflow: hidden;
      }
      .afm-bar > div { height: 100%; background: linear-gradient(to right, #3b82f6, #a855f7); transition: width 0.3s; }
      .afm-hist {
        background: #1e293b; border: 1px solid #334155; border-radius: 6px;
        padding: 10px 12px;
      }
      .afm-hist .row1 {
        display: flex; align-items: baseline; gap: 12px;
        font-variant-numeric: tabular-nums;
      }
      .afm-hist .name { color: #cbd5e1; font-size: 12px; font-weight: 600; flex: 1; }
      .afm-hist .pcts { color: #94a3b8; font-size: 11px; }
      .afm-hist .pcts b { color: #f1f5f9; }
      .afm-hist .summary { color: #64748b; font-size: 10px; margin-top: 2px; font-variant-numeric: tabular-nums; }
      .afm-hist .bars {
        display: flex; align-items: flex-end; gap: 1px; height: 36px;
        margin-top: 6px;
      }
      .afm-hist .bars > span {
        flex: 1; background: #475569; border-radius: 1px 1px 0 0; min-height: 1px;
      }
      .afm-hist .bars > span.hot { background: linear-gradient(to top, #3b82f6, #60a5fa); }
      .afm-spark {
        height: 30px; display: flex; align-items: flex-end; gap: 1px;
        margin-top: 8px;
      }
      .afm-spark > span {
        flex: 1; background: #34d399; border-radius: 1px 1px 0 0; min-height: 1px;
        opacity: 0.5;
      }
      .afm-spark > span:last-child { opacity: 1; }
      .afm-reasons { display: grid; gap: 4px; }
      .afm-reasons > div {
        display: grid; grid-template-columns: 110px 1fr 60px;
        align-items: center; gap: 8px; font-size: 12px;
        font-variant-numeric: tabular-nums;
      }
      .afm-reasons .name { color: #cbd5e1; }
      .afm-reasons .bar { height: 8px; background: #334155; border-radius: 4px; overflow: hidden; }
      .afm-reasons .bar > div { height: 100%; background: #34d399; transition: width 0.3s; }
      .afm-reasons .v { color: #f1f5f9; text-align: right; }
      .afm-status { font-size: 11px; color: #64748b; }
      .afm-status.err { color: #f87171; }
      .afm-empty { color: #64748b; font-size: 11px; font-style: italic; }
    </style>
    <button class="afm-dash-toggle" id="afm-dash-toggle" title="Open AFM /metrics dashboard (Esc to close)">📊</button>
    <aside class="afm-dash" id="afm-dash" aria-hidden="true">
      <header>
        <h1>AFM /metrics</h1>
        <div class="afm-dash-meta">
          <span id="afm-dash-model">—</span> ·
          <span id="afm-dash-status" class="afm-status">connecting…</span>
        </div>
        <button class="afm-dash-close" id="afm-dash-close">Close</button>
      </header>

      <section>
        <h2>Live</h2>
        <div class="grid" id="afm-live-grid"></div>
        <div style="margin-top:10px; font-size:11px; color:#64748b;">Decode rate (last 60s):</div>
        <div class="afm-spark" id="afm-spark"></div>
      </section>

      <section>
        <h2>Counters</h2>
        <div class="grid" id="afm-counter-grid"></div>
      </section>

      <section>
        <h2>Finished reasons</h2>
        <div class="afm-reasons" id="afm-reasons"></div>
      </section>

      <section>
        <h2>Latency histograms</h2>
        <div class="grid" style="grid-template-columns: 1fr;" id="afm-latency-grid"></div>
      </section>

      <section>
        <h2>Size + sampling histograms</h2>
        <div class="grid" style="grid-template-columns: 1fr;" id="afm-size-grid"></div>
      </section>

      <section style="font-size:10px; color:#475569;">
        Last scrape: <span id="afm-dash-ts">—</span>. Polling interval: 1s.
      </section>
    </aside>
    <script>
    (function(){
      // ── Toggle plumbing ────────────────────────────────────────────
      var dash = document.getElementById('afm-dash');
      var toggle = document.getElementById('afm-dash-toggle');
      var closeBtn = document.getElementById('afm-dash-close');
      function openDash() {
        dash.classList.add('open');
        dash.setAttribute('aria-hidden', 'false');
        startPolling();
      }
      function closeDash() {
        dash.classList.remove('open');
        dash.setAttribute('aria-hidden', 'true');
      }
      toggle.addEventListener('click', function(){
        dash.classList.contains('open') ? closeDash() : openDash();
      });
      closeBtn.addEventListener('click', closeDash);
      // Esc closes only when focus is NOT inside the chat input
      // (so the user's escape-to-cancel-typing doesn't accidentally close the panel).
      document.addEventListener('keydown', function(e){
        if (e.key !== 'Escape' || !dash.classList.contains('open')) return;
        var t = e.target;
        if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable)) return;
        closeDash();
      });

      // ── Prometheus text parser ─────────────────────────────────────
      // Returns: { name: { type, help, samples: [{labels:{...}, value, le?}] } }
      function parsePrometheus(txt) {
        var out = {}, name;
        var lines = txt.split(/\\r?\\n/);
        for (var i = 0; i < lines.length; i++) {
          var l = lines[i];
          if (!l) continue;
          if (l.charCodeAt(0) === 35) {  // '#'
            var m = l.match(/^# (HELP|TYPE) (\\S+) (.*)$/);
            if (!m) continue;
            name = m[2];
            out[name] = out[name] || { type: '', help: '', samples: [] };
            if (m[1] === 'HELP') out[name].help = m[3];
            else                 out[name].type = m[3];
            continue;
          }
          // sample line: NAME{labels} VALUE   or   NAME VALUE
          var sp = l.indexOf(' ');
          if (sp < 0) continue;
          var head = l.slice(0, sp);
          var val = parseFloat(l.slice(sp + 1));
          var lb = head.indexOf('{');
          var key, labels = {};
          if (lb < 0) {
            key = head;
          } else {
            key = head.slice(0, lb);
            var rb = head.lastIndexOf('}');
            var inner = head.slice(lb + 1, rb);
            // crude label parser — handles escaped \\" inside values
            var re = /([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\\\]|\\\\.)*)"/g;
            var lm;
            while ((lm = re.exec(inner)) !== null) {
              labels[lm[1]] = lm[2].replace(/\\\\(.)/g, '$1');
            }
          }
          // For histogram bucket lines, key ends with "_bucket"
          var rec = out[key] || (out[key] = { type: 'untyped', help: '', samples: [] });
          rec.samples.push({ labels: labels, value: val });
        }
        return out;
      }

      // ── Histogram percentile from cumulative buckets ───────────────
      // Buckets are emitted in ascending order; values are cumulative.
      // Returns an interpolated quantile value, or null if no data.
      function percentile(buckets, total, p) {
        if (!buckets || !buckets.length || !total) return null;
        var target = total * p;
        var prevCount = 0, prevLe = 0;
        for (var i = 0; i < buckets.length; i++) {
          var b = buckets[i];
          if (b.count >= target) {
            if (b.le === Infinity) return prevLe;  // cap at last finite bucket
            if (b.count === prevCount) return b.le;
            // linear interp inside (prevLe, le]
            var frac = (target - prevCount) / (b.count - prevCount);
            return prevLe + (b.le - prevLe) * frac;
          }
          prevCount = b.count; prevLe = (b.le === Infinity ? prevLe : b.le);
        }
        return prevLe;
      }
      function histBuckets(rec) {
        // Convert bucket samples into [{le, count}], sorted by le ascending,
        // with +Inf last.
        if (!rec || !rec.samples) return [];
        return rec.samples.map(function(s){
          var leStr = s.labels.le;
          var le = leStr === '+Inf' ? Infinity : parseFloat(leStr);
          return { le: le, count: s.value };
        }).filter(function(b){ return !isNaN(b.le); })
          .sort(function(a,b){ return a.le - b.le; });
      }

      // ── Format helpers ─────────────────────────────────────────────
      function fmtN(v) {
        if (v == null || isNaN(v)) return '—';
        if (v >= 1e9) return (v/1e9).toFixed(2)+'B';
        if (v >= 1e6) return (v/1e6).toFixed(2)+'M';
        if (v >= 1e3) return (v/1e3).toFixed(1)+'k';
        return Math.round(v).toString();
      }
      function fmtSec(v) {
        if (v == null || isNaN(v) || !isFinite(v)) return '—';
        if (v < 0.001) return (v*1e6).toFixed(0)+'µs';
        if (v < 1) return (v*1000).toFixed(0)+'ms';
        if (v < 60) return v.toFixed(2)+'s';
        return (v/60).toFixed(1)+'m';
      }
      function fmtPct(v) {
        if (v == null || isNaN(v)) return '—';
        return (v*100).toFixed(1)+'%';
      }

      // ── Tile + histogram + reason renderers ────────────────────────
      function renderTiles(host, tiles) {
        // tiles: [{key, lbl, val, sub?, cls?, barPct?}]
        var existing = {};
        host.querySelectorAll('[data-key]').forEach(function(el){
          existing[el.getAttribute('data-key')] = el;
        });
        tiles.forEach(function(t){
          var el = existing[t.key];
          if (!el) {
            el = document.createElement('div');
            el.className = 'afm-tile' + (t.cls ? ' '+t.cls : '');
            el.setAttribute('data-key', t.key);
            el.innerHTML = '<div class="lbl"></div><div class="val"></div><div class="sub"></div><div class="afm-bar" style="display:none"><div></div></div>';
            host.appendChild(el);
          }
          el.className = 'afm-tile' + (t.cls ? ' '+t.cls : '');
          el.querySelector('.lbl').textContent = t.lbl;
          el.querySelector('.val').textContent = t.val;
          el.querySelector('.sub').textContent = t.sub || '';
          var bar = el.querySelector('.afm-bar');
          if (t.barPct != null) {
            bar.style.display = '';
            bar.firstChild.style.width = Math.max(0, Math.min(100, t.barPct*100))+'%';
          } else {
            bar.style.display = 'none';
          }
          delete existing[t.key];
        });
        Object.values(existing).forEach(function(el){ el.remove(); });
      }

      function renderHistogram(host, key, label, rec, fmt) {
        var bs = histBuckets(rec);
        var sumRec = (state.metrics[key.replace(/_bucket$/,'')+'_sum']) || null;
        var countRec = (state.metrics[key.replace(/_bucket$/,'')+'_count']) || null;
        var sum = sumRec ? sumRec.samples.reduce(function(a,s){return a+s.value;},0) : null;
        var total = countRec ? countRec.samples.reduce(function(a,s){return a+s.value;},0) : null;
        var p50 = percentile(bs, total, 0.5);
        var p95 = percentile(bs, total, 0.95);
        var p99 = percentile(bs, total, 0.99);

        var el = host.querySelector('[data-key="'+key+'"]');
        if (!el) {
          el = document.createElement('div');
          el.className = 'afm-hist';
          el.setAttribute('data-key', key);
          el.innerHTML = '<div class="row1"><div class="name"></div><div class="pcts"></div></div><div class="summary"></div><div class="bars"></div>';
          host.appendChild(el);
        }
        el.querySelector('.name').textContent = label;
        el.querySelector('.pcts').innerHTML =
          'p50 <b>'+fmt(p50)+'</b> · p95 <b>'+fmt(p95)+'</b> · p99 <b>'+fmt(p99)+'</b>';
        el.querySelector('.summary').textContent =
          'count '+fmtN(total)+' · sum '+(sum!=null?fmt(sum):'—')+
          ' · mean '+(total>0?fmt(sum/total):'—');

        // Per-bucket bar widths (relative to most populated finite bucket)
        var bars = el.querySelector('.bars');
        bars.innerHTML = '';
        if (!total) {
          bars.style.display = 'none';
        } else {
          bars.style.display = '';
          // Compute non-cumulative deltas for visual clarity
          var prevCount = 0;
          var deltas = [];
          for (var i = 0; i < bs.length; i++) {
            deltas.push(bs[i].count - prevCount);
            prevCount = bs[i].count;
          }
          var max = Math.max.apply(Math, deltas);
          var hotIdx = deltas.indexOf(max);
          deltas.forEach(function(d, i){
            var s = document.createElement('span');
            s.style.height = max > 0 ? Math.max(1, (d/max)*100)+'%' : '1%';
            s.title = 'le=' + (bs[i].le === Infinity ? '+Inf' : bs[i].le) + ': ' + d + ' obs';
            if (i === hotIdx) s.classList.add('hot');
            bars.appendChild(s);
          });
        }
      }

      function renderReasons(host, sumByReason) {
        host.innerHTML = '';
        var keys = Object.keys(sumByReason).sort();
        if (!keys.length || keys.every(function(k){ return sumByReason[k] === 0; })) {
          host.innerHTML = '<div class="afm-empty">No completed requests yet.</div>';
          return;
        }
        var max = Math.max.apply(Math, keys.map(function(k){ return sumByReason[k]; }));
        keys.forEach(function(k){
          var v = sumByReason[k];
          var row = document.createElement('div');
          row.innerHTML = '<div class="name"></div><div class="bar"><div></div></div><div class="v"></div>';
          row.querySelector('.name').textContent = k;
          row.querySelector('.bar > div').style.width = max > 0 ? (v/max)*100+'%' : '0';
          row.querySelector('.v').textContent = fmtN(v);
          host.appendChild(row);
        });
      }

      // ── Polling state ─────────────────────────────────────────────
      var state = {
        polling: false,
        timer: null,
        prev: null,        // previous gen-token snapshot for spark rate
        prevTs: null,
        spark: [],         // last 60 tok/s samples (instantaneous, for sparkline only)
        peakClient: 0,     // client-side high-water for inflight (serial mode fallback)
        // ── Per-completed-request tracking ─────────────────────────────
        // We detect a request finishing by `requests_completed_total`
        // ticking up between polls; at that moment the dashboard's
        // shown rate becomes the CURRENT request's actual decode rate
        // (Δgen / Δdecode_time_sum), matching what the chat UI shows
        // per message (model-reported decode tok/s, not wall clock).
        // The value sticks until another request completes.
        prevSnap: null,    // { compTot, genTot, decodeSum }
        lastReqTps: null,  // tok/s of the most recent completed request
        lastReqGen: null,  // tokens of the most recent completed request
        lastReqAt: null,   // wall-clock instant we observed the completion
        // ── Sticky live values ─────────────────────────────────────────
        // Whatever Active was last seen non-zero, keep it visible so the
        // panel reflects "what just happened" instead of snapping to 0.
        lastActive: 0,
        lastActiveAt: null,
        metrics: {}
      };

      function tick() {
        fetch('/metrics', { headers: { Accept: 'text/plain' } })
          .then(function(r){
            if (!r.ok) throw new Error('HTTP '+r.status);
            return r.text();
          })
          .then(function(txt){
            state.metrics = parsePrometheus(txt);
            paint();
            var statusEl = document.getElementById('afm-dash-status');
            statusEl.textContent = 'live'; statusEl.classList.remove('err');
            document.getElementById('afm-dash-ts').textContent = new Date().toLocaleTimeString();
          })
          .catch(function(err){
            var statusEl = document.getElementById('afm-dash-status');
            statusEl.textContent = 'error: '+err.message; statusEl.classList.add('err');
          });
      }
      function startPolling() {
        if (state.polling) return;
        state.polling = true;
        tick();
        state.timer = setInterval(tick, 1000);
      }
      // Note: we keep polling even when closed so the spark stays continuous.

      // ── Paint loop ────────────────────────────────────────────────
      function paint() {
        var m = state.metrics;
        function single(name) {
          var rec = m[name];
          if (!rec || !rec.samples || !rec.samples.length) return null;
          // For our schema each gauge/counter has a single sample.
          return rec.samples[0].value;
        }
        function modelName() {
          var rec = m['afm:max_concurrent_slots'];
          return rec && rec.samples[0] ? rec.samples[0].labels.model_name : '—';
        }

        document.getElementById('afm-dash-model').innerHTML = '<b>'+modelName()+'</b>';

        var runningRaw = single('afm:num_requests_running') || 0;
        var waiting = single('afm:num_requests_waiting') || 0;
        var peakRaw = single('afm:batch_size_peak') || 0;
        var slots   = single('afm:max_concurrent_slots') || 0;
        var gpu     = single('afm:gpu_cache_usage_perc');
        var genTot  = single('afm:generation_tokens_total') || 0;
        var promTot = single('afm:prompt_tokens_total') || 0;
        var startTot = single('afm:requests_started_total') || 0;
        var compTot = single('afm:requests_completed_total') || 0;
        var hits    = single('afm:radix_cache_hits_total') || 0;
        var misses  = single('afm:radix_cache_misses_total') || 0;
        // Decode time accumulator from histogram _sum line — drives the
        // accurate per-request tok/s computation below.
        var decodeSum = (function(){
          var rec = state.metrics['afm:request_decode_time_seconds_sum'];
          if (!rec || !rec.samples || !rec.samples.length) return 0;
          return rec.samples.reduce(function(a,s){return a + s.value;}, 0);
        })();

        // In serial mode (no BatchScheduler) the running/peak gauge readers
        // are never registered and the server-side values stay at 0. Fall
        // back to (started - completed) so the live panel still tells the
        // user whether anything is actively generating.
        var inflightDerived = Math.max(0, startTot - compTot);
        var running = runningRaw > 0 ? runningRaw : inflightDerived;
        if (running > state.peakClient) state.peakClient = running;
        var peak = Math.max(peakRaw, state.peakClient);
        var serialMode = (slots === 0);

        var now = Date.now();
        if (running > 0) { state.lastActive = running; state.lastActiveAt = now; }

        // Detect a request completing on this poll: requests_completed_total
        // ticked up. When it does, compute that request's TRUE decode tok/s
        // from the deltas of cumulative counters — this matches the chat
        // UI's per-message "X.XX t/s" (which is gen_tokens / decode_time
        // straight from the model). The value sticks in state.lastReqTps
        // until another request completes.
        if (state.prevSnap == null) {
          state.prevSnap = { compTot: compTot, genTot: genTot, decodeSum: decodeSum };
        } else if (compTot > state.prevSnap.compTot) {
          var dGen = genTot - state.prevSnap.genTot;
          var dDec = decodeSum - state.prevSnap.decodeSum;
          if (dDec > 0 && dGen > 0) {
            state.lastReqTps = dGen / dDec;
            state.lastReqGen = dGen;
            state.lastReqAt = now;
          }
          state.prevSnap = { compTot: compTot, genTot: genTot, decodeSum: decodeSum };
        }

        // Spark (visual only): instantaneous wall-clock delta of genTot.
        var tpsInst = null;
        if (state.prev != null && state.prevTs != null) {
          var dt = (now - state.prevTs) / 1000;
          if (dt > 0) tpsInst = (genTot - state.prev) / dt;
        }
        state.prev = genTot; state.prevTs = now;
        if (tpsInst != null) {
          state.spark.push(Math.max(0, tpsInst));
          if (state.spark.length > 60) state.spark.shift();
        }

        // Display priority for the Decode rate tile:
        //   1. Actively generating now → use decodeSum-based live rate
        //      (Δgen since the active request started / Δdecode if we
        //       can compute it; fall back to the spark window).
        //   2. Last completed request's rate (sticky) — matches the chat
        //      UI's per-message t/s number.
        //   3. Lifetime cumulative gen / decode_sum.
        //   4. 0.
        var displayTps = null, displaySub = '', displayLabel = 'Decode rate';
        if (running > 0) {
          // Live: prefer the spark's recent peak so a fast in-progress
          // burst shows realistic tok/s instead of being averaged down
          // against the 1s polling window.
          var window = state.spark.slice(-5);
          var nz = window.filter(function(v){ return v > 0; });
          if (nz.length) {
            displayTps = Math.max.apply(Math, nz);
            displaySub = 'generating';
          } else if (state.lastReqTps != null) {
            displayTps = state.lastReqTps;
            displaySub = 'generating (last: ' + state.lastReqTps.toFixed(1) + ' tok/s)';
          } else {
            displayTps = 0; displaySub = 'generating';
          }
        } else if (state.lastReqTps != null) {
          displayTps = state.lastReqTps;
          var ageS = state.lastReqAt ? Math.max(0, Math.round((now - state.lastReqAt) / 1000)) : 0;
          displaySub = 'last request · ' + (state.lastReqGen ? state.lastReqGen + ' tok' : '') +
                      (ageS > 0 ? ' · ' + ageS + 's ago' : '');
          displayLabel = 'Decode rate (last)';
        } else if (decodeSum > 0) {
          displayTps = genTot / decodeSum;
          displaySub = 'lifetime avg';
        } else {
          displayTps = 0;
          displaySub = 'idle';
        }

        // Active tile: sticky high-water + sticky last-non-zero count.
        var activeVal = String(running);
        var activeSub;
        if (running > 0) {
          activeSub = 'peak ' + peak;
        } else if (state.lastActiveAt != null) {
          var aS = Math.max(0, Math.round((now - state.lastActiveAt) / 1000));
          activeSub = 'last ' + state.lastActive + ' · ' + aS + 's ago · peak ' + peak;
        } else {
          activeSub = 'peak ' + peak;
        }

        // Sustained (wall-clock) throughput — mean of the spark's recent
        // non-empty window. This is what Grafana's Token Throughput panel
        // computes via rate(generation_tokens_total[5m]). Always lower
        // than `displayTps` because it includes idle gaps between
        // requests; both numbers are useful, they answer different
        // questions ("how fast does the model decode?" vs "how many
        // tokens is the system sustaining?").
        var sustained = null;
        if (state.spark.length > 0) {
          var sum = state.spark.reduce(function(a, b) { return a + b; }, 0);
          sustained = sum / state.spark.length;
        }

        // Live tiles
        var liveTiles = [
          { key: 'tps', lbl: displayLabel,
            val: displayTps == null ? '—' : displayTps.toFixed(1) + ' tok/s',
            sub: displaySub, cls: 'live' },
          { key: 'sustained', lbl: 'Sustained throughput',
            val: sustained == null ? '—' : sustained.toFixed(1) + ' tok/s',
            sub: 'wall-clock · last ' + state.spark.length + 's',
            cls: 'live' },
          { key: 'inflight', lbl: serialMode ? 'Active' : 'In-flight',
            val: activeVal, sub: activeSub, cls: 'accent',
            barPct: slots > 0 ? running/slots : (running > 0 ? 1 : 0) },
        ];
        var conn = single('afm:num_active_connections') || 0;
        var connPeak = single('afm:active_connections_peak') || 0;
        liveTiles.push({ key: 'conn', lbl: 'Connections',
          val: String(conn),
          sub: 'peak ' + connPeak,
          cls: 'accent' });
        if (!serialMode) {
          liveTiles.push({ key: 'queue', lbl: 'Queue depth',
            val: String(waiting), sub: 'cap ' + slots });
        }
        liveTiles.push({ key: 'gpu', lbl: 'GPU memory',
          val: gpu == null ? '—' : fmtPct(gpu),
          sub: gpu == null ? 'not exported' : 'of recommended VRAM',
          barPct: gpu });
        var radixFill = single('afm:radix_cache_fill_perc');
        liveTiles.push({ key: 'radix', lbl: 'Prefix cache fill',
          val: radixFill == null ? '—' : fmtPct(radixFill),
          sub: radixFill == null ? '--enable-prefix-caching off' : 'radix tree',
          barPct: radixFill });
        renderTiles(document.getElementById('afm-live-grid'), liveTiles);
        var sparkHost = document.getElementById('afm-spark');
        sparkHost.innerHTML = '';
        var max = Math.max.apply(Math, state.spark.length ? state.spark : [1]);
        state.spark.forEach(function(v){
          var s = document.createElement('span');
          s.style.height = max > 0 ? Math.max(1, (v/max)*100)+'%' : '1%';
          s.title = v.toFixed(1)+' tok/s';
          sparkHost.appendChild(s);
        });

        // Counter tiles
        var hitRate = (hits + misses) > 0 ? hits / (hits + misses) : null;
        renderTiles(document.getElementById('afm-counter-grid'), [
          { key: 'gen', lbl: 'Generation tokens', val: fmtN(genTot) },
          { key: 'prompt', lbl: 'Prompt tokens', val: fmtN(promTot) },
          { key: 'started', lbl: 'Requests started', val: fmtN(startTot) },
          { key: 'completed', lbl: 'Requests completed', val: fmtN(compTot), sub: startTot ? fmtPct(compTot/startTot) + ' of started' : '' },
          { key: 'hits', lbl: 'Radix cache hits', val: fmtN(hits) },
          { key: 'misses', lbl: 'Radix cache misses', val: fmtN(misses), sub: hitRate == null ? '' : 'hit rate ' + fmtPct(hitRate), barPct: hitRate },
        ]);

        // Reasons
        var reasonRec = m['afm:request_success_total'];
        var reasonMap = {};
        if (reasonRec && reasonRec.samples) {
          reasonRec.samples.forEach(function(s){
            var k = s.labels.finished_reason || 'unknown';
            reasonMap[k] = (reasonMap[k] || 0) + s.value;
          });
        }
        renderReasons(document.getElementById('afm-reasons'), reasonMap);

        // Latency histograms
        var latencyHost = document.getElementById('afm-latency-grid');
        var latencyHists = [
          ['afm:e2e_request_latency_seconds_bucket',     'End-to-end latency', fmtSec],
          ['afm:request_queue_time_seconds_bucket',      'Queue time',          fmtSec],
          ['afm:request_inference_time_seconds_bucket',  'Inference time',      fmtSec],
          ['afm:request_prefill_time_seconds_bucket',    'Prefill time',        fmtSec],
          ['afm:request_decode_time_seconds_bucket',     'Decode time',         fmtSec],
          ['afm:time_to_first_token_seconds_bucket',     'Time to first token', fmtSec],
          ['afm:time_per_output_token_seconds_bucket',   'Time per output token', fmtSec],
        ];
        latencyHists.forEach(function(h){
          renderHistogram(latencyHost, h[0], h[1], m[h[0]], h[2]);
        });

        // Size + sampling
        var sizeHost = document.getElementById('afm-size-grid');
        var sizeHists = [
          ['afm:request_prompt_tokens_bucket',     'Prompt tokens / request',     fmtN],
          ['afm:request_generation_tokens_bucket', 'Generation tokens / request', fmtN],
          ['afm:request_params_n_bucket',          'Sampling param n',            fmtN],
          ['afm:request_params_best_of_bucket',    'Sampling param best_of',      fmtN],
        ];
        sizeHists.forEach(function(h){
          renderHistogram(sizeHost, h[0], h[1], m[h[0]], h[2]);
        });
      }

      // Don't auto-poll — only when opened.
      // (User can still click the toggle to peek.)
    })();
    </script>
    """

    /// Serve the webui with custom CSS injected
    private func serveWebuiWithCustomCSS(webuiFilePath: String, req: Request) async throws -> Response {
        let fileURL = URL(fileURLWithPath: webuiFilePath)
        let compressedData = try Data(contentsOf: fileURL)

        // Decompress gzip data
        guard let decompressedData = try? Self.gunzip(compressedData),
              var htmlString = String(data: decompressedData, encoding: .utf8) else {
            // Fallback: serve compressed if decompression fails
            var headers = HTTPHeaders()
            headers.add(name: .contentType, value: "text/html; charset=utf-8")
            headers.add(name: .contentEncoding, value: "gzip")
            headers.add(name: "Cache-Control", value: "no-cache")
            return Response(status: .ok, headers: headers, body: .init(data: compressedData))
        }

        // Inject custom CSS before </head>
        if let headEndRange = htmlString.range(of: "</head>") {
            htmlString.insert(contentsOf: customCSS, at: headEndRange.lowerBound)
        }

        var headers = HTTPHeaders()
        headers.add(name: .contentType, value: "text/html; charset=utf-8")
        headers.add(name: "Cache-Control", value: "no-cache")

        return Response(status: .ok, headers: headers, body: .init(string: htmlString))
    }
    
    public func start() async throws {
        // Print ASCII art splash screen
        let version = BuildInfo.version ?? "dev-build"

        // ANSI color codes - Apple Intelligence inspired gradient
        let cyan = "\u{001B}[36m"
        let blue = "\u{001B}[34m"
        let magenta = "\u{001B}[35m"
        let brightCyan = "\u{001B}[96m"
        let brightBlue = "\u{001B}[94m"
        let brightMagenta = "\u{001B}[95m"
        let white = "\u{001B}[97m"
        let gray = "\u{001B}[90m"
        let reset = "\u{001B}[0m"
        let bold = "\u{001B}[1m"

        // Center the version string properly (box content width is 68 chars)
        let boxContentWidth = 68
        let versionTextPadding = (boxContentWidth - version.count) / 2
        let versionLeftPad = String(repeating: " ", count: versionTextPadding)
        let versionRightPad = String(repeating: " ", count: boxContentWidth - version.count - versionTextPadding)

        print("")
        print("  \(brightCyan)╔════════════════════════════════════════════════════════════════════╗\(reset)")
        print("  \(brightCyan)║\(reset)                                                                    \(brightCyan)║\(reset)")
        print("  \(brightCyan)║\(reset)                    \(brightMagenta)█████╗\(reset) \(brightBlue)███████╗\(reset)\(brightCyan)███╗   ███╗\(reset)                      \(brightCyan)║\(reset)")
        print("  \(brightCyan)║\(reset)                   \(brightMagenta)██╔══██╗\(reset)\(brightBlue)██╔════╝\(reset)\(brightCyan)████╗ ████║\(reset)                      \(brightCyan)║\(reset)")
        print("  \(brightCyan)║\(reset)                   \(brightMagenta)███████║\(reset)\(brightBlue)█████╗\(reset)  \(brightCyan)██╔████╔██║\(reset)                      \(brightCyan)║\(reset)")
        print("  \(brightCyan)║\(reset)                   \(brightMagenta)██╔══██║\(reset)\(brightBlue)██╔══╝\(reset)  \(brightCyan)██║╚██╔╝██║\(reset)                      \(brightCyan)║\(reset)")
        print("  \(brightCyan)║\(reset)                   \(brightMagenta)██║  ██║\(reset)\(brightBlue)██║\(reset)     \(brightCyan)██║ ╚═╝ ██║\(reset)                      \(brightCyan)║\(reset)")
        print("  \(brightCyan)║\(reset)                   \(gray)╚═╝  ╚═╝╚═╝     ╚═╝     ╚═╝\(reset)                      \(brightCyan)║\(reset)")
        print("  \(brightCyan)║\(reset)                                                                    \(brightCyan)║\(reset)")
        print("  \(brightCyan)║\(reset)           \(white)Apple Foundation Models - OpenAI Compatible API\(reset)          \(brightCyan)║\(reset)")
        print("  \(brightCyan)║\(reset)\(versionLeftPad)\(bold)\(brightBlue)\(version)\(reset)\(versionRightPad)\(brightCyan)║\(reset)")
        print("  \(brightCyan)║\(reset)                                                                    \(brightCyan)║\(reset)")
        print("  \(brightCyan)╚════════════════════════════════════════════════════════════════════╝\(reset)")
        print("")

        // Initialize the Foundation Model Service once at startup
        if #available(macOS 26.0, *) {
            try await FoundationModelService.initialize(instructions: instructions, adapter: adapter, temperature: temperature, randomness: randomness, permissiveGuardrails: permissiveGuardrails, prewarm: prewarmEnabled)
        }

        let repoURL = "https://github.com/scouzi1966/maclocal-api"
        let link = "\u{001B}]8;;\(repoURL)\u{001B}\\\(repoURL)\u{001B}]8;;\u{001B}\\"
        print("  \(gray)🚀 Server: http://\(hostname):\(port)\(reset)")
        print("  \(gray)📦 \(link)\(reset)")
        print("")
        print("  📡 Endpoints:")
        print("     • POST   /v1/chat/completions    - Chat completion (streaming supported)")
        print("     • GET    /v1/models              - List available models")
        print("     • GET    /health                 - Health check")
        print("")
        print("  ⚙️  Configuration:")
        print("     • Streaming:          \(streamingEnabled ? "✓ enabled" : "✗ disabled")")
        print("     • Prewarm:            \(prewarmEnabled ? "✓ enabled" : "✗ disabled")")
        if webuiEnabled {
            if webuiPath != nil {
                print("     • WebUI:              ✓ enabled (with image/PDF upload)")
            } else {
                print("     • WebUI:              ⚠️  enabled but not found (run 'make webui')")
            }
        }
        if let temp = temperature {
            print("     • Temperature:        \(String(format: "%.1f", temp))")
        }
        if let rand = randomness {
            print("     • Randomness:         \(rand)")
        }
        if permissiveGuardrails {
            print("     • Guardrails:         ⚠️  permissive mode")
        }
        if let adapterPath = adapter {
            print("     • Adapter:            \(adapterPath)")
        }
        if gatewayEnabled {
            print("     • Gateway:            ✓ enabled (multi-backend proxy)")
        }
        if veryVerbose {
            let red = "\u{001B}[38;5;196m"
            let pink = "\u{001B}[38;5;213m"
            let purple = "\u{001B}[38;5;135m"
            let teal = "\u{001B}[38;5;43m"
            let orange = "\u{001B}[38;5;208m"
            print("  🎨 Log colors (-V):")
            print("     • \(red)Red\(reset)      User prompt")
            print("     • \(pink)Pink\(reset)     Full request JSON")
            print("     • \(purple)Purple\(reset)   Reasoning")
            print("     • \(teal)Teal\(reset)     Content / answer / usage")
            print("     • \(orange)Orange\(reset)   Start / done bookends")
        }
        print("")
        print("  ℹ️  Requires macOS 26+ with Apple Intelligence")
        print("  💡 Press Ctrl+C to stop the server")
        if let mlxModel = mlxModelID {
            print("  💡 OpenClaw:  afm mlx -m \(mlxModel) --openclaw-config")
        } else {
            print("  💡 OpenClaw:  afm mlx -m <model> --openclaw-config")
        }
        if gatewayEnabled {
            print("")
            let yellow = "\u{001B}[33m"
            print("  ⚠️  API Key for detected backends: \(yellow)\(afmAPIKey)\(reset)")
            print("     This is NOT a security measure and is considered unsafe and insecure.")
            print("     It is a shared passphrase for backends absolutely requiring API keys")
            print("     (e.g. Jan). Set this key in your backend's API")
            print("     key settings if it rejects requests.")
        }
        print("")
        print("  ─────────────────────────────────────────────────────────────────────────")
        print("")

        // Start backend discovery scanning (gateway mode only)
        if gatewayEnabled, let discovery = app.backendDiscovery {
            await discovery.startPeriodicScanning()

            let discovered = await discovery.allDiscoveredModels()
            if !discovered.isEmpty {
                print("  🔍 Discovered LLM Backends:")
                // Group by backend name
                var byBackend: [String: [String]] = [:]
                for model in discovered {
                    byBackend[model.backendName, default: []].append(model.id)
                }
                for (backend, modelIds) in byBackend.sorted(by: { $0.key < $1.key }) {
                    print("     • \(backend): \(modelIds.count) model(s)")
                    for id in modelIds.prefix(5) {
                        print("       - \(id)")
                    }
                    if modelIds.count > 5 {
                        print("       ... and \(modelIds.count - 5) more")
                    }
                }
                print("")
            }
        }

        // Start the server
        try await app.server.start(address: .hostname(hostname, port: port))

        if let telegramConfiguration {
            let bridge = try TelegramBridge(config: telegramConfiguration)
            try await bridge.start()
            self.telegramBridge = bridge
        }

        // Open browser if webui is enabled
        if webuiEnabled && webuiPath != nil {
            let url = "http://\(browserLaunchHost(for: hostname)):\(port)"
            print("  🌐 Opening WebUI in browser: \(url)")
            print("")
            Task { @MainActor in
                self.openBrowser(url: url)
            }
        }

        // Wait indefinitely (until shutdown is called)
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            // Store continuation for later use in shutdown
            app.storage[ContinuationKey.self] = continuation
        }
    }
    
    public func shutdown() {
        print("🛑 Shutting down server...")
        telegramBridge?.stop()
        telegramBridge = nil

        // Shutdown the server first
        Task {
            // Stop backend discovery
            if let discovery = app.backendDiscovery {
                await discovery.stopScanning()
            }

            await app.server.shutdown()

            if let mlxService = mlxModelService {
                await mlxService.shutdownAndReleaseResources(verbose: verbose)
            }

            print("Server shutdown complete")

            // Resume the continuation to exit the wait
            if let continuation = app.storage[ContinuationKey.self] {
                continuation.resume()
                app.storage[ContinuationKey.self] = nil
            }
        }
    }

    /// Find the webui index.html.gz file
    private static func findWebuiPath() -> String? {
        let fileManager = FileManager.default
        let cwd = fileManager.currentDirectoryPath

        // Get the executable's absolute directory. argv[0] is unreliable when invoked
        // via PATH (it is just "afm"), so go through the Mach-O loader.
        var size: UInt32 = 0
        _ = _NSGetExecutablePath(nil, &size)
        let executableURL: URL
        if size > 0 {
            var buffer = [CChar](repeating: 0, count: Int(size))
            if _NSGetExecutablePath(&buffer, &size) == 0 {
                executableURL = URL(fileURLWithPath: String(cString: buffer)).resolvingSymlinksInPath()
            } else if let bundleExec = Bundle.main.executableURL {
                executableURL = bundleExec.resolvingSymlinksInPath()
            } else {
                executableURL = URL(fileURLWithPath: cwd).appendingPathComponent(CommandLine.arguments[0])
            }
        } else if let bundleExec = Bundle.main.executableURL {
            executableURL = bundleExec.resolvingSymlinksInPath()
        } else {
            executableURL = URL(fileURLWithPath: cwd).appendingPathComponent(CommandLine.arguments[0])
        }
        let executableDir = executableURL.deletingLastPathComponent().standardized.path

        // Paths to check (in order of priority)
        let pathsToCheck = [
            // Bundled with executable (portable distribution)
            "\(executableDir)/Resources/webui/index.html.gz",
            // One level up from executable
            "\(executableDir)/../Resources/webui/index.html.gz",
            // Two levels up (e.g., .build/release -> .build -> project root)
            "\(executableDir)/../../Resources/webui/index.html.gz",
            // Three levels up for deeper nesting
            "\(executableDir)/../../../Resources/webui/index.html.gz",
            // pip: webui bundled in macafm package (sibling share directory)
            "\(executableDir)/../share/webui/index.html.gz",
            // Homebrew: share directory relative to bin (Apple Silicon)
            "\(executableDir)/../share/afm/webui/index.html.gz",
            // Homebrew: share directory relative to bin (Intel)
            "/usr/local/share/afm/webui/index.html.gz",
            // Homebrew: Apple Silicon path
            "/opt/homebrew/share/afm/webui/index.html.gz",
            // Development: Resources folder in current working directory
            "\(cwd)/Resources/webui/index.html.gz",
            // Development: vendored llama.cpp webui relative to executable
            "\(executableDir)/../../../vendor/llama.cpp/tools/server/public/index.html.gz",
            // Development: llama.cpp submodule public folder
            "\(cwd)/vendor/llama.cpp/tools/server/public/index.html.gz"
        ]

        for path in pathsToCheck {
            let standardizedPath = URL(fileURLWithPath: path).standardized.path
            if fileManager.fileExists(atPath: standardizedPath) {
                return standardizedPath
            }
        }

        return nil
    }

    private func browserLaunchHost(for hostname: String) -> String {
        switch hostname {
        case "0.0.0.0", "::", "[::]":
            return "127.0.0.1"
        default:
            return hostname
        }
    }

    /// Open URL in default browser
    @MainActor
    private func openBrowser(url: String) {
        guard let targetURL = URL(string: url) else { return }

        #if canImport(AppKit)
        if NSWorkspace.shared.open(targetURL) {
            return
        }
        #endif

        if runBrowserOpenProcess(executable: "/usr/bin/open", arguments: [targetURL.absoluteString]) {
            return
        }

        if runBrowserOpenProcess(executable: "/usr/bin/osascript", arguments: ["-e", "open location \"\(targetURL.absoluteString)\""]) {
            return
        }

        print("  ⚠️  Failed to open WebUI automatically. Open this URL manually: \(targetURL.absoluteString)")
    }

    @MainActor
    private func runBrowserOpenProcess(executable: String, arguments: [String]) -> Bool {
        let task = Process()
        task.executableURL = URL(fileURLWithPath: executable)
        task.arguments = arguments
        task.standardOutput = nil
        task.standardError = nil
        do {
            try task.run()
            task.waitUntilExit()
            return task.terminationReason == .exit && task.terminationStatus == 0
        } catch {
            return false
        }
    }

    /// Decompress gzip data
    private static func gunzip(_ data: Data) throws -> Data {
        // Gzip has a header we need to skip (minimum 10 bytes)
        guard data.count > 10 else { throw GzipError.invalidData }

        // Check gzip magic number
        guard data[0] == 0x1f && data[1] == 0x8b else { throw GzipError.invalidData }

        // Skip gzip header (10 bytes minimum, more if there are extra fields)
        var headerSize = 10
        let flags = data[3]

        // Check for extra field (FEXTRA)
        if flags & 0x04 != 0 {
            guard data.count > headerSize + 2 else { throw GzipError.invalidData }
            let extraLen = Int(data[headerSize]) | (Int(data[headerSize + 1]) << 8)
            headerSize += 2 + extraLen
        }

        // Check for original filename (FNAME)
        if flags & 0x08 != 0 {
            while headerSize < data.count && data[headerSize] != 0 {
                headerSize += 1
            }
            headerSize += 1 // skip null terminator
        }

        // Check for comment (FCOMMENT)
        if flags & 0x10 != 0 {
            while headerSize < data.count && data[headerSize] != 0 {
                headerSize += 1
            }
            headerSize += 1 // skip null terminator
        }

        // Check for header CRC (FHCRC)
        if flags & 0x02 != 0 {
            headerSize += 2
        }

        guard headerSize < data.count - 8 else { throw GzipError.invalidData }

        // Extract compressed data (excluding 8-byte trailer: CRC32 + original size)
        let compressedData = data.subdata(in: headerSize..<(data.count - 8))

        // Decompress using zlib raw deflate
        let destinationBufferSize = 10 * 1024 * 1024 // 10MB max
        let destinationBuffer = UnsafeMutablePointer<UInt8>.allocate(capacity: destinationBufferSize)
        defer { destinationBuffer.deallocate() }

        let decompressedSize = compressedData.withUnsafeBytes { sourcePtr -> Int in
            guard let baseAddress = sourcePtr.baseAddress else { return 0 }
            return compression_decode_buffer(
                destinationBuffer,
                destinationBufferSize,
                baseAddress.assumingMemoryBound(to: UInt8.self),
                compressedData.count,
                nil,
                COMPRESSION_ZLIB
            )
        }

        guard decompressedSize > 0 else { throw GzipError.decompressionFailed }

        return Data(bytes: destinationBuffer, count: decompressedSize)
    }
}

enum GzipError: Error {
    case invalidData
    case decompressionFailed
}

struct ModelsResponse: Content {
    let object: String
    let data: [ModelInfo]
    let models: [ModelDetails]?
}

struct ModelDetails: Content {
    let name: String
    let model: String
    let capabilities: [String]?
}

struct ModelInfo: Content {
    let id: String
    let object: String
    let created: Int
    let owned_by: String
    let status: ModelStatus
    let max_context_length: Int?
    init(id: String, object: String, created: Int, owned_by: String, loaded: Bool = true, max_context_length: Int? = nil) {
        self.id = id
        self.object = object
        self.created = created
        self.owned_by = owned_by
        self.status = ModelStatus(value: loaded ? "loaded" : "unloaded")
        self.max_context_length = max_context_length
    }
}

struct ModelStatus: Content {
    let value: String
}

public struct HealthResponse: Content {
    public let status: String
    public let timestamp: Double
    public let version: String
    public init(status: String, timestamp: Double, version: String) {
        self.status = status; self.timestamp = timestamp; self.version = version
    }
}

// MARK: - Props Response (llama.cpp webui compatibility)

struct PropsResponse: Content {
    let default_generation_settings: DefaultGenerationSettings
    let total_slots: Int
    let model_path: String
    let role: String
    let modalities: Modalities
    let chat_template: String
    let bos_token: String
    let eos_token: String
    let build_info: String
    let default_model: String
}

struct DefaultGenerationSettings: Content {
    let n_ctx: Int
    let params: GenerationParams
}

struct GenerationParams: Content {
    let n_predict: Int
    let temperature: Double
    let top_k: Int
    let top_p: Double
    let min_p: Double
    let stream: Bool
    let max_tokens: Int
}

struct Modalities: Content {
    let vision: Bool
    let audio: Bool
}

// Compact log handler that prints "[INFO]" instead of Vapor's padded "[ INFO ]"
public struct CompactLogHandler: LogHandler {
    public var metadata: Logger.Metadata = [:]
    public var logLevel: Logger.Level = .info
    let label: String

    public init(label: String) {
        self.label = label
    }

    public subscript(metadataKey key: String) -> Logger.Metadata.Value? {
        get { metadata[key] }
        set { metadata[key] = newValue }
    }

    private static let timestampFormatter: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "yyyy-MM-dd HH:mm:ss.SSS"
        f.locale = Locale(identifier: "en_US_POSIX")
        return f
    }()

    public func log(level: Logger.Level, message: Logger.Message, metadata: Logger.Metadata?,
             source: String, file: String, function: String, line: UInt) {
        let ts = Self.timestampFormatter.string(from: Date())
        let levelStr = level.rawValue.uppercased()
        let metaStr = Self.formatMetadata(self.metadata, metadata)
        if metaStr.isEmpty {
            print("[\(ts)] [\(levelStr)] \(message)")
        } else {
            print("[\(ts)] [\(levelStr)] \(message) \(metaStr)")
        }
    }

    private static func formatMetadata(_ base: Logger.Metadata, _ extra: Logger.Metadata?) -> String {
        var merged = base
        if let extra { merged.merge(extra) { _, new in new } }
        guard !merged.isEmpty else { return "" }
        return merged.sorted(by: { $0.key < $1.key })
            .map { "[\($0.key): \($0.value)]" }
            .joined(separator: " ")
    }
}
