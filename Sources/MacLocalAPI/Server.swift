import Vapor
import Foundation
import Compression

// Storage key for the continuation
struct ContinuationKey: StorageKey {
    typealias Value = CheckedContinuation<Void, Error>
}

// Middleware to handle payload too large errors with a user-friendly message
struct PayloadTooLargeMiddleware: AsyncMiddleware {
    func respond(to request: Request, chainingTo next: any AsyncResponder) async throws -> Response {
        do {
            return try await next.respond(to: request)
        } catch let abort as Abort where abort.status == .payloadTooLarge {
            // Return a JSON error response compatible with OpenAI format
            let errorResponse = OpenAIError(
                message: "Your conversation is too long. Please start a new conversation.",
                type: "payload_too_large"
            )
            let response = Response(status: .payloadTooLarge)
            response.headers.add(name: .contentType, value: "application/json")
            response.headers.add(name: .accessControlAllowOrigin, value: "*")
            try response.content.encode(errorResponse)
            return response
        }
    }
}

class Server {
    private let app: Application
    private let port: Int
    private let hostname: String
    private let verbose: Bool
    private let veryVerbose: Bool
    private let streamingEnabled: Bool
    private let instructions: String
    private let adapter: String?
    private let temperature: Double?
    private let randomness: String?
    private let permissiveGuardrails: Bool
    private let webuiEnabled: Bool
    private let webuiPath: String?
    private let gatewayEnabled: Bool
    private let prewarmEnabled: Bool
    private let mlxModelID: String?
    private let mlxModelService: MLXModelService?
    private let mlxRepetitionPenalty: Double?
    private let mlxTopP: Double?
    private let mlxMaxTokens: Int?
    private let mlxRawOutput: Bool

    init(port: Int, hostname: String, verbose: Bool, veryVerbose: Bool = false, streamingEnabled: Bool, instructions: String, adapter: String? = nil, temperature: Double? = nil, randomness: String? = nil, permissiveGuardrails: Bool = false, webuiEnabled: Bool = false, gatewayEnabled: Bool = false, prewarmEnabled: Bool = true, mlxModelID: String? = nil, mlxModelService: MLXModelService? = nil, mlxRepetitionPenalty: Double? = nil, mlxTopP: Double? = nil, mlxMaxTokens: Int? = nil, mlxRawOutput: Bool = false) async throws {
        self.port = port
        self.hostname = hostname
        self.verbose = verbose
        self.veryVerbose = veryVerbose
        self.streamingEnabled = streamingEnabled
        self.instructions = instructions
        self.adapter = adapter
        self.temperature = temperature
        self.randomness = randomness
        self.permissiveGuardrails = permissiveGuardrails
        self.webuiEnabled = webuiEnabled
        self.webuiPath = Server.findWebuiPath()
        self.gatewayEnabled = gatewayEnabled
        self.prewarmEnabled = prewarmEnabled
        self.mlxModelID = mlxModelID
        self.mlxModelService = mlxModelService
        self.mlxRepetitionPenalty = mlxRepetitionPenalty
        self.mlxTopP = mlxTopP
        self.mlxMaxTokens = mlxMaxTokens
        self.mlxRawOutput = mlxRawOutput

        // Create environment without command line arguments to prevent Vapor from parsing them
        var env = Environment(name: "development", arguments: ["afm"])
        try LoggingSystem.bootstrap(from: &env)

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

        // Add custom error middleware to handle payload too large errors
        app.middleware.use(PayloadTooLargeMiddleware())

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
            if let mlxModelID = self.mlxModelID {
                return ModelsResponse(
                    object: "list",
                    data: [
                        ModelInfo(
                            id: mlxModelID,
                            object: "model",
                            created: Int(Date().timeIntervalSince1970),
                            owned_by: "mlx",
                            loaded: true
                        )
                    ],
                    models: [
                        ModelDetails(
                            name: mlxModelID,
                            model: mlxModelID,
                            capabilities: ["chat", "completion", "vision"]
                        )
                    ]
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

        if let mlxModelID = mlxModelID, let mlxModelService = mlxModelService {
            let mlxController = MLXChatCompletionsController(
                streamingEnabled: streamingEnabled,
                modelID: mlxModelID,
                service: mlxModelService,
                temperature: temperature,
                topP: mlxTopP,
                maxTokens: mlxMaxTokens,
                repetitionPenalty: mlxRepetitionPenalty,
                veryVerbose: veryVerbose,
                rawOutput: mlxRawOutput
            )
            try app.register(collection: mlxController)
        } else {
            let chatController = ChatCompletionsController(
                streamingEnabled: streamingEnabled,
                instructions: instructions,
                adapter: adapter,
                temperature: temperature,
                randomness: randomness,
                permissiveGuardrails: permissiveGuardrails,
                veryVerbose: veryVerbose
            )
            try app.register(collection: chatController)
        }

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
                    modalities: Modalities(vision: true, audio: false),
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
            var hasVision = false
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
                modalities: Modalities(vision: hasVision, audio: false),
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

    /// Custom CSS/JS to inject into webui (branding + auto-select default model)
    private var customCSS: String { Self.customCSSTemplate.replacingOccurrences(of: "/*_IS_MLX_PLACEHOLDER*/false", with: mlxModelID != nil ? "true" : "false") }
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
                        badge.innerHTML = '<img src="data:image/png;base64,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" width="18" height="18" style="flex-shrink:0" />'
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
    
    func start() async throws {
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
        print("")
        print("  ℹ️  Requires macOS 26+ with Apple Intelligence")
        print("  💡 Press Ctrl+C to stop the server")
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

        // Open browser if webui is enabled
        if webuiEnabled && webuiPath != nil {
            let url = "http://\(hostname):\(port)"
            print("  🌐 Opening WebUI in browser: \(url)")
            print("")
            openBrowser(url: url)
        }

        // Wait indefinitely (until shutdown is called)
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            // Store continuation for later use in shutdown
            app.storage[ContinuationKey.self] = continuation
        }
    }
    
    func shutdown() {
        print("🛑 Shutting down server...")

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

        // Get the executable's absolute directory
        let executablePath = CommandLine.arguments[0]
        let executableURL: URL
        if executablePath.hasPrefix("/") {
            executableURL = URL(fileURLWithPath: executablePath)
        } else {
            executableURL = URL(fileURLWithPath: cwd).appendingPathComponent(executablePath)
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

    /// Open URL in default browser
    private func openBrowser(url: String) {
        let task = Process()
        task.executableURL = URL(fileURLWithPath: "/usr/bin/open")
        task.arguments = [url]
        try? task.run()
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

    init(id: String, object: String, created: Int, owned_by: String, loaded: Bool = true) {
        self.id = id
        self.object = object
        self.created = created
        self.owned_by = owned_by
        self.status = ModelStatus(value: loaded ? "loaded" : "unloaded")
    }
}

struct ModelStatus: Content {
    let value: String
}

struct HealthResponse: Content {
    let status: String
    let timestamp: Double
    let version: String
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
