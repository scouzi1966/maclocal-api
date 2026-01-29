import Vapor
import Foundation
import Network

actor BackendDiscoveryService {
    private let logger: Logger
    private let selfPort: Int
    private var discoveredBackends: [String: DiscoveredBackend] = [:]
    private var scanTask: Task<Void, Never>?
    private var capabilitiesCache: [String: ModelCapabilities] = [:]
    private var lastScanTime: Date = .distantPast
    /// Minimum interval between on-demand rescans triggered by /v1/models requests
    private let staleScanInterval: TimeInterval = 10

    init(logger: Logger, selfPort: Int) {
        self.logger = logger
        self.selfPort = selfPort
    }

    /// Scan only known backends (fast), then start periodic full scans in the background.
    /// Returns after known backends are discovered so the server can start immediately.
    func startPeriodicScanning() async {
        // Phase 1: Probe known backends only (fast — ~3 seconds max)
        await scanKnownBackends()

        // Phase 2: Port scan + periodic refresh runs in the background
        scanTask = Task { [weak self] in
            // Run initial port scan immediately (in background)
            await self?.scanOpenPorts()
            // Then periodic full scans
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 30_000_000_000) // 30 seconds
                guard !Task.isCancelled else { break }
                await self?.scanAllBackends()
            }
        }
    }

    func stopScanning() {
        scanTask?.cancel()
        scanTask = nil
    }

    func allDiscoveredModels() -> [DiscoveredModel] {
        discoveredBackends.values.flatMap { $0.models }
    }

    func backendForModel(_ modelId: String) -> DiscoveredModel? {
        for backend in discoveredBackends.values {
            if let model = backend.models.first(where: { $0.id == modelId }) {
                return model
            }
        }
        return nil
    }

    /// Rescan backends if enough time has passed since the last scan.
    /// Called by `/v1/models` so new backends/models appear without waiting
    /// for the full periodic scan interval.
    func refreshIfStale() async {
        let elapsed = Date().timeIntervalSince(lastScanTime)
        guard elapsed >= staleScanInterval else { return }
        await scanAllBackends()
    }

    /// Ports to scan for unknown OpenAI-compatible APIs (beyond the known backends).
    /// Focused on common LLM server ports to keep scans fast.
    private static let scanPortRanges: [ClosedRange<Int>] = [
        3000...3999,   // Node/Express dev servers
        4000...4999,   // Various dev servers
        5000...5999,   // Flask, FastAPI, vLLM
        7860...7899,   // Gradio/HuggingFace
        8000...8099,   // Common HTTP (8080 is known mlx-lm, 8081 is known llama.cpp)
        8082...8200,   // Above known llama.cpp
        8888...8899,   // Jupyter/misc
        9000...9099,   // Various
        9999...9999,   // Common alt port
    ]

    /// Probe only known backends (Ollama, LM Studio, etc.) — fast, ~3 seconds max.
    private func scanKnownBackends() async {
        let knownBackends = BackendDefinition.allKnown.filter { $0.defaultPort != selfPort }

        var newBackends: [String: DiscoveredBackend] = [:]

        await withTaskGroup(of: (String, DiscoveredBackend?).self) { group in
            for definition in knownBackends {
                group.addTask {
                    let backend = await self.probeBackend(definition)
                    return (definition.name, backend)
                }
            }
            for await (name, backend) in group {
                if let backend = backend {
                    newBackends[name] = backend
                }
            }
        }

        self.discoveredBackends = newBackends
        lastScanTime = Date()

        let modelCount = newBackends.values.reduce(0) { $0 + $1.models.count }
        if modelCount > 0 {
            let names = newBackends.keys.sorted().joined(separator: ", ")
            logger.info("Discovered \(modelCount) model(s) from known backends: \(names)")
        }
    }

    /// Scan port ranges for unknown OpenAI-compatible APIs and merge with existing backends.
    private func scanOpenPorts() async {
        let knownPorts = Set(BackendDefinition.allKnown.map { $0.defaultPort })
        let portsToScan = Self.scanPortRanges.flatMap { range in
            range.filter { $0 != selfPort && !knownPorts.contains($0) && !BackendDefinition.blacklistedPorts.contains($0) }
        }

        let openPorts = await findOpenPorts(portsToScan)
        guard !openPorts.isEmpty else {
            logger.info("Port scan complete — no additional OpenAI-compatible APIs found")
            return
        }

        logger.info("Port scan found \(openPorts.count) open port(s): \(openPorts.sorted())")

        var newBackends: [String: DiscoveredBackend] = [:]
        await withTaskGroup(of: (String, DiscoveredBackend?).self) { group in
            for port in openPorts {
                group.addTask {
                    let definition = BackendDefinition(name: "localhost:\(port)", defaultPort: port, hostname: "127.0.0.1")
                    let backend = await self.probeBackend(definition)
                    return (definition.name, backend)
                }
            }
            for await (name, backend) in group {
                if let backend = backend {
                    newBackends[name] = backend
                }
            }
        }

        if !newBackends.isEmpty {
            // Merge with existing discovered backends
            for (name, backend) in newBackends {
                discoveredBackends[name] = backend
            }

            let scannedModelCount = newBackends.values.reduce(0) { $0 + $1.models.count }
            let scannedNames = newBackends.keys.sorted().joined(separator: ", ")
            logger.info("Port scan discovered \(scannedModelCount) additional model(s) from: \(scannedNames)")
        }
    }

    /// Full rescan: known backends + port scan. Used for periodic refresh.
    func scanAllBackends() async {
        let previousModelIds = Set(allDiscoveredModels().map { $0.id })

        await scanKnownBackends()
        await scanOpenPorts()

        lastScanTime = Date()

        // Invalidate capabilities cache for models that disappeared
        let currentModelIds = Set(allDiscoveredModels().map { $0.id })
        let removedIds = previousModelIds.subtracting(currentModelIds)
        for id in removedIds {
            capabilitiesCache.removeValue(forKey: id)
        }
    }

    /// Fast TCP connect scan to find open ports before doing HTTP probes.
    /// Scans in batches to avoid overwhelming the system with thousands of concurrent connections.
    private nonisolated func findOpenPorts(_ ports: [Int]) async -> [Int] {
        let batchSize = 100
        var open: [Int] = []

        for batchStart in stride(from: 0, to: ports.count, by: batchSize) {
            let batchEnd = min(batchStart + batchSize, ports.count)
            let batch = Array(ports[batchStart..<batchEnd])

            let batchResults = await withTaskGroup(of: Int?.self) { group in
                for port in batch {
                    group.addTask {
                        await self.isTCPPortOpen(port) ? port : nil
                    }
                }
                var results: [Int] = []
                for await result in group {
                    if let port = result { results.append(port) }
                }
                return results
            }
            open.append(contentsOf: batchResults)
        }

        return open
    }

    /// Check if a TCP port is open using NWConnection with a short timeout.
    private nonisolated func isTCPPortOpen(_ port: Int) async -> Bool {
        let connection = NWConnection(
            host: NWEndpoint.Host("127.0.0.1"),
            port: NWEndpoint.Port(integerLiteral: UInt16(port)),
            using: .tcp
        )

        return await withCheckedContinuation { continuation in
            var resumed = false
            let lock = NSLock()

            @Sendable func finish(_ result: Bool) {
                lock.lock()
                defer { lock.unlock() }
                guard !resumed else { return }
                resumed = true
                connection.cancel()
                continuation.resume(returning: result)
            }

            connection.stateUpdateHandler = { state in
                switch state {
                case .ready:
                    finish(true)
                case .failed, .cancelled:
                    finish(false)
                default:
                    break
                }
            }

            connection.start(queue: .global(qos: .utility))

            // Timeout after 500ms
            DispatchQueue.global().asyncAfter(deadline: .now() + 0.5) {
                finish(false)
            }
        }
    }

    private func probeBackend(_ definition: BackendDefinition) async -> DiscoveredBackend? {
        let url = URL(string: definition.modelsURL)!
        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        request.setValue("Bearer \(afmAPIKey)", forHTTPHeaderField: "Authorization")
        request.timeoutInterval = 3

        do {
            let (data, response) = try await URLSession.shared.data(for: request)

            guard let httpResponse = response as? HTTPURLResponse,
                  httpResponse.statusCode == 200 else {
                return nil
            }

            let decoder = JSONDecoder()
            let modelsResponse = try decoder.decode(BackendModelsResponse.self, from: data)

            guard let entries = modelsResponse.data, !entries.isEmpty else {
                return nil
            }

            // Build a lookup from model ID → capabilities from the models array
            var detailsByModel: [String: BackendModelsResponse.BackendModelDetail] = [:]
            for detail in modelsResponse.models ?? [] {
                if let id = detail.model ?? detail.name {
                    detailsByModel[id] = detail
                }
            }

            let models = entries.map { entry in
                let displayId = "\(entry.id) · \(definition.name)"

                // Extract capabilities from models array and/or meta
                let detail = detailsByModel[entry.id]
                let caps = detail?.capabilities ?? []
                let hasVision = caps.contains("vision") || caps.contains("multimodal")
                let hasTools = caps.contains("tools")
                let ctxLength = entry.meta?.n_ctx_train

                // Pre-cache capabilities from /v1/models (may be incomplete for Ollama)
                let modelCaps = ModelCapabilities(vision: hasVision, tools: hasTools, contextLength: ctxLength)
                capabilitiesCache[displayId] = modelCaps

                return DiscoveredModel(
                    id: displayId,
                    originalId: entry.id,
                    ownedBy: entry.owned_by ?? definition.name.lowercased(),
                    backendName: definition.name,
                    baseURL: definition.baseURL,
                    created: entry.created ?? Int(Date().timeIntervalSince1970),
                    loaded: true
                )
            }

            // Eagerly probe backend-specific capabilities (Ollama /api/show, LM Studio /api/v0/models)
            // to get accurate vision/tools info that /v1/models doesn't provide
            let needsProbing = definition.name == "Ollama" || definition.name == "LM Studio"
            if needsProbing {
                for model in models {
                    let probedCaps = await probeModelCapabilities(model)
                    // Merge: prefer probed data but keep /v1/models context length if probe didn't find one
                    let existing = capabilitiesCache[model.id] ?? .default
                    let mergedCtx = probedCaps.contextLength ?? existing.contextLength
                    capabilitiesCache[model.id] = ModelCapabilities(
                        vision: probedCaps.vision || existing.vision,
                        tools: probedCaps.tools || existing.tools,
                        contextLength: mergedCtx
                    )
                }
            }

            return DiscoveredBackend(
                definition: definition,
                models: models,
                lastSeen: Date()
            )
        } catch {
            // Connection refused or timeout — backend not running, this is expected
            return nil
        }
    }

    // MARK: - Model Capability Probing

    /// Get capabilities for a model, probing the backend lazily on first request
    func capabilitiesForModel(_ displayId: String) async -> ModelCapabilities {
        if let cached = capabilitiesCache[displayId] {
            return cached
        }

        guard let model = backendForModel(displayId) else {
            return .default
        }

        let caps = await probeModelCapabilities(model)
        capabilitiesCache[displayId] = caps
        return caps
    }

    /// Probe a backend for model-specific capabilities
    private nonisolated func probeModelCapabilities(_ model: DiscoveredModel) async -> ModelCapabilities {
        switch model.backendName {
        case "Ollama":
            return await probeOllamaCapabilities(model)
        case "LM Studio":
            return await probeLMStudioCapabilities(model)
        default:
            return .default
        }
    }

    /// Probe Ollama via POST /api/show for model capabilities
    private nonisolated func probeOllamaCapabilities(_ model: DiscoveredModel) async -> ModelCapabilities {
        guard let url = URL(string: "\(model.baseURL)/api/show") else { return .default }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.timeoutInterval = 5
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("Bearer \(afmAPIKey)", forHTTPHeaderField: "Authorization")
        request.httpBody = try? JSONSerialization.data(withJSONObject: ["model": model.originalId])

        do {
            let (data, response) = try await URLSession.shared.data(for: request)
            guard let httpResponse = response as? HTTPURLResponse,
                  httpResponse.statusCode == 200 else {
                return .default
            }

            let decoded = try JSONDecoder().decode(OllamaShowResponse.self, from: data)
            let caps = decoded.capabilities ?? []
            let hasVision = caps.contains("vision")
            let hasTools = caps.contains("tools")

            // Try to extract context length from model_info
            var contextLength: Int? = nil
            if let info = decoded.model_info {
                // Common keys for context length in Ollama model_info
                for key in info.keys where key.contains("context_length") {
                    contextLength = info[key]?.intValue
                    if contextLength != nil { break }
                }
            }

            return ModelCapabilities(vision: hasVision, tools: hasTools, contextLength: contextLength)
        } catch {
            return .default
        }
    }

    /// Probe LM Studio via GET /api/v0/models/<id> for model capabilities
    private nonisolated func probeLMStudioCapabilities(_ model: DiscoveredModel) async -> ModelCapabilities {
        // URL-encode the model ID for the path
        let encodedId = model.originalId.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? model.originalId
        guard let url = URL(string: "\(model.baseURL)/api/v0/models/\(encodedId)") else { return .default }

        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        request.setValue("Bearer \(afmAPIKey)", forHTTPHeaderField: "Authorization")
        request.timeoutInterval = 5

        do {
            let (data, response) = try await URLSession.shared.data(for: request)
            guard let httpResponse = response as? HTTPURLResponse,
                  httpResponse.statusCode == 200 else {
                return .default
            }

            let decoded = try JSONDecoder().decode(LMStudioModelInfoResponse.self, from: data)
            let hasVision = decoded.type == "vlm" || (decoded.capabilities ?? []).contains("vision")
            let hasTools = (decoded.capabilities ?? []).contains("tools")

            return ModelCapabilities(vision: hasVision, tools: hasTools, contextLength: decoded.max_context_length)
        } catch {
            return .default
        }
    }

    /// Fetch currently running/loaded models from Ollama's /api/ps endpoint
    private nonisolated func fetchOllamaRunningModels(baseURL: String) async -> Set<String> {
        guard let url = URL(string: "\(baseURL)/api/ps") else { return [] }
        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        request.setValue("Bearer \(afmAPIKey)", forHTTPHeaderField: "Authorization")
        request.timeoutInterval = 3

        do {
            let (data, response) = try await URLSession.shared.data(for: request)
            guard let httpResponse = response as? HTTPURLResponse,
                  httpResponse.statusCode == 200 else {
                return []
            }
            let decoded = try JSONDecoder().decode(OllamaRunningModelsResponse.self, from: data)
            return Set((decoded.models ?? []).map { $0.name })
        } catch {
            return []
        }
    }
}

// MARK: - Vapor Storage Keys

struct BackendDiscoveryKey: StorageKey {
    typealias Value = BackendDiscoveryService
}

struct BackendProxyKey: StorageKey {
    typealias Value = BackendProxyService
}

extension Application {
    var backendDiscovery: BackendDiscoveryService? {
        get { storage[BackendDiscoveryKey.self] }
        set { storage[BackendDiscoveryKey.self] = newValue }
    }

    var backendProxy: BackendProxyService? {
        get { storage[BackendProxyKey.self] }
        set { storage[BackendProxyKey.self] = newValue }
    }
}
