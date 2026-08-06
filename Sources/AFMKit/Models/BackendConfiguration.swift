import Foundation

public struct BackendDefinition: Sendable {
    public let name: String
    public let defaultPort: Int
    public let hostname: String

    public var baseURL: String {
        "http://\(hostname):\(defaultPort)"
    }

    public var modelsURL: String {
        "\(baseURL)/v1/models"
    }

    public var chatCompletionsURL: String {
        "\(baseURL)/v1/chat/completions"
    }

    public static let allKnown: [BackendDefinition] = [
        BackendDefinition(name: "Ollama", defaultPort: 11434, hostname: "127.0.0.1"),
        BackendDefinition(name: "LM Studio", defaultPort: 1234, hostname: "127.0.0.1"),
        BackendDefinition(name: "Jan", defaultPort: 1337, hostname: "127.0.0.1"),
        BackendDefinition(name: "mlx-lm", defaultPort: 8080, hostname: "127.0.0.1"),
        BackendDefinition(name: "llama.cpp", defaultPort: 8081, hostname: "127.0.0.1"),
    ]

    /// Ports used internally by known backends (not directly usable)
    public static let blacklistedPorts: Set<Int> = [
        3570,  // Jan's internal llama.cpp server — rejects external requests
    ]
    public init(name: String, defaultPort: Int, hostname: String) {
        self.name = name
        self.defaultPort = defaultPort
        self.hostname = hostname
    }
}

public struct DiscoveredModel: Sendable {
    /// Display ID shown in the webui (may include provider suffix)
    public let id: String
    /// Original model ID as returned by the backend (used for proxying)
    public let originalId: String
    public let ownedBy: String
    public let backendName: String
    public let baseURL: String
    public let created: Int
    public let loaded: Bool
    public init(id: String, originalId: String, ownedBy: String, backendName: String, baseURL: String, created: Int, loaded: Bool) {
        self.id = id
        self.originalId = originalId
        self.ownedBy = ownedBy
        self.backendName = backendName
        self.baseURL = baseURL
        self.created = created
        self.loaded = loaded
    }
}

public struct DiscoveredBackend: Sendable {
    public let definition: BackendDefinition
    public let models: [DiscoveredModel]
    public let lastSeen: Date
    public init(definition: BackendDefinition, models: [DiscoveredModel], lastSeen: Date) {
        self.definition = definition
        self.models = models
        self.lastSeen = lastSeen
    }
}

/// Response from a backend's GET /v1/models endpoint (lenient decoding)
public struct BackendModelsResponse: Decodable, Sendable {
    public let data: [BackendModelEntry]?
    /// Extended model info (llama.cpp, some other backends)
    public let models: [BackendModelDetail]?

    public struct BackendModelEntry: Decodable, Sendable {
        public let id: String
        public let owned_by: String?
        public let created: Int?
        public let meta: BackendModelMeta?
        public init(id: String, owned_by: String?, created: Int?, meta: BackendModelMeta?) {
            self.id = id
            self.owned_by = owned_by
            self.created = created
            self.meta = meta
        }
    }

    public struct BackendModelMeta: Decodable, Sendable {
        public let n_ctx_train: Int?
        public init(n_ctx_train: Int?) {
            self.n_ctx_train = n_ctx_train
        }
    }

    public struct BackendModelDetail: Decodable, Sendable {
        public let model: String?
        public let name: String?
        public let capabilities: [String]?
        public init(model: String?, name: String?, capabilities: [String]?) {
            self.model = model
            self.name = name
            self.capabilities = capabilities
        }
    }
    public init(data: [BackendModelEntry]?, models: [BackendModelDetail]?) {
        self.data = data
        self.models = models
    }
}

/// Capabilities discovered for a specific model via backend-specific probing
public struct ModelCapabilities: Sendable {
    public let vision: Bool
    public let tools: Bool
    public let contextLength: Int?
    public let capabilities: [String]

    public init(vision: Bool = false, tools: Bool = false, contextLength: Int? = nil) {
        self.vision = vision
        self.tools = tools
        self.contextLength = contextLength
        var caps = ["completion"]
        if vision { caps.append("vision") }
        if tools { caps.append("tools") }
        self.capabilities = caps
    }

    /// Default capabilities when probing is not supported for a backend
    public static let `default` = ModelCapabilities()

    /// Foundation model capabilities
    public static let foundation = ModelCapabilities(vision: true, tools: true, contextLength: 4096)
}

/// Response from Ollama's POST /api/show endpoint
public struct OllamaShowResponse: Decodable, Sendable {
    public let capabilities: [String]?
    public let model_info: [String: OllamaModelInfoValue]?
    public init(capabilities: [String]?, model_info: [String: OllamaModelInfoValue]?) {
        self.capabilities = capabilities
        self.model_info = model_info
    }
}

/// Lenient value type for Ollama model_info fields (can be string, int, float, bool)
public enum OllamaModelInfoValue: Decodable, Sendable {
    case int(Int)
    case double(Double)
    case string(String)
    case bool(Bool)

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let v = try? container.decode(Int.self) { self = .int(v) }
        else if let v = try? container.decode(Double.self) { self = .double(v) }
        else if let v = try? container.decode(Bool.self) { self = .bool(v) }
        else if let v = try? container.decode(String.self) { self = .string(v) }
        else { self = .string("") }
    }

    public var intValue: Int? {
        switch self {
        case .int(let v): return v
        case .double(let v): return Int(v)
        default: return nil
        }
    }
}

/// Response from LM Studio's GET /api/v0/models/<id> endpoint
public struct LMStudioModelInfoResponse: Decodable, Sendable {
    public let type: String?
    public let max_context_length: Int?
    public let capabilities: [String]?
    public let state: String?
    public init(type: String?, max_context_length: Int?, capabilities: [String]?, state: String?) {
        self.type = type
        self.max_context_length = max_context_length
        self.capabilities = capabilities
        self.state = state
    }
}

/// Response from LM Studio's GET /api/v0/models endpoint (all models)
public struct LMStudioModelsListResponse: Decodable, Sendable {
    public let data: [LMStudioModelEntry]?

    public struct LMStudioModelEntry: Decodable, Sendable {
        public let id: String
        public let type: String?
        public let state: String?
        public let max_context_length: Int?
        public let capabilities: [String]?
        public init(id: String, type: String?, state: String?, max_context_length: Int?, capabilities: [String]?) {
            self.id = id
            self.type = type
            self.state = state
            self.max_context_length = max_context_length
            self.capabilities = capabilities
        }
    }
    public init(data: [LMStudioModelEntry]?) {
        self.data = data
    }
}

/// Response from Ollama's GET /api/ps endpoint (running models)
public struct OllamaRunningModelsResponse: Decodable, Sendable {
    public let models: [OllamaRunningModel]?

    public struct OllamaRunningModel: Decodable, Sendable {
        public let name: String
        public init(name: String) {
            self.name = name
        }
    }
    public init(models: [OllamaRunningModel]?) {
        self.models = models
    }
}
