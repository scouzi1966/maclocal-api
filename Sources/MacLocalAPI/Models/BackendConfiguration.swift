import Foundation

struct BackendDefinition: Sendable {
    let name: String
    let defaultPort: Int
    let hostname: String

    var baseURL: String {
        "http://\(hostname):\(defaultPort)"
    }

    var modelsURL: String {
        "\(baseURL)/v1/models"
    }

    var chatCompletionsURL: String {
        "\(baseURL)/v1/chat/completions"
    }

    static let allKnown: [BackendDefinition] = [
        BackendDefinition(name: "Ollama", defaultPort: 11434, hostname: "127.0.0.1"),
        BackendDefinition(name: "LM Studio", defaultPort: 1234, hostname: "127.0.0.1"),
        BackendDefinition(name: "Jan", defaultPort: 1337, hostname: "127.0.0.1"),
        BackendDefinition(name: "mlx-lm", defaultPort: 8080, hostname: "127.0.0.1"),
        BackendDefinition(name: "llama.cpp", defaultPort: 8081, hostname: "127.0.0.1"),
    ]

    /// Ports used internally by known backends (not directly usable)
    static let blacklistedPorts: Set<Int> = [
        3570,  // Jan's internal llama.cpp server — rejects external requests
    ]
}

struct DiscoveredModel: Sendable {
    /// Display ID shown in the webui (may include provider suffix)
    let id: String
    /// Original model ID as returned by the backend (used for proxying)
    let originalId: String
    let ownedBy: String
    let backendName: String
    let baseURL: String
    let created: Int
    let loaded: Bool
}

struct DiscoveredBackend: Sendable {
    let definition: BackendDefinition
    let models: [DiscoveredModel]
    let lastSeen: Date
}

/// Response from a backend's GET /v1/models endpoint (lenient decoding)
struct BackendModelsResponse: Decodable {
    let data: [BackendModelEntry]?
    /// Extended model info (llama.cpp, some other backends)
    let models: [BackendModelDetail]?

    struct BackendModelEntry: Decodable {
        let id: String
        let owned_by: String?
        let created: Int?
        let meta: BackendModelMeta?
    }

    struct BackendModelMeta: Decodable {
        let n_ctx_train: Int?
    }

    struct BackendModelDetail: Decodable {
        let model: String?
        let name: String?
        let capabilities: [String]?
    }
}

/// Capabilities discovered for a specific model via backend-specific probing
struct ModelCapabilities: Sendable {
    let vision: Bool
    let tools: Bool
    let contextLength: Int?
    let capabilities: [String]

    init(vision: Bool = false, tools: Bool = false, contextLength: Int? = nil) {
        self.vision = vision
        self.tools = tools
        self.contextLength = contextLength
        var caps = ["completion"]
        if vision { caps.append("vision") }
        if tools { caps.append("tools") }
        self.capabilities = caps
    }

    /// Default capabilities when probing is not supported for a backend
    static let `default` = ModelCapabilities()

    /// Foundation model capabilities
    static let foundation = ModelCapabilities(vision: false, contextLength: 4096)
}

/// Response from Ollama's POST /api/show endpoint
struct OllamaShowResponse: Decodable {
    let capabilities: [String]?
    let model_info: [String: OllamaModelInfoValue]?
}

/// Lenient value type for Ollama model_info fields (can be string, int, float, bool)
enum OllamaModelInfoValue: Decodable {
    case int(Int)
    case double(Double)
    case string(String)
    case bool(Bool)

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let v = try? container.decode(Int.self) { self = .int(v) }
        else if let v = try? container.decode(Double.self) { self = .double(v) }
        else if let v = try? container.decode(Bool.self) { self = .bool(v) }
        else if let v = try? container.decode(String.self) { self = .string(v) }
        else { self = .string("") }
    }

    var intValue: Int? {
        switch self {
        case .int(let v): return v
        case .double(let v): return Int(v)
        default: return nil
        }
    }
}

/// Response from LM Studio's GET /api/v0/models/<id> endpoint
struct LMStudioModelInfoResponse: Decodable {
    let type: String?
    let max_context_length: Int?
    let capabilities: [String]?
    let state: String?
}

/// Response from LM Studio's GET /api/v0/models endpoint (all models)
struct LMStudioModelsListResponse: Decodable {
    let data: [LMStudioModelEntry]?

    struct LMStudioModelEntry: Decodable {
        let id: String
        let type: String?
        let state: String?
        let max_context_length: Int?
        let capabilities: [String]?
    }
}

/// Response from Ollama's GET /api/ps endpoint (running models)
struct OllamaRunningModelsResponse: Decodable {
    let models: [OllamaRunningModel]?

    struct OllamaRunningModel: Decodable {
        let name: String
    }
}
