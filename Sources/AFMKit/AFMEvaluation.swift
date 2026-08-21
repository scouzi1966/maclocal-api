import Foundation
import AFMOpenAICompat

public struct AFMEvaluationSuite: Codable, Sendable {
    public let schemaVersion: Int
    public let name: String
    public let description: String
    public let defaults: AFMEvaluationParameters?
    public let cases: [AFMEvaluationCase]

    public init(
        schemaVersion: Int = 1,
        name: String,
        description: String,
        defaults: AFMEvaluationParameters? = nil,
        cases: [AFMEvaluationCase]
    ) {
        self.schemaVersion = schemaVersion
        self.name = name
        self.description = description
        self.defaults = defaults
        self.cases = cases
    }
}

public struct AFMEvaluationCase: Codable, Sendable {
    public let id: String
    public let description: String?
    public let prompt: String
    public let system: String?
    public let developer: String?
    public let parameters: AFMEvaluationParameters?
    public let expectations: AFMEvaluationExpectations?

    public init(
        id: String,
        description: String? = nil,
        prompt: String,
        system: String? = nil,
        developer: String? = nil,
        parameters: AFMEvaluationParameters? = nil,
        expectations: AFMEvaluationExpectations? = nil
    ) {
        self.id = id
        self.description = description
        self.prompt = prompt
        self.system = system
        self.developer = developer
        self.parameters = parameters
        self.expectations = expectations
    }
}

public struct AFMEvaluationParameters: Codable, Sendable {
    public let temperature: Double?
    public let maxTokens: Int?
    public let topP: Double?
    public let topK: Int?
    public let minP: Double?
    public let repetitionPenalty: Double?
    public let presencePenalty: Double?
    public let seed: Int?
    public let logprobs: Bool?
    public let topLogprobs: Int?
    public let stop: [String]?
    public let tools: [RequestTool]?
    public let responseFormat: ResponseFormat?
    public let streaming: Bool?

    public init(
        temperature: Double? = nil,
        maxTokens: Int? = nil,
        topP: Double? = nil,
        topK: Int? = nil,
        minP: Double? = nil,
        repetitionPenalty: Double? = nil,
        presencePenalty: Double? = nil,
        seed: Int? = nil,
        logprobs: Bool? = nil,
        topLogprobs: Int? = nil,
        stop: [String]? = nil,
        tools: [RequestTool]? = nil,
        responseFormat: ResponseFormat? = nil,
        streaming: Bool? = nil
    ) {
        self.temperature = temperature
        self.maxTokens = maxTokens
        self.topP = topP
        self.topK = topK
        self.minP = minP
        self.repetitionPenalty = repetitionPenalty
        self.presencePenalty = presencePenalty
        self.seed = seed
        self.logprobs = logprobs
        self.topLogprobs = topLogprobs
        self.stop = stop
        self.tools = tools
        self.responseFormat = responseFormat
        self.streaming = streaming
    }

    public func merging(_ override: AFMEvaluationParameters?) -> AFMEvaluationParameters {
        guard let override else { return self }
        return AFMEvaluationParameters(
            temperature: override.temperature ?? temperature,
            maxTokens: override.maxTokens ?? maxTokens,
            topP: override.topP ?? topP,
            topK: override.topK ?? topK,
            minP: override.minP ?? minP,
            repetitionPenalty: override.repetitionPenalty ?? repetitionPenalty,
            presencePenalty: override.presencePenalty ?? presencePenalty,
            seed: override.seed ?? seed,
            logprobs: override.logprobs ?? logprobs,
            topLogprobs: override.topLogprobs ?? topLogprobs,
            stop: override.stop ?? stop,
            tools: override.tools ?? tools,
            responseFormat: override.responseFormat ?? responseFormat,
            streaming: override.streaming ?? streaming
        )
    }
}

public struct AFMEvaluationExpectations: Codable, Sendable {
    public let exact: String?
    public let contains: [String]?
    public let notContains: [String]?
    public let validJSON: Bool?
    public let minimumCharacters: Int?
    public let maximumCharacters: Int?
    public let toolCallName: String?
    public let caseSensitive: Bool?
    public let matchesCase: String?

    public init(
        exact: String? = nil,
        contains: [String]? = nil,
        notContains: [String]? = nil,
        validJSON: Bool? = nil,
        minimumCharacters: Int? = nil,
        maximumCharacters: Int? = nil,
        toolCallName: String? = nil,
        caseSensitive: Bool? = nil,
        matchesCase: String? = nil
    ) {
        self.exact = exact
        self.contains = contains
        self.notContains = notContains
        self.validJSON = validJSON
        self.minimumCharacters = minimumCharacters
        self.maximumCharacters = maximumCharacters
        self.toolCallName = toolCallName
        self.caseSensitive = caseSensitive
        self.matchesCase = matchesCase
    }
}

public enum AFMEvaluationOutcome: String, Codable, Sendable {
    case passed
    case missed
    case observed
    case error
}

public struct AFMEvaluationCheckResult: Codable, Sendable {
    public let name: String
    public let passed: Bool
    public let detail: String

    public init(name: String, passed: Bool, detail: String) {
        self.name = name
        self.passed = passed
        self.detail = detail
    }
}

public struct AFMEvaluationToolCall: Codable, Sendable {
    public let name: String
    public let arguments: String

    public init(name: String, arguments: String) {
        self.name = name
        self.arguments = arguments
    }
}

public struct AFMEvaluationCaseResult: Codable, Sendable {
    public let suite: String
    public let caseID: String
    public let prompt: String
    public let system: String?
    public let output: String
    public let reasoning: String?
    public let toolCalls: [AFMEvaluationToolCall]
    public let outcome: AFMEvaluationOutcome
    public let checks: [AFMEvaluationCheckResult]
    public let error: String?
    public let startedAt: Date
    public let durationSeconds: Double
    public let timeToFirstTokenSeconds: Double?
    public let promptTimeSeconds: Double?
    public let generationTimeSeconds: Double?
    public let promptTokens: Int
    public let cachedPromptTokens: Int
    public let completionTokens: Int
    public let tokensPerSecond: Double?
    public let finishReason: String
    public let parameters: AFMEvaluationParameters

    public init(
        suite: String,
        caseID: String,
        prompt: String,
        system: String?,
        output: String,
        reasoning: String?,
        toolCalls: [AFMEvaluationToolCall],
        outcome: AFMEvaluationOutcome,
        checks: [AFMEvaluationCheckResult],
        error: String?,
        startedAt: Date,
        durationSeconds: Double,
        timeToFirstTokenSeconds: Double?,
        promptTimeSeconds: Double?,
        generationTimeSeconds: Double?,
        promptTokens: Int,
        cachedPromptTokens: Int,
        completionTokens: Int,
        tokensPerSecond: Double?,
        finishReason: String,
        parameters: AFMEvaluationParameters
    ) {
        self.suite = suite
        self.caseID = caseID
        self.prompt = prompt
        self.system = system
        self.output = output
        self.reasoning = reasoning
        self.toolCalls = toolCalls
        self.outcome = outcome
        self.checks = checks
        self.error = error
        self.startedAt = startedAt
        self.durationSeconds = durationSeconds
        self.timeToFirstTokenSeconds = timeToFirstTokenSeconds
        self.promptTimeSeconds = promptTimeSeconds
        self.generationTimeSeconds = generationTimeSeconds
        self.promptTokens = promptTokens
        self.cachedPromptTokens = cachedPromptTokens
        self.completionTokens = completionTokens
        self.tokensPerSecond = tokensPerSecond
        self.finishReason = finishReason
        self.parameters = parameters
    }
}

public struct AFMEvaluationSystemInfo: Codable, Sendable {
    public let operatingSystem: String
    public let architecture: String
    public let processorCount: Int
    public let physicalMemoryBytes: UInt64

    public init(
        operatingSystem: String,
        architecture: String,
        processorCount: Int,
        physicalMemoryBytes: UInt64
    ) {
        self.operatingSystem = operatingSystem
        self.architecture = architecture
        self.processorCount = processorCount
        self.physicalMemoryBytes = physicalMemoryBytes
    }
}

public struct AFMEvaluationRunReport: Codable, Sendable {
    public let schemaVersion: Int
    public let afmVersion: String
    public let model: String
    public let suites: [String]
    public let startedAt: Date
    public let finishedAt: Date
    public let interrupted: Bool
    public let reproducibilityCommand: String
    public let system: AFMEvaluationSystemInfo
    public let results: [AFMEvaluationCaseResult]

    public init(
        schemaVersion: Int = 1,
        afmVersion: String,
        model: String,
        suites: [String],
        startedAt: Date,
        finishedAt: Date,
        interrupted: Bool,
        reproducibilityCommand: String,
        system: AFMEvaluationSystemInfo,
        results: [AFMEvaluationCaseResult]
    ) {
        self.schemaVersion = schemaVersion
        self.afmVersion = afmVersion
        self.model = model
        self.suites = suites
        self.startedAt = startedAt
        self.finishedAt = finishedAt
        self.interrupted = interrupted
        self.reproducibilityCommand = reproducibilityCommand
        self.system = system
        self.results = results
    }
}

public struct AFMEvaluationSuiteDescriptor: Sendable {
    public enum Origin: String, Sendable { case bundled, custom }
    public let name: String
    public let description: String
    public let caseCount: Int
    public let origin: Origin
    public let url: URL
}

public enum AFMEvaluationError: LocalizedError {
    case invalidSuite(String)
    case suiteNotFound(String)
    case conflictingCLI(String)

    public var errorDescription: String? {
        switch self {
        case .invalidSuite(let message): return "Invalid evaluation suite: \(message)"
        case .suiteNotFound(let name):
            return "Evaluation suite '\(name)' was not found. Use --eval-list to see available suites."
        case .conflictingCLI(let message): return message
        }
    }
}

public enum AFMEvaluationCLIAction: Equatable, Sendable {
    case none
    case run(suites: [String], openReport: Bool)
    case list
    case scaffold(name: String)
    case validate(reference: String)
}

public enum AFMEvaluationCLIPlan {
    public static func resolve(
        evaluate: Bool,
        bench: Bool,
        suites: [String],
        list: Bool,
        scaffold: String?,
        validate: String?,
        noOpen: Bool
    ) throws -> AFMEvaluationCLIAction {
        let managementCount = [list, scaffold != nil, validate != nil].filter { $0 }.count
        guard managementCount <= 1 else {
            throw AFMEvaluationError.conflictingCLI(
                "Use only one of --eval-list, --eval-init, or --eval-validate at a time.")
        }
        if list { return .list }
        if let scaffold { return .scaffold(name: scaffold) }
        if let validate { return .validate(reference: validate) }
        if evaluate || bench || !suites.isEmpty {
            let selected = suites.isEmpty ? ["comprehensive"] : suites
            return .run(suites: selected, openReport: !noOpen)
        }
        if noOpen {
            throw AFMEvaluationError.conflictingCLI("--no-open is only valid with --eval or --bench.")
        }
        return .none
    }
}

public enum AFMEvaluationScorer {
    public static func score(
        output: String,
        toolCallNames: [String] = [],
        expectations: AFMEvaluationExpectations?
    ) -> (AFMEvaluationOutcome, [AFMEvaluationCheckResult]) {
        guard let expectations else { return (.observed, []) }
        let caseSensitive = expectations.caseSensitive == true
        let comparableOutput = caseSensitive ? output : output.lowercased()
        func comparable(_ value: String) -> String { caseSensitive ? value : value.lowercased() }
        var checks: [AFMEvaluationCheckResult] = []

        if let exact = expectations.exact {
            let passed = comparableOutput.trimmingCharacters(in: .whitespacesAndNewlines)
                == comparable(exact).trimmingCharacters(in: .whitespacesAndNewlines)
            checks.append(.init(name: "exact", passed: passed, detail: "Output matches the expected text"))
        }
        for value in expectations.contains ?? [] {
            checks.append(.init(
                name: "contains",
                passed: comparableOutput.contains(comparable(value)),
                detail: "Output contains '\(value)'"))
        }
        for value in expectations.notContains ?? [] {
            checks.append(.init(
                name: "notContains",
                passed: !comparableOutput.contains(comparable(value)),
                detail: "Output does not contain '\(value)'"))
        }
        if let expectedJSON = expectations.validJSON {
            let isJSON = output.data(using: .utf8).flatMap {
                try? JSONSerialization.jsonObject(with: $0, options: [.fragmentsAllowed])
            } != nil
            checks.append(.init(
                name: "validJSON",
                passed: isJSON == expectedJSON,
                detail: expectedJSON ? "Output is valid JSON" : "Output is not valid JSON"))
        }
        if let minimum = expectations.minimumCharacters {
            checks.append(.init(
                name: "minimumCharacters",
                passed: output.count >= minimum,
                detail: "Output has at least \(minimum) characters"))
        }
        if let maximum = expectations.maximumCharacters {
            checks.append(.init(
                name: "maximumCharacters",
                passed: output.count <= maximum,
                detail: "Output has no more than \(maximum) characters"))
        }
        if let expectedTool = expectations.toolCallName {
            checks.append(.init(
                name: "toolCallName",
                passed: toolCallNames.contains(expectedTool),
                detail: "Model calls tool '\(expectedTool)'"))
        }
        guard !checks.isEmpty else { return (.observed, []) }
        return (checks.allSatisfy(\.passed) ? .passed : .missed, checks)
    }
}

public struct AFMEvaluationSuiteStore {
    public let rootDirectory: URL
    private let fileManager: FileManager

    public init(
        rootDirectory: URL = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".afm/evals", isDirectory: true),
        fileManager: FileManager = .default
    ) {
        self.rootDirectory = rootDirectory
        self.fileManager = fileManager
    }

    public static func bundledSuiteURL(named name: String) -> URL? {
        Bundle.module.url(forResource: name, withExtension: "json", subdirectory: "Evals")
            ?? Bundle.module.url(forResource: name, withExtension: "json")
    }

    public func discover() throws -> [AFMEvaluationSuiteDescriptor] {
        var descriptors: [AFMEvaluationSuiteDescriptor] = []
        if let urls = Bundle.module.urls(forResourcesWithExtension: "json", subdirectory: "Evals") {
            for url in urls {
                let suite = try decode(url: url, origin: .bundled)
                descriptors.append(.init(
                    name: suite.name,
                    description: suite.description,
                    caseCount: suite.cases.count,
                    origin: .bundled,
                    url: url))
            }
        } else if let url = Self.bundledSuiteURL(named: "comprehensive") {
            let suite = try decode(url: url, origin: .bundled)
            descriptors.append(.init(
                name: suite.name,
                description: suite.description,
                caseCount: suite.cases.count,
                origin: .bundled,
                url: url))
        }
        if fileManager.fileExists(atPath: rootDirectory.path) {
            let urls = try fileManager.contentsOfDirectory(
                at: rootDirectory,
                includingPropertiesForKeys: [.isRegularFileKey],
                options: [.skipsHiddenFiles])
            for url in urls where url.pathExtension.lowercased() == "json" {
                let values = try? url.resourceValues(forKeys: [.isRegularFileKey])
                guard values?.isRegularFile == true else { continue }
                // Discovery is best-effort: one unrelated malformed custom file must
                // not prevent users from listing or loading every valid suite.
                guard let suite = try? decode(url: url, origin: .custom) else { continue }
                descriptors.removeAll { $0.name == suite.name }
                descriptors.append(.init(
                    name: suite.name,
                    description: suite.description,
                    caseCount: suite.cases.count,
                    origin: .custom,
                    url: url))
            }
        }
        return descriptors.sorted {
            $0.name.localizedCaseInsensitiveCompare($1.name) == .orderedAscending
        }
    }

    public func load(named name: String) throws -> AFMEvaluationSuite {
        try Self.validateSafeName(name, field: "suite name")
        let directCustom = rootDirectory.appendingPathComponent("\(name).json")
        if fileManager.fileExists(atPath: directCustom.path) {
            let suite = try decode(url: directCustom, origin: .custom)
            guard suite.name == name else {
                throw AFMEvaluationError.invalidSuite(
                    "\(directCustom.lastPathComponent) declares suite '\(suite.name)', expected '\(name)'")
            }
            return suite
        }
        if fileManager.fileExists(atPath: rootDirectory.path) {
            let urls = try fileManager.contentsOfDirectory(
                at: rootDirectory,
                includingPropertiesForKeys: [.isRegularFileKey],
                options: [.skipsHiddenFiles])
                .filter { $0.pathExtension.lowercased() == "json" }
                .sorted { $0.path < $1.path }
            for url in urls where url != directCustom {
                guard let suite = try? decode(url: url, origin: .custom) else { continue }
                if suite.name == name { return suite }
            }
        }
        if let url = Self.bundledSuiteURL(named: name) {
            return try decode(url: url, origin: .bundled)
        }
        if let bundled = try discover().first(where: { $0.name == name && $0.origin == .bundled }) {
            return try decode(url: bundled.url, origin: .bundled)
        }
        throw AFMEvaluationError.suiteNotFound(name)
    }

    public func load(reference: String) throws -> AFMEvaluationSuite {
        let expanded = NSString(string: reference).expandingTildeInPath
        if expanded.contains("/") || expanded.hasSuffix(".json") {
            return try decode(url: URL(fileURLWithPath: expanded).standardizedFileURL)
        }
        return try load(named: reference)
    }

    public func decode(url: URL) throws -> AFMEvaluationSuite {
        try decode(url: url, origin: .custom)
    }

    func decode(
        url: URL,
        origin: AFMEvaluationSuiteDescriptor.Origin
    ) throws -> AFMEvaluationSuite {
        let data: Data
        do { data = try Data(contentsOf: url, options: [.mappedIfSafe]) }
        catch { throw AFMEvaluationError.invalidSuite("Cannot read \(url.path): \(error.localizedDescription)") }
        guard data.count <= 5_000_000 else {
            throw AFMEvaluationError.invalidSuite("Suite files are limited to 5 MB")
        }
        try Self.validateKnownKeys(data)
        do {
            let suite = try JSONDecoder().decode(AFMEvaluationSuite.self, from: data)
            try Self.validate(suite, origin: origin)
            return suite
        } catch let error as AFMEvaluationError {
            throw error
        } catch {
            throw AFMEvaluationError.invalidSuite("\(url.lastPathComponent): \(error.localizedDescription)")
        }
    }

    public func scaffold(named name: String) throws -> URL {
        try Self.validateSafeName(name, field: "suite name")
        try fileManager.createDirectory(at: rootDirectory, withIntermediateDirectories: true)
        try? fileManager.setAttributes([.posixPermissions: 0o700], ofItemAtPath: rootDirectory.path)
        let url = rootDirectory.appendingPathComponent("\(name).json", isDirectory: false)
        guard !fileManager.fileExists(atPath: url.path) else {
            throw AFMEvaluationError.invalidSuite("Refusing to overwrite existing file \(url.path)")
        }
        let suite = AFMEvaluationSuite(
            name: name,
            description: "A custom deterministic AFM evaluation suite.",
            defaults: .init(temperature: 0, maxTokens: 256, seed: 42),
            cases: [
                .init(
                    id: "hello",
                    description: "Replace this example with your own local test.",
                    prompt: "Respond with exactly: hello",
                    expectations: .init(exact: "hello"))
            ])
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        try encoder.encode(suite).write(to: url, options: [.atomic])
        try? fileManager.setAttributes([.posixPermissions: 0o600], ofItemAtPath: url.path)
        return url
    }

    public func makeRunDirectory(model: String, suites: [String], date: Date = Date()) throws -> URL {
        try fileManager.createDirectory(at: rootDirectory, withIntermediateDirectories: true)
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = TimeZone(secondsFromGMT: 0)
        formatter.dateFormat = "yyyyMMdd-HHmmss-SSS"
        let modelPart = Self.sanitizePathComponent(model)
        let suitePart = Self.sanitizePathComponent(suites.joined(separator: "+"))
        let base = "\(formatter.string(from: date))-\(modelPart)-\(suitePart)"
        var candidate = rootDirectory.appendingPathComponent(base, isDirectory: true)
        var suffix = 2
        while fileManager.fileExists(atPath: candidate.path) {
            candidate = rootDirectory.appendingPathComponent("\(base)-\(suffix)", isDirectory: true)
            suffix += 1
        }
        try fileManager.createDirectory(at: candidate, withIntermediateDirectories: false)
        try? fileManager.setAttributes([.posixPermissions: 0o700], ofItemAtPath: candidate.path)
        return candidate
    }

    public static func sanitizePathComponent(_ value: String) -> String {
        let allowed = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "-_."))
        let result = value.unicodeScalars.map { allowed.contains($0) ? Character(String($0)) : "-" }
        var text = String(result).replacingOccurrences(of: "--", with: "-")
        while text.contains("--") { text = text.replacingOccurrences(of: "--", with: "-") }
        text = text.trimmingCharacters(in: CharacterSet(charactersIn: ".-_"))
        if text.isEmpty { text = "unnamed" }
        return String(text.prefix(80))
    }

    private static func validate(
        _ suite: AFMEvaluationSuite,
        origin: AFMEvaluationSuiteDescriptor.Origin
    ) throws {
        guard suite.schemaVersion == 1 else {
            throw AFMEvaluationError.invalidSuite("Unsupported schemaVersion \(suite.schemaVersion); expected 1")
        }
        try validateSafeName(suite.name, field: "suite name")
        guard !suite.description.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw AFMEvaluationError.invalidSuite("description must not be empty")
        }
        guard !suite.cases.isEmpty, suite.cases.count <= 1_000 else {
            throw AFMEvaluationError.invalidSuite("cases must contain 1...1000 entries")
        }
        var identifiers = Set<String>()
        try validate(suite.defaults, context: "defaults")
        for testCase in suite.cases {
            try validateSafeName(testCase.id, field: "case id")
            guard !identifiers.contains(testCase.id) else {
                throw AFMEvaluationError.invalidSuite("Duplicate case id '\(testCase.id)'")
            }
            guard !testCase.prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                throw AFMEvaluationError.invalidSuite("Case '\(testCase.id)' has an empty prompt")
            }
            guard testCase.prompt.utf8.count <= 65_536,
                  (testCase.system?.utf8.count ?? 0) <= 65_536,
                  (testCase.developer?.utf8.count ?? 0) <= 65_536 else {
                throw AFMEvaluationError.invalidSuite("Case '\(testCase.id)' prompt/system/developer text exceeds 64 KB")
            }
            try validate(testCase.parameters, context: "case '\(testCase.id)'")
            if let expectations = testCase.expectations {
                if let minimum = expectations.minimumCharacters, minimum < 0 {
                    throw AFMEvaluationError.invalidSuite("Case '\(testCase.id)' minimumCharacters must be >= 0")
                }
                if let maximum = expectations.maximumCharacters, maximum < 0 {
                    throw AFMEvaluationError.invalidSuite("Case '\(testCase.id)' maximumCharacters must be >= 0")
                }
                if let match = expectations.matchesCase {
                    guard origin == .bundled else {
                        throw AFMEvaluationError.invalidSuite(
                            "Case '\(testCase.id)' uses matchesCase, which is reserved for bundled suites")
                    }
                    guard identifiers.contains(match) else {
                        throw AFMEvaluationError.invalidSuite(
                            "Case '\(testCase.id)' matchesCase must reference an earlier case in the same suite")
                    }
                }
            }
            identifiers.insert(testCase.id)
        }
    }

    private static func validate(_ value: AFMEvaluationParameters?, context: String) throws {
        guard let value else { return }
        if let temperature = value.temperature, !(0...2).contains(temperature) {
            throw AFMEvaluationError.invalidSuite("\(context) temperature must be 0...2")
        }
        if let maxTokens = value.maxTokens, !(1...32_768).contains(maxTokens) {
            throw AFMEvaluationError.invalidSuite("\(context) maxTokens must be 1...32768")
        }
        if let topP = value.topP, !(0...1).contains(topP) {
            throw AFMEvaluationError.invalidSuite("\(context) topP must be 0...1")
        }
        if let topK = value.topK, topK < 0 || topK > 100_000 {
            throw AFMEvaluationError.invalidSuite("\(context) topK must be 0...100000")
        }
        if let minP = value.minP, !(0...1).contains(minP) {
            throw AFMEvaluationError.invalidSuite("\(context) minP must be 0...1")
        }
        if let topLogprobs = value.topLogprobs, !(0...20).contains(topLogprobs) {
            throw AFMEvaluationError.invalidSuite("\(context) topLogprobs must be 0...20")
        }
        if let stop = value.stop, stop.count > 16 || stop.contains(where: { $0.utf8.count > 1_024 }) {
            throw AFMEvaluationError.invalidSuite("\(context) stop allows at most 16 strings of 1 KB each")
        }
    }

    private static func validateSafeName(_ value: String, field: String) throws {
        let pattern = "^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$"
        guard value.range(of: pattern, options: .regularExpression) != nil,
              value != ".", value != ".." else {
            throw AFMEvaluationError.invalidSuite(
                "\(field) '\(value)' must use 1...64 letters, digits, dots, underscores, or hyphens")
        }
    }

    private static func validateKnownKeys(_ data: Data) throws {
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw AFMEvaluationError.invalidSuite("Top-level JSON value must be an object")
        }
        try rejectUnknown(root, allowed: ["schemaVersion", "name", "description", "defaults", "cases"], at: "suite")
        if let defaults = root["defaults"] as? [String: Any] {
            try rejectUnknown(defaults, allowed: parameterKeys, at: "defaults")
        }
        if let cases = root["cases"] as? [[String: Any]] {
            for (index, item) in cases.enumerated() {
                try rejectUnknown(item, allowed: ["id", "description", "prompt", "system", "developer", "parameters", "expectations"], at: "cases[\(index)]")
                if let parameters = item["parameters"] as? [String: Any] {
                    try rejectUnknown(parameters, allowed: parameterKeys, at: "cases[\(index)].parameters")
                }
                if let expectations = item["expectations"] as? [String: Any] {
                    try rejectUnknown(expectations, allowed: expectationKeys, at: "cases[\(index)].expectations")
                }
            }
        }
    }

    private static let parameterKeys: Set<String> = [
        "temperature", "maxTokens", "topP", "topK", "minP", "repetitionPenalty",
        "presencePenalty", "seed", "logprobs", "topLogprobs", "stop", "tools",
        "responseFormat", "streaming"
    ]
    private static let expectationKeys: Set<String> = [
        "exact", "contains", "notContains", "validJSON", "minimumCharacters",
        "maximumCharacters", "toolCallName", "caseSensitive", "matchesCase"
    ]

    private static func rejectUnknown(_ object: [String: Any], allowed: Set<String>, at path: String) throws {
        let unknown = Set(object.keys).subtracting(allowed).sorted()
        guard unknown.isEmpty else {
            throw AFMEvaluationError.invalidSuite("Unknown key(s) at \(path): \(unknown.joined(separator: ", "))")
        }
    }
}

public enum AFMEvaluationReportWriter {
    public static func jsonEncoder(pretty: Bool = true) -> JSONEncoder {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = pretty
            ? [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
            : [.sortedKeys, .withoutEscapingSlashes]
        return encoder
    }

    public static func html(for report: AFMEvaluationRunReport) -> String {
        let passed = report.results.filter { $0.outcome == .passed }.count
        let missed = report.results.filter { $0.outcome == .missed }.count
        let observed = report.results.filter { $0.outcome == .observed }.count
        let errors = report.results.filter { $0.outcome == .error }.count
        let totalPromptTokens = report.results.reduce(0) { $0 + $1.promptTokens }
        let totalTokens = report.results.reduce(0) { $0 + $1.completionTokens }
        let totalDuration = report.results.reduce(0) { $0 + $1.durationSeconds }
        let aggregateTPS = totalDuration > 0 ? Double(totalTokens) / totalDuration : 0
        let averageLatency = report.results.isEmpty ? 0 : totalDuration / Double(report.results.count)
        let rows = report.results.map { result in
            let checks = result.checks.map {
                "<li class=\"\($0.passed ? "ok" : "bad")\">\(escape($0.name)): \(escape($0.detail))</li>"
            }.joined()
            let tools = result.toolCalls.map {
                "<pre>\(escape($0.name))(\(escape($0.arguments)))</pre>"
            }.joined()
            return """
            <details class="case \(result.outcome.rawValue)">
              <summary><strong>\(escape(result.suite))/\(escape(result.caseID))</strong><span>\(escape(result.outcome.rawValue.uppercased()))</span><span>\(format(result.durationSeconds))s · \(result.completionTokens) tok · \(format(result.tokensPerSecond ?? 0)) tok/s</span></summary>
              <div class="grid"><section><h3>Prompt</h3><pre>\(escape(result.prompt))</pre></section><section><h3>Output</h3><pre>\(escape(result.output))</pre></section></div>
              \(result.reasoning.map { "<section><h3>Reasoning</h3><pre>\(escape($0))</pre></section>" } ?? "")
              \(tools.isEmpty ? "" : "<section><h3>Tool calls</h3>\(tools)</section>")
              \(checks.isEmpty ? "" : "<section><h3>Deterministic checks</h3><ul>\(checks)</ul></section>")
              \(result.error.map { "<p class=\"bad\">\(escape($0))</p>" } ?? "")
              <section><h3>Generation parameters</h3><pre>\(escape(parameterJSON(result.parameters)))</pre></section>
              <p class="meta">TTFT: \(result.timeToFirstTokenSeconds.map(format) ?? "n/a")s · prompt/prefill: \(result.promptTimeSeconds.map(format) ?? "n/a")s · generation: \(result.generationTimeSeconds.map(format) ?? "n/a")s · prompt/cached/output tokens: \(result.promptTokens)/\(result.cachedPromptTokens)/\(result.completionTokens) · finish: \(escape(result.finishReason))</p>
            </details>
            """
        }.joined(separator: "\n")

        return """
        <!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
        <title>AFM evaluation · \(escape(report.model))</title>
        <style>body{font:15px system-ui;margin:0;background:#0d1117;color:#e6edf3}main{max-width:1200px;margin:auto;padding:32px}h1{margin-bottom:4px}.muted,.meta{color:#8b949e}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:12px;margin:24px 0}.card,.case{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:14px}.card b{font-size:24px;display:block}.case{margin:12px 0}.case summary{display:grid;grid-template-columns:1fr auto auto;gap:18px;cursor:pointer}.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}pre{white-space:pre-wrap;word-break:break-word;background:#0d1117;padding:12px;border-radius:8px;overflow:auto}.ok,.passed{color:#3fb950}.bad,.missed,.error{color:#f85149}.observed{color:#d29922}code{word-break:break-all}@media(max-width:750px){.grid{grid-template-columns:1fr}.case summary{grid-template-columns:1fr}}</style></head>
        <body><main><h1>AFM model evaluation</h1><p class="muted">\(escape(report.model)) · \(escape(report.suites.joined(separator: ", "))) · \(escape(report.afmVersion))</p>
        <div class="cards"><div class="card"><b>\(report.results.count)</b>cases</div><div class="card"><b class="ok">\(passed)</b>passed</div><div class="card"><b class="bad">\(missed)</b>quality misses</div><div class="card"><b>\(observed)</b>observed</div><div class="card"><b class="bad">\(errors)</b>errors</div><div class="card"><b>\(totalPromptTokens)</b>prompt tokens</div><div class="card"><b>\(totalTokens)</b>output tokens</div><div class="card"><b>\(format(averageLatency))s</b>average latency</div><div class="card"><b>\(format(aggregateTPS))</b>aggregate tok/s</div></div>
        <p><strong>System:</strong> \(escape(report.system.operatingSystem)); \(escape(report.system.architecture)); \(report.system.processorCount) CPUs; \(formatBytes(report.system.physicalMemoryBytes)) RAM</p>
        <p><strong>Reproduce:</strong> <code>\(escape(report.reproducibilityCommand))</code></p>
        \(report.interrupted ? "<p class=\"bad\"><strong>Run interrupted; partial results preserved.</strong></p>" : "")
        \(rows)</main></body></html>
        """
    }

    public static func escape(_ value: String) -> String {
        value.replacingOccurrences(of: "&", with: "&amp;")
            .replacingOccurrences(of: "<", with: "&lt;")
            .replacingOccurrences(of: ">", with: "&gt;")
            .replacingOccurrences(of: "\"", with: "&quot;")
            .replacingOccurrences(of: "'", with: "&#39;")
    }

    private static func format(_ value: Double) -> String { String(format: "%.2f", value) }
    private static func parameterJSON(_ value: AFMEvaluationParameters) -> String {
        guard let data = try? jsonEncoder(pretty: false).encode(value),
              let text = String(data: data, encoding: .utf8) else { return "{}" }
        return text
    }
    private static func formatBytes(_ value: UInt64) -> String {
        String(format: "%.1f GB", Double(value) / 1_073_741_824)
    }
}
