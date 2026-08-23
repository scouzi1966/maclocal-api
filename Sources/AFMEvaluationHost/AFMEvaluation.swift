import Foundation
@_exported import AFMEvalKit

public struct AFMEvaluationSuiteDescriptor: Sendable {
    public enum Origin: String, Sendable { case bundled, custom }
    public let name: String
    public let description: String
    public let caseCount: Int
    public let origin: Origin
    public let url: URL
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
        let requestsRun = evaluate || bench || !suites.isEmpty
        if managementCount > 0, requestsRun || noOpen {
            throw AFMEvaluationError.conflictingCLI(
                "Evaluation management options cannot be combined with --eval, --bench, --eval-suite, or --no-open.")
        }
        if list { return .list }
        if let scaffold { return .scaffold(name: scaffold) }
        if let validate { return .validate(reference: validate) }
        if requestsRun {
            let selected = suites.isEmpty ? ["comprehensive"] : suites
            return .run(suites: selected, openReport: !noOpen)
        }
        if noOpen {
            throw AFMEvaluationError.conflictingCLI("--no-open is only valid with --eval or --bench.")
        }
        return .none
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
        try AFMEvaluationValidator.validateSafeName(name, field: "suite name")
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

    public func decode(
        url: URL,
        origin: AFMEvaluationSuiteDescriptor.Origin
    ) throws -> AFMEvaluationSuite {
        let data: Data
        do { data = try Data(contentsOf: url, options: [.mappedIfSafe]) }
        catch { throw AFMEvaluationError.invalidSuite("Cannot read \(url.path): \(error.localizedDescription)") }
        return try AFMEvaluationValidator.decode(
            data,
            allowsCrossCaseMatching: origin == .bundled,
            source: url.lastPathComponent)
    }

    public func scaffold(named name: String) throws -> URL {
        try AFMEvaluationValidator.validateSafeName(name, field: "suite name")
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

}
