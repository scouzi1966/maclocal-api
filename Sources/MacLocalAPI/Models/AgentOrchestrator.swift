import Foundation
import ArgumentParser

final class AgentOrchestrator {
    private let executablePath: String
    private let primaryModelID: String
    private let agentModelID: String
    private let maxSteps: Int
    private let skillsDirectory: String
    private let allowedHTTPHosts: Set<String>
    private let temperature: Double?
    private let topP: Double?
    private let maxTokens: Int?
    private let repetitionPenalty: Double?
    private let verbose: Bool

    private let agentService: MLXModelService
    private let skillCatalog: AgentSkillCatalog

    init(
        executablePath: String,
        primaryModelID: String,
        agentModelID: String,
        maxSteps: Int,
        skillsDirectory: String,
        allowedHTTPHosts: Set<String>,
        temperature: Double?,
        topP: Double?,
        maxTokens: Int?,
        repetitionPenalty: Double?,
        verbose: Bool
    ) {
        self.executablePath = executablePath
        self.primaryModelID = primaryModelID
        self.agentModelID = agentModelID
        self.maxSteps = maxSteps
        self.skillsDirectory = skillsDirectory
        self.allowedHTTPHosts = allowedHTTPHosts
        self.temperature = temperature
        self.topP = topP
        self.maxTokens = maxTokens
        self.repetitionPenalty = repetitionPenalty
        self.verbose = verbose

        let resolver = MLXCacheResolver()
        self.agentService = MLXModelService(resolver: resolver)
        self.skillCatalog = AgentSkillCatalog(rootDirectory: skillsDirectory)
    }

    func run(prompt: String, instructions: String) async throws -> String {
        defer {
            Task {
                await agentService.shutdownAndReleaseResources(verbose: verbose)
            }
        }

        try await validateAgentCapabilities()
        let skills = try skillCatalog.listSkills()

        var history: [String] = []
        var lastAgentContent = ""

        for step in 1...maxSteps {
            let agentMessages = buildAgentMessages(
                prompt: prompt,
                instructions: instructions,
                skills: skills,
                history: history,
                step: step
            )

            let result = try await agentService.generate(
                model: agentModelID,
                messages: agentMessages,
                temperature: temperature,
                maxTokens: maxTokens,
                topP: topP,
                repetitionPenalty: repetitionPenalty,
                tools: buildTools(),
                responseFormat: nil
            )

            let content = result.content.trimmingCharacters(in: .whitespacesAndNewlines)
            if !content.isEmpty {
                lastAgentContent = content
            }

            if let toolCalls = result.toolCalls, !toolCalls.isEmpty {
                for toolCall in toolCalls {
                    let args = parseArguments(toolCall.function.arguments)
                    let toolResult = await executeToolCall(name: toolCall.function.name, args: args)
                    history.append("TOOL \(toolCall.function.name) ARGS: \(toolCall.function.arguments)\nRESULT: \(toolResult)")
                }
                if history.count > 24 {
                    history = Array(history.suffix(24))
                }
                continue
            }

            if !content.isEmpty {
                return content
            }

            history.append("AGENT_NOTE: Empty response without tool calls.")
        }

        if !lastAgentContent.isEmpty {
            return lastAgentContent
        }
        throw ValidationError("Agent reached --agent-max-steps=\(maxSteps) without producing a final answer")
    }

    private func validateAgentCapabilities() async throws {
        let schema: [String: Any] = [
            "type": "object",
            "properties": [
                "status": ["type": "string"]
            ],
            "required": ["status"],
            "additionalProperties": false
        ]

        let guided = try await agentService.generate(
            model: agentModelID,
            messages: [
                Message(role: "system", content: "Return JSON only."),
                Message(role: "user", content: "Return JSON object with status=ok")
            ],
            temperature: 0.0,
            maxTokens: 64,
            topP: 1.0,
            repetitionPenalty: nil,
            responseFormat: ResponseFormat(
                type: "json_schema",
                jsonSchema: ResponseJsonSchema(
                    name: "capability_probe",
                    description: nil,
                    schema: AnyCodable(schema),
                    strict: true
                )
            )
        )

        guard let data = guided.content.data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              json["status"] != nil else {
            throw ValidationError("Agent model \(agentModelID) failed guided-json capability probe")
        }

        let probeTools = [
            RequestTool(
                type: "function",
                function: RequestToolFunction(
                    name: "probe_tool",
                    description: "Capability probe tool",
                    parameters: AnyCodable([
                        "type": "object",
                        "properties": ["text": ["type": "string"]],
                        "required": ["text"],
                        "additionalProperties": false
                    ])
                )
            )
        ]

        let toolProbe = try await agentService.generate(
            model: agentModelID,
            messages: [
                Message(role: "system", content: "Use tools when asked."),
                Message(role: "user", content: "Call probe_tool with text='ok'.")
            ],
            temperature: 0.0,
            maxTokens: 128,
            topP: 1.0,
            repetitionPenalty: nil,
            tools: probeTools,
            responseFormat: nil
        )

        guard let toolCalls = toolProbe.toolCalls, !toolCalls.isEmpty else {
            throw ValidationError("Agent model \(agentModelID) failed tool-calling capability probe")
        }
    }

    private func buildAgentMessages(
        prompt: String,
        instructions: String,
        skills: [AgentSkillSummary],
        history: [String],
        step: Int
    ) -> [Message] {
        let skillList: String
        if skills.isEmpty {
            skillList = "No local skills discovered at \(skillsDirectory)."
        } else {
            skillList = skills.map { "- \($0.name): \($0.description)" }.joined(separator: "\n")
        }

        let historyText = history.isEmpty ? "(none)" : history.joined(separator: "\n\n")
        let httpScope = allowedHTTPHosts.isEmpty ? "disabled" : allowedHTTPHosts.sorted().joined(separator: ", ")

        let system = """
You are AFM's agent intelligence layer.

Follow these rules:
1. Use tools when needed to gather facts, execute skills, call external APIs, or delegate to the primary model.
2. When you have enough information, answer directly for the user.
3. Keep answers concise and factual.
4. Never invent tool outputs.
"""

        let user = """
Step \(step) of max \(maxSteps).

Assistant instructions:
\(instructions)

Primary model:
\(primaryModelID)

Available skills:
\(skillList)

External HTTP allowlist:
\(httpScope)

User request:
\(prompt)

Prior tool history:
\(historyText)
"""

        return [
            Message(role: "system", content: system),
            Message(role: "user", content: user)
        ]
    }

    private func buildTools() -> [RequestTool] {
        [
            RequestTool(
                type: "function",
                function: RequestToolFunction(
                    name: "call_primary_model",
                    description: "Call the primary -m model via afm mlx subprocess and return its text output.",
                    parameters: AnyCodable([
                        "type": "object",
                        "properties": [
                            "prompt": ["type": "string"],
                            "instructions": ["type": "string"],
                            "max_tokens": ["type": "integer"]
                        ],
                        "required": ["prompt"],
                        "additionalProperties": false
                    ])
                )
            ),
            RequestTool(
                type: "function",
                function: RequestToolFunction(
                    name: "list_skills",
                    description: "List local skills discovered from SKILL.md files.",
                    parameters: AnyCodable([
                        "type": "object",
                        "properties": [String: Any](),
                        "additionalProperties": false
                    ])
                )
            ),
            RequestTool(
                type: "function",
                function: RequestToolFunction(
                    name: "read_skill",
                    description: "Read the SKILL.md content for a named local skill.",
                    parameters: AnyCodable([
                        "type": "object",
                        "properties": [
                            "name": ["type": "string"]
                        ],
                        "required": ["name"],
                        "additionalProperties": false
                    ])
                )
            ),
            RequestTool(
                type: "function",
                function: RequestToolFunction(
                    name: "http_request",
                    description: "Perform an allowlisted HTTP request to an external API.",
                    parameters: AnyCodable([
                        "type": "object",
                        "properties": [
                            "url": ["type": "string"],
                            "method": ["type": "string"],
                            "headers": ["type": "object"],
                            "body": ["type": "string"]
                        ],
                        "required": ["url", "method"],
                        "additionalProperties": false
                    ])
                )
            )
        ]
    }

    private func parseArguments(_ json: String) -> [String: Any] {
        guard let data = json.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return [:]
        }
        return obj
    }

    private func encodeJSON(_ object: Any) -> String {
        guard JSONSerialization.isValidJSONObject(object),
              let data = try? JSONSerialization.data(withJSONObject: object, options: [.sortedKeys]),
              let text = String(data: data, encoding: .utf8) else {
            return "{}"
        }
        return text
    }

    private func executeToolCall(name: String, args: [String: Any]) async -> String {
        do {
            switch name {
            case "call_primary_model":
                guard let prompt = args["prompt"] as? String,
                      !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                    return encodeJSON(["ok": false, "error": "prompt is required"])
                }
                let instructions = args["instructions"] as? String
                let maxTokens = args["max_tokens"] as? Int
                let output = try runPrimaryModel(prompt: prompt, instructions: instructions, maxTokens: maxTokens)
                return encodeJSON(["ok": true, "output": output])

            case "list_skills":
                let skills = try skillCatalog.listSkills()
                return encodeJSON([
                    "ok": true,
                    "skills": skills.map { ["name": $0.name, "description": $0.description] }
                ])

            case "read_skill":
                guard let name = args["name"] as? String, !name.isEmpty else {
                    return encodeJSON(["ok": false, "error": "name is required"])
                }
                let content = try skillCatalog.readSkill(named: name)
                return encodeJSON(["ok": true, "name": name, "content": content])

            case "http_request":
                return try await runHTTPRequest(args: args)

            default:
                return encodeJSON(["ok": false, "error": "Unknown tool: \(name)"])
            }
        } catch {
            return encodeJSON(["ok": false, "error": error.localizedDescription])
        }
    }

    private func runPrimaryModel(prompt: String, instructions: String?, maxTokens: Int?) throws -> String {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: executablePath)

        var args = ["mlx", "-m", primaryModelID, "-s", prompt]
        if let instructions, !instructions.isEmpty {
            args += ["-i", instructions]
        }
        if let maxTokens, maxTokens > 0 {
            args += ["--max-tokens", String(maxTokens)]
        }

        process.arguments = args
        let stdoutPipe = Pipe()
        let stderrPipe = Pipe()
        process.standardOutput = stdoutPipe
        process.standardError = stderrPipe

        try process.run()

        let timeoutSeconds: TimeInterval = 180
        let deadline = Date().addingTimeInterval(timeoutSeconds)
        while process.isRunning {
            if Date() > deadline {
                process.terminate()
                throw ValidationError("Primary model subprocess timed out")
            }
            Thread.sleep(forTimeInterval: 0.1)
        }

        let stdoutData = stdoutPipe.fileHandleForReading.readDataToEndOfFile()
        let stderrData = stderrPipe.fileHandleForReading.readDataToEndOfFile()
        let stdout = String(data: stdoutData, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        let stderr = String(data: stderrData, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""

        guard process.terminationStatus == 0 else {
            throw ValidationError("Primary model subprocess failed: \(stderr.isEmpty ? "exit \(process.terminationStatus)" : stderr)")
        }

        return stdout
    }

    private func runHTTPRequest(args: [String: Any]) async throws -> String {
        guard let rawURL = args["url"] as? String,
              let url = URL(string: rawURL),
              let methodRaw = args["method"] as? String else {
            return encodeJSON(["ok": false, "error": "url and method are required"])
        }

        let method = methodRaw.uppercased()
        guard method == "GET" || method == "POST" else {
            return encodeJSON(["ok": false, "error": "Only GET and POST are allowed in v1"])
        }

        guard let host = url.host?.lowercased() else {
            return encodeJSON(["ok": false, "error": "URL host is required"])
        }

        if allowedHTTPHosts.isEmpty {
            return encodeJSON(["ok": false, "error": "External HTTP tool is disabled; set --agent-allow-http"])
        }

        if !allowedHTTPHosts.contains(host) {
            return encodeJSON(["ok": false, "error": "Host not allowlisted: \(host)"])
        }

        var request = URLRequest(url: url)
        request.httpMethod = method
        request.timeoutInterval = 20

        if let headers = args["headers"] as? [String: Any] {
            for (k, v) in headers {
                request.setValue(String(describing: v), forHTTPHeaderField: k)
            }
        }

        if let body = args["body"] as? String {
            request.httpBody = body.data(using: .utf8)
        }

        let (data, response) = try await URLSession.shared.data(for: request)
        let statusCode = (response as? HTTPURLResponse)?.statusCode ?? 0
        let body = String(data: data.prefix(100_000), encoding: .utf8) ?? ""
        return encodeJSON(["ok": true, "status": statusCode, "body": body])
    }
}

struct AgentSkillSummary {
    let name: String
    let description: String
    let path: String
}

final class AgentSkillCatalog {
    private let rootDirectory: String

    init(rootDirectory: String) {
        self.rootDirectory = rootDirectory
    }

    func listSkills() throws -> [AgentSkillSummary] {
        let fm = FileManager.default
        guard fm.fileExists(atPath: rootDirectory) else {
            return []
        }

        let entries = try fm.contentsOfDirectory(at: URL(fileURLWithPath: rootDirectory), includingPropertiesForKeys: [.isDirectoryKey], options: [.skipsHiddenFiles])
        var out: [AgentSkillSummary] = []

        for entry in entries {
            let skillFile = entry.appendingPathComponent("SKILL.md")
            guard fm.fileExists(atPath: skillFile.path) else { continue }
            let content = (try? String(contentsOf: skillFile, encoding: .utf8)) ?? ""
            let parsed = parseFrontMatter(content)
            let name = parsed["name"] ?? entry.lastPathComponent
            let description = parsed["description"] ?? "No description"
            out.append(AgentSkillSummary(name: name, description: description, path: skillFile.path))
        }

        return out.sorted { $0.name < $1.name }
    }

    func readSkill(named name: String) throws -> String {
        let skills = try listSkills()
        guard let skill = skills.first(where: { $0.name == name || URL(fileURLWithPath: $0.path).deletingLastPathComponent().lastPathComponent == name }) else {
            throw ValidationError("Skill not found: \(name)")
        }

        let content = try String(contentsOfFile: skill.path, encoding: .utf8)
        let limit = 20_000
        if content.count <= limit {
            return content
        }
        return String(content.prefix(limit)) + "\n\n[truncated]"
    }

    private func parseFrontMatter(_ content: String) -> [String: String] {
        let lines = content.split(separator: "\n", omittingEmptySubsequences: false).map(String.init)
        guard lines.first == "---" else { return [:] }

        var index = 1
        var result: [String: String] = [:]
        while index < lines.count {
            let line = lines[index]
            if line == "---" { break }
            if let colon = line.firstIndex(of: ":") {
                let key = String(line[..<colon]).trimmingCharacters(in: .whitespacesAndNewlines)
                let value = String(line[line.index(after: colon)...]).trimmingCharacters(in: .whitespacesAndNewlines)
                if !key.isEmpty {
                    result[key] = value
                }
            }
            index += 1
        }
        return result
    }
}
