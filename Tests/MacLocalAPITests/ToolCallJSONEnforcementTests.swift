import Foundation
import Testing

@testable import MacLocalAPI

struct ToolCallJSONEnforcementTests {

    @Test("buildEnforcedToolCallSchema constrains selected named tool")
    func schemaConstrainsNamedTool() throws {
        let tools = [
            makeRequestTool(name: "search", schema: [
                "type": "object",
                "properties": [
                    "query": ["type": "string"]
                ],
                "required": ["query"]
            ]),
            makeRequestTool(name: "open", schema: [
                "type": "object",
                "properties": [
                    "path": ["type": "string"]
                ],
                "required": ["path"]
            ])
        ]

        let schema = try MLXModelService.buildEnforcedToolCallSchema(tools: tools, selectedToolName: "open")
        let branches = schema["oneOf"] as? [[String: Any]]
        let single = branches?.first
        let properties = single?["properties"] as? [String: Any]
        let nameSchema = properties?["name"] as? [String: Any]
        let enumValues = nameSchema?["enum"] as? [String]

        #expect(enumValues == ["open"])
    }

    @Test("parseEnforcedToolCallResponses parses single valid call")
    func parsesSingleValidCall() throws {
        let tools = [
            makeRequestTool(name: "search", schema: [
                "type": "object",
                "properties": [
                    "query": ["type": "string"],
                    "limit": ["type": "integer"]
                ],
                "required": ["query"]
            ])
        ]

        let calls = try MLXModelService.parseEnforcedToolCallResponses(
            from: #"{"name":"search","arguments":{"limit":3,"query":"swift testing"}}"#,
            tools: tools,
            toolChoice: nil
        )

        #expect(calls.count == 1)
        #expect(calls[0].function.name == "search")
        let args = parseArgs(calls[0].function.arguments)
        #expect(args?["query"] as? String == "swift testing")
        #expect(args?["limit"] as? Int == 3)
    }

    @Test("parseEnforcedToolCallResponses parses multiple calls")
    func parsesMultipleCalls() throws {
        let tools = [
            makeRequestTool(name: "search", schema: [
                "type": "object",
                "properties": ["query": ["type": "string"]],
                "required": ["query"]
            ]),
            makeRequestTool(name: "open", schema: [
                "type": "object",
                "properties": ["path": ["type": "string"]],
                "required": ["path"]
            ])
        ]

        let payload = #"[{"name":"search","arguments":{"query":"swift"}},{"name":"open","arguments":{"path":"README.md"}}]"#
        let calls = try MLXModelService.parseEnforcedToolCallResponses(from: payload, tools: tools, toolChoice: nil)

        #expect(calls.count == 2)
        #expect(calls.map(\.function.name) == ["search", "open"])
    }

    @Test("parseEnforcedToolCallResponses rejects unknown tool names")
    func rejectsUnknownToolName() {
        let tools = [
            makeRequestTool(name: "search", schema: [
                "type": "object",
                "properties": ["query": ["type": "string"]],
                "required": ["query"]
            ])
        ]

        #expect(throws: MLXServiceError.self) {
            _ = try MLXModelService.parseEnforcedToolCallResponses(
                from: #"{"name":"delete","arguments":{"path":"/tmp/x"}}"#,
                tools: tools,
                toolChoice: nil
            )
        }
    }

    @Test("parseEnforcedToolCallResponses rejects schema mismatches")
    func rejectsSchemaMismatch() {
        let tools = [
            makeRequestTool(name: "search", schema: [
                "type": "object",
                "properties": [
                    "query": ["type": "string"],
                    "options": [
                        "type": "object",
                        "properties": [
                            "caseSensitive": ["type": "boolean"]
                        ],
                        "required": ["caseSensitive"],
                        "additionalProperties": false
                    ]
                ],
                "required": ["query", "options"],
                "additionalProperties": false
            ])
        ]

        #expect(throws: MLXServiceError.self) {
            _ = try MLXModelService.parseEnforcedToolCallResponses(
                from: #"{"name":"search","arguments":{"query":"swift","options":{"caseSensitive":"yes"}}}"#,
                tools: tools,
                toolChoice: nil
            )
        }
    }

    @Test("parseEnforcedToolCallResponses rejects plain text fallback output")
    func rejectsPlainTextFallbackOutput() {
        let tools = [
            makeRequestTool(name: "search", schema: [
                "type": "object",
                "properties": ["query": ["type": "string"]],
                "required": ["query"]
            ])
        ]

        #expect(throws: MLXServiceError.self) {
            _ = try MLXModelService.parseEnforcedToolCallResponses(
                from: "I'll use the search tool for that.",
                tools: tools,
                toolChoice: nil
            )
        }
    }

    @Test("parseEnforcedToolCallResponses rejects stringified array arguments")
    func rejectsStringifiedArrayArguments() {
        let tools = [
            makeRequestTool(name: "add_tags", schema: [
                "type": "object",
                "properties": [
                    "item_id": ["type": "string"],
                    "tags": ["type": "array", "items": ["type": "string"]]
                ],
                "required": ["item_id", "tags"],
                "additionalProperties": false
            ])
        ]

        #expect(throws: MLXServiceError.self) {
            _ = try MLXModelService.parseEnforcedToolCallResponses(
                from: #"{"name":"add_tags","arguments":{"item_id":"TASK-456","tags":"[\"review\",\"pending\"]"}}"#,
                tools: tools,
                toolChoice: nil
            )
        }
    }

    @Test("legacy fallback extracts JSON-in-XML tool calls")
    func legacyFallbackExtractsJSONInXML() throws {
        let raw = """
        <tool_call>
        {"name":"get_weather","arguments":{"location":"Paris","unit":"celsius"}}
        </tool_call>
        """

        let (calls, remaining) = MLXModelService.extractToolCallsFallback(from: raw)

        #expect(calls.count == 1)
        #expect(calls[0].function.name == "get_weather")
        #expect(calls[0].function.arguments["location"] == .string("Paris"))
        #expect(calls[0].function.arguments["unit"] == .string("celsius"))
        #expect(remaining.isEmpty)
    }

    @Test("legacy coercion converts stringified JSON values to schema types")
    func legacyCoercionConvertsStringifiedJSONValues() throws {
        let tools = [
            makeRequestTool(name: "add_tags", schema: [
                "type": "object",
                "properties": [
                    "item_id": ["type": "string"],
                    "tags": ["type": "array", "items": ["type": "string"]],
                    "urgent": ["type": "boolean"]
                ],
                "required": ["item_id", "tags", "urgent"],
                "additionalProperties": false
            ])
        ]

        let malformed = ResponseToolCall(
            id: "call_test",
            type: "function",
            function: ResponseToolCallFunction(
                name: "add_tags",
                arguments: #"{"item_id":"TASK-456","tags":"[\"review\",\"pending\"]","urgent":"true"}"#
            )
        )

        let coerced = MLXModelService.coerceArgumentTypes(malformed, tools: tools)
        let args = parseArgs(coerced.function.arguments)

        #expect(args?["item_id"] as? String == "TASK-456")
        #expect(args?["tags"] as? [String] == ["review", "pending"])
        #expect(args?["urgent"] as? Bool == true)
    }

    private func makeRequestTool(name: String, schema: [String: Any]) -> RequestTool {
        let data = try! JSONSerialization.data(withJSONObject: schema)
        let anyCodable = try! JSONDecoder().decode(AnyCodable.self, from: data)
        return RequestTool(
            type: "function",
            function: RequestToolFunction(
                name: name,
                description: nil,
                parameters: anyCodable
            )
        )
    }

    private func parseArgs(_ json: String) -> [String: Any]? {
        guard let data = json.data(using: .utf8) else { return nil }
        return try? JSONSerialization.jsonObject(with: data) as? [String: Any]
    }
}
