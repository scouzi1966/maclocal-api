import Testing
import Foundation
@testable import MacLocalAPI

struct StrictGrammarWiringTests {
// dimensions: grammar_constraints=on, tool_call_format=xmlFunction

    // MARK: - Helpers

    private func makeTool(name: String, strict: Bool?) -> RequestTool {
        RequestTool(
            type: "function",
            function: RequestToolFunction(
                name: name,
                description: nil,
                parameters: nil,
                strict: strict
            )
        )
    }

    // MARK: - hasStrictTools()

    @Test func hasStrictTools_nilTools_returnsFalse() {
        #expect(MLXModelService.hasStrictTools(nil) == false)
    }

    @Test func hasStrictTools_emptyArray_returnsFalse() {
        #expect(MLXModelService.hasStrictTools([]) == false)
    }

    @Test func hasStrictTools_singleStrictTrue_returnsTrue() {
        let tools = [makeTool(name: "read_file", strict: true)]
        #expect(MLXModelService.hasStrictTools(tools) == true)
    }

    @Test func hasStrictTools_singleStrictFalse_returnsFalse() {
        let tools = [makeTool(name: "read_file", strict: false)]
        #expect(MLXModelService.hasStrictTools(tools) == false)
    }

    @Test func hasStrictTools_singleStrictNil_returnsFalse() {
        let tools = [makeTool(name: "read_file", strict: nil)]
        #expect(MLXModelService.hasStrictTools(tools) == false)
    }

    @Test func hasStrictTools_mixedStrictTrueAndNil_returnsTrue() {
        let tools = [
            makeTool(name: "read_file", strict: true),
            makeTool(name: "write_file", strict: nil),
        ]
        #expect(MLXModelService.hasStrictTools(tools) == true)
    }

    // MARK: - RequestToolFunction strict field decoding

    @Test func requestToolFunction_decodesStrictTrue() throws {
        let json = """
        {"name": "get_weather", "strict": true}
        """
        let decoded = try JSONDecoder().decode(
            RequestToolFunction.self,
            from: Data(json.utf8)
        )
        #expect(decoded.strict == true)
    }

    @Test func requestToolFunction_decodesStrictFalse() throws {
        let json = """
        {"name": "get_weather", "strict": false}
        """
        let decoded = try JSONDecoder().decode(
            RequestToolFunction.self,
            from: Data(json.utf8)
        )
        #expect(decoded.strict == false)
    }

    @Test func requestToolFunction_decodesStrictNilWhenAbsent() throws {
        let json = """
        {"name": "get_weather"}
        """
        let decoded = try JSONDecoder().decode(
            RequestToolFunction.self,
            from: Data(json.utf8)
        )
        #expect(decoded.strict == nil)
    }

    @Test func requestTool_fullArrayWithMixedStrict() throws {
        let json = """
        [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file",
                    "strict": true
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Write a file",
                    "strict": false
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "list_dir"
                }
            }
        ]
        """
        let tools = try JSONDecoder().decode(
            [RequestTool].self,
            from: Data(json.utf8)
        )
        #expect(tools.count == 3)
        #expect(tools[0].function.strict == true)
        #expect(tools[1].function.strict == false)
        #expect(tools[2].function.strict == nil)
    }

    // MARK: - ResponseJsonSchema strict field decoding

    @Test func responseJsonSchema_decodesStrictTrue() throws {
        let json = """
        {"name": "my_schema", "strict": true}
        """
        let decoded = try JSONDecoder().decode(
            ResponseJsonSchema.self,
            from: Data(json.utf8)
        )
        #expect(decoded.strict == true)
    }

    @Test func responseJsonSchema_decodesStrictFalse() throws {
        let json = """
        {"name": "my_schema", "strict": false}
        """
        let decoded = try JSONDecoder().decode(
            ResponseJsonSchema.self,
            from: Data(json.utf8)
        )
        #expect(decoded.strict == false)
    }

    @Test func responseJsonSchema_decodesStrictNilWhenAbsent() throws {
        let json = """
        {"name": "my_schema"}
        """
        let decoded = try JSONDecoder().decode(
            ResponseJsonSchema.self,
            from: Data(json.utf8)
        )
        #expect(decoded.strict == nil)
    }

    // MARK: - ResponseFormat json_schema strict detection

    @Test func responseFormat_jsonSchemaWithStrictTrue_detected() throws {
        let json = """
        {
            "type": "json_schema",
            "json_schema": {
                "name": "output",
                "strict": true
            }
        }
        """
        let format = try JSONDecoder().decode(
            ResponseFormat.self,
            from: Data(json.utf8)
        )
        #expect(format.type == "json_schema")
        #expect(format.jsonSchema?.strict == true)
    }

    @Test func responseFormat_jsonSchemaWithStrictFalse_notDetected() throws {
        let json = """
        {
            "type": "json_schema",
            "json_schema": {
                "name": "output",
                "strict": false
            }
        }
        """
        let format = try JSONDecoder().decode(
            ResponseFormat.self,
            from: Data(json.utf8)
        )
        #expect(format.type == "json_schema")
        #expect(format.jsonSchema?.strict == false)
    }

    @Test func responseFormat_jsonObjectType_noJsonSchema() throws {
        let json = """
        {"type": "json_object"}
        """
        let format = try JSONDecoder().decode(
            ResponseFormat.self,
            from: Data(json.utf8)
        )
        #expect(format.type == "json_object")
        #expect(format.jsonSchema == nil)
    }
}
