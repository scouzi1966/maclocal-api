import Foundation
import MLXLMCommon
import Testing

@testable import MacLocalAPI

/// Comprehensive unit tests for XML tool call parsing pipeline.
/// Covers every code path in: decodeXMLEntities, parseXMLFunction (XMLParser + regex fallback),
/// parseXMLFunctionRegex, parseJSONToolCall, extractToolCallsFallback, coerceArgumentTypes,
/// and coerceStringValue.
struct XMLToolCallParsingTests {

    // ═══════════════════════════════════════════════════════════════════
    // MARK: - decodeXMLEntities
    // ═══════════════════════════════════════════════════════════════════

    @Test("decodeXMLEntities decodes all 5 standard entities")
    func decodesAllStandardEntities() {
        let input = "if x &lt; 10 &amp;&amp; y &gt; 0 &quot;hello&quot; &apos;world&apos;"
        let result = MLXModelService.decodeXMLEntities(input)
        #expect(result == #"if x < 10 && y > 0 "hello" 'world'"#)
    }

    @Test("decodeXMLEntities passes through strings without entities")
    func passthroughNoEntities() {
        let input = "just plain text with no entities"
        let result = MLXModelService.decodeXMLEntities(input)
        #expect(result == input)
    }

    @Test("decodeXMLEntities handles empty string")
    func decodesEmptyString() {
        #expect(MLXModelService.decodeXMLEntities("") == "")
    }

    @Test("decodeXMLEntities handles multiple &lt; in Python code")
    func decodesPythonComparisons() {
        let input = "if size &lt; 1024:\n    return f&quot;{size} B&quot;"
        let result = MLXModelService.decodeXMLEntities(input)
        #expect(result == "if size < 1024:\n    return f\"{size} B\"")
    }

    @Test("decodeXMLEntities handles double-encoded entities: &amp;lt; → &lt;")
    func decodesDoubleEncoded() {
        // &amp;lt; → &lt; (amp decoded first, lt entity remains as literal text)
        let input = "&amp;lt;"
        let result = MLXModelService.decodeXMLEntities(input)
        #expect(result == "&lt;")
    }

    @Test("decodeXMLEntities passes through bare & without entity suffix")
    func passthroughBareAmpersand() {
        // "AT&T" has & but no matching entity — & gets replaced by &amp; decode? No.
        // Our function only replaces known entities. Bare & without matching suffix stays.
        let input = "AT&T Corp"
        let result = MLXModelService.decodeXMLEntities(input)
        #expect(result == "AT&T Corp", "bare & should pass through unchanged")
    }

    @Test("decodeXMLEntities handles entities at string boundaries")
    func decodesEntitiesAtBoundaries() {
        #expect(MLXModelService.decodeXMLEntities("&lt;") == "<")
        #expect(MLXModelService.decodeXMLEntities("&gt;") == ">")
        #expect(MLXModelService.decodeXMLEntities("&amp;") == "&")
        #expect(MLXModelService.decodeXMLEntities("&lt;tag&gt;") == "<tag>")
    }

    @Test("decodeXMLEntities handles consecutive entities")
    func decodesConsecutiveEntities() {
        let input = "&lt;&gt;&amp;&quot;&apos;"
        let result = MLXModelService.decodeXMLEntities(input)
        #expect(result == "<>&\"'")
    }

    // ═══════════════════════════════════════════════════════════════════
    // MARK: - extractToolCallsFallback: XML format (exercises parseXMLFunction)
    // ═══════════════════════════════════════════════════════════════════

    @Test("parses basic XML tool call")
    func parsesBasicXML() {
        let text = """
        <tool_call>
        <function=get_weather>
        <parameter=city>Berlin</parameter>
        </function>
        </tool_call>
        """
        let (calls, remaining) = MLXModelService.extractToolCallsFallback(from: text)
        #expect(calls.count == 1)
        #expect(calls[0].function.name == "get_weather")
        let city = calls[0].function.arguments["city"]?.anyValue as? String
        #expect(city == "Berlin")
        #expect(remaining.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
    }

    @Test("decodes XML entities in parameter values via XMLParser")
    func decodesEntitiesInValues() {
        let text = """
        <tool_call>
        <function=write_file>
        <parameter=path>/tmp/test.py</parameter>
        <parameter=content>if x &lt; 10 &amp;&amp; y &gt; 0:
            print(&quot;ok&quot;)</parameter>
        </function>
        </tool_call>
        """
        let (calls, _) = MLXModelService.extractToolCallsFallback(from: text)
        #expect(calls.count == 1)
        let content = calls[0].function.arguments["content"]?.anyValue as? String
        #expect(content?.contains("< 10") == true, "< entity should be decoded")
        #expect(content?.contains("&&") == true, "& entity should be decoded")
        #expect(content?.contains("> 0") == true, "> entity should be decoded")
        #expect(content?.contains("\"ok\"") == true, "quot entity should be decoded")
        #expect(content?.contains("&lt;") == false, "raw entity should not remain")
    }

    @Test("handles multiline parameter values")
    func handlesMultilineValues() {
        let text = """
        <tool_call>
        <function=write_file>
        <parameter=path>/tmp/app.py</parameter>
        <parameter=content>
        def main():
            print("hello")
            return 0
        </parameter>
        </function>
        </tool_call>
        """
        let (calls, _) = MLXModelService.extractToolCallsFallback(from: text)
        #expect(calls.count == 1)
        let content = calls[0].function.arguments["content"]?.anyValue as? String
        #expect(content?.contains("def main():") == true)
        #expect(content?.contains("return 0") == true)
    }

    @Test("handles multiple tool calls")
    func handlesMultipleToolCalls() {
        let text = """
        <tool_call>
        <function=read_file>
        <parameter=path>/tmp/a.txt</parameter>
        </function>
        </tool_call>
        <tool_call>
        <function=read_file>
        <parameter=path>/tmp/b.txt</parameter>
        </function>
        </tool_call>
        """
        let (calls, _) = MLXModelService.extractToolCallsFallback(from: text)
        #expect(calls.count == 2)
        #expect(calls[0].function.name == "read_file")
        #expect(calls[1].function.name == "read_file")
        let path0 = calls[0].function.arguments["path"]?.anyValue as? String
        let path1 = calls[1].function.arguments["path"]?.anyValue as? String
        #expect(path0 == "/tmp/a.txt")
        #expect(path1 == "/tmp/b.txt")
    }

    @Test("preserves content outside tool calls")
    func preservesNonToolContent() {
        let text = "Here is the weather:\n<tool_call>\n<function=get_weather>\n<parameter=city>Tokyo</parameter>\n</function>\n</tool_call>"
        let (calls, remaining) = MLXModelService.extractToolCallsFallback(from: text)
        #expect(calls.count == 1)
        #expect(remaining.contains("Here is the weather:"))
    }

    @Test("handles duplicate parameters (keeps first non-empty)")
    func handlesDuplicateParams() {
        let text = """
        <tool_call>
        <function=write>
        <parameter=content></parameter>
        <parameter=content>real content here</parameter>
        </function>
        </tool_call>
        """
        let (calls, _) = MLXModelService.extractToolCallsFallback(from: text)
        #expect(calls.count == 1)
        let content = calls[0].function.arguments["content"]?.anyValue as? String
        #expect(content == "real content here")
    }

    @Test("preserves JSON array parameter values")
    func preservesJSONArrayValues() {
        let text = """
        <tool_call>
        <function=ask_questions>
        <parameter=questions>[{"text":"Color?","options":["red","blue"]}]</parameter>
        </function>
        </tool_call>
        """
        let (calls, _) = MLXModelService.extractToolCallsFallback(from: text)
        #expect(calls.count == 1)
        let qs = calls[0].function.arguments["questions"]?.anyValue
        #expect(qs is [Any], "questions should be parsed as array, not string")
    }

    @Test("preserves JSON object parameter values")
    func preservesJSONObjectValues() {
        let text = """
        <tool_call>
        <function=create_config>
        <parameter=settings>{"debug":true,"timeout":30}</parameter>
        </function>
        </tool_call>
        """
        let (calls, _) = MLXModelService.extractToolCallsFallback(from: text)
        #expect(calls.count == 1)
        let settings = calls[0].function.arguments["settings"]?.anyValue
        #expect(settings is [String: Any], "settings should be parsed as dict, not string")
    }

    @Test("skips empty parameter values")
    func skipsEmptyParameterValues() {
        let text = """
        <tool_call>
        <function=write>
        <parameter=path>/tmp/out.txt</parameter>
        <parameter=content></parameter>
        </function>
        </tool_call>
        """
        let (calls, _) = MLXModelService.extractToolCallsFallback(from: text)
        #expect(calls.count == 1)
        // content is empty, should be skipped; path should exist
        let path = calls[0].function.arguments["path"]?.anyValue as? String
        #expect(path == "/tmp/out.txt")
        // content may or may not exist depending on implementation; if it exists it's empty
        let content = calls[0].function.arguments["content"]?.anyValue as? String
        if let content {
            // If it exists, it should not be empty (our implementation skips empty)
            #expect(!content.isEmpty, "empty content should have been skipped")
        }
    }

    @Test("handles multiple parameters with mixed string and JSON types")
    func handlesMixedParameterTypes() {
        let text = """
        <tool_call>
        <function=create_task>
        <parameter=title>My Task</parameter>
        <parameter=tags>["urgent","backend"]</parameter>
        <parameter=config>{"priority":1}</parameter>
        </function>
        </tool_call>
        """
        let (calls, _) = MLXModelService.extractToolCallsFallback(from: text)
        #expect(calls.count == 1)
        let title = calls[0].function.arguments["title"]?.anyValue as? String
        #expect(title == "My Task")
        let tags = calls[0].function.arguments["tags"]?.anyValue
        #expect(tags is [Any], "tags should be parsed as JSON array")
        let config = calls[0].function.arguments["config"]?.anyValue
        #expect(config is [String: Any], "config should be parsed as JSON object")
    }

    @Test("returns nil for content without <function=...> tag")
    func noFunctionTagReturnsEmpty() {
        let text = """
        <tool_call>
        just some random text without function tags
        </tool_call>
        """
        let (calls, _) = MLXModelService.extractToolCallsFallback(from: text)
        #expect(calls.count == 0, "no <function=...> tag means no tool calls parsed")
    }

    // ═══════════════════════════════════════════════════════════════════
    // MARK: - XMLParser failure → regex fallback
    // ═══════════════════════════════════════════════════════════════════

    @Test("regex fallback handles bare < in parameter value")
    func regexFallbackBareAngleBracket() {
        // Bare < (not entity-encoded) breaks XMLParser. Regex fallback should handle.
        let text = """
        <tool_call>
        <function=write>
        <parameter=content>if x < 10:
            pass</parameter>
        </function>
        </tool_call>
        """
        let (calls, _) = MLXModelService.extractToolCallsFallback(from: text)
        #expect(calls.count == 1)
        let content = calls[0].function.arguments["content"]?.anyValue as? String
        #expect(content?.contains("x < 10") == true, "bare < should be preserved by regex fallback")
    }

    @Test("regex fallback salvages unclosed parameter (max_tokens truncation)")
    func regexFallbackUnclosedParameter() {
        // Model hit max_tokens mid-content — no closing </parameter>.
        // Single param that's unclosed. Bare < forces XMLParser failure → regex fallback.
        let text = """
        <tool_call>
        <function=write_file>
        <parameter=content>if x < 10:
            print("world")
        </function>
        </tool_call>
        """
        let (calls, _) = MLXModelService.extractToolCallsFallback(from: text)
        #expect(calls.count == 1)
        let content = calls[0].function.arguments["content"]?.anyValue as? String
        #expect(content?.contains("x < 10") == true, "unclosed param should be salvaged via regex fallback")
    }

    @Test("XMLParser path drops unclosed parameter (no bare < to trigger fallback)")
    func xmlParserUnclosedParamDropped() {
        // Without bare <, XMLParser succeeds but never sees </parameter> for content.
        // Content param is lost — this documents current behavior.
        let text = """
        <tool_call>
        <function=write_file>
        <parameter=path>/tmp/code.py</parameter>
        <parameter=content>def hello():
            print("world")
        </function>
        </tool_call>
        """
        let (calls, _) = MLXModelService.extractToolCallsFallback(from: text)
        #expect(calls.count == 1)
        let path = calls[0].function.arguments["path"]?.anyValue as? String
        #expect(path == "/tmp/code.py")
        // Content is lost because XMLParser doesn't salvage unclosed params
        // This documents the limitation — only regex fallback salvages.
    }

    // ═══════════════════════════════════════════════════════════════════
    // MARK: - extractToolCallsFallback: JSON-in-XML format
    // ═══════════════════════════════════════════════════════════════════

    @Test("parses JSON-in-XML format with 'arguments' key")
    func parsesJSONInXMLArguments() {
        let text = """
        <tool_call>
        {"name":"get_weather","arguments":{"city":"Paris"}}
        </tool_call>
        """
        let (calls, _) = MLXModelService.extractToolCallsFallback(from: text)
        #expect(calls.count == 1)
        #expect(calls[0].function.name == "get_weather")
    }

    @Test("parses JSON-in-XML format with 'parameters' key")
    func parsesJSONInXMLParameters() {
        let text = """
        <tool_call>
        {"name":"write_file","parameters":{"path":"/tmp/a.txt","content":"hello"}}
        </tool_call>
        """
        let (calls, _) = MLXModelService.extractToolCallsFallback(from: text)
        #expect(calls.count == 1)
        #expect(calls[0].function.name == "write_file")
        let path = calls[0].function.arguments["path"]?.anyValue as? String
        #expect(path == "/tmp/a.txt")
    }

    @Test("JSON-in-XML with no arguments key still parses name")
    func parsesJSONInXMLNameOnly() {
        let text = """
        <tool_call>
        {"name":"list_files"}
        </tool_call>
        """
        let (calls, _) = MLXModelService.extractToolCallsFallback(from: text)
        #expect(calls.count == 1)
        #expect(calls[0].function.name == "list_files")
        #expect(calls[0].function.arguments.isEmpty)
    }

    @Test("JSON-in-XML with missing name returns no tool calls")
    func jsonInXMLMissingName() {
        let text = """
        <tool_call>
        {"arguments":{"city":"Paris"}}
        </tool_call>
        """
        let (calls, _) = MLXModelService.extractToolCallsFallback(from: text)
        #expect(calls.count == 0, "missing name field should not parse")
    }

    @Test("JSON-in-XML with invalid JSON falls through")
    func jsonInXMLInvalidJSON() {
        let text = """
        <tool_call>
        {not valid json at all}
        </tool_call>
        """
        let (calls, _) = MLXModelService.extractToolCallsFallback(from: text)
        #expect(calls.count == 0)
    }

    // ═══════════════════════════════════════════════════════════════════
    // MARK: - extractToolCallsFallback: Mistral [TOOL_CALLS] format
    // ═══════════════════════════════════════════════════════════════════

    @Test("parses Mistral [TOOL_CALLS] JSON array format")
    func parsesMistralJSONArray() {
        let text = #"[TOOL_CALLS][{"name":"get_weather","arguments":{"city":"London"}}]"#
        let (calls, remaining) = MLXModelService.extractToolCallsFallback(from: text)
        #expect(calls.count == 1)
        #expect(calls[0].function.name == "get_weather")
        #expect(remaining.isEmpty)
    }

    @Test("parses Mistral [TOOL_CALLS] with multiple items in array")
    func parsesMistralMultipleItems() {
        let text = #"[TOOL_CALLS][{"name":"read","arguments":{"path":"a.txt"}},{"name":"read","arguments":{"path":"b.txt"}}]"#
        let (calls, _) = MLXModelService.extractToolCallsFallback(from: text)
        #expect(calls.count == 2)
        #expect(calls[0].function.name == "read")
        #expect(calls[1].function.name == "read")
    }

    @Test("parses Mistral [TOOL_CALLS] func_name[ARGS]{...} format")
    func parsesMistralArgsFormat() {
        let text = #"[TOOL_CALLS]get_weather[ARGS]{"city":"Tokyo"}"#
        let (calls, remaining) = MLXModelService.extractToolCallsFallback(from: text)
        #expect(calls.count == 1)
        #expect(calls[0].function.name == "get_weather")
        #expect(remaining.isEmpty)
    }

    // ═══════════════════════════════════════════════════════════════════
    // MARK: - extractToolCallsFallback: Bare JSON (no wrapper tags)
    // ═══════════════════════════════════════════════════════════════════

    @Test("parses bare JSON tool call without wrapper tags")
    func parsesBareJSON() {
        let text = #"{"name":"get_weather","arguments":{"city":"Tokyo"}}"#
        let (calls, remaining) = MLXModelService.extractToolCallsFallback(from: text)
        #expect(calls.count == 1)
        #expect(calls[0].function.name == "get_weather")
        #expect(remaining.isEmpty)
    }

    @Test("bare JSON with 'parameters' key instead of 'arguments'")
    func parsesBareJSONWithParameters() {
        let text = #"{"name":"write","parameters":{"content":"hello"}}"#
        let (calls, _) = MLXModelService.extractToolCallsFallback(from: text)
        #expect(calls.count == 1)
        #expect(calls[0].function.name == "write")
        let content = calls[0].function.arguments["content"]?.anyValue as? String
        #expect(content == "hello")
    }

    @Test("bare JSON that's not a tool call doesn't match")
    func bareJSONNotToolCall() {
        let text = #"{"message":"hello","status":200}"#
        let (calls, remaining) = MLXModelService.extractToolCallsFallback(from: text)
        #expect(calls.count == 0, "JSON without 'name' should not parse as tool call")
        #expect(!remaining.isEmpty)
    }

    // ═══════════════════════════════════════════════════════════════════
    // MARK: - extractToolCallsFallback: Edge cases and remaining text
    // ═══════════════════════════════════════════════════════════════════

    @Test("returns empty array and original text when no tool calls found")
    func noToolCallsReturnsOriginal() {
        let text = "This is just a normal response with no tool calls."
        let (calls, remaining) = MLXModelService.extractToolCallsFallback(from: text)
        #expect(calls.count == 0)
        #expect(remaining == text)
    }

    @Test("strips <think></think> tags from remaining text")
    func stripsEmptyThinkTags() {
        let text = "<think>  </think>Here is the answer."
        let (calls, remaining) = MLXModelService.extractToolCallsFallback(from: text)
        #expect(calls.count == 0)
        #expect(remaining.contains("Here is the answer."))
        #expect(!remaining.contains("<think>"))
    }

    @Test("handles mixed XML and JSON in separate <tool_call> blocks")
    func handlesMixedFormatsInBlocks() {
        let text = """
        <tool_call>
        <function=read_file>
        <parameter=path>/tmp/a.txt</parameter>
        </function>
        </tool_call>
        <tool_call>
        {"name":"write_file","arguments":{"path":"/tmp/b.txt","content":"data"}}
        </tool_call>
        """
        let (calls, _) = MLXModelService.extractToolCallsFallback(from: text)
        #expect(calls.count == 2)
        #expect(calls[0].function.name == "read_file")
        #expect(calls[1].function.name == "write_file")
    }

    @Test("empty <tool_call></tool_call> block returns no calls")
    func emptyToolCallBlock() {
        let text = "<tool_call>\n</tool_call>"
        let (calls, _) = MLXModelService.extractToolCallsFallback(from: text)
        #expect(calls.count == 0)
    }

    @Test("multiple parameters with same non-empty value keeps first")
    func duplicateNonEmptyKeepsFirst() {
        let text = """
        <tool_call>
        <function=write>
        <parameter=content>first value</parameter>
        <parameter=content>second value</parameter>
        </function>
        </tool_call>
        """
        let (calls, _) = MLXModelService.extractToolCallsFallback(from: text)
        #expect(calls.count == 1)
        let content = calls[0].function.arguments["content"]?.anyValue as? String
        #expect(content == "first value", "first non-empty value should be kept")
    }

    @Test("handles parameter values with special characters in Python code")
    func handlesSpecialCharsInPython() {
        let text = """
        <tool_call>
        <function=write_file>
        <parameter=path>/tmp/test.py</parameter>
        <parameter=content>sizes = [f&quot;{s} B&quot; for s in range(10) if s &gt; 0]
        result = {k: v for k, v in data.items() if v &amp; mask}</parameter>
        </function>
        </tool_call>
        """
        let (calls, _) = MLXModelService.extractToolCallsFallback(from: text)
        #expect(calls.count == 1)
        let content = calls[0].function.arguments["content"]?.anyValue as? String
        #expect(content?.contains("s > 0") == true)
        #expect(content?.contains("v & mask") == true)
        #expect(content?.contains("\"") == true)
    }

    // ═══════════════════════════════════════════════════════════════════
    // MARK: - coerceArgumentTypes
    // ═══════════════════════════════════════════════════════════════════

    @Test("coerces string to integer")
    func coercesStringToInt() {
        let tools = [makeRequestTool(name: "search", properties: [
            "max_results": ["type": "integer"]
        ])]
        let rtc = makeResponseToolCall(name: "search", arguments: #"{"max_results":"5"}"#)
        let result = MLXModelService.coerceArgumentTypes(rtc, tools: tools)
        let args = parseArgs(result.function.arguments)
        #expect(args?["max_results"] is Int)
        #expect(args?["max_results"] as? Int == 5)
    }

    @Test("coerces string to negative integer")
    func coercesStringToNegativeInt() {
        let tools = [makeRequestTool(name: "move", properties: [
            "offset": ["type": "integer"]
        ])]
        let rtc = makeResponseToolCall(name: "move", arguments: #"{"offset":"-3"}"#)
        let result = MLXModelService.coerceArgumentTypes(rtc, tools: tools)
        let args = parseArgs(result.function.arguments)
        #expect(args?["offset"] as? Int == -3)
    }

    @Test("coerces string to boolean true and false")
    func coercesStringToBool() {
        let tools = [makeRequestTool(name: "search", properties: [
            "case_sensitive": ["type": "boolean"],
            "verbose": ["type": "boolean"]
        ])]
        let rtc = makeResponseToolCall(name: "search", arguments: #"{"case_sensitive":"true","verbose":"false"}"#)
        let result = MLXModelService.coerceArgumentTypes(rtc, tools: tools)
        let args = parseArgs(result.function.arguments)
        #expect(args?["case_sensitive"] as? Bool == true)
        #expect(args?["verbose"] as? Bool == false)
    }

    @Test("coerces boolean case-insensitively (TRUE, True, FALSE)")
    func coercesBoolCaseInsensitive() {
        let tools = [makeRequestTool(name: "cfg", properties: [
            "enabled": ["type": "boolean"]
        ])]
        let rtc = makeResponseToolCall(name: "cfg", arguments: #"{"enabled":"TRUE"}"#)
        let result = MLXModelService.coerceArgumentTypes(rtc, tools: tools)
        let args = parseArgs(result.function.arguments)
        #expect(args?["enabled"] as? Bool == true)
    }

    @Test("does NOT coerce 'yes'/'no'/'1'/'0' to boolean")
    func doesNotCoerceInvalidBoolStrings() {
        let tools = [makeRequestTool(name: "cfg", properties: [
            "enabled": ["type": "boolean"]
        ])]
        // "yes" is not a valid boolean string per our implementation
        let rtc = makeResponseToolCall(name: "cfg", arguments: #"{"enabled":"yes"}"#)
        let result = MLXModelService.coerceArgumentTypes(rtc, tools: tools)
        // Should be unchanged since coercion failed
        #expect(result.function.arguments == rtc.function.arguments)
    }

    @Test("coerces string to number (float)")
    func coercesStringToNumber() {
        let tools = [makeRequestTool(name: "set_temp", properties: [
            "celsius": ["type": "number"]
        ])]
        let rtc = makeResponseToolCall(name: "set_temp", arguments: #"{"celsius":"22.5"}"#)
        let result = MLXModelService.coerceArgumentTypes(rtc, tools: tools)
        let args = parseArgs(result.function.arguments)
        let val = args?["celsius"]
        #expect(val is Double || val is Int, "celsius should be numeric")
    }

    @Test("coerces whole-number string to Int for 'number' type")
    func coercesWholeNumberToInt() {
        let tools = [makeRequestTool(name: "cfg", properties: [
            "timeout": ["type": "number"]
        ])]
        let rtc = makeResponseToolCall(name: "cfg", arguments: #"{"timeout":"30"}"#)
        let result = MLXModelService.coerceArgumentTypes(rtc, tools: tools)
        let args = parseArgs(result.function.arguments)
        // 30.0 == Double(30) so should return Int
        #expect(args?["timeout"] as? Int == 30)
    }

    @Test("coerces string to array")
    func coercesStringToArray() {
        let tools = [makeRequestTool(name: "ask", properties: [
            "questions": ["type": "array"]
        ])]
        let rtc = makeResponseToolCall(name: "ask", arguments: #"{"questions":"[\"a\",\"b\"]"}"#)
        let result = MLXModelService.coerceArgumentTypes(rtc, tools: tools)
        let args = parseArgs(result.function.arguments)
        #expect(args?["questions"] is [Any], "questions should be coerced to array")
    }

    @Test("coerces string to object")
    func coercesStringToObject() {
        let tools = [makeRequestTool(name: "cfg", properties: [
            "settings": ["type": "object"]
        ])]
        let rtc = makeResponseToolCall(name: "cfg", arguments: #"{"settings":"{\"debug\":true}"}"#)
        let result = MLXModelService.coerceArgumentTypes(rtc, tools: tools)
        let args = parseArgs(result.function.arguments)
        #expect(args?["settings"] is [String: Any], "settings should be coerced to object")
    }

    @Test("leaves already-correct types alone (string stays string)")
    func leavesCorrectTypesAlone() {
        let tools = [makeRequestTool(name: "search", properties: [
            "query": ["type": "string"]
        ])]
        let rtc = makeResponseToolCall(name: "search", arguments: #"{"query":"hello"}"#)
        let result = MLXModelService.coerceArgumentTypes(rtc, tools: tools)
        #expect(result.function.arguments == rtc.function.arguments)
    }

    @Test("returns unchanged when tools is nil")
    func coercionToolsNil() {
        let rtc = makeResponseToolCall(name: "search", arguments: #"{"max_results":"5"}"#)
        let result = MLXModelService.coerceArgumentTypes(rtc, tools: nil)
        #expect(result.function.arguments == rtc.function.arguments)
    }

    @Test("returns unchanged when tools is empty")
    func coercionToolsEmpty() {
        let rtc = makeResponseToolCall(name: "search", arguments: #"{"max_results":"5"}"#)
        let result = MLXModelService.coerceArgumentTypes(rtc, tools: [])
        #expect(result.function.arguments == rtc.function.arguments)
    }

    @Test("returns unchanged when tool name not found in tools list")
    func coercionToolNotFound() {
        let tools = [makeRequestTool(name: "other_tool", properties: [
            "x": ["type": "integer"]
        ])]
        let rtc = makeResponseToolCall(name: "search", arguments: #"{"max_results":"5"}"#)
        let result = MLXModelService.coerceArgumentTypes(rtc, tools: tools)
        #expect(result.function.arguments == rtc.function.arguments)
    }

    @Test("returns unchanged when arguments JSON is invalid")
    func coercionInvalidJSON() {
        let tools = [makeRequestTool(name: "search", properties: [
            "max_results": ["type": "integer"]
        ])]
        let rtc = makeResponseToolCall(name: "search", arguments: "not json")
        let result = MLXModelService.coerceArgumentTypes(rtc, tools: tools)
        #expect(result.function.arguments == "not json")
    }

    @Test("mixed coercion: some params coerced, some left alone")
    func mixedCoercion() {
        let tools = [makeRequestTool(name: "search", properties: [
            "query": ["type": "string"],
            "max_results": ["type": "integer"],
            "case_sensitive": ["type": "boolean"]
        ])]
        let rtc = makeResponseToolCall(name: "search", arguments: #"{"query":"test","max_results":"10","case_sensitive":"false"}"#)
        let result = MLXModelService.coerceArgumentTypes(rtc, tools: tools)
        let args = parseArgs(result.function.arguments)
        #expect(args?["query"] as? String == "test", "string should stay string")
        #expect(args?["max_results"] as? Int == 10, "should be coerced to int")
        #expect(args?["case_sensitive"] as? Bool == false, "should be coerced to bool")
    }

    @Test("does not coerce non-string values")
    func doesNotCoerceNonStrings() {
        let tools = [makeRequestTool(name: "search", properties: [
            "max_results": ["type": "integer"]
        ])]
        // Already an integer, not a string
        let rtc = makeResponseToolCall(name: "search", arguments: #"{"max_results":5}"#)
        let result = MLXModelService.coerceArgumentTypes(rtc, tools: tools)
        #expect(result.function.arguments == rtc.function.arguments)
    }

    @Test("does not coerce when property not in schema")
    func doesNotCoerceUnknownProperty() {
        let tools = [makeRequestTool(name: "search", properties: [
            "query": ["type": "string"]
        ])]
        // "limit" is not in schema
        let rtc = makeResponseToolCall(name: "search", arguments: #"{"query":"test","limit":"5"}"#)
        let result = MLXModelService.coerceArgumentTypes(rtc, tools: tools)
        // Only query is in schema (as string), limit is not — no changes
        #expect(result.function.arguments == rtc.function.arguments)
    }

    // ═══════════════════════════════════════════════════════════════════
    // MARK: - coerceArgumentTypes: missing required params default fill
    // ═══════════════════════════════════════════════════════════════════

    @Test("does NOT fill missing required string param (let client report missing)")
    func doesNotFillMissingRequiredString() {
        // After the NSXMLParser fix, we stopped filling missing strings with empty defaults.
        // Missing string params should be left absent so the client can report "expected string, received undefined".
        // The EBNF grammar's named required param rules now prevent this from happening in practice.
        let tools = [makeRequestTool(name: "bash", properties: [
            "command": ["type": "string"],
            "description": ["type": "string"]
        ], required: ["command", "description"])]
        // Model only emits "command", omits "description"
        let rtc = makeResponseToolCall(name: "bash", arguments: #"{"command":"ls -la"}"#)
        let result = MLXModelService.coerceArgumentTypes(rtc, tools: tools)
        let args = parseArgs(result.function.arguments)
        #expect(args?["command"] as? String == "ls -la")
        #expect(args?["description"] == nil, "missing required string should NOT be filled — let client report the error")
    }

    @Test("fills missing required boolean param with false")
    func fillsMissingRequiredBool() {
        let tools = [makeRequestTool(name: "cfg", properties: [
            "name": ["type": "string"],
            "verbose": ["type": "boolean"]
        ], required: ["name", "verbose"])]
        let rtc = makeResponseToolCall(name: "cfg", arguments: #"{"name":"test"}"#)
        let result = MLXModelService.coerceArgumentTypes(rtc, tools: tools)
        let args = parseArgs(result.function.arguments)
        #expect(args?["verbose"] as? Bool == false)
    }

    @Test("fills missing required integer param with 0")
    func fillsMissingRequiredInt() {
        let tools = [makeRequestTool(name: "search", properties: [
            "query": ["type": "string"],
            "limit": ["type": "integer"]
        ], required: ["query", "limit"])]
        let rtc = makeResponseToolCall(name: "search", arguments: #"{"query":"test"}"#)
        let result = MLXModelService.coerceArgumentTypes(rtc, tools: tools)
        let args = parseArgs(result.function.arguments)
        #expect(args?["limit"] as? Int == 0)
    }

    @Test("does not fill non-required missing params")
    func doesNotFillOptionalParams() {
        let tools = [makeRequestTool(name: "bash", properties: [
            "command": ["type": "string"],
            "description": ["type": "string"]
        ], required: ["command"])]  // only command is required
        let rtc = makeResponseToolCall(name: "bash", arguments: #"{"command":"ls"}"#)
        let result = MLXModelService.coerceArgumentTypes(rtc, tools: tools)
        let args = parseArgs(result.function.arguments)
        #expect(args?["description"] == nil, "optional missing param should NOT be filled")
    }

    // ═══════════════════════════════════════════════════════════════════
    // MARK: - coerceStringValue edge cases
    // ═══════════════════════════════════════════════════════════════════

    @Test("coerceStringValue returns nil for non-numeric string with integer type")
    func coerceStringValueInvalidInt() {
        #expect(MLXModelService.coerceStringValue("abc", schemaType: "integer") == nil)
    }

    @Test("coerceStringValue returns nil for float string with integer type")
    func coerceStringValueFloatToInt() {
        // "22.5" is not a valid integer
        #expect(MLXModelService.coerceStringValue("22.5", schemaType: "integer") == nil)
    }

    @Test("coerceStringValue handles negative numbers")
    func coerceStringValueNegativeNumber() {
        let result = MLXModelService.coerceStringValue("-3.14", schemaType: "number")
        #expect(result != nil)
        if let d = result as? Double {
            #expect(abs(d - (-3.14)) < 0.001)
        }
    }

    @Test("coerceStringValue returns nil for non-numeric string with number type")
    func coerceStringValueInvalidNumber() {
        #expect(MLXModelService.coerceStringValue("abc", schemaType: "number") == nil)
    }

    @Test("coerceStringValue returns nil for invalid JSON with array type")
    func coerceStringValueInvalidArray() {
        #expect(MLXModelService.coerceStringValue("not json", schemaType: "array") == nil)
    }

    @Test("coerceStringValue returns nil for invalid JSON with object type")
    func coerceStringValueInvalidObject() {
        #expect(MLXModelService.coerceStringValue("{broken", schemaType: "object") == nil)
    }

    @Test("coerceStringValue returns nil for unknown schema type")
    func coerceStringValueUnknownType() {
        #expect(MLXModelService.coerceStringValue("test", schemaType: "custom_type") == nil)
    }

    @Test("coerceStringValue coerces '0' to Int 0 for integer type")
    func coerceStringValueZero() {
        let result = MLXModelService.coerceStringValue("0", schemaType: "integer")
        #expect(result as? Int == 0)
    }

    @Test("coerceStringValue coerces '0.0' to Int 0 for number type (whole number)")
    func coerceStringValueZeroPointZero() {
        let result = MLXModelService.coerceStringValue("0.0", schemaType: "number")
        // 0.0 == Double(0) so should return Int(0)
        #expect(result as? Int == 0)
    }

    @Test("coerceStringValue handles empty string for all types")
    func coerceStringValueEmptyString() {
        #expect(MLXModelService.coerceStringValue("", schemaType: "integer") == nil)
        #expect(MLXModelService.coerceStringValue("", schemaType: "number") == nil)
        #expect(MLXModelService.coerceStringValue("", schemaType: "boolean") == nil)
        #expect(MLXModelService.coerceStringValue("", schemaType: "array") == nil)
        #expect(MLXModelService.coerceStringValue("", schemaType: "object") == nil)
    }

    // ═══════════════════════════════════════════════════════════════════
    // MARK: - Real-world scenarios (from OpenCode test reports)
    // ═══════════════════════════════════════════════════════════════════

    @Test("question tool: array of objects with typed fields (OpenCode pattern)")
    func questionToolArrayCoercion() {
        // This is the exact pattern that failed in OpenCode testing:
        // The `questions` param should be an array of objects, not a string
        let text = """
        <tool_call>
        <function=question>
        <parameter=questions>[{"text":"What is your name?","key":"name","type":"text"},{"text":"Pick a color","key":"color","type":"select","options":["red","blue","green"]}]</parameter>
        </function>
        </tool_call>
        """
        let (calls, _) = MLXModelService.extractToolCallsFallback(from: text)
        #expect(calls.count == 1)
        #expect(calls[0].function.name == "question")
        let qs = calls[0].function.arguments["questions"]?.anyValue
        #expect(qs is [Any], "questions should be array, not string")
        if let arr = qs as? [[String: Any]] {
            #expect(arr.count == 2)
            #expect(arr[0]["text"] as? String == "What is your name?")
        }
    }

    @Test("write file with Python containing comparison operators (OpenCode pattern)")
    func writeFileWithPythonComparisons() {
        // This is the pattern that caused infinite error loops in OpenCode:
        // Models encode < and > as &lt; &gt; in XML. Without decoding, Python gets broken.
        let text = """
        <tool_call>
        <function=write>
        <parameter=path>/tmp/convert.py</parameter>
        <parameter=content>def format_size(size: int) -&gt; str:
            if size &lt; 1024:
                return f&quot;{size} B&quot;
            elif size &lt; 1024 * 1024:
                return f&quot;{size / 1024:.1f} KB&quot;
            return f&quot;{size / 1024 / 1024:.1f} MB&quot;</parameter>
        </function>
        </tool_call>
        """
        let (calls, _) = MLXModelService.extractToolCallsFallback(from: text)
        #expect(calls.count == 1)
        let content = calls[0].function.arguments["content"]?.anyValue as? String
        #expect(content?.contains("if size < 1024:") == true, "< should be decoded")
        #expect(content?.contains("-> str:") == true, "> should be decoded")
        #expect(content?.contains("&lt;") == false, "entities should not remain")
        #expect(content?.contains("&gt;") == false, "entities should not remain")
        #expect(content?.contains("&quot;") == false, "entities should not remain")
    }

    @Test("bash tool with special characters")
    func bashToolSpecialChars() {
        let text = """
        <tool_call>
        <function=bash>
        <parameter=command>ls -la /tmp &amp;&amp; echo &quot;done&quot; | grep -v &apos;test&apos;</parameter>
        </function>
        </tool_call>
        """
        let (calls, _) = MLXModelService.extractToolCallsFallback(from: text)
        #expect(calls.count == 1)
        let cmd = calls[0].function.arguments["command"]?.anyValue as? String
        #expect(cmd?.contains("&&") == true)
        #expect(cmd?.contains("\"done\"") == true)
        #expect(cmd?.contains("'test'") == true)
    }

    @Test("JSON format switch mid-conversation (Qwen3-Coder-Next common error)")
    func jsonFormatSwitch() {
        // Model switches from XML to JSON inside <tool_call> tags
        let text = """
        <tool_call>
        {"name":"write","arguments":{"path":"/tmp/test.py","content":"print('hello')"}}
        </tool_call>
        """
        let (calls, _) = MLXModelService.extractToolCallsFallback(from: text)
        #expect(calls.count == 1)
        #expect(calls[0].function.name == "write")
        let content = calls[0].function.arguments["content"]?.anyValue as? String
        #expect(content == "print('hello')")
    }

    @Test("text content before and after tool call is preserved")
    func textAroundToolCall() {
        let text = """
        I'll create the file now.
        <tool_call>
        <function=write>
        <parameter=path>/tmp/out.txt</parameter>
        <parameter=content>hello world</parameter>
        </function>
        </tool_call>
        The file has been created.
        """
        let (calls, remaining) = MLXModelService.extractToolCallsFallback(from: text)
        #expect(calls.count == 1)
        #expect(remaining.contains("I'll create the file now."))
        #expect(remaining.contains("The file has been created."))
    }

    // ═══════════════════════════════════════════════════════════════════
    // MARK: - decodeJSONEscapes
    // ═══════════════════════════════════════════════════════════════════

    @Test("decodeJSONEscapes unescapes literal \\n and \\\" when no real newlines present")
    func decodesJSONEscapedEditParams() {
        // This is the exact pattern from the Qwen3-Coder edit bug:
        // Model emits oldString with literal \n and \" instead of real newlines and quotes
        let input = #"if category == \"Other\":\n                actions[\"skipped\"].append({\n                    \"file\": file_path.name,\n                })"#
        let result = MLXModelService.decodeJSONEscapes(input)
        #expect(result.contains("\n"), "literal \\n should become real newlines")
        #expect(result.contains("\"Other\""), "escaped quotes should become real quotes")
        #expect(!result.contains("\\n"), "literal \\n should not remain")
        #expect(!result.contains("\\\""), "escaped quotes should not remain")
    }

    @Test("decodeJSONEscapes preserves values that already have real newlines")
    func preservesRealNewlines() {
        // Write tool content has real newlines — don't double-decode
        let input = "def main():\n    print(\"hello\")\n    return 0"
        let result = MLXModelService.decodeJSONEscapes(input)
        #expect(result == input, "values with real newlines should be unchanged")
    }

    @Test("decodeJSONEscapes handles empty string")
    func decodesJSONEscapesEmpty() {
        #expect(MLXModelService.decodeJSONEscapes("") == "")
    }

    @Test("decodeJSONEscapes passes through plain text without escapes")
    func decodesJSONEscapesPlainText() {
        let input = "just plain text no escapes"
        #expect(MLXModelService.decodeJSONEscapes(input) == input)
    }

    @Test("decodeJSONEscapes handles \\t escapes")
    func decodesTab() {
        // \\t and \\n on a single line (no real newlines)
        let input = #"col1\tcol2\nrow data"#
        let result = MLXModelService.decodeJSONEscapes(input)
        #expect(result.contains("\t"), "\\t should become real tab")
        #expect(result.contains("\n"), "\\n should become real newline")
        #expect(result == "col1\tcol2\nrow data")
    }

    @Test("decodeJSONEscapes preserves real backslashes (\\\\n stays as \\n literal)")
    func preservesRealBackslashes() {
        // Input has \\n (real backslash + n) — should become \n (backslash + n), NOT a newline
        let input = "regex pattern: \\\\n matches newline"
        let result = MLXModelService.decodeJSONEscapes(input)
        #expect(result == "regex pattern: \\n matches newline")
    }

    @Test("decodeJSONEscapes handles only \\\" without \\n")
    func decodesQuotesOnly() {
        let input = #"he said \"hello\" and \"goodbye\""#
        let result = MLXModelService.decodeJSONEscapes(input)
        #expect(result == "he said \"hello\" and \"goodbye\"")
    }

    @Test("decodeJSONEscapes full edit oldString from real Qwen3-Coder output")
    func decodesRealWorldEditOldString() {
        // Exact model output from test session: entire Python function on single line
        let input = #"if category == \"Other\":\n                actions[\"skipped\"].append({\n                    \"file\": file_path.name,\n                    \"reason\": \"Unknown file type\",\n                })\n                continue"#
        let result = MLXModelService.decodeJSONEscapes(input)

        // Should have real newlines
        let lines = result.split(separator: "\n", omittingEmptySubsequences: false)
        #expect(lines.count > 1, "should have multiple lines after decoding")

        // Should have real quotes
        #expect(result.contains("\"Other\""))
        #expect(result.contains("\"skipped\""))
        #expect(result.contains("\"file\""))
        #expect(result.contains("\"reason\""))

        // First line should be the if statement
        #expect(lines[0].contains("if category == \"Other\":"))
    }

    // ═══════════════════════════════════════════════════════════════════
    // MARK: - decodeJSONEscapes integration with extractToolCallsFallback
    // ═══════════════════════════════════════════════════════════════════

    @Test("edit tool with JSON-escaped oldString gets decoded (model pre-escaping bug)")
    func editToolJSONEscapedOldString() {
        // Model emits edit oldString with \n and \" on a single line (JSON pre-escaping)
        let text = """
        <tool_call>
        <function=edit>
        <parameter=filePath>/tmp/app.py</parameter>
        <parameter=oldString>if category == \\"Other\\":\\n                actions[\\"skipped\\"].append({\\n                    \\"file\\": file_path.name,\\n                })</parameter>
        <parameter=newString>if category == \\"Other\\":\\n                # Skip unknown files\\n                actions[\\"skipped\\"].append({\\n                    \\"file\\": file_path.name,\\n                })</parameter>
        </function>
        </tool_call>
        """
        let (calls, _) = MLXModelService.extractToolCallsFallback(from: text)
        #expect(calls.count == 1)
        #expect(calls[0].function.name == "edit")

        let oldString = calls[0].function.arguments["oldString"]?.anyValue as? String
        let newString = calls[0].function.arguments["newString"]?.anyValue as? String

        // Should have real newlines (not literal \n)
        #expect(oldString?.contains("\n") == true, "oldString should have real newlines")
        #expect(newString?.contains("\n") == true, "newString should have real newlines")

        // Should have real quotes (not \")
        #expect(oldString?.contains("\"Other\"") == true, "oldString should have real quotes")
        #expect(newString?.contains("# Skip unknown files") == true, "newString should have the added comment")

        // Should NOT have escaped sequences remaining
        #expect(oldString?.contains("\\n") == false, "literal \\n should not remain")
        #expect(oldString?.contains("\\\"") == false, "literal \\\" should not remain")
    }

    @Test("write tool with real newlines is NOT affected by decodeJSONEscapes")
    func writeToolRealNewlinesUnchanged() {
        // Write tool content has real newlines — decodeJSONEscapes should not touch it
        let text = """
        <tool_call>
        <function=write>
        <parameter=filePath>/tmp/app.py</parameter>
        <parameter=content>
        def main():
            if x < 10:
                print("hello")
            return 0
        </parameter>
        </function>
        </tool_call>
        """
        let (calls, _) = MLXModelService.extractToolCallsFallback(from: text)
        #expect(calls.count == 1)
        let content = calls[0].function.arguments["content"]?.anyValue as? String
        #expect(content?.contains("def main():") == true)
        #expect(content?.contains("print(\"hello\")") == true)
        // Verify real newlines are preserved
        let lines = content?.split(separator: "\n", omittingEmptySubsequences: false) ?? []
        #expect(lines.count > 1, "content should have multiple real lines")
    }

    @Test("bash tool with JSON-escaped command gets decoded")
    func bashToolJSONEscapedCommand() {
        // Model emits bash command with \n on single line
        let text = """
        <tool_call>
        <function=bash>
        <parameter=command>cd /tmp && python -c \\"import sys; print(sys.version)\\"</parameter>
        <parameter=description>Check Python version</parameter>
        </function>
        </tool_call>
        """
        let (calls, _) = MLXModelService.extractToolCallsFallback(from: text)
        #expect(calls.count == 1)
        let cmd = calls[0].function.arguments["command"]?.anyValue as? String
        // The \" should be decoded to real "
        #expect(cmd?.contains("\"import sys") == true, "escaped quotes in command should be decoded")
    }

    @Test("mixed: write content has real newlines, edit oldString has escaped — both decoded correctly")
    func mixedRealAndEscapedNewlines() {
        // First tool call: write with real newlines
        let text = """
        <tool_call>
        <function=write>
        <parameter=content>
        line1
        line2
        line3
        </parameter>
        <parameter=filePath>/tmp/test.txt</parameter>
        </function>
        </tool_call>
        <tool_call>
        <function=edit>
        <parameter=filePath>/tmp/test.txt</parameter>
        <parameter=oldString>line1\\nline2</parameter>
        <parameter=newString>LINE1\\nLINE2</parameter>
        </function>
        </tool_call>
        """
        let (calls, _) = MLXModelService.extractToolCallsFallback(from: text)
        #expect(calls.count == 2)

        // Write: real newlines preserved
        let writeContent = calls[0].function.arguments["content"]?.anyValue as? String
        #expect(writeContent?.contains("\n") == true)

        // Edit: escaped newlines decoded
        let editOld = calls[1].function.arguments["oldString"]?.anyValue as? String
        #expect(editOld == "line1\nline2", "escaped \\n should become real newline")
    }

    // ═══════════════════════════════════════════════════════════════════
    // MARK: - requiredParamNames ordering
    // ═══════════════════════════════════════════════════════════════════

    @Test("requiredParamNames preserves client schema order (not alphabetical)")
    func requiredParamNamesPreservesOrder() {
        // Edit tool: client sends required=["filePath","oldString","newString"]
        // Must NOT be sorted alphabetically (which would put newString before oldString)
        let tool = makeRequestTool(name: "edit", properties: [
            "filePath": ["type": "string"],
            "oldString": ["type": "string"],
            "newString": ["type": "string"]
        ], required: ["filePath", "oldString", "newString"])
        let names = MLXModelService.requiredParamNames(for: tool)
        #expect(names == ["filePath", "oldString", "newString"],
                "required params should preserve client order: filePath → oldString → newString")
        // Specifically verify oldString comes before newString (not alphabetical)
        if let oldIdx = names.firstIndex(of: "oldString"),
           let newIdx = names.firstIndex(of: "newString") {
            #expect(oldIdx < newIdx, "oldString must come before newString (model needs to know what to find before replacement)")
        }
    }

    @Test("requiredParamNames returns empty for tool with no required array")
    func requiredParamNamesEmptyWhenNoRequired() {
        let tool = makeRequestTool(name: "search", properties: [
            "query": ["type": "string"]
        ])
        let names = MLXModelService.requiredParamNames(for: tool)
        #expect(names.isEmpty)
    }

    @Test("normalizeToolCall remaps keys before coercing values")
    func normalizeToolCallRemapsThenCoerces() {
        let tool = makeRequestTool(name: "search", properties: [
            "query": ["type": "string"],
            "maxResults": ["type": "integer"]
        ], required: ["query"])
        let toolCall = ToolCall(function: .init(
            name: "search",
            arguments: [
                "query": "files",
                "max_results": "5"
            ]
        ))

        let result = MLXModelService.normalizeToolCall(
            toolCall,
            index: 0,
            paramNameMapping: ["max_results": "maxResults"],
            tools: [tool],
            fixToolArgs: true
        )

        let args = parseArgs(result.function.arguments)
        #expect((args?["query"] as? String) == "files")
        #expect((args?["maxResults"] as? Int) == 5)
        #expect(args?["max_results"] == nil)
    }

    @Test("normalizeToolCalls assigns sequential indices from startIndex")
    func normalizeToolCallsAssignsSequentialIndices() {
        let tool = makeRequestTool(name: "search", properties: [
            "query": ["type": "string"]
        ], required: ["query"])
        let calls = [
            ToolCall(function: .init(name: "search", arguments: ["query": "one"])),
            ToolCall(function: .init(name: "search", arguments: ["query": "two"])),
        ]

        let result = MLXModelService.normalizeToolCalls(
            calls,
            startIndex: 3,
            tools: [tool]
        )

        #expect(result.count == 2)
        #expect(result[0].index == 3)
        #expect(result[1].index == 4)
    }

    @Test("completed tool call parsing resolves adaptive XML tool names against offered tools")
    func parseCompletedToolCallsResolvesAdaptiveXMLNames() {
        let tool = makeRequestTool(name: "get_weather", properties: [
            "location": ["type": "string"]
        ], required: ["location"])
        let text = #"<tool_call>{"name":"get_weathr","arguments":{"location":"Berlin"}}</tool_call>"#

        let (calls, remaining) = ToolCallStreamingRuntime.parseCompletedToolCalls(
            from: text,
            toolCallParser: "afm_adaptive_xml",
            tools: [tool]
        )

        #expect(calls.count == 1)
        #expect(calls[0].function.name == "get_weather")
        #expect(calls[0].function.arguments["location"]?.anyValue as? String == "Berlin")
        #expect(remaining.isEmpty)
    }

    @Test("completed tool call parsing preserves remaining non-tool text")
    func parseCompletedToolCallsPreservesRemainingText() {
        let tool = makeRequestTool(name: "search", properties: [
            "query": ["type": "string"]
        ], required: ["query"])
        let text = """
        First inspect the repo.

        <tool_call>
        <function=search>
        <parameter=query>auth middleware</parameter>
        </function>
        </tool_call>

        Then summarize findings.
        """

        let (calls, remaining) = ToolCallStreamingRuntime.parseCompletedToolCalls(
            from: text,
            toolCallParser: "qwen3_xml",
            tools: [tool]
        )

        #expect(calls.count == 1)
        #expect(calls[0].function.name == "search")
        #expect(remaining.contains("First inspect the repo."))
        #expect(remaining.contains("Then summarize findings."))
        #expect(!remaining.contains("<tool_call>"))
    }

    // ═══════════════════════════════════════════════════════════════════
    // MARK: - EBNF grammar generation (buildToolCallEBNF)
    // ═══════════════════════════════════════════════════════════════════

    @Test("EBNF grammar generates named required param rules for edit tool")
    func ebnfNamedRequiredParamsEdit() {
        let tools = [makeRequestTool(name: "edit", properties: [
            "filePath": ["type": "string"],
            "oldString": ["type": "string"],
            "newString": ["type": "string"]
        ], required: ["filePath", "oldString", "newString"])]
        let grammar = MLXModelService.buildToolCallEBNF(tools: tools)

        // Should have named rules for each required param
        #expect(grammar.contains("edit_rp_filePath"), "should have named rule for filePath")
        #expect(grammar.contains("edit_rp_oldString"), "should have named rule for oldString")
        #expect(grammar.contains("edit_rp_newString"), "should have named rule for newString")

        // Should have the call rule referencing named params in order
        #expect(grammar.contains("edit_rp_filePath edit_rp_oldString edit_rp_newString"),
                "call rule should reference named params in client schema order")
    }

    @Test("EBNF grammar generates named required param rules for bash tool")
    func ebnfNamedRequiredParamsBash() {
        let tools = [makeRequestTool(name: "bash", properties: [
            "command": ["type": "string"],
            "description": ["type": "string"]
        ], required: ["command", "description"])]
        let grammar = MLXModelService.buildToolCallEBNF(tools: tools)

        #expect(grammar.contains("bash_rp_command"), "should have named rule for command")
        #expect(grammar.contains("bash_rp_description"), "should have named rule for description")
        #expect(grammar.contains("bash_rp_command bash_rp_description"),
                "grammar must force both required params")
    }

    @Test("EBNF grammar allows extra optional params after required ones")
    func ebnfExtraParamsAfterRequired() {
        let tools = [makeRequestTool(name: "bash", properties: [
            "command": ["type": "string"],
            "description": ["type": "string"],
            "timeout": ["type": "integer"]
        ], required: ["command", "description"])]
        let grammar = MLXModelService.buildToolCallEBNF(tools: tools)

        // Should have extra_params (or tool-specific extra) after required
        #expect(grammar.contains("bash_rp_command bash_rp_description"))
        // The call rule should allow additional params
        #expect(grammar.contains("extra") || grammar.contains("param*"),
                "grammar should allow optional extra params")
    }

    @Test("EBNF grammar handles tool with no required params")
    func ebnfNoRequiredParams() {
        let tools = [makeRequestTool(name: "list", properties: [
            "path": ["type": "string"]
        ])]
        let grammar = MLXModelService.buildToolCallEBNF(tools: tools)

        // Should still generate a valid grammar without named required rules
        #expect(grammar.contains("call_list"), "should have call rule")
        #expect(!grammar.contains("list_rp_"), "should NOT have named required rules")
    }

    @Test("EBNF grammar param_value uses permissive pv_char rule")
    func ebnfParamValuePermissive() {
        let tools = [makeRequestTool(name: "write", properties: [
            "content": ["type": "string"]
        ], required: ["content"])]
        let grammar = MLXModelService.buildToolCallEBNF(tools: tools)

        // param_value uses pv_char: allows all chars except \n</ boundary
        #expect(grammar.contains("param_value ::= pv_char+"), "should have pv_char-based param_value")
        #expect(grammar.contains(#"[^\n]"#), "pv_char should allow all non-newline chars")
    }

    // ═══════════════════════════════════════════════════════════════════
    // MARK: - Helpers
    // ═══════════════════════════════════════════════════════════════════

    private func makeRequestTool(name: String, properties: [String: [String: String]], required: [String]? = nil) -> RequestTool {
        var propsDict: [String: Any] = [:]
        for (key, val) in properties {
            propsDict[key] = val
        }
        var schemaDict: [String: Any] = [
            "type": "object",
            "properties": propsDict
        ]
        if let required {
            schemaDict["required"] = required
        }
        let schemaData = try! JSONSerialization.data(withJSONObject: schemaDict)
        let schema = try! JSONDecoder().decode(AnyCodable.self, from: schemaData)
        return RequestTool(
            type: "function",
            function: RequestToolFunction(
                name: name,
                description: nil,
                parameters: schema
            )
        )
    }

    private func makeResponseToolCall(name: String, arguments: String) -> ResponseToolCall {
        ResponseToolCall(
            index: nil,
            id: "call_test",
            type: "function",
            function: ResponseToolCallFunction(name: name, arguments: arguments)
        )
    }

    private func parseArgs(_ json: String) -> [String: Any]? {
        guard let data = json.data(using: .utf8) else { return nil }
        return try? JSONSerialization.jsonObject(with: data) as? [String: Any]
    }
}
