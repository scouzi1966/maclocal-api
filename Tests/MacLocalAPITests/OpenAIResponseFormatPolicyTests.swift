import Testing

@testable import AFMKit

struct OpenAIResponseFormatPolicyTests {
    @Test("response_format policy uses server default only when request omits format")
    func requestFormatOverridesServerDefault() {
        let serverDefault = Self.schemaFormat(name: "server")
        let perRequest = Self.schemaFormat(name: "request")

        #expect(OpenAIResponseFormatPolicy.effectiveResponseFormat(
            requestFormat: nil,
            serverDefault: serverDefault
        )?.jsonSchema?.name == "server")

        #expect(OpenAIResponseFormatPolicy.effectiveResponseFormat(
            requestFormat: perRequest,
            serverDefault: serverDefault
        )?.jsonSchema?.name == "request")
    }

    @Test("response_format policy exposes only strict json_schema schemas")
    func strictJsonSchemaResolution() {
        let strict = Self.schemaFormat(name: "strict", strict: true)
        let nonStrict = Self.schemaFormat(name: "non_strict", strict: false)

        #expect(OpenAIResponseFormatPolicy.effectiveStrictJsonSchema(
            requestFormat: nil,
            serverDefault: strict
        )?.name == "strict")

        #expect(OpenAIResponseFormatPolicy.effectiveStrictJsonSchema(
            requestFormat: nil,
            serverDefault: nonStrict
        ) == nil)

        #expect(OpenAIResponseFormatPolicy.effectiveStrictJsonSchema(
            requestFormat: ResponseFormat(type: "json_object"),
            serverDefault: strict
        ) == nil)
    }

    @Test("response_format policy strips fenced structured output")
    func structuredOutputSanitization() {
        let jsonObject = ResponseFormat(type: "json_object")
        let raw = "\n```json\n{\"ok\":true}\n```\n"

        #expect(OpenAIResponseFormatPolicy.requiresStructuredOutputSanitization(jsonObject))
        #expect(OpenAIResponseFormatPolicy.sanitizeStructuredOutput(
            raw,
            responseFormat: jsonObject
        ) == "{\"ok\":true}")

        #expect(OpenAIResponseFormatPolicy.sanitizeStructuredOutput(
            raw,
            responseFormat: nil
        ) == raw)
    }

    private static func schemaFormat(name: String, strict: Bool = true) -> ResponseFormat {
        ResponseFormat(
            type: "json_schema",
            jsonSchema: ResponseJsonSchema(
                name: name,
                description: nil,
                schema: AnyCodable(["type": "object"]),
                strict: strict
            )
        )
    }
}
