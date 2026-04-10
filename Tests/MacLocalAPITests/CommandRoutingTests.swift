import Foundation
import Testing

@testable import MacLocalAPI

struct CommandRoutingTests {

    @Test("ServeCommand parses guided-json flag")
    func serveCommandParsesGuidedJson() throws {
        let schema = #"{"type":"object","properties":{"planet":{"type":"string"}}}"#
        let command = try ServeCommand.parse(["--guided-json", schema])

        #expect(command.guidedJson == schema)
    }

    @Test("RootCommand forwards guided-json to ServeCommand args")
    func rootCommandForwardsGuidedJson() throws {
        let schema = #"{"type":"object","properties":{"planet":{"type":"string"}}}"#
        let parsed = try RootCommand.parseAsRoot(["--guided-json", schema, "--port", "9998"])
        guard let command = parsed as? RootCommand else {
            Issue.record("Expected RootCommand from parseAsRoot")
            return
        }
        let args = command.makeServeArgs()

        #expect(args.contains("--guided-json"))
        #expect(args.contains(schema))
        #expect(args.contains("9998"))
    }

    @Test("Foundation guided-json fallback only applies when request omits response_format")
    func effectiveGuidedJsonSchemaFallback() throws {
        let defaultSchema = try parseGuidedJsonSchema(
            #"{"type":"object","properties":{"planet":{"type":"string"}},"required":["planet"]}"#
        )

        let requestSchema = ResponseJsonSchema(
            name: "request",
            description: nil,
            schema: AnyCodable(["type": "object"]),
            strict: true
        )

        let requestResponseFormat = ResponseFormat(type: "json_schema", jsonSchema: requestSchema)
        let jsonObjectFormat = ResponseFormat(type: "json_object", jsonSchema: nil)

        let defaultResolved = ChatCompletionsController.effectiveGuidedJsonSchema(
            requestResponseFormat: nil,
            defaultGuidedJsonSchema: defaultSchema
        )
        #expect(defaultResolved?.name == "guided")

        let requestResolved = ChatCompletionsController.effectiveGuidedJsonSchema(
            requestResponseFormat: requestResponseFormat,
            defaultGuidedJsonSchema: defaultSchema
        )
        #expect(requestResolved?.name == "request")

        let nonSchemaResolved = ChatCompletionsController.effectiveGuidedJsonSchema(
            requestResponseFormat: jsonObjectFormat,
            defaultGuidedJsonSchema: defaultSchema
        )
        #expect(nonSchemaResolved == nil)
    }
}
