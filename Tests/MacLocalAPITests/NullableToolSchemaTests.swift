import Foundation
import Testing
import Jinja

@testable import MacLocalAPI

/// Proves that toJinjaCompatible() fixes the Jinja crash on nullable tool schemas (Issue #32).
///
/// Root cause: JSON null → AnyCodableValue.null → toAny() → NSNull() → stored in
/// [String: any Sendable] (ToolSpec) → Jinja Value.init(any:) has no case for NSNull → throws
/// "Cannot convert value of type Optional<Any> to Jinja Value"
///
/// Fix: toJinjaCompatible() returns non-optional Any, stripping null-valued keys from dicts
/// and null elements from arrays. This avoids both NSNull and Optional<Any>-boxed-in-Any.
struct NullableToolSchemaTests {
// dimensions: tool_call_format=xmlFunction, response_format=json_schema

    // MARK: - The bug: toAny() produces NSNull that Jinja can't handle

    @Test("toAny() wraps null as NSNull — Jinja throws on it")
    func toAnyProducesNSNullThatJinjaRejects() throws {
        let nullValue = AnyCodableValue.null
        let any = nullValue.toAny()

        #expect(any is NSNull, "toAny() should return NSNull for .null")

        #expect(throws: JinjaError.self) {
            _ = try Value(any: any)
        }
    }

    @Test("toAny() dict with null value — Jinja throws")
    func toAnyDictWithNullCrashesJinja() throws {
        let schema = AnyCodableValue.object([
            "type": .string("object"),
            "properties": .object([
                "units": .object([
                    "anyOf": .array([
                        .object(["type": .string("string")]),
                        .object(["type": .string("null")])
                    ]),
                    "default": .null
                ])
            ])
        ])

        let dict = schema.toAny()

        #expect(throws: JinjaError.self) {
            _ = try Value(any: dict)
        }
    }

    // MARK: - The fix: toJinjaCompatible() strips nulls so Jinja never sees them

    @Test("toJinjaCompatible() strips null keys from dicts — Jinja accepts it")
    func toJinjaCompatibleDictStripsNullKeys() throws {
        let schema = AnyCodableValue.object([
            "type": .string("object"),
            "properties": .object([
                "units": .object([
                    "anyOf": .array([
                        .object(["type": .string("string")]),
                        .object(["type": .string("null")])
                    ]),
                    "default": .null
                ])
            ])
        ])

        let compatible = schema.toJinjaCompatible()

        // Should NOT throw
        let jinjaValue = try Value(any: compatible)
        #expect(!jinjaValue.isNull)

        // Verify "default" key was stripped
        if let dict = compatible as? [String: Any],
           let props = dict["properties"] as? [String: Any],
           let units = props["units"] as? [String: Any] {
            #expect(units["default"] == nil, "null-valued 'default' key should be stripped")
            #expect(units["anyOf"] != nil, "non-null 'anyOf' key should be preserved")
        } else {
            Issue.record("Expected nested dict structure")
        }
    }

    @Test("toJinjaCompatible() deeply nested nulls — Jinja accepts them")
    func toJinjaCompatibleDeepNestedNulls() throws {
        let schema = AnyCodableValue.object([
            "type": .string("object"),
            "properties": .object([
                "filters": .object([
                    "type": .string("object"),
                    "properties": .object([
                        "category": .object([
                            "anyOf": .array([
                                .object(["type": .string("string")]),
                                .object(["type": .string("null")])
                            ])
                        ]),
                        "limit": .object([
                            "anyOf": .array([
                                .object(["type": .string("integer")]),
                                .object(["type": .string("null")])
                            ]),
                            "default": .null
                        ])
                    ])
                ])
            ])
        ])

        let compatible = schema.toJinjaCompatible()
        let jinjaValue = try Value(any: compatible)
        #expect(!jinjaValue.isNull)
    }

    @Test("toJinjaCompatible() strips null elements from arrays")
    func toJinjaCompatibleArrayStripsNulls() throws {
        let arr = AnyCodableValue.array([.string("hello"), .null, .int(42)])
        let compatible = arr.toJinjaCompatible()

        if let resultArr = compatible as? [Any] {
            #expect(resultArr.count == 2, "null element should be stripped from array")
        } else {
            Issue.record("Expected array result")
        }

        let jinjaValue = try Value(any: compatible)
        #expect(!jinjaValue.isNull)
    }

    // MARK: - Regression: non-null values still work

    @Test("toJinjaCompatible() preserves all non-null types")
    func toJinjaCompatiblePreservesTypes() throws {
        let schema = AnyCodableValue.object([
            "type": .string("object"),
            "properties": .object([
                "location": .object([
                    "type": .string("string"),
                    "description": .string("City name")
                ])
            ]),
            "required": .array([.string("location")])
        ])

        let compatible = schema.toJinjaCompatible()
        let jinjaValue = try Value(any: compatible)
        #expect(!jinjaValue.isNull)
    }

    // MARK: - End-to-end: ToolSpec stored in [String: any Sendable] → Jinja

    @Test("toJinjaCompatible() result survives ToolSpec → Jinja Value round-trip")
    func toolSpecRoundTrip() throws {
        // Simulate what convertToToolSpecs does: store result in [String: any Sendable]
        let params = AnyCodableValue.object([
            "type": .string("object"),
            "properties": .object([
                "location": .object(["type": .string("string")]),
                "units": .object([
                    "anyOf": .array([
                        .object(["type": .string("string")]),
                        .object(["type": .string("null")])
                    ]),
                    "default": .null
                ])
            ]),
            "required": .array([.string("location")])
        ])

        // This is what convertToToolSpecs does
        var funcDict: [String: any Sendable] = ["name": "get_weather"]
        funcDict["parameters"] = params.toJinjaCompatible()

        let toolSpec: [String: any Sendable] = [
            "type": "function",
            "function": funcDict
        ]

        // This is what swift-transformers does: Value(any: toolSpec)
        let jinjaValue = try Value(any: toolSpec)
        #expect(!jinjaValue.isNull, "ToolSpec should convert to Jinja Value without error")
    }

    @Test("toAny() result in ToolSpec — Jinja throws (proves the bug)")
    func toolSpecWithToAnyFails() throws {
        let params = AnyCodableValue.object([
            "type": .string("object"),
            "properties": .object([
                "units": .object([
                    "default": .null
                ])
            ])
        ])

        var funcDict: [String: any Sendable] = ["name": "get_weather"]
        funcDict["parameters"] = params.toAny()  // Uses toAny() → NSNull

        let toolSpec: [String: any Sendable] = [
            "type": "function",
            "function": funcDict
        ]

        // This is what crashes in production
        #expect(throws: JinjaError.self) {
            _ = try Value(any: toolSpec)
        }
    }
}
