import Foundation

#if canImport(FoundationModels) && !DISABLE_FOUNDATION_MODELS
import FoundationModels

/// Converts OpenAI JSON Schema (from `response_format.json_schema`) to Apple's
/// `GenerationSchema` via `DynamicGenerationSchema` for true constrained decoding.
@available(macOS 26.0, *)
enum JSONSchemaConverter {

    // MARK: - Error types

    enum SchemaConversionError: Error, LocalizedError {
        case unsupportedType(String)
        case missingRequiredField(String)
        case invalidSchema(String)

        var errorDescription: String? {
            switch self {
            case .unsupportedType(let type):
                return "Unsupported JSON Schema type: \(type)"
            case .missingRequiredField(let field):
                return "Missing required field in JSON Schema: \(field)"
            case .invalidSchema(let detail):
                return "Invalid JSON Schema: \(detail)"
            }
        }
    }

    // MARK: - Public entry point

    /// Convert an OpenAI `ResponseJsonSchema` to Apple's `GenerationSchema`.
    ///
    /// The incoming schema is the `json_schema` field from the OpenAI request:
    /// ```json
    /// { "name": "colors", "schema": { "type": "object", ... }, "strict": true }
    /// ```
    static func convert(_ responseSchema: ResponseJsonSchema) throws -> GenerationSchema {
        guard let schemaCodable = responseSchema.schema else {
            throw SchemaConversionError.missingRequiredField("schema")
        }

        let schemaValue = schemaCodable.value
        let name = responseSchema.name ?? "root"

        // Collect $defs for reference resolution
        var defs: [String: AnyCodableValue] = [:]
        if case .object(let obj) = schemaValue, let defsVal = obj["$defs"] {
            if case .object(let defsDict) = defsVal {
                defs = defsDict
            }
        }

        let root = try convertSchema(schemaValue, name: name, defs: defs)

        // Build dependency schemas from $defs
        var dependencies: [DynamicGenerationSchema] = []
        for (defName, defValue) in defs {
            dependencies.append(try convertSchema(defValue, name: defName, defs: defs))
        }

        return try GenerationSchema(root: root, dependencies: dependencies)
    }

    // MARK: - Recursive schema conversion

    /// Convert an `AnyCodableValue` JSON Schema node into a `DynamicGenerationSchema`.
    private static func convertSchema(
        _ schema: AnyCodableValue,
        name: String = "",
        defs: [String: AnyCodableValue]
    ) throws -> DynamicGenerationSchema {
        guard case .object(let obj) = schema else {
            throw SchemaConversionError.invalidSchema("Schema node must be an object")
        }

        // Handle $ref
        if let refVal = obj["$ref"], case .string(let ref) = refVal {
            let refName = extractRefName(ref)
            return DynamicGenerationSchema(referenceTo: refName)
        }

        // Handle enum (string enum → anyOf strings)
        if let enumVal = obj["enum"], case .array(let values) = enumVal {
            let stringValues = values.compactMap { v -> String? in
                if case .string(let s) = v { return s }
                return nil
            }
            if !stringValues.isEmpty {
                let enumName = name.isEmpty ? "enum" : name
                return DynamicGenerationSchema(name: enumName, anyOf: stringValues)
            }
        }

        // Handle anyOf / oneOf (polymorphic types)
        if let anyOfVal = obj["anyOf"] ?? obj["oneOf"], case .array(let choices) = anyOfVal {
            let converted = try choices.map { try convertSchema($0, name: "", defs: defs) }
            let anyOfName = name.isEmpty ? "choice" : name
            return DynamicGenerationSchema(name: anyOfName, anyOf: converted)
        }

        // Get the type field
        guard let typeVal = obj["type"], case .string(let type) = typeVal else {
            throw SchemaConversionError.missingRequiredField("type")
        }

        let description = extractString(obj["description"])

        switch type {
        case "string":
            return DynamicGenerationSchema(type: String.self)

        case "integer":
            return DynamicGenerationSchema(type: Int.self)

        case "number":
            return DynamicGenerationSchema(type: Double.self)

        case "boolean":
            return DynamicGenerationSchema(type: Bool.self)

        case "object":
            return try convertObject(obj, name: name, description: description, defs: defs)

        case "array":
            return try convertArray(obj, description: description, defs: defs)

        case "null":
            // Null type — represent as a string enum with empty value
            let nullName = name.isEmpty ? "null" : name
            return DynamicGenerationSchema(name: nullName, anyOf: [""])

        default:
            throw SchemaConversionError.unsupportedType(type)
        }
    }

    // MARK: - Object conversion

    private static func convertObject(
        _ obj: [String: AnyCodableValue],
        name: String,
        description: String?,
        defs: [String: AnyCodableValue]
    ) throws -> DynamicGenerationSchema {
        guard let propsVal = obj["properties"], case .object(let props) = propsVal else {
            // Object without properties — create empty object
            let objectName = name.isEmpty ? "object" : name
            return DynamicGenerationSchema(name: objectName, description: description, properties: [])
        }

        // Determine required fields
        let requiredFields: Set<String>
        if let reqVal = obj["required"], case .array(let reqArr) = reqVal {
            requiredFields = Set(reqArr.compactMap { v -> String? in
                if case .string(let s) = v { return s }
                return nil
            })
        } else {
            requiredFields = []
        }

        var properties: [DynamicGenerationSchema.Property] = []
        // Maintain order from sorted keys for deterministic output
        for (propName, propSchema) in props.sorted(by: { $0.key < $1.key }) {
            let propDescription = extractPropertyDescription(propSchema)
            let childSchema = try convertSchema(propSchema, name: propName, defs: defs)
            let isOptional = !requiredFields.contains(propName)
            properties.append(
                DynamicGenerationSchema.Property(
                    name: propName,
                    description: propDescription,
                    schema: childSchema,
                    isOptional: isOptional
                )
            )
        }

        let objectName = name.isEmpty ? "object" : name
        return DynamicGenerationSchema(name: objectName, description: description, properties: properties)
    }

    // MARK: - Array conversion

    private static func convertArray(
        _ obj: [String: AnyCodableValue],
        description: String?,
        defs: [String: AnyCodableValue]
    ) throws -> DynamicGenerationSchema {
        guard let itemsVal = obj["items"] else {
            throw SchemaConversionError.missingRequiredField("items (for array type)")
        }

        let itemSchema = try convertSchema(itemsVal, name: "item", defs: defs)

        let minItems: Int? = extractInt(obj["minItems"])
        let maxItems: Int? = extractInt(obj["maxItems"])

        return DynamicGenerationSchema(
            arrayOf: itemSchema,
            minimumElements: minItems,
            maximumElements: maxItems
        )
    }

    // MARK: - Helpers

    /// Extract `#/$defs/Foo` → `Foo`
    private static func extractRefName(_ ref: String) -> String {
        if ref.hasPrefix("#/$defs/") {
            return String(ref.dropFirst("#/$defs/".count))
        }
        if ref.hasPrefix("#/definitions/") {
            return String(ref.dropFirst("#/definitions/".count))
        }
        // Fallback: use the last path component
        return ref.components(separatedBy: "/").last ?? ref
    }

    private static func extractString(_ value: AnyCodableValue?) -> String? {
        guard let value else { return nil }
        if case .string(let s) = value { return s }
        return nil
    }

    private static func extractInt(_ value: AnyCodableValue?) -> Int? {
        guard let value else { return nil }
        if case .int(let i) = value { return i }
        if case .double(let d) = value { return Int(d) }
        return nil
    }

    private static func extractPropertyDescription(_ schema: AnyCodableValue) -> String? {
        guard case .object(let obj) = schema else { return nil }
        return extractString(obj["description"])
    }
}

#endif
