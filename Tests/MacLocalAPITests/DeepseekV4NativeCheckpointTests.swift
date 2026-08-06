import MLX
import MLXLLM
import MLXLMCommon
@testable import AFMKitMLX
import XCTest

final class DeepseekV4NativeCheckpointTests: XCTestCase {
    func testDwarfstarExecutorProfileHasStableCLIName() {
        XCTAssertEqual(
            DeepseekV4CheckpointConverter.Profile.dwarfstarExecutor.rawValue,
            "dwarfstar-executor"
        )
        XCTAssertEqual(
            DeepseekV4CheckpointConverter.Profile(rawValue: "dwarfstar-executor"),
            .dwarfstarExecutor
        )
    }

    func testDwarfstarMXFP4BlocksInterleaveScaleAndLanePayload() throws {
        let words = MLXArray([UInt32(0x76543210), 0xfedcba98, 0x01234567, 0x89abcdef])
            .reshaped([1, 4])
        let scales = MLXArray([UInt8(0x7f)])
        let packed = try DeepseekV4CheckpointConverter.dwarfstarMXFP4Blocks(
            weight: words, scales: scales)
        MLX.eval(packed)

        XCTAssertEqual(packed.dtype, .uint8)
        XCTAssertEqual(packed.shape, [1, 17])
        XCTAssertEqual(packed.asArray(UInt8.self), [
            0x7f,
            0x70, 0x61, 0x52, 0x43, 0x34, 0x25, 0x16, 0x07,
            0xf8, 0xe9, 0xda, 0xcb, 0xbc, 0xad, 0x9e, 0x8f,
        ])
    }

    func testDwarfstarExecutorRoutedPackingRemovesSidecars() throws {
        let base = "model.layers.2.mlp.switch_mlp.gate_proj"
        let weights: [String: MLXArray] = [
            "\(base).weight": MLXArray([UInt32](repeating: 0, count: 4))
                .reshaped([1, 4]),
            "\(base).scales": MLXArray([UInt8(127)]).reshaped([1, 1]),
            "\(base).biases": MLXArray([Float(0)]),
            "model.norm.weight": MLXArray([Float(1)]),
        ]

        let converted = try DeepseekV4CheckpointConverter
            .packDwarfstarRoutedMXFP4(weights)

        XCTAssertEqual(converted["\(base).weight"]?.shape, [1, 17])
        XCTAssertNil(converted["\(base).scales"])
        XCTAssertNil(converted["\(base).biases"])
        XCTAssertNotNil(converted["model.norm.weight"])
    }

    func testDwarfstarExecutorRoutingTableUsesI32ABI() {
        let key = "model.layers.0.mlp.gate.tid2eid"
        let unrelated = "model.layers.0.mlp.gate.bias"
        let normalized = DeepseekV4CheckpointConverter.normalizeDwarfstarExecutorIntegers([
            key: MLXArray([Int64(0), 1, 2, 3]).reshaped([2, 2]),
            unrelated: MLXArray([Float(0), 1]),
        ])
        MLX.eval(normalized[key]!, normalized[unrelated]!)

        XCTAssertEqual(normalized[key]?.dtype, .int32)
        XCTAssertEqual(normalized[key]?.asArray(Int32.self), [0, 1, 2, 3])
        XCTAssertEqual(normalized[unrelated]?.dtype, .float32)
    }

    func testDwarfstarExecutorMaterializesTiedQ80OutputHead() throws {
        let embedding = MLXArray((0..<64).map { Float($0 - 32) / 8 })
            .reshaped([2, 32])
        let converted = try DeepseekV4CheckpointConverter.addDwarfstarTiedOutputHead([
            "model.embed_tokens.weight": embedding,
        ])
        MLX.eval(
            converted["model.embed_tokens.weight"]!,
            converted["lm_head.weight"]!,
            converted["lm_head.scales"]!)

        XCTAssertEqual(converted["model.embed_tokens.weight"]?.shape, [2, 32])
        XCTAssertEqual(converted["lm_head.weight"]?.shape, [2, 34])
        XCTAssertEqual(converted["lm_head.weight"]?.dtype, .uint8)
        XCTAssertEqual(converted["lm_head.scales"]?.shape, [2, 1])
    }

    func testDwarfstarExecutorDetectsCheckpointOutputHeadAliases() {
        XCTAssertTrue(DeepseekV4CheckpointConverter.containsOutputHead(["head.weight"]))
        XCTAssertTrue(DeepseekV4CheckpointConverter.containsOutputHead(["lm_head.weight"]))
        XCTAssertTrue(DeepseekV4CheckpointConverter.containsOutputHead(["model.lm_head.weight"]))
        XCTAssertFalse(DeepseekV4CheckpointConverter.containsOutputHead(["embed.weight"]))
    }

    func testQ80RealTensorMatchesSymmetricKernel() throws {
        let environment = ProcessInfo.processInfo.environment
        guard environment["AFM_DSV4_REAL_TENSOR_TEST"] == "1" else {
            throw XCTSkip("Set AFM_DSV4_REAL_TENSOR_TEST=1 for the external checkpoint differential.")
        }
        let root = "/Volumes/edata/models/vesta-test-cache/deepseek-ai"
        let q80Directory = environment["AFM_DSV4_Q80_MODEL"]
            ?? "\(root)/DeepSeek-V4-Flash-0731-AFM-Q8-0-MLX"
        let symmetricDirectory = environment["AFM_DSV4_SYMMETRIC_MODEL"]
            ?? "\(root)/DeepSeek-V4-Flash-0731-AFM-DwarfStar-Symmetric-Q8-MLX"
        let shard = environment["AFM_DSV4_REAL_TENSOR_SHARD"]
            ?? "model-00004-of-00048.safetensors"
        let q80 = try MLX.loadArrays(
            url: URL(fileURLWithPath: q80Directory).appendingPathComponent(shard))
        let symmetric = try MLX.loadArrays(
            url: URL(fileURLWithPath: symmetricDirectory).appendingPathComponent(shard))

        let keys = q80.keys.filter { key in
            key.hasSuffix(".weight") && q80[key]?.dtype == .uint8
                && q80[key]?.ndim == 2 && (q80[key]?.dim(-1).isMultiple(of: 34) ?? false)
        }.sorted()
        XCTAssertFalse(keys.isEmpty)
        for key in keys {
            guard let q80Weight = q80[key], let symmetricWeight = symmetric[key] else {
                XCTFail("Missing matching weight for \(key)")
                continue
            }
            let base = String(key.dropLast(".weight".count))
            guard let q80Scales = q80["\(base).scales"],
                  let symmetricScales = symmetric["\(base).scales"]
            else {
                XCTFail("Missing matching scales for \(key)")
                continue
            }
            let inputDimensions = q80Weight.dim(1) / 34 * 32
            let inputValues = (0..<inputDimensions).map {
                Float(($0 % 23) - 11) / 12
            }
            let input = MLXArray(inputValues).reshaped([1, 1, inputDimensions])
                .asType(.bfloat16)
            let q80Layer = DeepseekV4QuantizedLinear(
                weight: q80Weight, bias: nil, scales: q80Scales, biases: nil,
                groupSize: 32, bits: 8, mode: .affine)
            let symmetricLayer = DeepseekV4QuantizedLinear(
                weight: symmetricWeight, bias: nil, scales: symmetricScales, biases: nil,
                groupSize: 32, bits: 8, mode: .affine)
            let q80Output = q80Layer(input).asType(.float32)
            let symmetricOutput = symmetricLayer(input).asType(.float32)
            MLX.eval(q80Output, symmetricOutput)
            let q80Values = q80Output.asArray(Float.self)
            let symmetricValues = symmetricOutput.asArray(Float.self)
            XCTAssertEqual(q80Values.count, symmetricValues.count, key)
            let maximumDifference = zip(q80Values, symmetricValues)
                .map { abs($0 - $1) }.max() ?? 0
            XCTAssertEqual(maximumDifference, 0, accuracy: 1.0e-3, key)

            if base.hasSuffix(".self_attn.wo_a") {
                let outputGroups = 8
                let sequenceLength = 3
                let groupedValues = (0..<(sequenceLength * outputGroups * inputDimensions)).map {
                    Float(($0 % 29) - 14) / 15
                }
                let groupedInput = MLXArray(groupedValues)
                    .reshaped([1, sequenceLength, outputGroups, inputDimensions])
                    .asType(.bfloat16)
                let q80Grouped = q80Layer.symmetricQ8Grouped(
                    groupedInput, outputGroups: outputGroups).asType(.float32)
                let symmetricGrouped = symmetricLayer.symmetricQ8Grouped(
                    groupedInput, outputGroups: outputGroups).asType(.float32)
                MLX.eval(q80Grouped, symmetricGrouped)
                let q80GroupedValues = q80Grouped.asArray(Float.self)
                let symmetricGroupedValues = symmetricGrouped.asArray(Float.self)
                XCTAssertEqual(q80GroupedValues.count, symmetricGroupedValues.count, key)
                let groupedMaximumDifference = zip(q80GroupedValues, symmetricGroupedValues)
                    .map { abs($0 - $1) }.max() ?? 0
                XCTAssertEqual(groupedMaximumDifference, 0, accuracy: 1.0e-3, key)
            }
        }
    }

    func testQ80BlockABIStoresScaleThenSignedWeights() throws {
        let values = MLXArray((0..<32).map { Float($0 - 16) })
            .reshaped([1, 32])
        let (blocks, scales) = try DeepseekV4CheckpointConverter.q80Blocks(values)
        MLX.eval(blocks, scales)

        XCTAssertEqual(blocks.shape, [1, 34])
        XCTAssertEqual(scales.shape, [1, 1])
        let bytes = blocks.asArray(UInt8.self)
        let scaleBytes = scales.view(dtype: .uint8).asArray(UInt8.self)
        XCTAssertEqual(Array(bytes.prefix(2)), scaleBytes)
        XCTAssertEqual(bytes.count, 34)
        XCTAssertEqual(Int8(bitPattern: bytes[2]), -127)
        XCTAssertEqual(Int8(bitPattern: bytes[18]), 0)
    }

    func testQ80UsesGGMLHalfAwayFromZeroRounding() throws {
        var values = [Float](repeating: 0, count: 32)
        values[0] = 127
        values[1] = 0.5
        values[2] = -0.5
        let (blocks, _) = try DeepseekV4CheckpointConverter.q80Blocks(
            MLXArray(values).reshaped([1, 32]))
        MLX.eval(blocks)

        let bytes = blocks.asArray(UInt8.self)
        XCTAssertEqual(Int8(bitPattern: bytes[2]), 127)
        XCTAssertEqual(Int8(bitPattern: bytes[3]), 1)
        XCTAssertEqual(Int8(bitPattern: bytes[4]), -1)
    }

    func testQ80KernelMatchesScalarReference() throws {
        let sourceWeightValues: [Float] = (0..<96).map { index in
            let numerator = Float((index % 29) - 14)
            let denominator = Float((index / 32) + 1)
            return numerator / denominator
        }
        let sourceWeights = MLXArray(sourceWeightValues).reshaped([3, 32])
        let inputValues = (0..<32).map { Float(($0 % 11) - 5) / 7 }
        let input = MLXArray(inputValues).reshaped([1, 1, 32])
        let (blocks, scales) = try DeepseekV4CheckpointConverter.q80Blocks(sourceWeights)
        MLX.eval(blocks, scales)

        let layer = DeepseekV4QuantizedLinear(
            weight: blocks, bias: nil, scales: scales, biases: nil,
            groupSize: 32, bits: 8, mode: .affine)
        let output = layer(input)
        MLX.eval(output)

        let bytes = blocks.asArray(UInt8.self)
        let scaleValues = scales.asArray(Float16.self).map(Float.init)
        let actual = output.asArray(Float.self)
        var expected = [Float]()
        for row in 0..<3 {
            var total: Float = 0
            let blockStart = row * 34
            for column in 0..<32 {
                let q = Float(Int8(bitPattern: bytes[blockStart + 2 + column]))
                total += q * scaleValues[row] * inputValues[column]
            }
            expected.append(total)
        }

        XCTAssertEqual(actual.count, expected.count)
        for (value, reference) in zip(actual, expected) {
            XCTAssertEqual(value, reference, accuracy: 1.0e-3)
        }
    }

    func testQ80GroupedKernelMatchesScalarReference() throws {
        let sourceWeightValues: [Float] = (0..<128).map { index in
            let numerator = Float((index % 31) - 15)
            let denominator = Float((index / 32) + 1)
            return numerator / denominator
        }
        let sourceWeights = MLXArray(sourceWeightValues).reshaped([4, 32])
        let firstInput = (0..<32).map { Float(($0 % 13) - 6) / 8 }
        let secondInput = (0..<32).map { Float(($0 % 7) - 3) / 5 }
        let inputValues = firstInput + secondInput
        let input = MLXArray(inputValues).reshaped([1, 2, 32])
        let (blocks, scales) = try DeepseekV4CheckpointConverter.q80Blocks(sourceWeights)
        MLX.eval(blocks, scales)

        let layer = DeepseekV4QuantizedLinear(
            weight: blocks, bias: nil, scales: scales, biases: nil,
            groupSize: 32, bits: 8, mode: .affine)
        let output = layer.symmetricQ8Grouped(input, outputGroups: 2)
        MLX.eval(output)

        let bytes = blocks.asArray(UInt8.self)
        let scaleValues = scales.asArray(Float16.self).map(Float.init)
        let actual = output.asArray(Float.self)
        var expected = [Float]()
        for row in 0..<4 {
            let groupInput = row < 2 ? firstInput : secondInput
            var total: Float = 0
            let blockStart = row * 34
            for column in 0..<32 {
                let q = Float(Int8(bitPattern: bytes[blockStart + 2 + column]))
                total += q * scaleValues[row] * groupInput[column]
            }
            expected.append(total)
        }

        XCTAssertEqual(actual.count, expected.count)
        for (value, reference) in zip(actual, expected) {
            XCTAssertEqual(value, reference, accuracy: 1.0e-3)
        }
    }

    func testQ80KernelMatchesBF16MultiBlockReference() throws {
        let columns = 128
        let sourceWeightValues: [Float] = (0..<(3 * columns)).map { index in
            Float((index % 37) - 18) / Float((index / columns) + 3)
        }
        let sourceWeights = MLXArray(sourceWeightValues).reshaped([3, columns])
        let inputValues = (0..<columns).map { Float(($0 % 17) - 8) / 9 }
        let input = MLXArray(inputValues).reshaped([1, 1, columns]).asType(.bfloat16)
        let (blocks, scales) = try DeepseekV4CheckpointConverter.q80Blocks(sourceWeights)
        MLX.eval(blocks, scales)

        let layer = DeepseekV4QuantizedLinear(
            weight: blocks, bias: nil, scales: scales, biases: nil,
            groupSize: 32, bits: 8, mode: .affine)
        let output = layer(input).asType(.float32)
        MLX.eval(output)

        let bytes = blocks.asArray(UInt8.self)
        let scaleValues = scales.asArray(Float16.self).map(Float.init)
        let actual = output.asArray(Float.self)
        let groups = columns / 32
        var expected = [Float]()
        for row in 0..<3 {
            var total: Float = 0
            for group in 0..<groups {
                let blockStart = (row * groups + group) * 34
                for offset in 0..<32 {
                    let q = Float(Int8(bitPattern: bytes[blockStart + 2 + offset]))
                    let inputValue = inputValues[group * 32 + offset]
                    total += q * scaleValues[row * groups + group] * inputValue
                }
            }
            expected.append(total)
        }

        XCTAssertEqual(actual.count, expected.count)
        for (value, reference) in zip(actual, expected) {
            XCTAssertEqual(value, reference, accuracy: 0.2)
        }
    }

    func testNativeCheckpointMarkerDecodes() throws {
        let data = Data(#"{"model_type":"deepseek_v4","afm_native_checkpoint":true}"#.utf8)
        let config = try JSONDecoder().decode(DeepseekV4Configuration.self, from: data)
        XCTAssertTrue(config.afmNativeCheckpoint)
    }

    func testNativeCheckpointMarkerDefaultsToFalse() throws {
        let data = Data(#"{"model_type":"deepseek_v4"}"#.utf8)
        let config = try JSONDecoder().decode(DeepseekV4Configuration.self, from: data)
        XCTAssertFalse(config.afmNativeCheckpoint)
    }

    func testNativeCheckpointBypassesSanitizer() {
        var config = DeepseekV4Configuration()
        config.afmNativeCheckpoint = true
        config.vocabSize = 16
        config.hiddenSize = 8
        config.numHiddenLayers = 0
        config.numAttentionHeads = 1
        config.numKeyValueHeads = 1
        config.headDim = 8
        config.qkRopeHeadDim = 2
        config.qLoraRank = 4
        config.oGroups = 1
        config.oLoraRank = 4
        config.nRoutedExperts = 2
        config.numExpertsPerTok = 1
        config.moeIntermediateSize = 4

        let model = DeepseekV4Model(config)
        let input = ["model.already_normalized": MLXArray([Float(1)])]
        let output = model.sanitize(weights: input)

        XCTAssertEqual(Set(output.keys), Set(input.keys))
        XCTAssertEqual(output["model.already_normalized"]?.shape, [1])
    }

    func testDwarfstarQ8ProfileSelectsOnlyAdvertisedRoles() {
        XCTAssertTrue(DeepseekV4CheckpointConverter.usesDwarfstarQ8Control("lm_head"))
        XCTAssertTrue(DeepseekV4CheckpointConverter.usesDwarfstarQ8Control(
            "model.layers.4.self_attn.wq_a"))
        XCTAssertTrue(DeepseekV4CheckpointConverter.usesDwarfstarQ8Control(
            "model.layers.4.mlp.shared_experts.gate_proj"))

        XCTAssertFalse(DeepseekV4CheckpointConverter.usesDwarfstarQ8Control(
            "model.layers.4.mlp.switch_mlp.gate_proj"))
        XCTAssertFalse(DeepseekV4CheckpointConverter.usesDwarfstarQ8Control(
            "model.layers.4.self_attn.compressor.kv_a_proj"))
        XCTAssertFalse(DeepseekV4CheckpointConverter.usesDwarfstarQ8Control(
            "model.layers.4.self_attn.indexer.weight"))
        XCTAssertFalse(DeepseekV4CheckpointConverter.usesDwarfstarQ8Control(
            "model.embed_tokens"))
    }

    func testAlignedMXFP4SuperblocksPrefixScalesAndPreserveWords() throws {
        let words = MLXArray((0..<64).map(UInt32.init)).reshaped([1, 64])
        let scales = MLXArray((0..<16).map(UInt8.init)).reshaped([1, 16])

        let aligned = try DeepseekV4CheckpointConverter.alignedMXFP4Superblocks(
            weight: words, scales: scales)
        MLX.eval(aligned)

        XCTAssertEqual(aligned.shape, [1, 68])
        let values = aligned.asArray(UInt32.self)
        XCTAssertEqual(values[0...3], [0x03020100, 0x07060504, 0x0b0a0908, 0x0f0e0d0c])
        XCTAssertEqual(Array(values[4...]), (0..<64).map(UInt32.init))
    }
}
