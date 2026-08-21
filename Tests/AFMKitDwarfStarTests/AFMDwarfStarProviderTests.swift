import XCTest
import AFMKitCore
import CDwarfStar
@testable import AFMKitDwarfStar

final class AFMDwarfStarProviderTests: XCTestCase {
    func testModelErasureRetainsRawCompletionCapability() {
        let model = AFMDwarfStarModel(modelID: "raw", modelPath: "/missing/model.gguf")
        XCTAssertNotNil(AnyAFMModel(model).rawTextGenerator)
        XCTAssertNotNil(AnyAFMModel(model).generationAdmitter)
    }

    func testGenerationAdmissionDoesNotOverAdmitOrReleaseAReplacement() async throws {
        let model = AFMDwarfStarModel(
            modelID: "admission",
            modelPath: "/missing/model.gguf",
            maxConcurrent: 2,
            runtime: AFMDwarfStarRuntimeCoordinator()
        )
        let admitter = try XCTUnwrap(AnyAFMModel(model).generationAdmitter)
        let first = try await admitter.admitGeneration(timeout: .zero)
        let second = try await admitter.admitGeneration(timeout: .zero)

        await assertCapacityRejected(admitter)

        first.release()
        let replacement = try await admitter.admitGeneration(timeout: .zero)
        first.release()

        await assertCapacityRejected(admitter)

        second.release()
        replacement.release()
        let final = try await admitter.admitGeneration(timeout: .zero)
        final.release()
    }

    private func assertCapacityRejected(
        _ admitter: AnyAFMGenerationAdmitter,
        file: StaticString = #filePath,
        line: UInt = #line
    ) async {
        do {
            _ = try await admitter.admitGeneration(timeout: .zero)
            XCTFail("admission exceeded configured capacity", file: file, line: line)
        } catch let error as AFMGenerationAdmissionError {
            XCTAssertEqual(error, .capacity, file: file, line: line)
        } catch {
            XCTFail("unexpected admission error: \(error)", file: file, line: line)
        }
    }

    func testIgnoreEOSSuppressesOnlyTheEOSTerminalCandidate() {
        XCTAssertTrue(AFMDwarfStarStoppingPolicy.shouldStop(
            isEndOfSequence: true,
            isRuntimeStop: true,
            ignoreEndOfSequence: false
        ))
        XCTAssertFalse(AFMDwarfStarStoppingPolicy.shouldStop(
            isEndOfSequence: true,
            isRuntimeStop: true,
            ignoreEndOfSequence: true
        ))
        XCTAssertTrue(AFMDwarfStarStoppingPolicy.shouldStop(
            isEndOfSequence: false,
            isRuntimeStop: true,
            ignoreEndOfSequence: true
        ))
        XCTAssertFalse(AFMDwarfStarStoppingPolicy.shouldExposeToken(
            isEndOfSequence: true,
            ignoreEndOfSequence: true
        ))
        XCTAssertTrue(AFMDwarfStarStoppingPolicy.shouldExposeToken(
            isEndOfSequence: false,
            ignoreEndOfSequence: true
        ))
    }

    func testStreamingIgnoreEOSDoesNotConsumeVisibleBudgetOrTelemetry() {
        var accounting = AFMDwarfStarOutputAccounting(maximumTokens: 3)
        var telemetryTokens = 0
        var streamedText = ""

        for _ in 1..<AFMDwarfStarOutputAccounting
            .maximumConsecutiveSuppressedEndOfSequenceTokens
        {
            let disposition = accounting.disposition(
                isEndOfSequence: true,
                isRuntimeStop: true,
                ignoreEndOfSequence: true
            )
            if disposition == .expose {
                streamedText += "<eos>"
            }
            XCTAssertEqual(disposition, .suppress)
        }

        XCTAssertEqual(streamedText, "")
        XCTAssertEqual(accounting.visibleTokens, 0)
        XCTAssertEqual(telemetryTokens, 0)
        XCTAssertFalse(accounting.isExhausted)
    }

    func testRepeatedSuppressedEOSStopsWithoutChangingUsage() {
        var accounting = AFMDwarfStarOutputAccounting(maximumTokens: 4)
        var telemetryTokens = 0

        var disposition = AFMDwarfStarOutputAccounting.Disposition.suppress
        for _ in 0..<AFMDwarfStarOutputAccounting
            .maximumConsecutiveSuppressedEndOfSequenceTokens
        {
            disposition = accounting.disposition(
                isEndOfSequence: true,
                isRuntimeStop: true,
                ignoreEndOfSequence: true
            )
        }

        let usage = AFMUsage(outputTokens: accounting.visibleTokens)
        XCTAssertEqual(disposition, .stop)
        XCTAssertEqual(usage.outputTokens, 0)
        XCTAssertEqual(telemetryTokens, 0)
        XCTAssertFalse(accounting.isExhausted)
    }

    func testVisibleTokenAfterSuppressedEOSConsumesOneOutputToken() {
        var accounting = AFMDwarfStarOutputAccounting(maximumTokens: 2)
        var telemetryTokens = 0

        XCTAssertEqual(accounting.disposition(
            isEndOfSequence: true,
            isRuntimeStop: true,
            ignoreEndOfSequence: true
        ), .suppress)
        XCTAssertEqual(accounting.disposition(
            isEndOfSequence: false,
            isRuntimeStop: false,
            ignoreEndOfSequence: true
        ), .expose)
        accounting.recordVisible { telemetryTokens += 1 }

        XCTAssertEqual(accounting.visibleTokens, 1)
        XCTAssertEqual(telemetryTokens, 1)
        XCTAssertEqual(accounting.consecutiveSuppressedEndOfSequenceTokens, 0)
    }

    func testRawStopPolicyWithholdsStopAcrossTokenPieces() {
        var buffer = ""
        let first = AFMDwarfStarRawStopPolicy.consume(
            buffer: &buffer,
            piece: "answer EN",
            stopSequences: ["END"]
        )
        XCTAssertEqual(first, .init(visibleText: "answer ", stopped: false))
        XCTAssertEqual(buffer, "EN")

        let second = AFMDwarfStarRawStopPolicy.consume(
            buffer: &buffer,
            piece: "D trailing text",
            stopSequences: ["END"]
        )
        XCTAssertEqual(second, .init(visibleText: "", stopped: true))
        XCTAssertEqual(buffer, "")
    }

    func testRawStopPolicyDrainsAnIncompletePrefixAtLengthBoundary() {
        var buffer = ""
        let result = AFMDwarfStarRawStopPolicy.consume(
            buffer: &buffer,
            piece: "value ST",
            stopSequences: ["STOP"]
        )
        XCTAssertEqual(result, .init(visibleText: "value ", stopped: false))
        XCTAssertEqual(AFMDwarfStarRawStopPolicy.drain(buffer: &buffer), "ST")
        XCTAssertEqual(buffer, "")
    }
    func testProviderContractDescribesInProcessDeviceRuntime() {
        let descriptor = AFMDwarfStarProviderFactory().descriptor

        XCTAssertEqual(descriptor.id, "dwarfstar")
        XCTAssertEqual(descriptor.privacyBoundary, .device)
        XCTAssertEqual(descriptor.metadata["runtime"], .string("in-process-ds4"))
        XCTAssertEqual(descriptor.metadata["execution"], .string("fixed-metal-schedule"))
        XCTAssertEqual(descriptor.metadata["checkpointFormat"], .string("native-gguf"))
        XCTAssertTrue(descriptor.configurationKeys.contains("modelPath"))
        XCTAssertTrue(descriptor.configurationKeys.contains("enablePrefixCaching"))
        XCTAssertTrue(descriptor.configurationKeys.contains("maxConcurrent"))
        XCTAssertTrue(descriptor.configurationKeys.contains("dsparkSupportPath"))
        XCTAssertTrue(descriptor.configurationKeys.contains("dsparkDraftTokens"))
        XCTAssertTrue(descriptor.configurationKeys.contains("dsparkConfidenceThreshold"))
        XCTAssertTrue(descriptor.configurationKeys.contains("dsparkStrict"))
        XCTAssertFalse(descriptor.configurationKeys.contains("templateGGUF"))
        XCTAssertFalse(descriptor.configurationKeys.contains("projectionMetadataPath"))
        XCTAssertFalse(descriptor.configurationKeys.contains("externalMapGGUF"))
    }

    func testBundledMetalRuntimeContainsEveryRequiredSource() throws {
        let root = try XCTUnwrap(AFMDwarfStarRuntime.metalSourceDirectory)
        let requiredSources = [
            "flash_attn.metal", "dense.metal", "moe.metal", "dsv4_hc.metal",
            "unary.metal", "dsv4_kv.metal", "dsv4_rope.metal", "dsv4_misc.metal",
            "argsort.metal", "cpy.metal", "concat.metal", "get_rows.metal",
            "sum_rows.metal", "softmax.metal", "repeat.metal", "glu.metal",
            "norm.metal", "bin.metal", "set_rows.metal",
        ]

        for source in requiredSources {
            XCTAssertTrue(
                FileManager.default.fileExists(atPath: root.appendingPathComponent(source).path),
                "missing bundled DwarfStar Metal source \(source)"
            )
        }
    }

    func testMissingModelIsUnavailableWithoutLoadingRuntime() async {
        let model = AFMDwarfStarModel(
            modelID: "missing",
            modelPath: "/path/that/does/not/exist.gguf",
            runtime: AFMDwarfStarRuntimeCoordinator()
        )

        let availability = await model.availability()
        XCTAssertFalse(availability.isAvailable)
        XCTAssertEqual(model.descriptor.providerID, "dwarfstar")
        XCTAssertTrue(model.descriptor.capabilities.contains(.reasoning))
        XCTAssertTrue(model.descriptor.capabilities.contains(.streaming))
        XCTAssertTrue(model.descriptor.capabilities.contains(.toolCalling))
    }

    func testModelDescriptorPublishesResidentSessionConfiguration() {
        let model = AFMDwarfStarModel(
            modelID: "configured",
            modelPath: "/missing.gguf",
            dsparkSupportPath: "/support.gguf",
            dsparkDraftTokens: 8,
            dsparkConfidenceThreshold: 0.9,
            dsparkStrict: true,
            enablePrefixCaching: true,
            maxConcurrent: 4,
            runtime: AFMDwarfStarRuntimeCoordinator()
        )

        XCTAssertEqual(model.descriptor.metadata["enablePrefixCaching"], .bool(true))
        XCTAssertEqual(model.descriptor.metadata["checkpointFormat"], .string("native-gguf"))
        XCTAssertEqual(model.descriptor.metadata["maxConcurrent"], .integer(4))
        XCTAssertEqual(model.descriptor.metadata["dsparkEnabled"], .bool(true))
        XCTAssertEqual(model.descriptor.metadata["dsparkDraftTokens"], .integer(8))
        XCTAssertEqual(model.descriptor.metadata["dsparkConfidenceThreshold"], .number(0.9))
        XCTAssertEqual(model.descriptor.metadata["dsparkStrict"], .bool(true))
    }

    func testReasoningModeDefaultsToChatWithoutThinkingControls() {
        XCTAssertEqual(AFMDwarfStarReasoningMode.resolve(metadata: [:]), .chat)
    }

    func testReasoningModeUsesOfficialReasoningEffort() {
        XCTAssertEqual(
            AFMDwarfStarReasoningMode.resolve(metadata: [
                "chatTemplateKwargs": .object(["reasoning_effort": .string("max")])
            ]),
            .max)
    }

    func testReasoningModeTreatsEnableThinkingAsLowEffort() {
        XCTAssertEqual(
            AFMDwarfStarReasoningMode.resolve(metadata: [
                "chatTemplateKwargs": .object(["enable_thinking": .bool(true)])
            ]),
            .low)
    }

    func testNoThinkingOverridesReasoningEffort() {
        XCTAssertEqual(
            AFMDwarfStarReasoningMode.resolve(metadata: [
                "chatTemplateKwargs": .object([
                    "enable_thinking": .bool(false),
                    "reasoning_effort": .string("max")
                ])
            ]),
            .chat)
    }

    func testReasoningModesMapToNativeDwarfStarModes() {
        XCTAssertEqual(AFMDwarfStarReasoningMode.chat.thinkMode, DS4_THINK_NONE)
        XCTAssertEqual(AFMDwarfStarReasoningMode.low.thinkMode, DS4_THINK_HIGH)
        XCTAssertEqual(AFMDwarfStarReasoningMode.high.thinkMode, DS4_THINK_HIGH)
        XCTAssertEqual(AFMDwarfStarReasoningMode.max.thinkMode, DS4_THINK_MAX)
    }

    func testSlotPolicyUsesFirstAvailableSlotWithoutPrefixCaching() {
        XCTAssertEqual(
            AFMDwarfStarSlotPolicy.bestSlot(
                commonPrefixes: [nil, 12, 30],
                prefixCachingEnabled: false),
            1)
    }

    func testSlotPolicyPrefersLongestReusablePrefixWithStableTieBreak() {
        XCTAssertEqual(
            AFMDwarfStarSlotPolicy.bestSlot(
                commonPrefixes: [12, nil, 30, 30],
                prefixCachingEnabled: true),
            2)
    }

    func testSchedulerUsesLargePrefillQuantumWhenNoDecodeIsActive() {
        XCTAssertEqual(
            AFMDwarfStarSchedulingPolicy.prefillQuantum(activeDecodeCount: 0),
            2_048)
    }

    func testSchedulerUsesBoundedPrefillQuantumDuringContinuousDecode() {
        XCTAssertEqual(
            AFMDwarfStarSchedulingPolicy.prefillQuantum(activeDecodeCount: 3),
            128)
    }

    func testSchedulerEstablishesCheckpointBeforeMixedPrefill() {
        XCTAssertFalse(
            AFMDwarfStarSchedulingPolicy.canMixPrefill(
                currentPosition: 0, activeDecodeCount: 3))
        XCTAssertTrue(
            AFMDwarfStarSchedulingPolicy.canMixPrefill(
                currentPosition: 128, activeDecodeCount: 3))
        XCTAssertFalse(
            AFMDwarfStarSchedulingPolicy.canMixPrefill(
                currentPosition: 128, activeDecodeCount: 0))
    }

    func testDSparkAvailabilityUsesGeneralizedDraftDepth() {
        XCTAssertFalse(
            AFMDwarfStarSpeculativePolicy.isAvailable(
                requested: false,
                draftTokenCount: 5))
        XCTAssertFalse(
            AFMDwarfStarSpeculativePolicy.isAvailable(
                requested: true,
                draftTokenCount: 0))
        XCTAssertTrue(
            AFMDwarfStarSpeculativePolicy.isAvailable(
                requested: true,
                draftTokenCount: 5))
    }

    func testSchedulerRotatesAcrossWaitingPrefills() {
        XCTAssertEqual(
            AFMDwarfStarSchedulingPolicy.nextPrefillSlot(
                lastSlot: 0,
                waiting: [true, true, false, true]),
            1)
        XCTAssertEqual(
            AFMDwarfStarSchedulingPolicy.nextPrefillSlot(
                lastSlot: 1,
                waiting: [true, true, false, true]),
            3)
        XCTAssertEqual(
            AFMDwarfStarSchedulingPolicy.nextPrefillSlot(
                lastSlot: 3,
                waiting: [true, true, false, true]),
            0)
    }

    func testSchedulerReturnsNilWhenNoPromptNeedsPrefill() {
        XCTAssertNil(
            AFMDwarfStarSchedulingPolicy.nextPrefillSlot(
                lastSlot: 2,
                waiting: [false, false, false]))
    }

    func testPrefixCacheCheckpointIdentitySeparatesModelRevisions() {
        let original = AFMDwarfStarPrefixCachePolicy.checkpointKey(
            path: "/models/deepseek.gguf", size: 100, modified: 10)
        XCTAssertEqual(
            original,
            AFMDwarfStarPrefixCachePolicy.checkpointKey(
                path: "/models/deepseek.gguf", size: 100, modified: 10))
        XCTAssertNotEqual(
            original,
            AFMDwarfStarPrefixCachePolicy.checkpointKey(
                path: "/models/deepseek.gguf", size: 101, modified: 10))
        XCTAssertNotEqual(
            original,
            AFMDwarfStarPrefixCachePolicy.checkpointKey(
                path: "/models/deepseek.gguf", size: 100, modified: 11))
    }

    func testPrefixCacheBudgetUsesSafeDefaultAndEnvironmentOverride() {
        XCTAssertEqual(AFMDwarfStarPrefixCachePolicy.budgetMB(environment: [:]), 4_096)
        XCTAssertEqual(
            AFMDwarfStarPrefixCachePolicy.budgetMB(
                environment: ["AFM_DWARFSTAR_PREFIX_CACHE_MB": "8192"]),
            8_192)
        XCTAssertEqual(
            AFMDwarfStarPrefixCachePolicy.budgetMB(
                environment: ["AFM_DWARFSTAR_PREFIX_CACHE_MB": "0"]),
            4_096)
    }

    func testToolPromptPublishesSchemasInDeepSeekDSMLFormat() throws {
        let prompt = try AFMDwarfStarToolCodec.systemPrompt(for: [
            AFMToolDefinition(
                name: "weather",
                description: "Look up weather.",
                inputSchema: .object([
                    "type": .string("object"),
                    "properties": .object([
                        "city": .object(["type": .string("string")])
                    ])
                ])
            )
        ])

        XCTAssertTrue(prompt.contains("<｜DSML｜tool_calls>"))
        XCTAssertTrue(prompt.contains("\"name\":\"weather\""))
        XCTAssertTrue(prompt.contains("\"city\""))
    }

    func testToolParserHandlesSplitMarkersAndParallelCalls() throws {
        var parser = AFMDwarfStarToolCodec.StreamParser()
        XCTAssertEqual(try parser.consume("I will check. <｜DSML｜tool_"), [.text("I will check. ")])
        XCTAssertEqual(try parser.consume("calls>\n<｜DSML｜invoke name=\"weather\">\n"), [])
        XCTAssertEqual(
            try parser.consume(
                "<｜DSML｜parameter name=\"city\" string=\"true\">Paris</｜DSML｜parameter>\n"
                    + "</｜DSML｜invoke>\n<｜DSML｜invoke name=\"clock\">\n"
                    + "<｜DSML｜parameter name=\"offset\" string=\"false\">-4</｜DSML｜parameter>\n"
                    + "</｜DSML｜invoke>\n</｜DSML｜tool_calls>"
            ),
            [
                .toolCalls([
                    AFMToolCall(
                        id: "call_1",
                        name: "weather",
                        arguments: "{\"city\":\"Paris\"}"
                    ),
                    AFMToolCall(
                        id: "call_2",
                        name: "clock",
                        arguments: "{\"offset\":-4}"
                    )
                ])
            ]
        )
    }

    func testToolMarkupInsideReasoningIsNotExecuted() throws {
        var parser = AFMDwarfStarToolCodec.StreamParser(startsInReasoning: true)
        let fakeCall = "<｜DSML｜tool_calls><｜DSML｜invoke name=\"unsafe\"></｜DSML｜invoke></｜DSML｜tool_calls>"

        XCTAssertEqual(
            try parser.consume("consider \(fakeCall)</think>answer"),
            [.reasoning("consider \(fakeCall)"), .text("answer")]
        )
    }

    func testCompletedToolCallTerminatesParserAndDiscardsTrailingOutput() throws {
        var parser = AFMDwarfStarToolCodec.StreamParser()
        let call = "<｜DSML｜tool_calls><｜DSML｜invoke name=\"weather\"></｜DSML｜invoke></｜DSML｜tool_calls>"

        let outputs = try parser.consume(call + "must not become response text")
        guard case .toolCalls(let calls) = try XCTUnwrap(outputs.first) else {
            return XCTFail("Expected a completed tool call")
        }
        XCTAssertEqual(calls.map(\.name), ["weather"])
        XCTAssertEqual(outputs.count, 1)
        XCTAssertEqual(try parser.consume("also ignored"), [])
        XCTAssertEqual(parser.finish(), [])
    }

    func testReasoningMarkersCanSpanChunks() throws {
        var parser = AFMDwarfStarToolCodec.StreamParser(startsInReasoning: true)

        XCTAssertEqual(try parser.consume("analysis</thi"), [.reasoning("analysis")])
        XCTAssertEqual(try parser.consume("nk>final"), [.text("final")])
    }

    func testResponseCanEnterAndLeaveExplicitReasoning() throws {
        var parser = AFMDwarfStarToolCodec.StreamParser()

        XCTAssertEqual(
            try parser.consume("before<think>private</think>after"),
            [.text("before"), .reasoning("private"), .text("after")]
        )
    }

    func testAssistantToolCallsRoundTripThroughDSML() throws {
        let message = AFMMessage(
            role: .assistant,
            content: [.text("Checking")],
            toolCalls: [
                AFMToolCall(
                    id: "original",
                    name: "weather",
                    arguments: "{\"city\":\"Montréal\",\"days\":2}"
                )
            ]
        )
        let rendered = try AFMDwarfStarToolCodec.assistantContent(for: message)

        XCTAssertTrue(rendered.hasPrefix("Checking\n\n<｜DSML｜tool_calls>"))
        XCTAssertTrue(rendered.contains("name=\"weather\""))
        XCTAssertTrue(rendered.contains("name=\"city\" string=\"true\">Montréal"))
        XCTAssertTrue(rendered.contains("name=\"days\" string=\"false\">2"))
    }

    func testAssistantToolReplayUsesCanonicalSeparatorAndBoundary() throws {
        let message = AFMMessage(
            role: .assistant,
            content: [.text("Checking")],
            toolCalls: [
                AFMToolCall(id: "call", name: "weather", arguments: "{\"city\":\"Paris\"}")
            ]
        )

        let suffix = try AFMDwarfStarToolCodec.assistantReplaySuffix(for: message)
        XCTAssertTrue(suffix.hasPrefix("\n\n<｜DSML｜tool_calls>"))
        XCTAssertTrue(suffix.hasSuffix("<｜end▁of▁sentence｜>"))
    }

    func testAssistantTextReplayEndsWithConversationBoundary() throws {
        let message = AFMMessage(role: .assistant, content: [.text("Done")])
        XCTAssertEqual(
            try AFMDwarfStarToolCodec.assistantReplaySuffix(for: message),
            "<｜end▁of▁sentence｜>"
        )
    }

}
