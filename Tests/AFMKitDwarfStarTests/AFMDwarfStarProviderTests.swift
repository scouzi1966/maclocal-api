import XCTest
@testable import AFMKitDwarfStar

final class AFMDwarfStarProviderTests: XCTestCase {
    func testProviderContractDescribesInProcessDeviceRuntime() {
        let descriptor = AFMDwarfStarProviderFactory().descriptor

        XCTAssertEqual(descriptor.id, "dwarfstar")
        XCTAssertEqual(descriptor.privacyBoundary, .device)
        XCTAssertEqual(descriptor.metadata["runtime"], .string("in-process-ds4"))
        XCTAssertEqual(descriptor.metadata["execution"], .string("fixed-metal-schedule"))
        XCTAssertTrue(descriptor.configurationKeys.contains("modelPath"))
        XCTAssertTrue(descriptor.configurationKeys.contains("enablePrefixCaching"))
        XCTAssertTrue(descriptor.configurationKeys.contains("maxConcurrent"))
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
        XCTAssertFalse(model.descriptor.capabilities.contains(.reasoning))
        XCTAssertTrue(model.descriptor.capabilities.contains(.streaming))
    }

    func testModelDescriptorPublishesResidentSessionConfiguration() {
        let model = AFMDwarfStarModel(
            modelID: "configured",
            modelPath: "/missing.gguf",
            enablePrefixCaching: true,
            maxConcurrent: 4,
            runtime: AFMDwarfStarRuntimeCoordinator()
        )

        XCTAssertEqual(model.descriptor.metadata["enablePrefixCaching"], .bool(true))
        XCTAssertEqual(model.descriptor.metadata["maxConcurrent"], .integer(4))
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

}
