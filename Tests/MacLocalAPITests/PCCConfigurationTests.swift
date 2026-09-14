import AFMKit
import AFMKitCore
import AFMKitFoundationModels
@testable import AFMServer
import XCTest

final class PCCConfigurationTests: XCTestCase {
    func testUnsupportedMacOSFailsWithActionableMessage() {
        for majorVersion in [15, 25, 26] {
            XCTAssertThrowsError(try PCCConfiguration.requireSupportedRuntime(
                version: .init(majorVersion: majorVersion, minorVersion: 9, patchVersion: 0))) { error in
                XCTAssertTrue(error.localizedDescription.contains("requires macOS 27 or later"))
                XCTAssertTrue(error.localizedDescription.contains("on-device Foundation Models"))
            }
        }
        XCTAssertNoThrow(try PCCConfiguration.requireSupportedRuntime(
            version: .init(majorVersion: 27, minorVersion: 0, patchVersion: 0)))
        XCTAssertNoThrow(try PCCConfiguration.requireSupportedRuntime(
            version: .init(majorVersion: 28, minorVersion: 0, patchVersion: 0)))
    }

    @available(macOS 27.0, *)
    private func snapshot(
        entitlement: Bool = true,
        availability: AFMFoundationNativeAvailability = .available,
        localeSupported: Bool = true,
        quotaReached: Bool = false
    ) -> AFMFoundationPrivateCloudComputeSnapshot {
        .init(hasEntitlement: entitlement,
              entitlement: AFMFoundationNativeProviderCapabilities.privateCloudComputeEntitlement,
              availability: availability, localeIdentifier: "en_CA", localeSupported: localeSupported,
              quotaStatus: quotaReached ? "limitReached" : "belowLimit", quotaIsLimitReached: quotaReached,
              quotaLimitDetail: quotaReached ? "Try after quota resets" : nil)
    }

    func testConfigurationSelectsPCCAndPreservesReasoningAndInstructions() throws {
        guard #available(macOS 27.0, *) else { throw XCTSkip("macOS 27 required") }
        let configuration = PCCConfiguration(instructions: "Be concise", reasoning: .deep)
        XCTAssertEqual(configuration.providerConfiguration.values["systemPrompt"], .string("Be concise"))
        XCTAssertEqual(configuration.providerConfiguration.values["reasoningLevel"], .string("deep"))
        let model = try configuration.makeModel()
        XCTAssertEqual(model.descriptor.modelID.rawValue, "apple.private-cloud-compute")
        XCTAssertTrue(model.descriptor.requiresNetwork)
        // Constructing a model/engine must neither bypass the entitlement nor generate a request.
        let engine = try configuration.makeEngine()
        guard case .provider(let provider, let modelID) = engine.backend else {
            return XCTFail("PCC must use the provider route")
        }
        XCTAssertEqual(provider, AFMFoundationProviderFactory.providerID)
        XCTAssertEqual(modelID, model.descriptor.modelID)
    }

    func testStatusRequiresEntitlementLocaleAndQuotaEvenWhenModelIsAvailable() throws {
        guard #available(macOS 27.0, *) else { throw XCTSkip("macOS 27 required") }
        XCTAssertTrue(PCCStatus(snapshot: snapshot()).available)
        let missing = PCCStatus(snapshot: snapshot(entitlement: false))
        XCTAssertFalse(missing.available)
        XCTAssertEqual(missing.reason, "missingEntitlement")
        let locale = PCCStatus(snapshot: snapshot(localeSupported: false))
        XCTAssertFalse(locale.available)
        XCTAssertEqual(locale.reason, "unsupportedLocale")
        let quota = PCCStatus(snapshot: snapshot(quotaReached: true))
        XCTAssertFalse(quota.available)
        XCTAssertEqual(quota.reason, "quotaLimitReached")
        XCTAssertEqual(quota.detail, "Try after quota resets")
    }

    func testStatusPreservesSystemFailureAndRoundTripsJSON() throws {
        guard #available(macOS 27.0, *) else { throw XCTSkip("macOS 27 required") }
        let status = PCCStatus(snapshot: snapshot(availability: .unavailable(reason: "systemNotReady", detail: "Enable Apple Intelligence")))
        XCTAssertFalse(status.available)
        XCTAssertEqual(status.reason, "systemNotReady")
        XCTAssertEqual(status.detail, "Enable Apple Intelligence")
        XCTAssertEqual(try JSONDecoder().decode(PCCStatus.self, from: JSONEncoder().encode(status)), status)
    }

    func testProviderErrorsDistinguishInvalidInputFromServiceFailure() {
        XCTAssertEqual(AFMProviderHTTPError(.invalidRequest("bad input")).status, .badRequest)
        XCTAssertEqual(AFMProviderHTTPError(.unsupportedCapability("tools")).status, .badRequest)
        XCTAssertEqual(AFMProviderHTTPError(.unavailable("quota reached")).status, .serviceUnavailable)
        XCTAssertEqual(AFMProviderHTTPError(.generationFailed("service unavailable")).status, .badGateway)
        XCTAssertEqual(AFMProviderHTTPError(.generationFailed("network failure")).type, "provider_error")
    }
}
