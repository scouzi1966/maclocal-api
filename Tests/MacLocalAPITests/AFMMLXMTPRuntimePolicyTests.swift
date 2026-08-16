import Foundation
@testable import AFMKitMLX
import XCTest

final class AFMMLXMTPRuntimePolicyTests: XCTestCase {
    func testFailedMTPSetupCannotReuseBaseOnlyLoadedStateOnRetry() {
        XCTAssertFalse(
            AFMMLXMTPRuntimePolicy.canReuseLoadedModel(
                loadedModelID: "qwen",
                requestedModelID: "qwen",
                mtpEnabled: true,
                bindingModelID: nil
            )
        )
    }

    func testBaseOnlyLoadedStateRemainsReusableWhenMTPIsDisabled() {
        XCTAssertTrue(
            AFMMLXMTPRuntimePolicy.canReuseLoadedModel(
                loadedModelID: "qwen",
                requestedModelID: "qwen",
                mtpEnabled: false,
                bindingModelID: nil
            )
        )
    }

    func testBindingFromAnotherModelIsNeverUsable() {
        XCTAssertFalse(
            AFMMLXMTPRuntimePolicy.bindingIsUsable(
                for: "model-b",
                mtpEnabled: true,
                bindingModelID: "model-a"
            )
        )
        XCTAssertFalse(
            AFMMLXMTPRuntimePolicy.canReuseLoadedModel(
                loadedModelID: "model-b",
                requestedModelID: "model-b",
                mtpEnabled: true,
                bindingModelID: "model-a"
            )
        )
    }

    func testDisabledMTPUsesBackgroundRatherThanSynchronousDownload() {
        XCTAssertFalse(
            AFMMLXMTPRuntimePolicy.allowSynchronousSidecarDownload(mtpEnabled: false)
        )
        XCTAssertTrue(
            AFMMLXMTPRuntimePolicy.shouldPrefetchInBackground(
                mtpEnabled: false,
                resolvedSidecar: nil,
                automaticRepositoryID: "mlx-community/Qwen3.8-27B-MTP-4bit"
            )
        )
    }

    func testExplicitMTPAllowsSynchronousDownloadAndNoBackgroundDuplicate() {
        XCTAssertTrue(
            AFMMLXMTPRuntimePolicy.allowSynchronousSidecarDownload(mtpEnabled: true)
        )
        XCTAssertFalse(
            AFMMLXMTPRuntimePolicy.shouldPrefetchInBackground(
                mtpEnabled: true,
                resolvedSidecar: nil,
                automaticRepositoryID: "mlx-community/Qwen3.8-27B-MTP-4bit"
            )
        )
    }

    func testDirectSidecarRequiresQuantizationMetadata() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: directory) }
        let sidecar = directory.appendingPathComponent("head.safetensors")
        try Data().write(to: sidecar)

        XCTAssertFalse(AFMMLXMTPRuntimePolicy.directSidecarHasRequiredMetadata(sidecar))

        let config: [String: Any] = [
            "quantization_config": ["group_size": 32, "bits": 4, "mode": "mxfp4"]
        ]
        try JSONSerialization.data(withJSONObject: config)
            .write(to: directory.appendingPathComponent("config.json"))
        XCTAssertTrue(AFMMLXMTPRuntimePolicy.directSidecarHasRequiredMetadata(sidecar))
    }
}
