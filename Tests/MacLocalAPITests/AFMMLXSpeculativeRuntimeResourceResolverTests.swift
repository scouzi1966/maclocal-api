import XCTest
import AFMKitMLX

final class AFMMLXSpeculativeRuntimeResourceResolverTests: XCTestCase {
    private func makeQwen38Config(
        quantization: [String: Any]? = ["mode": "affine", "bits": 4],
        hiddenSize: Int = 5_120,
        layerCount: Int = 64,
        mtpLayerCount: Int = 1
    ) throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        var config: [String: Any] = [
            "model_type": "qwen3_5",
            "text_config": [
                "model_type": "qwen3_5_text",
                "hidden_size": hiddenSize,
                "num_hidden_layers": layerCount,
                "mtp_num_hidden_layers": mtpLayerCount
            ]
        ]
        if let quantization {
            config["quantization"] = quantization
        }
        let data = try JSONSerialization.data(withJSONObject: config)
        try data.write(to: directory.appendingPathComponent("config.json"))
        addTeardownBlock { try? FileManager.default.removeItem(at: directory) }
        return directory
    }

    func testCurrentLoadedModelDirectoryIsUnavailableForMissingOrBlankID() {
        XCTAssertNil(
            AFMMLXSpeculativeRuntimeResourceResolver.currentLoadedModelDirectory(
                loadedModelRepoID: nil,
                repositoryDirectory: { _ in XCTFail("Repository resolver should not be called"); return nil }
            )
        )

        XCTAssertNil(
            AFMMLXSpeculativeRuntimeResourceResolver.currentLoadedModelDirectory(
                loadedModelRepoID: "   ",
                repositoryDirectory: { _ in XCTFail("Repository resolver should not be called"); return nil }
            )
        )
    }

    func testCurrentLoadedModelDirectoryUsesImportedPathDirectly() {
        let path = "/Volumes/edata/models/Qwen3.5"

        let directory = AFMMLXSpeculativeRuntimeResourceResolver.currentLoadedModelDirectory(
            loadedModelRepoID: "  \(path)  ",
            repositoryDirectory: { _ in XCTFail("Repository resolver should not be called"); return nil }
        )

        XCTAssertEqual(directory?.path, path)
    }

    func testCurrentLoadedModelDirectoryResolvesRepositoryID() {
        let expected = URL(fileURLWithPath: "/cache/mlx-community/Qwen3.5")

        let directory = AFMMLXSpeculativeRuntimeResourceResolver.currentLoadedModelDirectory(
            loadedModelRepoID: "  mlx-community/Qwen3.5  ",
            repositoryDirectory: { repoID in
                XCTAssertEqual(repoID, "mlx-community/Qwen3.5")
                return expected
            }
        )

        XCTAssertEqual(directory, expected)
    }

    func testMTPSidecarPathRequiresDirectoryAndExistingSidecar() {
        let directory = URL(fileURLWithPath: "/cache/model", isDirectory: true)
        let expectedPath = "/cache/model/\(AFMMLXSpeculativeRuntimeResourceResolver.mtpSidecarFilename)"

        XCTAssertNil(
            AFMMLXSpeculativeRuntimeResourceResolver.mtpSidecarPath(
                modelDirectory: nil,
                fileExists: { _ in true }
            )
        )

        XCTAssertNil(
            AFMMLXSpeculativeRuntimeResourceResolver.mtpSidecarPath(
                modelDirectory: directory,
                fileExists: { path in
                    XCTAssertEqual(path, expectedPath)
                    return false
                }
            )
        )

        XCTAssertEqual(
            AFMMLXSpeculativeRuntimeResourceResolver.mtpSidecarPath(
                modelDirectory: directory,
                fileExists: { path in
                    XCTAssertEqual(path, expectedPath)
                    return true
                }
            ),
            expectedPath
        )
    }

    func testAutomaticQwen38MTPRepositoryMatchesPublishedQuantization() throws {
        let cases: [([String: Any]?, String)] = [
            (["mode": "affine", "bits": 4], "4bit"),
            (["mode": "affine", "bits": 8], "8bit"),
            (["mode": "mxfp4"], "mxfp4"),
            (["mode": "mxfp8"], "mxfp8"),
            (["mode": "nvfp4"], "nvfp4"),
            (nil, "bf16")
        ]

        for (quantization, suffix) in cases {
            let directory = try makeQwen38Config(quantization: quantization)
            XCTAssertEqual(
                AFMMLXSpeculativeRuntimeResourceResolver.automaticMTPRepositoryID(
                    modelDirectory: directory
                ),
                "mlx-community/Qwen3.8-27B-MTP-\(suffix)"
            )
        }
    }

    func testAutomaticQwen38MTPRepositoryUsesConfigRatherThanDirectoryName() throws {
        let directory = try makeQwen38Config(quantization: ["mode": "mxfp4"])

        XCTAssertEqual(
            AFMMLXSpeculativeRuntimeResourceResolver.automaticMTPRepositoryID(
                modelDirectory: directory
            ),
            "mlx-community/Qwen3.8-27B-MTP-mxfp4"
        )
    }

    func testMTPQuantizationReadsExactCheckpointLayout() throws {
        let directory = try makeQwen38Config(
            quantization: ["mode": "mxfp4", "bits": 4, "group_size": 32]
        )

        XCTAssertEqual(
            AFMMLXSpeculativeRuntimeResourceResolver.mtpQuantization(
                resourceDirectory: directory
            ),
            .init(groupSize: 32, bits: 4, mode: "mxfp4")
        )
    }

    func testAutomaticQwen38MTPRepositoryRejectsIncompatibleArchitecture() throws {
        let wrongShape = try makeQwen38Config(hiddenSize: 4_096)
        let noMTP = try makeQwen38Config(mtpLayerCount: 0)

        XCTAssertNil(
            AFMMLXSpeculativeRuntimeResourceResolver.automaticMTPRepositoryID(
                modelDirectory: wrongShape
            )
        )
        XCTAssertNil(
            AFMMLXSpeculativeRuntimeResourceResolver.automaticMTPRepositoryID(
                modelDirectory: noMTP
            )
        )
    }

    func testRepositorySidecarResolutionPrefersLegacyBundledHead() {
        let directory = URL(fileURLWithPath: "/cache/model", isDirectory: true)
        let legacy = directory.appendingPathComponent("mtp.safetensors").path
        let repository = directory.appendingPathComponent("model.safetensors").path

        XCTAssertEqual(
            AFMMLXSpeculativeRuntimeResourceResolver.mtpSidecarPath(
                resourceDirectory: directory,
                fileExists: { $0 == legacy || $0 == repository }
            ),
            legacy
        )
        XCTAssertEqual(
            AFMMLXSpeculativeRuntimeResourceResolver.mtpSidecarPath(
                resourceDirectory: directory,
                fileExists: { $0 == repository }
            ),
            repository
        )
    }
}
