import Foundation
import XCTest
@testable import AFMKitMLX

final class AFMMLXImageInputPolicyTests: XCTestCase {
    func testNilImageURLProducesNoImagePlan() {
        let plan = AFMMLXImageInputPolicy.plan(
            imageURL: nil,
            cacheDirectory: URL(fileURLWithPath: "/tmp/cache"),
            uniqueSuffix: "abc"
        )

        XCTAssertEqual(plan, .none)
        XCTAssertFalse(plan.hasImages)
    }

    func testLocalFileURLPassesThroughWithoutCachePlan() throws {
        let imageURL = URL(fileURLWithPath: "/Users/test/image.png")
        let plan = AFMMLXImageInputPolicy.plan(
            imageURL: imageURL,
            cacheDirectory: URL(fileURLWithPath: "/tmp/cache"),
            uniqueSuffix: "abc"
        )

        XCTAssertEqual(plan, .localFile(imageURL))
        XCTAssertTrue(plan.hasImages)
    }

    func testRemoteURLCreatesCacheDestination() throws {
        let imageURL = try XCTUnwrap(URL(string: "https://example.com/path/photo.PNG?token=1"))
        let cacheDirectory = URL(fileURLWithPath: "/tmp/cache")
        let plan = AFMMLXImageInputPolicy.plan(
            imageURL: imageURL,
            cacheDirectory: cacheDirectory,
            uniqueSuffix: "20260728-abcdef"
        )

        guard case .remoteImage(let cachePlan) = plan else {
            return XCTFail("Expected remote image plan")
        }
        XCTAssertEqual(cachePlan.sourceURL, imageURL)
        XCTAssertEqual(cachePlan.cacheDirectory, cacheDirectory)
        XCTAssertEqual(
            cachePlan.destinationURL.path,
            "/tmp/cache/WebImage_20260728-abcdef.png"
        )
        XCTAssertTrue(plan.hasImages)
    }

    func testRemoteDetectionIsCaseInsensitiveAndDefaultsMissingExtensionToJPG() throws {
        let imageURL = try XCTUnwrap(URL(string: "HTTP://example.com/image"))

        XCTAssertTrue(AFMMLXImageInputPolicy.isRemoteImageURL(imageURL))
        XCTAssertEqual(AFMMLXImageInputPolicy.fileExtension(for: imageURL), "jpg")
        XCTAssertEqual(
            AFMMLXImageInputPolicy.cacheFileName(for: imageURL, uniqueSuffix: " "),
            "WebImage_image.jpg"
        )
    }
}
