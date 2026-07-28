import XCTest
@testable import AFMKitMLX

final class AFMMLXRuntimeMemoryControllerTests: XCTestCase {
    func testOptimalGPUCacheLimitScalesWithUnifiedMemory() {
        let gb = UInt64(AFMMLXRuntimeMemoryController.bytesPerGB)

        XCTAssertEqual(
            AFMMLXRuntimeMemoryController.optimalGPUCacheLimitMB(physicalMemoryBytes: 8 * gb),
            128
        )
        XCTAssertEqual(
            AFMMLXRuntimeMemoryController.optimalGPUCacheLimitMB(physicalMemoryBytes: 16 * gb),
            256
        )
        XCTAssertEqual(
            AFMMLXRuntimeMemoryController.optimalGPUCacheLimitMB(physicalMemoryBytes: 32 * gb),
            512
        )
        XCTAssertEqual(
            AFMMLXRuntimeMemoryController.optimalGPUCacheLimitMB(physicalMemoryBytes: 64 * gb),
            1024
        )
    }

    func testOptimalGPUCacheLimitBytesUsesSameThresholds() {
        let gb = UInt64(AFMMLXRuntimeMemoryController.bytesPerGB)

        XCTAssertEqual(
            AFMMLXRuntimeMemoryController.optimalGPUCacheLimitBytes(physicalMemoryBytes: 24 * gb),
            512 * AFMMLXRuntimeMemoryController.bytesPerMB
        )
    }

    func testWiredLimitUsesConfiguredPercentOfWorkingSet() {
        XCTAssertEqual(
            AFMMLXRuntimeMemoryController.wiredLimitBytes(
                maxRecommendedWorkingSetSize: 10_000,
                percent: 90
            ),
            9_000
        )
        XCTAssertEqual(
            AFMMLXRuntimeMemoryController.wiredLimitBytes(
                maxRecommendedWorkingSetSize: 10_000,
                percent: 75
            ),
            7_500
        )
    }
}
