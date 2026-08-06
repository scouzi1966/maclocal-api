import Foundation
import CDwarfStar

/// In-process access to DwarfStar's fixed-schedule DeepSeek runtime.
public enum AFMDwarfStarRuntime {
    public static var metalSourceDirectory: URL? {
        if let override = ProcessInfo.processInfo.environment["AFM_DWARFSTAR_METAL_SOURCE_ROOT"],
           !override.isEmpty {
            return URL(fileURLWithPath: override, isDirectory: true)
        }
        if let directory = Bundle.module.url(forResource: "metal", withExtension: nil) {
            return directory
        }
        // SwiftPM `.process` flattens this resource directory. Processing is
        // intentional because it dereferences the vendor symlink into portable
        // product resources instead of preserving a broken relative link.
        if let resources = Bundle.module.resourceURL,
           FileManager.default.fileExists(
               atPath: resources.appendingPathComponent("flash_attn.metal").path)
        {
            return resources
        }
        return nil
    }

    public static var backendName: String {
        String(cString: ds4_backend_name(DS4_BACKEND_METAL))
    }
}
