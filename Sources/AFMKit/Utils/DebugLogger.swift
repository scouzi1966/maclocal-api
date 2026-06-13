import Foundation

/// Centralized debug logging utility for MacLocal API
///
/// Debug logging is controlled by the AFM_DEBUG environment variable.
/// When AFM_DEBUG=1, debug messages are printed to stdout with a "DEBUG:" prefix.
/// This allows for consistent debugging across all components without affecting normal operation.
public struct DebugLogger {

    /// Logs a debug message if AFM_DEBUG environment variable is set to "1"
    ///
    /// - Parameter message: The debug message to log
    public static func log(_ message: String) {
        if ProcessInfo.processInfo.environment["AFM_DEBUG"] == "1" {
            print("DEBUG: \(message)")
        }
    }

    /// Logs a debug message with additional context information
    ///
    /// - Parameters:
    ///   - message: The primary debug message
    ///   - context: Additional context information (e.g., function name, component)
    public static func log(_ message: String, context: String) {
        if ProcessInfo.processInfo.environment["AFM_DEBUG"] == "1" {
            print("DEBUG [\(context)]: \(message)")
        }
    }
}