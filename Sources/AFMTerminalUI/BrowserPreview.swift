import AppKit
import Foundation
import WebKit

public struct TUIBrowserNavigationBoundary: Equatable, Sendable {
    public let artifactURL: URL

    public init(artifactURL: URL) {
        self.artifactURL = artifactURL.standardizedFileURL
    }

    public func allows(_ candidate: URL?) -> Bool {
        guard let candidate else { return false }
        if candidate.scheme?.lowercased() == "about" {
            return candidate.absoluteString == "about:blank" || candidate.absoluteString == "about:srcdoc"
        }
        return candidate.isFileURL && candidate.standardizedFileURL == artifactURL
    }
}

@MainActor
private final class TUIBrowserPreviewController: NSObject, NSWindowDelegate, WKNavigationDelegate {
    private let boundary: TUIBrowserNavigationBoundary
    private let window: NSWindow
    private let webView: WKWebView

    init(artifactURL: URL) {
        boundary = TUIBrowserNavigationBoundary(artifactURL: artifactURL)
        let configuration = WKWebViewConfiguration()
        configuration.websiteDataStore = .nonPersistent()
        configuration.preferences.javaScriptCanOpenWindowsAutomatically = false
        webView = WKWebView(frame: .zero, configuration: configuration)
        window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 960, height: 720),
            styleMask: [.titled, .closable, .miniaturizable, .resizable],
            backing: .buffered,
            defer: false
        )
        super.init()
        webView.navigationDelegate = self
        window.delegate = self
        window.title = "AFM TUI Preview"
        window.contentView = webView
        window.center()
    }

    func show(artifactURL: URL) {
        let application = NSApplication.shared
        application.setActivationPolicy(.regular)
        window.makeKeyAndOrderFront(nil)
        application.activate(ignoringOtherApps: true)
        webView.loadFileURL(artifactURL, allowingReadAccessTo: artifactURL)
    }

    func windowWillClose(_ notification: Notification) {
        NSApplication.shared.terminate(nil)
    }

    func webView(
        _ webView: WKWebView,
        decidePolicyFor navigationAction: WKNavigationAction
    ) async -> WKNavigationActionPolicy {
        boundary.allows(navigationAction.request.url) ? .allow : .cancel
    }

    func webView(
        _ webView: WKWebView,
        decidePolicyFor navigationResponse: WKNavigationResponse
    ) async -> WKNavigationResponsePolicy {
        boundary.allows(navigationResponse.response.url) ? .allow : .cancel
    }
}

public enum TUIBrowserPreview {
    @MainActor
    public static func run(artifactURL: URL) throws {
        try TUIArtifactActions.preflightRegularFile(at: artifactURL, maximumBytes: 10_000_000)
        let controller = TUIBrowserPreviewController(artifactURL: artifactURL)
        controller.show(artifactURL: artifactURL)
        withExtendedLifetime(controller) {
            NSApplication.shared.run()
        }
    }
}
