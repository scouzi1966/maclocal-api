import NIOCore
import Vapor

final class AFMRequestConnection: @unchecked Sendable {
    private let closeFuture: EventLoopFuture<Void>

    init(closeFuture: EventLoopFuture<Void>) {
        self.closeFuture = closeFuture
    }

    func whenClosed(_ callback: @escaping @Sendable () -> Void) {
        closeFuture.whenComplete { _ in callback() }
    }
}

private struct AFMRequestConnectionKey: StorageKey {
    typealias Value = AFMRequestConnection
}

extension Request {
    func onAFMConnectionClose(_ callback: @escaping @Sendable () -> Void) {
        storage[AFMRequestConnectionKey.self]?.whenClosed(callback)
    }

    func attachAFMConnection(_ connection: AFMRequestConnection) {
        storage[AFMRequestConnectionKey.self] = connection
    }
}
