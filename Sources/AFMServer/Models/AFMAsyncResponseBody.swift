import Foundation
import Vapor

final class AFMAsyncResponseBody: @unchecked Sendable {
    typealias Callback = @Sendable (AsyncBodyStreamWriter) async throws -> Void

    static let shared = AFMAsyncResponseBody()

    private let lock = NSLock()
    private var callbacks: [ObjectIdentifier: Callback] = [:]

    private init() {}

    func install(on response: Response, callback: @escaping Callback) {
        let identifier = ObjectIdentifier(response)
        lock.withLock { callbacks[identifier] = callback }
        response.body = .init(asyncStream: { writer in
            defer { self.remove(identifier) }
            try await callback(writer)
        })
    }

    func take(for response: Response) -> Callback? {
        lock.withLock { callbacks.removeValue(forKey: ObjectIdentifier(response)) }
    }

    private func remove(_ identifier: ObjectIdentifier) {
        _ = lock.withLock { callbacks.removeValue(forKey: identifier) }
    }
}

extension Response {
    func useAFMAsyncBody(
        _ callback: @escaping AFMAsyncResponseBody.Callback
    ) {
        AFMAsyncResponseBody.shared.install(on: self, callback: callback)
    }
}
