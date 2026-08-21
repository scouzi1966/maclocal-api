import Foundation
import NIOCore
import NIOHTTP1
import NIOPosix
import Vapor

final class AFMHTTPServer: Vapor.Server, @unchecked Sendable {
    private let application: Application
    private let responder: Responder
    private let eventLoopGroup: EventLoopGroup
    private let configuration: HTTPServer.Configuration
    private let stateLock = NSLock()
    private let shutdownPromise: EventLoopPromise<Void>
    private var serverChannel: Channel?

    init(
        application: Application,
        responder: Responder,
        configuration: HTTPServer.Configuration,
        eventLoopGroup: EventLoopGroup
    ) {
        self.application = application
        self.responder = responder
        self.configuration = configuration
        self.eventLoopGroup = eventLoopGroup
        self.shutdownPromise = eventLoopGroup.next().makePromise(of: Void.self)
    }

    var localAddress: SocketAddress? {
        stateLock.withLock { serverChannel?.localAddress }
    }

    var onShutdown: EventLoopFuture<Void> {
        shutdownPromise.futureResult
    }

    @available(*, noasync, message: "Use the async start() method instead.")
    func start(address: BindAddress?) throws {
        try finishStarting(bootstrap().bind(to: resolvedAddress(address)).wait())
    }

    func start(address: BindAddress?) async throws {
        try finishStarting(await bootstrap().bind(to: resolvedAddress(address)).get())
    }

    @available(*, noasync, message: "Use the async shutdown() method instead.")
    func shutdown() {
        guard let channel = stateLock.withLock({ serverChannel }) else { return }
        try? channel.close().wait()
    }

    func shutdown() async {
        guard let channel = stateLock.withLock({ serverChannel }) else { return }
        try? await channel.close().get()
    }

    private func bootstrap() -> ServerBootstrap {
        ServerBootstrap(group: eventLoopGroup)
            .serverChannelOption(.backlog, value: Int32(configuration.backlog))
            .serverChannelOption(
                .socketOption(.so_reuseaddr),
                value: configuration.reuseAddress ? Int32(1) : Int32(0)
            )
            .childChannelInitializer { [application, responder] channel in
                channel.pipeline.configureHTTPServerPipeline(withErrorHandling: true).flatMap {
                    channel.pipeline.addHandler(
                        AFMHTTPChannelHandler(
                            application: application,
                            responder: responder
                        )
                    )
                }
            }
            .childChannelOption(
                .socketOption(.so_reuseaddr),
                value: configuration.reuseAddress ? Int32(1) : Int32(0)
            )
            .childChannelOption(
                .tcpOption(.tcp_nodelay),
                value: configuration.tcpNoDelay ? Int32(1) : Int32(0)
            )
            .childChannelOption(.maxMessagesPerRead, value: 1)
    }

    private func resolvedAddress(_ override: BindAddress?) -> BindAddress {
        override ?? configuration.address
    }

    private func finishStarting(_ channel: Channel) throws {
        let alreadyStarted = stateLock.withLock { () -> Bool in
            guard serverChannel == nil else { return true }
            serverChannel = channel
            return false
        }
        guard !alreadyStarted else {
            try? channel.close().wait()
            throw AFMHTTPServerError.alreadyStarted
        }
        channel.closeFuture.cascade(to: shutdownPromise)
        application.logger.notice("AFM HTTP server started on \(channel.localAddress?.description ?? "unknown")")
    }
}

private enum AFMHTTPServerError: Error {
    case alreadyStarted
}

private extension ServerBootstrap {
    func bind(to address: BindAddress) -> EventLoopFuture<Channel> {
        switch address {
        case .hostname(let hostname, let port):
            bind(
                host: hostname ?? HTTPServer.Configuration.defaultHostname,
                port: port ?? HTTPServer.Configuration.defaultPort
            )
        case .unixDomainSocket(let path):
            bind(unixDomainSocketPath: path)
        }
    }
}

private final class AFMHTTPChannelHandler: ChannelInboundHandler, @unchecked Sendable {
    typealias InboundIn = HTTPServerRequestPart
    typealias OutboundOut = HTTPServerResponsePart

    private static let maximumRequestBodyBytes = 100 * 1_024 * 1_024

    private struct PendingRequest {
        let head: HTTPRequestHead
        var body: ByteBuffer
        var isTooLarge: Bool
    }

    private let application: Application
    private let responder: Responder
    private var pendingRequest: PendingRequest?

    init(application: Application, responder: Responder) {
        self.application = application
        self.responder = responder
    }

    func channelRead(context: ChannelHandlerContext, data: NIOAny) {
        switch unwrapInboundIn(data) {
        case .head(let head):
            pendingRequest = PendingRequest(
                head: head,
                body: context.channel.allocator.buffer(capacity: 0),
                isTooLarge: false
            )
        case .body(var body):
            guard var pendingRequest else { return }
            if pendingRequest.body.readableBytes + body.readableBytes
                > Self.maximumRequestBodyBytes
            {
                pendingRequest.isTooLarge = true
            } else if !pendingRequest.isTooLarge {
                pendingRequest.body.writeBuffer(&body)
            }
            self.pendingRequest = pendingRequest
        case .end:
            guard let pendingRequest else { return }
            self.pendingRequest = nil
            if pendingRequest.isTooLarge {
                serialize(
                    response: Response(status: .payloadTooLarge),
                    requestHead: pendingRequest.head,
                    context: context
                )
                return
            }
            respond(to: pendingRequest, context: context)
        }
    }

    func errorCaught(context: ChannelHandlerContext, error: Error) {
        application.logger.debug("AFM HTTP connection error: \(error)")
        context.close(promise: nil)
    }

    private func respond(
        to pendingRequest: PendingRequest,
        context: ChannelHandlerContext
    ) {
        let request = Request(
            application: application,
            method: pendingRequest.head.method,
            url: .init(path: pendingRequest.head.uri),
            version: pendingRequest.head.version,
            headersNoUpdate: pendingRequest.head.headers,
            collectedBody: pendingRequest.body,
            remoteAddress: context.channel.remoteAddress,
            peerCertificateChain: nil,
            logger: application.logger,
            byteBufferAllocator: context.channel.allocator,
            on: context.eventLoop
        )
        request.attachAFMConnection(
            AFMRequestConnection(closeFuture: context.channel.closeFuture)
        )

        let box = NIOLoopBound((context, self), eventLoop: context.eventLoop)
        responder.respond(to: request).hop(to: context.eventLoop).whenComplete { result in
            let (context, handler) = box.value
            switch result {
            case .success(let response):
                handler.serialize(
                    response: response,
                    requestHead: pendingRequest.head,
                    context: context
                )
            case .failure(let error):
                handler.application.logger.error("AFM HTTP responder error: \(error)")
                handler.serialize(
                    response: Response(status: .internalServerError),
                    requestHead: pendingRequest.head,
                    context: context
                )
            }
        }
    }

    private func serialize(
        response: Response,
        requestHead: HTTPRequestHead,
        context: ChannelHandlerContext
    ) {
        let keepAlive = requestHead.isKeepAlive
        if let callback = AFMAsyncResponseBody.shared.take(for: response) {
            var headers = response.headers
            headers.remove(name: .contentLength)
            headers.replaceOrAdd(name: .connection, value: keepAlive ? "keep-alive" : "close")
            context.write(
                wrapOutboundOut(.head(.init(
                    version: response.version,
                    status: response.status,
                    headers: headers
                ))),
                promise: nil
            )
            context.flush()

            let writer = AFMChannelResponseBodyWriter(
                channel: context.channel,
                closeAfterResponse: !keepAlive
            )
            Task {
                do {
                    try await callback(writer)
                } catch {
                    try? await writer.write(.error(error))
                }
            }
            return
        }

        let box = NIOLoopBound((context, self), eventLoop: context.eventLoop)
        response.body.collect(on: context.eventLoop).whenComplete { result in
            let (context, handler) = box.value
            switch result {
            case .success(let body):
                handler.serializeCollectedBody(
                    body,
                    response: response,
                    requestHead: requestHead,
                    keepAlive: keepAlive,
                    context: context
                )
            case .failure(let error):
                handler.application.logger.error("AFM HTTP response body error: \(error)")
                context.close(promise: nil)
            }
        }
    }

    private func serializeCollectedBody(
        _ body: ByteBuffer?,
        response: Response,
        requestHead: HTTPRequestHead,
        keepAlive: Bool,
        context: ChannelHandlerContext
    ) {
        var headers = response.headers
        let suppressBody = requestHead.method == .HEAD || response.status == .noContent
        let readableBytes = suppressBody ? 0 : (body?.readableBytes ?? 0)
        headers.remove(name: .transferEncoding)
        headers.replaceOrAdd(name: .contentLength, value: String(readableBytes))
        headers.replaceOrAdd(name: .connection, value: keepAlive ? "keep-alive" : "close")
        context.write(
            wrapOutboundOut(.head(.init(
                version: response.version,
                status: response.status,
                headers: headers
            ))),
            promise: nil
        )
        if !suppressBody, let body, body.readableBytes > 0 {
            context.write(wrapOutboundOut(.body(.byteBuffer(body))), promise: nil)
        }
        let completion = context.eventLoop.makePromise(of: Void.self)
        let contextBox = NIOLoopBound(context, eventLoop: context.eventLoop)
        completion.futureResult.whenComplete { _ in
            if !keepAlive {
                contextBox.value.close(promise: nil)
            }
        }
        context.writeAndFlush(wrapOutboundOut(.end(nil)), promise: completion)
    }
}

private final class AFMChannelResponseBodyWriter: AsyncBodyStreamWriter, @unchecked Sendable {
    private let channel: Channel
    private let closeAfterResponse: Bool
    private let lock = NSLock()
    private var isComplete = false

    init(channel: Channel, closeAfterResponse: Bool) {
        self.channel = channel
        self.closeAfterResponse = closeAfterResponse
    }

    func write(_ result: BodyStreamResult) async throws {
        switch result {
        case .buffer(let buffer):
            guard !lock.withLock({ isComplete }) else {
                throw AFMHTTPBodyWriterError.alreadyComplete
            }
            try await channel.writeAndFlush(
                HTTPServerResponsePart.body(.byteBuffer(buffer))
            ).get()
        case .end:
            let shouldFinish = lock.withLock { () -> Bool in
                guard !isComplete else { return false }
                isComplete = true
                return true
            }
            guard shouldFinish else { throw AFMHTTPBodyWriterError.alreadyComplete }
            try await channel.writeAndFlush(HTTPServerResponsePart.end(nil)).get()
            if closeAfterResponse {
                try? await channel.close().get()
            }
        case .error(let error):
            lock.withLock { isComplete = true }
            try? await channel.close().get()
            throw error
        }
    }
}

private enum AFMHTTPBodyWriterError: Error {
    case alreadyComplete
}

private struct AFMHTTPServerKey: StorageKey {
    typealias Value = AFMHTTPServer
}

extension Application {
    var afmHTTPServer: AFMHTTPServer? {
        storage[AFMHTTPServerKey.self]
    }

    func useAFMHTTPServer() {
        let server = AFMHTTPServer(
            application: self,
            responder: responder.current,
            configuration: http.server.configuration,
            eventLoopGroup: eventLoopGroup
        )
        storage[AFMHTTPServerKey.self] = server
        servers.use { _ in server }
    }
}
