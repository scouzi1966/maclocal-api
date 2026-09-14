import AFMKitCore
import Vapor

/// Preserve provider failures at the HTTP boundary instead of reporting every
/// PCC availability or service failure as an invalid MLX request.
struct AFMProviderHTTPError {
    let status: HTTPStatus
    let type: String

    init(_ error: AFMError) {
        switch error {
        case .invalidRequest, .unsupportedCapability:
            status = .badRequest
            type = "invalid_request_error"
        case .unavailable, .loadingFailed:
            status = .serviceUnavailable
            type = "provider_unavailable"
        case .generationFailed:
            status = .badGateway
            type = "provider_error"
        case .modelNotFound:
            status = .notFound
            type = "model_not_found"
        case .providerNotRegistered, .providerAlreadyRegistered:
            status = .internalServerError
            type = "provider_error"
        }
    }
}
