// Compatibility facade for applications that adopted the original maclocal-api
// macOS 27 product before AFMKit became an independent package.
#if compiler(>=6.4)
@_exported import AFMKitApple
@_exported import AFMKitFoundationModelsMLX
#endif
