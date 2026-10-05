# Splash integration

AFM exposes both Splash interfaces:

- `afm splash …` forwards to the bundled upstream CLI, including its Python
  frontend and native C++/Metal engine. It preserves all native flags, stdio,
  environment, working directory, PID, exit status, and signals.
- `afm splash-api --model /path/to/package` serves text chat and streaming through
  AFM's existing HTTP API using the standalone `AFMKitSplash` provider. Swift
  handles tokenization and talks directly to the native engine's binary protocol;
  no Python frontend or second HTTP server is started in this mode.

## Release pin and build

Both modes use **Splash 1.0**, upstream commit
`c675ed23e6942b5353961246e68b08cd63fb4ee9`. AFMKit owns the version, protocol,
archive URL, and SHA-256 in `Sources/AFMKitSplash/Resources/splash-release.json`.
The archive digest is
`dc752f0aab8419c46fe2803a0e7059c1515e5df48def11b9d517bbbf1fb2dddc`.

`make build`, `make debug`, and `build.sh` stage the verified precompiled release
beside AFM as `splash-runtime/`. No Splash source build, model download, or model
execution occurs during staging. For a direct SwiftPM build, stage explicitly:

```sh
Scripts/swiftpm-reliable.sh build -c release --product afm
Scripts/stage-splash-runtime.sh "$(dirname "$(Scripts/find-afm-binary.sh release)")"
```

Release archives, native wheels, Homebrew formula generation, and local installs
carry that directory and `AFMKit_AFMKitSplash.bundle`. The complete upstream
runtime adds approximately 220 MB unpacked (including its Python runtime and
license). It is independent of model weights. The default command never searches
PATH for an arbitrary Splash version or runs an unpinned Homebrew install.

This paired development PR requires [AFMKit #126](https://github.com/scouzi1966/AFMKit/pull/126). Before merging,
publish a qualified AFMKit version and bump AFM's single exact AFMKit pin and lock
using the usual release workflow. Until then use the supported local workspace:

```sh
MACLOCAL_AFMKIT_PATH=/path/to/AFMKit-with-splash make build
```

The tracked AFMKit version remains the existing published release during paired
development; an ordinary release build cannot consume the new provider until
that exact-version bump. No unpublished or unqualified tag is fabricated.

## CLI

```sh
afm splash --help
afm splash --version
afm help splash
```

All arguments after `splash`, including `--help`, `--version`, future flags, and
`--`, pass unchanged. `afm help splash` shows AFM's integration help. The default
Splash server address is `127.0.0.1:8000`.

When ready to run a model, use Splash's own interface:

```sh
afm splash serve --model incoai/Qwen3.8-27B-Splash --max-context 32K
```

An explicit development override remains available:

```sh
AFM_SPLASH_EXECUTABLE=/absolute/path/to/splash afm splash --help
```

This deliberately opts out of the bundled release pin for the CLI only. It is
an executable path, never a shell command. Invalid paths and recursive references
to AFM itself fail with a readable error. The native AFMKit provider always uses
the verified bundled release.

## AFM API through the native provider

When ready to run a model:

```sh
afm splash-api --model /absolute/path/to/existing/splash-package \
  --model-id splash-local --port 9999 --max-context 32768
```

The package must already contain `target/`, `draft/`, and `tokenizer/` directories.
This command does not automatically download weights. It registers
`AFMSplashProviderFactory` and passes the resulting `AnyAFMModel` to the existing
AFM server, exposing its usual `/v1/models` and `/v1/chat/completions` routes.

The initial native adapter supports text chat, streaming, temperature, top-p,
top-k (maximum 32), seed, max tokens, usage, and native speculative decoding.
It serializes requests and supports cancellation, timeout, and process cleanup.
Thinking is disabled. Tools, reasoning, structured output, images, custom stops,
logprobs, and unsupported penalties are explicitly rejected; use `afm splash`
for the full upstream feature set.

Both commands fail gracefully before starting a runtime on macOS versions older
than 26.4. Splash also requires Apple Silicon M3 or newer and sufficient memory;
its native engine enforces those hardware requirements.

## Validation without models or GPU

The provider's CPU-only test harness compiles the actual Core/Splash sources and
Swift Tokenizers without MLX or the HTTP server. An inert protocol peer tests
framing, streaming, failures, cancellation, and shutdown. Staging tests check
archive hashes and altered runtime files. The CLI process test uses an inert
executable and does not need a real Splash installation:

```sh
python3 Scripts/test-splash-cli.py --binary /path/to/compiled/afm
```

The provider's integration notes document `Scripts/test-splash-cpu.sh`. Live
model/tokenizer parity, inference performance, and the full AFM executable still
require separate qualification. No real model was loaded during this work.

Sources: [Splash 1.0](https://github.com/incoai/splash/releases/tag/1.0),
[native protocol](https://github.com/incoai/splash/blob/1.0/runtime/engine/Protocol.hpp).
