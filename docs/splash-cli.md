# Splash CLI integration

`afm splash …` runs Splash's existing user-facing CLI. AFM finds the installed
`splash` executable on PATH and replaces its process with it. Splash owns its
arguments, HTTP server, model installation, Python environment, and native
C++/Metal inference runtime.

Install Splash separately using its upstream instructions:

```sh
brew install incoai/tap/splash
afm splash --help
afm splash --version
```

When you want to run a model:

```sh
afm splash serve --model incoai/Qwen3.8-27B-Splash
afm splash serve --model incoai/Qwen3.8-27B-Splash --max-context 32K
```

These are Splash's flags, model package format, and defaults. For example, its
documented server address is `127.0.0.1:8000`, not AFM's usual port 9999. All
arguments after `splash`, including `--help`, `--version`, future flags, and `--`,
are passed unchanged. AFM does not add arguments or automatically install or
upgrade Splash. `afm help splash` shows AFM's integration help even if Splash
isn't installed; `afm splash --help` asks the installed Splash itself.

For a source checkout or custom installation:

```sh
AFM_SPLASH_EXECUTABLE=/absolute/path/to/splash/splash afm splash --help
```

The override is an executable path, not a shell command. Paths with spaces work
when quoted. An invalid override fails rather than selecting a different copy
from PATH. A missing installation produces an actionable error. Pointing the
override at `afm` itself, including through a symlink or hard link, is rejected.

Standard input/output/error, environment variables, current working directory,
terminal ownership, and exit status are inherited. AFM uses `execv`, so signals
sent to the AFM PID go directly to Splash. No shell interpolates the forwarded
arguments. Splash enforces its own hardware, macOS, memory, and model requirements.

## What this validates

The application now has a dependency-free `AFMExternalCLI` target (Foundation
and Darwin only) and a small `SplashCommand` adapter in AFMCLI. Adding this
external runtime requires no changes to AFMKit's provider implementations,
AFMEngine, AFMServer, MLX, DwarfStar, or the pinned dependency versions.

This validates **CLI/process modularity**. It does not make Splash an
`AFMProviderFactory`, route it through AFM's HTTP controllers, or turn it into
an in-process Swift provider. Splash's user-facing CLI still uses Python.

The separate C++ command, `serve-native TARGET_DIRECTORY DRAFT_DIRECTORY
MAX_CONTEXT MAX_MEMORY_BYTES`, is a binary stdin/stdout token protocol, not the
user-facing server CLI. Integrating that protocol into AFMKit would be a separate
provider implementation involving tokenization, prompt formatting, streaming,
cancellation, and lifecycle management. This PR does not claim that integration.

## Validation without models or GPU use

`ExternalCLIInvocationTests` covers executable discovery, custom paths, argument
preservation, invalid installations, and self-recursion. It uses temporary files
only; it never invokes a model or Splash.

```sh
Scripts/swiftpm-reliable.sh test -c release --filter ExternalCLIInvocationTests
python3 Scripts/test-splash-cli.py --binary .build/release/afm
```

The Python regression script supplies a temporary fixture executable through
`AFM_SPLASH_EXECUTABLE`. It checks argv, stdin/stdout/stderr, environment, working
directory, preserved PID, help/version forwarding, exit status, SIGTERM delivery,
and missing-install/recursive-executable failures. It does not need Splash or
download model weights. It does not access the GPU.

Sources: [Splash](https://github.com/incoai/splash),
[launcher](https://github.com/incoai/splash/blob/main/splash),
[native entry point](https://github.com/incoai/splash/blob/main/runtime/main.mm),
[development guide](https://github.com/incoai/splash/blob/main/DEVELOPMENT.md).
