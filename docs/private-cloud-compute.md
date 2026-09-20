# Private Cloud Compute from the AFM CLI

`afm pcc` uses the native macOS 27 Private Cloud Compute (PCC) provider already
shipped by the pinned AFMKit package. Requests leave the Mac for Apple's PCC;
there is no automatic fallback to on-device generation and no API key to configure.
The existing `afm` and `afm mlx` commands retain their existing behavior.

## Commands

Run the executable **inside a correctly signed app bundle**:

```bash
AFM_PCC="/path/to/AFM.app/Contents/MacOS/afm"
"$AFM_PCC" pcc status
"$AFM_PCC" pcc status --json
"$AFM_PCC" pcc respond "Explain Swift actors" --reasoning moderate
printf '%s\n' 'Explain Swift actors' | "$AFM_PCC" pcc respond -
"$AFM_PCC" pcc chat --reasoning deep
"$AFM_PCC" pcc serve --hostname 127.0.0.1 --port 9999
```

`afm pcc` defaults to `status`. Status performs no generation and exits nonzero
when unavailable. Its JSON includes `available`, `hasEntitlement`, `reason`,
`detail`, and quota diagnostics. Entitlement presence in a signature does not
prove provisioning authorization or guarantee a successful subsequent request.

`respond`, `chat`, and `serve` accept `--instructions` and `--reasoning`
(`automatic`, `light`, `moderate`, `deep`). They stream by default; use
`--no-streaming` to wait for the complete answer. `chat` uses AFM's native TUI
and requires an interactive terminal. `respond` writes answer text to stdout.

On macOS 26, all PCC execution commands exit with an explanatory error before
creating a PCC model: **Private Cloud Compute requires macOS 27 or later.**
AFM itself continues to require macOS 26 or later; this does not introduce
support for running the binary on older systems.

## Development signing

Device access and an entitlement approved for another app do not authorize
AFM's identity. AFM's embedded Info.plist uses `com.scouzi1966.afm`.

1. Under your approved Apple Developer team, enable the managed Private Cloud
   Compute capability on the explicit AFM App ID.
2. Generate/download a macOS development provisioning profile for that App ID.
   It must contain `com.apple.developer.private-cloud-compute = true`, include
   this Mac, and authorize your Apple Development signing certificate.
3. Ensure the matching Apple Development certificate/private key is available
   in your keychain. Xcode's Accounts settings can manage development identities.
4. Run the build-and-sign entry point from this PR's checkout:

```bash
python3 Scripts/build-pcc-app.py
```

The script prompts without echo for the local provisioning-profile path and the
Apple Development identity name or SHA-1. Inspect identities locally with
`security find-identity -v -p codesigning`; do not post that output. The private
key stays in your keychain. No Apple account password or API key is needed.

It checks the platform, invokes the consumer-boundary check and
`Scripts/swiftpm-reliable.sh build -c release --product afm`, using a fresh
isolated scratch directory under `.build-pcc/`. It then runs the existing
packager's `--check-only` preflight, packages/signs the app, and verifies the
signature. It does not clean another build directory, run status, or generate
any response. Build products are retained for inspection.

The default output is a new timestamped app under
`~/Library/Developer/AFM-PCC/`. Use `--output /path/to/new/AFM.app` to choose a
location. To sign an existing build without starting any compilation:

```bash
python3 Scripts/build-pcc-app.py --binary /absolute/path/to/afm
```

`--profile /path/to/development.provisionprofile` and
`--identity "LOCAL-SIGNING-IDENTITY"` are also accepted for local automation;
prefer the prompts to keep real values out of command text and shell history.
The script never prints the complete credential-bearing command on failure.
Keep shell tracing disabled. Generated apps, embedded profiles, entitlements,
and raw signing diagnostics can contain developer identifiers and should stay
private; share only redacted error summaries.

For preflight only on an existing binary, use the lower-level
`Scripts/package-pcc-app.py --binary ... --profile ... --identity ...
--output ... --check-only`. The build script and this command share the same
provisioning/signature validation implementation.

The packager rejects expired profiles, missing PCC, a different App ID (including
Vesta), wildcard App IDs, unregistered devices, and unmatched certificates. It
does not modify keychains, create certificates, accept licenses, or overwrite
existing apps. It preserves the profile's App ID prefix and Team ID, and its
App Attest opt-in claim when present. It does not copy unrelated capabilities.

The bundle contains `Contents/Info.plist`, `Contents/embedded.provisionprofile`,
`Contents/MacOS/afm`, SwiftPM bundles under `Contents/Resources`, and required
adjacent runtime libraries/MLX metallib. Nested Mach-O files are signed before
the app. The exact entitlement claims are saved in
`Contents/Resources/AFM.entitlements`; signature and entitlement verification
run after signing. The bundle is for local development, not notarized distribution.

```bash
PCC_APP="/absolute/path/to/the/generated/AFM.app"
codesign --verify --deep --strict "$PCC_APP"
"$PCC_APP/Contents/MacOS/afm" pcc status --json
# Optional live inference, only when you explicitly intend to send a PCC request:
"$PCC_APP/Contents/MacOS/afm" pcc respond "Reply with OK" --no-streaming
```

Keep the binary inside the bundle. To expose it on PATH, use a shell launcher
that `exec`s the absolute bundled executable path with `"$@"`. Moving the binary
alone loses the profile-bearing app context. Rebuilds require repackaging/signing.

## OpenAI HTTP interface and limits

```bash
curl http://127.0.0.1:9999/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"apple.private-cloud-compute","messages":[{"role":"user","content":"Explain Swift actors"}],"stream":true}'
```

PCC uses the existing AFMKit provider-backed chat route. Reasoning is configured
with `pcc serve --reasoning`; the model ID is `apple.private-cloud-compute`.
This initial CLI integration supports text chat and the provider's supported
structured output. It does not register native tools. Arbitrary client tool
definitions, MLX-specific sampling flags, and media support are not promised.
The provider rejects unsupported options instead of silently changing models.

PCC availability and quota can change after startup. Provider unavailability
returns HTTP 503; provider generation failures return 502; unsupported requests
return 400. After SSE headers are sent, failures use the existing streaming error
channel. The pinned AFMKit provider converts native generation errors to generic
AFM errors, so this integration does **not** infer HTTP 429 from localized quota
error text. A future typed provider error contract can make that distinction.

Public Developer ID/Homebrew/pip distribution is a separate qualification from
local development. Apple's current eligibility page specifies App Store
distribution and TestFlight/ad hoc testing. An ordinary AFM download is not
automatically PCC-authorized by adding an entitlement or notarizing it.

## Validation

```bash
python3 Scripts/test-pcc-build.py
python3 Scripts/test-pcc-packaging.py
Scripts/swiftpm-reliable.sh test -c release --filter 'PCCConfigurationTests|PCCHTTPTests'
python3 Scripts/test-pcc-cli.py --binary .build/release/afm
```

The CLI regression script expects an ordinary, non-PCC-entitled binary. Run it
on macOS 26 to exercise real unsupported-OS errors, and macOS 27 to exercise
missing-entitlement errors. XCTest covers runtime policy, status mapping,
provider configuration, and HTTP requests with a fixture provider. These checks
do not prove successful live PCC inference; use the signed smoke test above.

References: [PCC access](https://developer.apple.com/private-cloud-compute/),
[PCC API](https://developer.apple.com/documentation/foundationmodels/privatecloudcomputelanguagemodel),
[profile-authorized command-line executables](https://developer.apple.com/documentation/xcode/signing-a-daemon-with-a-restricted-entitlement).
