---
name: afm-release-wheel
description: Use when user wants to build a PyPI wheel from an existing compiled afm binary and publish to PyPI. Covers staging assets, building the wheel, and providing the uv publish command. Only for official stable releases, not nightly builds.
user_invocable: true
---

# AFM Release Wheel

Build a PyPI-distributable wheel from an existing compiled `afm` binary and provide the publish command.

**Only official stable releases can be published to PyPI. Nightly builds are NOT distributed via pip.**

## Usage

- `/afm-release-wheel` — build wheel from existing release binary
- `/afm-release-wheel 0.9.7` — build wheel and set version

## Instructions

### Step 1: Ask Target Registry

Use AskUserQuestion:

**Question:** "Which PyPI registry should this wheel target?"

**Options:**
1. "TestPyPI (Recommended for first-time validation)" — publish to https://test.pypi.org
2. "PyPI Production" — publish to https://pypi.org (official release)

Save the choice for Step 5.

### Step 2: Determine Version

If version was provided as argument, use it. Otherwise read current version:

```bash
grep '^version = ' pyproject.toml
grep '^__version__' macafm/__init__.py
```

Use AskUserQuestion to confirm: "Current version is X.Y.Z. Use this version or specify a new one?"

**Both files must agree.** If they differ, update both before proceeding:
```bash
sed -i '' 's/^version = ".*"/version = "NEW_VERSION"/' pyproject.toml
sed -i '' 's/^__version__ = ".*"/__version__ = "NEW_VERSION"/' macafm/__init__.py
```

### Step 3: Locate and Validate Binary

```bash
BIN=".build/arm64-apple-macosx/release/afm"
[ -x "$BIN" ] || BIN=".build/release/afm"

if [ ! -x "$BIN" ]; then
  echo "ERROR: No compiled binary found. Run /build-afm first."
  exit 1
fi

echo "Binary: $(cd "$(dirname "$BIN")" && pwd)/$(basename "$BIN")"
$BIN --version
ls -lh "$BIN"
```

Also locate required assets:
```bash
# Metallib
METALLIB="$(dirname "$BIN")/MacLocalAPI_MacLocalAPI.bundle/default.metallib"
test -f "$METALLIB" && echo "Metallib: OK ($(ls -lh "$METALLIB" | awk '{print $5}'))" || echo "WARNING: No metallib found"

# WebUI
test -f Resources/webui/index.html.gz && echo "WebUI: OK" || echo "WARNING: No webui found"
```

If binary is missing, **STOP** and tell user to run `/build-afm` first.

### Step 4: Stage Assets and Build Wheel

Stage compiled artifacts into `macafm/` for wheel bundling, build, then clean:

```bash
# Stage binary + metallib
mkdir -p macafm/bin
cp "$BIN" macafm/bin/
if [ -f "$METALLIB" ]; then
  cp "$METALLIB" macafm/bin/
fi

# Stage webui
if [ -f Resources/webui/index.html.gz ]; then
  mkdir -p macafm/share/webui
  cp Resources/webui/index.html.gz macafm/share/webui/
fi

# Build wheel + sdist
uv build

# Clean staged assets (never commit binaries to git)
rm -rf macafm/bin macafm/share
```

Verify the wheel was created and has real content (not an empty 9KB wheel):
```bash
ls -lh dist/macafm-*.whl dist/macafm-*.tar.gz
# Wheel should be ~15-25 MB (contains binary + metallib)
```

If wheel is under 1 MB, **STOP** — assets were not staged correctly.

### Step 5: Provide Publish Command

Based on the registry choice from Step 1, provide the exact command with the wheel path.

**For TestPyPI:**
```
uv publish --publish-url https://test.pypi.org/legacy/ --token <YOUR_TEST_PYPI_TOKEN> dist/macafm-VERSION*
```

After publishing, tell user to verify:
```
pip install --index-url https://test.pypi.org/simple/ macafm==VERSION
```

**For PyPI Production:**
```
uv publish --token <YOUR_PYPI_TOKEN> dist/macafm-VERSION*
```

After publishing, tell user to verify:
```
pip install macafm==VERSION
```

### Step 6: Report

Report to the user:
- Version published
- Registry (TestPyPI or Production)
- Wheel file path and size
- The exact publish command to run (with actual paths, only token placeholder)
- The verification install command

Remind the user:
- **TestPyPI**: Good for validation. Install with `--index-url https://test.pypi.org/simple/`
- **Production**: This is the real release. `pip install macafm` will get this version.
- **After publishing to production**, commit the version bump if not already done:
  ```
  git add pyproject.toml macafm/__init__.py
  git commit -m "Bump version to VERSION"
  ```

### Error Handling

- **No binary found**: Direct user to `/build-afm`
- **`uv build` fails**: Check that `uv` is installed (`pip install uv`), check pyproject.toml syntax
- **Wheel too small (<1 MB)**: Assets not staged — re-run Step 4
- **Version conflict**: Both `pyproject.toml` and `macafm/__init__.py` must match
