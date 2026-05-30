#!/bin/bash
# Compatibility shim.
#
# The build script now lives at the repository root as ./build.sh so that anyone
# who does `git clone` can build immediately without hunting through Scripts/.
# This wrapper forwards to it, preserving every existing caller
# (publish-stable.sh, publish-next.sh, skills, docs, CI) unchanged.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

exec "$ROOT_DIR/build.sh" "$@"
