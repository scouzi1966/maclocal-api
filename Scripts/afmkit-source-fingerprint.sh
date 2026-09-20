#!/usr/bin/env bash
# Print a stable fingerprint for a writable AFMKit checkout and its DS4
# submodule. This intentionally hashes Git state instead of the complete
# vendored source tree so local incremental builds remain fast.

set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <afmkit-source-root> <source-id>" >&2
    exit 2
fi

AFMKIT_SOURCE_ROOT="$(cd "$1" && pwd)"
AFMKIT_SOURCE_ID="$2"

git -C "$AFMKIT_SOURCE_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1 || {
    echo "AFMKit source root is not a Git worktree: $AFMKIT_SOURCE_ROOT" >&2
    exit 2
}

emit_git_state() {
    local repository_root="$1"
    shift

    git -C "$repository_root" rev-parse HEAD
    git -C "$repository_root" diff \
        --no-ext-diff --binary --submodule=diff HEAD -- "$@"
    while IFS= read -r -d '' relative_path; do
        local digest
        digest="$(shasum -a 256 "$repository_root/$relative_path" | awk '{print $1}')"
        printf '%s\0%s\n' "$relative_path" "$digest"
    done < <(
        git -C "$repository_root" ls-files -z \
            --others --exclude-standard -- "$@"
    )
}

{
    printf '%s\n' "$AFMKIT_SOURCE_ID"
    emit_git_state "$AFMKIT_SOURCE_ROOT" \
        Package.swift Sources Packages vendor/ds4 vendor/MLX

    DS4_ROOT="$AFMKIT_SOURCE_ROOT/vendor/ds4"
    if git -C "$DS4_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        # The parent repository records only the DS4 gitlink. Include the
        # submodule's own tracked diff and untracked, non-ignored files so a
        # local provider edit cannot reuse stale compiled products.
        printf '%s\n' "vendor/ds4-working-tree"
        emit_git_state "$DS4_ROOT" .
    fi
} | shasum -a 256 | awk '{print $1}'
