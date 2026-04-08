#!/usr/bin/env bash
# Generate versioned Homebrew formulae for the scouzi1966/afm tap.
#
# Creates files like:
#   afm@0.9.9.rb            (class AfmAT099)        pinned to v0.9.9
#   afm-next@20260408.rb    (class AfmNextAT20260408) pinned to nightly-20260408-*
#
# Backfills from GitHub releases and keeps only the most recent N per family.
# Fetches sha256 from the release asset digest (no tarball download needed).
#
# Usage:
#   ./Scripts/generate-tap-versioned.sh                 # backfill both families (last 10 each)
#   ./Scripts/generate-tap-versioned.sh --stable-only
#   ./Scripts/generate-tap-versioned.sh --next-only
#   ./Scripts/generate-tap-versioned.sh --add-next <tag> <version> <url> <sha256>
#   ./Scripts/generate-tap-versioned.sh --keep 10
#
# After generation, the tap repo has uncommitted changes. Review with `git diff`
# in $TAP_DIR and commit/push when ready.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TAP_DIR="${TAP_DIR:-$ROOT_DIR/../homebrew-afm}"
REPO="scouzi1966/maclocal-api"
KEEP=10
MODE="all"
ADD_NEXT_ARGS=()

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'
log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

while [ $# -gt 0 ]; do
  case "$1" in
    --stable-only) MODE="stable" ;;
    --next-only) MODE="next" ;;
    --keep) shift; KEEP="$1" ;;
    --add-next)
      MODE="add-next"
      shift; ADD_NEXT_ARGS=("$1" "$2" "$3" "$4")
      shift 3
      ;;
    -h|--help)
      sed -n '1,25p' "$0"; exit 0 ;;
    *) log_error "Unknown option: $1"; exit 1 ;;
  esac
  shift
done

if [ ! -d "$TAP_DIR/.git" ]; then
  log_error "Tap repo not found at $TAP_DIR"
  exit 1
fi

# ---------------------------------------------------------------------------
# Homebrew class-name helpers
# ---------------------------------------------------------------------------

# Compute the Homebrew-canonical class name for a given "name@version" string.
# Replicates the exact transform in Homebrew::Formulary::class_s so the class
# name in the rendered .rb file matches what brew expects when parsing the file.
#
# Example: "afm-next@0.9.10-next.628c2bb.20260408" -> "AfmNextAT0910Next628c2bb20260408"
class_name_for() {
  local base="$1" version="$2"
  # Use double-quoted bash heredoc so we can escape Ruby backrefs as \\1/\\2 from
  # bash's perspective (which reach Ruby as \1/\2 inside a single-quoted Ruby string).
  ruby -e "
    full = ARGV[0] + '@' + ARGV[1]
    class_name = full.capitalize
    class_name.gsub!(/[-_.\\s]([a-zA-Z0-9])/) { \$1.upcase }
    class_name.tr!('+', 'x')
    class_name.sub!(/(.)@(\\d)/, '\\1AT\\2')
    puts class_name
  " -- "$base" "$version"
}

# ---------------------------------------------------------------------------
# Formula body rendering
# ---------------------------------------------------------------------------

render_afm_versioned() {
  # Render afm@X.Y.Z.rb content
  local version="$1" url="$2" sha256="$3"
  local class
  class="$(class_name_for "afm" "$version")"
  cat <<EOF
class ${class} < Formula
  desc "Apple Foundation Models + MLX local models — OpenAI-compatible API, WebUI, all Swift (pinned v${version})"
  homepage "https://github.com/scouzi1966/maclocal-api"
  url "${url}"
  version "${version}"
  sha256 "${sha256}"

  depends_on arch: :arm64
  depends_on :macos

  conflicts_with "afm", because: "both install an \`afm\` binary"
  conflicts_with "afm-next", because: "both install an \`afm\` binary"

  def install
    bin.install "afm"
    if File.directory?("MacLocalAPI_MacLocalAPI.bundle")
      (libexec/"MacLocalAPI_MacLocalAPI.bundle").install Dir["MacLocalAPI_MacLocalAPI.bundle/*"]
    end
    if File.exist?("Resources/webui/index.html.gz")
      (share/"afm/webui").install "Resources/webui/index.html.gz"
    end
  end

  def post_install
    bundle_src = libexec/"MacLocalAPI_MacLocalAPI.bundle"
    bundle_dst = HOMEBREW_PREFIX/"bin/MacLocalAPI_MacLocalAPI.bundle"
    bundle_dst.unlink if bundle_dst.symlink? || bundle_dst.exist?
    bundle_dst.make_symlink(bundle_src) if bundle_src.exist?
  end

  def caveats
    <<~EOS
      This is a pinned historical release of afm (v${version}).
      For the latest stable: brew install scouzi1966/afm/afm
      For the latest nightly: brew install scouzi1966/afm/afm-next
    EOS
  end

  test do
    assert_match "afm", shell_output("#{bin}/afm --version")
  end
end
EOF
}

render_next_versioned() {
  # Render afm-next@<full_version>.rb content
  # full_version is the canonical string from the release body, e.g. "0.9.10-next.628c2bb.20260408"
  # datestr is the YYYYMMDD part only (for the caveats message).
  local datestr="$1" full_version="$2" url="$3" sha256="$4"
  local class
  class="$(class_name_for "afm-next" "$full_version")"
  cat <<EOF
class ${class} < Formula
  desc "AFM next — OpenAI-compatible local LLM API (pinned nightly ${datestr})"
  homepage "https://github.com/scouzi1966/maclocal-api"
  url "${url}"
  version "${full_version}"
  sha256 "${sha256}"

  depends_on arch: :arm64
  depends_on :macos

  conflicts_with "afm", because: "both install an \`afm\` binary"
  conflicts_with "afm-next", because: "both install an \`afm\` binary"

  def install
    bin.install "afm"
    if File.directory?("MacLocalAPI_MacLocalAPI.bundle")
      (libexec/"MacLocalAPI_MacLocalAPI.bundle").install Dir["MacLocalAPI_MacLocalAPI.bundle/*"]
    end
    if File.exist?("Resources/webui/index.html.gz")
      (share/"afm/webui").install "Resources/webui/index.html.gz"
    end
  end

  def post_install
    bundle_src = libexec/"MacLocalAPI_MacLocalAPI.bundle"
    bundle_dst = HOMEBREW_PREFIX/"bin/MacLocalAPI_MacLocalAPI.bundle"
    bundle_dst.unlink if bundle_dst.symlink? || bundle_dst.exist?
    bundle_dst.make_symlink(bundle_src) if bundle_src.exist?
  end

  def caveats
    <<~EOS
      This is a pinned historical nightly (${datestr}).
      For the latest nightly: brew install scouzi1966/afm/afm-next
      For the latest stable:  brew install scouzi1966/afm/afm
    EOS
  end

  test do
    assert_match "afm", shell_output("#{bin}/afm --version")
  end
end
EOF
}

# ---------------------------------------------------------------------------
# Release enumeration
# ---------------------------------------------------------------------------

# Get asset download URL + sha256 for a given release tag and exact asset name.
# Echoes "url sha256" (space-separated) or empty if not found.
get_asset_info() {
  local tag="$1" name="$2"
  gh api "repos/${REPO}/releases/tags/${tag}" \
    --jq ".assets[] | select(.name == \"${name}\") | \"\(.browser_download_url) \(.digest | sub(\"^sha256:\"; \"\"))\"" \
    2>/dev/null | head -1
}

backfill_stable() {
  log_info "Backfilling stable versioned formulae (last ${KEEP})"
  # List last KEEP stable releases (tag starts with v, not pre-release)
  mapfile -t tags < <(
    gh release list --repo "$REPO" --exclude-drafts --exclude-pre-releases --limit 50 2>/dev/null \
      | awk -F'\t' '$3 ~ /^v[0-9]+\.[0-9]+\.[0-9]+$/ { print $3 }' \
      | head -n "$KEEP"
  )
  if [ ${#tags[@]} -eq 0 ]; then
    log_warn "No stable tags found"
    return
  fi
  for tag in "${tags[@]}"; do
    local version="${tag#v}"
    local outfile="$TAP_DIR/afm@${version}.rb"
    local info url sha256
    info="$(get_asset_info "$tag" "afm-v${version}-arm64.tar.gz")"
    if [ -z "$info" ]; then
      log_warn "  skip $tag: no matching asset"
      continue
    fi
    url="${info%% *}"
    sha256="${info##* }"
    render_afm_versioned "$version" "$url" "$sha256" > "$outfile"
    log_info "  wrote afm@${version}.rb"
  done
  prune_formulae "afm" "$KEEP"
}

backfill_next() {
  log_info "Backfilling nightly versioned formulae (last ${KEEP})"
  # List last KEEP nightly releases (newest first)
  mapfile -t tags < <(
    gh release list --repo "$REPO" --limit 60 2>/dev/null \
      | awk -F'\t' '$3 ~ /^nightly-[0-9]{8}-/ { print $3 }' \
      | head -n "$KEEP"
  )
  if [ ${#tags[@]} -eq 0 ]; then
    log_warn "No nightly tags found"
    return
  fi
  for tag in "${tags[@]}"; do
    # tag format: nightly-YYYYMMDD-SHORTSHA
    local datestr="${tag#nightly-}"
    datestr="${datestr%%-*}"
    local info url sha256
    info="$(get_asset_info "$tag" "afm-next-arm64.tar.gz")"
    if [ -z "$info" ]; then
      log_warn "  skip $tag: no matching asset"
      continue
    fi
    url="${info%% *}"
    sha256="${info##* }"
    # Extract the canonical full version string from the release body:
    #   **Version:** 0.9.10-next.628c2bb.20260408
    # This matches exactly what users see on the GitHub release page.
    local full_version
    full_version="$(
      gh release view "$tag" --repo "$REPO" --json body -q .body 2>/dev/null \
        | grep -E '\*\*Version:\*\*' \
        | head -1 \
        | sed -E 's/.*\*\*Version:\*\*[[:space:]]*(.+)$/\1/' \
        | tr -d '\r'
    )"
    if [ -z "$full_version" ]; then
      log_warn "  $tag: no Version in release body, skipping"
      continue
    fi
    local outfile="$TAP_DIR/afm-next@${full_version}.rb"
    render_next_versioned "$datestr" "$full_version" "$url" "$sha256" > "$outfile"
    log_info "  wrote afm-next@${full_version}.rb"
  done
  prune_formulae "afm-next" "$KEEP"
}

# Remove versioned formulae for a family beyond the KEEP most recent.
#
# Stable formulae (afm@X.Y.Z.rb): sort by SemVer — highest first.
# Nightly formulae (afm-next@X.Y.Z-next.SHA.YYYYMMDD.rb): sort by the trailing
# date field (YYYYMMDD) — newest first. The base SemVer at the front can be
# smaller than a later nightly when the base version hasn't bumped yet, so we
# sort by date specifically to keep the 10 most recently built nightlies.
prune_formulae() {
  local family="$1" keep="$2"
  local files_sorted=()
  case "$family" in
    afm)
      # Standard SemVer sort
      while IFS= read -r f; do
        files_sorted+=("$f")
      done < <(cd "$TAP_DIR" && ls -1 afm@*.rb 2>/dev/null | grep -v '^afm-next@' | sort -V -r)
      ;;
    afm-next)
      # Sort by the trailing .YYYYMMDD field in the filename (key: date desc)
      while IFS= read -r f; do
        files_sorted+=("$f")
      done < <(
        cd "$TAP_DIR" && ls -1 'afm-next@'*.rb 2>/dev/null \
          | awk -F'.' '{
              # Last field before .rb is the date; second-to-last when .rb is at the end
              # Filename: afm-next@0.9.10-next.628c2bb.20260408.rb
              # Split on "." → ["afm-next@0","9","10-next","628c2bb","20260408","rb"]
              # Date = $(NF-1)
              printf "%s\t%s\n", $(NF-1), $0
            }' \
          | sort -r \
          | cut -f2-
      )
      ;;
    *) return ;;
  esac

  if [ ${#files_sorted[@]} -le "$keep" ]; then
    return
  fi
  # Delete everything past the keep threshold
  for f in "${files_sorted[@]:$keep}"; do
    log_info "  pruning old formula: $f"
    rm -f "$TAP_DIR/$f"
  done
}

add_single_next() {
  # Usage: --add-next <tag> <full_version> <url> <sha256>
  # full_version should be the canonical string shown on the release page,
  # e.g. "0.9.10-next.628c2bb.20260408". The filename will be afm-next@<full_version>.rb.
  local tag="$1" full_version="$2" url="$3" sha256="$4"
  local datestr="${tag#nightly-}"
  datestr="${datestr%%-*}"
  local outfile="$TAP_DIR/afm-next@${full_version}.rb"
  render_next_versioned "$datestr" "$full_version" "$url" "$sha256" > "$outfile"
  log_info "Wrote afm-next@${full_version}.rb"
  prune_formulae "afm-next" "$KEEP"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

case "$MODE" in
  all)
    backfill_stable
    backfill_next
    ;;
  stable)
    backfill_stable
    ;;
  next)
    backfill_next
    ;;
  add-next)
    add_single_next "${ADD_NEXT_ARGS[@]}"
    ;;
esac

echo ""
log_info "Done. Review changes: cd $TAP_DIR && git status && git diff"
