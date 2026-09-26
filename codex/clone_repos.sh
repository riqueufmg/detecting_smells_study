#!/usr/bin/env bash
#
# Clone every repository listed in repo_head_hashes.txt and check it out at
# its pinned commit. Idempotent and resumable: a repo already sitting at the
# pinned commit is skipped; a repo present at the wrong commit is fetched and
# re-checked-out (it is NOT re-cloned).
#
# Input file format, one entry per line (blank lines / `#` comments ignored):
#
#     owner/repo:commit-sha
#
# Each repo is cloned to  <DEST_ROOT>/<owner>/<repo>  so the on-disk layout is
# unambiguous and collision-free. Those paths are what tasks.jsonl's
# `repo_host_path` should point at.
#
# Usage:
#     ./clone_repos.sh [DEST_ROOT] [LIST_FILE]
#
# Defaults: DEST_ROOT=./repos   LIST_FILE=./repo_head_hashes.txt
# (both resolved relative to this script's directory).

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST_ROOT="${1:-$HERE/repos}"
LIST_FILE="${2:-$HERE/repo_head_hashes.txt}"

if [[ ! -f "$LIST_FILE" ]]; then
  echo "ERROR: list file not found: $LIST_FILE" >&2
  exit 1
fi

mkdir -p "$DEST_ROOT"

n_ok=0 n_skip=0 n_fail=0
failures=()

while IFS= read -r raw || [[ -n "$raw" ]]; do
  # strip whitespace; skip blank lines and comments
  line="${raw#"${raw%%[![:space:]]*}"}"
  line="${line%"${line##*[![:space:]]}"}"
  [[ -z "$line" || "$line" == \#* ]] && continue

  if [[ "$line" != */*:* ]]; then
    echo "ERROR: malformed entry (expected owner/repo:sha): '$line'" >&2
    failures+=("$line"); n_fail=$((n_fail + 1)); continue
  fi

  slug="${line%%:*}"        # owner/repo
  sha="${line##*:}"         # commit-sha
  owner="${slug%%/*}"
  repo="${slug##*/}"
  dest="$DEST_ROOT/$owner/$repo"
  url="https://github.com/$slug.git"

  echo "==> $slug @ ${sha:0:12}"

  # Already at the pinned commit?
  if [[ -d "$dest/.git" ]]; then
    have="$(git -C "$dest" rev-parse HEAD 2>/dev/null || echo '')"
    if [[ "$have" == "$sha" ]]; then
      echo "    already at pinned commit — skip"
      n_skip=$((n_skip + 1)); continue
    fi
    echo "    present at ${have:0:12}, repinning to ${sha:0:12}"
  else
    echo "    cloning $url"
    if ! git clone --quiet "$url" "$dest"; then
      echo "    ERROR: clone failed" >&2
      failures+=("$slug (clone)"); n_fail=$((n_fail + 1)); continue
    fi
  fi

  # Make sure the pinned commit is local (fetch it if the clone didn't have it),
  # then detach exactly onto it.
  if ! git -C "$dest" cat-file -e "${sha}^{commit}" 2>/dev/null; then
    git -C "$dest" fetch --quiet origin "$sha" 2>/dev/null \
      || git -C "$dest" fetch --quiet --all 2>/dev/null || true
  fi

  if ! git -C "$dest" checkout --quiet --detach "$sha" 2>/dev/null; then
    echo "    ERROR: commit $sha not found after fetch" >&2
    failures+=("$slug (checkout $sha)"); n_fail=$((n_fail + 1)); continue
  fi

  got="$(git -C "$dest" rev-parse HEAD)"
  if [[ "$got" != "$sha" ]]; then
    echo "    ERROR: HEAD is $got, expected $sha" >&2
    failures+=("$slug (HEAD mismatch)"); n_fail=$((n_fail + 1)); continue
  fi
  echo "    OK -> $dest"
  n_ok=$((n_ok + 1))
done < "$LIST_FILE"

echo
echo "done: $n_ok checked out, $n_skip skipped, $n_fail failed"
if ((n_fail > 0)); then
  printf '  - %s\n' "${failures[@]}" >&2
  exit 1
fi
