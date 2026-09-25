#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

OUT="${1:-$ROOT/iclr_anonymized_code.zip}"
ARCHIVE_ROOT="iclr_anonymized_code"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "== Building archive =="
rm -f "$OUT"
git archive --format=zip --prefix="${ARCHIVE_ROOT}/" -o "$OUT" HEAD
echo "Wrote: $OUT"

echo
echo "== Verifying archive contents =="
unzip -q "$OUT" -d "$TMP"

if [[ -e "$TMP/$ARCHIVE_ROOT/.git" ]]; then
  echo "FAIL: archive contains a .git directory."
  exit 1
fi
echo "OK: archive has no .git directory."

(
  cd "$TMP/$ARCHIVE_ROOT"
  scripts/verify_anonymization.sh
)

echo
echo "Archive is ready for anonymous submission: $OUT"
