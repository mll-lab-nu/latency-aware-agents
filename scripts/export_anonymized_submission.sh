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
python3 - "$OUT" <<'PY'
import os
import sys
import tempfile
import zipfile

archive_path = sys.argv[1]
archive_dir = os.path.dirname(os.path.abspath(archive_path))
fd, scrubbed_path = tempfile.mkstemp(prefix=".anonymous-", suffix=".zip", dir=archive_dir)
os.close(fd)

try:
    with zipfile.ZipFile(archive_path, "r") as src, zipfile.ZipFile(scrubbed_path, "w") as dst:
        dst.comment = b""
        for info in src.infolist():
            data = src.read(info.filename)
            clean = zipfile.ZipInfo(info.filename, date_time=(1980, 1, 1, 0, 0, 0))
            clean.compress_type = info.compress_type
            clean.external_attr = info.external_attr
            clean.create_system = info.create_system
            clean.comment = b""
            clean.extra = b""
            dst.writestr(clean, data)
    os.replace(scrubbed_path, archive_path)
finally:
    if os.path.exists(scrubbed_path):
        os.unlink(scrubbed_path)
PY
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
