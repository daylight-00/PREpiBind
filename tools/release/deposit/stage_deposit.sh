#!/usr/bin/env bash
# Assemble the PREpiBind archival deposit into one directory.
#
#   bash tools/release/deposit/stage_deposit.sh --out DIR   assemble and verify
#   bash tools/release/deposit/stage_deposit.sh --check     verify the sources, write nothing
#
# It copies two files, packs a third, writes README.md and SHA256SUMS, and re-verifies everything
# it wrote. It does not upload, does not tag, does not mint a DOI, does not delete anything, and
# writes nothing outside DIR.
#
# Sources default to where they sit on abc and are overridable:
#   --snapshot PATH   --iedb PATH   --draft PATH
#
# Every source is checked against the sha256 recorded on 2026-09-10 before it is used. A mismatch
# is a hard stop: it means the artifact moved on and the inventory in README.md is stale.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"

SNAPSHOT=${PREPIBIND_DEPOSIT_SNAPSHOT:-/home/hwjang/project2/IMG/paperwork/plot_package/0_raw_260910.tar.zst}
IEDB=${PREPIBIND_DEPOSIT_IEDB:-/home/hwjang/project2/260901/0_snapshots/snapshotA_250422_mhc_ligand_full_single_file.zip}
DRAFT=${PREPIBIND_DEPOSIT_DRAFT:-$REPO/pipeline/preprocess/input/draft.csv}

# Verified on abc, 2026-09-10. See README.md in this directory for how each was checked.
SNAPSHOT_SHA=a7aa11473824cee741fc1116569c5ab26fbe0c8ef00d3d67d9b5b56759149142
IEDB_SHA=7ce2af4d57a60c6a5f5fc474b5636fc6471479dd3f204170c3ef26760d6370dc
DRAFT_SHA=ce3b894c8c620920733514f6dd367e6a7af67573d70854c06918b827c316dde3

# Names the deposit publishes under. The IEDB file is renamed back to what IEDB served.
OUT_SNAPSHOT=0_raw_260910.tar.zst
OUT_IEDB=mhc_ligand_full_single_file.zip
OUT_DRAFT=draft.csv.tar.zst

OUT=""
CHECK_ONLY=0
FORCE=0

die() { echo "stage_deposit: $*" >&2; exit 1; }

usage() { sed -n '2,15p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out)      OUT=${2:?--out needs a directory}; shift 2 ;;
    --check)    CHECK_ONLY=1; shift ;;
    --force)    FORCE=1; shift ;;
    --snapshot) SNAPSHOT=${2:?}; shift 2 ;;
    --iedb)     IEDB=${2:?}; shift 2 ;;
    --draft)    DRAFT=${2:?}; shift 2 ;;
    -h|--help)  usage ;;
    *)          die "unknown argument: $1  (try --help)" ;;
  esac
done

for t in zstd tar sha256sum unzip; do
  command -v "$t" >/dev/null || die "$t is not on PATH"
done

expect_sha() {   # path expected_sha label
  local path=$1 want=$2 label=$3 got
  [[ -f $path ]] || die "$label not found: $path"
  echo "  hashing $label ..."
  got=$(sha256sum "$path" | cut -d' ' -f1)
  [[ $got == "$want" ]] || die "$label sha256 changed
    path     $path
    expected $want
    got      $got
  The artifact moved on since 2026-09-10. Re-verify it and update this script and README.md
  before depositing anything."
  echo "  ok      $label"
}

echo "sources"
expect_sha "$SNAPSHOT" "$SNAPSHOT_SHA" "prediction snapshot"
expect_sha "$IEDB"     "$IEDB_SHA"     "IEDB export zip"
expect_sha "$DRAFT"    "$DRAFT_SHA"    "draft.csv"

echo "integrity"
zstd -t "$SNAPSHOT" 2>&1 | sed 's/^/  /'
unzip -tqq "$IEDB" && echo "  ok      IEDB zip decompresses"

if [[ $CHECK_ONLY == 1 ]]; then
  echo
  echo "check only: all three sources are present and unchanged. Nothing was written."
  exit 0
fi

[[ -n $OUT ]] || die "--out is required (or use --check)"
if [[ -e $OUT && -n $(ls -A "$OUT" 2>/dev/null) && $FORCE == 0 ]]; then
  die "$OUT exists and is not empty. Use a fresh directory, or --force."
fi
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)

echo
echo "assembling -> $OUT"
cp -f "$SNAPSHOT" "$OUT/$OUT_SNAPSHOT"
echo "  copied  $OUT_SNAPSHOT"
cp -f "$IEDB" "$OUT/$OUT_IEDB"
echo "  copied  $OUT_IEDB"

echo "  packing $OUT_DRAFT  (zstd -19, a few minutes)"
tar -C "$(dirname "$DRAFT")" -cf - "$(basename "$DRAFT")" | zstd -19 -T0 -q -f -o "$OUT/$OUT_DRAFT"

cp -f "$HERE/DEPOSIT_README.md" "$OUT/README.md"
echo "  wrote   README.md"

( cd "$OUT" && sha256sum "$OUT_SNAPSHOT" "$OUT_IEDB" "$OUT_DRAFT" > SHA256SUMS )
echo "  wrote   SHA256SUMS"

echo
echo "verifying what was written"
( cd "$OUT" && sha256sum -c SHA256SUMS ) | sed 's/^/  /'
zstd -t "$OUT/$OUT_SNAPSHOT" 2>&1 | sed 's/^/  /'
zstd -t "$OUT/$OUT_DRAFT" 2>&1 | sed 's/^/  /'
unzip -tqq "$OUT/$OUT_IEDB" && echo "  ok      $OUT_IEDB"

echo
echo "deposit staged. Nothing has been uploaded."
( cd "$OUT" && ls -l && echo && du -sh . )
cat <<'NEXT'

Next, and none of it is done here:
  * choose a host, create the record, and paste tools/release/deposit/description.md into it
  * upload the four files, then reserve or publish the DOI
  * fill <DATA-DOI> in tools/release/deposit/data-availability.draft.md and apply it
  * only after the DOI resolves, delete the superseded
    IMG/paperwork/plot_package/0_raw.tar.zst
NEXT
