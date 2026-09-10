#!/usr/bin/env bash
# Pack / unpack / verify the 0_raw snapshot for transfer between clusters.
#
#   tools/snapshot.sh pack [--lean] [outfile]   create a tar.zst archive
#   tools/snapshot.sh unpack <file> [dest]      restore a snapshot
#   tools/snapshot.sh verify                    md5-check 0_raw against the manifest
#
# --lean omits the *_curve-*.png diagnostic plots (~271 MB, 8850 files). Nothing in
# the analysis reads them and they can be regenerated from pred-* plus the test sets,
# so a lean snapshot still reproduces every figure. They stay listed in the manifest
# either way, so provenance is unaffected.
set -euo pipefail

PP="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW="${PREPIBIND_RAW_ROOT:-$PP/0_raw}"
MANIFEST="$PP/raw_manifest.csv"
ZSTD_OPTS=${ZSTD_OPTS:--12 -T0}

usage() { sed -n '2,12p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 1; }

cmd=${1:-}; shift || usage
case "$cmd" in
pack)
  lean=0
  if [[ ${1:-} == --lean ]]; then lean=1; shift; fi
  out=${1:-$PP/0_raw$([[ $lean == 1 ]] && echo .lean).tar.zst}
  [[ -f $MANIFEST ]] || { echo "manifest missing; run tools/build_manifest.py first" >&2; exit 1; }
  excl=()
  [[ $lean == 1 ]] && excl=(--exclude='*_curve-*.png')
  echo "packing -> $out  (lean=$lean, zstd $ZSTD_OPTS)"
  tar -I "zstd $ZSTD_OPTS" -cf "$out" \
      -C "$PP" "${excl[@]}" 0_raw raw_manifest.csv
  ls -la "$out"
  ;;
unpack)
  f=${1:?archive path required}; dest=${2:-$PP}
  echo "unpacking $f -> $dest"
  tar -I zstd -xf "$f" -C "$dest"
  echo "done. now run: tools/snapshot.sh verify"
  ;;
verify)
  python3 - "$RAW" "$MANIFEST" <<'PYEOF'
import csv, hashlib, os, sys
raw, man = sys.argv[1], sys.argv[2]
rows = [r for r in csv.DictReader(open(man)) if r['state'] == 'snapshot' and r['rel']]
missing = bad = ok = 0
for r in rows:
    p = os.path.join(raw, r['rel'])
    if not os.path.exists(p):
        missing += 1
        continue
    h = hashlib.md5()
    with open(p, 'rb') as fh:
        while (b := fh.read(1 << 20)):
            h.update(b)
    if h.hexdigest() == r['md5']:
        ok += 1
    else:
        bad += 1
        if bad <= 5:
            print(f'  MD5 MISMATCH {r["rel"]}')
print(f'manifest {len(rows)} rows: {ok} ok, {missing} missing, {bad} mismatched')
if missing:
    print('  (a lean snapshot legitimately omits *_curve-*.png)')
sys.exit(1 if bad else 0)
PYEOF
  ;;
*) usage ;;
esac
