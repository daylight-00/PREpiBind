#!/usr/bin/env python3
"""
Generator for raw_manifest.csv.

Because 0_raw mirrors the scratch path verbatim (see tools/sync_snapshot.py), a
file's provenance *is* its path: `src_path == SCRATCH / rel`. So this script no
longer has to back-trace anything. It records integrity and usage:

    rel          path within 0_raw, identical to the scratch-relative path
    state        snapshot | external | lost
    md5, size    integrity, so a transferred snapshot can be verified
    src_cluster  cluster that produced the data
    src_path     absolute scratch path
    src_mtime    when the original was written
    used_by      which analysis referenced this file, if any
    note         why an external/lost entry is not in the snapshot

    python tools/build_manifest.py           # regenerate
    python tools/build_manifest.py --check   # compare only, write nothing
"""
from __future__ import annotations

import argparse
import collections
import csv
import datetime
import glob
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # analysis/
sys.path.insert(0, HERE)
import rawpath as rp  # noqa: E402

# Same resolution rawpath uses, so PREPIBIND_RAW_ROOT points this at an unpacked snapshot that
# lives outside the checkout instead of silently reporting every entry as removed.
RAW = rp.RAW_ROOT
MANIFEST = os.path.join(HERE, 'raw_manifest.csv')

SCRATCH = os.environ.get('IMG_SCRATCH', '')   # the scratch tree the snapshot mirrors
# Cluster that produced the training/inference data. The other clusters are
# snapshot consumers only, so they are not recorded.
SRC_CLUSTER = os.environ.get('IMG_SRC_CLUSTER') or 'abc'

# Dependencies deliberately kept out of the snapshot, and ones already gone.
EXTERNAL = [
    ('EMB/emb_hla_chai_jack_single_0430.h5', 'figures/figS2_umap.py',
     'HLA embedding, 58MB - reproduced from umap_cache/*.npz'),
    ('EMB/emb_hla_chai_jack_pair_side_0430.h5', 'figures/figS2_umap.py',
     'HLA embedding, 35MB - reproduced from umap_cache/*.npz'),
    ('EMB/esm_large/emb_hla_esm3_small_2408_0430.h5', 'figures/figS2_umap.py',
     'HLA embedding, 229MB - reproduced from umap_cache/*.npz'),
    ('250601/2_full_hla_esm/emb_hla_esmc_small_0601.h5', 'figures/figS2_umap.py',
     'HLA embedding, 1017MB - reproduced from umap_cache/*.npz'),
]
# Recovered after they were deleted from scratch, and small enough to ship: they are inside
# figures/data/figure_inputs.tar.zst, which is why rp.at() resolves them with no snapshot.
BUNDLED = [
    ('250511/2_dataset/1_full_bal/hum_ani_full.csv', 'figures/figS1_dataset_overlap.ipynb', 'in figures/data/figure_inputs.tar.zst'),
    ('250511/2_dataset/2_ic50/hum_ani_full.csv', 'figures/figS1_dataset_overlap.ipynb', 'in figures/data/figure_inputs.tar.zst'),
    ('250520/1_dataset/3_ms_ql/hum_ani_full.csv', 'figures/figS1_dataset_overlap.ipynb', 'in figures/data/figure_inputs.tar.zst'),
]
LOST: list = []

FIELDS = ['rel', 'state', 'md5', 'size', 'src_cluster', 'src_path',
          'src_mtime', 'used_by', 'note']


def md5(path: str, buf: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, 'rb') as fh:
        while (b := fh.read(buf)):
            h.update(b)
    return h.hexdigest()


def mtime(path: str) -> str:
    try:
        return datetime.datetime.fromtimestamp(os.path.getmtime(path)).strftime('%Y-%m-%d')
    except OSError:
        return ''


def read_ledger() -> dict[str, str]:
    """scoring/*/per_run_results.csv records which analysis consumed which file.
    Older CSVs carry an absolute `file_path`; ones written by pipeline.py carry
    `rel` (already snapshot-relative) and `src_path`. Keyed by snapshot rel."""
    led: dict[str, str] = {}
    for p in sorted(glob.glob(os.path.join(HERE, 'scoring/*/per_run_results.csv'))):
        try:
            rows = list(csv.DictReader(open(p)))
        except OSError:
            continue
        if not rows:
            continue
        ref = os.path.relpath(p, HERE)
        for r in rows:
            if r.get('rel'):
                led.setdefault(r['rel'], ref)
            else:
                fp = r.get('file_path') or r.get('src_path') or ''
                if fp.startswith(SCRATCH + '/'):
                    led.setdefault(os.path.relpath(fp, SCRATCH), ref)
    return led


def build() -> list[dict]:
    ledger = read_ledger()
    out: list[dict] = []

    for dirpath, _, files in os.walk(RAW):
        for f in files:
            p = os.path.join(dirpath, f)
            rel = os.path.relpath(p, RAW)
            src = os.path.join(SCRATCH, rel)
            out.append({'rel': rel, 'state': 'snapshot', 'md5': md5(p),
                        'size': os.path.getsize(p), 'src_cluster': SRC_CLUSTER,
                        'src_path': src, 'src_mtime': mtime(src),
                        'used_by': ledger.get(rel, ''), 'note': ''})

    for sub, used, note in EXTERNAL:
        p = os.path.join(SCRATCH, sub)
        out.append({'rel': '', 'state': 'external', 'md5': '',
                    'size': os.path.getsize(p) if os.path.exists(p) else '',
                    'src_cluster': SRC_CLUSTER, 'src_path': p,
                    'src_mtime': mtime(p), 'used_by': used, 'note': note})
    for sub, used, note in BUNDLED:
        p = rp.at(sub, missing_ok=True)
        out.append({'rel': sub, 'state': 'bundled',
                    'md5': md5(p) if os.path.exists(p) else '',
                    'size': os.path.getsize(p) if os.path.exists(p) else '',
                    'src_cluster': SRC_CLUSTER, 'src_path': os.path.join(SCRATCH, sub),
                    'src_mtime': mtime(os.path.join(SCRATCH, sub)), 'used_by': used, 'note': note})
    for sub, used, note in LOST:
        out.append({'rel': '', 'state': 'lost', 'md5': '', 'size': '',
                    'src_cluster': SRC_CLUSTER, 'src_path': os.path.join(SCRATCH, sub),
                    'src_mtime': '', 'used_by': used, 'note': note})

    snap = [r for r in out if r['state'] == 'snapshot']
    used = sum(1 for r in snap if r['used_by'])
    orphan = sorted(set(ledger) - {r['rel'] for r in snap})
    print(f'\n  snapshot files      {len(snap)}', file=sys.stderr)
    print(f'  referenced by an analysis {used}', file=sys.stderr)
    print(f'  ledger paths missing from the snapshot: {len(orphan)}', file=sys.stderr)
    if orphan:
        why = collections.Counter('/'.join(o.split('/')[:3]) for o in orphan)
        for k, v in why.most_common(8):
            print(f'    {v:5d}  {k}', file=sys.stderr)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true',
                    help='compare against the existing manifest without writing')
    a = ap.parse_args()

    rows = sorted(build(), key=lambda r: (r['state'], r['rel'], r['src_path']))
    if a.check:
        old = {r['rel']: r for r in csv.DictReader(open(MANIFEST))} \
            if os.path.exists(MANIFEST) else {}
        new = {r['rel']: r for r in rows}
        add, rm = set(new) - set(old), set(old) - set(new)
        ch = {k for k in set(old) & set(new) if old[k].get('md5') != new[k]['md5']}
        print(f'\n--check: {len(add)} added / {len(rm)} removed / {len(ch)} changed',
              file=sys.stderr)
        return 1 if (add or rm or ch) else 0

    # Without this, pointing PREPIBIND_RAW_ROOT at a path that holds no snapshot rewrites the
    # manifest as a handful of rows and the index is gone.
    snap = sum(1 for r in rows if r['state'] == 'snapshot')
    if snap == 0:
        print(f'refusing to write: no snapshot files under {RAW}', file=sys.stderr)
        return 2

    with open(MANIFEST, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f'\n-> {MANIFEST}  ({os.path.getsize(MANIFEST) / 1048576:.1f} MB, '
          f'{len(rows)} rows)', file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main())
