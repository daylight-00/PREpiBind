"""
Path resolver for the 0_raw snapshot.

Design
------
* `0_raw/` is a **portable snapshot** holding only what is needed to reproduce the
  paper, so analyses run unchanged on clusters that have no experiment data.
* The snapshot **mirrors the scratch path verbatim**:

      0_raw/<scratch-relative-path>

  so a file's provenance is its path. `source(rel)` is just `SCRATCH / rel`, and
  directories that share a `<lr>_<seed>` name can never overwrite each other -
  the flaw that silently dropped 194 files from the old flattened layout.
* `raw_manifest.csv` records integrity (md5, size) and which analysis used each
  file, plus dependencies deliberately left out (`state=external`), ones small enough to ship
  inside figures/data/figure_inputs.tar.zst (`state=bundled`), and ones whose originals are gone
  (`state=lost`).
* All training/inference data was produced on cluster `abc`, and the scratch root
  is the scratch tree named by IMG_SCRATCH. The other clusters are snapshot
  consumers only, so they are not recorded.

Environment
-----------
    IMG_RAW_ROOT   location of 0_raw   (default: 0_raw next to this file)
    IMG_SCRATCH    scratch root the snapshot mirrors (no default)

Usage
-----
    import rawpath as rp

    rp.at('250513/2_lomo/plots_X/e5_s100/pred-esmc_small_fold0.csv')  # snapshot path
    rp.at('250513/lomo/HLA-DRB1~15:01_HLA-DRA~01:01/test.csv')        # inputs too
    rp.data('dataset/full/test.csv')   # the repo's own data/, not the snapshot
    rp.source(rel)                    # original scratch path
    rp.find(abs_path)                 # reverse: scratch path -> snapshot rel
    rp.table(root='250513/2_lomo')    # manifest slice as a DataFrame
    rp.verify(rel)                    # md5 against the manifest
    rp.status()                       # what is visible here
"""
from __future__ import annotations

import csv
import fnmatch
import hashlib
import os
import socket

_HERE = os.path.dirname(os.path.abspath(__file__))          # .../analysis


def _find_repo_root(start):
    """Walk up until a directory holding pyproject.toml is found, so this package works
    wherever it is placed inside the repository."""
    d = start
    while True:
        if os.path.exists(os.path.join(d, 'pyproject.toml')):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            return os.path.abspath(os.path.join(start, '..'))
        d = parent


_REPO = _find_repo_root(_HERE)

#: Prediction snapshot. Not distributed with the repository; see analysis/README.md.
RAW_ROOT = os.environ.get('PREPIBIND_RAW_ROOT') or os.environ.get('IMG_RAW_ROOT') or os.path.join(_HERE, '0_raw')
SCRATCH = os.environ.get('IMG_SCRATCH', '')
DATA_ROOT = os.path.join(_REPO, 'data')
MANIFEST = os.path.join(_HERE, 'raw_manifest.csv')

#: The handful of raw tables the figure notebooks read directly, shipped with the repository so
#: that every figure reproduces without the 1.2 GB prediction snapshot. Extracted on first use.
FIGURE_BUNDLE = os.path.join(_HERE, 'figures', 'data', 'figure_inputs.tar.zst')
BUNDLE_CACHE = os.path.join(_HERE, 'figures', 'data', 'extracted')


def _from_bundle(rel):
    """Resolve `rel` out of the shipped bundle, extracting it once. Returns None if absent."""
    cached = os.path.join(BUNDLE_CACHE, rel)
    if os.path.exists(cached):
        return cached
    if not os.path.exists(FIGURE_BUNDLE):
        return None
    import tarfile
    try:
        import zstandard
    except ImportError:
        return None
    os.makedirs(BUNDLE_CACHE, exist_ok=True)
    with open(FIGURE_BUNDLE, 'rb') as fh:
        with zstandard.ZstdDecompressor().stream_reader(fh) as z:
            with tarfile.open(fileobj=z, mode='r|') as tar:
                tar.extractall(BUNDLE_CACHE, filter='data')
    return cached if os.path.exists(cached) else None


#: cluster that produced the training/inference data
SRC_CLUSTER = 'abc'
#: host we are running on right now
HOST = socket.gethostname()

_M: dict[str, dict] | None = None


# ---------------------------------------------------------------- manifest
def manifest() -> dict[str, dict]:
    """rel -> manifest row, snapshot rows only. Read once and cached."""
    global _M
    if _M is None:
        with open(MANIFEST, newline='') as fh:
            _M = {r['rel']: r for r in csv.DictReader(fh)
                  if r['state'] == 'snapshot' and r['rel']}
    return _M


def all_rows() -> list[dict]:
    """Every row, including the external and lost entries."""
    with open(MANIFEST, newline='') as fh:
        return list(csv.DictReader(fh))


def row(rel: str) -> dict:
    try:
        return manifest()[rel]
    except KeyError:
        raise KeyError(f'{rel} is not in raw_manifest.csv') from None


# ---------------------------------------------------------------- reading
def _declared(rel: str) -> dict | None:
    """The manifest row for a path that is knowingly absent from the snapshot."""
    for r in all_rows():
        if r['state'] in ('external', 'lost') and r['src_path'] == source(rel):
            return r
    return None


def at(rel: str, missing_ok: bool = False) -> str:
    """Snapshot path for a scratch-relative path. Works on every cluster."""
    p = os.path.join(RAW_ROOT, rel)
    if os.path.exists(p):
        return p
    # The few raw tables the figure notebooks read are shipped with the repository, so the figure
    # layer works with no snapshot at all. Everything else needs it.
    b = _from_bundle(rel)
    if b:
        return b
    if not missing_ok:
        d = _declared(rel)
        if d and d['state'] == 'lost':
            raise FileNotFoundError(
                f'{rel} is recorded as lost: {d["note"]}.'
            )
        if d and d['state'] == 'external':
            raise FileNotFoundError(
                f'{rel} is deliberately outside the snapshot: {d["note"]}.\n'
                f'Use rp.external({rel!r}) to reach the original on '
                f'cluster {SRC_CLUSTER}.'
            )
        raise FileNotFoundError(
            f'{rel} is not in the snapshot ({RAW_ROOT}). Check that the snapshot was '
            f'unpacked and IMG_RAW_ROOT is correct; if this is a new experiment, add '
            f'its root to tools/sync_snapshot.py and re-sync.'
        )
    return p


def external(rel: str) -> str:
    """An `external` dependency, reachable only where the scratch tree is mounted.

    These are the HLA embedding .h5 files, too large to ship. figures/6_umap.py
    falls back to figures/umap_cache/*.npz when they are unavailable.
    """
    p = source(rel)
    if not os.path.exists(p):
        d = _declared(rel)
        raise FileNotFoundError(
            f'{rel} is not on this host ({HOST}); it lives on cluster {SRC_CLUSTER}.'
            + (f' {d["note"]}' if d else '')
        )
    return p


#: kept so older calls keep working
raw = at


def data(rel: str) -> str:
    """A file under the repo's own data/, replacing old
    scratch-tree references."""
    p = os.path.join(DATA_ROOT, rel)
    if not os.path.exists(p):
        raise FileNotFoundError(f'{rel} is not under the repo data/ ({DATA_ROOT})')
    return p


# ---------------------------------------------------------------- provenance
def source(rel: str) -> str:
    """Original scratch path. Identity mapping, since the snapshot mirrors it."""
    return os.path.join(SCRATCH, rel)


def source_exists(rel: str) -> bool:
    return os.path.exists(source(rel))


def resolve(rel: str, prefer: str = 'raw') -> str:
    """A path that can actually be opened. prefer='source' favours the original."""
    if prefer == 'source' and source_exists(rel):
        return source(rel)
    p = os.path.join(RAW_ROOT, rel)
    if os.path.exists(p):
        return p
    if source_exists(rel):
        return source(rel)
    raise FileNotFoundError(
        f'{rel}: present neither in the snapshot nor at the source '
        f'(raw={p}, source={source(rel)}, host={HOST})'
    )


def find(abs_path: str) -> str | None:
    """Reverse lookup: scratch (or snapshot) absolute path -> snapshot rel."""
    for base in (SCRATCH, RAW_ROOT):
        if abs_path.startswith(base.rstrip('/') + '/'):
            rel = os.path.relpath(abs_path, base)
            return rel if rel in manifest() else None
    return None


# ---------------------------------------------------------------- query / verify
def glob(pattern: str) -> list[str]:
    """Glob over rel keys, e.g. rp.glob('250513/2_lomo/plots_*/e5_*/pred-*_fold?.csv')"""
    return sorted(r for r in manifest() if fnmatch.fnmatch(r, pattern))


def verify(rel: str) -> bool:
    """Whether the snapshot file still matches the md5 recorded in the manifest."""
    h = hashlib.md5()
    with open(at(rel), 'rb') as fh:
        while (b := fh.read(1 << 20)):
            h.update(b)
    return h.hexdigest() == row(rel)['md5']


def table(root: str = '', pattern: str = ''):
    """Manifest slice as a pandas DataFrame; 'path' holds an openable path.

    `root` matches a leading scratch path (e.g. '250513/2_lomo'), `pattern` is a
    glob over the whole rel.
    """
    import pandas as pd

    df = pd.DataFrame(list(manifest().values()))
    if root:
        df = df[df['rel'].str.startswith(root.rstrip('/') + '/')]
    if pattern:
        df = df[df['rel'].map(lambda r: fnmatch.fnmatch(r, pattern))]
    df = df.copy()
    df['path'] = df['rel'].map(lambda r: os.path.join(RAW_ROOT, r))
    df['size'] = df['size'].astype(int)
    return df.reset_index(drop=True)


def status() -> dict:
    rows = all_rows()
    snap = [r for r in rows if r['state'] == 'snapshot']
    return {
        'host': HOST,
        'src_cluster': SRC_CLUSTER,
        'raw_root': RAW_ROOT,
        'scratch': SCRATCH,
        'data_root': DATA_ROOT,
        'snapshot_rows': len(snap),
        'snapshot_present': sum(1 for r in snap
                                if os.path.exists(os.path.join(RAW_ROOT, r['rel']))),
        'source_present': sum(1 for r in snap if os.path.exists(r['src_path'])),
        'used_by_analysis': sum(1 for r in snap if r['used_by']),
        'external': sum(1 for r in rows if r['state'] == 'external'),
        'bundled': sum(1 for r in rows if r['state'] == 'bundled'),
        'lost': sum(1 for r in rows if r['state'] == 'lost'),
    }


if __name__ == '__main__':
    for k, v in status().items():
        print(f'{k:20s} {v}')
