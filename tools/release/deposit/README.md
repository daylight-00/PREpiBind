# Archival deposit — prepared, not deposited

**Nothing here has been uploaded.** No host is chosen and no DOI exists. This directory is the
inventory, the staging script, and the text that ships with the deposit.

```
inventory.csv       every candidate artifact, with size and sha256
stage_deposit.sh    assembles the deposit directory; writes only inside --out
DEPOSIT_README.md   ships inside the deposit, for whoever downloads it
description.md      abstract, file manifest, and what differs between hosts
```

```bash
bash tools/release/deposit/stage_deposit.sh --out /some/scratch/prepibind-deposit
bash tools/release/deposit/stage_deposit.sh --check      # verify the sources, write nothing
```

It copies two files, packs a third, writes `SHA256SUMS` and `README.md` into the output directory,
and re-verifies everything it wrote. Both modes were run end to end on 2026-09-10: `--check` in 34 s,
the full assembly in 2 m 07 s, producing 546 MiB in five files.

## What goes in

Three files, 545.3 MiB, all verified on 2026-09-10 — full checksums in `inventory.csv`.

| deposit name | bytes | what it is |
|---|---:|---|
| `0_raw_260910.tar.zst` | 308,022,032 | the prediction snapshot, 12,035 files, 1.16 GiB unpacked. `PREPIBIND_RAW_ROOT`, consumed by `make scoring` |
| `mhc_ligand_full_single_file.zip` | 251,184,299 | the IEDB export of 2025-04-21. `PREPIBIND_IEDB_EXPORT` after unzipping, consumed by `make datasets` stage 1 |
| `draft.csv.tar.zst` | 12,605,261 | the stage-1 intermediate, so a downloader can skip the 7.7 GB stage 1 |

## What it does not cover

| target | external input | where it comes from |
|---|---|---|
| `figures`, `supplementary` | none | tracked in the repository |
| `scoring` | prediction snapshot | this deposit |
| `datasets` | IEDB export | this deposit |
| `verify`, `demo-assets` | research HLA store | HuggingFace `daylight-00/prepibind-embeddings` |

Two exclusions worth stating rather than leaving to inference:

- `analysis/scoring/0_ref` is not part of `make scoring`. Rescoring NetMHCIIpan-4.3 and
  MixMHC2pred-2.0 needs those tools under their own licences; their predictions are in the snapshot
  and their derived tables are tracked.
- The ~21 GB of epitope embedding stores are in no deposit and no repository. Without them,
  "re-derive our numbers" stops at "re-aggregate our predictions".

## What a deposit still needs

1. **A host, an account and a DOI.** Deferred; everything else is ready.
2. **A tag.** A deposit that says "reproduces commit X" needs X to be a tag, not a moving `main`.
3. **A decision on the IEDB export.** Redistributing a 251 MB IEDB export under our own DOI is
   allowed by CC BY 4.0 but has not been decided here. If the answer is no, the deposit ships
   `draft.csv.tar.zst` plus the export's checksums and a download URL, and only `make datasets`
   stage 1 stops being reproducible from the deposit alone.
4. Open, and nobody has asked for it: whether the 1.25 GiB of released checkpoints belong in the
   archive as well, or whether the code gets its own DOI through a GitHub–Zenodo integration.

The older `0_raw.tar.zst` (272,919,446 B, 9,552 files, no `260905/` paths) is superseded by the
snapshot above and is **still on disk**. Delete it only once the new one is deposited and its DOI
resolves.
