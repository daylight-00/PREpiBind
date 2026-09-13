# `tools/release/`

What produces the published artifacts, and the text that ships with them. Nothing here talks to
HuggingFace; uploading is a human step with the `hf` CLI.

| file | what it is |
|---|---|
| `convert_checkpoints.py` | training checkpoint -> release file. `--check` compares against what is on disk, `--verify` loads each one into the model its predict config builds |
| `model_card_prepibind.md` | card for [`daylight-00/prepibind`](https://huggingface.co/daylight-00/prepibind), the float32 checkpoints |
| `model_card_prepibind_demo.md` | card for [`daylight-00/prepibind-demo`](https://huggingface.co/daylight-00/prepibind-demo), the float16 copies |
| `dataset_card_prepibind_embeddings.md` | card for [`daylight-00/prepibind-embeddings`](https://huggingface.co/datasets/daylight-00/prepibind-embeddings), the full-length HLA store |
| `deposit/` | the archival deposit: inventory, staging script, and the text that ships inside it |

Each card is uploaded **as `README.md`** into its repository; they are kept here under descriptive
names so all three can live in one directory.

## The two tiers

`convert_checkpoints.py` writes both from the same training checkpoint, into `models/`, which
`.gitignore` excludes:

| tier | name | precision | size each | read by |
|---|---|---|---|---|
| research | `prepibind_<arm>_s<seed>_f<fold>.pt` | float32 | 213.0 MiB | anyone reproducing the paper's path |
| demo | `prepibind_<arm>_s<seed>_f<fold>_fp16.pt` | float16 | 106.5 MiB | `configs/predict/*.py`, both notebooks |

They land in `models/` rather than a directory of their own because that is where the predict
configs look, so `python -m prepibind.inference configs/predict/config_demo.py` works straight after
a write. The optimizer state and epoch counter are dropped; nothing else is changed.

Which checkpoint each arm releases is the lowest-validation-loss run of its fifteen, from
`analysis/val_metrics.csv`. `.github/workflows/ci.yml` checks that the four predict configs, the
README and both notebooks all name the same four files.
