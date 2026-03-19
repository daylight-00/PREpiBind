# PREpiBind: Protein Representation-integrated Epitope-MHC Class II Binding Prediction

![banner](https://github.com/user-attachments/assets/94e3dc94-b173-40cd-9a7c-ed6c46923883)

> **PREpiBind: Protein Language Model-based MHC Class II Epitope Binding Prediction**
> Hyunyoo Jang et al. *bioRxiv* (2025) — [Paper](#) · [HuggingFace Models](https://huggingface.co/daylight00/prepibind-esmc-300m)

PREpiBind predicts MHC class II–peptide binding by leveraging pre-trained protein language model (PLM) representations. It encodes epitope sequences on-the-fly with [ESMC 300M](https://huggingface.co/daylight00/esmc-300m-2024-12) and uses pre-computed HLA embeddings for both alpha and beta chains, feeding them into a lightweight cross-attention architecture to produce a binding score.

---

## Quick Start

### Option A — Local Jupyter Notebook

```bash
git clone https://github.com/daylight-00/PREpiBind
cd PREpiBind
uv sync
```

Open `run.ipynb` and run all cells.
Cell 1 automatically downloads model weights from HuggingFace. Cell 2 runs inference and saves results to `outputs/`.

### Option B — Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/daylight-00/PREpiBind/blob/main/run_colab.ipynb)

The Colab notebook provides an interactive widget interface for entering MHC II allele pairs and epitope sequences. No local setup required.

### Option C — CLI

**1. Install dependencies**

```bash
git clone https://github.com/daylight-00/PREpiBind
cd PREpiBind
uv sync
```

**2. Download model weights**

```python
from huggingface_hub import hf_hub_download

hf_hub_download(repo_id="daylight00/esmc-300m-2024-12",      filename="esmc_300m_2024_12_v0_fp16.pth",          local_dir="models")
hf_hub_download(repo_id="daylight00/prepibind-esmc-300m",    filename="prepi_esmc_small_e5_s128_f4_fp16.pth",   local_dir="models")
```

**3. Run inference**

```bash
python code/inference.py config_demo.py --plot
```

Results are saved to `outputs/prediction.csv`. Use `--help` for all options:

```
python code/inference.py --help

positional arguments:
  config_path           Path to the config.py file.

options:
  --batch_size          Batch size (default: 512)
  --test_path           Path to input data CSV
  --hla_path            Path to HLA mapping CSV
  --hla_emb_path        Path to HLA embedding HDF5 file
  --chkp_path           Path to PREpiBind model checkpoint
  --esm_chkp_path       Path to ESMC model checkpoint
  --out_path            Path to save output files
  --num_workers         Number of DataLoader workers
  --use_compile         Enable torch.compile
  --plot                Save KDE plot of prediction scores
```

---

## Input Format

The input CSV must contain at least two columns: an MHC allele pair column and an epitope sequence column.
The MHC allele pair is two allele names joined by an underscore (e.g., `HLA-DRA*01:01_HLA-DRB1*01:01`).
Allele names must exactly match the keys in the HLA mapping file (`data/mhc_mapping.csv`).

```csv
MHC,Epitope
HLA-DRA*01:01_HLA-DRB1*01:01,PKYVKQNTLKLAT
HLA-DRA*01:01_HLA-DRB5*01:01,AYSAVTTLAEEMK
```

See `data/dataset_demo.csv` for a full example (48,352 samples).

---

## HLA Allele Coverage

| Set | Alleles | File | Notes |
|-----|--------:|------|-------|
| Light (bundled) | 134 | `data/emb_hla_esmc_small_light_0601_fp16.h5` | Included in repo, used by default |
| Full | 7,282 | `emb_hla_esmc_small_0601_fp16.h5` | ~1 GB, download from HuggingFace |

To use the full allele set, download the embedding file and update `hla_emb_path` and `hla_path` in your config:

```python
hf_hub_download(repo_id="daylight00/prepibind-esmc-300m", filename="emb_hla_esmc_small_0601_fp16.h5", local_dir="data")
```

---

## Requirements

| | Minimum | Recommended |
|-|---------|-------------|
| Python | 3.10+ | 3.11+ |
| GPU VRAM | 4 GB | 8 GB+ |
| RAM | 8 GB | 16 GB+ |

CPU-only inference is supported but significantly slower for large datasets.
A CUDA-capable GPU is strongly recommended.

## Output

| File | Description |
|------|-------------|
| `outputs/prediction.csv` | Input data with appended `Logits` and `Score` (sigmoid) columns |
| `outputs/plot.png` | KDE distribution of prediction scores (generated with `--plot`) |

---

## Citation

If you use PREpiBind in your work, please cite:

```bibtex
@article{jang2025prepibind,
  title   = {PREpiBind: Protein Language Model-based MHC Class II Epitope Binding Prediction},
  author  = {Jang, Hyunyoo and others},
  journal = {bioRxiv},
  year    = {2025},
  doi     = {10.1101/XXXX}
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
