# Train

One config per representation. The filenames use the size words the code was written with; this is
what each one actually is, and the same mapping is published as
`supplementary_data/prepibind_D12_method_key.csv`:

| config | representation | embedding dim | trained parameters |
|---|---|---:|---:|
| `config_esmc_small.py` | ESMC 300M | 960 | 55,820,161 |
| `config_esmc_medium.py` | ESMC 600M | 1152 | 80,365,825 |
| `config_esmc_large.py` | ESMC 6B | 2560 | 396,661,761 |
| `config_esm3_small.py` | ESM3 Small | 1536 | 142,838,785 |
| `config_esm3_medium.py` | ESM3 Medium | 2560 | 396,661,761 |
| `config_esm3_large.py` | ESM3 Large | 6144 | 2,284,204,033 |
| `config_af3.py` | AlphaFold 3 | — | 24,823,041 |
| `config_boltz.py` | Boltz-1 | — | 24,823,041 |
| `config_chai.py` | Chai-1 | — | 48,629,505 |
| `config_blosum.py` | BLOSUM62 baseline | — | 39,450 |
| `config_deepneo.py` | DeepNeo re-implementation | — | — |

`config_esmc_small.py` is the released model's config. The embedding dimension is `hla_dim_s` /
`epi_dim_s` in each file; the structure backends read side-chain stores, whose width depends on the
pair reduction.

## Workflow
```mermaid
flowchart TD
    A[run] --> B(config)
    B --> D(model)
    B --> E(encoder)

    subgraph modules
    D
    E
    end
    
    B --> X[(dataset)]
    X --> C(dataprovider)
    C --> E
    D --> F(train)
    E --> G(collate)
    B --> G
    G --> F
```

## How to run
- Pick a config. Each `config_*.py` imports `prepibind.model` and `prepibind.encoder`; change those
  two imports to use your own.
- The embedding stores are not distributed with the repository: set `PREPIBIND_EMB_ROOT` to the
  directory holding the `emb_*.h5` files, or see `pipeline/embeddings/`.
- `run.sh` trains and then evaluates, one config or several in sequence.
```bash
configs/train/run.sh configs/train/config_esmc_small.py    # one
configs/train/run.sh configs/train/config_*.py             # all eleven

# the two steps run.sh runs per config
python -m prepibind.train configs/train/config_esmc_small.py
python -m prepibind.test  configs/train/config_esmc_small.py
```
```python
# from a notebook; main takes the loaded config, not a path
from prepibind.train import load_config, main as train
train(load_config('configs/train/config_esmc_small.py'))
```

### Description of Selected Arguments in `config_*.py`
| Arguemnt | Description |
| --- | --- |
| model | Select specific class in `model.py` |
| encoder | Select specific class in `encoder.py` |
| epi_args | Arguments for epitope dataset. You can set the header name of epitope sequence, HLA name, and target in the dataset file. Specific sperator should be set according to the dataset file. (e.g. `,` or `\t`) |
| regularize | If the model has a regularization method, set it to True. (e.g. `model.DeepNeo`) |