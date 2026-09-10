import os
import torch.nn as nn
import torch.optim as optim
from prepibind import model as model                       # Change here if you have a different `model.py` file
from prepibind import encoder as encoder                   # Change here if you have a different `encoder.py` file
from prepibind import collate as collate

# The HDF5 embedding stores are not distributed with the repository, so no path in a fresh clone
# holds them and there is no default: PREPIBIND_EMB_ROOT must name the directory they are in.
# pipeline/embeddings/ is the code that produces them; the ESMC store is also published as the
# HuggingFace dataset daylight-00/prepibind-embeddings.
def emb_path(*parts):
    """A path inside the embedding store root, or a clear error if it was never named."""
    root = os.environ.get("PREPIBIND_EMB_ROOT")
    if not root:
        raise SystemExit(
            "PREPIBIND_EMB_ROOT is not set. Point it at the directory holding the HDF5 embedding "
            "stores (emb_*.h5); they are not distributed with this repository. "
            "See pipeline/embeddings/README.md.")
    return os.path.join(root, *parts)

config = {
    "chkp_name"         : "esm3_medium",
    "model"             : model.plm_cat_mean,
    "model_args"        : {
        "hla_dim_s"       : 2560,
        "hla_dim_p"       : 0,
        "epi_dim_s"       : 2560,
        "epi_dim_p"       : 0,
        "head_div"        : 64,
    },
    "encoder"           : encoder.plm_plm_mask_msa_pair,
    "encoder_args"      : {
        "hla_emb_path_s" : emb_path("esm_large/emb_hla_esm3_medium_2408_0430.h5"),
        "epi_emb_path_s" : emb_path("esm_large/emb_epi_esm3_medium_2408_0430.h5"),
        # "hla_emb_path_p" : emb_path("emb_hla_af3_pair_side_0430.h5"),
        # "epi_emb_path_p" : emb_path("emb_epi_af3_pair_side_0430.h5"),
    },
    "collate_fn"         : collate.pad_and_mask_collate_fn,
    "Train": {
        "regularize"    : False,            # true if regularize method is implemented in the model
        "criterion"     : nn.BCEWithLogitsLoss,
        "optimizer"     : optim.AdamW,
    },
    "Test": {
    },
}

def deep_update(source, overrides):
    for key, value in overrides.items():
        if isinstance(value, dict) and key in source and isinstance(source[key], dict):
            deep_update(source[key], value)
        else:
            source[key] = value

import os
import json
current_dir = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(current_dir, "config_global.json")
with open(path, 'r') as f:
    config_global = json.load(f)
    deep_update(config, config_global)

# Learning rate as used for the published runs: 1e-3 for BLOSUM62 and DeepNeo, 1e-5 for the rest.
deep_update(config, {"Train": {"optimizer_args": {"lr": 1e-05}}})

# config["Train"]["batch_size"] = 16
