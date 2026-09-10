import os
import torch.nn as nn
import torch.optim as optim
from prepibind import model as model                       # Change here if you have a different `model.py` file
from prepibind import encoder as encoder                   # Change here if you have a different `encoder.py` file
from prepibind import collate as collate

# This arm is the DeepNeo baseline: it encodes from a contact matrix and reads no HDF5
# embedding store, so it needs no PREPIBIND_EMB_ROOT and no store-path helper. Every other
# config in this directory defines one and requires the variable.

config = {
    "chkp_name"         : "deepneo",
    "model"             : model.DeepNeo,
    "model_args"        : {
        "kernel_size"   : (8, 133),
    },
    "encoder"           : encoder.deepneo,
    "encoder_args"      : {
        "matrix_size"   : (15, 269),
    },
    "Train": {
        "regularize"    : True,            # true if regularize method is implemented in the model
        "criterion"     : nn.BCELoss,
        "optimizer"     : optim.SGD,
        "optimizer_args": {
            "momentum"  : 0.9,
        },
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
deep_update(config, {"Train": {"optimizer_args": {"lr": 0.001}}})

# config["Train"]["optimizer_args"]["lr"] = 0.01
