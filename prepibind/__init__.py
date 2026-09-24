"""PREpiBind: protein-representation-integrated epitope-MHC class II binding prediction.

    from prepibind import model, encoder, collate

The training and inference entry points are modules, so they run either way:

    python -m prepibind.train   configs/train/config_esmc_small.py
    python -m prepibind.test    configs/train/config_esmc_small.py
    python -m prepibind.inference configs/predict/config_demo.py
"""
__version__ = "1.0.1"
