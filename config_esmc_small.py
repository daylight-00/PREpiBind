import torch.nn as nn
import torch.optim as optim
import model as model                       # Change here if you have a different `model.py` file
import encoder as encoder                   # Change here if you have a different `encoder.py` file
import collate as collate

config = {
    "chkp_name"         : "esmc_small",
    "chkp_path"         : "models",
    # "plot_path"         : "plots",
    # "seed"              : 128,
    "model"             : model.plm_cat_mean_inf,
    "model_args"        : {
        "hla_dim_s"       : 960,
        "hla_dim_p"       : 0,
        "epi_dim_s"       : 960,
        "epi_dim_p"       : 0,
        "head_div"        : 64,
    },
    "encoder"           : encoder.plm_plm_mask_msa_pair_inf,
    "encoder_args"      : {
        "hla_emb_path_s" : "data/emb_hla_esmc_small_light_0601.h5",
    },
    "collate_fn"         : collate.pad_and_mask_collate_fn_inf,
    "Data": {
        "hla_path"      : "data/mhc_mapping.csv",
        "hla_args"      : {
            "hla_header": 'HLA_Name',
            "seq_header": 'HLA_Seq',
            "seperator" : ",",
        },
        "test_path"     : "data/dataset.csv",
        "test_args"     : {
            "epi_header": 'Epitope',
            "hla_header": 'MHC',
            "tgt_header": 'Target',
            "seperator" : ",",
        },
        "num_workers"   : 4,
    },
    "Test": {
        "batch_size"    : 128,
        "chkp_prefix"   : "best",
        "plot"          : True,
    },
}