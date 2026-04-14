import torch.nn as nn
import torch.optim as optim
import model as model                       # Change here if you have a different `model.py` file
import encoder as encoder                   # Change here if you have a different `encoder.py` file
import collate as collate

config = {
    "chkp_name"         : "esmc_small",
    "chkp_path"         : "models",
    "log_file"          : "train.log",
    "plot_path"         : "plots",
    "seed"              : 128,

    "model"             : model.plm_cat_mean,
    "model_args"        : {
        "hla_dim_s"       : 960,
        "hla_dim_p"       : 0,
        "epi_dim_s"       : 960,
        "epi_dim_p"       : 0,
        "head_div"        : 64,
    },

    "encoder"           : encoder.plm_plm_mask_msa_pair,
    "encoder_args"      : {
        "hla_emb_path_s" : "../emb/emb_hla_esmc_small.h5",
        "epi_emb_path_s" : "../emb/emb_epi_esmc_small.h5",
        # "hla_emb_path_p" : "",
        # "epi_emb_path_p" : "",
    },
    "collate_fn"        : collate.pad_and_mask_collate_fn,
    "CrossValidation": {
        "num_folds"     : 5,
    },

    "Data": {
        "epi_path"      : "../data/dataset/full/train.csv",
        "epi_args"      : {
            "epi_header": 'Epi_Seq',
            "hla_header": 'HLA_Name',
            "tgt_header": 'Target',
            "separator" : ",",
        },
        "hla_path"      : "../data/mhc_mapping/HLA2_IMGT_MSA_idx_edit.csv",
        "hla_args"      : {
            "hla_header": 'HLA_Name',
            "seq_header": 'HLA_Seq',
            "separator" : ",",
        },
        "test_path"     : "../data/dataset/full/test.csv",
        "test_args"     : {
            "epi_header": 'Epi_Seq',
            "hla_header": 'HLA_Name',
            "tgt_header": 'Target',
            "separator" : ",",
        },
        "num_workers"   : 8,
        "val_size"      : 0.2,
    },

    "Train": {
        "batch_size"    : 128,
        "num_epochs"    : 500,
        "patience"      : 10,
        "regularize"    : False,            # true if regularize method is implemented in the model
        "criterion"     : nn.BCEWithLogitsLoss,
        "optimizer"     : optim.AdamW,
        "optimizer_args": {
            "lr"        : 1e-5,
        },
        "use_scheduler" : False,
    },
    
    "Test": {
        "batch_size"    : 256,
        "chkp_prefix"   : "best",
        "plot"          : True,
        "feat_extract"  : False,
        "feat_path"     : "feat_extract",
        "target_layer"  : "output_layer",
        "save_pred"     : False,
    },
}
