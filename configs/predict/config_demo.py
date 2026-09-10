import os

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DEMO = os.path.join(_REPO, "demo", "data")
_MODELS = os.path.join(_REPO, "models")

from prepibind import model as model
from prepibind import encoder as encoder
from prepibind import collate as collate

config = {
    "model"             : model.plm_cat_mean_inf,
    "model_args"        : {
        "hla_dim"       : 960,
        "epi_dim"       : 960,
        "head_div"      : 64,
    },
    "encoder"           : encoder.plm_plm_mask_msa_pair_inf,
    "encoder_args"      : {
        "hla_emb_path"  : os.path.join(_DEMO, "emb_hla_esmc_small_demo_fp16.h5"),
    },
    "collate_fn"        : collate.pad_and_mask_collate_fn_inf,
    "Data": {
        "hla_path"      : os.path.join(_DEMO, "mhc_mapping_demo.csv"),
        "hla_args"      : {
            "hla_header": 'HLA_Name',
            "seq_header": 'HLA_Seq',
            "separator" : ",",
        },
        "test_path"     : os.path.join(_DEMO, "dataset_demo.csv"),
        "test_args"     : {
            "epi_header": 'Epitope',
            "hla_header": 'MHC',
            "separator" : ",",
        },
        "num_workers"   : 8,
    },
    "Test": {
        "batch_size"    : 512,
        "chkp_path"     : os.path.join(_MODELS, "prepibind_qualitative_s100_f0_fp16.pt"),
        "esm_chkp_path" : os.path.join(_MODELS, "esmc_300m_2024_12_v0_fp16.pth"),
        "plot"          : True,
        "use_compile"   : False,
        "out_path"      : "outputs",
        # Half precision throughout: this config is the Colab demo, and it encodes epitopes with
        # ESMC at run time, so it cannot match the precomputed-embedding path bit for bit whatever
        # it does. See prepibind.inference.PRECISION.
        "precision"     : "fp16",
    },
}
