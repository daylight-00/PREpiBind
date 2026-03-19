import model as model
import encoder as encoder
import collate as collate

config = {
    "model"             : model.plm_cat_mean_inf,
    "model_args"        : {
        "hla_dim"       : 960,
        "epi_dim"       : 960,
        "head_div"      : 64,
    },
    "encoder"           : encoder.plm_plm_mask_msa_pair_inf,
    "encoder_args"      : {
        "hla_emb_path"  : "data/emb_hla_esmc_small_light_0601_fp16.h5",
    },
    "collate_fn"        : collate.pad_and_mask_collate_fn_inf,
    "Data": {
        "hla_path"      : "data/mhc_mapping_light.csv",
        "hla_args"      : {
            "hla_header": 'HLA_Name',
            "seq_header": 'HLA_Seq',
            "separator" : ",",
        },
        "test_path"     : "data/dataset_demo.csv",
        "test_args"     : {
            "epi_header": 'Epitope',
            "hla_header": 'MHC',
            "separator" : ",",
        },
        "num_workers"   : 8,
    },
    "Test": {
        "batch_size"    : 512,
        "chkp_path"     : "models/prepi_esmc_small_e5_s128_f4_fp16.pth",
        "esm_chkp_path" : "models/esmc_300m_2024_12_v0_fp16.pth",
        "plot"          : True,
        "use_compile"   : False,
        "out_path"      : "output",
    },
}
