from config_demo import config
import copy

config = copy.deepcopy(config)
config["Test"]["chkp_path"] = "models/prepi_esmc_small_ic50_500_e5_s128_f4_fp16.pth"
