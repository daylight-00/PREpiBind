from config_demo import config
import copy

config = copy.deepcopy(config)
config["Test"]["chkp_path"] = "models/prepi_esmc_small_ms_e5_s100_f0_fp16.pth"
