import importlib.util as _il, os as _os
# Load the base config by path so this file works from any working directory.
_spec = _il.spec_from_file_location("_base", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "config_demo.py"))
_base = _il.module_from_spec(_spec); _spec.loader.exec_module(_base)
config = _base.config
import copy

config = copy.deepcopy(config)
config["Test"]["chkp_path"] = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))), "models/prepibind_ic50_500_s128_f2_fp16.pt")
