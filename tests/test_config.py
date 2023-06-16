from netsanut.config import ConfigLoader
from omegaconf import OmegaConf, DictConfig

cfg: DictConfig = ConfigLoader.load_from_file("tests/example_config.py")

try:
    cfg.safe_cfg.count = "oops"
except ValueError:
    pass 
except:
    raise

print(OmegaConf.to_yaml(cfg))
# note the type safe config will not be preserved, as it requires omegaconf 
ConfigLoader.save_cfg(cfg, "tests/saved_config.py")