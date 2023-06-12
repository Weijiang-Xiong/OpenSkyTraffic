from netsanut.config import ConfigLoader
from omegaconf import OmegaConf, DictConfig

cfg: DictConfig = ConfigLoader.load_from_file("config/LSTM_TF_stable.py")
print(OmegaConf.to_yaml(cfg))
ConfigLoader.save_cfg(cfg, "tests/saved_cfg.py")