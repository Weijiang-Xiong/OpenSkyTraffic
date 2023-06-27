from netsanut.models import NeTSFormer
from netsanut.config import ConfigLoader
from netsanut.solver import build_optimizer

model  = NeTSFormer()

cfg = ConfigLoader.load_from_file("config/NeTSFormer_stable.py")
optimizer = build_optimizer(model, cfg.optimizer)

cfg = ConfigLoader.load_from_file("config/NeTSFormer_uncertainty.py")
optimizer = build_optimizer(model, cfg.optimizer)

cfg = ConfigLoader.load_from_file("config/TTNet_stable.py")
optimizer = build_optimizer(model, cfg.optimizer)