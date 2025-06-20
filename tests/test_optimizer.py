import unittest
import torch.nn as nn 

from omegaconf import OmegaConf
from netsanut.config import ConfigLoader
from netsanut.solver import build_optimizer

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 10)
        self.linear2 = nn.Linear(10, 10)
        self.linear3 = nn.Linear(10, 10)
        
    def forward(self, x):
        return x

class TestOptimizer(unittest.TestCase):
    
    def test_build_optimizer(self):
        model  = SimpleModel()

        cfg = ConfigLoader.load_from_file("config/common_cfg.py")
        
        sgd = build_optimizer(model, cfg.sgd)
        self.assertIsNotNone(sgd)

        adam = build_optimizer(model, cfg.adam)
        self.assertIsNotNone(adam)
        
        adamw = build_optimizer(model, cfg.adamw)
        self.assertIsNotNone(adamw)
        
    def test_hyperparam_overrides(self):
        model  = SimpleModel()
        cfg = OmegaConf.create({
            "type": "adam",
            "lr": 0.001,
            "weight_decay": 0.001,
            "betas": (0.9, 0.999),
            "overrides": {
                "linear1": {"lr": 0.01, "weight_decay": 0.01},
            }
        })
        optimizer = build_optimizer(model, cfg)
        self.assertIsNotNone(optimizer)
        
if __name__ == "__main__":
    unittest.main()