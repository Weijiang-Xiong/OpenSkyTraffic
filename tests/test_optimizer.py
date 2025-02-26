import unittest
import torch.nn as nn 

from netsanut.config import ConfigLoader
from netsanut.solver import build_optimizer

class TestOptimizer(unittest.TestCase):
    
    def test_build_optimizer(self):
        model  = nn.Sequential(
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 10)
        )

        cfg = ConfigLoader.load_from_file("config/common_cfg.py")
        
        sgd = build_optimizer(model, cfg.sgd)
        self.assertIsNotNone(sgd)

        adam = build_optimizer(model, cfg.adam)
        self.assertIsNotNone(adam)
        
        adamw = build_optimizer(model, cfg.adamw)
        self.assertIsNotNone(adamw)
        
if __name__ == "__main__":
    unittest.main()