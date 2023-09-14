import unittest

from netsanut.models import NeTSFormer
from netsanut.config import ConfigLoader
from netsanut.solver import build_optimizer

class TestOptimizer(unittest.TestCase):
    
    def test_build_optimizer(self):
        model  = NeTSFormer()

        cfg = ConfigLoader.load_from_file("config/NeTSFormer_prediction.py")
        optimizer = build_optimizer(model, cfg.optimizer)
        self.assertIsNotNone(optimizer)

        cfg = ConfigLoader.load_from_file("config/NeTSFormer_uncertainty.py")
        optimizer = build_optimizer(model, cfg.optimizer)
        self.assertIsNotNone(optimizer)

        cfg = ConfigLoader.load_from_file("config/TTNet_stable.py")
        optimizer = build_optimizer(model, cfg.optimizer)
        self.assertIsNotNone(optimizer)

if __name__ == "__main__":
    unittest.main()