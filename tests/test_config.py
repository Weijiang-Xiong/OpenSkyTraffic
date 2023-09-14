import copy
import unittest
from netsanut.config import ConfigLoader
from omegaconf import OmegaConf, DictConfig

class TestConfig(unittest.TestCase):
    
    def test_cfg_load_and_save(self):

        cfg: DictConfig = ConfigLoader.load_from_file("tests/example_config.py")
        self.assertIsInstance(cfg, DictConfig)
        
        org_count = copy.deepcopy(cfg.safe_cfg.count)
        try:
            cfg.safe_cfg.count = "oops"
        except ValueError:
            pass 
        except:
            raise
        self.assertEqual(org_count, cfg.safe_cfg.count)
        
        print(OmegaConf.to_yaml(cfg))
        # note the type safe config will be saved as dictionary, and the runtime safety will be lost 
        ConfigLoader.save_cfg(cfg, "tests/__NO_GIT_saved_config.py")

if __name__ == "__main__":
    unittest.main()