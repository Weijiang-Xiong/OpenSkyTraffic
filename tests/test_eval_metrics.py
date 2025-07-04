import unittest

import torch

from skytraffic.evaluation.metrics import common_metrics

class TestLoss(unittest.TestCase):

    def test_error_values(self):
        """ In the first example, the pred-label pairs (2, nan) and (4, nan) are ignored. 
            (1.0, 0.5) is ignored at MAPE (not the other two) because the label is less than threshold
            
            In the second example, we lowerd the MAPE threshould to count in the (1.0, 0.5) pair. 
        """
        pred = torch.tensor([1, 2, 3, 4, 5, 1.0])
        real = torch.tensor([1, torch.nan, 3, torch.nan, 5, 0.5])
        res = common_metrics(pred, real, ignore_value=torch.nan, mape_threshold=0.7)
        self.assertAlmostEqual(res['mae'], 0.125)
        self.assertAlmostEqual(res['mape'], 0) # MAPE is ignored for label < 1
        self.assertAlmostEqual(res['rmse'], 0.25)
        
        res = common_metrics(pred, real, ignore_value=torch.nan, mape_threshold=0.1)
        self.assertAlmostEqual(res['mae'], 0.125)
        self.assertAlmostEqual(res['mape'], 0.25)
        self.assertAlmostEqual(res['rmse'], 0.25)
    
    def test_masked_error_values(self):
        pred = torch.tensor([1, 2, 3, 4, 5, 1.0])
        real = torch.tensor([1, torch.nan, 3, torch.nan, 5, 0.5])
        res = common_metrics(pred, real, ignore_value=torch.nan)
        self.assertAlmostEqual(res['mae'], 0.125)
        self.assertAlmostEqual(res['mape'], 0.25)
        self.assertAlmostEqual(res['rmse'], 0.25)
    
if __name__ == "__main__":
    unittest.main()