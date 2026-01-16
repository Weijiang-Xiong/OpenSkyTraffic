import unittest
import torch
import numpy as np
import tempfile
import os
import torch.nn as nn

from skytraffic.evaluation.metr_gmm_evaluation import MetrGMMEvaluator

from torch.utils.data import Dataset, DataLoader

class DummyDataset(Dataset):
    """Simple dataset that generates random data"""
    
    def __init__(self, num_samples=64, num_nodes=207, input_steps=12, pred_steps=12):
        self.num_samples = num_samples
        self.num_nodes = num_nodes
        self.input_steps = input_steps
        self.pred_steps = pred_steps
        
        # Create dummy metadata once
        self.metadata = {
            'adjacency': [torch.randn(num_nodes, num_nodes)],
            'mean': torch.tensor(0.0),
            'std': torch.tensor(1.0),
            'data_dim': 0,
            'geo_loc': np.random.randn(num_nodes, 2),
            'invalid_value': 0.0
        }
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return {
            "source": torch.randn(self.input_steps, self.num_nodes, 2),
            "target": torch.randn(self.pred_steps, self.num_nodes),
        }
    
    def collate_fn(self, batch):
        """ assume the input is a list of (x, y), pack the x's and y's into two tensors
        """
        xs = [torch.as_tensor(xy["source"]).unsqueeze(0) for xy in batch]
        ys = [torch.as_tensor(xy["target"]).unsqueeze(0) for xy in batch]
        
        xs, ys = torch.cat(xs, dim=0), torch.cat(ys, dim=0)
        
        return {"source": xs.contiguous(), "target": ys.contiguous()}

def dummy_dataloader(batch_size=16, num_nodes=207, input_steps=12, pred_steps=12, num_samples=64):
    """Create a DataLoader with dummy data"""
    dataset = DummyDataset(
        num_samples=num_samples,
        num_nodes=num_nodes,
        input_steps=input_steps,
        pred_steps=pred_steps
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate_fn)

class DummyModel(nn.Module):
    def __init__(self, num_nodes=207, pred_steps=12, num_mixtures=5):
        super().__init__()
        self.num_nodes = num_nodes
        self.pred_steps = pred_steps
        self.num_mixtures = num_mixtures

    def forward(self, x):
        """Simple function that creates dummy GMM outputs"""
        # Generate dummy GMM outputs
        batch_size = x['source'].size(0)
        pred = torch.randn(batch_size, self.pred_steps, self.num_nodes)
        mixing = torch.softmax(torch.randn(batch_size, self.pred_steps, self.num_nodes, self.num_mixtures), dim=-1)
        means = torch.randn(batch_size, self.pred_steps, self.num_nodes, self.num_mixtures)
        log_var = torch.randn(batch_size, self.pred_steps, self.num_nodes, self.num_mixtures)
        
        return {
            "pred": pred,
            "mixing": mixing,
            "means": means,
            "log_var": log_var
        }

    
class TestMetrGMMEvaluator(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.num_nodes = 207
        self.input_steps = 12
        self.pred_steps = 12
        self.batch_size = 16
        
        # Create simple dataloader
        self.dataloader = dummy_dataloader(
            batch_size=self.batch_size,
            num_nodes=self.num_nodes,
            input_steps=self.input_steps,
            pred_steps=self.pred_steps
        )
        
        # Model parameters for creating dummy outputs
        self.num_mixtures = 5
        
        # Create simple mock model
        self.model = DummyModel(num_nodes=self.num_nodes, pred_steps=self.pred_steps, num_mixtures=self.num_mixtures)
        
        # Create temporary directory for evaluator
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures"""
        # Clean up temporary directory
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)


    def test_collect_predictions(self):
        """Test that the evaluator can collect predictions correctly"""
        evaluator = MetrGMMEvaluator(save_dir=self.temp_dir)
        
        all_preds, all_data = evaluator.collect_predictions(
            self.model, 
            self.dataloader,
            pred_seqs=["pred", "mixing", "means", "log_var"],
            data_seqs=["target"]
        )
        
        # Check that predictions were collected
        self.assertIn("pred", all_preds)
        self.assertIn("mixing", all_preds)
        self.assertIn("means", all_preds)
        self.assertIn("log_var", all_preds)
        self.assertIn("target", all_data)
        
        # Check shapes are correct
        expected_samples = 4 * self.batch_size  # 4 batches * batch_size
        self.assertEqual(all_preds["pred"].shape[0], expected_samples)
        self.assertEqual(all_data["target"].shape[0], expected_samples)
    
    def test_evaluate_method(self):
        """Test the main evaluate method"""
        evaluator = MetrGMMEvaluator(save_dir=self.temp_dir)
        
        results = evaluator.evaluate(self.model, self.dataloader, verbose=False)
        
        # Check that results contain expected metrics
        self.assertIsInstance(results, dict)
        self.assertIn("mae", results)
        self.assertIn("mape", results)
        self.assertIn("rmse", results)
        self.assertIn("CRPS_GMM_GT", results)
        self.assertIn("mCCE", results)
        self.assertIn("mAW", results)
        
        # Check that all metrics are numeric
        for key, value in results.items():
            self.assertIsInstance(value, (int, float))
            self.assertFalse(np.isnan(value))


if __name__ == "__main__":
    unittest.main()
