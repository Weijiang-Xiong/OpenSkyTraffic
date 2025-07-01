import unittest

from skytraffic.data.datasets import MetrDataset
import torch
from torch.utils.data import DataLoader

class TestDataset(unittest.TestCase):
    
    def test_init_and_iteration(self):
        dataset = MetrDataset()
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True, collate_fn=dataset.collate_fn)

        for batch in dataloader:
            self.assertIsInstance(batch['source'], torch.Tensor)
            self.assertIsInstance(batch['target'], torch.Tensor)
            break

if __name__ == "__main__":
    unittest.main()