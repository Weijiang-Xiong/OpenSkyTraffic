import unittest

from netsanut.data import NetworkedTimeSeriesDataset, tensor_collate, tensor_to_contiguous
import torch
from torch.utils.data import DataLoader

class TestDataset(unittest.TestCase):
    
    def test_init_and_iteration(self):
        dataset = NetworkedTimeSeriesDataset(compute_metadata=True)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True, collate_fn=tensor_collate)

        for data, label in dataloader:
            self.assertIsInstance(data, torch.Tensor)
            self.assertIsInstance(label, torch.Tensor)
            break

        dataloader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True, collate_fn=tensor_to_contiguous)

        for data in dataloader:
            self.assertIsInstance(data['source'], torch.Tensor)
            self.assertIsInstance(data['target'], torch.Tensor)
            break

if __name__ == "__main__":
    unittest.main()