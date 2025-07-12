import unittest
import torch
import numpy as np

from skytraffic.data.datasets.simbarca_comm import SimBarcaSpeed


class TestSimBarcaSpeed(unittest.TestCase):
    
    def test_dataset_initialization(self):
        dataset = SimBarcaSpeed(split="train", input_nan_to_global_avg=True)
        sample = dataset[77]
        self.assertTrue(sample['source'].isnan().sum() == 0)
        
        dataset_nan_input = SimBarcaSpeed(split="train", input_nan_to_global_avg=False)
        sample = dataset_nan_input[77]
        self.assertTrue(sample['source'].isnan().sum() > 0)

    def test_dataset_collate_fn(self):
        dataset = SimBarcaSpeed(split="train", input_nan_to_global_avg=True)
        batch = [dataset[i] for i in range(10)]
        collated_batch = dataset.collate_fn(batch)
        self.assertTrue(collated_batch['source'].shape == (10, 10, 1570, 2))


if __name__ == '__main__':
    unittest.main()
