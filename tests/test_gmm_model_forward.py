import unittest
import torch

from skytraffic.data.datasets import MetrDataset
from skytraffic.models import LSTMGCNConv
from skytraffic.models.stid import STID
from skytraffic.models.gwnet import GWNET
from skytraffic.models.mtgnn import MTGNN
from skytraffic.models.staeformer import STAEformer

from skytraffic.models import LSTMGCNConv_GMM
from skytraffic.models.stid_gmm import STID_GMM
from skytraffic.models.gwnet_gmm import GWNET_GMM
from skytraffic.models.mtgnn_gmm import MTGNN_GMM
from skytraffic.models.staeformer_gmm import STAEformer_GMM



class TestModels(unittest.TestCase):

    def setUp(self):
        self.device = torch.device("cuda")
        self.dataset = MetrDataset(split="test")
        self.batch = self.dataset.collate_fn([self.dataset[0], self.dataset[10], self.dataset[20]])
        self.batch = {k: v.to(self.device) for k, v in self.batch.items()}
        self.N, self.T, self.P, _ = self.batch['source'].shape

    def test_deterministic_models(self):
        print("Testing deterministic models forward pass")
        for model_class in [LSTMGCNConv, STID, GWNET, MTGNN, STAEformer]:
            print(f"Testing {model_class.__name__}")
            
            model = model_class(
                metadata=self.dataset.metadata
                ).to(self.device)
            model.train()
            loss = model(self.batch)
            model.eval()
            pred = model(self.batch)
            self.assertEqual(pred['pred'].shape, (self.N, self.T, self.P))
            print(f"{model_class.__name__} test OK")

    def test_probabilistic_models(self):
        print("Testing probabilistic models forward pass")
        for model_class in [LSTMGCNConv_GMM, STID_GMM, GWNET_GMM, MTGNN_GMM, STAEformer_GMM]:
            print(f"Testing {model_class.__name__}")
            model = model_class(
                metadata=self.dataset.metadata
                ).to(self.device)
            model.train()
            loss = model(self.batch)
            model.eval()
            pred = model(self.batch)
            self.assertEqual(pred['pred'].shape, (self.N, self.T, self.P))
            print(f"{model_class.__name__} test OK")

if __name__ == "__main__":
    unittest.main()