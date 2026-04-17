import unittest
import torch

from skytraffic.data.datasets import MetrDataset
from skytraffic.models import (
    ForecastModel,
    GMMTensorDataNormalizer,
    GWNET,
    GWNET_GMM,
    LSTMGCNConv,
    LSTMGCNConv_GMM,
    MTGNN,
    MTGNN_GMM,
    STAEformer,
    STAEformer_GMM,
    STIDGMMNet,
    STIDNet,
    TensorDataNormalizer,
)



class TestModels(unittest.TestCase):

    def setUp(self):
        self.device = torch.device("cuda")
        self.dataset = MetrDataset(split="test")
        self.batch = self.dataset.collate_fn([self.dataset[0], self.dataset[10], self.dataset[20]])
        self.batch = {k: v.to(self.device) for k, v in self.batch.items()}
        self.N, self.T, self.P, _ = self.batch['source'].shape

    def build_wrapper(self, model, normalizer):
        return ForecastModel(
            model=model,
            normalizer=normalizer,
            data_null_value=self.dataset.data_null_value,
            metadata=self.dataset.metadata,
        )

    def build_lgc_model(self):
        return self.build_wrapper(
            LSTMGCNConv(
                input_steps=self.dataset.input_steps,
                pred_steps=self.dataset.pred_steps,
                num_nodes=self.dataset.num_nodes,
                assume_clean_input=True,
                metadata=self.dataset.metadata,
            ),
            TensorDataNormalizer(),
        )

    def build_stid_model(self):
        return self.build_wrapper(
            STIDNet(
                input_steps=self.dataset.input_steps,
                pred_steps=self.dataset.pred_steps,
                num_nodes=self.dataset.num_nodes,
            ),
            TensorDataNormalizer(),
        )

    def build_gwnet_model(self):
        return self.build_wrapper(
            GWNET(
                input_steps=self.dataset.input_steps,
                pred_steps=self.dataset.pred_steps,
                num_nodes=self.dataset.num_nodes,
                metadata=self.dataset.metadata,
            ),
            TensorDataNormalizer(),
        )

    def build_mtgnn_model(self):
        return self.build_wrapper(
            MTGNN(
                input_steps=self.dataset.input_steps,
                pred_steps=self.dataset.pred_steps,
                num_nodes=self.dataset.num_nodes,
                metadata=self.dataset.metadata,
            ),
            TensorDataNormalizer(),
        )

    def build_staeformer_model(self):
        return self.build_wrapper(
            STAEformer(
                input_steps=self.dataset.input_steps,
                pred_steps=self.dataset.pred_steps,
                num_nodes=self.dataset.num_nodes,
                metadata=self.dataset.metadata,
            ),
            TensorDataNormalizer(),
        )

    def build_lgc_gmm_model(self):
        return self.build_wrapper(
            LSTMGCNConv_GMM(
                input_steps=self.dataset.input_steps,
                pred_steps=self.dataset.pred_steps,
                num_nodes=self.dataset.num_nodes,
                assume_clean_input=True,
                metadata=self.dataset.metadata,
            ),
            GMMTensorDataNormalizer(),
        )

    def build_stid_gmm_model(self):
        return self.build_wrapper(
            STIDGMMNet(
                input_steps=self.dataset.input_steps,
                pred_steps=self.dataset.pred_steps,
                num_nodes=self.dataset.num_nodes,
            ),
            GMMTensorDataNormalizer(),
        )

    def build_gwnet_gmm_model(self):
        return self.build_wrapper(
            GWNET_GMM(
                input_steps=self.dataset.input_steps,
                pred_steps=self.dataset.pred_steps,
                num_nodes=self.dataset.num_nodes,
                metadata=self.dataset.metadata,
            ),
            GMMTensorDataNormalizer(),
        )

    def build_mtgnn_gmm_model(self):
        return self.build_wrapper(
            MTGNN_GMM(
                input_steps=self.dataset.input_steps,
                pred_steps=self.dataset.pred_steps,
                num_nodes=self.dataset.num_nodes,
                metadata=self.dataset.metadata,
            ),
            GMMTensorDataNormalizer(),
        )

    def build_staeformer_gmm_model(self):
        return self.build_wrapper(
            STAEformer_GMM(
                input_steps=self.dataset.input_steps,
                pred_steps=self.dataset.pred_steps,
                num_nodes=self.dataset.num_nodes,
                metadata=self.dataset.metadata,
            ),
            GMMTensorDataNormalizer(),
        )

    def test_deterministic_models(self):
        print("Testing deterministic models forward pass")
        model_builders = [
            ("LSTMGCNConv", self.build_lgc_model),
            ("STID", self.build_stid_model),
            ("GWNET", self.build_gwnet_model),
            ("MTGNN", self.build_mtgnn_model),
            ("STAEformer", self.build_staeformer_model),
        ]

        for model_name, build_model in model_builders:
            print(f"Testing {model_name}")
            model = build_model().to(self.device)
            model.train()
            loss = model(self.batch)
            model.eval()
            pred = model(self.batch)
            self.assertEqual(pred['pred'].shape, (self.N, self.T, self.P))
            print(f"{model_name} test OK")

    def test_probabilistic_models(self):
        print("Testing probabilistic models forward pass")
        model_builders = [
            ("LSTMGCNConv_GMM", self.build_lgc_gmm_model),
            ("STID_GMM", self.build_stid_gmm_model),
            ("GWNET_GMM", self.build_gwnet_gmm_model),
            ("MTGNN_GMM", self.build_mtgnn_gmm_model),
            ("STAEformer_GMM", self.build_staeformer_gmm_model),
        ]

        for model_name, build_model in model_builders:
            print(f"Testing {model_name}")
            model = build_model().to(self.device)
            model.train()
            loss = model(self.batch)
            model.eval()
            pred = model(self.batch)
            self.assertEqual(pred['pred'].shape, (self.N, self.T, self.P))
            print(f"{model_name} test OK")

if __name__ == "__main__":
    unittest.main()
