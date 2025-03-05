import unittest
import pickle

import torch 

from netsanut.models import GMMPred, GMMPredictionHead

def gaussian_pdf(mean, var, x):
    return torch.exp(-0.5 * ((x - mean) ** 2) / var) / torch.sqrt(2 * torch.pi * var)

class TestGMM(unittest.TestCase):

    def test_gmm_confidence_interval(self):
        # unittest.main()
        gmm_head = GMMPredictionHead(in_dim=32, hid_dim=32, anchors=[1.0, 2.0, 3.0], sizes=[1.0, 1.0, 1.0])
        N, P, C = 4, 19, 32
        rand_input = torch.rand(N, P, C)
        mixing, means, log_var = gmm_head(rand_input)
        xs = torch.linspace(0, 14, 1000).reshape(*([1] * 4), -1)
        densities_by_component = gaussian_pdf(means.unsqueeze(-1), torch.exp(log_var).unsqueeze(-1), xs)
        gmm_density = (mixing.unsqueeze(-1) * densities_by_component).sum(dim=-2)
        
        pass 

    def test_forward(self):
        
        with open("tests/simbarca_batch.pkl", "rb") as f:
            batch = pickle.load(f)

        model = GMMPred(adjacency_hop=5)
        model.train()
        model.adapt_to_metadata(batch["metadata"]) # this should do nothing, as the metadata is already set 

        loss_dict = model(batch)
        # print(loss_dict.keys())
        loss = sum(loss_dict.values())
        loss.backward()
        # print("Loss:", loss.item())

        model.eval()
        pred = model(batch)
        self.assertTrue(isinstance(pred, dict))
        # print(pred.keys())
        state_dict = model.state_dict()
        self.assertTrue(isinstance(state_dict, dict))
        model.load_state_dict(state_dict)
    
if __name__ == "__main__":
    unittest.main()