import unittest
import pickle

import torch 

from netsanut.models import GMMPred, GMMPredictionHead

class TestGMM(unittest.TestCase):

    def test_gmm_density(self):
        N, T, P, C = 4, 10, 19, 32
        xmin, xmax, n_points = 0, 6, 1000

        mixing = torch.ones(N, T, P, 3) / 3 # equal weights everywhere
        means = torch.ones(N, T, P, 3) * torch.tensor([1.0, 3.0, 5.0]) # evenly spaced from 0 to 6
        log_var = -2 * torch.ones(N, T, P, 3) # variances are e^-2 everywhere
        
        xs = torch.linspace(xmin, xmax, n_points)
        mixture_density = GMMPredictionHead.get_mixture_density(mixing, means, log_var, xs)
        self.assertTrue(mixture_density.shape == (N, T, P, 1000))
        # check the value at 0
        var = torch.tensor(-2).exp()
        value = sum(1/(3*torch.sqrt(torch.tensor([2*torch.pi*var]))) * torch.tensor([-1/(2*var), -9/(2*var), -25/(2*var)]).exp())
        self.assertTrue(torch.allclose(mixture_density[..., 0], value))
    
    def test_gmm_confidence_interval(self):
        N, T, P, C = 4, 10, 19, 32
        xmin, xmax, n_points = 0, 6, 1000

        mixing = torch.ones(N, T, P, 3) / 3 # equal weights everywhere
        means = torch.ones(N, T, P, 3) * torch.tensor([1.0, 3.0, 5.0]) # evenly spaced from 0 to 6
        log_var = -2 * torch.ones(N, T, P, 3) # variances are e^-2 everywhere
        
        lb, ub = GMMPredictionHead.get_confidence_interval(mixing, means, log_var, xmin, xmax, n_points, 0.70)
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