import unittest
import pickle

import torch 
import numpy as np
from scipy.stats import norm

from skytraffic.models import HiMSNet_GMM
from skytraffic.models.gmmpred import GMMPredictionHead, gaussian_density

class TestGMM(unittest.TestCase):

    def test_gaussian_density_against_scipy(self):
        """Verify our implementation matches scipy's normal distribution PDF."""
        
        # Test cases with different means, log variances, and evaluation points
        test_cases = [
            {"mean": 0.0, "log_var": 0.0, "x": torch.linspace(-3, 3, 100)},  # Standard normal
            {"mean": 2.5, "log_var": np.log(2.0), "x": torch.linspace(-2, 7, 100)},  # Mean=2.5, var=2.0
            {"mean": -1.0, "log_var": np.log(0.5), "x": torch.linspace(-5, 3, 100)},  # Mean=-1.0, var=0.5
            {"mean": 0.0, "log_var": np.log(5.0), "x": torch.linspace(-10, 10, 100)},  # Mean=0, var=5.0
        ]
        
        for case in test_cases:
            mean = torch.tensor(case["mean"], dtype=torch.float32)
            log_var = torch.tensor(case["log_var"], dtype=torch.float32)
            x = case["x"].to(torch.float32)
            
            # Calculate density using our function
            density = gaussian_density(mean, log_var, x)
            
            # Calculate density using scipy (which uses standard deviation)
            std = np.sqrt(torch.exp(log_var).item())
            scipy_density = torch.tensor(norm.pdf(x.numpy(), loc=mean.item(), scale=std), dtype=torch.float32)
            
            # Check that results are very close
            self.assertTrue(torch.allclose(density, scipy_density, rtol=1e-5, atol=1e-5), 
                            msg=f"Failed for mean={mean.item()}, log_var={log_var.item()}")

    def test_gaussian_density(self):
        import numpy as np
        from scipy.stats import norm
        
        density_value = gaussian_density(torch.tensor([1.0]), torch.log(torch.tensor([2.0])), torch.tensor([3.0]))
        # Check if the density value is close to the expected value
        expected_value = norm.pdf(3.0, loc=1.0, scale=np.sqrt(2.0))
        
        self.assertTrue(np.allclose(density_value.numpy(), expected_value, atol=1e-6))
        self.addCleanup(lambda: torch.cuda.empty_cache())  # Clear GPU memory after test
        
    def test_gmm_density(self):
        N, T, P, C = 4, 10, 19, 32
        xmin, xmax, n_points = 0, 6, 1000

        mixing = torch.ones(N, T, P, 3) / 3 # equal weights everywhere
        means = torch.ones(N, T, P, 3) * torch.tensor([1.0, 3.0, 5.0]) # evenly spaced from 0 to 6
        log_var = -2 * torch.ones(N, T, P, 3) # variances are e^-2 everywhere
        
        xs = torch.linspace(xmin, xmax, n_points)
        mixture_density = GMMPredictionHead.get_mixture_density(mixing, means, log_var, xs)
        self.assertTrue(mixture_density.shape == (N, T, P, 1000))
        var = torch.tensor(-2).exp().item()
        scipy_value = 1/3 * (norm.pdf(xs.numpy(), loc=1.0, scale=np.sqrt(var)) + 
                             norm.pdf(xs.numpy(), loc=3.0, scale=np.sqrt(var)) +
                             norm.pdf(xs.numpy(), loc=5.0, scale=np.sqrt(var)))
        self.assertTrue(torch.allclose(mixture_density[0, 0, 0, :], torch.tensor(scipy_value, dtype=torch.float32)))
    
    def test_gmm_confidence_interval(self):
        N, T, P, C = 4, 10, 19, 32
        xmin, xmax, n_points = 0, 6, 1000

        mixing = torch.ones(N, T, P, 3) / 3 # equal weights everywhere
        means = torch.ones(N, T, P, 3) * torch.tensor([1.0, 3.0, 5.0]) # evenly spaced from 0 to 6
        log_var = -2 * torch.ones(N, T, P, 3) # variances are e^-2 everywhere
        
        xs = torch.linspace(xmin, xmax, n_points)
        lb, ub = GMMPredictionHead.get_confidence_interval(mixing, means, log_var, xs, 0.70)
        pass 
        

    def test_forward(self):
        
        with open("tests/simbarca_batch.pkl", "rb") as f:
            batch = pickle.load(f)

        model = HiMSNet_GMM(adjacency_hop=5)
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