import unittest
import torch 
from skytraffic.models.utils.transform import TensorDataScaler

class TestScaler(unittest.TestCase):
    
    def test_scaling_tensor(self):
        
        rand_data  = torch.rand((32, 12, 207, 2), requires_grad=False)
        data_dim = 0

        scaler = TensorDataScaler(torch.mean(rand_data, dim=(0, 1, 2))[data_dim], torch.std(rand_data, dim=(0, 1, 2))[data_dim], data_dim=data_dim)
        scaled_data = scaler.transform(rand_data.clone())
        self.assertTrue(torch.allclose(rand_data[..., 1], scaled_data[..., 1], atol=1e-7))

        scaled_back_full = scaler.inverse_transform(scaled_data.clone(), datadim_only=True)
        self.assertTrue(torch.allclose(scaled_back_full, rand_data, atol=1e-7))

        scaled_back_input_data_only = scaler.inverse_transform(scaled_data.clone()[..., data_dim])
        self.assertTrue(torch.allclose(scaled_back_input_data_only, rand_data[..., data_dim], atol=1e-7))


    def test_multiple_dims(self):

        xs = (2 * torch.arange(1, 13, 1).view(2, 2, 3) + 1).type(torch.float32)
        scaler = TensorDataScaler(mean=[1, 2], std=[2, 4], data_dim=[0, 1])

        scaled_xs = scaler.transform(xs.clone(), datadim_only=True)
        self.assertTrue(torch.allclose(scaled_xs[:,:,0], (xs[:,:,0] - 1) / 2))
        self.assertTrue(torch.allclose(scaled_xs[:,:,1], (xs[:,:,1] - 2) / 4))
        self.assertTrue(torch.allclose(scaled_xs[:,:,2], xs[:,:,2]))

        scaled_back = scaler.inverse_transform(scaled_xs.clone(), datadim_only=True)
        self.assertTrue(torch.allclose(scaled_back, xs))

        
if __name__ == "__main__":
    unittest.main()
