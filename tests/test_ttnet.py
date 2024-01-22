import unittest
import torch 
from netsanut.models import TTNet

class TestTTNet(unittest.TestCase):
    
    def test_forward(self):
    
        device = torch.device("cuda")
            
        print("Testing TTNet forward pass...")
        N, T, M, C = 32, 12, 211, 2
        rand_input = torch.rand(size=(N, T, M, C), device=device)
        rand_label = torch.rand(size=(N, T, M), device=device)
        rand_data = {'source': rand_input, 'target': rand_label}

        adjacencies = [torch.randint(0, 2, (M, M)).bool() for _ in range(2)]
        model = TTNet().to(device).eval()
        model.set_fixed_mask(adjacencies)

        model.train()
        loss_dict = model(rand_data)
        loss = sum(loss_dict.values())
        loss.backward()

        model.eval()
        result_dict = model(rand_data)
        assert result_dict['pred'].shape == (N, T, M)
        assert result_dict['plog_sigma'].shape == (N, T, M)

        print("Num Params. {:.2f}M".format(model.num_params/1e6))
        print("Test OK")

