import torch 
import unittest
from netsanut.models import NeTSFormer
from netsanut.models.netsformer import TemporalAggregate
from einops import rearrange

class TestNetsFormer(unittest.TestCase):
    
    def setUp(self) -> None:
        self.device = torch.device("cuda")
        self.model = NeTSFormer().to(self.device)
        return super().setUp()
    
    def test_model_forward(self):
        
        N, T, M, C = 32, 12, 211, 2
        rand_input = torch.rand(size=(N, T, M, C), device=self.device)
        rand_label = torch.rand(size=(N, T, M), device=self.device)

        in_rearrange = rearrange(rand_input.clone(), 'N T M C -> (N M) T C', N=N)
        in_reshape = rand_input.clone().permute((0, 2, 1, 3)).reshape(N*M, T, C)
        self.assertTrue(in_rearrange.shape == in_reshape.shape) 
        self.assertTrue(torch.allclose(in_rearrange, in_reshape))

        rand_data = {'source': rand_input, 'target': rand_label}

        
        self.model.adapt_to_metadata({'adjacency': [torch.randint(0, 5, size=(M, M)) for _ in range(2)],
                                'mean': torch.tensor([0.0, 0.0]),
                                'std': torch.tensor([1.0, 0.0])})

        self.model.train()
        loss_dict = self.model(rand_data)
        loss = sum(loss_dict.values())
        loss.backward()

        self.model.eval()
        result_dict = self.model(rand_data)
        self.assertTrue(result_dict['pred'].shape == (N, T, M))
        self.assertTrue(result_dict['logvar'].shape == (N, T, M))
        
        for mode in ['linear', 'last', 'avg']:        
            agg = TemporalAggregate(in_dim=T, mode=mode).to(self.device)
            out = agg(rand_input)
            self.assertTrue(out.shape == (N, M, C))
            
    def test_param_group(self):
        param_groups = self.model.get_param_groups()
        all_params = set(self.model.parameters())
        self.assertTrue(sum([len(g) for g in param_groups.values()]) == len(all_params))
        for group_name, group_params in param_groups.items():
            for param in group_params:
                self.assertTrue(param in all_params)

        print("Num Params. {:.2f}M".format(self.model.num_params/1e6))
        
if __name__ == "__main__":
    unittest.main()