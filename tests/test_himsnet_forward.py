import unittest

import torch
from netsanut.models import HiMSNet, ValueEmbedding

class TestHimsNetForward(unittest.TestCase):
    
    def test_value_embedding(self):
        emb_layer = ValueEmbedding(d_model=3)
        emb_layer.time_emb_w.data.fill_(2.0)
        emb_layer.time_emb_b.data.fill_(0.5)
        emb_layer.value_emb_w.data.fill_(1.0)
        emb_layer.value_emb_b.data.fill_(0.5)
        emb_layer.empty_token.data.fill_(6.0)
        emb_layer.unmonitored_token.data.fill_(7.0)

        # shape (N, C, P, T)
        example_input = torch.ones((16,)).reshape(2, 2, 2, 2)
        example_input[0, 1, 0, 0] = torch.nan
        example_input[1, 0, 1, 0] = torch.nan
        # (N, P, T)
        unmonitored = torch.empty((2, 2, 2)).fill_(0.0).bool()
        unmonitored[0, 0, 1] = True
        
        out = emb_layer(example_input, unmonitored=unmonitored)
        self.assertTrue(torch.all(out[0, 1, 0, :]== 8.5).item()) # 6 for empty token, 2.5 for time 
        self.assertTrue(torch.all(out[1, 0, 1, :]== 8.5).item()) 
        self.assertTrue(torch.all(out[0, 0, 1, :]== 9.5).item()) # 7 for empty token, 2.5 for time 
        self.assertTrue(torch.all(out[1, 0, 0, :]== 4.0).item()) # 1.5 from value_emb, 2.5 for time 
        
    
    def test_forward(self):
        adjacency = torch.randint(low=0, high=2, size=(1570, 1570)).cuda()
        edge_index = adjacency.nonzero().t().contiguous()
        fake_data_dict = {
            "drone_speed": torch.rand(size=(2, 360, 1570, 2)).cuda(),
            "ld_speed": torch.rand(size=(2, 10, 1570, 2)).cuda(),
            "pred_speed": torch.rand(size=(2, 10, 1570)).cuda(),
            "pred_speed_regional": torch.rand(size=(2, 10, 4)).cuda(),
            "metadata": {
                "adjacency": adjacency,
                "edge_index": edge_index,
                "cluster_id": torch.randint(low=0, high=4, size=(1570,)).cuda(),
                "grid_id": torch.randint(low=0, high=150, size=(1570,)).cuda(),
                "mean_and_std": {
                    "drone_speed": (0.0, 1.0),
                    "ld_speed": (0.0, 1.0),
                    "pred_speed": (0.0, 1.0),
                    "pred_speed_regional": (0.0, 1.0),
                },
                "input_seqs": ["drone_speed", "ld_speed"],
                "output_seqs": ["pred_speed", "pred_speed_regional"],
            },
        }
        empty_mask = torch.rand_like(fake_data_dict['drone_speed'][:,:,:,0]) < 0.1
        unmonitored_mask = torch.rand_like(fake_data_dict['drone_speed']) > 0.8
        fake_data_dict['drone_speed'][:,:,:,0][empty_mask] = torch.nan
        fake_data_dict['drone_unmonitored'] = unmonitored_mask
        
        model = HiMSNet().cuda()
        model.adapt_to_metadata(fake_data_dict['metadata'])
        model.train()
        loss_dict = model(fake_data_dict)
        loss = sum(loss_dict.values())
        loss.backward()

        model.eval()
        res = model(fake_data_dict)

if __name__ == "__main__":
    unittest.main()