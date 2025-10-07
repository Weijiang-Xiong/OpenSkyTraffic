import unittest
import pickle
import torch
from skytraffic.models import HiMSNet
from skytraffic.models.layers import ValueEmbedding

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
        
        out = emb_layer(example_input, monitor_mask=torch.logical_not(unmonitored))
        self.assertTrue(torch.all(out[0, 1, 0, :]== 8.5).item()) # 6 for empty token, 2.5 for time 
        self.assertTrue(torch.all(out[1, 0, 1, :]== 8.5).item()) 
        self.assertTrue(torch.all(out[0, 0, 1, :]== 9.5).item()) # 7 for empty token, 2.5 for time 
        self.assertTrue(torch.all(out[1, 0, 0, :]== 4.0).item()) # 1.5 from value_emb, 2.5 for time 
        
    
    def test_forward(self):
        with open("tests/simbarca_batch.pkl", "rb") as f:
            batch = pickle.load(f)

        model = HiMSNet(adjacency_hop=5, tf_glb=True).cuda()
        model.train()
        model.adapt_to_metadata(batch["metadata"])
        loss_dict = model(batch)
        print(loss_dict.keys())
        loss = sum(loss_dict.values())
        loss.backward()

        model.eval()
        res = model(batch)
        self.assertTrue("pred_speed" in res.keys())
        self.assertTrue("pred_speed_regional" in res.keys())
        
    def test_state_dict(self):
        
        model = HiMSNet().cuda()
        adjacency = torch.randint(low=0, high=2, size=(1570, 1570)).cuda()
        edge_index = adjacency.nonzero().t().contiguous()
        metadata = {
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
        }
        model.adapt_to_metadata(metadata)
        
        state_dict = model.state_dict()
        model.load_state_dict(state_dict)

    def test_gnn_forward(self):
        from torch_geometric.nn import GCNConv
        gcn = GCNConv(5, 5, node_dim=1)
        gcn.lin.weight.data = torch.eye(5)
        gcn.bias.data.fill_(0.0)
        
        # create one hot node encoding as input
        # this is equivalent to torch.eye(5).unsqueeze(0).tile((3, 1, 1)), just try advanced indexing
        x = torch.zeros(2, 5, 5)
        indexes = torch.arange(5).tile((2, 1)).reshape(-1, 5)
        x[torch.arange(2).reshape(-1, 1), torch.arange(5).reshape(1, -1), indexes] = 1
        self.assertTrue(torch.allclose(x, torch.eye(5).unsqueeze(0).tile((2, 1, 1))))
        
        edge_index = torch.tensor([
            [0, 1, 2, 3, 4, 4],
            [1, 2, 3, 4, 0, 1]
        ]) # basically a directed ring with a special case 4 -> 2
        
        # one can create adjacency matrix from edge_index with advanced indexing technique
        # adj = torch.zeros(3, 3)
        # adj[edge_index[0], edge_index[1]] = 1
        
        out = gcn(x, edge_index)
        for org, dst in zip(edge_index[0], edge_index[1]):
            self.assertTrue(torch.all(out[:, dst, org]))
            out[:, dst, org] = 0.0
        # self-loops are enabled by default initialization settings, so we need to zero them out
        out[:, torch.arange(5), torch.arange(5)] = 0.0
        self.assertTrue(torch.all(out == 0.0))
        
if __name__ == "__main__":
    unittest.main()