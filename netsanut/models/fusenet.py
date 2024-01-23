import torch 
import torch.nn as nn 
import torch_geometric.nn as gnn
from scipy.stats import rv_continuous, gennorm

from netsanut.data.transform import TensorDataScaler
from typing import Dict, List, Tuple
from einops import rearrange

class FuseNet(nn.Module):
    
    def __init__(self):
        
        super().__init__()
        self._calibrated_intervals: Dict[float, float] = dict() 
        # self._beta: int = beta
        # self._distribution: rv_continuous = None
        self._set_distribution(beta=1)
    
        self.metadata: Dict[str, torch.Tensor] = None
        
        self.drone_data_scaler = TensorDataScaler(mean=[3, 0], std=[2, 0], data_dim=0)
        self.ld_data_scaler = TensorDataScaler(mean=[5, 0], std=[3, 0], data_dim=0)
        self.output_scalar = TensorDataScaler(mean=[5, 0], std=[3, 0], data_dim=0)
        
        self.drone_embedding = nn.Conv2d(in_channels=2, out_channels=64, kernel_size=(1,1))
        # drone data has higher temporal resolution, so we use two conv layers to down sample
        # note that we set the kernel size and stride both to be (3,1) to keep the spatial dimension unchanged
        self.drone_t_patching_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 1), stride=(3, 1))
        self.drone_t_patching_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 1), stride=(3, 1))
        self.ld_embedding = nn.Conv2d(in_channels=2, out_channels=64, kernel_size=(1,1))
        
        self.drone_temporal = nn.GRU(input_size=64, hidden_size=64, batch_first=True)
        self.ld_temporal = nn.GRU(input_size=64, hidden_size=64, batch_first=True)
        
        self.channel_down_sample = nn.Linear(in_features=64*2, out_features=32)
        self.gcn = gnn.GCNConv(in_channels=32, out_channels=32)
        self.channel_up_sample = nn.Linear(in_features=32, out_features=64)
        self.prediction = nn.Linear(in_features=64, out_features=10)
        self.prediction_regional = nn.Linear(in_features=64, out_features=10)
        
        self.loss = nn.L1Loss()
        
    @property   
    def device(self):
        return list(self.parameters())[0].device
    
    @property
    def num_params(self):
        return sum([p.numel() for p in self.parameters() if p.requires_grad])
    
    @property
    def is_probabilistic(self):
        return self._distribution is not None
    
    def forward(self, data: dict[str, torch.Tensor]):
        """
            time series forecasting task, 
            data is assumed to have (N, T, P, C) shape (assumed to be unnormalized)
            label is assumed to have (N, T, P) shape (assumed to be unnormalized)
            
            compute loss in training mode, predict future values in inference
        """
        
        # preprocessing (if any)
        source, target = self.preprocess(data)
        
        if self.training:
            assert target is not None, "label should be provided for training"
            return self.compute_loss(source, target)
        else:
            # we should not have target sequences in inference
            return self.inference(source)
        
    def preprocess(self, data: dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        
        if self.metadata is None:
            self.adapt_to_metadata(data['metadata'])
        
        source = {'drone': self.drone_data_scaler.transform(data['drone_speed']), 
                  'ld': self.ld_data_scaler.transform(data['ld_speed'])}
        target = {'section': data['pred_speed'][..., 0], 
                  'regional': data['pred_speed_regional'][..., 0]}
        
        return source, target

    def post_process(self, prediction: torch.Tensor) -> torch.Tensor:
        return prediction
    
    def make_prediction(self, source: dict[str, torch.Tensor]) -> torch.Tensor:

        x_drone, x_ld = source['drone'], source['ld']
        N, T_drone, P, C = source['drone'].shape
        T_ld = source['ld'].shape[1]
        
        x_drone = rearrange(x_drone, 'N T P C -> N C P T')
        x_drone = self.drone_embedding(x_drone)
        x_drone = rearrange(x_drone, 'N C P T -> N C T P')
        x_drone = self.drone_t_patching_1(x_drone)
        x_drone = self.drone_t_patching_2(x_drone)
        x_drone = rearrange(x_drone, 'N C T P -> (N P) T C')
        x_drone, _ = self.drone_temporal(x_drone) # we take the last cell output
        x_drone = rearrange(x_drone[:, -1, :], '(N P) C -> N P C', N=N)
        
        x_ld = rearrange(x_ld, 'N T P C -> N C P T')
        x_ld = self.ld_embedding(x_ld)
        x_ld = rearrange(x_ld, 'N C P T -> (N P) T C')
        x_ld, _ = self.ld_temporal(x_ld)
        x_ld = rearrange(x_ld[:, -1, :], '(N P) C -> N P C', N=N)
        
        x_joint = torch.cat([x_drone, x_ld], dim=-1)
        x_joint = self.channel_down_sample(x_joint)
        
        # graph convolution
        x_joint = self.gcn(x_joint, self.metadata['edge_index'])

        x_joint = self.channel_up_sample(x_joint)
        x_joint = torch.relu(x_joint)
        
        fused_features = x_joint + x_drone + x_ld
        pred = self.prediction(fused_features)
        regional_feature = torch.cat(
                [torch.mean(fused_features[:, self.metadata['cluster_id'] == region_id, :], dim=1).unsqueeze(1)
                 for region_id in self.metadata['cluster_id'].unique()],
                dim=1
            )
        pred_regional = self.prediction_regional(regional_feature)
        
        return {'section': rearrange(pred, 'N P T -> N T P'),
                'regional': rearrange(pred_regional, 'N P T -> N T P')}
        
        
    
    def compute_loss(self, source: dict[str, torch.Tensor], target: torch.Tensor) -> torch.Tensor:
        
        pred_res = self.make_prediction(source)
        loss = self.loss(pred_res['section'], target['section'])
        loss_regional = self.loss(pred_res['regional'], target['regional'])
        
        return {'loss': loss, 'loss_regional': loss_regional}

    def inference(self, source: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.post_process(self.make_prediction(source))
    
    def adapt_to_metadata(self, metadata):
        
        self.metadata = metadata
        self.metadata['edge_index'] = metadata['adjacency'].nonzero().t().contiguous()

    def _set_distribution(self, beta:int) -> rv_continuous:
        
        if beta is None:
            self._distribution = None 
        else:
            self._distribution = gennorm(beta=int(beta))

        return self._distribution
    
def build_model(cfg):
    
    print("build FuseNet")
    
    return FuseNet()

# some initial test codes
if __name__ == "__main__":
    
    fake_data_dict = {
        "drone_speed": torch.rand(size=(2, 360, 1570, 2)).cuda(),
        "ld_speed": torch.rand(size=(2, 10, 1570, 2)).cuda(),
        "pred_speed": torch.rand(size=(2, 10, 1570, 2)).cuda(),
        "pred_speed_regional": torch.rand(size=(2, 10, 4, 2)).cuda(),
        "metadata": {
            "adjacency": torch.randint(low=0, high=2, size=(1570, 1570)).cuda(),
            "cluster_id": torch.randint(low=0, high=4, size=(1570,)).cuda(),
            "grid_id": torch.randint(low=0, high=150, size=(1570,)).cuda()
        },
    }
    
    model = FuseNet().cuda()
    
    model.train()
    loss_dict = model(fake_data_dict)
    loss = sum(loss_dict.values())
    loss.backward()
    
    model.eval()
    res = model(fake_data_dict)
