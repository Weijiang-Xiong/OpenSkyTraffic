""" Customized policy network for Stable Baselines3 PPO agent.
    see https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html#advanced-example
"""

from typing import Callable, Dict, Tuple

import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces.utils import flatdim

import torch
import torch.nn as nn
import torch_geometric.nn as gnn

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class SimpleFeatureExtractor(BaseFeaturesExtractor):

    def __init__(self, observation_space: spaces.Dict, hidden_dim: int = 256):
        base_dim = max(8, hidden_dim // 8)
        time_dim = max(4, hidden_dim // 32)
        pos_emb_dim = max(4, hidden_dim // 32)

        flow_in = int(flatdim(observation_space["flow"]))
        density_in = int(flatdim(observation_space["density"]))
        time_in = int(flatdim(observation_space["time_in_day"]))
        coverage_in = int(flatdim(observation_space["coverage_mask"]))

        num_x = int(observation_space["positions_x"].nvec.max())
        num_y = int(observation_space["positions_y"].nvec.max())

        features_dim = (base_dim * 3) + time_dim + (2 * pos_emb_dim)
        super().__init__(observation_space, features_dim=features_dim)

        self.flow_encoder = nn.Linear(flow_in, base_dim)
        self.density_encoder = nn.Linear(density_in, base_dim)
        self.time_encoder = nn.Linear(time_in, time_dim)
        self.coverage_encoder = nn.Linear(coverage_in, base_dim)

        self.x_embed = nn.Embedding(num_x, pos_emb_dim)
        self.y_embed = nn.Embedding(num_y, pos_emb_dim)

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        flow = observations["flow"].float().flatten(start_dim=1)
        density = observations["density"].float().flatten(start_dim=1)
        time_in_day = observations["time_in_day"].float().flatten(start_dim=1)
        coverage = observations["coverage_mask"].float().flatten(start_dim=1)

        flow_feat = self.flow_encoder(flow)
        density_feat = self.density_encoder(density)
        time_feat = self.time_encoder(time_in_day)
        coverage_feat = self.coverage_encoder(coverage)

        pos_x = observations["positions_x"].long()
        pos_y = observations["positions_y"].long()
        pos_x_emb = self.x_embed(pos_x).mean(dim=1)
        pos_y_emb = self.y_embed(pos_y).mean(dim=1)

        return torch.cat(
            [flow_feat, density_feat, time_feat, coverage_feat, pos_x_emb, pos_y_emb],
            dim=1,
        )


class GraphFeatureExtractor(BaseFeaturesExtractor):

    def __init__(
        self,
        observation_space: spaces.Dict,
        adjacency_matrix,
        hidden_dim: int = 256,
        gnn_hidden_dim: int = None,
    ):
        if adjacency_matrix is None:
            raise ValueError("adjacency_matrix is required for GraphFeatureExtractor.")
        if gnn_hidden_dim is None:
            gnn_hidden_dim = hidden_dim

        pos_emb_dim = max(4, hidden_dim // 32)
        time_dim = int(flatdim(observation_space["time_in_day"]))
        aux_dim = time_dim + (2 * pos_emb_dim)

        super().__init__(observation_space, features_dim=hidden_dim)

        adj = torch.as_tensor(adjacency_matrix)
        edge_index = (adj > 0).nonzero(as_tuple=False).t().contiguous()
        self.register_buffer("edge_index", edge_index)

        num_x = int(observation_space["positions_x"].nvec.max())
        num_y = int(observation_space["positions_y"].nvec.max())

        self.x_embed = nn.Embedding(num_x, pos_emb_dim)
        self.y_embed = nn.Embedding(num_y, pos_emb_dim)

        self.gcn = gnn.GCNConv(in_channels=3, out_channels=gnn_hidden_dim, node_dim=1)
        self.relu = nn.ReLU()
        self.encoder = nn.Linear(gnn_hidden_dim + aux_dim, hidden_dim)

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        flow = observations["flow"].float()
        density = observations["density"].float()
        coverage = observations["coverage_mask"].float()

        node_features = torch.stack([flow, density, coverage], dim=-1)
        graph_features = self.relu(self.gcn(node_features, self.edge_index))
        graph_features = graph_features.mean(dim=1)

        time_feat = observations["time_in_day"].float().flatten(start_dim=1)
        pos_x = observations["positions_x"].long()
        pos_y = observations["positions_y"].long()
        pos_x_emb = self.x_embed(pos_x).mean(dim=1)
        pos_y_emb = self.y_embed(pos_y).mean(dim=1)

        aux_features = torch.cat(
            [
                time_feat,
                pos_x_emb,
                pos_y_emb,
            ],
            dim=1,
        )

        return self.encoder(torch.cat([graph_features, aux_features], dim=1))


class GridCNNFeatureExtractor(BaseFeaturesExtractor):

    def __init__(
        self,
        observation_space: spaces.Dict,
        map_structure,
        hidden_dim: int = 256,
        pos_emb_dim: int = 16,
        cnn_hidden_dim: int = None,
    ):
        if map_structure is None:
            raise ValueError("map_structure is required for GridCNNFeatureExtractor.")
        if cnn_hidden_dim is None:
            cnn_hidden_dim = hidden_dim

        grid_xy = torch.as_tensor(map_structure.grid_xy, dtype=torch.long)
        grid_width = int(grid_xy[:, 0].max().item()) + 1
        grid_height = int(grid_xy[:, 1].max().item()) + 1
        num_cells = grid_width * grid_height

        grid_index = grid_xy[:, 0] + (grid_xy[:, 1] * grid_width)
        grid_counts = torch.zeros(num_cells, dtype=torch.float32)
        grid_counts.scatter_add_(0, grid_index, torch.ones_like(grid_index, dtype=torch.float32))
        grid_counts = torch.clamp(grid_counts, min=1.0)

        grid_pos = torch.arange(num_cells, dtype=torch.long)
        grid_x = grid_pos % grid_width
        grid_y = grid_pos // grid_width

        super().__init__(observation_space, features_dim=hidden_dim)

        self.grid_width = grid_width
        self.grid_height = grid_height
        self.num_cells = num_cells

        self.register_buffer("grid_index", grid_index)
        self.register_buffer("grid_counts", grid_counts)
        self.register_buffer("grid_x", grid_x)
        self.register_buffer("grid_y", grid_y)

        self.x_embed = nn.Embedding(self.grid_width, pos_emb_dim)
        self.y_embed = nn.Embedding(self.grid_height, pos_emb_dim)

        in_channels = 2 + (2 * pos_emb_dim)
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, cnn_hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(cnn_hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def _grid_mean(self, values: torch.Tensor) -> torch.Tensor:
        batch_size = values.shape[0]
        idx = self.grid_index.unsqueeze(0).expand(batch_size, -1)
        grid = torch.zeros(batch_size, self.num_cells, device=values.device, dtype=values.dtype)
        grid.scatter_add_(1, idx, values)
        return grid / self.grid_counts

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        flow = observations["flow"].float()
        density = observations["density"].float()

        flow_grid = self._grid_mean(flow)
        density_grid = self._grid_mean(density)

        grid = torch.stack([flow_grid, density_grid], dim=1)
        grid = grid.reshape(flow.shape[0], 2, self.grid_height, self.grid_width)

        pos_emb = torch.cat(
            [self.x_embed(self.grid_x), self.y_embed(self.grid_y)],
            dim=1,
        )
        pos_emb = pos_emb.view(self.grid_height, self.grid_width, -1).permute(2, 0, 1).unsqueeze(0)
        pos_emb = pos_emb.expand(flow.shape[0], -1, -1, -1)

        features = torch.cat([grid, pos_emb], dim=1)
        encoded = self.cnn(features)
        return encoded.mean(dim=(2, 3))


class CustomMLPExtractor(nn.Module):
    """Two-head MLP for PPO's actor and critic."""

    def __init__(self, input_dim: int, policy_dim: int = 256, value_dim: int = 256):
        super().__init__()
        self.policy_net = nn.Sequential(
            nn.Linear(input_dim, policy_dim),
            nn.ReLU(),
            nn.Linear(policy_dim, policy_dim),
            nn.ReLU(),
        )
        self.value_net = nn.Sequential(
            nn.Linear(input_dim, value_dim),
            nn.ReLU(),
            nn.Linear(value_dim, value_dim),
            nn.ReLU(),
        )
        self.latent_dim_pi = policy_dim
        self.latent_dim_vf = value_dim

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ The forward function should return the feature for policy and value separately.
        """
        return self.forward_actor(features), self.forward_critic(features)
    
    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return self.policy_net(features)
    
    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.value_net(features) 


class SimpleDronePolicy(ActorCriticPolicy):
    """Actor-critic policy that plugs the monitoring extractor into PPO."""

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        lr_schedule: Callable[[float], float],
        policy_hidden_dim: int = 256,
        value_hidden_dim: int = 256,
        feature_extractor_class: BaseFeaturesExtractor = SimpleFeatureExtractor,
        feature_extractor_kwargs: Dict = None,
        *args,
        **kwargs,
    ):

        self.policy_hidden_dim = policy_hidden_dim
        self.value_hidden_dim = value_hidden_dim

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=[], # leave it empty as the custom MLP extractor will consume required parameters directly.
            features_extractor_class=feature_extractor_class,
            features_extractor_kwargs=feature_extractor_kwargs,
            *args,
            **kwargs,
        )
    
    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomMLPExtractor(
            input_dim=self.features_dim, # defined in the parent class, equal to the hidden dim of feature extractor
            policy_dim=self.policy_hidden_dim,
            value_dim=self.value_hidden_dim,
        )

class GraphDronePolicy(SimpleDronePolicy):
    """Actor-critic policy that encodes graph features with a GNN."""

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        lr_schedule: Callable[[float], float],
        policy_hidden_dim: int = 256,
        value_hidden_dim: int = 256,
        feature_extractor_class: BaseFeaturesExtractor = GraphFeatureExtractor,
        feature_extractor_kwargs: Dict = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            policy_hidden_dim=policy_hidden_dim,
            value_hidden_dim=value_hidden_dim,
            feature_extractor_class=feature_extractor_class,
            feature_extractor_kwargs=feature_extractor_kwargs,
            *args,
            **kwargs,
        )

class GridDronePolicy(SimpleDronePolicy):
    """Actor-critic policy that encodes grid features with a CNN."""

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        lr_schedule: Callable[[float], float],
        policy_hidden_dim: int = 256,
        value_hidden_dim: int = 256,
        feature_extractor_class: BaseFeaturesExtractor = GridCNNFeatureExtractor,
        feature_extractor_kwargs: Dict = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            policy_hidden_dim=policy_hidden_dim,
            value_hidden_dim=value_hidden_dim,
            feature_extractor_class=feature_extractor_class,
            feature_extractor_kwargs=feature_extractor_kwargs,
            *args,
            **kwargs,
        )
