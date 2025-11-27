""" Customized policy network for Stable Baselines3 PPO agent.
    see https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html#advanced-example
"""

from typing import Callable, Dict, Tuple

import gymnasium as gym
from gymnasium import spaces

import torch
import torch.nn as nn

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class GridFeatureExtractor(BaseFeaturesExtractor):

    def __init__(self, observation_space: spaces.Dict, hidden_dim: int = 256):
        traffic_shape = observation_space["observed_traffic"].shape
        coverage_shape = observation_space["coverage_mask"].shape
        self.num_locations = int(coverage_shape[-1])
        self.feature_dim = int(traffic_shape[-1])
        flattened_dim = (self.num_locations * self.feature_dim) + self.num_locations

        super().__init__(observation_space, features_dim=hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(flattened_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        traffic = observations["observed_traffic"].mean(dim=1)
        traffic_flat = traffic.flatten(start_dim=1)
        coverage = observations["coverage_mask"].float().view(traffic_flat.shape[0], -1)
        features = torch.cat([traffic_flat, coverage], dim=1)
        return self.mlp(features)


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


class DronePolicy(ActorCriticPolicy):
    """Actor-critic policy that plugs the monitoring extractor into PPO."""

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        lr_schedule: Callable[[float], float],
        policy_hidden_dim: int = 256,
        value_hidden_dim: int = 256,
        feature_extractor_class: BaseFeaturesExtractor = GridFeatureExtractor,
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