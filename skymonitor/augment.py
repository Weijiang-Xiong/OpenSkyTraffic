import numpy as np
import torch 
from typing import Dict, List, Tuple

class RandomGridCoverage:
    """
    Randomly sample monitoring positions and build coverage masks.

    Init parameters:
        pts_per_step: number of time series data points per monitoring time step, e.g., traffic data every 5 seconds, then in a 3 minutes monitoring window, pts_per_step=36
        cvg_num: number of monitoring positions to sample at each time step ( number of drones )
        empty_value: the value to fill in the unmonitored positions (by default 0.0)
        data_dims: number of data dimensions (excluding time feature, by assuming the input shape is N T P C, applied to the C axis)
    """

    def __init__(
        self,
        pts_per_step: int, 
        cvg_num: int = 10,
        empty_value: float = 0.0,
        data_dims: int = 2, 
    ) -> None:
        self.pts_per_step = pts_per_step
        self.cvg_num = cvg_num
        self.empty_value = empty_value
        self.data_dims = data_dims

    def set_grid(self, grid_xy_to_id: Dict[Tuple[int, int], int], grid_id_of_nodes: np.ndarray):
        self.available_positions: List[Tuple[int, int]] = list(grid_xy_to_id.keys())
        if not self.available_positions:
            raise ValueError("No available drone positions defined.")

        self.total_positions = len(self.available_positions)

        gids = [grid_xy_to_id[pos] for pos in self.available_positions]
        self.available_gids = torch.tensor(gids, dtype=torch.long)
        self.grid_id_of_nodes = torch.as_tensor(grid_id_of_nodes, dtype=torch.long)

    def sample(self, batch_size: int) -> torch.Tensor:
        random_scores = torch.rand(batch_size, self.num_time_steps, self.total_positions)
        sampled_indices = random_scores.argsort(dim=-1, descending=True)[..., : self.cvg_num]
        sampled_gids = self.available_gids[sampled_indices]
        masks = sampled_gids.unsqueeze(-1).eq(self.grid_id_of_nodes).any(dim=2)

        return masks

    def __call__(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        source = data_dict["source"]

        masks = self.sample(batch_size=source.shape[0])
        expanded_masks = masks.repeat_interleave(self.pts_per_step, dim=1)
        expanded_masks = expanded_masks[:, : source.shape[1], :]
        data_values = source.clone()[..., :self.data_dims]
        data_values[expanded_masks == 0] = self.empty_value
        masked_source = torch.cat([data_values, source[..., self.data_dims:]], dim=-1)  # keep time feature unchanged

        data_dict["source"] = masked_source

        return data_dict


class RandomWalkCoverage(RandomGridCoverage):
    """Sample monitoring positions by performing simple random walks."""

    def __init__(
        self,
        pts_per_step: int, 
        cvg_num: int = 10,
        empty_value: float = 0.0,
    ) -> None:
        super().__init__(pts_per_step, cvg_num, empty_value)

    def set_grid(self, grid_xy_to_id: Dict[Tuple[int, int], int], grid_id_of_nodes: np.ndarray):
        super().set_grid(grid_xy_to_id, grid_id_of_nodes)

        index_lookup = {pos: idx for idx, pos in enumerate(self.available_positions)}
        neighbor_offsets = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        self.neighbor_indices: List[torch.Tensor] = []
        for pos in self.available_positions:
            neighbors: List[int] = []
            x, y = pos
            for dx, dy in neighbor_offsets:
                neighbor = (x + dx, y + dy)
                if neighbor in index_lookup:
                    neighbors.append(index_lookup[neighbor])
            if not neighbors:
                neighbors.append(index_lookup[pos])
            self.neighbor_indices.append(torch.tensor(neighbors, dtype=torch.long))

    def sample(self, batch_size: int) -> torch.Tensor:
        walk_indices = torch.empty(batch_size, self.num_time_steps, self.cvg_num, dtype=torch.long)

        # randomly initialize starting positions
        current_indices = torch.randint(
            low=0,
            high=self.total_positions,
            size=(batch_size, self.cvg_num),
            dtype=torch.long,
        )
        walk_indices[:, 0, :] = current_indices

        # randomly move to a neighbor
        for step in range(1, self.num_time_steps):
            next_indices = torch.empty(batch_size, self.cvg_num, dtype=torch.long)
            for batch_idx in range(batch_size):
                all_neighbors = []
                for pos_idx in range(self.cvg_num):
                    current_idx = current_indices[batch_idx, pos_idx].item()
                    neighbors = self.neighbor_indices[current_idx]
                    all_neighbors.append(neighbors)

                # we need to avoid overlapping, so we gather all neighbors first and sample without replacement
                unique_neighbors = torch.cat(all_neighbors, dim=0).unique()
                chosen_neighbors = unique_neighbors[torch.randperm(unique_neighbors.shape[0])[: self.cvg_num]]
                next_indices[batch_idx, :] = chosen_neighbors

            walk_indices[:, step, :] = next_indices
            current_indices = next_indices

        sampled_gids = self.available_gids[walk_indices]
        coverage_masks = sampled_gids.unsqueeze(-1).eq(self.grid_id_of_nodes).any(dim=2)

        return coverage_masks
