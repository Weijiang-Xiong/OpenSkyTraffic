import numpy as np
import torch 
from typing import Dict, List, Tuple

class RandomGridCoverage:
    """Randomly sample monitoring positions and build coverage masks."""

    def __init__(
        self,
        input_window: int,  # in minutes
        step_size: int,  # in minutes
        num_positions: int = 10,
        empty_value: float = 0.0,
    ) -> None:
        self.input_window = input_window
        self.step_size = step_size
        self.num_positions = num_positions
        self.empty_value = empty_value

    def set_grid(self, grid_xy_to_id: Dict[Tuple[int, int], int], grid_id_of_nodes: np.ndarray):
        self.available_positions: List[Tuple[int, int]] = list(grid_xy_to_id.keys())
        if not self.available_positions:
            raise ValueError("No available drone positions defined.")

        self.total_positions = len(self.available_positions)
        self.num_time_steps = max(1, self.input_window // self.step_size)
        self.repeats_per_step = max(1, (self.step_size * 60) // 5)

        gids = [grid_xy_to_id[pos] for pos in self.available_positions]
        self.available_gids = torch.tensor(gids, dtype=torch.long)
        self.grid_id_of_nodes = torch.as_tensor(grid_id_of_nodes, dtype=torch.long)

    def sample(self, batch_size: int) -> torch.Tensor:
        random_scores = torch.rand(batch_size, self.num_time_steps, self.total_positions)
        sampled_indices = random_scores.argsort(dim=-1, descending=True)[..., : self.num_positions]
        sampled_gids = self.available_gids[sampled_indices]
        coverage_masks = sampled_gids.unsqueeze(-1).eq(self.grid_id_of_nodes).any(dim=2)

        return coverage_masks

    def __call__(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        source = data_dict["source"]

        masks = self.sample(batch_size=source.shape[0])
        expanded_masks = masks.repeat_interleave(self.repeats_per_step, dim=1)
        expanded_masks = expanded_masks[:, : source.shape[1], :]
        data_values = source.clone()[..., :-1]
        data_values[expanded_masks == 0] = self.empty_value
        masked_source = torch.cat([data_values, source[..., -1:]], dim=-1)  # keep time feature unchanged

        data_dict["source"] = masked_source

        return data_dict


class RandomWalkCoverage(RandomGridCoverage):
    """Sample monitoring positions by performing simple random walks."""

    def __init__(
        self,
        input_window: int,
        step_size: int,
        num_positions: int = 1,
        empty_value: float = 0.0,
    ) -> None:
        super().__init__(input_window, step_size, num_positions, empty_value)

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
        walk_indices = torch.empty(batch_size, self.num_time_steps, self.num_positions, dtype=torch.long)

        # randomly initialize starting positions
        current_indices = torch.randint(
            low=0,
            high=self.total_positions,
            size=(batch_size, self.num_positions),
            dtype=torch.long,
        )
        walk_indices[:, 0, :] = current_indices

        # randomly move to a neighbor
        for step in range(1, self.num_time_steps):
            next_indices = torch.empty(batch_size, self.num_positions, dtype=torch.long)
            for batch_idx in range(batch_size):
                all_neighbors = []
                for pos_idx in range(self.num_positions):
                    current_idx = current_indices[batch_idx, pos_idx].item()
                    neighbors = self.neighbor_indices[current_idx]
                    all_neighbors.append(neighbors)

                # we need to avoid overlapping, so we gather all neighbors first and sample without replacement
                unique_neighbors = torch.cat(all_neighbors, dim=0).unique()
                chosen_neighbors = unique_neighbors[torch.randperm(unique_neighbors.shape[0])[: self.num_positions]]
                next_indices[batch_idx, :] = chosen_neighbors

            walk_indices[:, step, :] = next_indices
            current_indices = next_indices

        sampled_gids = self.available_gids[walk_indices]
        coverage_masks = sampled_gids.unsqueeze(-1).eq(self.grid_id_of_nodes).any(dim=2)

        return coverage_masks
