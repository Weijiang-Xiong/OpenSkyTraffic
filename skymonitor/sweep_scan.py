import os
from typing import Optional, Sequence, Tuple

import numpy as np
from matplotlib import pyplot as plt

from skymonitor.simbarca_explore import initialize_dataset
from skymonitor.monitor_env import TrafficMonitorEnv, build_traffic_monitor_env
from skymonitor.agents import SweepingAgent


def visualize_sweep_scan_trajectories(
    grid_xy: np.ndarray,
    trajectories: Sequence[Sequence[Tuple[int, int]]],
    save_path: Optional[str] = None,
    title: str = "Sweeping-Scan Trajectories",
):
    """Plot sweep-scan trajectories on the grid."""
    positions = {(int(x), int(y)) for x, y in np.asarray(grid_xy)}
    if not positions:
        return None, None

    xs = [pos[0] for pos in positions]
    ys = [pos[1] for pos in positions]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(min_x - 0.5, max_x + 0.5)
    ax.set_ylim(min_y - 0.5, max_y + 0.5)
    for x in range(min_x, max_x + 2):
        ax.axvline(x - 0.5, color="gray", linewidth=0.5, alpha=0.3)
    for y in range(min_y, max_y + 2):
        ax.axhline(y - 0.5, color="gray", linewidth=0.5, alpha=0.3)

    grid_points = np.asarray(list(positions))
    ax.scatter(grid_points[:, 0], grid_points[:, 1], s=18, color="lightgray", alpha=0.6, marker="s")

    cmap = plt.get_cmap("tab10", max(1, len(trajectories)))
    for idx, path in enumerate(trajectories):
        if not path:
            continue
        coords = np.asarray(path)
        color = cmap(idx)
        ax.plot(coords[:, 0], coords[:, 1], color=color, linewidth=2, alpha=0.9)
        ax.scatter(coords[0, 0], coords[0, 1], color=color, s=60, marker="o")
        ax.scatter(coords[-1, 0], coords[-1, 1], color=color, s=70, marker="X")

    ax.set_title(title)
    ax.set_xlabel("Grid X")
    ax.set_ylabel("Grid Y")
    ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()
    if save_path is not None:
        dir_name = os.path.dirname(save_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig, ax

if __name__ == "__main__":

    trainset, _ = initialize_dataset()
    # Use available grid cells for planning/visualization
    env: TrafficMonitorEnv = build_traffic_monitor_env(trainset, num_drones=10, env_type='train')

    trajectories = SweepingAgent.build_sweep_scan_plan(env.all_positions, num_drones=10)
    print("Generated sweep-scan trajectories for 10 drones.")
    print("Max length of trajectories:", max(len(t) for t in trajectories))

    # Static trajectory plot
    visualize_sweep_scan_trajectories(
        env.all_positions,
        trajectories,
        save_path="figures/skymonitor/sweep_scan.pdf",
    )
