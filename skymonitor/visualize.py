import os
import logging
from collections import defaultdict
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import animation

from skytraffic.utils.structure import norm_coords_and_grid_xy

# Use seaborn darkgrid style
import seaborn as sns
sns.set_theme(style="darkgrid")

FIGURE_DIR = os.path.join("figures", "skymonitor")
logger = logging.getLogger(__name__)

def init_canvas(
    grid_xy,
    *,
    figsize: Tuple[int, int] = (12, 8),
    title="Grid Visualization",
    line_alpha=0.4,
    draw_grid_lines=True,
    mark_grids=True,
):
    
    fig, ax = plt.subplots(figsize=figsize)

    grid_xy = np.asarray(grid_xy)
    x_min = int(grid_xy[:, 0].min())
    x_max = int(grid_xy[:, 0].max())
    y_min = int(grid_xy[:, 1].min())
    y_max = int(grid_xy[:, 1].max())

    ax.set_xlim(x_min - 0.5, x_max + 0.5)
    ax.set_ylim(y_min - 0.5, y_max + 0.5)
    ax.grid(False)

    if draw_grid_lines:
        for x in range(x_min, x_max + 2):
            ax.axvline(x - 0.5, color="gray", linewidth=0.5, alpha=line_alpha)
        for y in range(y_min, y_max + 2):
            ax.axhline(y - 0.5, color="gray", linewidth=0.5, alpha=line_alpha)

    if mark_grids:
        ax.scatter(
            grid_xy[:, 0],
            grid_xy[:, 1],
            marker="s",
            s=20,
            color="lightgray",
            linewidths=1,
            alpha=0.6,
        )

    ax.set_title(title)
    ax.set_xlabel("Grid X")
    ax.set_ylabel("Grid Y")
    ax.set_aspect("equal", adjustable="box")

    fig.tight_layout()

    return fig, ax

def plot_flow_density(flow_array: np.ndarray, density_array: np.ndarray, flow_weight:np.ndarray, note="example"):
    """
    Plot flow and density distributions plus their regional average relationship.

    Args:
        flow_array: Flow values shaped like the 3-min flow tensor, shape (batch, time, location).
        density_array: Density values shaped like the 3-min density tensor, shape (batch, time, location).
    """
    # Draw a histogram for the 3 min flow and density values separately,
    # and a scatter plot for flow vs density
    flow_values = flow_array.flatten()
    density_values = density_array.flatten()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].hist(flow_values, bins=50, color='blue', alpha=0.7)
    axes[0].set_title('3-min Flow Distribution')
    axes[0].set_xlabel('Flow (veh/s)')
    axes[0].set_ylabel('Data point count')

    axes[1].hist(density_values, bins=50, color='green', alpha=0.7)
    axes[1].set_title('3-min Density Distribution')
    axes[1].set_xlabel('Density (veh/m)')
    axes[1].set_ylabel('Data point count')

    axes[2].scatter(np.average(density_array, weights=flow_weight, axis=-1).flatten(),
                    np.average(flow_array, weights=flow_weight, axis=-1).flatten(),
                    alpha=0.5) 
    axes[2].set_title('Regional Avg Flow vs Density')
    axes[2].set_xlabel('Density (veh/m)')
    axes[2].set_ylabel('Flow (veh/s)')

    fig.tight_layout()
    os.makedirs(FIGURE_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIGURE_DIR, "{}.pdf".format(note)), bbox_inches="tight")
    print(f"Saved flow-density plots to {os.path.join(FIGURE_DIR, '{}.pdf'.format(note))}")
    plt.close(fig)


def visualize_data_as_grid(grid_xy, node_data, agg="max", font_size=10, note="data"):
    """
    Visualize data as a grid map

    Args:
        grid_xy: Array of (x, y) grid coordinates for each item. shape (num_nodes, 2).
        node_data: Sequence of node data values aligned with grid_xy. shape (num_nodes,).
    """
    fig, ax = init_canvas(
        grid_xy=grid_xy,
        line_alpha=1.0,
        mark_grids=False,
        title="Grid Data Visualization",
    )

    # Visualization
    grid_height = int(grid_xy[:, 1].max() + 1)
    grid_width = int(grid_xy[:, 0].max() + 1)

    # put the data to grid cells according
    grid_data_container = defaultdict(list)
    for i, (x, y) in enumerate(grid_xy):
        grid_data_container[(int(y), int(x))].append(node_data[i])
    grid_map = np.full((grid_height, grid_width), -1, dtype=int)
    for (y, x), data_list in grid_data_container.items():
        if agg == "sum":
            grid_map[y, x] = sum(data_list)
        elif agg == "mean":
            grid_map[y, x] = np.mean(data_list)
        elif agg == "max":
            grid_map[y, x] = max(data_list)
        else:
            raise ValueError(f"Unsupported aggregation method: {agg}")

    # Render grid cells with colors, leaving empty cells white.
    grid_map = grid_map.astype(float)
    grid_map[grid_map == -1] = np.nan
    cmap = cm.turbo.copy()
    cmap.set_bad(color="white")
    ax.imshow(
        grid_map, # nans will be ignored
        origin="lower",
        cmap=cmap,
        extent=(-0.5, grid_width - 0.5, -0.5, grid_height - 0.5),
        interpolation="none",
    )

    # Add text annotations for grid IDs
    for y in range(grid_height):
        for x in range(grid_width):
            if grid_map[y, x] != -1:
                ax.text(
                    x,
                    y,
                    str(grid_map[y, x]),
                    ha="center",
                    va="center",
                    fontsize=font_size,
                    color="black",
                )

    os.makedirs(FIGURE_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIGURE_DIR, "{}.pdf".format(note)), bbox_inches="tight")
    print(f"Saved grid data visualization to {os.path.join(FIGURE_DIR, '{}.pdf'.format(note))}")
    plt.close(fig)


def animate_trajectory(
    grid_xy: np.ndarray,
    trajectory: Sequence,
    save_path: Optional[str] = None,
    title: str = "Drone Coverage Summary",
    fps: int = 2,
):
    """Visualize drone paths with matplotlib in terminal mode.
    If save_path is provided, saves a GIF to that location.
    """
    positions_frames = [np.asarray(frame) for frame in trajectory]
    steps = len(positions_frames)
    num_agents = positions_frames[0].shape[0]

    fig, ax = init_canvas(
        grid_xy=grid_xy,
        line_alpha=0.4,
        title=title,
    )
    scatter = ax.scatter(
        [],
        [],
        c=[],
        s=80,
        cmap=plt.get_cmap("tab10"),
        vmin=0,
        vmax=max(1, num_agents) - 1,
        label="Drones",
    )
    trails = [ax.plot([], [], linestyle="-", marker="X", alpha=0.5)[0] for _ in range(num_agents)]
    annotation = ax.text(
        0.02,
        0.95,
        "",
        transform=ax.transAxes,
        fontsize=10,
        ha="left",
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.7},
    )

    def init():
        annotation.set_text("")
        for line in trails:
            line.set_data([], [])
        return [scatter, annotation, *trails]

    def update(frame_idx):
        scatter.set_offsets(positions_frames[frame_idx])
        scatter.set_array(np.arange(num_agents))
        annotation.set_text(f"Step {frame_idx} / {steps - 1}")
        for drone_idx, line in enumerate(trails):
            history = np.asarray([frame[drone_idx] for frame in positions_frames[: frame_idx + 1]])
            line.set_data(history[:, 0], history[:, 1])
        return [scatter, annotation, *trails]

    animation_obj = animation.FuncAnimation(
        fig,
        update,
        frames=steps,
        init_func=init,
        blit=False,
        repeat=False,
    )
    if save_path is not None:
        animation_obj.save(save_path, writer="pillow", fps=fps)
        logger.info('Saved trajectory visualization to {}'.format(save_path))

    return animation_obj


def visualize_sweep_scan_trajectories(
    grid_xy: np.ndarray,
    trajectories: Sequence[Sequence[Tuple[int, int]]],
    save_path: Optional[str] = None,
):
    """Plot sweep-scan trajectories on the grid."""
    fig, ax = init_canvas(
        grid_xy=grid_xy,
        line_alpha=0.3,
        title="Sweeping-Scan Trajectories",
    )

    cmap = plt.get_cmap("tab10", max(1, len(trajectories)))
    for idx, path in enumerate(trajectories):
        if not path:
            continue
        coords = np.asarray(path)
        color = cmap(idx)
        ax.plot(coords[:, 0], coords[:, 1], color=color, linewidth=2, alpha=0.9)
        ax.scatter(coords[0, 0], coords[0, 1], color=color, s=60, marker="o")
        ax.scatter(coords[-1, 0], coords[-1, 1], color=color, s=70, marker="X")

    if save_path is not None:
        dir_name = os.path.dirname(save_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        logger.info('Saved sweep-scan trajectory visualization to {}'.format(save_path))

    return fig, ax


def visualize_map_scores(
    scores: np.ndarray,
    node_coordinates: np.ndarray,
    grid_size: float = 220.0,
    note: str = "note",
    info: Dict[str, Sequence] = None,
    draw_grid: bool = True,
):

    """
    Visualize and animate congestion scores for each simulation session.

    scores: array with shape (num_sessions, num_timesteps, num_locations).
    node_coordinates: array with shape (num_locations, 2).
    session_ids: sequence of session identifiers with length num_sessions.


    For every session, this function creates a scatter plot of all road segments where
    colors encode the congestion score at each time step, then saves the
    animation under `./figures/skymonitor/<note>/`.

    Args:
        congestion_scores: Array with shape (num_sessions, num_timesteps, num_locations).
        node_coordinates: Array with shape (num_locations, 2).
        info: a dictionary containing additional information for each session, for example:
            {
                "session_ids": [...],  # sequence of session identifiers
                "demand_scales": [...],  # sequence of demand scale factors
            }
        note: Label used for the output directory name.
    """


    output_dir = os.path.join(FIGURE_DIR, f"{note}")
    os.makedirs(output_dir, exist_ok=True)

    scores = np.asarray(scores, dtype=np.float32)
    coords, grid_xy = norm_coords_and_grid_xy(node_coordinates, grid_size)

    if info is not None:
        session_info = ["_".join("{}={}".format(k, v[i]) for k, v in info.items())
                        for i in range(scores.shape[0])]
    else:
        session_info = ["Session {}".format(i+1) for i in range(scores.shape[0])]

    num_sessions = scores.shape[0]

    for idx in range(0, num_sessions, 8):
        print(f"Visualizing session ({idx+1}/{num_sessions})...")
        session_scores = scores[idx]
        title_prefix = session_info[idx]

        fig, ax = init_canvas(
            grid_xy=grid_xy,
            title=f"{title_prefix} (t=0)",
            draw_grid_lines=draw_grid,
            mark_grids=False,
        )
        scatter = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=session_scores[0],
            cmap="coolwarm",
            vmin=0.0,
            vmax=1.0,
            s=8,
        )

        def update(frame):
            scatter.set_array(session_scores[frame])
            ax.set_title(f"{title_prefix} (t={frame})")
            return scatter,

        anim = animation.FuncAnimation(
            fig,
            update,
            frames=session_scores.shape[0],
            blit=False,
            repeat=False,
        )

        save_path = os.path.join(output_dir, f"{session_info[idx]}.gif")
        anim.save(save_path, writer=animation.PillowWriter(fps=2))
        plt.close(fig)

def scatter_plot(
	x: np.ndarray,
	y: np.ndarray,
	title: str = 'Scatter Plot', 
	save_path: str = "./example.pdf",
	xlabel: str = 'X',
	ylabel: str = 'Y',
):
	fig, ax = plt.subplots(figsize=(6, 4))
	scatter = ax.scatter(x, y, s=10)
	ax.set_title(title)
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	fig.tight_layout()
	if save_path:
		fig.savefig("{}".format(save_path))
	return fig, ax, scatter


def historgram_plot(
    data: np.ndarray,
    title: str = 'Histogram', 
    save_path: str = "./example.pdf",
    xlabel: str = 'Value',
    ylabel: str = 'Count',
    bins: int = 30,
):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(data, bins=bins, color='blue', alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    if save_path:
        fig.savefig("{}".format(save_path))
    return fig, ax


if __name__ == "__main__":
    from skymonitor.agents import SweepingAgent
    from skymonitor.monitor_env import TrafficMonitorEnv, build_traffic_monitor_env
    from skymonitor.simbarca_explore import initialize_dataset
    from skytraffic.utils.event_logger import setup_logger

    logger = setup_logger(name='skymonitor', level=logging.INFO)

    trainset, _ = initialize_dataset()
    env: TrafficMonitorEnv = build_traffic_monitor_env(trainset, num_drones=10, env_type="train")

    trajectories = SweepingAgent.build_sweep_scan_plan(env.all_positions, num_drones=10)
    logger.info("Generated sweep-scan trajectories for 10 drones.")
    logger.info("Max length of trajectories: {}".format(max(len(t) for t in trajectories)))

    visualize_sweep_scan_trajectories(
        env.all_positions,
        trajectories,
        save_path="{}/sweep_scan.pdf".format(FIGURE_DIR),
    )
