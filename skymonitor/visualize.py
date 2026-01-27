import os
from typing import Dict, Sequence
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# Use seaborn darkgrid style
import seaborn as sns
sns.set_theme(style="darkgrid")

FIGURE_DIR = os.path.join("figures", "skymonitor")

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


    plt.figure(figsize=(12, 8))

    # Render grid cells with colors, leaving empty cells white.
    masked_map = np.ma.masked_where(grid_map == -1, grid_map)
    cmap = plt.cm.turbo.copy()
    cmap.set_bad(color="white")
    plt.imshow(
        masked_map,
        origin="lower",
        cmap=cmap,
        extent=(-0.5, grid_width - 0.5, -0.5, grid_height - 0.5),
        interpolation="none",
    )

    # Set up the plot
    plt.xlim(-0.5, grid_width - 0.5)
    plt.ylim(-0.5, grid_height - 0.5)
    plt.grid(False)

    # Add grid lines
    for x in range(grid_width):
        plt.axvline(x - 0.5, color="gray", linewidth=0.5)
    for y in range(grid_height):
        plt.axhline(y - 0.5, color="gray", linewidth=0.5)

    plt.title("Grid Data Visualization")
    plt.xlabel("Grid X")
    plt.ylabel("Grid Y")

    # Add text annotations for grid IDs
    for y in range(grid_height):
        for x in range(grid_width):
            if grid_map[y, x] != -1:
                plt.text(
                    x,
                    y,
                    str(grid_map[y, x]),
                    ha="center",
                    va="center",
                    fontsize=font_size,
                    color="black",
                )

    plt.tight_layout()
    os.makedirs(FIGURE_DIR, exist_ok=True)
    plt.savefig(os.path.join(FIGURE_DIR, "{}.pdf".format(note)), bbox_inches="tight")
    print(f"Saved grid data visualization to {os.path.join(FIGURE_DIR, '{}.pdf'.format(note))}")
    plt.close()


def visualize_map_scores(
    scores: np.ndarray,
    node_coordinates: np.ndarray,
    note: str = "note",
    info: Dict[str, Sequence] = None,
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
    coords = np.asarray(node_coordinates, dtype=np.float32)
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

        fig, ax, scatter = map_visualize(
            coords=coords,
            values=session_scores[0],
            figure_note=f"{title_prefix} score (t=0)",
            save_path=None,
            cmap="coolwarm",
            vmin=0.0,
            vmax=1.0,
            colorbar_label="Score",
        )

        def update(frame):
            scatter.set_array(session_scores[frame])
            ax.set_title(f"{title_prefix} score (t={frame})")
            return scatter,

        anim = animation.FuncAnimation(
            fig,
            update,
            frames=session_scores.shape[0],
            interval=300,
            blit=False,
        )

        save_path = os.path.join(output_dir, f"{session_info[idx]}.gif")
        anim.save(save_path, writer=animation.PillowWriter(fps=4))
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


def map_visualize(
    coords: np.ndarray,
    values: np.ndarray,
    figure_note: str,
    save_path: str = None,
    cmap: str = "coolwarm",
    vmin: float = None,
    vmax: float = None,
    colorbar_label: str = "Value",
):
    """
    Scatter map helper used for static plots and animation frames.

    Args:
        node_coordinate: Array of shape (num_locations, 2) with XY coordinates.
        values: Array of shape (num_locations,) containing values to visualize.
        figure_note: Title text for the figure.
        save_path: Optional file path to save the figure.
        cmap: Matplotlib colormap name.
        vmin: Optional minimum for color scaling.
        vmax: Optional maximum for color scaling.
        colorbar_label: Label for the colorbar.

    Returns:
        Tuple of (figure, axis, scatter) for reuse (e.g., animation updates).
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="darkgrid")

    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=values, cmap=cmap, vmin=vmin, vmax=vmax, s=8)
    fig.colorbar(scatter, ax=ax, label=colorbar_label)
    ax.set_title(figure_note)
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.set_aspect("equal")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200)

    return fig, ax, scatter


def plot_coverage_masks(coverage_mask):
    """ 
    Visualize monitoring coverage masks for multiple data samples.

    Example usage:
    ```
    with open("figures/skymonitor/data.pkl", "rb") as f:
        data = pickle.load(f)

    source = data["source"]  # shape: (N, T, P, C)
    coverage_mask = data["coverage_mask"]  # shape: (N, T, P)
    N, T, P, C = source.shape
    plot_coverage_masks(coverage_mask)
    ```
    """
    # coverage_mask shape: (N, T, P)
    N, T, P = coverage_mask.shape
    mask_per_fig = 4
    num_figs = np.ceil(N / mask_per_fig).astype(int)

    for fig_idx in range(num_figs):
        start = fig_idx * mask_per_fig
        end = min(start + mask_per_fig, N)
        batch_slice = coverage_mask[start:end]

        cols = mask_per_fig
        fig, axes = plt.subplots(1, cols, figsize=(2 * cols, 6))
        axes = axes.ravel()

        for i, mask in enumerate(batch_slice):
            ax = axes[i]
            ax.imshow(mask.T, origin="lower", cmap="viridis", aspect="auto")
            # ax.set_title(f"Monitoring Mask {start + i}")
            ax.set_xlabel("Input Time Steps")
            if i == 0:
                ax.set_ylabel("Road Segment Number")
            else:
                ax.set_ylabel("")
                ax.set_yticks([])

        # Hide any unused subplots
        for j in range(len(batch_slice), len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        fig.savefig(f"{FIGURE_DIR}/coverage_mask_{fig_idx}.pdf", dpi=300)
        plt.close(fig)
