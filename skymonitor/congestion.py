import os
import argparse

from typing import Dict, Sequence

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from einops import rearrange

from skymonitor.simbarca_explore import initialize_dataset

FIGURE_DIR = os.path.join("figures", "skymonitor", "congestion")
SPEED_LIMIT = 50/3.6 # 50 km/h given in unit m/s because of SI units used in flow and density

def get_congestion_score(
    flow: np.ndarray,
    density: np.ndarray,
    density_threshold: float = 1e-3,
    ff_quantile: float = 1.0,
) -> np.ndarray:

    assert flow.shape == density.shape, "Flow and density arrays must have the same shape."

    S, T, P = flow.shape
    flow, density = rearrange(flow, 'S T P -> P (S T)'), rearrange(density, 'S T P -> P (S T)')

    # we put the nan here temporarily for computing free-flow speed from valid observations only
    speed = np.where(density > density_threshold, flow / density, np.nan)
    
    # free flow speed of each road segment (ff_quantile observed speed)
    # 1.0 means maximum speed 
    ff_speed = np.nanquantile(speed, ff_quantile, axis=1)

    # empty road segments have free-flow speed equal to speed limit
    is_empty_segments = np.isnan(speed).all(axis=1)
    ff_speed[is_empty_segments] = SPEED_LIMIT

    speed = np.nan_to_num(speed, nan=SPEED_LIMIT) # we have nans in speed because the density is zero
    congestion_score = np.clip(1 - (speed / ff_speed[:, None]), 0.0, 1.0)

    congestion_score = rearrange(congestion_score, 'P (S T) -> S T P', S=S)

    return congestion_score


def get_congestion_change(congestion_score: np.ndarray, entry_thd: float, exit_thd: float) -> np.ndarray:
    """Compute congestion state using a 2-step lookahead and hysteresis thresholds.

    Args:
        congestion_score: Array with shape (S, T, P) representing congestion scores.
        entry_thd: Congestion entry threshold (between 0 and 1).
        exit_thd: Congestion exit threshold (between 0 and 1).
    """
    assert congestion_score.ndim == 3, "Expected congestion_score with shape (S, T, P)."
    S, T, P = congestion_score.shape
    state, enters, exits = [np.zeros((S, T, P), dtype=bool) for _ in range(3)]
    current = np.zeros((S, P), dtype=bool)

    for t in range(T):

        # the last two time steps will be kept the same as previous time step
        if t + 2 < T:
            # a segment ENTERS congestion if it is not currently congested and then becomes congested for the next two time steps
            enter_congestion = (
                (~current)
                & (congestion_score[:, t + 1, :] > entry_thd)
                & (congestion_score[:, t + 2, :] > entry_thd)
            )
            # a segment EXITS congestion if it is currently congested and then becomes uncongested for the next two time steps
            # this criterion is stricter than the entering because we want to avoid rapid toggling
            exit_congestion = (
                current
                & (congestion_score[:, t + 1, :] < exit_thd)
                & (congestion_score[:, t + 2, :] < exit_thd)
            )
            # update the congestion state. 
            # a segment is congested if it was previously uncongested and enters congestion (first part), 
            # if it exits congestion it becomes uncongested (second part)
            current = (current | enter_congestion) & ~exit_congestion
            enters[:, t, :] = enter_congestion
            exits[:, t, :] = exit_congestion

        state[:, t, :] = current

    return state, enters, exits

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

    for idx in range(0, num_sessions, 15):
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

if __name__ == "__main__":


    parser = argparse.ArgumentParser(description='Visualize congestion scores and state.')
    parser.add_argument('--figure-dir', type=str, default=FIGURE_DIR, help='Directory to save the figures.')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'test'], help='Dataset split to visualize.')
    parser.add_argument('--density-threshold', type=float, default=1e-3, help='Minimum density to compute speed; lower values treated as empty.')
    parser.add_argument('--ff-quantile', type=float, default=0.9, help='Quantile used to estimate free-flow speed per segment. 1.0 means maximum observed speed.')
    parser.add_argument('--entry-thd', type=float, default=0.5, help='Congestion entry threshold. Value above is considered congested.')
    parser.add_argument('--exit-thd', type=float, default=0.6, help='Congestion exit threshold. Value below is not congested.')
    args = parser.parse_args()

    trainset, test_set = initialize_dataset()
    vis_set = trainset if args.split == 'train' else test_set

    flow, density = vis_set.veh_flow_3min, vis_set.veh_density_3min
    congestion_score = get_congestion_score(flow, density, density_threshold=args.density_threshold, ff_quantile=args.ff_quantile)
    visualize_map_scores(
        congestion_score,
        node_coordinates=vis_set.node_coordinates,
        info={
            'session_ids': vis_set.session_ids,
            'demand_scales': vis_set.demand_scales,
        },
        note='congestion_score_{}_ffq{}_en{}_ex{}'.format(args.split, args.ff_quantile, args.entry_thd, args.exit_thd),
    )

    congestion_state = get_congestion_change(congestion_score, entry_thd=args.entry_thd, exit_thd=args.exit_thd)
    visualize_map_scores(
        congestion_state.astype(float),
        node_coordinates=vis_set.node_coordinates,
        info={
            'session_ids': vis_set.session_ids,
            'demand_scales': vis_set.demand_scales,
        },
        note='congestion_state_{}_ffq{}_en{}_ex{}'.format(args.split, args.ff_quantile, args.entry_thd, args.exit_thd),
    )
