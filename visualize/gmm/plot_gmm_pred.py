import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from einops import rearrange

from skytraffic.models.layers import GMMPredictionHead

sns.set_style("darkgrid")

PRED_PATH = Path("mtgnn_gmm_simbarcaspd_outputs.pkl")
OUTPUT_DIR = Path("figures/gmm/mtgnn")
SAMPLE_PER_SESSION = 20
POSITION_TO_VISUALIZE = [113, 345, 1068, 806]
SESSION_TO_VISUALIZE = range(20, 26) # index of sessions to visualize, from 0 to num_sessions - 1
VIS_POINTS = 300
DENSITY_SCALE = 2.0  # scale factor for better visualization
DATA_TIME_STEP = 3  # minutes between consecutive samples
INPUT_WINDOW = 30   # minutes of history used for prediction
OUTPUT_WINDOW = 30
Y_MIN, Y_MAX = 0.0, 14.0

def plot_k_steps_ahead(plot_step: int):
    with PRED_PATH.open("rb") as f:
        data = pickle.load(f)

    preds = data["preds"]
    targets = data["labels"]["target"]

    avg_pred = torch.as_tensor(preds["pred"]).cpu()
    mixing = torch.as_tensor(preds["mixing"]).cpu()
    means = torch.as_tensor(preds["means"]).cpu()
    log_var = torch.as_tensor(preds["log_var"]).cpu()
    targets = torch.as_tensor(targets).cpu()

    # Drop trailing output dimension if present
    if avg_pred.ndim == 4:
        avg_pred = avg_pred[..., 0]
    if targets.ndim == 4:
        targets = targets[..., 0]

    assert avg_pred.shape[0] % SAMPLE_PER_SESSION == 0, "Batch size must be divisible by SAMPLE_PER_SESSION"
    num_sessions = avg_pred.shape[0] // SAMPLE_PER_SESSION

    pred_by_session = torch.tensor_split(avg_pred[:, plot_step - 1], num_sessions)
    gt_by_session = torch.tensor_split(targets[:, plot_step - 1], num_sessions)
    mixing_by_session = torch.tensor_split(mixing[:, plot_step - 1], num_sessions)
    means_by_session = torch.tensor_split(means[:, plot_step - 1], num_sessions)
    logvar_by_session = torch.tensor_split(log_var[:, plot_step - 1], num_sessions)

    y_vals = torch.linspace(Y_MIN, Y_MAX, VIS_POINTS)
    xx = np.arange(SAMPLE_PER_SESSION)
    palette = sns.color_palette("husl", SAMPLE_PER_SESSION)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for position in POSITION_TO_VISUALIZE:
        for session_idx in SESSION_TO_VISUALIZE:
            pdf_matrix = GMMPredictionHead.get_mixture_density(
                rearrange(mixing_by_session[session_idx][:, position], "T K -> () T () K"),
                rearrange(means_by_session[session_idx][:, position], "T K -> () T () K"),
                rearrange(logvar_by_session[session_idx][:, position], "T K -> () T () K"),
                y_vals,
            ).squeeze().numpy()

            fig, ax = plt.subplots(figsize=(6, 4))
            for t in range(SAMPLE_PER_SESSION):
                x_baseline = t
                ridge_x = t + DENSITY_SCALE * pdf_matrix[t, :]
                ax.fill_betweenx(y_vals, x_baseline, ridge_x, color=palette[t], alpha=0.6)

            ax.plot(xx, pred_by_session[session_idx][:, position], "o-", label=f"{plot_step} Step Pred")
            ax.plot(xx, gt_by_session[session_idx][:, position], "x-", label="Ground Truth")
            ax.set_xticks(xx, (plot_step + xx) * DATA_TIME_STEP + INPUT_WINDOW)
            ax.set_xlabel("Time from Simulation Start (min)")
            ax.set_ylabel("Speed (m/s)")
            ax.tick_params(axis="x", labelrotation=90)
            ax.legend()
            ax.set_ylim(Y_MIN, Y_MAX)
            ax.set_xlim(-0.5, SAMPLE_PER_SESSION + 1.5)

            fig.tight_layout()
            save_path = OUTPUT_DIR / f"session{session_idx}_pos{position}_step{plot_step}.pdf"
            print("Saving figure to", save_path)
            fig.savefig(save_path)
            plt.close(fig)


if __name__ == "__main__":
    # the model predicts OUTPUT_WINDOW // DATA_TIME_STEP steps into the future
    # we plot the prediction from this many steps 
    plot_step = 10
    assert (
        isinstance(plot_step, int)
        and plot_step >= 1
        and plot_step <= (OUTPUT_WINDOW // DATA_TIME_STEP)
    ), "Invalid prediction horizon to plot"
    
    for plot_step in range(1, 11):
        plot_k_steps_ahead(plot_step)
