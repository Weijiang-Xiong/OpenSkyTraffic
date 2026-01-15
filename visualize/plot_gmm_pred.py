import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from einops import rearrange

from skytraffic.models.layers import GMMPredictionHead

sns.set_style("darkgrid")

PRED_PATH = Path("scratch/_save/lgc_gmm_simbarcaspd_test_preds.pkl")
OUTPUT_DIR = Path("figures/gmm_pred_sessions")
SAMPLE_PER_SESSION = 20
POSITION_TO_VISUALIZE = [113, 345, 1068]
VIS_POINTS = 300
DENSITY_SCALE = 5
DATA_TIME_STEP = 3  # minutes between consecutive samples
INPUT_WINDOW = 30   # minutes of history used for prediction
Y_MIN, Y_MAX = 0.0, 14.0


def main():
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

    pred_by_session = torch.tensor_split(avg_pred[:, -1], num_sessions)
    gt_by_session = torch.tensor_split(targets[:, -1], num_sessions)
    mixing_by_session = torch.tensor_split(mixing[:, -1], num_sessions)
    means_by_session = torch.tensor_split(means[:, -1], num_sessions)
    logvar_by_session = torch.tensor_split(log_var[:, -1], num_sessions)

    y_vals = torch.linspace(Y_MIN, Y_MAX, VIS_POINTS)
    xx = np.arange(SAMPLE_PER_SESSION)
    palette = sns.color_palette("husl", SAMPLE_PER_SESSION)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for position in POSITION_TO_VISUALIZE:
        for session_idx in range(num_sessions):
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

            ax.plot(xx, pred_by_session[session_idx][:, position], "o-", label="30min Pred")
            ax.plot(xx, gt_by_session[session_idx][:, position], "x-", label="Ground Truth")
            ax.set_xticks(xx, xx * DATA_TIME_STEP + INPUT_WINDOW)
            ax.set_xlabel("Time in Simulation")
            ax.set_ylabel("Speed (m/s)")
            ax.legend()
            ax.set_ylim(Y_MIN, Y_MAX)

            fig.tight_layout()
            save_path = OUTPUT_DIR / f"session{session_idx}_pos{position}.pdf"
            fig.savefig(save_path)
            plt.close(fig)


if __name__ == "__main__":
    main()
