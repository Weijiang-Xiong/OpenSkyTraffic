import torch
import matplotlib.pyplot as plt
import seaborn as sns
from skytraffic.data.datasets import SimBarca

sns.set_theme(style="darkgrid")

if __name__ == "__main__":
    simbarca_dataset = SimBarca(split="test", force_reload=False)
    # [..., 0] takes the value not the normalized time step
    pred_speed = simbarca_dataset.pred_speed[..., 0].flatten() 
    pred_speed = pred_speed[torch.logical_not(torch.isnan(pred_speed))]
    regional = simbarca_dataset.pred_speed_regional[..., 0].flatten()
    regional = regional[torch.logical_not(torch.isnan(regional))]

    # plot a histogram for pred_speed
    fig, ax = plt.subplots(figsize=(6, 4))
    # ignore outliers
    ax.hist(pred_speed, bins=50, alpha=0.8, density=True)
    ax.set_xlabel("Speed (m/s)")
    ax.set_ylabel("Frequency")
    ax.set_title("Histogram of Segment-level Speed")
    fig.tight_layout()
    fig.savefig("visualize/figures/pred_speed_histogram.pdf")
    
    # plot a histogram for regional
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(regional, bins=50, alpha=0.8, density=True)
    ax.set_xlabel("Speed (m/s)")
    ax.set_ylabel("Frequency")
    ax.set_title("Histogram of Regional Speed")
    fig.tight_layout()
    fig.savefig("visualize/figures/regional_speed_histogram.pdf")