""" This is a file for some visualizations that are not really integrated
"""
import os
import glob

import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")


error_keys = ["15min_mae_segment", "15min_mape_segment", "15min_rmse_segment", "30min_mae_segment", "30min_mape_segment", "30min_rmse_segment", "15min_mae_region", "15min_mape_region", "15min_rmse_region", "30min_mae_region", "30min_mape_region", "30min_rmse_region"]

default_output_dir = "scratch/himsnet_3hop"

def get_error_metrics_from_log(log_dir):
    log_file = "{}/experiment.log".format(log_dir)
    with open(log_file, "r") as f:
        all_lines = f.readlines()
        for row_idx, line in enumerate(all_lines):
            if "Results to copy in excel:" in line:
                error_metric_string = all_lines[row_idx+1]
    
    # an example for error_metric_string: 
    # ' example 1.03 23.8% 1.73 1.21 27.1% 2.01 0.20 6.1% 0.28 0.30 10.2% 0.47 \n'
    # the numbers are (MAE, MAPE, RMSE) for 15 min, 30 min prediction intervals. 
    # the first half is for segment leve, the second half is for regional level
    error_metrics = error_metric_string.strip().split(" ")[1:]
    error_metrics_num = [float(e[:-1]) if e.endswith("%") else float(e) for e in error_metrics]
    
    return {k: v for k, v in zip(error_keys, error_metrics_num)}

different_hops = ["scratch/himsnet_{}hop".format(i) for i in (1, 3, 5)]
mae_segment_30min, mae_regional_30min = [], []
for f in different_hops:
    error_metrics = get_error_metrics_from_log(f)
    mae_segment_30min.append(error_metrics["30min_mae_segment"])
    mae_regional_30min.append(error_metrics["30min_mae_region"])

# draw a bar plot, put segment-level and regional-level MAE to different bar groups
fig, ax = plt.subplots()
bar_width = 0.25
index = np.arange(len(different_hops))
bar1 = ax.bar(index, mae_segment_30min, bar_width, label="Segment-level")
ax.bar_label(bar1, padding=3)
bar2 = ax.bar(index + bar_width, mae_regional_30min, bar_width, label="Regional")
ax.bar_label(bar2, padding=3)
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(["{} hops".format(i) for i in (1, 3, 5)])
ax.set_ylim(0.2, 1.4)
ax.legend(ncols=2)
ax.set_ylabel("MAE")
ax.set_title("30-min MAE of different adjacency hops")
plt.savefig("datasets/simbarca/figures/mae_30min.pdf")



different_epochs = [default_output_dir] + ["scratch/himsnet_ep{}_3hop".format(e) for e in [60, 90, 120, 150]]

mae_segment_30min, mae_regional_30min = [], []
for f in different_epochs:
    error_metrics = get_error_metrics_from_log(f)
    mae_segment_30min.append(error_metrics["30min_mae_segment"])
    mae_regional_30min.append(error_metrics["30min_mae_region"])

# draw a bar plot, put segment-level and regional-level MAE to different bar groups
fig, ax = plt.subplots()
bar_width = 0.25
index = np.arange(len(different_epochs))
bar1 = ax.bar(index, mae_segment_30min, bar_width, label="Segment-level")
ax.bar_label(bar1, padding=3)
bar2 = ax.bar(index + bar_width, mae_regional_30min, bar_width, label="Regional")
ax.bar_label(bar2, padding=3)
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(["{} epochs".format(e) for e in [30, 60, 90, 120, 150]])
ax.set_ylim(0.2, 1.4)
ax.legend(ncols=2)
ax.set_ylabel("MAE")
ax.set_title("30-min MAE of different training epochs")
plt.savefig("datasets/simbarca/figures/mae_30min_epochs.pdf")

all_coverages = np.arange(0.1, 1.0, 0.1).tolist()
different_coverage = ["scratch/himsnet_rnd_noise_3hop_{}cvg".format(round(c*100)) for c in all_coverages] + [default_output_dir]
mae_segment_30min, mae_regional_30min = [], []
for f in different_coverage:
    error_metrics = get_error_metrics_from_log(f)
    mae_segment_30min.append(error_metrics["30min_mae_segment"])
    mae_regional_30min.append(error_metrics["30min_mae_region"])

all_coverages = all_coverages + [1.0]

# draw a line plot for different coverage
fig, ax = plt.subplots()
ax.plot(all_coverages, mae_segment_30min, label="Segment-level")
ax.plot(all_coverages, mae_regional_30min, label="Regional-level")
ax.set_xlabel("Coverage")
ax.set_ylabel("MAE")
ax.set_title("30-min MAE of different coverage")
ax.legend()
plt.savefig("datasets/simbarca/figures/mae_30min_coverage.pdf")
