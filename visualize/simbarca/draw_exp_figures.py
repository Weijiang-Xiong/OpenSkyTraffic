""" This is a file for some visualizations that are not really integrated
"""
import os
import glob

import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")


error_keys = ["15min_mae_segment", "15min_mape_segment", "15min_rmse_segment", "30min_mae_segment", "30min_mape_segment", "30min_rmse_segment", "15min_mae_region", "15min_mape_region", "15min_rmse_region", "30min_mae_region", "30min_mape_region", "30min_rmse_region"]

default_output_dir = "scratch/himsnet_5hop"

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

def draw_hops():
    adj_hops = (1, 3, 5, 7, 9)
    different_hops = ["scratch/himsnet_{}hop".format(i) for i in adj_hops]
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
    ax.set_xticklabels(["{} hops".format(i) for i in adj_hops])
    ax.set_ylim(0.2, 1.4)
    ax.legend(ncols=2)
    ax.set_xlabel("Number of Adjacency Hops")
    ax.set_ylabel("MAE")
    # ax.set_title("30-min MAE of different adjacency hops")
    plt.tight_layout()
    plt.savefig("datasets/simbarca/figures/mae_30min_adj_hops.pdf")
    print("figure saved to datasets/simbarca/figures/mae_30min_adj_hops.pdf")

def draw_hidden_dimension():
    hidden_dims = [32, 128, 256]
    folders_for_hidden_dims = ["scratch/himsnet_d{}_5hop".format(i) for i in hidden_dims]
    hidden_dims.insert(1, 64)
    folders_for_hidden_dims.insert(1, default_output_dir)
    
    mae_segment_30min, mae_regional_30min = [], []
    for f in folders_for_hidden_dims:
        error_metrics = get_error_metrics_from_log(f)
        mae_segment_30min.append(error_metrics["30min_mae_segment"])
        mae_regional_30min.append(error_metrics["30min_mae_region"])
    
    # draw a bar plot, put segment-level and regional-level MAE to different bar groups
    fig, ax = plt.subplots()
    bar_width = 0.25
    index = np.arange(len(folders_for_hidden_dims))
    bar1 = ax.bar(index, mae_segment_30min, bar_width, label="Segment-level")
    ax.bar_label(bar1, padding=3)
    bar2 = ax.bar(index + bar_width, mae_regional_30min, bar_width, label="Regional")
    ax.bar_label(bar2, padding=3)
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(["{}".format(i) for i in hidden_dims])
    ax.set_xlabel("Hidden Dimension Size")
    ax.set_ylim(0.2, 1.4)
    ax.legend(ncols=2)
    ax.set_ylabel("MAE")
    # ax.set_title("30-min MAE of different hidden dimensions")
    plt.tight_layout()
    plt.savefig("datasets/simbarca/figures/mae_30min_hidden_dim.pdf")
    

def draw_epochs():
    different_epochs = [default_output_dir] + ["scratch/himsnet_ep{}_5hop".format(e) for e in [60, 90, 120, 150]]

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
    ax.set_xticklabels(["{}".format(e) for e in [30, 60, 90, 120, 150]])
    ax.set_xlabel("Training Epochs")
    ax.set_ylim(0.2, 1.4)
    ax.legend(ncols=2)
    ax.set_ylabel("MAE")
    # ax.set_title("30-min MAE of different training epochs")
    plt.tight_layout()
    plt.savefig("datasets/simbarca/figures/mae_30min_epochs.pdf")

def draw_coverage(include_no_emb=False):
    
    all_coverages = [0.01, 0.03, 0.05, 0.07] + np.arange(0.1, 1.0, 0.1).tolist()
    different_coverage = ["scratch/himsnet_rnd_noise_5hop_{}cvg".format(round(c*100)) for c in all_coverages]
    mae_segment_30min, mae_regional_30min = [], []
    for f in different_coverage:
        error_metrics = get_error_metrics_from_log(f)
        mae_segment_30min.append(error_metrics["30min_mae_segment"])
        mae_regional_30min.append(error_metrics["30min_mae_region"])

    if include_no_emb:
        coverage_no_emb = ["scratch/himsnet_rnd_noise_3hop_no_emb_{}cvg".format(round(c*100)) for c in all_coverages]
        mae_segment_30min_no_emb, mae_regional_30min_no_emb = [], []
        for f in coverage_no_emb:
            error_metrics = get_error_metrics_from_log(f)
            mae_segment_30min_no_emb.append(error_metrics["30min_mae_segment"])
            mae_regional_30min_no_emb.append(error_metrics["30min_mae_region"])
        
    # draw a line plot for different coverage
    fig, ax = plt.subplots()
    ax.plot(all_coverages, mae_segment_30min, label="Segment-level")
    ax.plot(all_coverages, mae_regional_30min, label="Regional")
    if include_no_emb:
        ax.plot(all_coverages, mae_segment_30min_no_emb, label="Segment (No Emb)", linestyle="-")
        ax.plot(all_coverages, mae_regional_30min_no_emb, label="Regional (No Emb)", linestyle="-")
    ax.set_xlabel("Coverage")
    ax.set_ylabel("MAE")
    # ax.set_title("30-min MAE by coverage")
    ax.legend()
    plt.tight_layout()
    plt.savefig("datasets/simbarca/figures/mae_30min_coverage.pdf")

def draw_loss_weight():
    # plot for different loss weight
    all_loss_weights = [0.2, 0.5, 2, 5]
    different_weights = ["scratch/himsnet_regional_loss_{}_5hop".format(str(weight).replace(".", "p")) for weight in all_loss_weights]
    different_weights.insert(2, default_output_dir)
    all_loss_weights.insert(2, 1.0)

    mae_segment_30min, mae_regional_30min = [], []
    for f in different_weights:
        error_metrics = get_error_metrics_from_log(f)
        mae_segment_30min.append(error_metrics["30min_mae_segment"])
        mae_regional_30min.append(error_metrics["30min_mae_region"])
        
    # draw a bar plot, put segment-level and regional-level MAE to different bar groups
    fig, ax = plt.subplots()
    bar_width = 0.25
    index = np.arange(len(different_weights))
    bar1 = ax.bar(index, mae_segment_30min, bar_width, label="Segment-level")
    ax.bar_label(bar1, padding=3)
    bar2 = ax.bar(index + bar_width, mae_regional_30min, bar_width, label="Regional")
    ax.bar_label(bar2, padding=3)
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(["{}".format(weight) for weight in all_loss_weights])
    ax.set_xlabel("Weight of Regional Loss")
    ax.set_ylim(0.2, 1.4)
    ax.legend(ncols=2)
    ax.set_ylabel("MAE")
    # ax.set_title("30-min MAE of different loss weights")
    plt.tight_layout()
    plt.savefig("datasets/simbarca/figures/mae_30min_loss_weight.pdf")
    
def draw_ablation_emb_with_10reps():
    # so we basically confirmed that EMB has just a little effect on the performance
    # but it looks positive.
    default_group = sorted(glob.glob("scratch/himsnet_3hop_r*"))
    no_emb_group = sorted(glob.glob("scratch/himsnet_no_emb_3hop_r*"))
    pn_full_group = sorted(glob.glob("scratch/himsnet_rnd_noise_fix_3hop_r*"))
    pn_no_emb_group = sorted(glob.glob("scratch/himsnet_rnd_no_emb_noise_fix_3hop_r*"))
    
    def get_10step_mae(folder):
        
        exp_log = "{}/experiment.log".format(folder)
        log_contents = open(exp_log, "r").readlines()
        
        segment_line_idx = [idx for idx, line in enumerate(log_contents) if "Evaluate model on pred_speed\n" in line][-1]
        regional_line_idx = [idx for idx, line in enumerate(log_contents) if "Evaluate model on pred_speed_regional\n" in line][-1]
        
        segment_eval_res = log_contents[segment_line_idx:regional_line_idx]
        for idx, line in enumerate(segment_eval_res):
            if "Evaluate model on test data at 10 time step" in line:
                # obtain the number after "MAE" from a string like 
                # [11/15 16:23:13 default]: MAE: 1.2141, MAPE: 0.2748, RMSE: 1.9926
                mae_at_30min = float(segment_eval_res[idx+1].split("MAE: ")[1].split(",")[0])
    
        regional_eval_res = log_contents[regional_line_idx:]
        for idx, line in enumerate(regional_eval_res):
            if "Evaluate model on test data at 10 time step" in line:
                regional_mae_at_30min = float(regional_eval_res[idx+1].split("MAE: ")[1].split(",")[0])
                
        return mae_at_30min, regional_mae_at_30min
    
    def get_mean_variance_of_30_min_mae(group_of_folders:list[str]):
        segment_mae, regional_mae = [], []
        for folder in group_of_folders:
            mae_segment, mae_regional = get_10step_mae(folder)
            segment_mae.append(mae_segment)
            regional_mae.append(mae_regional)
        return np.mean(segment_mae), np.std(segment_mae), np.mean(regional_mae), np.std(regional_mae)
    
    for group, name in zip([default_group, no_emb_group, pn_full_group, pn_no_emb_group], ["Default", "No Emb", "Noise", "Noise No Emb"]):
        segment_mae, segment_std, regional_mae, regional_std = get_mean_variance_of_30_min_mae(group)
        print("Evaluating the group of {} with 10 repetitions".format(name))
        print("Segment Level MAE at 10 steps (30 min): {:.4f} ± {:.4f}".format(segment_mae, segment_std))
        print("Regional Level MAE at 10 steps (30 min): {:.4f} ± {:.4f}".format(regional_mae, regional_std))
        
    
def check_emb_params_update():
    import torch 
    
    default_ckpts = sorted(glob.glob("scratch/himsnet_3hop/himsnet_epoch_*.pth"), key=lambda x: int(x.split("_")[-1].split(".")[0]))
    no_emb_ckpts = sorted(glob.glob("scratch/himsnet_no_emb_3hop/himsnet_epoch_*.pth"), key=lambda x: int(x.split("_")[-1].split(".")[0]))

    def see_cosine_similarity_tokens(ckpts):
        emptys, unmonitored_tokens = [], []
        for ckpt in ckpts:
            print(ckpt)
            state_dict = torch.load(ckpt)
            empty = state_dict['model']['model_params']['drone_embedding.empty_token'].cpu().numpy()
            unmonitored = state_dict['model']['model_params']['drone_embedding.unmonitored_token'].cpu().numpy()
            emptys.append(empty)
            unmonitored_tokens.append(unmonitored)

        # consine similarity between the first empty token and the rest
        inner_products = [np.dot(emptys[0], empty) / (np.linalg.norm(emptys[0]) * np.linalg.norm(empty)) for empty in emptys]
        inner_products_unmonitored = [np.dot(unmonitored_tokens[0], unmonitored) / (np.linalg.norm(unmonitored_tokens[0]) * np.linalg.norm(unmonitored)) for unmonitored in unmonitored_tokens]
        print(inner_products)
        print(inner_products_unmonitored)
        
    print("checking default setup")
    see_cosine_similarity_tokens(default_ckpts)
    print("checking no emb setup")
    see_cosine_similarity_tokens(no_emb_ckpts)
    

def draw_ld_only_regional_mae_example():
    import json
    
    data_ld_cvg10_train = json.load(open('scratch/himsnet_rnd_no_drone_noise_fix_3hop/figures/log_pred_speed_regional_mae_train.json', 'r'))
    data_ld_cvg10_test = json.load(open('scratch/himsnet_rnd_no_drone_noise_fix_3hop/figures/log_pred_speed_regional_mae_test.json', 'r'))
    data_ld_full_train = json.load(open('scratch/himsnet_no_drone_3hop/figures/log_pred_speed_regional_mae_train.json', 'r'))
    data_ld_full_test = json.load(open('scratch/himsnet_no_drone_3hop/figures/log_pred_speed_regional_mae_test.json', 'r'))
    
    epochs = data_ld_cvg10_train['epoch']
    fig, ax = plt.subplots(figsize=(6,5))
    ax.plot(epochs, data_ld_cvg10_train['value'], label="10% Coverage train")
    ax.plot(epochs, data_ld_cvg10_test['value'], label="10% Coverage test")
    ax.plot(epochs, data_ld_full_train['value'], label="Full Coverage train")
    ax.plot(epochs, data_ld_full_test['value'], label="Full Coverage test")
    ax.set_xlabel("Epoch")
    ax.legend()
    fig.tight_layout()
    save_path = "datasets/simbarca/figures/ld_only_regional_mae_example.pdf"
    fig.savefig(save_path)
    print("figure saved to {}".format(save_path))
    plt.close(fig)

def compare_norm_input_at_different_hops():
    hops = range(1, 27, 2)
    repeats = range(5)
    
    folders = ["scratch/himsnet_{}hop_r{}".format(i, r) for i in hops for r in repeats]
    
    mae_segment_30min, mae_regional_30min = [], []
    mae_segment_30min_no_dropout, mae_regional_30min_no_dropout = [], []
    for f in folders:
        error_metrics = get_error_metrics_from_log(f)
        mae_segment_30min.append(error_metrics["30min_mae_segment"])
        mae_regional_30min.append(error_metrics["30min_mae_region"])
        
    mae_segment_30min = np.array(mae_segment_30min).reshape(-1, 5).mean(axis=1)
    mae_regional_30min = np.array(mae_regional_30min).reshape(-1, 5).mean(axis=1)
    
    # plot the errors for different hops using a line plot, with and without normalized input
    fig, ax = plt.subplots()
    ax.plot(hops, mae_segment_30min, label="Segment-level")
    ax.plot(hops, mae_regional_30min, label="Regional")

    ax.set_xlabel("Number of Adjacency Hops")
    ax.set_ylabel("MAE")
    ax.legend()
    fig.tight_layout()
    save_path = "datasets/simbarca/figures/mae_30min_norm_input.pdf"
    fig.savefig(save_path)
    print("figure saved to {}".format(save_path))
    plt.close(fig)

if __name__ == "__main__":
    draw_hops()
    draw_epochs()
    draw_coverage(include_no_emb=False)
    draw_loss_weight()
    # draw_ablation_emb_with_10reps()
    # check_emb_params_update()
    draw_hidden_dimension()
    # draw_ld_only_regional_mae_example()
    # compare_norm_input_at_different_hops()