import json
import argparse

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

# Set style for academic publication
sns.set_style("darkgrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 12


def load_evaluation_results(dataset: str, res_group_file: str = "visualize/result_groups.json"):
    result_groups = json.load(open(res_group_file))
    dataset_results = result_groups.get(dataset, {})

    """Load evaluation results from all method folders."""
    result_dir = Path("scratch")
    results = {}
    for method_dir, method_name in dataset_results.items():
        eval_file = result_dir / method_dir / "evaluation" / "final_evaluation_scores.json"
        if eval_file.exists():
            with open(eval_file, 'r') as f:
                data = json.load(f)
                results[method_name] = data
    return results

def plot_ci_coverage(results, save_note="dataset"):
    """Plot confidence interval coverage vs confidence level."""
    plt.figure(figsize=(8, 6))
    
    # Add reference line where coverage equals confidence level
    x_ref = np.linspace(0.5, 0.95, 100)
    plt.plot(x_ref, x_ref, 'k--', alpha=0.6, linewidth=1, label='Ideal Coverage')

    for method, data in results.items():
        ci_keys = [k for k in data["average"].keys() if k.startswith("CI_COVER_")]
        confidence_levels = [float(k.split("_")[-1]) for k in ci_keys]
        coverage_values = [data["average"][k] for k in ci_keys]
        if len(confidence_levels) == 0:
            continue
        # Sort by confidence level
        sorted_pairs = sorted(zip(confidence_levels, coverage_values))
        confidence_levels, coverage_values = zip(*sorted_pairs)
        plt.plot(confidence_levels, coverage_values, marker='o', label=method, linewidth=2)

    
    plt.xlabel('Confidence')
    plt.ylabel('Data Coverage')
    plt.legend()
    plt.tight_layout()
    save_path = f'visualize/figures/{save_note}_data_coverage_by_ci_level.pdf'
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved figure to {save_path}")
    plt.show()

def plot_cce_horizon(results, save_note="dataset"):
    """Plot CCE values over prediction horizon."""
    plt.figure(figsize=(8, 6))
    
    # Get prediction horizon from the first method's first horizon metric
    first_method_data = list(results.values())[0]
    first_horizon_metric = list(first_method_data["horizon"].values())[0]
    pred_horizon = len(first_horizon_metric)

    for method, data in results.items():
        if "mCCE" in data["horizon"]:
            horizons = list(range(1, pred_horizon + 1))
            cce_values = data["horizon"]["mCCE"]
            plt.plot(horizons, cce_values, marker='o', label=method, linewidth=2)
    
    plt.xlabel('Prediction Horizon')
    plt.ylabel('Confidence Calibration Error (CCE)')
    plt.xticks(range(1, pred_horizon + 1))
    plt.legend()
    plt.tight_layout()
    save_path = f'visualize/figures/{save_note}_mcce_by_pred_horizon.pdf'
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved figure to {save_path}")
    plt.show()

def plot_aw_horizon(results, save_note="dataset"):
    """Plot AW values over prediction horizon."""
    plt.figure(figsize=(8, 6))
    
    # Get prediction horizon from the first method's first horizon metric
    first_method_data = list(results.values())[0]
    first_horizon_metric = list(first_method_data["horizon"].values())[0]
    pred_horizon = len(first_horizon_metric)

    for method, data in results.items():
        if "mAW" in data["horizon"]:
            horizons = list(range(1, pred_horizon + 1))
            aw_values = data["horizon"]["mAW"]
            plt.plot(horizons, aw_values, marker='o', label=method, linewidth=2)
    
    plt.xlabel('Prediction Horizon')
    plt.ylabel('Average Width (AW) of CI')
    plt.xticks(range(1, pred_horizon + 1))
    plt.legend()
    plt.tight_layout()
    save_path = f'visualize/figures/{save_note}_maw_by_pred_horizon.pdf'
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved figure to {save_path}")
    plt.show()

def plot_crps_gt(results, save_note="dataset", det_results=None):
    """Plot CRPS_GMM_GT values over prediction horizon."""
    plt.figure(figsize=(8, 6))
    
    # Get prediction horizon from the first method's first horizon metric
    first_method_data = list(results.values())[0]
    first_horizon_metric = list(first_method_data["horizon"].values())[0]
    pred_horizon = len(first_horizon_metric)

    for method, data in results.items():
        # draw the CRPS values for probabilistic methods
        if "CRPS_GMM_GT" in data["horizon"]:
            horizons = list(range(1, pred_horizon + 1))
            crps_values = data["horizon"]["CRPS_GMM_GT"]
            plt.plot(horizons, crps_values, marker='o', label=method, linewidth=2)
        # if there is no CRPS, then the method is deterministic, we draw the MAE values,
        # MAE is a special case of CRPS where the prediction is a point estimate
        elif "mae" in data["horizon"]:
            pred_horizon = len(data["horizon"]["mae"])
            horizons = list(range(1, pred_horizon + 1))
            mae_values = data["horizon"]["mae"]
            plt.plot(horizons, mae_values, marker='o', label=f"{method}-Det (MAE)", linewidth=2)
    
    plt.xlabel('Prediction Horizon')
    plt.ylabel('CRPS')
    plt.xticks(range(1, pred_horizon + 1))
    plt.legend()
    plt.tight_layout()
    save_path = f'visualize/figures/{save_note}_crps_wrt_gt_by_pred_horizon.pdf'
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved figure to {save_path}")
    plt.show()


def plot_crps_emp(results, save_note="dataset"):
    """Plot CRPS_GMM_EMP values over prediction horizon."""
    plt.figure(figsize=(8, 6))

    # Get prediction horizon from the first method's first horizon metric
    first_method_data = list(results.values())[0]
    first_horizon_metric = list(first_method_data["horizon"].values())[0]
    pred_horizon = len(first_horizon_metric)

    for method, data in results.items():
        if "CRPS_GMM_EMP" in data["horizon"]:
            horizons = list(range(1, pred_horizon + 1))
            crps_values = data["horizon"]["CRPS_GMM_EMP"]
            plt.plot(horizons, crps_values, marker='o', label=method, linewidth=2)
    
    plt.xlabel('Prediction Horizon')
    plt.ylabel('CRPS')
    plt.xticks(range(1, pred_horizon + 1))
    plt.legend()
    plt.tight_layout()
    save_path = f'visualize/figures/{save_note}_crps_wrt_emp_by_pred_horizon.pdf'
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved figure to {save_path}")
    plt.show()


def plot_mae_mape_rmse_horizon(results, save_note="dataset"):
    """Plot MAE, MAPE, and RMSE values over prediction horizon."""
    for metric in ["mae", "mape", "rmse"]:
        fig, ax = plt.subplots(figsize=(8, 6))

        # Get prediction horizon from the first method's first horizon metric
        first_method_data = list(results.values())[0]
        first_horizon_metric = list(first_method_data["horizon"].values())[0]
        pred_horizon = len(first_horizon_metric)

        for method, data in results.items():
            if metric in data["horizon"]:
                horizons = list(range(1, pred_horizon + 1))
                values = data["horizon"][metric]
                ax.plot(horizons, values, marker='o', label=method, linewidth=2)
        
        ax.set_xlabel('Prediction Horizon')
        ax.set_ylabel(metric)
        ax.legend()
        fig.tight_layout()
        save_path = f'visualize/figures/{save_note}_{metric}_by_pred_horizon.pdf'
        fig.savefig(save_path, bbox_inches='tight')
        print(f"Saved figure to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="metr")
    args = parser.parse_args()

    # Load results from all methods (probabilistic methods and deterministic methods)
    results = load_evaluation_results(args.dataset)
    
    if not results:
        print("No evaluation results found!")
        exit()
    
    print(f"Found results for {len(results)} methods: {list(results.keys())}")
    
    # Generate plots
    plot_ci_coverage(results, save_note=args.dataset)
    plot_cce_horizon(results, save_note=args.dataset)
    plot_aw_horizon(results, save_note=args.dataset)
    plot_crps_gt(results, save_note=args.dataset)
    plot_crps_emp(results, save_note=args.dataset)
    plot_mae_mape_rmse_horizon(results, save_note=args.dataset)
