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

SIMBARCA_RESULTS = {
    "gmmpred_bayes_avg_fullinfo": "Full-Both",
    "gmmpred_bayes_avg_no_drone_fullinfo": "Full-LD",
    "gmmpred_bayes_avg_no_ld_fullinfo": "Full-Drone",
    "gmmpred_bayes_avg_rndobs": "Partial-Both",
    "gmmpred_bayes_avg_no_drone_rndobs": "Partial-LD",
    "gmmpred_bayes_avg_no_ld_rndobs": "Partial-Drone",
    "barcaspd_lgc_gmm": "LGC-GMM (Low Res)",
    "barcaspd_lgc_single": "LGC-Normal (Low Res)",
    }
SIMBARCA_DET_RESULTS = {
    "barcaspd_lgc": "LGC (Low Res)",
}

METR_RESULTS = {
    "metr_lgc_single": "LGC-Normal",
    "metr_lgc_gmm": "LGC-GMM",
}

METR_DET_RESULTS = {
    "metr_lgc": "LGC",
}

PEMSBAY_RESULTS = {
    "pemsbay_lgc_single": "LGC-Normal",
    "pemsbay_lgc_gmm": "LGC-GMM",
}

PEMSBAY_DET_RESULTS = {
    "pemsbay_lgc": "LGC",
}

RESULT_GROUPS = {
    "simbarca": SIMBARCA_RESULTS,
    "metr": METR_RESULTS,
    "pemsbay": PEMSBAY_RESULTS,
}

DET_RESULT_GROUPS = {
    "simbarca": SIMBARCA_DET_RESULTS,
    "metr": METR_DET_RESULTS,
    "pemsbay": PEMSBAY_DET_RESULTS,
}

PRED_HORIZON = {
    "simbarca": 10,
    "metr": 12,
    "pemsbay": 12,
}

def load_evaluation_results(result_group: dict):
    """Load evaluation results from all method folders."""
    result_dir = Path("scratch")
    results = {}
    for method_dir, method_name in result_group.items():
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

def plot_cce_horizon(results, save_note="dataset", pred_horizon=10):
    """Plot CCE values over prediction horizon."""
    plt.figure(figsize=(8, 6))
    
    for method, data in results.items():
        if "mCCE" in data["horizon"]:
            horizons = list(range(1, len(data["horizon"]["mCCE"]) + 1))
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

def plot_aw_horizon(results, save_note="dataset", pred_horizon=10):
    """Plot AW values over prediction horizon."""
    plt.figure(figsize=(8, 6))
    
    for method, data in results.items():
        if "mAW" in data["horizon"]:
            horizons = list(range(1, len(data["horizon"]["mAW"]) + 1))
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

def plot_crps_gt(results, save_note="dataset", pred_horizon=10, det_results=None):
    """Plot CRPS_GMM_GT values over prediction horizon."""
    plt.figure(figsize=(8, 6))
    
    for method, data in results.items():
        if "CRPS_GMM_GT" in data["horizon"]:
            horizons = list(range(1, len(data["horizon"]["CRPS_GMM_GT"]) + 1))
            crps_values = data["horizon"]["CRPS_GMM_GT"]
            plt.plot(horizons, crps_values, marker='o', label=method, linewidth=2)
    
    if det_results is not None:
        for method, data in det_results.items():
            if "mae" in data["horizon"]:
                horizons = list(range(1, len(data["horizon"]["mae"]) + 1))
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


def plot_crps_emp(results, save_note="dataset", pred_horizon=10):
    """Plot CRPS_GMM_EMP values over prediction horizon."""
    plt.figure(figsize=(8, 6))
    
    for method, data in results.items():
        if "CRPS_GMM_EMP" in data["horizon"]:
            horizons = list(range(1, len(data["horizon"]["CRPS_GMM_EMP"]) + 1))
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="simbarca")
    args = parser.parse_args()

    # Load results from all methods (probabilistic methods and deterministic methods)
    prob_results = load_evaluation_results(RESULT_GROUPS[args.dataset])
    det_results = load_evaluation_results(DET_RESULT_GROUPS[args.dataset])
    pred_horizon = PRED_HORIZON[args.dataset]
    
    if not prob_results:
        print("No evaluation results found!")
        exit()
    
    print(f"Found results for {len(prob_results)} methods: {list(prob_results.keys())}")
    
    # Generate plots
    plot_ci_coverage(prob_results, save_note=args.dataset)
    plot_cce_horizon(prob_results, save_note=args.dataset, pred_horizon=pred_horizon)
    plot_aw_horizon(prob_results, save_note=args.dataset, pred_horizon=pred_horizon)
    plot_crps_gt(prob_results, save_note=args.dataset, pred_horizon=pred_horizon, det_results=det_results)
    plot_crps_emp(prob_results, save_note=args.dataset, pred_horizon=pred_horizon)
