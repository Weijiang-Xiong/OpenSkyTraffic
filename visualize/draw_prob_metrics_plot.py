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
        "gmmpred_bayes_avg_fullinfo": "Bayes-100%",
        "gmmpred_bayes_avg_rndobs": "Bayes-10%",
    }


RESULT_GROUPS = {
    "simbarca": SIMBARCA_RESULTS,
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

def plot_cce_horizon(results, save_note="dataset"):
    """Plot CCE values over prediction horizon."""
    plt.figure(figsize=(8, 6))
    
    for method, data in results.items():
        if "CCE_horizon" in data["horizon"]:
            horizons = list(range(1, len(data["horizon"]["CCE_horizon"]) + 1))
            cce_values = data["horizon"]["CCE_horizon"]
            plt.plot(horizons, cce_values, marker='o', label=method, linewidth=2)
    
    plt.xlabel('Prediction Horizon')
    plt.ylabel('Confidence Calibration Error (CCE)')
    plt.xticks(range(1, 11))
    plt.legend()
    plt.tight_layout()
    save_path = f'visualize/figures/{save_note}_mcce_by_pred_horizon.pdf'
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved figure to {save_path}")
    plt.show()

def plot_aw_horizon(results, save_note="dataset"):
    """Plot AW values over prediction horizon."""
    plt.figure(figsize=(8, 6))
    
    for method, data in results.items():
        if "AW_horizon" in data["horizon"]:
            horizons = list(range(1, len(data["horizon"]["AW_horizon"]) + 1))
            aw_values = data["horizon"]["AW_horizon"]
            plt.plot(horizons, aw_values, marker='o', label=method, linewidth=2)
    
    plt.xlabel('Prediction Horizon')
    plt.ylabel('Average Width (AW) of CI')
    plt.xticks(range(1, 11))
    plt.legend()
    plt.tight_layout()
    save_path = f'visualize/figures/{save_note}_maw_by_pred_horizon.pdf'
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved figure to {save_path}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="simbarca")
    args = parser.parse_args()

    # Load results from all methods
    results = load_evaluation_results(RESULT_GROUPS[args.dataset])
    
    if not results:
        print("No evaluation results found!")
        exit()
    
    print(f"Found results for {len(results)} methods: {list(results.keys())}")
    
    # Generate plots
    plot_ci_coverage(results, save_note=args.dataset)
    plot_cce_horizon(results, save_note=args.dataset)
    plot_aw_horizon(results, save_note=args.dataset)
