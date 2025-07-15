""" This script is used to format the average results into a latex tables.
    One for deterministic metrics and one for probabilistic metrics.
"""
import pandas as pd
from draw_prob_metrics_plot import load_evaluation_results

def format_det_metrics_latex_table(res_dir, groups):
    """Format the results into a latex table."""
    metrics = ['mae', 'mape', 'rmse']
    df = pd.DataFrame(columns=sorted(["{}_{}".format(g, m) for m in metrics for g in groups]))
    for group in groups:
        results = load_evaluation_results(res_dir=res_dir, dataset=group)
        results = dict(sorted(results.items(), key=lambda x: x[0]))
        for method, eval_res in results.items():
            avg_res = eval_res["average"]
            df.loc[method, "{}_{}".format(group, "mae")] = avg_res["mae"]
            df.loc[method, "{}_{}".format(group, "mape")] = avg_res["mape"]*100
            df.loc[method, "{}_{}".format(group, "rmse")] = avg_res["rmse"]
            
    with pd.option_context('display.float_format', '{:.2f}'.format):
        print(df)
    df.to_latex(buf="visualize/result_latex_table_det.tex", index=True, float_format='{:.2f}'.format)

def format_prob_metrics_latex_table(res_dir, groups):
    """Format the results into a latex table."""
    metrics = ['CRPS', 'mAW', 'mCCE']
    df = pd.DataFrame(columns=sorted(["{}_{}".format(g, m) for m in metrics for g in groups]))
    for group in groups:
        results = load_evaluation_results(res_dir=res_dir, dataset=group)
        results = dict(sorted(results.items(), key=lambda x: x[0]))
        for method, eval_res in results.items():
            avg_res = eval_res["average"]
            if "CRPS_GMM_GT" in avg_res:
                df.loc[method, "{}_{}".format(group, "CRPS")] = avg_res["CRPS_GMM_GT"]
                df.loc[method, "{}_{}".format(group, "mAW")] = avg_res["mAW"]
                df.loc[method, "{}_{}".format(group, "mCCE")] = avg_res["mCCE"]
            elif 'mae' in avg_res:
                df.loc[method, "{}_{}".format(group, "CRPS")] = avg_res["mae"]
                df.loc[method, "{}_{}".format(group, "mAW")] = "-"
                df.loc[method, "{}_{}".format(group, "mCCE")] = "-"
            else:
                raise ValueError("CRPS_GMM_GT or mae not found in average results")
            
    with pd.option_context('display.float_format', '{:.2f}'.format):
        print(df)
    df.to_latex(buf="visualize/result_latex_table_prob.tex", index=True, float_format='{:.2f}'.format)

if __name__ == "__main__":
    format_det_metrics_latex_table(res_dir="scratch/result_collection", groups = ['metr', 'pemsbay', 'simbarcaspd'])
    print("\n\n")
    format_prob_metrics_latex_table(res_dir="scratch/result_collection", groups = ['metr', 'pemsbay', 'simbarcaspd'])