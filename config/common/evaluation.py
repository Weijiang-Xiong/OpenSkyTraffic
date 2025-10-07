from skytraffic.config import LazyCall as L
from skytraffic.evaluation import (
    MetrEvaluator,
    SimBarcaEvaluator,
    SimBarcaSpeedEvaluator,
)

metr_evaluator = L(MetrEvaluator)(
    # we assume that evaluator will be a top-level config, and in the same level we have `train`
    save_dir="${train.output_dir}/evaluation",
    visualize=False,
    collect_pred=["pred"],
    collect_data=["target"]
)

simbarca_evaluator = L(SimBarcaEvaluator)(
    ignore_value=float("nan"), 
    mape_threshold=1.0, 
    save_dir="${train.output_dir}/evaluation", 
    visualize=False
)

simbarca_speed_evaluator = L(SimBarcaSpeedEvaluator)(
    save_dir="${train.output_dir}/evaluation", 
    visualize=False, 
    collect_pred=["pred"], 
    collect_data=["target"]
)
