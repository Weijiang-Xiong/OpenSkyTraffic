from skytraffic.config import LazyCall as L
from skytraffic.evaluation import (
    MetrEvaluator,
    MetrGMMEvaluator,
    SimBarcaEvaluator,
    SimBarcaGMMEvaluator,
    SimBarcaSpeedEvaluator,
    SimBarcaSpeedGMMEvaluator,
)

metr_evaluator = L(MetrEvaluator)(
    # we assume that evaluator will be a top-level config, and in the same level we have `train`
    save_dir="${train.output_dir}/evaluation",
    visualize=False,
    collect_pred=["pred"],
    collect_data=["target"]
)

metr_gmm_evaluator = L(MetrGMMEvaluator)(
    save_dir="${train.output_dir}/evaluation", 
    visualize=False, 
    collect_pred=["pred", "mixing", "means", "log_var"], 
    collect_data=["target"],
    data_min=0.0, 
    data_max=70.0,
    sp_size=5,
    gpu=True,
    ci_pts=500
)

simbarca_evaluator = L(SimBarcaEvaluator)(
    ignore_value=float("nan"), 
    mape_threshold=1.0, 
    save_dir="${train.output_dir}/evaluation", 
    visualize=False
)

simbarca_gmm_evaluator = L(SimBarcaGMMEvaluator)(
    ignore_value=float("nan"),
    mape_threshold=1.0,
    save_dir="${train.output_dir}/evaluation",
    visualize=False,
    add_output_seq=None
)

simbarca_speed_evaluator = L(SimBarcaSpeedEvaluator)(
    save_dir="${train.output_dir}/evaluation", 
    visualize=False, 
    collect_pred=["pred"], 
    collect_data=["target"]
)

simbarca_speed_gmm_evaluator = L(SimBarcaSpeedGMMEvaluator)(
    save_dir="${train.output_dir}/evaluation", 
    visualize=False, 
    collect_pred=["pred", "mixing", "means", "log_var"], 
    collect_data=["target"],
    data_min=0.0, 
    data_max=70.0,
    sp_size=5,
    gpu=True,
    ci_pts=500
)