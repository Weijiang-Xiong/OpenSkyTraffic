from skytraffic.config import LazyCall as L
from skytraffic.evaluation import (
    MetrEvaluator,
    MetrGMMEvaluator,
    SimBarcaEvaluator,
    SimBarcaGMMEvaluator,
    SimBarcaSpeedEvaluator,
    SimBarcaSpeedGMMEvaluator,
)
from .train import train as train_cfg

metr_evaluator = L(MetrEvaluator)(
    save_dir=train_cfg.output_dir,
    visualize=False,
    collect_pred=["pred"],
    collect_data=["target"]
)

metr_gmm_evaluator = L(MetrGMMEvaluator)(
    save_dir=train_cfg.output_dir, 
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
    save_dir=train_cfg.output_dir, 
    visualize=False
)

simbarca_gmm_evaluator = L(SimBarcaGMMEvaluator)(
    ignore_value=float("nan"),
    mape_threshold=1.0,
    save_dir=train_cfg.output_dir,
    save_note="",
    visualize=False,
    add_output_seq=None
)

simbarca_speed_evaluator = L(SimBarcaSpeedEvaluator)(
    save_dir=train_cfg.output_dir, 
    visualize=False, 
    collect_pred=["pred"], 
    collect_data=["target"]
)

simbarca_speed_gmm_evaluator = L(SimBarcaSpeedGMMEvaluator)(
    save_dir=train_cfg.output_dir, 
    visualize=False, 
    collect_pred=["pred", "mixing", "means", "log_var"], 
    collect_data=["target"],
    data_min=0.0, 
    data_max=70.0,
    sp_size=5,
    gpu=True,
    ci_pts=500
)