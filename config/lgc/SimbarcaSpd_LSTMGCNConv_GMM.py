from .LSTMGCNConv_GMM import (
    train, 
    model,
    optimizer,
    scheduler,
    dataloader,
)
from ..common.data import simbarcaspd as dataset
from skytraffic.data.datasets import SimBarcaSpeed
from ..common.evaluation import simbarca_speed_gmm_evaluator as evaluator

dataset.train.input_nan_to_global_avg = False
train.output_dir = "scratch/simbarcaspd_lgc_gmm"
model.model.num_nodes = SimBarcaSpeed.num_nodes
model.model.input_steps = SimBarcaSpeed.input_steps
model.model.pred_steps = SimBarcaSpeed.pred_steps
model.model.assume_clean_input = False
model.data_null_value = SimBarcaSpeed.data_null_value
