""" SimBarcaSpeed dataset has similar structure as METR-LA, but with different specifications.
    
    In general, when applying a model to a new data set, we need to change:
        - the dataset itself (from ..common.data)
        - the evaluator for the dataset (from ..common.evaluation)
        - the output directory, choose a good name so they can be easily identified
        - dataset-specific parameters of the model, e.g., number of nodes, input and output steps, and data null value.
"""
from .LSTMGCNConv import (
    train, 
    model,
    optimizer,
    scheduler,
    dataloader,
)
from ..common.data import simbarcaspd as dataset
from ..common.evaluation import simbarca_speed_evaluator as evaluator
from skytraffic.data.datasets import SimBarcaSpeed

dataset.train.input_nan_to_global_avg = False
train.output_dir = "scratch/simbarcaspd_lgc"
model.num_nodes = SimBarcaSpeed.num_nodes
model.input_steps = SimBarcaSpeed.input_steps
model.pred_steps = SimBarcaSpeed.pred_steps
model.data_null_value = SimBarcaSpeed.data_null_value