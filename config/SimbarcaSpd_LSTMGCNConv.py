from .LSTMGCNConv import train, scheduler, evaluation, optimizer, dataloader, model
from .datasets import simbarcaspd as dataset

train.output_dir = "scratch/lgc_simbarcaspd"
dataloader.train.batch_size=32
evaluation.evaluator_type='simbarcaspd' 
model.data_null_value=float('nan') 
model.pred_steps=10
model.input_steps=10
model.num_nodes=1570