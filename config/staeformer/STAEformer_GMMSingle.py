from ..common.train import train
from ..common.evaluation import metr_gmm_evaluator as evaluator
from ..common.optim import AdamW as optimizer
from ..common.schedule import scheduler
from .STAEformer import dataset, dataloader
from .STAEformer_GMM import model

# Override train settings
train.output_dir = "scratch/metr_staeformer_gmm_single"

model.anchors = [0.0]
model.sizes = [3.0] 