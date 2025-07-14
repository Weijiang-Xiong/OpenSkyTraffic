from ..common.train import train
from ..common.evaluation import metr_gmm_evaluator as evaluator
from ..common.optim import AdamW as optimizer
from ..common.schedule import scheduler
from .GWNET import dataset, dataloader
from .GWNET_GMM import model

# Override train settings
train.output_dir = "scratch/metr_gwnet_gmm_single"

model.anchors = [0.0]
model.sizes = [3.0] 