import torch 
from skymonitor.patch_lgc import PatchedMVLSTMGCNConv
from skymonitor.simbarca_explore import SimBarcaExplore
from skymonitor.augment import RandomGridCoverage, RandomWalkCoverage

from torch.nn import Conv2d
from torch.utils.data import DataLoader


rgc = RandomWalkCoverage(input_window=30, step_size=3, num_positions=10)
dataset = SimBarcaExplore(split="train", input_window=30, step_size=3, augmentations=rgc)

model = PatchedMVLSTMGCNConv(
    use_global=True,
    feature_dim=3,
    d_model=64,
    temp_patching=3,
    global_downsample_factor=1,
    layernorm=True,
    adjacency_hop=1,
    dropout=0.1,
    loss_ignore_value = float("nan"),
    norm_label_for_loss = True,
    input_steps=360,
    pred_steps=10,
    num_nodes=1570,
    pred_feat=2,
    data_null_value=0.0,
    metadata=dataset.metadata
)
batch = dataset.collate_fn([dataset[i] for i in range(2)])
model.to(torch.device("cuda"))
model.train()
loss = model(batch)

model.eval()
with torch.no_grad():
    out = model(batch)