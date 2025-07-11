import torch

from skytraffic.config import LazyCall as L

SGD = L(torch.optim.SGD)(
    params=None,  # params will be set dynamically in the training loop
    lr=0.001,
    momentum=0.9,
    weight_decay=1e-5,
)


AdamW = L(torch.optim.AdamW)(
    params=None,  # params will be set dynamically in the training loop
    lr=5e-4,
    betas=(0.9, 0.999),
    weight_decay=1e-4,
    eps=1e-8,
)

Adam = L(torch.optim.Adam)(
    params=None,  # params will be set dynamically in the training loop
    lr=1e-3,
    betas=(0.9, 0.999),
    weight_decay=1e-4,
)