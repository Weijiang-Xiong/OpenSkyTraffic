import unittest

from typing import Any
from collections import Counter
from bisect import bisect_right

import torch 
import torch.nn as nn 
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision.datasets import FakeData
from torchvision.transforms import ToTensor

from netsanut.solver.lr_schedule import WarmupMultiStepScaler

class Model(nn.Module):
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.linear1 = nn.Linear(3*224*224, 100)
        self.linear2 = nn.Linear(100, 10)
    
    def forward(self, x):
        return self.linear2(self.linear1(x))

class TestScheduler(unittest.TestCase):
    
    def test_lr_schedule(self):
        model = Model()

        dataset = FakeData(size=200, transform=ToTensor())
        loader = DataLoader(
            dataset,
            batch_size=10,
            shuffle=False,
            num_workers=0
        )

        optimizer = optim.SGD([{'params': model.linear1.parameters(), "lr":2.0},
                            {'params': model.linear2.parameters()}], lr=1.0, momentum=0.9)
        scaler1 = WarmupMultiStepScaler(0, 6,  [4, 5], 0.5, 1)
        scaler2 = WarmupMultiStepScaler(5, 10, [8, 9], 0.5, 1)

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[scaler1, scaler2])

        for scaler in scheduler.lr_lambdas:
            scaler.to_iteration_based(len(loader))

        criterion = nn.NLLLoss()

        param_lrs = [] 
        for epoch in range(12):
            for batch_idx, (data, target) in enumerate(loader):
                param_lrs.append((epoch+batch_idx/len(loader), optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr']))
                optimizer.zero_grad()
                output = model(data.view(10, -1))
                loss = criterion(output, target.long())
                loss.backward()
                optimizer.step()
                scheduler.step()

        import numpy as np 
        import matplotlib.pyplot as plt

        param_lrs = np.array(param_lrs)
        xs = param_lrs[:, 0]
        lr1 = param_lrs[:, 1]
        lr2 = param_lrs[:, 2]

        # fig, ax = plt.subplots(figsize=(5, 4))
        # ax.plot(xs, lr1, label="lr1")
        # ax.plot(xs, lr2, label='lr2')
        # ax.set_ylabel("Learning Rate")
        # ax.set_xlabel("Epochs")
        # plt.savefig("lr_change.png")


