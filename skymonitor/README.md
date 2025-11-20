### 任务划分

pretrain的目的是，训练一个交通流预测模型来应对数据大量mask的情况。10%随机mask，训练和测试都是。

训练的时候当数据增强（假设我们已经有full info），测试可能要测试10轮取平均。

RL：从pretrain的初始化，freeze住预测模型，用RL训练收集数据的agent；

从不同的随机位置rollout agent（run on train set），用收集到的数据微调预测模型，repeat

### TODO

- [ ]  train policy for 1 round, see if it's better than random

- [ ]  iteratively train RL agent and fine-tune prediction model on the agent's behavior

- [ ]  evaluate MFD deviation? eucledian distance between (gt_density, gt_flow) and (pred_density, pred_flow)

- [X]  baselines: LSTM without GNN, global historical average, per-location historical average

- [X]  drones only observation for HiMSNet, with lower coverage (like 3, 5, 7, 10 drones

### Notes

Right now the baseline agent moves from the "good" agent's previous position.

So the reward is given only based on the advantage made by the immediate step.

This may be a bit problematic, as the advantage may require a few more steps to manifest. So perhaps I need to implement a history storage for the baseline agent as well, and let the baseline and "good" agent have diverged paths for a few steps.
