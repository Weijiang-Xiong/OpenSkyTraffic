### 任务划分

pretrain的目的是，训练一个交通流预测模型来应对数据大量mask的情况。10%随机mask，训练和测试都是。训练的时候当数据增强（假设我们已经有full info），测试可能要测试10轮取平均。

RL：从pretrain的初始化，freeze住预测模型，用RL训练收集数据的agent；从不同的随机位置rollout agent（run on train set），用收集到的数据微调预测模型，repeat