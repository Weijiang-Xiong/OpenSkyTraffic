from netsanut.data import NetworkedTimeSeriesDataset, tensor_collate, tensor_to_gpu
from torch.utils.data import DataLoader

dataset = NetworkedTimeSeriesDataset(compute_metadata=True)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True, collate_fn=tensor_collate)

for data, label in dataloader:
    print(data.shape)
    print(label.shape)
    break

dataloader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True, collate_fn=tensor_to_gpu)

for data, label in dataloader:
    print(data.device, data.shape)
    print(label.device, data.shape)
    break