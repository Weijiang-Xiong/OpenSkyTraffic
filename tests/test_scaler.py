from netsanut.data import TensorDataScaler
import torch 

rand_data  = torch.rand((32, 12, 207, 2), requires_grad=False)
data_dim = 0

scaler = TensorDataScaler(torch.mean(rand_data, dim=(0, 1, 2)), torch.std(rand_data, dim=(0, 1, 2)), data_dim=data_dim)
scaled_data = scaler.transform(rand_data.clone())
assert torch.allclose(rand_data[..., 1], scaled_data[..., 1], atol=1e-7)

scaled_back_full = scaler.inverse_transform(scaled_data.clone())
assert torch.allclose(scaled_back_full, rand_data, atol=1e-7)

scaled_back_data_only = scaler.inverse_transform(scaled_data.clone()[..., 0])
assert torch.allclose(scaled_back_data_only, rand_data[..., 0], atol=1e-7)

rand_logvar = torch.rand((32, 12, 207), requires_grad=False)
scaler.inverse_transform_logvar(rand_logvar)