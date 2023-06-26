import torch 
from netsanut.models import NeTSFormer
from einops import rearrange

device = torch.device("cuda")

N, T, M, C = 32, 12, 211, 2
rand_input = torch.rand(size=(N, T, M, C), device=device)
rand_label = torch.rand(size=(N, T, M), device=device)

in_rearrange = rearrange(rand_input.clone(), 'N T M C -> (N M) T C', N=N)
in_reshape = rand_input.clone().permute((0, 2, 1, 3)).reshape(N*M, T, C)
assert in_rearrange.shape == in_reshape.shape 
assert torch.allclose(in_rearrange, in_reshape)

model = NeTSFormer().to(device)
model.adapt_to_metadata({'adjacency': [torch.randint(0, 5, size=(M, M)) for _ in range(2)],
                         'mean': torch.tensor([0.0, 0.0]),
                         'std': torch.tensor([1.0, 0.0])})

model.train()
loss_dict = model(rand_input, rand_label)
loss = sum(loss_dict.values())
loss.backward()

model.eval()
pred = model(rand_input, rand_label)
metrics = model.pop_auxiliary_metrics(scalar_only=False)
assert pred.shape == (N, T, M)
assert metrics['logvar'].shape == (N, T, M)

param_groups = model.det_and_sto_params()
all_params = set(model.parameters())
assert sum([len(g) for g in param_groups.values()]) == len(all_params)
for group_name, group_params in param_groups.items():
    for param in group_params:
        assert param in all_params