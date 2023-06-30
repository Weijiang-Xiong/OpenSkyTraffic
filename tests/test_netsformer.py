import torch 
from netsanut.models import NeTSFormer
from netsanut.models.netsformer import TemporalAggregate
from einops import rearrange

device = torch.device("cuda")

N, T, M, C = 32, 12, 211, 2
rand_input = torch.rand(size=(N, T, M, C), device=device)
rand_label = torch.rand(size=(N, T, M), device=device)

in_rearrange = rearrange(rand_input.clone(), 'N T M C -> (N M) T C', N=N)
in_reshape = rand_input.clone().permute((0, 2, 1, 3)).reshape(N*M, T, C)
assert in_rearrange.shape == in_reshape.shape 
assert torch.allclose(in_rearrange, in_reshape)

rand_data = {'source': rand_input, 'target': rand_label}

model = NeTSFormer().to(device)
model.adapt_to_metadata({'adjacency': [torch.randint(0, 5, size=(M, M)) for _ in range(2)],
                         'mean': torch.tensor([0.0, 0.0]),
                         'std': torch.tensor([1.0, 0.0])})

model.train()
loss_dict = model(rand_data)
loss = sum(loss_dict.values())
loss.backward()

model.eval()
result_dict = model(rand_data)
metrics = model.pop_auxiliary_metrics(scalar_only=False)
assert result_dict['pred'].shape == (N, T, M)
assert result_dict['logvar'].shape == (N, T, M)
assert isinstance(metrics, dict)

param_groups = model.get_param_groups()
all_params = set(model.parameters())
assert sum([len(g) for g in param_groups.values()]) == len(all_params)
for group_name, group_params in param_groups.items():
    for param in group_params:
        assert param in all_params

for mode in ['linear', 'last', 'avg']:        
    agg = TemporalAggregate(in_dim=T, mode=mode).to(device)
    out = agg(rand_input)
    assert out.shape == (N, M, C)

print("Num Params. {:.2f}M".format(model.num_params/1e6))