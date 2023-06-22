import torch 
from netsanut.models import TTNet

def test_model_forward():
    
    print("Testing TTNet forward pass...")
    N, C, M, T = 8, 2, 207, 12
    random_input = torch.rand((N, T, M, C))
    adjacencies = [torch.randint(0, 2, (M, M)).bool() for _ in range(2)]
    model = TTNet().eval()
    model.set_fixed_mask(adjacencies)
    mean = model(random_input)
    var = model.pop_auxiliary_metrics()
    assert mean.shape == (N, T, M)
    assert isinstance(var, dict)
    print("Test OK")

if __name__ == "__main__":

    test_model_forward()