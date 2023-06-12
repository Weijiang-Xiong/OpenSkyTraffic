import torch 
from netsanut.models.lstm_tf import TTNet, PositionalEncoding, LearnedPositionalEncoding

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

def test_posenc():
    
    print("Testing positional encoding...")
    
    encoders_batch_first = [
        PositionalEncoding(64, max_len=100, batch_first=True),
        LearnedPositionalEncoding(64, max_len=100, batch_first=True)
        ]
    encoders_not_batch_first = [
        PositionalEncoding(64, max_len=100, batch_first=False),
        LearnedPositionalEncoding(64, max_len=100, batch_first=False)
        ]
    
    for enc in encoders_batch_first:
        print(enc.__class__.__name__)
        out = enc(torch.ones(size=(2, 12, 64)))
        
    for enc in encoders_not_batch_first:
        print(enc.__class__.__name__)
        out = enc(torch.ones(size=(12, 2, 64)))

if __name__ == "__main__":

    test_posenc()
    test_model_forward()