import torch
from netsanut.loss import GeneralizedProbRegLoss

def test_forward():
    rand_output = torch.rand(size=(5,5), requires_grad=True)
    rand_label  = torch.rand_like(rand_output)
    rand_logvar = torch.rand_like(rand_output, requires_grad=True)
    
    criterion = GeneralizedProbRegLoss(reduction='mean', aleatoric=True)
    loss = criterion(rand_output, rand_label, rand_logvar)
    loss.backward() 
    
    assert rand_output.grad is not None
    assert rand_logvar.grad is not None

def test_forward_without_logvar():
    rand_output = torch.rand(size=(5,5), requires_grad=True)
    rand_label  = torch.rand_like(rand_output)
    rand_logvar = torch.rand_like(rand_output, requires_grad=True)
    
    criterion = GeneralizedProbRegLoss(reduction='mean', aleatoric=True)
    loss = criterion(rand_output, rand_label)
    loss.backward() 
    
    assert rand_output.grad is not None
    assert rand_logvar.grad is None
    
def test_ignore_value():
    rand_output = torch.ones(size=(5,5), requires_grad=True)
    rand_label  = torch.ones_like(rand_output)
    rand_label[0, ...] = 0.0
    
    criterion = GeneralizedProbRegLoss(reduction=None, aleatoric=False, ignore_value=0.0)
    loss = criterion(rand_output, rand_label)
    assert torch.allclose(loss, torch.tensor(0.0))

def test_nan_to_num():
    rand_output = torch.ones(size=(5,5), requires_grad=True)
    rand_label  = torch.zeros_like(rand_output)
    
    criterion = GeneralizedProbRegLoss(reduction="mean", aleatoric=False, ignore_value=0.0)
    loss = criterion(rand_output, rand_label)
    assert torch.allclose(loss, torch.tensor(0.0))
    
if __name__ == "__main__":
    test_forward()
    test_forward_without_logvar()
    test_ignore_value()
    test_nan_to_num()