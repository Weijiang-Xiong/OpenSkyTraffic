import unittest

import torch
from netsanut.loss import GeneralizedProbRegLoss

class TestLoss(unittest.TestCase):

    def test_forward(self):
        rand_output = torch.rand(size=(5,5), requires_grad=True)
        rand_label  = torch.rand_like(rand_output)
        rand_plog_sigma = torch.rand_like(rand_output, requires_grad=True)
        
        criterion = GeneralizedProbRegLoss(reduction='mean', aleatoric=True)
        loss = criterion(rand_output, rand_label, rand_plog_sigma)
        loss.backward() 
        
        self.assertIsNotNone(rand_output.grad)
        self.assertIsNotNone(rand_plog_sigma.grad)

    def test_forward_without_plog_sigma(self):
        rand_output = torch.rand(size=(5,5), requires_grad=True)
        rand_label  = torch.rand_like(rand_output)
        rand_plog_sigma = torch.rand_like(rand_output, requires_grad=True)
        
        criterion = GeneralizedProbRegLoss(reduction='mean', aleatoric=True)
        loss = criterion(rand_output, rand_label)
        loss.backward() 
        
        self.assertIsNotNone(rand_output.grad)
        self.assertIsNone(rand_plog_sigma.grad)
        
    def test_ignore_value(self):
        rand_output = torch.ones(size=(5,5), requires_grad=True)
        rand_label  = torch.ones_like(rand_output)
        rand_label[0, ...] = 0.0
        
        criterion = GeneralizedProbRegLoss(reduction=None, aleatoric=False, ignore_value=0.0)
        loss = criterion(rand_output, rand_label)
        self.assertTrue(torch.allclose(loss, torch.tensor(0.0)))

    def test_nan_to_num(self):
        rand_output = torch.ones(size=(5,5), requires_grad=True)
        rand_label  = torch.zeros_like(rand_output)
        
        criterion = GeneralizedProbRegLoss(reduction="mean", aleatoric=False, ignore_value=0.0)
        loss = criterion(rand_output, rand_label)
        self.assertTrue(torch.allclose(loss, torch.tensor(0.0))) 
    
if __name__ == "__main__":
    unittest.main()