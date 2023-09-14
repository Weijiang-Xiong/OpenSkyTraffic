import unittest

import torch
from netsanut.models import PositionalEncoding, LearnedPositionalEncoding

class TestPosEncoding(unittest.TestCase):
    
    def test_posenc(self):
        
        # print("Testing positional encoding...")
        
        encoders_batch_first = [
            PositionalEncoding(64, max_len=100, batch_first=True),
            LearnedPositionalEncoding(64, max_len=100, batch_first=True)
            ]
        encoders_not_batch_first = [
            PositionalEncoding(64, max_len=100, batch_first=False),
            LearnedPositionalEncoding(64, max_len=100, batch_first=False)
            ]
        
        rand_input = torch.rand(size=(2, 12, 64))
        for enc in encoders_batch_first:
            # print(enc.__class__.__name__)
            encodings = enc.encodings(rand_input)
            self.assertTrue(torch.allclose(encodings[0], encodings[1]))
            out = enc(rand_input)
            self.assertTrue(out.shape == rand_input.shape)
            
        for enc in encoders_not_batch_first:
            # print(enc.__class__.__name__)
            encodings = enc.encodings(rand_input)
            self.assertTrue(torch.allclose(encodings[0], encodings[1]))
            out = enc(rand_input)
            self.assertTrue(out.shape == rand_input.shape)

if __name__ == "__main__":
    unittest.main()