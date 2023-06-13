from .common_cfg import train, data, optimizer, test

model = {
    # these are information about the model
    "name": "lstm_tf",
    "device": "cuda",
    # these are model parameters
    "in_dim": 2, 
    "out_dim": 12,
    "rnn_layers": 3,
    "hid_dim": 64,
    "enc_layers":2,
    "dec_layers": 4,
    "heads": 2,
    # these are related to loss 
    "aleatoric": False, 
    "exponent": 1,
    "alpha": 1.0,
    "ignore_value": 0.0
}