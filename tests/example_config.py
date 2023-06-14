from dataclasses import dataclass
from omegaconf import OmegaConf

@dataclass
class __TypeSafeConfig:
    name: str = "Config"
    count: int = 10

# this config is type-safe at runtime, e.g., one can't modify `count` to a string
# https://omegaconf.readthedocs.io/en/latest/usage.html#from-structured-config
safe_cfg = OmegaConf.structured(__TypeSafeConfig)

train = {
	'max_epoch': 100,
	'output_dir': 'scratch/longer_training1',
	'checkpoint': '',
	'device': 'cuda',
} 

data = {
	'dataset': 'metr-la',
	'adj_type': 'doubletransition',
	'batch_size': 128,
} 

optimizer = {
	'learning_rate': 0.001,
	'lr_milestone': [0.7, 0.85],
	'lr_decrease': 0.1,
	'dropout': 0.1,
	'weight_decay': 0.0001,
	'grad_clip': 3.0,
} 

model = {
	'name': 'lstm_tf',
	'device': 'cuda',
	'in_dim': 2,
	'out_dim': 12,
	'rnn_layers': 3,
	'hid_dim': 64,
	'enc_layers': 2,
	'dec_layers': 4,
	'heads': 2,
	'aleatoric': False,
	'exponent': 1,
	'alpha': 1.0,
	'ignore_value': 0.0,
}