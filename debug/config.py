train = {
	'max_epoch': 30,
	'output_dir': 'debug',
	'checkpoint': '',
	'device': 'cuda',
	'test_best_ckpt': False,
	'grad_clip': 3.0,
	'eval_train': True,
} 

scheduler = {
	'start': 0,
	'end': '${..train.max_epoch}',
	'lr_milestone': [0.7, 0.85],
	'lr_decrease': 0.1,
	'warmup': 1.0,
} 

optimizer = {
	'type': 'adam',
	'lr': 0.001,
	'weight_decay': 0.0001,
	'betas': [0.9, 0.999],
} 

model = {
	'name': 'himsnet',
	'device': 'cuda',
	'use_drone': True,
	'use_ld': True,
	'use_global': True,
	'scale_output': False,
	'normalize_input': False,
	'd_model': 64,
} 

data = {
	'train': {
		'batch_size': 8,
	},
	'test': {
		'batch_size': 8,
	},
} 

