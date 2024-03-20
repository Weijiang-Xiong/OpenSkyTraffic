""" this is a debug script, which will test the model with a few samples from the dataset

    it is supposed to be lauched with vscode python debugger
"""


import torch
import numpy as np
from torch.utils.data import DataLoader

from netsanut.config import default_argument_parser, default_setup, ConfigLoader
from netsanut.models import build_model
from netsanut.data.datasets import SimBarca
from netsanut.evaluation import SimBarcaEvaluator
from netsanut.solver import build_optimizer

def build_evaluator(evaluator_type, save_dir=None, save_res=True, save_note=None, **kwargs):
    match evaluator_type:
        case None:
            raise ValueError('No evaluator is specified')
        case 'simbarca':
            return SimBarcaEvaluator(save_dir, save_res, save_note)

if __name__ == "__main__":
    # the argument parser requires a `--config-file` which specifies how to configure
    # models and training pipeline, and other overrides to the config file can be passed
    # as `something.to.modify=new_value`
    args = default_argument_parser().parse_args("--config-file ./config/HiMSNet.py train.output_dir=scratch/debug model.scale_output=True model.layernorm=True model.normalize_input=False model.d_model=64 optimizer.lr=0.001 model.normalize_input=True".split())
    # the config file will be loaded first and overrides will be applied after that
    # then, the logger and save folder will be setup
    cfg = ConfigLoader.load_from_file(args.config_file)
    cfg = ConfigLoader.apply_overrides(cfg, overrides=args.opts)
    default_setup(cfg, args)
    
    model = build_model(cfg.model)
    evaluator = build_evaluator(**cfg.evaluation)
    
    trainset = SimBarca(split="train", force_reload=False)
    for seq in trainset.sequence_names:
        seq_data = getattr(trainset, seq)[:2, ...]
        setattr(trainset, seq, seq_data)
    train_loader = DataLoader(trainset, batch_size=2, shuffle=False, collate_fn=trainset.collate_fn)

    # build optimizer and scheduler using the corresponding configurations
    optimizer = build_optimizer(model, cfg.optimizer)
    
    model.train()
    
    for epoch in range(2000):
        
        loss_list = []
        
        for batch in train_loader:
            loss_dict = model(batch)
            loss = sum(loss_dict.values())
            loss.backward() 
            
            optimizer.step() 
            optimizer.zero_grad()
            
            loss_list.append(loss.item())
        
        print(f"Epoch {epoch}: {np.mean(loss_list)}")
    
    model.eval() 
    eval_res = evaluator(model, train_loader)
    print(eval_res)