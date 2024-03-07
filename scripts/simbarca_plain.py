import torch
import numpy as np
from torch.utils.data import DataLoader

from netsanut.config import default_argument_parser, default_setup, ConfigLoader
from netsanut.models import build_model
from netsanut.data.datasets.simbarca import SimBarca, evaluate
from netsanut.solver import build_optimizer

    
if __name__ == "__main__":
    # the argument parser requires a `--config-file` which specifies how to configure
    # models and training pipeline, and other overrides to the config file can be passed
    # as `something.to.modify=new_value`
    args = default_argument_parser().parse_args("--config-file ./config/HiMSNet.py train.output_dir=debug".split())
    # the config file will be loaded first and overrides will be applied after that
    # then, the logger and save folder will be setup
    cfg = ConfigLoader.load_from_file(args.config_file)
    cfg = ConfigLoader.apply_overrides(cfg, overrides=args.opts)
    default_setup(cfg, args)
    
    model = build_model(cfg.model)
    
    trainset = SimBarca(split="train", force_reload=False)
    train_loader = DataLoader(trainset, batch_size=8, shuffle=True, collate_fn=trainset.collate_fn)
    testset = SimBarca(split="test", force_reload=False)
    test_loader = DataLoader(testset, batch_size=8, shuffle=False, collate_fn=testset.collate_fn)

    # build optimizer and scheduler using the corresponding configurations
    optimizer = build_optimizer(model, cfg.optimizer)
    
    model.train()
    
    for epoch in range(1):
        
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
    eval_res = evaluate(model, test_loader)
