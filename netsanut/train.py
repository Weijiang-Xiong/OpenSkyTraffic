import torch
import numpy as np

from netsanut.config import default_argument_parser, default_setup, ConfigLoader
from netsanut.engine import DefaultTrainer
from netsanut.models import build_model
from netsanut.data import build_trainvaltest_loaders

def main(args):
    
    cfg = ConfigLoader.load_from_file(args.config_file)
    cfg = ConfigLoader.apply_overrides(cfg, overrides=args.opts)
    default_setup(cfg, args)
    
    dataloaders, metadata = build_trainvaltest_loaders(**cfg.data)
    model = build_model(cfg.model)
    model.adapt_to_metadata(metadata)
    
    trainer = DefaultTrainer(cfg, model, dataloaders)
    trainer.load_checkpoint(cfg.train.checkpoint, resume=args.resume)
        
    if args.eval_only:
        trainer.load_checkpoint(cfg.train.checkpoint)
        eval_res = trainer.evaluate(
            trainer.model, 
            trainer.train_val_test_loaders['test'], 
        )
        return eval_res
        
    return trainer.train()
    
if __name__ == "__main__":

    args = default_argument_parser().parse_args("--config-file config/TTNet_stable.py".split())
    main(args)
