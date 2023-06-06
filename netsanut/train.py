import torch
import numpy as np

from netsanut.config import default_argument_parser, default_setup
from netsanut.engine import DefaultTrainer
from netsanut.model import build_model
from netsanut.data import build_trainvaltest_loaders, StandardScaler

def main(args):
    
    default_setup(args)
    
    dataloaders, metadata = build_trainvaltest_loaders(args.dataset, args.batch_size, args.adj_type)
    scaler = StandardScaler(mean=metadata['mean'], std=metadata['std'])

    model = build_model(args, metadata['adjacency'], scaler)
    
    trainer = DefaultTrainer(args, model, dataloaders)
    
    if args.eval_only:
        trainer.load_checkpoint(args.checkpoint)
        trainer.evaluate(
            trainer.model, 
            trainer.train_val_test_loaders['test'], 
        )
        
    return trainer.train()
    
if __name__ == "__main__":

    args = default_argument_parser("--eval-only --checkpoint scratch/test/ntsmodel_epoch_23_2.73.pth".split())
    main(args)
