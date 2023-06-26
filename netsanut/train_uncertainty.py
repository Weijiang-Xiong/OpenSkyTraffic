import torch 
import torch.nn as nn 
import torch.optim as optim

from typing import Dict, List, Tuple
from torch.utils.data import DataLoader

from netsanut.config import default_argument_parser, default_setup, ConfigLoader
from netsanut.engine import DefaultTrainer
from netsanut.models import build_model
from netsanut.data import build_trainvaltest_loaders


class UncertaintyTrainer(DefaultTrainer):
    """ This implementation can not preserve the event storage from the first-stage training
        is this important ??? 
    """
    def __init__(self, cfg, model: nn.Module, dataloaders: Dict[str, DataLoader]):
        super().__init__(cfg, model, dataloaders)
        self.configure_uncertainty_training(cfg)
        
    def configure_uncertainty_training(self, cfg):
        # configure the optimizer, scheduler, loss etc. for the second stage
        # remember to include stochastic parameters only
        self.optimizer_2nd: optim.Optimizer
        self.scheduler_2nd: optim.lr_scheduler.LRScheduler
        self.model_2nd: nn.Module
        # extend the max epoch 
        self.max_epoch: int 

    def train_uncertainty(self):
        self.optimizer = self.optimizer_2nd
        self.scheduler = self.scheduler_2nd
        self.model_2nd.load_state_dict(self.model.state_dict())
        self.model = self.model_2nd

        return self.train()
        
    def train(self):
        # train without uncertainty for good initialization
        super().train()
        # now train uncertainty part 
        return self.train_uncertainty()
    
    def load_checkpoint(self, ckpt_path: str, resume=False):
        # when resuming, be careful to deal with the training stage, first or second??? 
        return super().load_checkpoint(ckpt_path, resume)

def main(args):
    
    cfg = ConfigLoader.load_from_file(args.config_file)
    cfg = ConfigLoader.apply_overrides(cfg, overrides=args.opts)
    default_setup(cfg, args)
    
    dataloaders, metadata = build_trainvaltest_loaders(**cfg.data)
    model = build_model(cfg.model)
    model.adapt_to_metadata(metadata)
    
    trainer = UncertaintyTrainer(cfg, model, dataloaders)
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

    args = default_argument_parser().parse_args()
    main(args)