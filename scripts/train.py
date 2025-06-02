""" 
This script is the entry point for training and evaluation of the models.
In the __main__ function, the argument parser requires a `--config-file` which specifies how to configure
models and training pipeline, and other overrides to the config file can be passed as 

    something.to.modify=new_value 

To run a training in the terminal, use the following command:

    python scripts/train.py --config-file config/HiMSNet.py model.adjacency_hop=5 train.output_dir=scratch/himsnet_5hop

wAfter the training, you can run evaluation on the trained model as follows (you don't need to pass config override again, just load the config file saved to the output directory).

    python scripts/train.py --eval-only --config-file scratch/himsnet_5hop/config.py evaluation.visualize=True

It is possible to override the checkpoint in case you want to evaluate another checkpoint rather than the finally saved one by adding `train.checkpoint=scratch/himsnet_5hop/model_final.pth`

You can also run debugging using VS Code. For example, to debug the evaluation of a trained HiMSNet model, put the following to parse_args(...). 

    "--eval-only --config-file saved_folder/config.py evaluation.visualize=True".split()
"""

import torch
import numpy as np
from pathlib import Path

from netsanut.config import default_argument_parser, default_setup, ConfigLoader
from netsanut.engine import DefaultTrainer, hooks
from netsanut.models import build_model
from netsanut.data import build_train_loader, build_test_loader, build_dataset
from netsanut.evaluation import build_evaluator
from netsanut.solver import build_optimizer, build_scheduler

def get_checkpoint_path(cfg, args) -> str:
    """
    Get the path to the checkpoint file based on the config file and output directory.
    If the checkpoint is specified in the config, use that; otherwise, use the default path.
    """
    if Path(cfg.train.checkpoint).is_file():
        return cfg.train.checkpoint
    else:
        return "{}/model_final.pth".format(Path(args.config_file).parent)

def main(args):
    
    # the config file will be loaded first and overrides will be applied after that
    # then, the logger and save folder will be setup
    cfg = ConfigLoader.load_from_file(args.config_file)
    cfg = ConfigLoader.apply_overrides(cfg, overrides=args.opts)
    default_setup(cfg, args)
    
    model = build_model(cfg.model)
    
    if args.eval_only:
        test_loader = build_test_loader(build_dataset(cfg.dataset.test), cfg.dataloader.test)
        evaluator = build_evaluator(**cfg.evaluation)
        # in evaluation mode, just load the checkpoint and call evaluation function
        state_dict = DefaultTrainer.load_file(ckpt_path=get_checkpoint_path(cfg, args))
        model.load_state_dict(state_dict['model'])
        eval_res = evaluator(
            model, 
            test_loader,
            verbose=True
        )
        return eval_res
    else:
        train_set, test_set = build_dataset(cfg.dataset.train), build_dataset(cfg.dataset.test)
        train_loader = build_train_loader(train_set, cfg.dataloader.train)
        test_loader = build_test_loader(test_set, cfg.dataloader.test)
        # build optimizer and scheduler using the corresponding configurations
        optimizer = build_optimizer(model, cfg.optimizer)
        scheduler = build_scheduler(optimizer, cfg.scheduler)
        evaluator = build_evaluator(**cfg.evaluation)
        
        # the trainer runs a training loop, which iterates the model through batches in 
        # the dataloader, compute loss and call optimizer to update model parameters
        trainer = DefaultTrainer(cfg, model, train_loader, optimizer)
        # it is possible to load from a pretrained model or resume from previous training
        trainer.load_checkpoint(cfg.train.checkpoint, resume=args.resume)
        # the hooks have functions that are executed before/after each epoch/batch or the whole training, 
        # and will be called by the trainer in the same order as they are initialized here
        trainer.register_hooks([
            hooks.EpochTimer(),
            hooks.StepBasedLRScheduler(scheduler=scheduler),
            hooks.EvalHook(lambda m: evaluator(m, train_loader), metric_suffix='train', eval_after_train=False) if cfg.train.eval_train else None,
            hooks.EvalHook(lambda m: evaluator(m, test_loader), metric_suffix='test', eval_after_train=False),
            hooks.MetricPrinter(),
            hooks.CheckpointSaver(cfg.train.best_metric, cfg.train.test_best_ckpt, cfg.train.save_period),
            # after training, we print the results on the test set
            hooks.EvalHook(lambda m: evaluator(m, test_loader, verbose=True), metric_suffix='final_test', eval_after_epoch=False),
            hooks.GradientClipper(clip_value=cfg.train.grad_clip),
            hooks.PlotTrainingLog(),
            hooks.MetadataHook(metadata=train_set.metadata)
        ])

        return trainer.train()
    
if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    main(args)