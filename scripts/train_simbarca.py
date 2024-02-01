import torch
import numpy as np

from netsanut.config import default_argument_parser, default_setup, ConfigLoader
from netsanut.engine import DefaultTrainer, hooks
from netsanut.models import build_model
from netsanut.data.datasets.simbarca import build_train_loader, build_test_loader, evaluate
from netsanut.solver import build_optimizer, build_scheduler

def main(args):
    
    # the config file will be loaded first and overrides will be applied after that
    # then, the logger and save folder will be setup
    cfg = ConfigLoader.load_from_file(args.config_file)
    cfg = ConfigLoader.apply_overrides(cfg, overrides=args.opts)
    default_setup(cfg, args)
    
    model = build_model(cfg.model)
    
    if args.eval_only:
        test_loader = build_test_loader(**cfg.data.test)
        # in evaluation mode, just load the checkpoint and call evaluation function
        state_dict = DefaultTrainer.load_file(ckpt_path=cfg.train.checkpoint)
        model.load_state_dict(state_dict['model'])
        eval_res = evaluate(
            model, 
            test_loader,
            verbose=True
        )
        return eval_res
    else:
        train_loader = build_train_loader(**cfg.data.train)
        test_loader = build_test_loader(**cfg.data.test)
        # build optimizer and scheduler using the corresponding configurations
        optimizer = build_optimizer(model, cfg.optimizer)
        scheduler = build_scheduler(optimizer, cfg.scheduler)
        
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
            hooks.EvalHook(lambda m: evaluate(m, train_loader), metric_suffix='train', eval_after_train=False) if cfg.train.eval_train else None,
            hooks.EvalHook(lambda m: evaluate(m, test_loader), metric_suffix='test', eval_after_train=False),
            hooks.CheckpointSaver(test_best_ckpt=cfg.train.test_best_ckpt),
            hooks.MetricLogger(),
            # after training, we print the results on the test set
            hooks.EvalHook(lambda m: evaluate(m, test_loader, verbose=True), metric_suffix='test', eval_after_epoch=False),
            hooks.GradientClipper(clip_value=cfg.train.grad_clip),
            hooks.PlotTrainingLog()
        ])

        return trainer.train()
    
if __name__ == "__main__":
    # the argument parser requires a `--config-file` which specifies how to configure
    # models and training pipeline, and other overrides to the config file can be passed
    # as `something.to.modify=new_value`
    args = default_argument_parser().parse_args("--config-file ./config/HiMSNet.py train.output_dir=debug".split())
    main(args)