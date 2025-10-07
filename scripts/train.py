""" 
This script is the entry point for training and evaluation of the models.
In the __main__ function, the argument parser requires a `--config-file` which specifies how to configure
models and training pipeline, and other overrides to the config file can be passed as 

    something.to.modify=new_value

To run a training in the terminal, use the following command:

    python scripts/train.py --config-file config/NAME_OF_CONFIG_FILE.py THE.OPTION.TO.MODIFY=NEW_VALUE

After the training, you can run evaluation on the trained model and enable visualization as follows.
You don't need to pass config override again, just load the config file saved to the output directory.

    python scripts/train.py --eval-only --config-file SAVE_FOLDER/config.yaml evaluation.visualize=True

It is possible to override the checkpoint in case you want to evaluate another checkpoint rather than the finally saved one by adding `train.checkpoint=PATH_TO_CHECKPOINT/FILE_NAME.pth`

You can also run debugging using VS Code. For example, to debug the evaluation of a trained HiMSNet model, put the following to parse_args(...). 

    "--eval-only --config-file SAVE_FOLDER/config.yaml evaluation.visualize=True".split()
"""

from pathlib import Path

from skytraffic.config import default_argument_parser, default_setup, LazyConfig, instantiate
from skytraffic.engine import DefaultTrainer, hooks

def get_checkpoint_path(cfg, args) -> str:
    """
    Get the path to the checkpoint file based on the config file and output directory.
    If the checkpoint is specified in the config, use that; otherwise, use the default path.
    """
    if Path(cfg.train.checkpoint).is_file():
        return cfg.train.checkpoint
    else:
        return "{}/model_final.pth".format(cfg.train.output_dir)

def main(args):
    
    # the config file will be loaded first and overrides will be applied after that
    # then, the logger and save folder will be setup
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, overrides=args.opts)
    default_setup(cfg, args)

    if args.eval_only:

        test_set = instantiate(cfg.dataset.test)

        cfg.dataloader.test.dataset = test_set
        cfg.dataloader.test.collate_fn = test_set.collate_fn
        test_loader = instantiate(cfg.dataloader.test)

        cfg.model.metadata = test_set.metadata
        model = instantiate(cfg.model).to(cfg.train.device)

        evaluator = instantiate(cfg.evaluator)
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
        train_set = instantiate(cfg.dataset.train)
        test_set = instantiate(cfg.dataset.test)

        cfg.dataloader.train.dataset, cfg.dataloader.train.collate_fn = train_set, train_set.collate_fn
        cfg.dataloader.test.dataset, cfg.dataloader.test.collate_fn = test_set, test_set.collate_fn
        train_loader, test_loader = instantiate(cfg.dataloader.train), instantiate(cfg.dataloader.test)
        
        cfg.model.metadata = train_set.metadata
        model = instantiate(cfg.model).to(cfg.train.device)

        # build optimizer and scheduler using the corresponding configurations
        cfg.optimizer.params = model.parameters()
        optimizer = instantiate(cfg.optimizer)
        cfg.scheduler.optimizer = optimizer
        scheduler = instantiate(cfg.scheduler)
        evaluator = instantiate(cfg.evaluator)
        
        # the trainer runs a training loop, which iterates the model through batches in 
        # the dataloader, compute loss and call optimizer to update model parameters
        trainer = DefaultTrainer(
            model=model, 
            dataloader=train_loader, 
            optimizer=optimizer, 
            max_epoch=cfg.train.max_epoch, 
            output_dir=cfg.train.output_dir
        )
        # it is possible to load from a pretrained model or resume from previous training
        trainer.load_checkpoint(cfg.train.checkpoint, resume=args.resume)
        # the hooks have functions that are executed before/after each epoch/batch or the whole training, 
        # and will be called by the trainer in the same order as they are initialized here
        trainer.register_hooks([
            hooks.EpochTimer(),
            hooks.StepBasedLRScheduler(scheduler=scheduler),
            (
                hooks.EvalHook(
                    lambda m: evaluator(m, train_loader),
                    metric_suffix="train",
                    period=cfg.train.eval_period,
                    eval_after_train=False,
                )
                if cfg.train.eval_train
                else None
            ),
            hooks.EvalHook(lambda m: evaluator(m, test_loader), metric_suffix='test', period=cfg.train.eval_period, eval_after_train=False),
            hooks.MetricPrinter(),
            hooks.CheckpointSaver(cfg.train.best_metric, cfg.train.test_best_ckpt, cfg.train.save_period),
            # after training, we print the results on the test set
            hooks.EvalHook(lambda m: evaluator(m, test_loader, verbose=True), metric_suffix='final_test'),
            hooks.GradientClipper(clip_value=cfg.train.grad_clip),
            hooks.PlotTrainingLog(),
            hooks.TensorboardWriter(period=cfg.train.eval_period, save_dir=cfg.train.output_dir)
        ])

        return trainer.train()
    
if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    main(args)