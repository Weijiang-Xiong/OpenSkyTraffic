import torch
import numpy as np

from netsanut.config import default_argument_parser, default_setup, ConfigLoader
from netsanut.engine import DefaultTrainer, hooks
from netsanut.models import build_model, GGDModel
from netsanut.data import build_trainvaltest_loaders
from netsanut.evaluation import evaluate
from netsanut.solver import build_optimizer, build_scheduler

def main(args):
    
    cfg = ConfigLoader.load_from_file(args.config_file)
    cfg = ConfigLoader.apply_overrides(cfg, overrides=args.opts)
    default_setup(cfg, args)
    
    dataloaders, metadata = build_trainvaltest_loaders(**cfg.data)
    model = build_model(cfg.model)
    model.adapt_to_metadata(metadata)
    
    if args.eval_only:
        state_dict = DefaultTrainer.load_file(ckpt_path=cfg.train.checkpoint)
        model.load_state_dict(state_dict['model'])
        eval_res = evaluate(
            model, 
            dataloaders['test'], 
            verbose=True
        )
        return eval_res
    else:
        optimizer = build_optimizer(model, cfg.optimizer)
        scheduler = build_scheduler(optimizer, cfg.scheduler)
        
        trainer = DefaultTrainer(cfg, model, dataloaders['train'], optimizer)
        trainer.load_checkpoint(cfg.train.checkpoint, resume=args.resume)
        trainer.register_hooks([
            hooks.EpochTimer(),
            hooks.TrainingStageManager(getattr(cfg.train, "milestone", None), 
                                       getattr(cfg.train, "milestone_cfg", None)),
            # hooks.TrainMetricRecorder(),
            hooks.StepBasedLRScheduler(scheduler=scheduler),
            hooks.ValidationHook(lambda m: evaluate(m, dataloaders['train'], eval_uncertainty=isinstance(model, GGDModel)), 
                                 suffix='train') if cfg.train.eval_train else None,
            hooks.ValidationHook(lambda m: evaluate(m, dataloaders['val'], eval_uncertainty=isinstance(model, GGDModel))),
            hooks.CheckpointSaver(test_best_ckpt=cfg.train.test_best_ckpt),
            hooks.MetricLogger(),
            hooks.TestHook(lambda m: evaluate(m, dataloaders['test'], verbose=True, eval_uncertainty=isinstance(model, GGDModel))),
            hooks.GradientClipper(clip_value=cfg.train.grad_clip),
            hooks.PlotTrainingLog()
        ])
        
        return trainer.train()
    
if __name__ == "__main__":

    args = default_argument_parser().parse_args()
    main(args)
