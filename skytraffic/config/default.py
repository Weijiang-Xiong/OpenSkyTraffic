import os
import sys
import yaml
import argparse

from omegaconf import OmegaConf, DictConfig
from ..utils.event_logger import setup_logger
from ..utils.io import make_dir_if_not_exist
from .lazy import LazyConfig

def default_argument_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(epilog=f"""
An example to train with `NeTSFormer_prediction.py` and apply overrides to max_epoch and batch_size

python {sys.argv[0]} --config-file config/NeTSFormer_prediction.py train.max_epoch=30 data.batch_size=64

""")

    parser.add_argument('--config-file', type=str, default="", help='path to config file')
    parser.add_argument('--eval-only', action="store_true", help="skip training and run evaluation only")
    parser.add_argument('--resume', action="store_true", default=False, help='resume training from checkpoint (inc. model, optimizer and scheduler)')
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER, help="config override".strip())
    
    return parser


def default_setup(cfg: DictConfig, args):
    """common setup at the beginning of experiments:
    1. setup logger
    2. log command line arguments and the experiment config
    3. save the config (after merging with command line inputs) to output folder
    4. make the save folder if it doesn't exist
    5. replace the referenced values in the config with literal values
    """
    OmegaConf.resolve(cfg)
    save_dir = cfg.train.output_dir
    make_dir_if_not_exist(save_dir)
    logger = setup_logger(name="default", log_file="{}/experiment.log".format(save_dir))
    logger.info("Command Line Arguments: {}".format(args))
    cfg_save_path = "{}/config.yaml".format(save_dir)
    if not args.eval_only:  # save the training config only
        LazyConfig.save(cfg, cfg_save_path)
        logger.info("Config saved to {}".format(cfg_save_path))
    # cfg will contain python objects when loaded, and printing a loaded cfg is not readable in terminal
    # so we read the saved config file and log the readable format
    logger.info(
        "Start experiment with the following configurations: \n {}".format(
            yaml.dump(
                yaml.safe_load(open(cfg_save_path, "r")),
                sort_keys=False,
                indent=2,
                default_flow_style=False,
            )
        )
    )
    
    
    
