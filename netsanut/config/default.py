import os
import sys
import argparse

from omegaconf import OmegaConf, DictConfig
from netsanut.event_logger import setup_logger

def default_argument_parser(input_args=None) -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(epilog=f"""
Example usage:

python {sys.argv[0]} --config-file config/LSTM_TF_stable.py

""")

    parser.add_argument('--config-file', type=str, default="", help='path to config file')
    parser.add_argument('--eval-only', action="store_true", help="skip training and run evaluation only")
    parser.add_argument('--resume', action="store_true", default=False, help='resume training from checkpoint (inc. model, optimizer and scheduler)')
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER, help="config override".strip())
    
    return parser

def make_dir_if_not_exist(path):
    if not os.path.exists(path):
        print("Creating directory: {}".format(path))
        os.makedirs(path)

def default_setup(cfg: DictConfig, args):
    
    """ common setup at the beginning of experiments: 
            1. setup logger
            2. log command line arguments and the experiment config
            3. save the config (after merging with command line inputs) to output folder
    """
    save_dir = cfg.train.output_dir
    make_dir_if_not_exist(save_dir)
    logger = setup_logger(name="default", log_file="{}/experiment.log".format(save_dir))
    logger.info("Command Line Arguments: {}".format(args))
    logger.info("Start Training with the following configurations:")
    logger.info(OmegaConf.to_yaml(cfg))
    
    
