import torch
import numpy as np
import argparse
import time
import util

from config import default_argument_parser, default_setup
from engine import DefaultTrainer
from event_logger import setup_logger
from model import build_model

def main(args):
    
    default_setup(args)
    
    logger = setup_logger(name="default", log_file="{}/experiment.log".format(args.save_dir))
    logger.info("Using these configurations {}".format(args))
    
    device = torch.device(args.device)
    sensor_ids, sensor_id_to_ind, adj_mx = util.load_adj(args.adjdata, args.adjtype)
    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader['scaler']
    adjacencies = [torch.tensor(i).to(device) for i in adj_mx]

    model = build_model(args, adjacencies)
    
    trainer = DefaultTrainer(args, model, dataloader, scaler)
    
    if args.eval_only:
        trainer.load_checkpoint(args.checkpoint)
        trainer.evaluate(trainer.model, 
                         trainer.dataloader['test_loader'], 
                         trainer.scaler)
        
    return trainer.train()
    
if __name__ == "__main__":

    args = default_argument_parser()
    main(args)
