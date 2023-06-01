import torch
import numpy as np

from netsanut import util
from netsanut.config import default_argument_parser, default_setup
from netsanut.engine import DefaultTrainer
from netsanut.model import build_model

def main(args):
    
    default_setup(args)
    
    device = torch.device(args.device)
    sensor_ids, sensor_id_to_ind, adj_mx = util.load_adj(args.adjdata, args.adjtype)
    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader['scaler']
    adjacencies = [torch.tensor(i).to(device) for i in adj_mx]

    model = build_model(args, adjacencies, scaler)
    
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
