import os
import argparse

from netsanut.event_logger import setup_logger

def default_argument_parser(input_args=None):

    parser = argparse.ArgumentParser()
    
    # parameters related to data loading and saving
    parser.add_argument('--device', type=str, default='cuda', help='default to cuda')
    parser.add_argument('--dataset', type=str, default='metr-la', help='data path')
    parser.add_argument('--adj-type', type=str, default='doubletransition', help='adj type')
    
    # model related parameters 
    parser.add_argument('--in-dim', type=int, default=2, help='inputs dimension')
    parser.add_argument('--rnn', type=int, default=3, help="number of rnn layers, for temporal encoding")
    parser.add_argument('--hid-dim', type=int, default=64, help='model dimension of transformer')
    parser.add_argument('--enc', type=int, default=2, help='number of Encoder layers')
    parser.add_argument('--dec', type=int, default=4, help='number of Decoder layers')
    parser.add_argument('--num-head', type=int, default=2, help='number of heads in multi-head attention')
    parser.add_argument('--pred-win', type=int, default=12, help='number of time step to predict (depends on the dataset)')
    
    # loss related arguments
    parser.add_argument('--aleatoric', type=bool, default=False, help='whether to use aleatoric uncertainty in loss')
    parser.add_argument('--exponent', type=int, default=1, help='exponent in the regression loss')
    parser.add_argument('--alpha', type=float, default=1.0, help='regularization parameter, multiplied with logvar')
    parser.add_argument('--ignore-value', type=float, default=0.0, help='the value to ignore in labels, e.g., 0.0')
    
    # parameters related to optimizer and scheduler
    parser.add_argument('--max-epoch', type=int, default=30, help='number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--lr-milestone', default=[0.7, 0.85], nargs='+', type=float, help="decrease learning rate at percentage of epochs, default 0.7 and 0.85 max-epoch.")
    parser.add_argument("--lr-decrease", default=0.1, type=float, help="at each milestone, decrease to what proportion of previous learning rate, default to 0.1")
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--weight-decay', type=float, default=0.0001, help='weight decay rate')
    parser.add_argument('--grad-clip', type=float, default=3.0, help="clip gradients at this value")
    
    # training profiling, model saving and loading
    parser.add_argument('--print-every', type=int, default=1000, help='')
    parser.add_argument('--save-dir', type=str, default='scratch/test/', help='save path')
    parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint path')
    parser.add_argument('--eval-only', action="store_true", help="skip training and run evaluation only")
    # load weights only or load everything from the checkpoint
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--load-weights', action="store_true", default=False, help='load pretrained weights from checkpoint')
    group.add_argument('--resume', action="store_true", default=False, help='resume training from checkpoint (inc. model, optimizer and scheduler)')
    
    args = parser.parse_args(input_args)
    
    return args

def make_dir_if_not_exist(path):
    if not os.path.exists(path):
        print("Creating directory: {}".format(path))
        os.makedirs(path)

def default_setup(args):
    
    """ if there are any environment-related codes, put them here, 
        such as checking directories and folders
    """
    make_dir_if_not_exist(args.save_dir)
    logger = setup_logger(name="default", log_file="{}/experiment.log".format(args.save_dir))
    logger.info("Using these configurations {}".format(args))
    

