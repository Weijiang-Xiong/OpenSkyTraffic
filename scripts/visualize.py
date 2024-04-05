import os

import seaborn as sns
sns.set_style("darkgrid")

from netsanut.config import default_argument_parser, ConfigLoader
from netsanut.engine import DefaultTrainer
from netsanut.models import build_model
from netsanut.data import build_trainvaltest_loaders
from netsanut.predict import Visualizer
from netsanut.utils.event_logger import setup_logger

if __name__ == "__main__":
    
    parser = default_argument_parser()
    parser.add_argument("--result-dir", type=str, default=None, help="the folder containing all training results")
    parser.add_argument("--cfg-name", type=str, default="config.py", help="file name of the config file")
    parser.add_argument("--ckpt", type=str, default="model_final.pth", help="checkpoint file name")
    args = parser.parse_args()
    
    config_file = "{}/{}".format(args.result_dir, args.cfg_name)
    checkpoint = "{}/{}".format(args.result_dir, args.ckpt)
    
    cfg = ConfigLoader.load_from_file(config_file)
    cfg = ConfigLoader.apply_overrides(cfg, overrides=args.opts)
    logger = setup_logger(name="default", log_file="{}/experiment.log".format(args.result_dir))
    
    dataloaders, metadata = build_trainvaltest_loaders(**cfg.data)
    model = build_model(cfg.model)
    model.adapt_to_metadata(metadata)

    state_dict = DefaultTrainer.load_file(ckpt_path=checkpoint)
    model.load_state_dict(state_dict['model'])
        
    visualizer = Visualizer(model, save_dir=os.path.dirname(checkpoint))
    
    logger.info("Evaluating uncertainty on test set")
    visualizer.inference_on_dataset(dataloaders['test'])
    test_res = visualizer.calculate_metrics(verbose=True)
    visualizer.visualize_calibration(test_res, visualizer.save_dir, save_hint="test")
    visualizer.visualize_day(save_name="example")
    per_loc_test_res = visualizer.calculate_metrics(verbose=False, per_loc=True)
    visualizer.visualize_map(per_loc_test_res, metadata, visualizer.save_dir, save_hint="test")
    
    logger.info("Evaluating uncertainty on train set")
    visualizer.inference_on_dataset(dataloaders['train'])
    train_res = visualizer.calculate_metrics(verbose=True)
    visualizer.visualize_calibration(train_res, visualizer.save_dir, save_hint="train")
    
    logger.info("Calibrating the confidence intervals and re-evaluate on test set")
    # we can use the training set to calibrate the width of confidence interval
    calibrated_res = visualizer.calibrate_scale_offset()
    visualizer.inference_on_dataset(dataloaders['test'])
    visualizer.visualize_calibration(calibrated_res, visualizer.save_dir, save_hint="test_calibrated")