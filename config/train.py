train = {
    "max_epoch": 50,
    "output_dir": "scratch/debug", 
    "checkpoint": "",
    "device": "cuda",
    # select model by validation set performance, and test the best model
    # will test the final model if set to False
    "best_metric": None,
    "test_best_ckpt": False, 
    "grad_clip": 3.0,
    "eval_train": False, # run evaluation on train set after each epoch
    "eval_period": 1, # run evaluation every n epochs
    "save_period": 5, # save checkpoint every n epochs
}