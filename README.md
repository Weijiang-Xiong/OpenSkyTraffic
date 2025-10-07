## SkyTraffic: Urban Traffic Monitoring From the Sky

This repo aims to present a solution for drone-based urban traffic monitoring.

### Installation and dataset preparation

We assume there is a conda environment called `pytorch` with [PyTorch](https://pytorch.org/get-started/locally/) correctly installed.

```bash
git clone https://github.com/Weijiang-Xiong/OpenSkyTraffic.git
python -m pip install -e .
```

To prepare METR-LA and PEMS-Bay dataset, please refer to this [README.md](preprocess/metr-style/README.md).

To prepare SimBarca dataset, follow this [README.md](preprocess/simbarca/README.md)
For using the SimBarca dataset, one can simply download the processed version from [Zenodo](https://zenodo.org/records/17159241) and skip the preparations. 

The file structure should look like
```
datasets
├── metr 
│   ├── test.npz
│   ├── train.npz
│   └── val.npz
├── pems
└── simbarca 
    ├── metadata
    └── simulation_sessions
```

### Basic Usage

The main entrance scripts are placed under `scripts`.

* `train.py` provides the training pipeline and evaluation
* `run_multiple.py` is a script to format and run multiple commands in sequence using `subprocess.run`

For example, the following command will run a training task with `PATH/TO/CFG/FILE.py`, and override the output folder to `scratch/SAVE_DIR`.

The actal configuration files can be found as python files under the [config](./config) folder. 

```
python scripts/train.py --config-file PATH/TO/CFG/FILE.py train.output_dir=scratch/SAVE_DIR
```

After training, the results can be visualized using the following command

```
python scripts/train.py --eval-only --config-file scratch/SAVE_DIR/config.py evaluation.visualize=True
```

For a complete walk through of the training and visualization workflow, please look at the comments in the scripts.

### Project Structure

```text
project_root
├── config : experiment configuration files
├── datasets : the dataset files (raw and processed) and metadata
├── skytraffic : codes
│   ├── config : configuration management
│   ├── data : dataset loading and transform
│   ├── engine : training loop
│   ├── evaluation: evaluation codes
│   ├── models : neural network models
│   ├── solver: optimizer and scheduler
│   └── utils: utility functions
├── preprocess : data preprocessing (before training)
├── scratch : training results
├── scripts : train scripts
├── tests : unit tests
└── visualize : visualization for data and models
```