## Netsanut: Networked Time Series Analysis for Urban Transportation

In a modern transporation network, various sensors can be installed on roads and vehicles to collect traffic information of a specific location, whose record will be a time series. Meanwhile, different locations are connected via the network, which results in spatial relations among the time series. We coin this data representation as Networked Time Series (NeTS), and his package aims to provide a framework for training deep learning models for Networked Time Series Analysis in the domain of Urban Transportation.

### Installation and dataset preparation

We assume there is a conda environment called `pytorch` with [PyTorch](https://pytorch.org/get-started/locally/) correctly installed.

```bash
git clone https://github.com/Weijiang-Xiong/netsanut.git
python -m pip install -e .
```

For the preprocessing and usage of datasets, please follow the [readme file](datasets/README.md) in `datasets` folder.

### Basic Usage

The main entrance scripts are placed under `scripts`.

* `train.py` provides the training pipeline and evaluation
* `run_multiple.py` is a script to format and run multiple commands in sequence using `subprocess.run`

For example, the following command will train a model using the specifications in `config/HiMSNet.py` , and override the output dir to `scratch/himsnet_5hop`.

```
python scripts/train.py --config-file config/HiMSNet.py model.adjacency_hop=5 train.output_dir=scratch/himsnet_5hop
```

After training, the results can be visualized using the following command

```
python scripts/train.py --eval-only --config-file scratch/himsnet_5hop/config.py evaluation.visualize=True
```

For a complete walk through of the training and visualization workflow, please look at the comments in the scripts.

### File Structure

```text
project_root
├── config : experiment configuration files
├── datasets : the dataset files (raw and processed) and metadata
├── netsanut : codes
│   ├── config : configuration management
│   ├── data : dataset loading and transform
│   ├── engine : training loop
│   ├── evaluation
│   ├── models
│   ├── solver
│   └── utils
├── preprocess : data preprocessing (before training)
├── scratch : training results
├── scripts : train scripts
├── tests : unit tests
└── visualize : visualization for data and models
```