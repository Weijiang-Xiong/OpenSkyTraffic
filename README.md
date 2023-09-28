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

The main entrance scripts are placed under `tools`.

* `train.py` provides the training pipeline and evaluation
* `visualize.py` provides visualization about the model calibration and predictions
* `run_multiple.py` is a script to format and run multiple commands in sequence using `subprocess.run`

For example, the following command will train a model using the specifications in `config/NeTSFormer_uncertainty.py` , and override the output dir to `results/debug`.

```
python tools/train.py --config-file config/NeTSFormer_uncertainty.py train.output_dir=results/debug
```

After training, the results can be analysed

```
python tools/visualize.py --result-dir results/debug
```

For a complete walk through of the training and visualization workflow, please look at the comments in the scripts.

### TODO

* [ ] rewrite the data pipeline, and docs for dataset
* [ ] separate functionalities of calibration and visualization
* [ ] adapt he data processing from [LargeST](https://github.com/liuxu77/LargeST)
* [ ] add processing for UTD19
* [ ] change the spatial positional encoding into longitude and latitude-based, like vision transformer
* [ ] adapt the lazy call system from  detectron2

### Reference

The structure and code framework (config, logging, trainer-hooks, etc.) follows [detectron2](https://github.com/facebookresearch/detectron2).

The model in `netsanut/models/ttnet.py` is modified from [Traffic Transformer](https://github.com/R0oup1iao/Traffic-Transformer).
