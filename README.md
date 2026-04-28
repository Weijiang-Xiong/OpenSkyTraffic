# 🚥 SkyTraffic: Urban Traffic Monitoring and Prediction From the Sky

This repo provides the implementation of our work on drone-based urban traffic prediction and monitoring.

## 📰 News

🎉 **2026-4-17**: Codes and preprint for Gaussian Mixture Model prediction released ! 

🎉 **2025-8-8**: Paper accepted to IEEE T-ITS !

## 🛠️ Installation and dataset preparation

We assume there is a conda environment called `pytorch` with [PyTorch](https://pytorch.org/get-started/locally/) correctly installed.

```bash
git clone https://github.com/Weijiang-Xiong/OpenSkyTraffic.git
python -m pip install -e .
```

To prepare METR-LA and PEMS-Bay dataset, please refer to this [README.md](preprocess/metr-style/README.md).

For using the SimBarca dataset, one can simply download the processed version from [Hugging Face](https://huggingface.co/datasets/Greatriver/SimBarca) (**recommended**) or [Zenodo](https://zenodo.org/records/17159241) and skip the preparations below.

To prepare SimBarca dataset, follow this [README.md](preprocess/simbarca/README.md)

## 🚀 Basic Usage

The main entrance scripts are placed under `scripts`.

* `train.py` provides the training pipeline and evaluation
* `run_multiple.py` is a script to format and run multiple commands in sequence using `subprocess.run`

For example, the following command will run a training task with `PATH/TO/CFG/FILE.py`, and override the output folder to `scratch/SAVE_DIR`.

The actual configuration files can be found as python files under the [config](./config) folder, and the configuration files used in the papers are linked in the [Publications and Citation](#-publications-and-citation) section below.

```
python scripts/train.py --config-file PATH/TO/CFG/FILE.py train.output_dir=scratch/SAVE_DIR
```

After training, the results can be visualized using the following command

```
python scripts/train.py --eval-only --config-file scratch/SAVE_DIR/config.yaml evaluation.visualize=True
```

For a complete walk through of the training and visualization workflow, please look at the comments in the scripts.

## 📁 File Structure

```text
project_root
├── config : experiment configuration files
├── datasets : the dataset files (raw and processed) and metadata
│   ├── <dataset_1>
│   ├── <dataset_2>
├── skytraffic : codes
│   ├── config : configuration management
│   ├── data : dataset loading and transform
│   ├── engine : training loop
│   ├── evaluation: evaluation codes
│   ├── models : neural network models
│   ├── solver: optimizer and scheduler
│   └── utils: utility functions
├── preprocess : data preprocessing (before training)
├── scripts : train scripts
├── tests : unit tests
├── visualize : visualization for data and models
├── scratch : training logs and checkpoints
└── figures : visualization of data and results
```


## 📚 Publications and Citation

Weijiang Xiong, Robert Fonod, Alexandre Alahi and Nikolas Geroliminis, "Multi-Source Urban Traffic Flow Forecasting With Drone and Loop Detector Data," in IEEE Transactions on Intelligent Transportation Systems. 
[[Paper]](https://ieeexplore.ieee.org/document/11174070)
[[Exp. Config]](config/himsnet/HiMSNet.py)


Weijiang Xiong, Robert Fonod and Nikolas Geroliminis, "Unveiling Stochasticity: Universal Multi-modal Probabilistic Modeling for Traffic Forecasting", ArXiv preprint 2026.
[[Preprint]](https://arxiv.org/abs/2604.16084)
[[Exp. Config]](config/gwnet/GWNET_GMM.py)

```
@ARTICLE{xiong2025himsnet,
  author={Xiong, Weijiang and Fonod, Robert and Alahi, Alexandre and Geroliminis, Nikolas},
  journal={IEEE Transactions on Intelligent Transportation Systems}, 
  title={Multi-Source Urban Traffic Flow Forecasting With Drone and Loop Detector Data}, 
  year={2025},
  volume={26},
  number={11},
  pages={18637-18652},
  doi={10.1109/TITS.2025.3605014}}
```
