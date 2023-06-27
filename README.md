## Netsanut: Networked Time Series Analysis for Urban Transportation

### Installation

We assume there is a conda environment called `pytorch` with [PyTorch](https://pytorch.org/get-started/locally/) correctly installed.

```bash
git clone https://github.com/Weijiang-Xiong/netsanut.git
python -m pip install -e .
```

### Dataset Preparation

Please follow the [readme file](datasets/README.md) in `datasets` folder.

### TODO

* [X] complete the NetsFormer model
* [X] Add a two stage trainer to train deterministic part first and then variance prediction
* [ ] visualization of prediction results
* [ ] adapt he data processing from [LargeST](https://github.com/liuxu77/LargeST)
* [ ] add processing for UTD19
* [ ] change the spatial positional encoding into longitude and latitude-based, like vision transformer
* [ ] adapt the lazy call system from  detectron2

### Reference

The structure and code framework (config, logging, trainer-hooks, etc.) follows [detectron2](https://github.com/facebookresearch/detectron2).

The model in `netsanut/models/ttnet.py` is modified from [Traffic Transformer](https://github.com/R0oup1iao/Traffic-Transformer).
