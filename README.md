## Netsanut: Networked Time Series Analysis for Urban Transportation

### Installation

```bash
python -m pip install -e .
```

### Dataset Preparation

METR-LA and PEMS-BAY can be downloaded from DCRNN

UTD19 can be downloaded from https://www.research-collection.ethz.ch/handle/20.500.11850/437802


TODO:

* [ ] complete the NetsFormer model
* [ ] Add a two stage trainer to train deterministic part first and then variance prediction
* [ ] add processing for UTD19
* [ ] adapt the lazy call system from  detectron2



Modified from [Traffic Transformer](https://github.com/R0oup1iao/Traffic-Transformer).

How it works:

1. process fixed time dimension with LSTM and take the last memory
2. Use adjacency matrix as decoder mask in transformer
