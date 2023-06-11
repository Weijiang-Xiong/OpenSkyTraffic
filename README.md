# Netsanut: Networked Time Series Analysis for Urban Transportation

TODO:

- [ ] try to adapt the lazy call system from  detectron2
- [ ] improve documentation



Modified from [Traffic Transformer](https://github.com/R0oup1iao/Traffic-Transformer).

How it works:

1. process fixed time dimension with LSTM and take the last memory
2. Use adjacency matrix as decoder mask in transformer
