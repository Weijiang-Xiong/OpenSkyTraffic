

Many datasets use `.h5` files to store raw data, and preprocessing `.h5` files with pandas [requires PyTables](https://pandas.pydata.org/docs/getting_started/install.html#dependencies), one can install it individually or install all dependencies of pandas at once

```
python -m pip install pandas[all]
```

The sensor graphs for METR-LA and PEMS-Bay can be downloaded from DCRNN repo

https://github.com/liyaguang/DCRNN/tree/master/data/sensor_graph

The download link for UTD19

https://www.research-collection.ethz.ch/handle/20.500.11850/437802

For the LargeST dataset, follow its own github repo https://github.com/liuxu77/LargeST
