### File Structure

The dataset files, either raw or processed, are placed under `datasets`, and each dataset should have its own folder (or the symbolic link to the actual storage folder).

The codes for data preprocessing and utilization are placed under `netsanut/data`.

### SimBarca

**It is better to create a separate conda environment for creating and processing SimBarca, as my code strictly depends on pandarallel and pandas 2.2.0. And they won't be used after the dataset is created.**

Many datasets use `.h5` files to store raw data, and preprocessing `.h5` files with pandas [requires PyTables](https://pandas.pydata.org/docs/getting_started/install.html#dependencies), one can install it individually or install all dependencies of pandas at once.

```
python -m pip install pandas[all]==2.2.0 pandarallel
```

You can check the MD5SUM of the npz file using `md5sum` command in linux. 

The MD5SUM for `train.npz` is `16f8293a9489a8466f1289b2819c469b`, and for `test.npz`, it is `c3e1002219dec74a9a06e816fd3f8df0`

### METR-LA and PEMS-Bay

The sensor graphs for METR-LA and PEMS-Bay can be downloaded from DCRNN repo

https://github.com/liyaguang/DCRNN/tree/master/data/sensor_graph

The download link for UTD19

https://www.research-collection.ethz.ch/handle/20.500.11850/437802

For the LargeST dataset, follow its own github repo https://github.com/liuxu77/LargeST
