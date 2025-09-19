METR-LA and PEMS-Bay datasets are from the [DCRNN](https://github.com/liyaguang/DCRNN?tab=readme-ov-file#data-preparation) repo, instructions summarized here for convenience.

Traffic data can be downloaded from [Google Drive](https://drive.google.com/open?id=10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX) or [Baidu Yun](https://pan.baidu.com/s/14Yy9isAIZYdU__OYEQGa_g).
The metadata files for graph structure are available at [DCRNN](https://github.com/liyaguang/DCRNN/tree/master/data/sensor_graph) repo

These files should be put under `dataset` folder as follows

```
datasets
├── metr
│   ├── adj_mx_metr.pkl
│   ├── distances_la_2012.csv
│   ├── graph_sensor_ids.txt
│   ├── graph_sensor_locations.csv
│   ├── metr-la.h5
├── pems
│   ├── adj_mx_bay.pkl
│   ├── distances_bay_2017.csv
│   ├── graph_sensor_ids.txt # this is extracted from graph_sensor_locations_bay.csv
│   ├── graph_sensor_locations_bay.csv
│   ├── pems-bay.h5
```

To work with `.h5` files, we need to install pandas with certain [dependencies](https://pandas.pydata.org/docs/getting_started/install.html#dependencies). 
To save trouble, one can create a separate python environment and install pandas with all dependencies.
Ideally, the preprocessing only needs to be done once, and then this environment is no longer needed.

```
python -m pip install pandas[all]
```

Then create the training samples from raw data

```
# METR-LA
python preprocess/metr-style/generate_training_data.py --output_dir=datasets/metr --traffic_df_filename=data/metr-la.h5

# PEMS-BAY
python preprocess/metr-style/generate_training_data.py --output_dir=datasets/pems --traffic_df_filename=data/pems-bay.h5
```