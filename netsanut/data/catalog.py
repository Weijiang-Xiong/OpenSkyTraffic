from collections import defaultdict

DATA_ROOT = "datasets"
DATASET_CATALOG = defaultdict(dict)

METR_LA = {
    'train'    : 'metr/train.npz',
    'val'      : 'metr/val.npz',
    'test'     : 'metr/test.npz',
    'adjacency': 'metr/adj_mx_metr.pkl',
    'geo_locations': 'metr/graph_sensor_locations.csv'
}

PEMS_BAY = {
    'train'    : 'pems/train.npz',
    'val'      : 'pems/val.npz',
    'test'     : 'pems/test.npz',
    'adjacency': 'pems/adj_mx_bay.pkl',
    'geo_locations': 'pems/graph_sensor_locations_bay.csv'
}

DATASET_CATALOG['metr-la'] = METR_LA
DATASET_CATALOG['pems-bay'] = PEMS_BAY

# add DATA_ROOT to the split path
for dname, dpaths in DATASET_CATALOG.items():
    for split_name, split_path in dpaths.items():
        DATASET_CATALOG[dname][split_name] = "{}/{}".format(DATA_ROOT, split_path)
