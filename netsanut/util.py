import os 
from collections.abc import Mapping

def make_dir_if_not_exist(path):
    if not os.path.exists(path):
        print("Creating directory: {}".format(path))
        os.makedirs(path)

def flatten_results_dict(results):
    """
    Expand a hierarchical dict of scalars into a flat dict of scalars.
    If results[k1][k2][k3] = v, the returned dict will have the entry
    {"k1_k2_k3": v}.

    Args:
        results (dict):
    """
    r = {}
    for k, v in results.items():
        if isinstance(v, Mapping):
            v = flatten_results_dict(v)
            for kk, vv in v.items():
                r[k + "_" + kk] = vv
        else:
            r[k] = v
    return r