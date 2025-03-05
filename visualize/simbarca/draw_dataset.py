""" this is to check the dataset matches the samples, doesn't really need to check..
    cuz almost everything is done before creating the samples, the dataset literally does a collation
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from netsanut.data.datasets import SimBarca

with open("datasets/simbarca/metadata/sections_of_interest.txt", "r") as f:
    IDS_OF_INTEREST = [int(x) for x in f.read().split(",")]

test_set = SimBarca(split="test", force_reload=False, filter_short=None)
session_id_to_index = {v:k for k, v in test_set.index_to_section_id.items()}

with open("datasets/simbarca/metadata/train_test_split.json", "r") as f:
    session_split = json.load(f)
session_order = [0, 7, 14, 21]
sample_files = ["datasets/simbarca/simulation_sessions/session_{:03}/timeseries/samples.npz".format(
    session_split['test'][i]
    ) for i in session_order]

sample_data = []
for f in sample_files:
    with np.load(f) as npz_data:
        data = {k: npz_data[k] for k in npz_data.keys()}
    sample_data.append(data)
    
for sec_id in IDS_OF_INTEREST:
    p = session_id_to_index[sec_id]
    all_labels = test_set.pred_speed[:, -1, p, 0]
    sec_clips = [data['pred_vdist'][:, -1, p, 0] / data['pred_vtime'][:, -1, p, 0] 
                 for data in sample_data]
    # session 75 starts from index 0, each session has 20 samples
    ts_clips = [np.arange(20)+x*20 for x in session_order]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(len(all_labels)), all_labels, label="Dataset")
    for ts, sec, session_id in zip(ts_clips, sec_clips, session_order):
        ax.plot(ts, sec, label="Samples_{}".format(session_id))
    ax.set_title("The labels for 30-min-ahead Predictions of Section ID: {}".format(sec_id))
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Speed (m/s)")
    ax.legend()
    fig.tight_layout()
    fig.savefig("datasets/simbarca/figures/dataset_vs_sample_{}.pdf".format(sec_id))
    plt.close(fig)

