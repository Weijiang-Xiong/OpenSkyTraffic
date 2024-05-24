""" this is to check the dataset matches the samples, doesn't really need to check..
    cuz almost everything is done before creating the samples, the dataset literally does a collation
"""
import numpy as np
import matplotlib.pyplot as plt
from netsanut.data import SimBarca

IDS_OF_INTEREST = [int(x) for x in open("datasets/simbarca/metadata/sections_of_interest.txt", "r").read().split(",")]

test_set = SimBarca(split="test", force_reload=False, filter_short=None)
session_id_to_index = {v:k for k, v in test_set.index_to_section_id.items()}
# session 0 ~ 74 are training sessions (75), 75 ~ 100 are testing sessions (26)
sessions_to_exam = list(range(75, 101, 10))
sample_files = ["datasets/simbarca/simulation_sessions/session_{:03}/timeseries/samples.npz".format(i) 
                for i in sessions_to_exam]

sample_data = []
for f in sample_files:
    compressed = np.load(f)
    data = {k: compressed[k] for k in compressed.keys()}
    sample_data.append(data)
    
for sec_id in IDS_OF_INTEREST:
    p = session_id_to_index[sec_id]
    all_labels = test_set.pred_speed[:, -1, p, 0]
    sec_clips = [data['pred_vdist'][:, -1, p, 0] / data['pred_vtime'][:, -1, p, 0] 
                 for data in sample_data]
    # session 75 starts from index 0, each session has 20 samples
    ts_clips = [np.arange(20)+(x-75)*20 for x in sessions_to_exam]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(len(all_labels)), all_labels, label="Dataset")
    for ts, sec, session_id in zip(ts_clips, sec_clips, sessions_to_exam):
        ax.plot(ts, sec, label="Samples_{}".format(session_id))
    ax.set_title("The labels for 30-min-ahead Predictions of Section ID: {}".format(sec_id))
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Speed (m/s)")
    ax.legend()
    fig.tight_layout()
    fig.savefig("datasets/simbarca/figures/dataset_vs_sample_{}.pdf".format(sec_id))

