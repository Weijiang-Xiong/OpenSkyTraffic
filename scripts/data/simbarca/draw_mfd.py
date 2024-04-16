import glob
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from typing import Dict
from multiprocessing import Pool

# total length of all road segments, in km
TOTAL_LENGTH = pd.read_csv("datasets/simbarca/metadata/link_bboxes_clustered.csv")['length'].sum() / 1000 

def draw_mfd(folder):
    
    file_path = "{}/timeseries/agg_timeseries.pkl".format(folder)
    
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    vdist_5s, vtime_5s = data['drone_vdist'], data['drone_vtime']
    vdist_3min, vtime_3min = data['pred_vdist'], data['pred_vtime']
    
    fig, ax = plt.subplots(figsize=(5, 4))
    d1 = vdist_5s.sum(axis=1).values / (5 * TOTAL_LENGTH) # normalize by time window
    t1 = vtime_5s.sum(axis=1).values / (5  * TOTAL_LENGTH)
    ax.plot(t1, d1, label='Every 5s')
    d2 = vdist_3min.sum(axis=1).values / (180 * TOTAL_LENGTH)
    t2 = vtime_3min.sum(axis=1).values / (180  * TOTAL_LENGTH)
    ax.plot(t2, d2, label='Every 3min')
    
    # annotate the data points with the time
    # but this doesn't look good, many texts will overlap
    for i in range(len(t2)):
        if i % 5 != 0:
            continue
        ax.annotate("{} min".format((i+1)*3), (t2[i], d2[i]), textcoords="offset points", xytext=(0,10), ha='center')
    
    ax.set_xlabel('Vehicle Density (veh/km)')
    ax.set_ylabel('Vehicle Flow (veh/s)')
    ax.legend()
    ax.set_title('MFD')
    fig.tight_layout()
    fig.savefig('{}/MFD.pdf'.format(folder))
    
    return data

if __name__ == "__main__":
    
    folder_pattern =  "datasets/simbarca/simulation_sessions/session_*"
    all_folders = sorted(glob.glob(folder_pattern))
    
    # add TQDM for progress bar
    with Pool(processes=8) as pool:
        all_data = list(tqdm(pool.imap(draw_mfd, all_folders), total=len(all_folders)))
    
    # now we take all the MFDs and plot them together
    print("Plotting all MFDs together in one figure")
    fig, ax = plt.subplots(figsize=(5, 4))
    for data in all_data:
        vdist_3min, vtime_3min = data['pred_vdist'], data['pred_vtime']
        d2 = vdist_3min.sum(axis=1).values / (180 * TOTAL_LENGTH)
        t2 = vtime_3min.sum(axis=1).values / (180 * TOTAL_LENGTH)
        ax.plot(t2, d2)
        
    ax.set_xlabel('Vehicle Density')
    ax.set_ylabel('Vehicle Flow')
    ax.legend()
    ax.set_title('MFD')
    fig.tight_layout()
    fig.savefig('{}/all_MFD_combined.pdf'.format("datasets/simbarca/figures"))