import glob
import pickle

import seaborn
seaborn.set_style('darkgrid')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from typing import Dict
from multiprocessing import Pool

# length of all road segments
ALL_LENGTHS = pd.read_csv("datasets/simbarca/metadata/link_bboxes_clustered.csv")[['id', 'length']].set_index('id')
TOTAL_LENGTH = ALL_LENGTHS['length'].sum() / 1000

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
    plt.close(fig)
    
    fig, ax = plt.subplots(figsize=(5, 4))
    ld_speed = data['ld_speed']
    gt_speed = (vdist_3min / vtime_3min)
    speed_diff = (ld_speed - gt_speed).to_numpy().flatten()
    speed_diff = speed_diff[~np.isnan(speed_diff)] # remove NaN values
    ax.hist(speed_diff, bins=20)
    ax.set_xlabel('Speed Difference (m/s)')
    ax.set_ylabel('Frequency')
    ax.set_title('Speed Difference Distribution')
    fig.tight_layout()
    fig.savefig('{}/speed_diff.pdf'.format(folder))
    plt.close(fig)

    
    fig, ax = plt.subplots(figsize=(5, 4))
    avg_speed_diff = (ld_speed - gt_speed).mean(axis=0).to_numpy()
    avg_speed_diff = avg_speed_diff[~np.isnan(avg_speed_diff)] # remove NaN values
    ax.hist(avg_speed_diff, bins=20)    
    ax.set_xlabel('Speed Difference (m/s), averaged over time')
    ax.set_ylabel('Frequency')
    ax.set_title('Average Speed Difference Distribution by Location')
    fig.tight_layout()
    fig.savefig('{}/avg_speed_diff.pdf'.format(folder))
    plt.close(fig)

    # scatter plot with length as x and speed difference as y
    fig, ax = plt.subplots(figsize=(5, 4))
    valid_indexes = (ld_speed - gt_speed).mean(axis=0).notna()
    ax.scatter(ALL_LENGTHS[valid_indexes], avg_speed_diff, s=2)
    ax.set_xlabel('Road Segment Length (m)')
    ax.set_ylabel('Speed Difference (m/s)')
    ax.set_title('Speed Difference vs. Road Segment Length')
    fig.tight_layout()
    fig.savefig('{}/speed_diff_wrt_length.pdf'.format(folder))
    plt.close(fig)

    # plot the speed of one segment
    for sec_id in np.random.choice(data['ld_speed'].columns, 3, replace=False):
        fig, ax = plt.subplots(figsize=(5, 4))
        ld_speed_seg = data['ld_speed'][sec_id]
        gt_speed_seg = (vdist_3min / vtime_3min)[sec_id]
        ld_speed_seg.plot(ax=ax, label='LD Speed')
        gt_speed_seg.plot(ax=ax, label='GT Speed')
        ax.legend()
        ax.set_title("LD Speed vs. GT Speed")
        ax.set_xlabel("Time stamp")
        ax.set_ylabel("Speed (m/s)")
        ax.legend()
        fig.tight_layout()
        fig.savefig('{}/speed_comparison_{}.pdf'.format(folder, sec_id))
        plt.close(fig)
    
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
        ax.scatter(t2, d2, s=2)
        
    ax.set_xlabel('Vehicle Density')
    ax.set_ylabel('Vehicle Flow')
    ax.set_title('MFD')
    fig.tight_layout()
    fig.savefig('{}/all_MFD_combined.pdf'.format("datasets/simbarca/figures"))