""" This script draws the time that vehicles spend traveling through the network (i.e., the time from entry to the network until exit).
"""
import os 
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

def network_travel_time_histogram(folder: str):
    IO_time_data = pickle.load(open("{}/timeseries/network_in_out.pkl".format(folder), "rb"))
    entrance_time = pd.DataFrame(IO_time_data['entrance'], columns=["vehicle_id", "in_section_id", "in_time"])
    exit_time = pd.DataFrame(IO_time_data['exit'], columns=["vehicle_id", "out_section_id", "out_time"])
    # join the two dataframes by matching vehicle ID
    traverse_df = pd.merge(entrance_time, exit_time, on="vehicle_id")
    traverse_df['travel_time'] = (traverse_df['out_time'] - traverse_df['in_time']) / 60
    
    # plot the histogram of travel time
    tt = traverse_df['travel_time']
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(tt, bins=25)
    ax.set_xlabel('Travel Time (min) ')
    ax.set_ylabel('Num of Vehicles')
    ax.set_title('Network Travel Time (max {:.1f} mean {:.1f})'.format(tt.max(), tt.mean()))
    fig.tight_layout()
    fig.savefig("{}/figures/travel_time_histogram.pdf".format(folder))

if __name__ == "__main__":
    all_folders = sorted(glob.glob("datasets/simbarca/simulation_sessions/session_*"))
    # add TQDM for progress bar
    with Pool(processes=8) as pool:
        all_data = list(tqdm(pool.imap(network_travel_time_histogram, all_folders), total=len(all_folders)))