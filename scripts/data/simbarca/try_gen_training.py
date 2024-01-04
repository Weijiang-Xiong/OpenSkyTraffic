""" This script aggregate the per-time-step statistics (the results of `time_series_from_traj.py`) to different intervals.
    These aggregated values will represent different sensor modalities, and they will be used to create training data. 
"""

import json
import numpy as np 
import pandas as pd

from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
from functools import partial

PREDICT_INTERVAL = 180 # predict future values every 3 mins
DETECTOR_INTERVAL = 180 # detector readings are typically every 3~5 mins
DRONE_INTERVAL = 5
PROBE_INTERVAL = 30
SIM_START_TIME, SIM_TIME_STEP = "2005-05-10T07:45", 0.5

sec_cols = ['time_steps', 'total_dist', 'total_time', 'num_in', 'num_out']
ld_cols = ['time_steps', 'LD_count', 'LD_speed']
probe_cols = ['pv_time_steps', 'pv_dist', 'pv_time']

def get_modality_step(modality:str):
    
    """ calculate the number of time steps for each modality
    """
    
    match modality.split('_')[0]:
        case 'drone':
            modality_steps = DRONE_INTERVAL / SIM_TIME_STEP
        case 'ld':
            modality_steps = DETECTOR_INTERVAL / SIM_TIME_STEP
        case 'probe':
            modality_steps = PROBE_INTERVAL / SIM_TIME_STEP
        case 'pred':
            modality_steps = PREDICT_INTERVAL / SIM_TIME_STEP
        
    return int(modality_steps)

def agg_raw_to_intervals(item):
    
    """ Aggregate the raw statistics (per simulation time step) to different intervals as defined above.
        Input: the statistics of a section per simulation time step. 
        Output: aggregated statistics for different time windows  
            `sec_*` for drone data (section level)
            'ld_*' for loop detector data
            'pv_*' for probe vehicle data
    """
    
    section, sec_data = item
    if section == '10001':
        print() 
    sec_df = pd.DataFrame({k:v for k, v in sec_data.items() if k in sec_cols}, columns=sec_cols)
    # an alternative way is to use DataFrame.resample(), needs absolute time instead of time steps
    drone_groups = sec_df.groupby(sec_df['time_steps']//get_modality_step("drone_speed"))
    pred_groups = sec_df.groupby(sec_df['time_steps']//get_modality_step("pred_speed"))
    
    df_ld = pd.DataFrame({k:v for k, v in sec_data.items() if k in ld_cols}, columns=ld_cols)
    # delete the rows with zero LD_count, where the LD_speed is nan
    df_ld = df_ld[df_ld['LD_count'] > 0]
    ld_groups = df_ld.groupby(sec_df['time_steps']//get_modality_step("ld_speed"))
    
    # average speed by statistic time window (5s for link speed, 90s for detector speed)
    drone_speed = drone_groups['total_dist'].sum() / drone_groups['total_time'].sum()
    pred_speed = pred_groups['total_dist'].sum() / pred_groups['total_time'].sum()
    ld_speed = ld_groups['LD_speed'].sum()  / ld_groups['LD_count'].sum()

    # if drone_speed.size > 0 and (drone_speed < -1e-3).any():
    #     print('negative speed in section {}'.format(section))
    #     print(sec_df[sec_df['total_dist'] < 0])
    # # if there are any nan values
    # if drone_speed.isna().any() or ld_speed.isna().any():
    #     print('nan values in section {}'.format(section))
    #     print(drone_speed)
    #     print(ld_speed)
        
    return {'section': section, 
            'drone_speed': drone_speed, 
            'ld_speed': ld_speed, 
            'pred_speed': pred_speed,
            }


if __name__ == '__main__':
    
    file_path = 'datasets/simbarca/session_000/section_statistics.json'
    data = json.load(open(file_path, 'r'))
    start_time = np.datetime64(SIM_START_TIME)
    time_step_delta = np.timedelta64(int(SIM_TIME_STEP * 1e3), 'ms')
    
    with ProcessPoolExecutor(max_workers=8) as executor:
        # if the wrapped function requires any arguments, use functools.partial
        partial_func = partial(agg_raw_to_intervals)
        # process the data in parallel
        results = list(tqdm(executor.map(partial_func, data['statistics'].items()),
                            total=len(data['statistics'].items())))


    stats_all_sec = defaultdict(list)
    for modality in ['drone_speed', 'ld_speed', 'pred_speed']:
        # store the data as a DF with columns (time_step, section, value) put some process here if suitable
        section_data = []
        for r in results:
            df = pd.DataFrame({'time_step': r[modality].index, 
                               'value': r[modality].values})
            df['section'] = r['section']
            section_data.append(df)
        tuple_all_sec = pd.concat(section_data, ignore_index=True, copy=False)
        tuple_all_sec.sort_values(by=['time_step'], inplace=True)
        modality_steps = get_modality_step(modality)
        # add absolute time to the time steps 
        tuple_all_sec['time'] = start_time + (tuple_all_sec['time_step']+1) * modality_steps * time_step_delta

        # switch to a DF with time as rows, sections as columns, and values as the entries
        modality_data = tuple_all_sec.pivot(index='time', columns='section', values='value')
        # a nan means no vehicles are observed by that kind of sensor, and this is different 
        # from 0, which means the vehicles are stopped at red lights.
        modality_data.fillna(-1, inplace=True) 
        stats_all_sec[modality] = modality_data
        
    # now begin to construct X and Y for the samples, by modality. 
    # X: original records for the last 30 minutes
    