""" 
    This script aggregate the per-time-step statistics (the results of `time_series_from_traj.py`) to different time intervals. The spatial dimension is kept untouched, which may be aggregated again in the dataset class.
    
    These aggregated values will represent different sensor modalities, and they will be used to create training data. 
"""

import json
import pickle
import argparse
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
    
    """ calculate the number of time steps for each modality, the original data (per SIM_TIME_STEP) will
        be divided to non-overlapping intervals of this length and then aggregated
    """
    
    # see https://docs.python.org/3/tutorial/controlflow.html#match-statements
    match modality.split('_')[0]:
        case 'drone':
            modality_steps = DRONE_INTERVAL / SIM_TIME_STEP
        case 'ld':
            modality_steps = DETECTOR_INTERVAL / SIM_TIME_STEP
        case 'probe':
            modality_steps = PROBE_INTERVAL / SIM_TIME_STEP
        case 'pred':
            modality_steps = PREDICT_INTERVAL / SIM_TIME_STEP
        case _:
            raise ValueError("Unknown modality {}".format(modality))
        
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
    sec_df = pd.DataFrame({k:v for k, v in sec_data.items() if k in sec_cols}, columns=sec_cols)
    # an alternative way is to use DataFrame.resample(), needs absolute time instead of time steps
    drone_groups = sec_df.groupby(sec_df['time_steps']//get_modality_step("drone_speed"))
    pred_groups = sec_df.groupby(sec_df['time_steps']//get_modality_step("pred_speed"))
    
    df_ld = pd.DataFrame({k:v for k, v in sec_data.items() if k in ld_cols}, columns=ld_cols)
    # delete the rows with zero LD_count, where the LD_speed is nan
    df_ld = df_ld[df_ld['LD_count'] > 0]
    # weight the speed with the number of vehicles (then we can divide by vehicle count per group)
    df_ld['LD_spd_mul_cnt'] = df_ld['LD_speed'] * df_ld['LD_count']
    ld_groups = df_ld.groupby(sec_df['time_steps']//get_modality_step("ld_speed"))
    
    # average speed by statistic time window 
    # 5s for vehicle travel distance and time, 90s for detector speed
    drone_vdist = drone_groups['total_dist'].sum() 
    drone_vtime = drone_groups['total_time'].sum()
    pred_vdist = pred_groups['total_dist'].sum()
    pred_vtime = pred_groups['total_time'].sum()
    ld_speed = ld_groups['LD_spd_mul_cnt'].sum() / ld_groups['LD_count'].sum()

    return {'section': section, 
            'drone_vdist': drone_vdist,
            'drone_vtime': drone_vtime, 
            'ld_speed': ld_speed, 
            'pred_vdist': pred_vdist,
            'pred_vtime': pred_vtime
            }

def dataframe_to_array(df: pd.DataFrame, add_time_in_day=True):
    
    time_in_day = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
    time_in_day_2d = np.tile(time_in_day.reshape(-1, 1), [1, len(df.columns)])
    values_with_time = np.concatenate([np.expand_dims(df.to_numpy(), -1), 
                                       np.expand_dims(time_in_day_2d, -1)], 
                                      axis=-1)
    return values_with_time

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Aggregate the raw statistics to different intervals.')
    parser.add_argument('--session', type=str, default='datasets/simbarca/session_000/', help='path to session folder')
    args = parser.parse_args()
    
    file_path = '{}/timeseries/section_statistics.json'.format(args.session)
    data = json.load(open(file_path, 'r'))
    start_time = np.datetime64(SIM_START_TIME)
    sim_time_step = np.timedelta64(int(SIM_TIME_STEP * 1e3), 'ms')
    
    print("Aggregating the raw statistics to different intervals...")
    with ProcessPoolExecutor(max_workers=8) as executor:
        # if the wrapped function requires any arguments, use functools.partial
        partial_func = partial(agg_raw_to_intervals)
        # process the data in parallel
        results = list(tqdm(executor.map(partial_func, data['statistics'].items()),
                            total=len(data['statistics'].items())))
        
    section_ids_sorted = np.sort(pd.read_csv("datasets/simbarca/metadata/link_bboxes.csv")['id'].to_numpy())
    
    print("Constructing time series for each modality ...")
    stats_all_sec = defaultdict(list)
    for modality in ['drone_vdist', 'drone_vtime', 'ld_speed', 'pred_vdist', 'pred_vtime']: # pred_speed
        # store the data as a DF with columns (time_step, section, value) 
        # put some process here if suitable
        section_data = []
        for r in results:
            df = pd.DataFrame({'time_step': r[modality].index, 'value': r[modality].values})
            df['section'] = int(r['section'])
            section_data.append(df)
        all_secs = pd.concat(section_data, ignore_index=True, copy=False)
        all_secs.sort_values(by=['time_step'], inplace=True)
        modality_steps = get_modality_step(modality)
        # add absolute time to the time steps 
        all_secs['time'] = start_time + (all_secs['time_step']+1) * modality_steps * sim_time_step

        # switch to a DF with time as rows, sections as columns, and values as the entries
        modality_data = all_secs.pivot(index='time', columns='section', values='value')
        # set columns to integers, and sort the columns by section id
        modality_data.columns = modality_data.columns.astype(int)
        # some sections may never have a vehicle, so there are no recorded trajectory data for them
        columns_without_vehicles = np.setdiff1d(section_ids_sorted, modality_data.columns)
        modality_data[columns_without_vehicles] = np.nan
        modality_data = modality_data[section_ids_sorted]
        # we keep the nan values for the timesteps when no vehicles are present
        # this will be handled in the dataset class
        # modality_data.fillna(0, inplace=True)
        stats_all_sec[modality] = modality_data
    
    # save this, you can plot MFD with it 
    save_file = "{}/timeseries/agg_timeseries.pkl".format(args.session)
    print("Saving aggregated time series to {}".format(save_file))
    with open(save_file, "wb") as f:
        pickle.dump(stats_all_sec, f)
        
    # using the sliding window approach, and take 30 mins for input and 30 mins for prediction
    print("Extracting samples from time series ...")
    samples = [] 
    t0 = start_time + np.timedelta64(15, 'm') # the simulation has 15 mins warm-up time
    t1 = t0 + np.timedelta64(30, 'm')
    t2 = t1 + np.timedelta64(30, 'm')
    dt = np.timedelta64(3, 'm') # shift time window every 3 mins
    # the vehicles are generated for 75 mins, no vehicle will enter the system after that
    # we suppose the data is valid for 120 mins after warmup (45 mins after demand ends)
    # e.g., if we take 30 min, predict next 30 min with 3 min steps, we have (120-60) / 5 samples 
    num_samples = int((np.timedelta64(120, 'm') - (t2-t0)) / dt)
    
    for offset in range(num_samples):
        t_s = t0 + dt * offset # start 
        t_m = t1 + dt * offset # middle
        t_e = t2 + dt * offset # end
        # the time steps for the data in the input time window (t_s, t_m)
        drone_ts_in = (stats_all_sec['drone_vdist'].index>t_s) & (stats_all_sec['drone_vdist'].index<=t_m)
        ld_ts_in = (stats_all_sec['ld_speed'].index>t_s) & (stats_all_sec['ld_speed'].index<=t_m)
        # the time steps for the data in the prediction time window (t_m, t_e )
        pred_ts_out = (stats_all_sec['pred_vdist'].index>t_m) & (stats_all_sec['pred_vdist'].index<=t_e)
        ld_ts_out = (stats_all_sec['ld_speed'].index>t_m) & (stats_all_sec['ld_speed'].index<=t_e)
        sample_dict = {
            "drone_vdist": dataframe_to_array(stats_all_sec['drone_vdist'].loc[drone_ts_in]),
            "drone_vtime": dataframe_to_array(stats_all_sec['drone_vtime'].loc[drone_ts_in]),
            "ld_speed": dataframe_to_array(stats_all_sec['ld_speed'].loc[ld_ts_in]),
            "pred_vdist": dataframe_to_array(stats_all_sec['pred_vdist'].loc[pred_ts_out]),
            "pred_vtime": dataframe_to_array(stats_all_sec['pred_vtime'].loc[pred_ts_out]),
            "pred_ld_speed": dataframe_to_array(stats_all_sec['ld_speed'].loc[ld_ts_out]),
        }
        samples.append(sample_dict)
    print('number of samples: {}'.format(len(samples)))
    
    print('Packing the samples into numpy arrays ...')
    # convert the samples to numpy arrays
    packed_samples = defaultdict(list)
    for sample in samples:
        for k, v in sample.items():
            # expand the array to (1, num_time_steps, num_sections, num_features)
            # the first dimension will be used as batch dimension
            packed_samples[k].append(np.expand_dims(v, 0)) 
    # concatenate the arrays
    for k, v in packed_samples.items():
        packed_samples[k] = np.concatenate(v, axis=0)
    
    # save the samples to a compressed npz file
    with open('{}/timeseries/samples.npz'.format(args.session), 'wb') as f:
        np.savez_compressed(f, **packed_samples)