import json
import numpy as np 
import pandas as pd

from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

DETECTOR_INTERVAL = 90
LINK_SPEED_INTERVAL = 5
SPEED_LIMIT = 50 # m/s, this is clearly too much for a car velocity
sec_cols = ['time_steps', 'total_dist', 'total_time', 'num_in', 'num_out']
ld_cols = ['time_steps', 'LD_count', 'LD_speed']
probe_cols = ['pv_time_steps', 'pv_dist', 'pv_time']

def get_link_speed(df):
    # use total_time because total_dist can be zero when vehicles are stopped 
    df_non_zero = df[df['total_time']!=0]
    if df_non_zero.size == 0:
        return 0
    else:
        speed = df_non_zero['total_dist'].sum() / df_non_zero['total_time'].sum()
        return speed

def get_ld_speed(df):
    df_non_zero = df[df['LD_count']!=0]
    if df_non_zero.size == 0:
        return 0
    else:
        speed = df_non_zero['LD_speed'].sum() / df_non_zero['LD_count'].sum()
        return speed

def process_section(item):
    sec, sec_data = item
    sec_df = pd.DataFrame({k:v for k, v in sec_data.items() if k in sec_cols}, columns=sec_cols)
    sec_groups = sec_df.groupby(sec_df['time_steps']//drone_steps)
    df_ld = pd.DataFrame({k:v for k, v in sec_data.items() if k in ld_cols}, columns=ld_cols)
    ld_groups = df_ld.groupby(sec_df['time_steps']//detector_steps)
    
    # take the mean of the non-zero elements in the groups
    sec_speed = sec_groups['total_dist'].sum() / sec_groups['total_time'].sum()
    # sec_in_flow = sec_groups['num_in'].sum() / (LINK_SPEED_INTERVAL/60)
    # sec_out_flow = sec_groups['num_out'].sum() / (LINK_SPEED_INTERVAL/60)
    ld_speed = ld_groups['LD_speed'].sum()  / ld_groups['LD_count'].sum()
    # ld_flow = ld_groups['LD_count'].sum() / (DETECTOR_INTERVAL/60)

    if sec_speed.size > 0 and (sec_speed < -1e-3).any():
        print('negative speed in section {}'.format(sec))
        print(sec_df[sec_df['total_dist'] < 0])
    
    return {'sec': sec, 
            'sec_speed': sec_speed, 
            # 'sec_in_flow': sec_in_flow, 
            # 'sec_out_flow': sec_out_flow,
            'ld_speed': ld_speed, 
            # 'ld_flow': ld_flow
            }


file_path = 'datasets/simbarca/session_000/vehicle_dist_time.json'
data = json.load(open(file_path, 'r'))
start_time = np.datetime64(data['sim_start_time'])
time_step = np.timedelta64(int(1000*data['sim_time_step_second']), 'ms')
detector_steps = int(np.timedelta64(DETECTOR_INTERVAL, 's') / time_step)
drone_steps = int(np.timedelta64(LINK_SPEED_INTERVAL, 's') / time_step)

with ProcessPoolExecutor(max_workers=16) as executor:
    results = list(tqdm(executor.map(process_section, data['vehicle_dist_time'].items()),
                        total=len(data['vehicle_dist_time'].items())))


ld_speed_all_sec = pd.DataFrame({r['sec']:r['ld_speed'] for r in results})
sec_speed_all_sec = pd.DataFrame({r['sec']:r['sec_speed'] for r in results})
# in_flow_all_sec = pd.DataFrame({r['sec']:r['sec_in_flow'] for r in results})
# ld_flow_all_sec = pd.DataFrame({r['sec']:r['ld_flow'] for r in results})
# out_flow_all_sec = pd.DataFrame({r['sec']:r['sec_out_flow'] for r in results})

# now begin to construct X and Y for the samples.
# X: original records for the last 30 minutes