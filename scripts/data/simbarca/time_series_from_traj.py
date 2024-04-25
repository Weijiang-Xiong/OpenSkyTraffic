""" This script generates some most basic traffic statistics at each simulation time step, 
    using the trajectory data (vehicle positions over time) from Aimsun
    
    The statistics are separately computed for each section, and include (e.g.):
        total distance traveled by all vehicles
        total time traveled by all vehicles
        number of vehicles passing the loop detector
    
    This script will be called by `sim_vehicle_traj.py`, please refer to that script for usage. 
"""
import re
import time
import json
import gzip
import glob
import argparse
import numpy as np
import pandas as pd

from copy import deepcopy
from typing import List, Dict, Tuple
from collections import defaultdict
from pandarallel import pandarallel
pandarallel.initialize(nb_workers=8, progress_bar=False)

# simulation starts from 7:45am with time step 0.5s
SIM_START_TIME, SIM_TIME_STEP = "2005-05-10T07:45", 0.5
# the main demand matrix in Aimsun has roughly 1e5 vehicles, which can be scaled by 1.5
# so here we choose a safe upper bound to randomly sample probe vehicles
MAX_NUM_VEHICLE = int(2e5)
CONNECTION_DTYPE = {'turn': int, 'org': int, 'dst': int, 'intersection': int, 'length': float}
LINK_DTYPE = {'id': int, 'from_x': float, 'from_y': float, 'to_x': float, 'to_y': float, 'length': float, 'out_ang': float, 'num_lanes': int}
VEHINFO_DTYPE = {'time': float, 'vehicle_id': int, 'speed': float, 'section': int, 'lane': int, 'junction': int, 'section_from': int, 'section_to': int, 'position': float, 'dist2end': float, 'total_dist': float, }
COLUMNS_FOR_TIME_SERIES = ['vehicle_id', 'section', 'time', 'position', 'speed', 'total_dist']

class Section:

    def __init__(self, road_id, length, num_lanes=None, 
                 lane_lengths=None, entrance_len=None, exit_len=None) -> None:

        self.id = int(road_id)
        self.length = length
        self.detector_place = 0.5 * length
        
        self.num_lanes = int(num_lanes)
        self.lane_start, self.lane_end, self.lane_length = [[] for _ in range(3)]
        self.set_lane_info(lane_lengths, entrance_len, exit_len)
        
        self.in_turns = []
        self.out_turns = []


        self.clips = None  # take the clips at this section from a TrajectoryClip
        
        self.clips = None  # take the clips at this section from a TrajectoryClip

    def set_detector_place(self, loc=0.5):
        self.detector_place = self.length * loc
        
    def set_lane_info(self, lane_lengths, entrance_len, exit_len):
        """ 
            These information are saved from the section objects in Aimsun, and their orders 
            are from left to right, so we need to reverse them to match the API at the end.
            For API, see the `numberLane` at https://docs.aimsun.com/next/22.0.2/UsersManual/ApiVehicleTracking.html#read-the-information-of-a-tracked-vehicle
            For the section class, see https://api.aimsun.com.br/classGKSection.html#getLane
            
            lane_lengths: a list of lane lengths, from left to right
            entrance_len: the length of the entrance lane, from left to right
            exit_len: the length of the exit lane, from left to right
            
            one may check the lane_end is equal to the position + dist2end of the vehicles in 
            a lane. 
        """
        lane_start = np.array(entrance_len) - min(entrance_len)
        lane_end = lane_start + np.array(lane_lengths)
        
        # API records count from right to left, so reverse it
        self.lane_start = lane_start[::-1].tolist() 
        self.lane_end = lane_end[::-1].tolist()
        self.lane_length = lane_lengths[::-1]
    
class Junction:

    def __init__(self, junction_id) -> None:
        self.id = junction_id
        self.turns = dict()
        self.orgs = set()
        self.dsts = set()

    def add_turn(self, org, dst, length):
        self.orgs.add(org)
        self.dsts.add(dst)
        self.turns["{}-{}".format(org, dst)] = length

    def get_turn_length(self, org, dst):
        return self.turns.get("{}-{}".format(org, dst), np.nan)


class RoadNetwork:

    """ this is a class to generate count vehicle number, average speed, etc. for each section
        and then summarize time series data for each section
    """

    def __init__(self, sections: Dict[str, Section], junctions: Dict[str, Junction]) -> None:
        self.sections: dict = sections
        self.junctions: dict = junctions
        
    def get_section(self, key):
        return self.sections[key]

    def get_junction(self, key):
        return self.junctions[key]

    def get_turn_length(self, junction, org, dst):
        return self.junctions[junction].get_turn_length(org, dst)


def get_statistics(df: pd.DataFrame, road_network=None, probe=False):
    """ compute the statistics for each time step

    Args:
        df (pd.DataFrame): data frame containing the trajectory data of all vehicles
        road_network (_type_): road network
        probe (bool, optional): if this dataframe is for probe vehicles. Defaults to False.
    """
    
    #########################################################
    # step 1: trajectory clip per time step, previous=>now  #
    #########################################################
    
    section = road_network.get_section(df['section'].iloc[0])
    # compose trajectory segment (start_time, end_time, start_position, end_position)
    vehicle_groups = df.groupby(['vehicle_id'])
    df['p_time'] = vehicle_groups['time'].shift(1)
    df['p_position'] = vehicle_groups['position'].shift(1)
    df['p_speed'] = vehicle_groups['speed'].shift(1)
    df['p_total_dist'] = vehicle_groups['total_dist'].shift(1)
    # drop the first time step of each vehicle, whose 'p_time' is nan
    df.drop(df[df['p_time'].isna()].index, inplace=True)
    
    # we will calculate traffic variables for each time step
    df['time_step'] = np.ceil(df['time'] / SIM_TIME_STEP).astype(int)
    time_groups = df.groupby('time_step')
    
    # calculate in-out first, as it is easy to identify in-out with 0 and -1
    df['in_flag'] = df['p_position'].eq(0)
    df['out_flag'] = df['position'].eq(-1)
    num_in = time_groups['in_flag'].sum()
    num_out = time_groups['out_flag'].sum()
    
    # assume constant speed and extend observed points to entry and exit points
    # use the speed and time to calculate the vehicle position for entry and exit
    # this will be used to determine if the vehicle passes the loop detector
    df.loc[df['in_flag']==True, 'p_position'] = df['position'] - df['speed'] * (df['time'] - df['p_time'])
    df.loc[df['out_flag']==True, 'position'] = df['p_position'] + df['p_speed'] * (df['time'] - df['p_time'])
    
    assert df['position'].eq(-1).sum()== 0
    
    # use total_dist to calculate vehicle distance traveled, df['dx'], because the position in section can have sudden changes due to lane change, which may result in negative distance traveled
    df.loc[df['in_flag']==True, 'p_total_dist'] = df['total_dist'] - df['speed'] * (df['time'] - df['p_time'])
    df.loc[df['out_flag']==True, 'total_dist'] = df['p_total_dist'] + df['p_speed'] * (df['time'] - df['p_time'])

    #########################################################
    # step 2: stats per time step, speed, flow, count  
    #########################################################

    df['dx'] = df['total_dist'] - df['p_total_dist']
    df['dt'] = df['time'] - df['p_time']
    # when a link is very short, it is possible that a vehicle can pass the link between two consecutive simulation time steps, and in these cases, dx will be `nan``, as we have no speed to extrapolate the distance. So we simply fill these nan values with the length of the section.
    df.loc[df['dx'].isna(), 'dx'] = section.length
    # up to now, dx can still be negative, because Aimsun has a very unrealistic lane change behavior.
    # a vehicle can magically float to another lane when it is already stopped and inside a queue.
    # this can make the vehicle move backward and cause negative dx, but this is very very rare. 
    # so we just set them to 0 
    df.loc[df['dx'] < 0, 'dx'] = 0
    assert (df['dx'].ge(0).all() and df['dt'].ge(0).all()) # dx and dt should always be positive
    # a flag wheter the vehicle passes the loop detector
    df['pass_ld'] = (df['position'] >= section.detector_place) & (df['p_position'] < section.detector_place)
    # keep the speed of vehicles passing the loop detector, set the speed of other vehicles to nan
    df['ld_spd'] = ((df['dx'] / df['dt']) * df['pass_ld']).replace(0, np.nan)
    
    num_vehicle = time_groups['vehicle_id'].nunique()
    total_dist = time_groups['dx'].sum()
    total_time = time_groups['dt'].sum()
    LD_count = time_groups['pass_ld'].sum()
    # groupby.mean() always ignores nan, that's the reason we replace 0 with nan above
    # note that the LD_speed will still be nan if LD_count is 0
    # https://pandas.pydata.org/docs/reference/api/pandas.core.groupby.DataFrameGroupBy.mean.html
    LD_speed = time_groups['ld_spd'].mean() 
    
    #########################################################
    # step 3: organize results 
    #########################################################
    
    time_steps = list(time_groups.groups.keys())
    # the grouby.apply above returns empty dataframe (not series) if `df` is empty, and 
    # .tolist() will give an error. So we need to check if the dataframe/series is empty.
    # this can happen after the time with demand when there is no vehicle in some sections
    if not probe:
        return {"time_steps": time_steps,
                "num_vehicle": num_vehicle.to_list() if num_vehicle.size > 0 else [],
                "total_dist": total_dist.tolist() if total_dist.size > 0 else [], 
                "total_time": total_time.tolist() if total_time.size > 0 else [], 
                "num_in": num_in.tolist() if num_in.size > 0 else [],
                "num_out": num_out.tolist() if num_out.size > 0 else [],
                "LD_count": LD_count.tolist() if LD_count.size > 0 else [], 
                "LD_speed": LD_speed.tolist() if LD_speed.size > 0 else []}
    else:
        return {"pv_time_steps": time_steps, # "pv" stands for "probe vehicle"
                "pv_dist": total_dist.tolist() if total_dist.size > 0 else [], 
                "pv_time": total_time.tolist() if total_time.size > 0 else []}
    

def find_num_vehicle(session_folder):
    log_file = "{}/aimsun_log.log".format(session_folder)
    log_content = open(log_file, "r").read()
    try: # extract this number from log file
        num_vehicle = int(re.findall(r"Number of generated trips: (\d+)", log_content)[0])
    except:
        num_vehicle = MAX_NUM_VEHICLE
    return num_vehicle 

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Generate time series data from trajectory data")
    parser.add_argument('--metadata_folder', type=str, default='datasets/simbarca/metadata', help='Path to metadata folder')
    parser.add_argument('--session_folder', type=str, default='datasets/simbarca/simulation_sessions/session_000', help='Path to session folder')
    parser.add_argument('--penetration_rate', type=float, default=0.05, help='Penetration rate of probe vehicles')
    args = parser.parse_args()
    
    #########################################################
    # load metadata, and construct a graph for road network #
    #########################################################
    
    connections = pd.read_csv("{}/connections.csv".format(args.metadata_folder), dtype=CONNECTION_DTYPE)
    link_bboxes = pd.read_csv("{}/link_bboxes.csv".format(args.metadata_folder), dtype=LINK_DTYPE)
    with open("{}/intersec_polygon.json".format(args.metadata_folder), "r") as f:
        intersection_polygon = json.load(f)
    with open("{}/lane_info.json".format(args.metadata_folder), "r") as f:
        lane_info = json.load(f)
    
    # construct a graph from the files
    sections = {row.id: Section(row.id, 
                                row.length, 
                                row.num_lanes, 
                                lane_info['lane_lengths'][str(row.id)],
                                lane_info['entrance_len'][str(row.id)],
                                lane_info['exit_len'][str(row.id)])
                for row in link_bboxes.itertuples()}
    
    junctions = {junction_id: Junction(junction_id) for junction_id in set(connections['intersection'])}
    for row in connections.itertuples(): # iterrows does not preserve data type
        junctions[row.intersection].add_turn(row.org, row.dst, row.length) 
        sections[row.org].out_turns.append(row.dst)
        sections[row.dst].in_turns.append(row.org)

    # check by comparing with Aimsun Next Model
    road_network = RoadNetwork(sections, junctions)
    
    #########################################################
    #########################################################
    
    #########################################################
    # loop over the saved file and process them one by one  #
    #########################################################
    
    # select a subset of vehicles to be probe vehicles
    with open("{}/settings.json".format(args.session_folder), "r") as f:
        settings = json.load(f)
        rng = np.random.default_rng(settings['random_seed'])
        num_vehicle = find_num_vehicle(args.session_folder)
        selected_vehicles = rng.choice(range(num_vehicle), 
                                       size=int(args.penetration_rate*num_vehicle), 
                                       replace=False)
        print("\n Selecting {}% vehicles from {} vehicles".format(args.penetration_rate*100, num_vehicle))
        
    # get the names of trajectory files
    data_files = sorted(glob.glob("{}/trajectory/*.json.gz".format(args.session_folder)))
    # these files are organized chronologically by time step, so keep the last time step of
    # this file as the init_time_step for the next file
    previous_time_step: pd.DataFrame = None
    # vehicle distance and time traveled by section: section -> series_name -> series_data
    per_section_stat: Dict[str, Dict[str, List]] = defaultdict(lambda: defaultdict(list)) 
    # process files one by one, as reading multiple files using multiple threads won't reduce
    # the data reading time (bottleneck is the disk), and the memory consumption is higher
    for file_name in data_files:
        
        print("Working on file {}".format(file_name))

        start_time = time.perf_counter()
        with gzip.open(file_name, "rt") as f:
            data = json.load(f)
        print("Loading a file takes {:.2f}s".format(time.perf_counter() - start_time))
        
        # concatenate the last time step with the current data frame
        start_time = time.perf_counter()
        if previous_time_step is not None:
            df = pd.concat([previous_time_step,
                            pd.DataFrame(data=data['trajectory'], columns=data['info_columns'])],
                           copy=False, ignore_index=True)
        else:
            df = pd.DataFrame(data=data['trajectory'], columns=data['info_columns'])
            # check the init_time_step['time'] is the same as the number in the file name (the time step when this file is saved)
            previous_time_step= deepcopy(df.groupby('time').get_group(df['time'].max()))
        
        entering = pd.DataFrame(data=data['entering'], columns=data['inout_columns'])
        exiting = pd.DataFrame(data=data['exiting'], columns=data['inout_columns'])
        print("Creating data frames takes {:.2f}s".format(time.perf_counter() - start_time))

        start_time = time.perf_counter()
        df = df[COLUMNS_FOR_TIME_SERIES]
        df = df[df['section'] != -1] # section==-1 means the vehicle is in a junction, so ignore it
        # we replace the position of entering and exiting vehicles with 0 and -1 respectively, but 
        # both of them are not valid because Aimsun handles the entering and exiting by queues.
        # when a queue of vehicles enters a section, all vehicles are assigned to the section as the 
        # first vehicle is in. At this time, the 'position' of subsequent vehicles can be negative values
        # our solution is to extrapolate the entry position using the closest position inside the section,
        # in order to keep the speed consistent.
        entering['position'] = 0
        exiting['position'] = -1
        df = pd.concat([df, entering, exiting], ignore_index=True, copy=False)
        # reduce lane index by 1, as the records count from 1 in Aimsun. 
        # lane info can be helpful for debugging, but we don't need it afterwards
        # df['lane'] = (df['lane'] - 1).fillna(-1)
        # this speed will be used to extrapolate the entry and exit position, but we should avoid 
        # using it for other purposes because in reality, we can not directly measure the speed of vehicles
        # from drone images, instead, we need to use distance/time 
        df['speed'] = df['speed'] / 3.6 # convert speed from km/h to m/s
        
        df.sort_values(by=['vehicle_id', 'time'], inplace=True)
        df = df.astype({k:v for k, v in VEHINFO_DTYPE.items() if k in df.columns}, copy=False)
        print("Filling data types and values takes {:.2f}s".format(time.perf_counter() - start_time))
        
        # pandarallel uses process-based parallelism, so the child process will fork a copy 
        # of the address space, modifications to `road_network` here will not be available 
        # in the parent process.
        # The OS usually use "copy on write" to prevent unnecessary copying of memory, so the 
        # data will not be copied until modification happens (same principle applies to pandas).
        start_time = time.perf_counter()
        av_ts_in_file = df.groupby('section').parallel_apply(get_statistics, road_network=road_network, probe=False)
        probe_vehicle_df = df[df['vehicle_id'].isin(selected_vehicles)]
        pv_ts_in_file = probe_vehicle_df.groupby('section').parallel_apply(get_statistics,road_network=road_network, probe=True)
        print("\n Processing a file takes {:.2f}s".format(time.perf_counter() - start_time))
        
        # concatenate the time series with previous file
        start_time = time.perf_counter()
        for ts_in_file in [av_ts_in_file, pv_ts_in_file]:
            for section, contents in ts_in_file.items():
                for series_name, series_data in contents.items():
                    per_section_stat[section][series_name].extend(series_data)
        print("Concatenating time series takes {:.2f}s".format(time.perf_counter() - start_time))
                
                
    # save the time series data
    start_time = time.perf_counter()
    with open("{}/timeseries/section_statistics.json".format(args.session_folder), "w") as f:
        json.dump({"sim_start_time": SIM_START_TIME, 
                   "sim_time_step_second": SIM_TIME_STEP,
                   "statistics": per_section_stat}, f)
    print("Saving all the time series takes {:.2f}s".format(time.perf_counter() - start_time))
        