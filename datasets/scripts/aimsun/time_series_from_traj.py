import time
import json
import glob
import argparse
import numpy as np
import pandas as pd

from copy import deepcopy
from typing import List, Dict, Tuple
from collections import defaultdict
from pandarallel import pandarallel
pandarallel.initialize(nb_workers=32, progress_bar=True)

SIM_START_TIME = np.datetime64("2005-05-10T07:45")
SIM_TIME_STEP = 0.5  # simulation time step is set to be 0.5s in Aimsun
# the main demand matrix in Aimsun has roughly 1e5 vehicles, which can be scaled by 1.5
# so here we choose a safe upper bound to randomly sample probe vehicles
MAX_NUM_VEHICLE = int(2e5)
CONNECTION_DTYPE = {'turn': int, 'org': int, 'dst': int, 'intersection': int, 'length': float}
LINK_DTYPE = {'id': int, 'from_x': float, 'from_y': float, 'to_x': float, 'to_y': float, 'length': float, 'out_ang': float, 'num_lanes': int}
VEHINFO_DTYPE = {'time': float, 'vehicle_id': int, 'speed': float, 'section': int, 'junction': int, 'section_from': int, 'section_to': int, 'position': float, 'dist2end': float, 'total_dist': float, }
TIME_STEP_COLUMNS = ['section', 'junction', 'position', 'dist2end', 'total_dist']


class Section:

    def __init__(self, road_id, length, num_lanes=None) -> None:

        self.id = int(road_id)
        self.length = length
        self.detector_place = 0.5 * length
        self.num_lanes = int(num_lanes)
        self.in_turns = []
        self.out_turns = []

        self.clips = None  # take the clips at this section from a TrajectoryClip

    def set_detector_place(self, loc=0.5):
        self.detector_place = self.length * loc

    
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


def get_total_dist(df: pd.DataFrame):
    return (df['position'] - df['p_position']).abs().sum()

def get_total_time(df: pd.DataFrame):
    return (df['time'] - df['p_time']).abs().sum()

def get_LD_count(df: pd.DataFrame, place=0):
    """ vehicle count observed by loop detector at `place` from start
    """
    return ((df['position'] >= place) & (df['p_position'] < place)).sum()

def get_LD_speed(df: pd.DataFrame, place):
    """ vehicle speed observed by loop detector at `place` from start
    """
    vs = df[(df['position'] >= place) & (df['p_position'] < place)]
    return np.nan_to_num(((vs['position'] - vs['p_position']) / (vs['time'] - vs['p_time'])).mean())

def from_all_vehicles(df, road_network):
    
    section = road_network.get_section(df['section'].iloc[0])
    # compose trajectory segment (start_time, end_time, start_position, end_position)
    vehicle_groups = df.groupby(['vehicle_id'])
    df['p_time'] = vehicle_groups['time'].shift(1)
    df['p_position'] = vehicle_groups['position'].shift(1)
    # drop the first time step of each vehicle
    df.dropna(inplace=True)
    
    df['time_step'] = np.ceil(df['time'] / SIM_TIME_STEP).astype(int)
    
    time_groups = df.groupby('time_step')
    time_steps = list(time_groups.groups.keys())
    total_dist = time_groups.apply(get_total_dist)
    total_time = time_groups.apply(get_total_time)
    LD_count = time_groups.apply(get_LD_count, place=section.detector_place)
    LD_speed = time_groups.apply(get_LD_speed, place=section.detector_place)
    
    # the grouby.apply above returns empty dataframe (not series) if `df` is empty, and 
    # .tolist() will give an error. So we need to check if the dataframe/series is empty.
    # this can happen after the time with demand when there is no vehicle in some sections
    return {"time_steps": time_steps,
            "total_dist": total_dist.tolist() if total_dist.size > 0 else [], 
            "total_time": total_time.tolist() if total_time.size > 0 else [], 
            "LD_count": LD_count.tolist() if LD_count.size > 0 else [], 
            "LD_speed": LD_speed.tolist() if LD_speed.size > 0 else []}

def from_probe_vehicles(df: pd.DataFrame):
    """ the procedure is similar to `get_traffic_variables_of_section`, but the dataframe
        should be downsampled before applying this function. 
    """
    # compose trajectory segment (start_time, end_time, start_position, end_position)
    vehicle_groups = df.groupby(['vehicle_id'])
    df['p_time'] = vehicle_groups['time'].shift(1)
    df['p_position'] = vehicle_groups['position'].shift(1)
    # drop the first time step of each vehicle
    df.dropna(inplace=True)
    
    df['time_step'] = np.ceil(df['time'] / SIM_TIME_STEP).astype(int)
    
    time_groups = df.groupby('time_step')
    time_steps = list(time_groups.groups.keys())
    total_dist = time_groups.apply(get_total_dist)
    total_time = time_groups.apply(get_total_time)
    
    # the grouby.apply above returns empty dataframe (not series) if `df` is empty, and 
    # .tolist() will give an error. So we need to check if the dataframe/series is empty.
    # this can happen after the time with demand when there is no vehicle in some sections
    return {"pv_time_steps": time_steps, # "pv" stands for "probe vehicle"
            "pv_dist": total_dist.tolist() if total_dist.size > 0 else [], 
            "pv_time": total_time.tolist() if total_time.size > 0 else []}
    


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Generate time series data from trajectory data")
    parser.add_argument('--metadata_folder', type=str, default='datasets/simbarca/metadata', help='Path to metadata folder')
    parser.add_argument('--session_folder', type=str, default='datasets/simbarca/debug_session', help='Path to session folder')
    parser.add_argument('--penetration_rate', type=float, default=0.05, help='Penetration rate of probe vehicles')
    args = parser.parse_args()
    
    connections = pd.read_csv("{}/connections.csv".format(args.metadata_folder), dtype=CONNECTION_DTYPE)
    link_bboxes = pd.read_csv("{}/link_bboxes.csv".format(args.metadata_folder), dtype=LINK_DTYPE)
    with open("{}/intersec_polygon.json".format(args.metadata_folder), "r") as f:
        intersection_polygon = json.load(f)
    
    # construct a graph from the files
    sections = {row.id: Section(row.id, row.length, row.num_lanes) 
                for row in link_bboxes.itertuples()}
    junctions = {junction_id: Junction(junction_id) for junction_id in set(connections['intersection'])}
    
    for row in connections.itertuples(): # iterrows does not preserve data type
        junctions[row.intersection].add_turn(row.org, row.dst, row.length) 
        sections[row.org].out_turns.append(row.dst)
        sections[row.dst].in_turns.append(row.org)

    # check by comparing with Aimsun Next Model
    road_network = RoadNetwork(sections, junctions)

    with open("{}/settings.json".format(args.session_folder), "r") as f:
        settings = json.load(f)
        rng = np.random.default_rng(settings['random_seed'])
        selected_vehicles = rng.choice(range(MAX_NUM_VEHICLE), 
                                       size=int(args.penetration_rate*MAX_NUM_VEHICLE), 
                                       replace=False)
    
    # get the names of trajectory files
    data_files = sorted(glob.glob("{}/trajectory/*.json".format(args.session_folder)))
    # these files are organized chronologically by time step, so keep the last time step of
    # this file as the init_time_step for the next file
    previous_time_step: pd.DataFrame = None
    # section -> series_name -> series_data
    per_section_ts: Dict[str, Dict[str, List]] = defaultdict(lambda: defaultdict(list)) 
    # read the json data files
    for file_name in data_files:
        
        print("Working on file {}".format(file_name))

        start_time = time.perf_counter()
        with open(file_name, "r") as f:
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
        df = df[['vehicle_id', 'section', 'time', 'position']]
        df = df[df['section'] != -1] # section==-1 means the vehicle is in a junction, so ignore it
        entering['position'] = 0
        # use -1 to indicate the vehicle is at the end, will be replaced by section length
        exiting['position'] = -1
        df = pd.concat([df, entering, exiting], ignore_index=True, copy=False)
        
        df.sort_values(by=['vehicle_id', 'time'], inplace=True)
        df = df.astype({k:v for k, v in VEHINFO_DTYPE.items() if k in df.columns}, copy=False)
        print("Filling data types and values takes {:.2f}s".format(time.perf_counter() - start_time))
        
        # pandarallel uses process-based parallelism, so the child process will fork a copy 
        # of the address space, modifications to `road_network` here will not be available 
        # in the parent process.
        # The OS usually use "copy on write" to prevent unnecessary copying of memory, so the 
        # data will not be copied until modification happens (same principle applies to pandas).
        start_time = time.perf_counter()
        av_ts_in_file = df.groupby('section').parallel_apply(from_all_vehicles, road_network=road_network)
        probe_vehicle_df = df[df['vehicle_id'].isin(selected_vehicles)]
        pv_ts_in_file = probe_vehicle_df.groupby('section').parallel_apply(from_probe_vehicles)
        print("\n Processing a file takes {:.2f}s".format(time.perf_counter() - start_time))
        
        # concatenate the time series with previous file
        start_time = time.perf_counter()
        for ts_in_file in [av_ts_in_file, pv_ts_in_file]:
            for section, contents in ts_in_file.items():
                for series_name, series_data in contents.items():
                    per_section_ts[section][series_name].extend(series_data)
        print("Concatenating time series takes {:.2f}s".format(time.perf_counter() - start_time))
                
                
    # save the time series data
    start_time = time.perf_counter()
    with open("{}/time_series.json".format(args.session_folder), "w") as f:
        json.dump(per_section_ts, f)
    print("Saving all the time series takes {:.2f}s".format(time.perf_counter() - start_time))
        