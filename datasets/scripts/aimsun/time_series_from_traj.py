import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from copy import deepcopy
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor

SIM_TIME_STEP = 0.5  # simulation time step is set to be 0.5s in Aimsun
CONNECTION_DTYPE = {'turn': int, 'org': int, 'dst': int, 'intersection': int, 'length': float}
LINK_DTYPE = {'id': int, 'from_x': float, 'from_y': float, 'to_x': float, 'to_y': float, 'length': float, 'out_ang': float, 'num_lanes': int}
VEHINFO_DTYPE = {'time': float, 'vehicle_id': int, 'speed': float,
              'section': int, 'junction': int, 'section_from': int, 'section_to': int}
COLUMNS_TO_RECORD = ['speed', 'dist2end', 'section', 'junction', 'section_from', 'section_to']


class Section:

    def __init__(self, road_id, length, num_lanes=None) -> None:

        self.id = int(road_id)
        self.length = length
        self.num_lanes = int(num_lanes)
        self.in_turns = []
        self.out_turns = []

        self.clips = []  # take the clips at this section from a TrajectoryClip

    def add_detector(self, loc=0.5):
        pass


class Junction:

    def __init__(self, junction_id) -> None:
        self.id = junction_id
        self.turns = dict()

    def add_turn(self, org, dst, length):
        self.turns["{}-{}".format(org, dst)] = length

    def get_turn_length(self, org, dst):
        return self.turns.get("{}-{}".format(org, dst), 9999)


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

class TrajectoryClip:

    def __init__(self, clips: List[Tuple]) -> None:
        
        self.clips = clips  # list of (vehicle_id, section_id, start_time, end_time, start_loc, end_loc)


# use BFS to find the shortest path ,return the path as a list of sections
def BFS_shortest_path(graph: RoadNetwork, start: str, end: str):
    
    if start == end:
        return [start, end]
    else:
        
        depth, queue = 0, [[start]]
        
        while queue:
            path = queue.pop(0)
            node = path[-1]
            for next_node in graph.get_section(node).out_turns:
                new_path = list(path)
                new_path.append(next_node)
                queue.append(new_path)
                if next_node == end:
                    return new_path

            # avoid infinite loop
            depth += 1 
            if depth > 10:
                return None


def get_trajectory_clips(graph: RoadNetwork, row: pd.Series, mode=1):

    match mode:
        case 1:  # (start, end) = (section, section)
            # a) same section
            if row['section'] == row['p_section']:
                row['traj_clip'] = TrajectoryClip([(
                    row['vehicle_id'], row['section'], row['time'] - SIM_TIME_STEP, row['time'],
                    row['p_dist2end'], row['dist2end']
                )])
            # b) search(start, end)
            else:
                path = BFS_shortest_path(graph, row['p_section'], row['section'])
                
                clips = [()]
                for i in range(1, len(path) - 1):
                    pass 
                
                    
                
        case 2:  # (start, end) = (junction, section)
            # start.dist2end + search(start.dst, end)
            pass
        case 3:  # (start, end) = (section, junction)
            # search(start, end.org)
            pass
        case 4:  # (start, end) = (junction, junction)
            # a) same junction, b) search(start.dst, end.org)
            pass


if __name__ == "__main__":
    
    # create multiple thread executor
    executor = ThreadPoolExecutor(max_workers=8)
    
    metadata_folder = "datasets/simbarca/metadata"
    session_folder = "datasets/simbarca/debug_session"

    connections = pd.read_csv("{}/connections.csv".format(metadata_folder), dtype=CONNECTION_DTYPE)
    link_bboxes = pd.read_csv("{}/link_bboxes.csv".format(metadata_folder), dtype=LINK_DTYPE)
    with open("{}/intersec_polygon.json".format(metadata_folder), "r") as f:
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
    

    # get the names of trajectory files
    all_data_files = sorted(glob.glob("{}/trajectory/*.json".format(session_folder)))
    assert "traj_end" in all_data_files[-1]

    init_time_step: pd.DataFrame = None
    # read the json data files
    for data_file in all_data_files:

        with open(data_file, "r") as f:
            data = json.load(f)

        # concatenate the last time step with the current data frame
        if init_time_step is not None:
            df = pd.concat([init_time_step,
                            pd.DataFrame(data=data['trajectory'], columns=data['column_names'])],
                           copy=False)
        else:
            df = pd.DataFrame(data=data['trajectory'], columns=data['column_names'])
        df.sort_values(by=['vehicle_id', 'time'], inplace=True)
        df.astype(VEHINFO_DTYPE, copy=False)
        # add empty columns for previous time step
        for col in COLUMNS_TO_RECORD:
            df['p_{}'.format(col)] = np.nan

        time_groups = df.groupby('time')
        # these files are organized chronologically by time step, so keep the last time step of
        # this file as the init_time_step for the next file
        # check the init_time_step['time'] is the same as the number in the file name (the time step when this file is saved)
        init_time_step: pd.DataFrame = time_groups.get_group(list(time_groups.groups.keys())[-1])

        # to check the shifting works see in debug console or print
        # vehicle_groups.get_group(77)[['vehicle_id', 'speed', 'p_speed']]
        vehicle_groups = df.groupby(['vehicle_id'])
        for col in COLUMNS_TO_RECORD:
            df['p_{}'.format(col)] = vehicle_groups[col].shift(1)

        # drop the initial time step, since it has no previous time step
        # check that when reading the first file of the simulated trajectories,
        # the len(df) is decreased by the number of vehicle after this step
        df.dropna(inplace=True)

        df['traj_clip'] = None  # add an empty column for trajectory clips
        # split the trajectory by section
        # see the note https://pandas.pydata.org/docs/user_guide/indexing.html#boolean-indexing
        df[(df['p_section'] > 0) & (df['section'] > 0)].apply(
            lambda row: get_trajectory_clips(graph=road_network, row=row, mode=1), axis='columns')
        df[(df['p_junction'] > 0) & (df['section'] > 0)].apply(
            lambda row: get_trajectory_clips(graph=road_network, row=row, mode=2), axis='columns')
        df[(df['p_section'] > 0) & (df['junction'] > 0)].apply(
            lambda row: get_trajectory_clips(graph=road_network, row=row, mode=3), axis='columns')
        df[(df['p_junction'] > 0) & (df['junction'] > 0)].apply(
            lambda row: get_trajectory_clips(graph=road_network, row=row, mode=4), axis='columns')
