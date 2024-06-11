""" this API file should be added to the "Aimsun Next APIs" tab in a scenario property (right click a scenario in the project tab). Aimsun will use an absolute path to identify this API extension, therefore when this file is moved elsewhere, the path in the .ang file should also be changed accordingly.

We generally record two kinds of information, one is the vehicle information at each time step, which contains the vehicle position at each time step. Please see `get_info` for details. The other is entering and exiting information which records when a vehicle goes into and out of a section. The latter is used to calculate the travel time of a vehicle within a section (see `AAPIEnterVehicleSection` and `AAPIExitVehicleSection`).

"""
import json
import gzip
import logging
import numpy as np
import pandas as pd

from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor

from AAPI import *
from PyANGKernel import GKSystem

# in the barcelona network, this is the ID of the car demand matrix
OD_MTX_ID = 73600

def setup_logger(name, log_file, level=logging.INFO):

    # file handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(level)
    plainer_formatter = logging.Formatter(
        "[%(asctime)s %(name)s]: %(message)s", datefmt="%m/%d %H:%M:%S"
    )
    fh.setFormatter(plainer_formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    # 
    if len(logger.handlers) > 0:
        logger.handlers.clear()
    logger.addHandler(fh)

    return logger

def get_info(time, vehicle_id):
    """ This function defines what information to get from the Aimsun API
        should match DataStorage.COLUMN_NAMES

        Recorded information:
            time: simulation time step
            idVeh: ID of vehicle
            xCurrentPos: x coordinate of vehicle
            yCurrentPos: y coordinate of vehicle
            CurrentSpeed: speed of vehicle
            idSection: ID of section (road segment) vehicle is in. -1 if it is not in a section.
            numberLane: Lane number in the segment (from 1, the rightmost lane, to N, the leftmost lane).
            idJunction: ID of junction (intersection) vehicle is in. -1 if it is not in a junction.
            idSectionFrom: ID of section vehicle is coming from, when it is in a junction. -1 if it is not in a junction.
            idSectionTo: ID of section vehicle is going to, when it is in a junction. -1 if it is not in a junction.
            CurrentPos: Position inside the section, distance from the beginning of the section or junction.
            distance2End: distance to the end of the section or to the end of the turning between two sections, depending on whether the vehicle is in a section or in a junction. This is the difference between section length and CurrentPos.
            TotalDistance: Total distance traveled.
    """
    vi = AKIVehTrackedGetInf(vehicle_id)
    return (time, 
            vi.idVeh, 
            vi.xCurrentPos,
            vi.yCurrentPos, 
            vi.CurrentSpeed, 
            vi.idSection, 
            vi.numberLane,
            vi.idJunction,
            vi.idSectionFrom, 
            vi.idSectionTo,
            vi.CurrentPos,
            vi.distance2End,
            vi.TotalDistance)


class DataStorage:

    MAX_TRAJ_ROWS = 1e7
    INFO_COLUMNS = ["time", "vehicle_id", "x", "y", "speed", "section", "lane", "junction",
                    "section_from", "section_to", "position", "dist2end", "total_dist"]
    INOUT_COLUMNS = ["vehicle_id", "section", "time"]
    
    def __init__(self, logger):
        # the speed is in KPH according to the definition of the barcelona network
        # a vehicle is either in a section or a junction, and -1 indicates not in
        self.traj_info = []
        self.entering = []
        self.exiting = []
        self.network_entrance = [] 
        self.network_exit = []
        self.logger = logger
        self.num_saved = 0 # number of saved files

    def update_traj(self, time: float, new_locs: List[Tuple]):

        self.traj_info.extend(new_locs)
        if len(self.traj_info) > self.MAX_TRAJ_ROWS:
            self.save_to_file("./trajectory/{:02d}_traj_{}.json".format(self.num_saved, time))
            
    def save_to_file(self, file_name):
        compressed_file_name = "{}.gz".format(file_name)
        contents_to_save = {
                "info_columns": self.INFO_COLUMNS, 
                "trajectory": self.traj_info,
                "inout_columns": self.INOUT_COLUMNS,
                "entering": self.entering, 
                "exiting": self.exiting,
                "network_entrance": self.network_entrance,
                "network_exit": self.network_exit}
        with gzip.open(compressed_file_name, "wt") as f:
            json.dump(contents_to_save, f)
        self.logger.info("file saved to {}".format(compressed_file_name))
        self.traj_info.clear()
        self.entering.clear()
        self.exiting.clear()
        self.network_entrance.clear()
        self.network_exit.clear()
        self.num_saved += 1

    def clean_up(self):
        self.save_to_file("./trajectory/{:02d}_traj_end.json".format(self.num_saved))

class TrafficDemand:

    def __init__(self, rng, vehicle_type=1, time_slice=1, logger=None):
        """
        Args:
            vehicle_type (int, optional): from 1 to AKIVehGetNbVehTypes(). Defaults to 1 (car).
            time_slice (int, optional): from 0 to AKIODDemandGetNumSlicesOD(vehicle_type). Defaults to 1 (after warmup).
        """
        self.vehicle_type = vehicle_type
        self.time_slice = time_slice
        self.od_pairs: np.array = None
        self.rng: np.random.Generator = rng
        self.logger = logger
        
    
    def get_od_from_model(self, model):
        
        catalog = model.getCatalog()
        all_centroids = catalog.getObjectsByType( model.getType( "GKCentroid" ) )
        od_mtx = catalog.getObjectsByType( model.getType( "GKODMatrix" ) )[OD_MTX_ID]

        # this takes about a minute, as the function doesn't allow reading the whole matrix all at once
        od_pairs = []
        for org_key, org in all_centroids.items():
            for dst_key, dst in all_centroids.items():
                # this function works when the vehicles have been generated, at AAPISimulationReady()
                num_trips = od_mtx.getTrips(org, dst)
                if num_trips > 0:
                    od_pairs.append((org_key, dst_key, num_trips))
                    
        self.od_pairs = np.array(od_pairs)
    
    def apply_random_transform(self, settings: dict):
        """ modifies the self.od_pairs data in place
        """
        orgs, dsts, trips = self.od_pairs[:, 0], self.od_pairs[:, 1], self.od_pairs[:, 2]
        
        trips[self.rng.random(size=trips.shape) < settings["mask_p"]] = 0 # random mask-out
        trips = np.where( # random noise with certain probability and scale
            self.rng.random(size=trips.shape) < settings["noise_p"], 
            trips * (1 + self.rng.uniform(low=-1.0, high=1.0, size=trips.shape) * settings["noise_scale"]), 
            trips)
        trips = trips * settings["global_scale"]
        
        self.od_pairs[:, 2] = trips
        

    def write_changes(self):
        for org, dst, num_trips in self.od_pairs:
            ret = AKIODDemandSetDemandODPair(int(org), int(dst), self.vehicle_type, self.time_slice, int(num_trips))
            if ret < 0:
                self.logger.info("Error updating the demand org {} dst {} num_trips".format(
                    org, dst, num_trips
                ))
    
    def check_changes(self):
        for org, dst, num_trips in self.od_pairs:
            ret = AKIODDemandGetDemandODPair(int(org), int(dst), self.vehicle_type, self.time_slice)
            if abs(ret - num_trips) > 10 and abs(ret-num_trips)/num_trips > 0.1:
                self.logger.info("The demand with org {} dst {} num_trips is set to be {} but turns out to be {}".format(
                    org, dst, num_trips, ret
                ))
    
model = GKSystem.getSystem().getActiveModel()
vehicles_inside = []

logger = setup_logger(name="default", log_file="./api_log.log")
storage = DataStorage(logger)

with open("./settings.json", "r") as f:
    settings = json.load(f)
rng = np.random.default_rng(seed=settings["random_seed"])

demand = TrafficDemand(rng, logger=logger)
demand.get_od_from_model(model)
demand.apply_random_transform(settings)

executors = ThreadPoolExecutor(max_workers=settings["num_thread"])

def AAPILoad():
    """
    Called when the module is loaded by Aimsun Next.
    """
    # AKIPrintString("AAPILoad")
    demand.write_changes()
    rep = model.getCatalog().find( ANGConnGetReplicationId() )
    rep.setRandomSeed(settings["random_seed"])
    logger.info("Running Experiment {} Replication {} Random seed {}".format(
        ANGConnGetExperimentId(),
        ANGConnGetReplicationId(),
        rep.getRandomSeed()
    ))
    return 0


def AAPIInit():
    """ Called when Aimsun Next starts the simulation and can be used to initialize the module.
    """
    # AKIPrintString("AAPIInit")
    return 0


def AAPISimulationReady():
    """Called when Aimsun Next has initialized all and vehicles are ready to start moving.
    """
    # AKIPrintString("AAPISimulationReady")
    logger.info("Simulation Ready")
    demand.check_changes()
    return 0


def AAPIManage(time, timeSta, timeTrans, acycle):
    """
    Called in every simulation step at the beginning of the cycle, and can be used to request detector measures, vehicle information and interact with junctions, meterings and VMS in order to implement the control and management policy. This function takes four time related parameters:

    Args:
            - *time*: Absolute time of simulation in seconds. At the beginning of the simulation (beginning of the warm-up, if any), it takes the value 0.
            - *timeSta*: Time of simulation in stationary period, in seconds from midnight.
            - *timeTrans*: Duration of warm-up period, in seconds.
            - *cycle*: Duration of each simulation step in seconds.

    Returns:
            _type_: _description_
    """
    # AKIPrintString("AAPIManage")
    return 0


def AAPIPostManage(time, timeSta, timeTrans, acycle):
    """ 
    Called in every simulation step at the end of the cycle, and can be used to request detector measures, vehicle information and interact with junctions, meterings and VMS to implement the control and management policy. This function takes four time related parameters:

            time: Absolute time of simulation in seconds. At the beginning of the simulation (beginning of the warm-up, if any), it takes the value 0.
            timeSta: Time of simulation in stationary period, in seconds from midnight.
            timeTrans: Duration of warm-up period, in seconds.
            cycle: Duration of each simulation step in seconds.
    """
    # AKIPrintString("AAPIPostManage")
    # record the locations of all vehicles at this time step

    all_veh_info = list(executors.map(lambda vid: get_info(time, vehicle_id=vid), vehicles_inside))
    storage.update_traj(time, all_veh_info)
    
    # report vehicle number every 1000 steps
    if int(time / acycle) % 1000 == 1:
        logger.info("{} vehicles in network at time {}".format(len(vehicles_inside), time))
        
    return 0


def AAPIFinish():
    """Called when Aimsun Next finishes the simulation and can be used to terminate the module operations, write summary information, close files, etc.
    """
    # AKIPrintString("AAPIFinish")
    storage.clean_up()
    logger.info("Simulation Finished")
    return 0


def AAPIUnLoad():
    """Called when the module is unloaded by Aimsun Next.
    """
    # AKIPrintString("AAPIUnLoad")
    return 0


def AAPIPreRouteChoiceCalculation(time, timeSta):
    """ Called just before a new cycle of route choice calculation is about to begin. It can be used to modify the sections and turnings costs to affect the route choice calculation. This function takes two parameters in relation to time:

    - *time*: Absolute time of simulation in seconds. At the beginning of the simulation (beginning of the warm-up, if any), it takes the value 0.
    - *timeSta*: Time of simulation in stationary period, in seconds from midnight.
    """
    # AKIPrintString("AAPIPreRouteChoiceCalculation")
    return 0


def AAPIEnterVehicle(idveh, idsection):
    """ Called when a new vehicle enters the system, that is, when the vehicle enters its first section, not when it enters a Virtual queue if one is present. This function takes two parameters:

    idveh: Identifier of the vehicle entering the network.
    idsection: Identifier of the section where the vehicle enters the network.
    """
    # setting the vehicle as tracked gives 10x speedup for information look up 
    # with AKIVehTrackedGetInf(), compared to AKIVehGetInf()
    AKIVehSetAsTracked(idveh)
    vehicles_inside.append(idveh)
    storage.network_entrance.append((idveh, idsection, AKIGetCurrentSimulationTime()))
    return 0


def AAPIExitVehicle(idveh, idsection):
    """ Called when a vehicle exits the network. This function takes two parameters:

    idveh: Identifier of the vehicle exiting the network.
    idsection: Identifier of the section where the vehicle exits the network.
    """
    AKIVehSetAsNoTracked(idveh)
    vehicles_inside.remove(idveh)
    storage.network_exit.append((idveh, idsection, AKIGetCurrentSimulationTime()))
    return 0


def AAPIEnterPedestrian(idPedestrian, originCentroid):
    """ Called when a new pedestrian enters the system, that is, when the pedestrian enters through its entrance. This function takes two parameters:

    - *idPedestrian*: Identifier of the new pedestrian entering the network.
    - *originCentroid*: Identifier of the pedestrian entrance where the pedestrian enters the network.

    """
    return 0


def AAPIExitPedestrian(idPedestrian, destinationCentroid):
    """Called when a pedestrian exits the network. This function takes two parameters:

    - *idPedestrian*: Identifier of the pedestrian exiting the network.
    - *destinationCentroid*: Identifier of the pedestrian exit where the pedestrian exits the network.
    """
    return 0


def AAPIEnterVehicleSection(idveh, idsection, atime):
    """ Called when a vehicle enters a new section. This function takes three parameters:

            - *idveh*: Identifier of the vehicle.
            - *idsection*: Identifier of the section the vehicle is entering.
            - *atime*: Absolute time of the simulation when the vehicle enters the section. At the beginning of the simulation (beginning of the warm-up, if any), it takes the value 0.
    """
    storage.entering.append((idveh, idsection, atime))
    return 0


def AAPIExitVehicleSection(idveh, idsection, atime):
    """ Called when a vehicle exits a section. This function receives two parameters:

            - *idveh*: Identifier of the vehicle.
            - *idsection*: Identifier of the section the vehicle is exiting.
            - *atime*: Absolute time of the simulation when the vehicle is exits the section. At the beginning of the simulation (beginning of the warm-up, if any), it takes the value 0.
    """
    storage.exiting.append((idveh, idsection, atime))
    return 0


def AAPIVehicleStartParking(idveh, idsection, time):
    """Called when a vehicle starts a parking maneuver.

    - *idveh*: Vehicle identifier.
    - *idsection*: Section identifier where vehicle is doing the parking.
    - *time*: Current simulation time.
    """
    return 0
