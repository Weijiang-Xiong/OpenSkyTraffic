""" this API file should be added to the "API" tab in a scenario property (right click a scenario)
"""
import logging
import numpy as np
import pandas as pd 
from typing import Dict, List, Tuple

from AAPI import *
from PyANGKernel import GKSystem

model = GKSystem.getSystem().getActiveModel()
vehicles_inside = []

def AAPILoad():
	"""
	Called when the module is loaded by Aimsun Next.
	"""
	AKIPrintString( "AAPILoad" )
	return 0

def AAPIInit():
	""" Called when Aimsun Next starts the simulation and can be used to initialize the module.
	"""
	AKIPrintString( "AAPIInit" )
	return 0

def AAPISimulationReady():
	"""Called when Aimsun Next has initialized all and vehicles are ready to start moving.
	"""
	AKIPrintString( "AAPISimulationReady" )
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
	AKIPrintString( "AAPIManage" )
	return 0

def AAPIPostManage(time, timeSta, timeTrans, acycle):
	""" 
	Called in every simulation step at the end of the cycle, and can be used to request detector measures, vehicle information and interact with junctions, meterings and VMS to implement the control and management policy. This function takes four time related parameters:

		time: Absolute time of simulation in seconds. At the beginning of the simulation (beginning of the warm-up, if any), it takes the value 0.
		timeSta: Time of simulation in stationary period, in seconds from midnight.
		timeTrans: Duration of warm-up period, in seconds.
		cycle: Duration of each simulation step in seconds.
	"""
	AKIPrintString( "AAPIPostManage" )
	for vehicle_id in vehicles_inside:
		print(f'Vehicle: {vehicle_id}')
		vehicle_info = AKIVehGetInf(vehicle_id)
		xPos = vehicle_info.xCurrentPos
		print(f'xPos: {xPos}')
		yPos = vehicle_info.yCurrentPos
		print(f'yPos: {yPos}')
	return 0

def AAPIFinish():
	AKIPrintString( "AAPIFinish" )
	return 0

def AAPIUnLoad():
	AKIPrintString( "AAPIUnLoad" )
	return 0
	
def AAPIPreRouteChoiceCalculation(time, timeSta):
	""" Called just before a new cycle of route choice calculation is about to begin. It can be used to modify the sections and turnings costs to affect the route choice calculation. This function takes two parameters in relation to time:
 
	- *time*: Absolute time of simulation in seconds. At the beginning of the simulation (beginning of the warm-up, if any), it takes the value 0.
	- *timeSta*: Time of simulation in stationary period, in seconds from midnight.
	"""
	AKIPrintString( "AAPIPreRouteChoiceCalculation" )
	return 0

def AAPIEnterVehicle(idveh, idsection):
	""" Called when a new vehicle enters the system, that is, when the vehicle enters its first section, not when it enters a Virtual queue if one is present. This function takes two parameters:

	idveh: Identifier of the vehicle exiting the network.
	idsection: Identifier of the section where the vehicle exits the network.
	"""
	vehicles_inside.append(idveh)
	return 0

def AAPIExitVehicle(idveh, idsection):
	""" Called when a vehicle exits the network. This function takes two parameters:

	idveh: Identifier of the vehicle exiting the network.
	idsection: Identifier of the section where the vehicle exits the network.
	"""
	vehicles_inside.pop(idveh)
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
	return 0

def AAPIExitVehicleSection(idveh, idsection, atime):
	""" Called when a vehicle exits a section. This function receives two parameters:

		- *idveh*: Identifier of the vehicle.
		- *idsection*: Identifier of the section the vehicle is exiting.
		- *atime*: Absolute time of the simulation when the vehicle is exits the section. At the beginning of the simulation (beginning of the warm-up, if any), it takes the value 0.
	"""
	return 0

def AAPIVehicleStartParking (idveh, idsection, time):
	"""Called when a vehicle starts a parking maneuver.

    - *idveh*: Vehicle identifier.
    - *idsection*: Section identifier where vehicle is doing the parking.
    - *time*: Current simulation time.
	"""
	return 0
