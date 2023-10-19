""" 
    Export the road graph of an aimsun network to readable files, and export the 
    origin-demand matrix for the car demand to json (for information and debug). 
    
    The ID of the demand matrix is manually chosen to be the main demand (not warm up),
    for other network models, this ID will be different, modify `OD_MTX_ID`
    
    this script should be added to the `scripts` section under Aimsun project panel
	variables like model and GKSystem are initialized by aimsun, see here 
 	https://docs.aimsun.com/next/22.0.2/UsersManual/ScriptExecute.html#initialization
 
	to run it, click the execute button in the script editor window in Aimsun
"""

# in the barcelona network, this is for the car demand matrix
OD_MTX_ID = 73600

import json
import pandas as pd
from typing import List, Dict


link_bboxes = pd.DataFrame(columns=["id", "from_x", "from_y","to_x", "to_y", "length", "out_ang","num_lanes"])
connections = pd.DataFrame(columns=["turn", "intersection", "org", "dst"])
intersec_polygon = dict()

catalog = model.getCatalog()

# obtain all node objects from the network model
# catalog.getUsedSubTypesFromType will return GKNode and its subclasses as a list of dict
all_nodes: Dict = catalog.getObjectsByType( model.getType( "GKNode" ) )
# A node in Aimsun is correspond to an intersection in real-world, its shape is described 
# using a polygon `node.getPolygon()` and it contains one or more `turning`. 
# Each `turning` links a road segment to another (turn.getOrigin() and turn.getDestination())
idx = 0
for key, node in all_nodes.items():
    # print("Node ID {}".format(key)) # the dict key is ID, can be verified by node.getId()
    intersec_polygon[node.getId()] = {"polygon": [(p.x, p.y) for p in node.getPolygon()]}
    for turn in node.getTurnings():
        connections.loc[idx, :] = [turn.getId(), node.getId(), turn.getOrigin().getId(), turn.getDestination().getId()]
        idx += 1

# A section in Aimsun is correspond to a road segment, it has an ID, and its bounding boxes can
# be obtained using .getBBox()
all_links: Dict = catalog.getObjectsByType( model.getType( "GKSection" ) )
for idx, (key, link) in enumerate(all_links.items()):
    box = link.getBBox()
    # the two corners of the bbox are called from and to, but from is a reserved keyword in python
    # so we use getattr to get the attribute
    pt_from, pt_to = getattr(box, "from"), getattr(box, "to")
    link_bboxes.loc[idx, :] = [link.getId(), 
                               pt_from.x if not box.isUndefined() else -1, 
                               pt_from.y if not box.isUndefined() else -1, 
                               pt_to.x if not box.isUndefined() else -1, 
                               pt_to.y if not box.isUndefined() else -1,
                               link.length2D(),
                               link.getExitAngle(),
                               len(link.getLanes())]


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


link_bboxes.set_index("id", inplace=True)
connections.set_index("turn", inplace=True)
link_bboxes.to_csv("link_bboxes.csv")
connections.to_csv("connections.csv")
with open("intersec_polygon.json", "w") as f:
    f.write(json.dumps(intersec_polygon, indent=4))
with open("od_pairs.json", "w") as f:
    f.write(json.dumps({"od_pairs": od_pairs}))