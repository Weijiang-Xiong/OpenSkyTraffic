""" 
    reads nodes and sections in a Aimsun network,  and record the road graph into a file. 
    
    this script should be added to the `scripts` section under Aimsun project panel
	variables like model and GKSystem are initialized by aimsun, see here 
 	https://docs.aimsun.com/next/22.0.2/UsersManual/ScriptExecute.html#initialization
 
	to run it, click the execute button in the script editor window in Aimsun
"""
import pandas as pd
from typing import List, Dict

link_poses = pd.DataFrame(columns=["id", "from_x", "from_y","to_x", "to_y"])
connections = pd.DataFrame(columns=["turn", "node", "org", "dst"])

catalog = model.getCatalog()
# obtain all node objects from the network model
# catalog.getUsedSubTypesFromType will return GKNode and its subclasses as a list of dict
all_nodes: Dict = catalog.getObjectsByType( model.getType( "GKNode" ) )

# A node in Aimsun is correspond to an intersection in real-world, and it contains one or more `turning`
# each `turning` links a road segment to another (turn.getOrigin() and turn.getDestination())
idx = 0
for ikey, node in all_nodes.items():
    # print("Node ID {}".format(key)) # the dict key is ID, can be verified by node.getId()
    for turn in node.getTurnings():
        connections.loc[idx, :] = [turn.getId(), node.getId(), turn.getOrigin().getId(), turn.getDestination().getId()]
        idx += 1

# A section in Aimsun is correspond to a road segment, it has an ID, and its bounding boxes can
# be obtained using .getBBox()
idx = 0
all_links: Dict = catalog.getObjectsByType( model.getType( "GKSection" ) )
for idx, (key, link) in enumerate(all_links.items()):
    box = link.getBBox()
    # the two corners of the bbox are called from and to
    pt_from, pt_to = getattr(box, "from"), getattr(box, "to")
    link_poses.loc[idx, :] = [link.getId(), pt_from.x, pt_from.y, pt_to.x, pt_to.y]
    idx += 1

link_poses.set_index("id", inplace=True)
connections.set_index("turn", inplace=True)
link_poses.to_csv("link_sizes.csv")
connections.to_csv("connections.csv")