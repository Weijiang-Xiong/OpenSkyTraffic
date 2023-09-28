counter=0
detector_length=2

sectionType = model.getType( "GKSection" )
# Iterate over all the sections in the network.
for types in model.getCatalog().getUsedSubTypesFromType( sectionType ):
	for section in types.itervalues():
		max = section.getNbFullLanes ()
		for i in range(max):
			counter=counter+1
			detector = GKSystem.getSystem().newObject( "GKDetector", model )
			#Sets parameters.
			detector.setLanes( i, i )
			detector.setLength( detector_length)
			detector.setPosition( section.length2D() / 2.0)

			section.addTopObject(detector)
			model.getGeoModel().add(section.getLayer(), detector)

# Be sure that you reset the UNDO buffer after a modification that cannot be undone
model.getCommander().addCommand( None )
print(counter)
print("Done")