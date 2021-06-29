from iblviewer.application import Viewer
import numpy as np

viewer = Viewer()
viewer.initialize(embed_ui=True, jupyter=False)
# Test data
points = np.random.random((500, 3)) * 1000
viewer.add_points(points)
viewer.show().close()