from iblviewer.mouse_brain import MouseBrainViewer

viewer = MouseBrainViewer()
viewer.initialize(resolution=50, mapping='Allen', embed_ui=True, jupyter=False)
viewer.show().close()