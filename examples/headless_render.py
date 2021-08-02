from iblviewer.mouse_brain import MouseBrainViewer

# This example starts the viewer and renders a view
viewer = MouseBrainViewer()
viewer.initialize(resolution=50, mapping='Allen', offscreen=True)
# Put objects on the scene
viewer.show()

# Add more code to add data and control the viewer here

# Render a 4K image. 1920*2 by 1080*2 pixels.
# Scaling is used to keep text at a legible size but
# you can also disable info overlay and ignore scaling.
#viewer.set_info_visibility(False)
viewer.render('./test.jpg', 1920, 1080, 2)
viewer.close()