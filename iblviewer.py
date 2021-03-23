
"""
Project: IBL Viewer
Description: Interactive volumetric and surface data visualizer based 
on vedo (a VTK wrapper), matplotlib, VTK itself and Python, 
a common environment for scientific visualization.

Description: this is a simple viewer dedicated to neuroscience and
based on Allen Atlases. The base case is the Mouse CCF v3. You can
load the mouse atlas and visualize your own data, such as:
- custom scalar values per brain region
- point neurons
- neuronal probes (as lines or sets of lines for instance)
Users have full control in Python over the given data and can take
advantage of vedo wrapper for more visualization features.

Copyright: 2021 Nicolas Antille, International Brain Laboratory
License: MIT
"""

__version__ = '1.0.1'
__author__ = 'Nicolas Antille'
__credits__ = 'International Brain Laboratory'

from iblviewer.atlas_controller import AtlasController

if __name__ == '__main__':
    controller = AtlasController(50, embed_ui=True)