import argparse
from iblviewer.mouse_brain import MouseBrainViewer

"""
Project: IBL Viewer
Description: Interactive volumetric and surface data visualizer based 
on VTK, vedo (a VTK wrapper) and Python.

Description: this is a simple viewer for 3D volumetric and surface
data visualization. 

In the context of the International Brain Laboratory, this viewer is
used by neuroscientists to perform analysis of data models registered
to the Allen Mouse CCF v3 atlas. 

Copyright: 2021 Nicolas Antille, International Brain Laboratory
License: MIT
"""

# From stackoverflow
def str2bool(v):
    if isinstance(v, bool):
        return v
    elif v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data mapping on the Allen CCF volume atlas')
    parser.add_argument('-r', dest='resolution', type=int, default=50, 
    help='Volume resolution. Possible values: 100, 50, 25, and 10. Units are in microns. The 10um volume takes a lot of RAM (and some time to load)')
    
    parser.add_argument('-m', dest='mapping', type=str, default='Allen', 
    help='Volume mapping name. Either Allen (default value) or Beryl (IBL specific simplified mapping).')
    
    parser.add_argument('-d', dest='dark_mode', type=int, default=0, 
    help='Enable (1) or disable (0) dark mode.')
    
    parser.add_argument('-e', dest='embed_ui', type=int, default=1, 
    help='Whether you want to see a UI within the VTK window')
    
    parser.add_argument('-v', dest='volume_mode', type=str2bool, default=False, 
    help='Whether the Allen Atlas volume (1) or the DWI (0) is loaded')
    
    parser.add_argument('-cm', dest='color_map', type=str, default='viridis', 
    help='Color map for the volume')
    
    parser.add_argument('-nc', dest='nan_color', type=float, default=0.65, 
    help='Gray color (between 0 and 1) for regions that have no assigned value')
    
    parser.add_argument('-na', dest='nan_alpha', type=float, default=0.5, 
    help='Alpha (opacity) value for regions that have no assigned value')

    args = parser.parse_args()
    viewer = MouseBrainViewer()
    viewer.initialize(resolution=args.resolution, mapping=args.mapping, add_dwi=not args.volume_mode,
    add_atlas=args.volume_mode, embed_ui=bool(args.embed_ui), dark_mode=bool(args.dark_mode))
    viewer.show()