import argparse
import numpy as np
from iblviewer.qt_application import ViewerApp
from iblviewer.application import Viewer
got_ibllib = True
from iblviewer.mouse_brain import MouseBrainViewer



"""
Project: IBL Viewer

Description: this is a fast and interactive 3D viewer for exploring
and analysing volumes, surfaces, points and lines.
It's based on Python, VTK and partly on vedo (a VTK wrapper).

In the context of the International Brain Laboratory, this viewer is
used by neuroscientists to perform analysis of data models in the
context of the Allen Mouse CCF v3 atlas. 

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

class IBLViewer():

    def __init__(self):
        self.viewer = None
        self.qt_app = None
        self.args = None

    def parse_args(self, parser=None):
        """
        Standard argument parser for iblviewer. Make sure you do not use the below argument shortcuts:
        -s, -t, -ui, -r, -m, -d, -atlas, -dwi, -cm, -nc, -na
        :param parser: An existing ArgumentParser that will be updated with standard arguments.
            If None, a new one is created (default).
        :return: ArgumentParser
        """
        if parser is None:
            parser = argparse.ArgumentParser(description='International Brain Viewer based on VTK')

        parser.add_argument('-neuro', dest='neuroscience', type=str2bool, default=True, 
        help='Whether the viewer starts in Neuroscience mode with Mouse Brain volume preset (1) or as generic 3D viewer (0)')

        parser.add_argument('-t', dest='test_data', type=str2bool, default=False, 
        help='Whether a simple random set of points is added as test data')

        parser.add_argument('-ui', dest='ui', type=int, default=1, 
        help='User interface. 0 for none, 1 for Qt, 2 for embed')

        # More command-line options are given in the context of neuroscience
        parser.add_argument('-r', dest='resolution', type=int, default=50, 
        help='Volume resolution. Possible values: 100, 50, 25, and 10. Units are in microns.\
            The 10um volume takes a lot of RAM (and some time to load)')
        
        parser.add_argument('-m', dest='mapping', type=str, default='Allen', 
        help='Volume mapping name. Either Allen (default value) or Beryl (IBL specific simplified mapping).')
        
        parser.add_argument('-d', dest='dark_mode', type=str2bool, default=True, 
        help='Enable (1) or disable (0) dark mode.')
        
        parser.add_argument('-atlas', dest='add_atlas', type=str2bool, default=True, 
        help='If the Allen Atlas volume should be added to the viewer')
        
        parser.add_argument('-dwi', dest='add_dwi', type=str2bool, default=False, 
        help='If the Allen Atlas raw DWI volume should be added to the viewer')
        
        parser.add_argument('-cm', dest='color_map', type=str, default='viridis', 
        help='Color map for custom data mapped on to the Allen atlas volume')
        
        parser.add_argument('-nc', dest='nan_color', type=float, default=0.65, 
        help='Gray color (between 0 and 1) for regions that have no assigned value')
        
        parser.add_argument('-na', dest='nan_alpha', type=float, default=0.5, 
        help='Alpha (opacity) value for regions that have no assigned value')

        args = parser.parse_args()
        self.args = args
        return args

    def launch(self, callable=None, stats_callable=None, args=None, 
                jupyter=False, neuroscience=True, **kwargs):
        """
        Start the 3D viewer according to parameters given in the console
        :param callable: Function that will be called when the 3D viewer is initialized
        :param stats_callable: Function that will be called when statistics are updated,
            when a selection or sub selection changes in the 3D viewer. Available when 
            you use the Qt UI only.
        :param args: Any existing ArgumentParser. If None, a new IBL standard one is created.
        :param jupyter: Whether you launch the viewer within a jupyter notebook or lab
        :param neuroscience: Whether the viewer in jupyter is started in neuroscience mode or not
        :param kwargs: All further keyword arguments set to viewer.initialize() method (for jupyter mode)
        :return: Either a qt_application.ViewerApp (if Qt) or a viewer instance (mouse_brain.MouseBrainViewer
            or application.)
        """
        ibllib_msg = 'The viewer is set to start in neuroscience mode but you do not have ibllib '
        ibllib_msg += 'optional module installed.\n\nPlease run pip install ibllib and run the viewer '
        ibllib_msg += 'again if you want to start in neuroscience mode.\n\n'
        ibllib_msg += 'Alternatively, you may use the viewer in standard mode with random points for test: iblviewer -neuro 0 -t 1\n'
        if jupyter:
            if neuroscience and not got_ibllib:
                print(ibllib_msg)
                exit()
            if neuroscience:
                # This a computational neuroscience environment, in this case focused
                # on the Allen Brain Atlas and International Brain Laboratory data models
                self.viewer = MouseBrainViewer()
            else:
                # This is a generic 3D viewer, not related to neuroscience
                self.viewer = Viewer()
            self.viewer.initialize(**kwargs)
            return self.viewer.show()

        if args is None:
            args = self.args
            if args is None:
                args = self.parse_args()
                self.args = args

        if args.neuroscience:
            # This a computational neuroscience environment, in this case focused
            # on the Allen Brain Atlas and International Brain Laboratory data models
            self.viewer = MouseBrainViewer()
        else:
            # This is a generic 3D viewer, not related to neuroscience
            self.viewer = Viewer()

        qt_mode = args.ui == 1
        if qt_mode:
            self.qt_app = ViewerApp()
            if args.neuroscience:
                # viewer.initialize(...) method will be called internally with the expanded args
                self.qt_app.initialize(viewer=self.viewer, callable=callable, stats_callable=stats_callable, 
                                embed_ui=args.ui==2, offscreen=False, dark_mode=args.dark_mode, 
                                resolution=args.resolution, mapping=args.mapping, add_dwi=args.add_dwi, 
                                add_atlas=args.add_atlas)
            else:
                new_function = callable
                if args.test_data:
                    # Test data
                    points = np.random.random((500, 3)) * 1000
                    def new_function(viewer):
                        if callable is not None:
                            callable(viewer)
                        viewer.add_spheres(points, radius=10)
                # viewer.initialize(...) method will be called internally with the expanded args
                self.qt_app.initialize(viewer=self.viewer, callable=new_function, stats_callable=stats_callable,
                                embed_ui=args.ui==2, offscreen=False, dark_mode=args.dark_mode)
            return self.qt_app
            # Any code below here is only executed once you quit the Qt application

        else:
            if args.neuroscience:
                self.viewer.initialize(resolution=args.resolution, mapping=args.mapping, add_dwi=args.add_dwi,
                                    add_atlas=args.add_atlas, embed_ui=args.ui==2, offscreen=False, 
                                    jupyter=jupyter, dark_mode=args.dark_mode)
            else:
                self.viewer.initialize(embed_ui=args.ui==2, offscreen=qt_mode, jupyter=jupyter)
                if args.test_data:
                    # Test data
                    points = np.random.random((500, 3)) * 1000
                    self.viewer.add_points(points, radius=10)

            if callable is not None:
                callable(self.viewer)
            self.viewer.show()
            return self.viewer

def main(auto_close_viewer=True):
    app = IBLViewer()
    app.launch()
    if auto_close_viewer:
        app.viewer.close()
    return app

if __name__ == '__main__':
    app = main()