
from datetime import datetime
import logging
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle

import vtk
from vedo import *
from vedo.addons import *
import ibllib.atlas as atlas

from iblviewer.atlas_model import AtlasModel
from iblviewer.volume_view import VolumeView
from iblviewer.slicer_view import SlicerView
import iblviewer.utils as utils


class AtlasView():
    
    BASE_PATH = utils.split_path(os.path.realpath(__file__))[0]
    SLICE_MESH_SUFFIX = 'Slice'

    def __init__(self, plot, model):
        """
        Constructor
        :parma plot: Plotter instance
        :param model: AtlasModel instance
        """
        self.plot = plot
        self.model = model
        self.volume_view = None

        self.slicer_views = []

        # Embed UI stuff (optional)
        self.region_info_text = None
        self.region_info_point = None

        self.ui_actors = []
        self.scalars_button = None
        self.coronal_button = None
        self.sagittal_button = None
        self.axial_button = None
        self.ortho_button = None
        self.axes_button = None

    def set_renderer(self):
        """
        Set VTK renderer.
        OSPRay is not supported (2021) by default and there is no
        pip wheel for it with vtk, or paraview or any vtk-based tool.
        So you can only rely on OSPRay if you compile it alongside VTK
        """
        renderer = self.plot.renderer
        try:
            if(use_ospray):
                osprayPass= vtkOSPRayPass()
                renderer.SetPass(osprayPass)

                osprayNode=vtkOSPRayRendererNode()
                osprayNode.SetSamplesPerPixel(4,renderer)
                osprayNode.SetAmbientSamples(4,renderer)
                osprayNode.SetMaxFrames(4, renderer)
                logging.info("Render info: using OSPRay.")
            else:
                logging.info("Render info: using OpenGL.")
        except (ImportError, NameError):
            logging.info("Render info: VTK is not built with OSPRay support. Using OpenGL.")

        settings.useDepthPeeling = True
        settings.useFXAA = True
        settings.multiSamples = 0
        #print('Renderer size', renderer.GetSize())

    def add_lines(self, start_points, end_points=None, spherical_angles=None, radians=True, values=None, use_origin=True, relative=False, add_to_scene=True):
        """
        Add a set of lines with given end points
        :param start_points: 3D numpy array of points of length n
        :param end_points: 3D numpy array of points of length n
        :param spherical_angles: 3D numpy array of spherical angle data of length n 
        In case end_points is None, this replaces end_points by finding the relative
        coordinate to each start point with the given radius/depth, theta and phi
        :param values: 1D list of length n, for one scalar value per line
        :param radians: Whether the given spherical angle data is in radians or in degrees
        :param use_origin: Whether the current origin (not necessarily absolute 0) is used as offset
        :param relative: Whether the given end point is relative to the start point. False by default,
        except is spherical coordinates are given
        :param add_to_scene: Whether the new lines are added to scene/plot and rendered
        :return: Lines
        """
        #points = df[['x', 'y', 'z']].to_numpy()
        #angles = df[['depth', 'theta', 'phi']].to_numpy()
        #probes_data = df[['x', 'y', 'z', 'depth', 'theta', 'phi']].to_numpy()
        if len(start_points) != len(end_points):
            logging.error('Mismatch between start and end points length. Fix your data and call add_lines() again.')
            return

        if use_origin:
            start_points = -1 * start_points + self.model.origin

        if values is None:
            values = np.arange(len(start_points))
            
        if end_points is None:
            relative = True
            if radians:
                end_points = spherical_angles.apply(spher2cart)
            else:
                end_points = spherical_angles.apply(utils.spherical_degree_angles_to_xyz)
        
        if relative:
            end_points += start_points

        distances = np.linalg.norm(end_points - start_points)
        # Lines is a single object. It's the same principle as grouping particles into one object
        lines = Lines(start_points, end_points).lw(1).cmap('Accent', distances, on='cells')
        #lines.addCellArray(values, 'scalars')
        lines.name = 'probes'
        if add_to_scene:
            self.plot.add(lines)
        return lines

    def add_points(self, positions, values, radii=5, color_map='viridis', name='neurons', use_origin_as_offset=True, noise_amount=0, add_to_scene=True, as_spheres=True):
        """
        Add points to the View
        :param positions: 3D array of coordinates
        :param values: 1D array of values, one per neuron
        :param radii: List same length as positions of radii. The default size is 5um, or 5 pixels
        in case as_spheres is False.
        :param color_map: A color map, can be a color map built with model.build_color_map(), 
        a color map name (see vedo documentation), a list of values, etc.
        :param name: All point neurons are grouped into one object, you can give it a custom name
        :parma use_origin_as_offset: Whether the origin is added as offset to the given positions
        :param noise_amount: Amount of 3D random noise applied to each point. Defaults to 0
        :param add_to_scene: Whether the new lines are added to scene/plot and rendered
        :param as_spheres: Whether the points are spheres, which means their size is relative to 
        the 3D scene (they will get bigger if you move the camera closer to them). On the other hand,
        if points are rendered as points, their radius in pixels will be constant, however close or
        far you are from them (which can lead to unwanted visual results)
        :return: Either Points or Spheres, depending on as_sphere param
        """
        if isinstance(radii, int) or isinstance(radii, float):
            radii = [radii] * len(positions)

        if use_origin_as_offset:
            positions += self.model.origin
        if noise_amount > 0:
            positions += np.random.rand(len(positions), 3) * noise_amount

        if as_spheres:
            points = Spheres(positions, r=radii)
        else:
            points = Points(positions, r=radii)
        points.lighting(0).pickable(True).cmap(color_map, values, on='cells')
        points.name = name
        if add_to_scene:
            self.plot.add(points)
        return points

    def update(self):
        # if some 
        pass
        self.plot.show(self.plot.actors, at=0, interactive=False)

    def initialize(self):
        """
        Initialize the atlas viewer
        """
        logging.info('\n\nStarting brain atlas View...\n')

        # N is the number of windows/plots
        self.set_renderer()
        
        """
        self.origin = Sphere([0, 0, 0], r=50, c='black').pickable(0)
        self.origin.name = 'Origin'
        self.origin.pickable(True)
        self.plot.add(self.origin, render=False)
        """
        
        # E.g.: KeyPressEvent, RightButtonPressEvent, MouseMoveEvent, ..etc
        #self.plot.interactor.RemoveAllObservers()
        
        #self.atlas_origin = Cross3D(self.model.origin, s=50, c='yellow').lighting(0)
        #self.atlas_origin_label = Text('Bregma origin', pos=self.atlas_origin.pos()+[.3,0,0], s=self.model.ui.font_scale*100, c='k').followCamera()
        #self.plot.add([self.atlas_origin])#, self.atlas_origin_label])


    # -----------------------------------------------------------------
    # WIP: use this as a utility
    import trimesh
    def ray_cast(self, origins, directions):
        locations, index_ray, index_tri = mesh.ray.intersects_location(
            ray_origins=origins, ray_directions=directions)
    # -----------------------------------------------------------------