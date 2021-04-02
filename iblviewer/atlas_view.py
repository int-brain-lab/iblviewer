
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
import vedo
from vedo import settings
import ibllib.atlas as atlas

from iblviewer.atlas_model import AtlasModel
from iblviewer.volume_view import VolumeView
from iblviewer.slicer_view import SlicerView
import iblviewer.utils as utils


class AtlasView():
    
    BASE_PATH = utils.split_path(os.path.realpath(__file__))[0]

    def __init__(self, plot, model):
        """
        Constructor
        :parma plot: Plotter instance
        :param model: AtlasModel instance
        """
        self.plot = plot
        self.model = model
        self.volume = None

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
                #logging.info("Render info: using OSPRay.")
            else:
                #logging.info("Render info: using OpenGL.")
                pass
        except (ImportError, NameError):
            #logging.info("Render info: VTK is not built with OSPRay support. Using OpenGL.")
            pass

        settings.useDepthPeeling = True
        settings.useFXAA = True
        settings.multiSamples = 0

    def new_segments(self, start_points, end_points=None, line_width=2, relative_end_points=False, 
    spherical_angles=None, radians=True, values=None, use_origin=True, trim_outliers=True, add_to_scene=False):
        """
        [Please use add_segments instead of new_segments]
        """
        return self.add_segments(start_points, end_points, line_width, relative_end_points, spherical_angles, radians, values, use_origin, trim_outliers, add_to_scene)

    def add_segments(self, start_points, end_points=None, line_width=2, relative_end_points=False, 
    spherical_angles=None, radians=True, values=None, use_origin=True, trim_outliers=True, add_to_scene=False):
        """
        Add a set of segments
        :param start_points: 3D numpy array of points of length n
        :param end_points: 3D numpy array of points of length n
        :param line_width: Line width, defaults to 2px
        :param relative_end_points: Whether the given end point is relative to the start point. False by default,
        except is spherical coordinates are given
        :param spherical_angles: 3D numpy array of spherical angle data of length n 
        In case end_points is None, this replaces end_points by finding the relative
        coordinate to each start point with the given radius/depth, theta and phi
        :param radians: Whether the given spherical angle data is in radians or in degrees
        :param values: 1D list of length n, for one scalar value per line
        :param use_origin: Whether the current origin (not necessarily absolute 0) is used as offset
        :param trim_outliers: Whether segments that are out of the brain envelope are trimmed or not. True by default
        :param add_to_scene: Whether the new lines are added to scene/plot and rendered
        :return: Lines
        """
        if end_points is None and spherical_angles is not None:
            relative_end_points = True
            spherical_angles = np.array(spherical_angles)
            if radians:
                end_points = spherical_angles.apply(vedo.spher2cart)
            else:
                end_points = spherical_angles.apply(utils.spherical_degree_angles_to_xyz)
        elif end_points is None:
            # We assume start_points are segments (arrays of two 3d arrays)
            end_points = start_points[:, -1]
            start_points = start_points[:, 0]
        elif end_points is not None and len(start_points) != len(end_points):
            logging.error('Mismatch between start and end points length. Fix your data and call add_lines() again.')
            return
        
        if relative_end_points:
            end_points += start_points

        point_sets = np.c_[start_points, end_points].reshape(-1, 2, 3)
        return self.add_lines(point_sets, line_width, values, use_origin, trim_outliers, add_to_scene)

    def new_lines(self, point_sets, line_width=2, values=None, use_origin=True, trim_outliers=True, add_to_scene=False):
        """
        [Please use add_lines instead of new_lines]
        """
        return self.add_lines(point_sets, line_width, values, use_origin, trim_outliers, add_to_scene)

    def add_lines(self, point_sets, line_width=2, values=None, use_origin=True, trim_outliers=True, add_to_scene=False):
        """
        Create a set of lines with given point sets
        :param point_sets: List of lists of 3D coordinates
        :param line_width: Line width, defaults to 2px
        :param spherical_angles: 3D numpy array of spherical angle data of length n 
        In case end_points is None, this replaces end_points by finding the relative
        coordinate to each start point with the given radius/depth, theta and phi
        :param values: 1D list of length n, for one scalar value per line
        :param radians: Whether the given spherical angle data is in radians or in degrees
        :param use_origin: Whether the current origin (not necessarily absolute 0) is used as offset
        :param trim_outliers: Whether segments that are out of the brain envelope are trimmed or not. True by default
        :param add_to_scene: Whether the new lines are added to scene/plot and rendered
        :return: Lines
        """
        target =  list(point_sets.keys()) if isinstance(point_sets, dict) else range(len(point_sets))
        points_lists = []
        indices = []
        line_id = 0
        for index in target:
            point_set = point_sets[index]
            point_set = np.array(point_set).astype(float)
            if use_origin:
                point_set = point_set * [[1, -1, -1]] + self.model.origin
            points_lists.append(point_set)
            indices.append(line_id)
            line_id += 1

        if values is None:
            values = np.arange(len(point_sets))

        #distances = np.linalg.norm(end_points - start_points)
        # Lines is a single object. It's the same principle as grouping particles into one object
        lines = utils.LinesExt(points_lists).cmap('Accent', indices, on='cells')
        lines.addCellArray(values, 'ids')
        lines.lighting(0)
        lines.pickable(True)
        lines.lw(line_width)
        lines.name = AtlasModel.LINES_PREFIX

        if trim_outliers and self.volume.bounding_mesh is not None:
            lines.cutWithMesh(self.volume.bounding_mesh)
        if add_to_scene:
            self.plot.add(lines)
        return lines
        
    def new_points(self, positions, radius=10, values=None, color_map='viridis', use_origin=True, noise_amount=0, as_spheres=True, add_to_scene=False):
        """
        [Please use add_points instead of new_points]
        """
        return self.add_points(positions, radius, values, color_map, use_origin, noise_amount, as_spheres, add_to_scene)

    def add_points(self, positions, radius=10, values=None, color_map='viridis', use_origin=True, noise_amount=0, as_spheres=True, add_to_scene=False):
        """
        Add new points as circles or spheres
        :param positions: 3D array of coordinates
        :param radius: List same length as positions of radii. The default size is 5um, or 5 pixels
        in case as_spheres is False.
        :param values: 1D array of values, one per neuron or a time series
        :param color_map: A color map, can be a color map built with model.build_color_map(), 
        a color map name (see vedo documentation), a list of values, etc.
        :param name: All point neurons are grouped into one object, you can give it a custom name
        :parma use_origin: Whether the origin is added as offset to the given positions
        :param noise_amount: Amount of 3D random noise applied to each point. Defaults to 0
        :param as_spheres: Whether the points are spheres, which means their size is relative to 
        the 3D scene (they will get bigger if you move the camera closer to them). On the other hand,
        if points are rendered as points, their radius in pixels will be constant, however close or
        far you are from them (which can lead to unwanted visual results)
        :param add_to_scene: Whether the new lines are added to scene/plot and rendered
        :return: Either Points or Spheres, depending on as_sphere param
        """
        if use_origin:
            positions = positions * [[1, -1, -1]] + self.model.origin
        if noise_amount is not None:
            positions += np.random.rand(len(positions), 3) * noise_amount

        if values is not None:
            if isinstance(values, np.ndarray) and values.shape[1] > 1:
                # handle case where we want a time series
                # temporary default
                values = np.array(values[:, 0]) # TODO: change this and handle time series
                min_value = min(values)
                max_value = max(values)
                colors = []
                for value in values:
                    colors.append(list(vedo.colorMap(value, color_map, min_value, max_value)))

        if as_spheres:
            points = utils.SpheresExt(positions, r=radius, c=colors)
            points.cmap(color_map, values, on='points')
        else:
            points = vedo.Points(positions, r=radius)
            points.cmap(color_map, values, on='points')
        points.lighting('off')
        points.pickable(True)
        #points.color([0.5, 0.5, 0.5])
        points.name = AtlasModel.POINTS_PREFIX
        if add_to_scene:
            self.plot.add(points)
        return points

    def add_point_cloud(self, positions, point_radius=2, auto_xy_rotate=True, add_to_scene=False):
        """
        Test method that validates that VTK is fast enough for displaying 10 million points interactively
        """
        if positions is None:
            try:
                points_path = utils.get_local_data_file_path('mouse_brain_neurons', extension='npz')
                positions = np.load(points_path, allow_pickle=True)
                positions = positions['arr_0']
            except Exception:
                # We sample a cube if you don't have the pickle file for neurons in the brain
                positions = np.random.rand(1000000, 3) * 10000
        values = np.random.rand(len(positions)) * 1.0
        point_cloud = self.add_points(positions, point_radius, values, use_origin=False, as_spheres=False)
        if auto_xy_rotate:
            point_cloud.rotateX(90)
            point_cloud.rotateZ(90)
        if add_to_scene:
            self.plot.add(point_cloud)
        return point_cloud

    def add_glyphs(self, positions):
        raise NotImplementedError
        # WIP
        num_points = len(positions)
        polydata = vtk.vtkPolyData()
        points = vtk.vtkPoints()
        points.SetNumberOfPoints(num_points)

        cells = vtk.vtkCellArray()
        scalars = vtk.vtkDoubleArray()
        scalars.SetName('ScalarArray')

        for p_id in range(num_points):
            points.SetPoint(p_id, *positions[p_id])
            cells.InsertNextCell(1)
            cells.InsertCellPoint(p_id)
        #points.InsertPoints(positions)

        polydata.SetPoints(points)
        polydata.SetVerts(cells)
        polydata.GetPointData().SetScalars(scalars)
        polydata.GetPointData().SetActiveScalars('ScalarArray')

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        mapper.SetColorModeToDefault()
        mapper.SetScalarRange(0, num_points)
        mapper.SetScalarVisibility(1)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        return actor
        '''
        if glyph_actor is None:
            glyph_actor = Ellipsoid()
        #glyph = utils.SpheresExt(positions, r=10)
        glyph = Glyph(positions, glyph_actor)
        glyph.name = '[Glyph]'
        glyph.lighting('off')
        #glyph.cmap('viridis')#addScalarBar
        '''

    def initialize(self):
        """
        Initialize the atlas viewer
        """
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
