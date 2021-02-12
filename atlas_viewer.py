from vedo import load, datadir
from vedo.applications import RayCaster, Slicer

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from datetime import datetime
import numpy as np
import nrrd
import os
import pandas as pd
import pickle
#import pyvista as pv
from vedo import *
from vedo.addons import *
import vtk
import math
import random

import ibllib.atlas as atlas

"""
Project: IBL Viewer
Description: Interactive volumetric and surface data visualizer that 
integrates with the Python ecosystem that is widely used nowadays

Copyright: 2021 Nicolas Antille, International Brain Laboratory
License: MIT
"""

#pv.rcParams['use_ipyvtk'] = True

def change_file_name(file_path, prefix=None, name=None, suffix=None):
    """
    Change the file name from the given file path
    :param file_path: Input file path
    :param prefix: Prefix to the file name
    :param name: Whether a new name is set instead of the current name.
    If None, the current file name is used.
    :param suffix: Suffix to the file name
    :return: New file path
    """
    path, file_name, extension = split_path(file_path)
    if prefix is None:
        prefix = ''
    if suffix is None:
        suffix = ''
    if name is None or name == '' or not isinstance(name, str):
        name = file_name
    return os.path.join(path, prefix + name + suffix + extension)


def split_path(path):
    """
    Split any given file path to folder path, file name and extension
    :return: Folder path, file name and extension
    """
    base_name = os.path.basename(path)
    file_name, extension = os.path.splitext(base_name)
    return path[:-len(base_name)], file_name, extension

def time_diff(t):
    now  = datetime.now()
    duration = now - t
    return duration.total_seconds()


class Slicer():

    slices = []

    def __init__(self, plot, volume_actor, resolution, color_map=None, lut=None, surface=None):
        """
        Constructor
        """
        Slicer.slices.append(self)
        self.id = len(Slicer.slices) - 1

        self.plot = plot
        self.slice = None
        self.normal = np.array([1.0, 0.0, 0.0])
        self.volume_actor = volume_actor
        self.color_map = color_map
        self.opacity_map = None
        self.resolution = resolution
        self.lut = lut

        self.interactor_plane = None

        self.volume_range = 0
        self.volume_dimensions = np.array(self.volume_actor.dimensions()).astype(np.float64) * self.resolution
        self.volume_center = np.array(self.volume_actor.pos()) + np.array(self.volume_actor.center())

        # First method
        self.value = 0
        self.brain_surface = surface
        self.cut_brain_surface = None

        # Second method wip
        self.origin = np.array([0.0, 0.0, 0.0])

    def get_transformation_matrix(self, origin, normal):
        """
        Get transformation matrix for a plane given by its origin and normal
        :param origin: Origin 3D vector
        :param normal: Normal 3D vector
        :return: Matrix and Translation
        """
        newaxis = utils.versor(normal)
        initaxis = (0, 0, 1)
        crossvec = np.cross(initaxis, newaxis)
        angle = np.arccos(np.dot(initaxis, newaxis))
        T = vtk.vtkTransform()
        T.PostMultiply()
        T.RotateWXYZ(np.rad2deg(angle), crossvec)
        T.Translate(np.array(origin))
        M = T.GetMatrix()
        return M, T

    def reslice(self, volume, origin, normal):
        """
        Reslice volume, can be slow on large volumes
        """
        reslice = vtk.vtkImageReslice()
        #reslice.SetInputData(image)
        reslice.SetOutputDimensionality(2)
        reslice.SetAutoCropOutput(False)
        #reslice.SetInterpolator(interpolateMethod)
        reslice.SetInputData(self.volume_actor._data)

        M, T = self.get_transformation_matrix(origin, normal)
        reslice.SetResliceAxes(M)
        
        reslice.SetInterpolationModeToLinear()
        reslice.SetSlabNumberOfSlices(1)
        reslice.Update()

        slice_image = vtk.vtkImageDataGeometryFilter()
        slice_image.SetInputData(reslice.GetOutput())
        slice_image.Update()

        mesh = Mesh(slice_image.GetOutput())
        mesh.SetOrientation(T.GetOrientation())
        mesh.SetPosition(origin)
        return mesh

    def update(self, raw_value=None, normal=None, axis=None, plot=None):
        """
        Update slicer with given value
        """
        normal_changed = normal is not None and (normal != self.normal).any()
        normal = normal if normal_changed else self.normal

        last_value = self.value

        if raw_value is None:
            if self.value is None:
                self.value = -self.volume_range / self.resolution if raw_value != 0 else 0
        else:
            self.value = raw_value / self.resolution if raw_value != 0 else 0

        in_volume_position = np.array([0.0, 0.0, 0.0])
        in_volume_position[:] = np.array(self.volume_actor.center()) #IBLViewer.BREGMA
        in_volume_position += normal * self.value * self.resolution # - IBLViewer.BREGMA

        if axis == 0:
            axis_center = int(self.volume_dimensions[0] / 2)
            in_volume_slice = int(self.value + axis_center/self.resolution)
            current_slice = self.volume_actor.xSlice(in_volume_slice)
            offset = np.array([axis_center, 0.0, 0.0])
            normal = np.array([1.0, 0.0, 0.0])
            #self.reslice(self.volume_actor, in_volume_position, normal)
        elif axis == 2:
            # Default to Z
            axis_center = int(self.volume_dimensions[2] / 2)
            in_volume_slice = int(self.value + axis_center/self.resolution)
            current_slice = self.volume_actor.zSlice(in_volume_slice)
            offset = np.array([0.0, 0.0, axis_center])
            normal = np.array([0.0, 0.0, 1.0])
        else:
            axis_center = int(self.volume_dimensions[1] / 2)
            in_volume_slice = int(self.value + axis_center/self.resolution)
            current_slice = self.volume_actor.ySlice(in_volume_slice)
            offset = np.array([0.0, axis_center, 0.0])
            normal = np.array([0.0, 1.0, 0.0])
        #current_slice = self.volume_actor.slicePlane(origin=in_volume_position, normal=normal)
        current_slice.pickable(True)
        #current_slice.UseBoundsOff() # avoid resetting the cam
        current_slice.name = IBLViewer.SLICE_MESH_SUFFIX + '_' + str(self.id)
        current_slice.lighting('off')
        
        current_slice._mapper.SetScalarVisibility(1)
        # Without setting scalar range, the mapping will be off
        current_slice._mapper.SetScalarRange(0, len(self.color_map))
        current_slice._mapper.SetLookupTable(self.lut)
        current_slice._mapper.SetColorModeToMapScalars()
        current_slice._mapper.SetScalarModeToUsePointData()
        #current_slice.cmap(self.color_map, alpha=self.opacity_map)

        current_slice.ForceOpaqueOn()
        current_slice.pickable(True)
        
        #current_slice.rotateZ(180)
        #current_slice.scale([-1, 1, 1])
        slice_position = np.array([0.0, 0.0, 0.0])
        #slice_position[:] = self.volume_actor.pos() / self.resolution * normal
        #slice_position[:] = np.array(self.volume_actor.center())#IBLViewer.BREGMA# - IBLViewer.MESH_OFFSET #[0, 1000, 5000]# - IBLViewer.MESH_OFFSET[[1, 2, 0]]
        #slice_position[2] += self.value * scale + normal[2]
        slice_position = normal * in_volume_slice * self.resolution - offset #* self.resolution # arbitrary slicing
        #slice_position += np.array([1.0, 0.0, 0.0]) * self.value * self.resolution # x slicing
        #current_slice.pos(slice_position)

        clipping_planes = vtk.vtkPlaneCollection()
        slice_plane = vtk.vtkPlane()
        slice_position = normal * (self.value * self.resolution + offset)
        slice_plane.SetOrigin(slice_position)
        #slice_plane.SetOrigin(slice_position * self.resolution + offset)
        slice_plane.SetNormal(normal)
        clipping_planes.AddItem(slice_plane)
        self.volume_actor.mapper().SetClippingPlanes(clipping_planes)

        try:
            self.plot.remove(self.slice, render=False)
            self.plot.remove(self.interactor_plane, render=False)
        except Exception as e:
            print(e)
            pass

        #cut_brain_surface.alpha(0.1)

        self.plot.add([current_slice])
        """ if cut_brain_surface is not None:
            self.plot.add([cut_brain_surface, current_slice])
            self.cut_brain_surface = cut_brain_surface
        else:
            self.plot.add([current_slice]) """

        self.slice = current_slice
        #self.interactor_plane = interactor_plane

class IBLViewer():
    
    BASE_PATH = split_path(os.path.realpath(__file__))[0]
    VIZ_VOLUME_SUFFIX = '_m'
    SLICE_MESH_SUFFIX = 'Slice'

    ALLEN_ATLAS_RESOLUTIONS = [10, 25, 50, 100]
    MESH_OFFSET = np.array([-860.0, 1500.0, -5400.0])
    #MESH_OFFSET = np.array([0.0, -1200.0, -3650.0])

    DEFAULT_CAMERA_DISTANCE = 35000.0
    DEFAULT_CAMERA_POSITION = np.array([32000.0, 8000.0, 5000.0]) #np.array([4000.0, -4000.0, 32000.0])
    DEFAULT_FOCAL_POINT = np.array([7200.0, 4500.0, 5800.00])
    
    BREGMA = atlas.ALLEN_CCF_LANDMARKS_MLAPDV_UM['bregma'] #np.array([5739.0, 5400.0, 332.0])
    REVERSED_REFERENCE = True 

    def __init__(self):
        """
        Constructor
        """
        self.plot = None
        self.origin = None
        
        self.atlas = None # IBL AllenAtlas wrapper
        self.metadata = None # actual .csv file as pandas dataframe
        self.resolution = 0 # should be removed in favor of self.atlas.res_um
        self.lut = None
        self.volume_unique_scalars = []
        self.volume_dimensions = None
        #self.volume_scalar_min = 0.0
        #self.volume_scalar_max = 1.0
        
        self.scalars = None

        self.color_map = None
        self.volume_actor = None
        self.regions_surfaces = None

        self.slicers = []
        self.active_slicer = None

        self.normal = np.array([1.0, 0.0, 0.0])
        self.selected_region = None

        self.region_info_text = None
        self.region_info_point = None

        self.brain_surface = None
        self.decimated_brain_surface = None
        self.cut_brain_surface = None
        self.current_region_surface = None

        self.time = 0
        self.font = 'Source Sans Pro'

        self.per_region_priors_median_dsquare_test_data = None
        self.probes_insertions_test_data = None

        # UI
        self.scalars_button = None
        self.coronal_button = None
        self.sagittal_button = None
        self.axial_button = None
        self.ortho_button = None
        self.axes_button = None
        self.hollow_button = None
        self.atlas_button = None
        self.atlas_visible = True

        self.last_camera_settings = dict()

        self.mode = None
        self.last_mode = None

    def get_region_and_row_id(self, acronym):
        region_data = self.metadata[self.metadata.acronym == acronym]
        if region_data is None or len(region_data) < 1:
            return
        return region_data.id.to_numpy()[0], region_data.index[0]

    def get_atlas(self):
        return pd.read_csv(atlas.regions.FILE_REGIONS)
        """ if file_path is None:
            file_path = os.path.join(IBLViewer.BASE_PATH, 'data/allen_structure_tree.csv')
        # Can be replaced later on with IBL atlas, but if we depend on ibllib for external data only,
        # it's best to wait until we add another dependency (even if currently only IBL people have access to this viewer)
        return pd.read_csv(file_path) """

    def import_volume(self, file_path):
        # Forcing index order to C to run faster values reassignment than with default Fortran order
        volume, header = nrrd.read(file_path, index_order='C')
        return volume, header

    def set_resolution(self, resolution):
        if resolution not in IBLViewer.ALLEN_ATLAS_RESOLUTIONS:
            resolution = IBLViewer.ALLEN_ATLAS_RESOLUTIONS[-1]
        self.resolution = resolution

    def add_slicer(self, normal=None):
        """
        Initialize the UI for slicing
        """
        slicer = Slicer(self.plot, self.volume_actor, self.resolution, self.color_map, self.lut, self.brain_surface)
        self.slicers.append(slicer)
        self.active_slicer = self.slicers[-1]
        return slicer

    def get_region_color(self, region_id, not_found_color=[0.0, 0.0, 0.0]):
        """
        Get RGB color of a brain region from the Allen brain atlas
        """
        # Note @Allen brain devs: when you store non uniform data, people down the line have to bear with it
        not_always_hex_col = self.metadata.color_hex_triplet[region_id]
        rgb = not_found_color
        if isinstance(not_always_hex_col, str) and not_always_hex_col != '0':
            rgb = colors.getColor('#' + str(not_always_hex_col))
        return rgb

    def build_color_map(self, mode=-1, only_custom_data=False, alpha_factor=1.0, rand=0.0):
        """
        Build a color map for the atlas volume and slices
        """
        rgb = []
        alpha = []
        if mode == 'atlas':            
            for r_id in range(len(self.metadata)):
                rgb.append([r_id, self.get_region_color(r_id)])
                a = 1.0 if r_id > 0 else 0.0
                alpha.append([r_id, a * alpha_factor])
           
            #print('Building color map with len colors', self.metadata.color_hex_triplet.size, 'versus len regions', self.metadata.id.size)
            self.scalars = None
            
        elif mode == 'priors':
            df = self.per_region_priors_median_dsquare_test_data
            values = df['value']
            min_p = float(np.amin(values, axis=0))
            max_p = float(np.amax(values, axis=0))
            rng_p = max_p - min_p

            # Init all to clear gray (90% white)
            #c = np.ones((self.metadata.id.size, 4)).astype(np.float32) * 0.9
            #c[:, -1] = 0.0 if only_custom_data else alpha_factor
            #print('Assigning', values.size, 'to atlas ids', self.metadata.id.size)

            trans = vtk.vtkMath()
            for r_id in range(len(self.metadata)):
                color = self.get_region_color(r_id)
                hsv_col = matplotlib.colors.rgb_to_hsv(color)
                hsv_col[1] = 0.0
                desaturated_color = matplotlib.colors.hsv_to_rgb(hsv_col)
                rgb.append([r_id, desaturated_color])
                a = 1.0 if r_id > 0 and not only_custom_data else 0.0
                alpha.append([r_id, a * alpha_factor])
            
            self.scalars = {}
            random.seed(rand)
            for acronym, value in df.iterrows():
                region_id, row_id = self.get_region_and_row_id(acronym)
                #region = scene.add_brain_region(acronym, silhouette=True)
                if rand is not None and rand > 0:
                    value = float(value.to_numpy()[0]) + random.random() * (max_p - min_p)
                else:
                    value = float(value.to_numpy()[0])
                #r = (float(value.to_numpy()[0]) - min_p) / rng_p
                #c[row_id] = list(colorMap(value, "jet", min_p, max_p)) + [1.0]
                scalar_color = list(colorMap(value, "hot", min_p, max_p))
                rgb[row_id] = [row_id, scalar_color]
                alpha[row_id] = [row_id, scalar_color[0] * alpha_factor] #1.0 * alpha_factor]
                self.scalars[int(row_id)] = value

        elif isinstance(mode, int) and mode > 0:
            for r_id in range(len(self.metadata)):
                if r_id == mode:
                    rgb.append([r_id, self.get_region_color(r_id)])
                    alpha.append([r_id, 1.0 * alpha_factor])
                else:
                    rgb.append([r_id, [0.0, 0.0, 0.0]])
                    alpha.append([r_id, 0.0 * alpha_factor])
            #c = np.zeros((self.metadata.id.size, 4))
            #c[mode, :] = [1.0, 0.2, 0.2, 1.0]
        else:
            pass
            #c = np.random.rand(self.metadata.id.size, 3)
            #c = np.c_[c.astype(np.float32), np.ones(self.metadata.id.size)/2]

        """ elif isinstance(mode, int) and mode > 0:
            c = np.zeros((self.metadata.id.size, 4))
            c[mode, :] = [1.0, 0.2, 0.2, 1.0]
        else:
            c = np.random.rand(self.metadata.id.size, 3)
            c = np.c_[c.astype(np.float32), np.ones(self.metadata.id.size)/2] """

        rgb = np.array(rgb, dtype=object)
        alpha = np.array(alpha)
        # First region in atlas is not a region (void)
        alpha[0, 1] = 0.0
        #alpha[0] = 0.0
        #rgb[0, 0, :] = [1.0, 1.0, 1.0, 0.0]
        
        num_regions = len(rgb)
        lut = vtk.vtkLookupTable()
        lut.SetRange(0, num_regions)
        lut.SetNumberOfTableValues(num_regions)
        for r_id in range(num_regions):
            lut.SetTableValue(r_id, *rgb[r_id, 1], alpha[r_id, 1])
            #lut.SetTableValue(r_id, *rgb[r_id], alpha[r_id])
        lut.Build()

        return rgb, alpha, lut
    
    def add_region_surface(self, region_id=997, meshes_path='./data/allen/structure/structure_meshes/clean_ply/', ext='ply'):
        #if region_id == 997:
            #region_id = str(region_id) + 'm'
        region_mesh_path = meshes_path + str(region_id) + '.' + ext
        if os.path.exists(region_mesh_path):
            return load(region_mesh_path)

    def clip_volume(self, obj, event):

        obj.GetPlanes(self.clipping_planes)
        """ plane = planes.GetPlane(0)
        pos = plane.GetOrigin()
        normal = plane.GetNormal()) """
        vtk_n = self.clipping_planes.GetNormals()
        vtk_pts = self.clipping_planes.GetPoints()
        normals = [vtk_n.GetTuple(i) for i in range(vtk_n.GetNumberOfTuples())]
        points = [vtk_pts.GetPoint(i) for i in range(vtk_pts.GetNumberOfPoints())]
        self.volume_actor.mapper().SetClippingPlanes(self.clipping_planes)

        try:
            self.plot.remove(self.active_slicer.slice, render=False)
        except Exception:
            pass
        self.active_slicer.update(origin=points[0], normal=normals[0])

        self.plot.add([self.active_slicer.slice])

    def test_load_priors(self, file_path='./data/ibl/completefits_2020-11-09.p', verbose=False):
        pickles = []
        with (open(file_path, 'rb')) as openfile:
            while True:
                try:
                    pickles.append(pickle.load(openfile))
                except EOFError:
                    break
        
        df = pickles[0]
        if verbose:
            print('')
            for stuff in df:
                print('')
                print(stuff)
                print(df[stuff])
            print('')
            print('')

        self.per_region_priors_median_dsquare_test_data = df['rawpoints'].groupby('acronym').agg({'value':'median'})
        if verbose:
            print(self.per_region_priors_median_dsquare_test_data)
            print('Min', np.amin(self.per_region_priors_median_dsquare_test_data, axis=0))
            print('Max', np.amax(self.per_region_priors_median_dsquare_test_data, axis=0))
            print('')
        return self.per_region_priors_median_dsquare_test_data

    def get_probe_vector(self, x, y, z, depth=1, theta=0, phi=0):
        """
        Get insertion probe line
        """
        #With raw data, not "ibl-wise" [[1, 2, 0]]  #np.array([-y, -z, x]) + IBLViewer.BREGMA[[1, 2, 0]]
        start_point = np.array([x, y, z])[[2, 0, 1]] * [-1, 1, -1] + IBLViewer.BREGMA[[2, 0, 1]]
        if IBLViewer.REVERSED_REFERENCE:
            start_point = np.array([x, y, z]) * [-1, -1, -1] + IBLViewer.BREGMA
        sph = spher2cart(depth, theta / 180 * math.pi, phi / 180 * math.pi) 

        end_point = start_point - sph[[2, 0, 1]] * [-1, 1, -1] #* [-1, 1, 1]
        if IBLViewer.REVERSED_REFERENCE:
            end_point = start_point - sph * [-1, -1, -1]
        
        #line = Line(start_point, end_point).c(color).lw(2).alpha(alpha)
        #line = Cylinder(pos=[start_point, end_point], c=color, r=7, alpha=alpha)
        # Add cutoff because lines go beyond the mouse brain volume, but it's rather slow
        #line.cutWithMesh(self.decimated_brain_surface)
        return [start_point, end_point]

    def test_load_ibl_probes_data(self, file_path='./data/ibl/ephys_aligned_session_insertions.p'):
        pickles = []
        with (open(file_path, 'rb')) as openfile:
            while True:
                try:
                    pickles.append(pickle.load(openfile))
                except EOFError:
                    break
        
        df = pickles[0]
        self.probes_insertions_test_data = df
        points = df[['x', 'y', 'z']].to_numpy()
        angles = df[['depth', 'theta', 'phi']].to_numpy()
        probes_data = df[['x', 'y', 'z', 'depth', 'theta', 'phi']].to_numpy()

        starts = []
        ends = []
        distances = []
        for single_probe_data in probes_data:
            pt1, pt2 = self.get_probe_vector(*single_probe_data)
            starts.append(pt1)
            ends.append(pt2)
            distances.append(np.linalg.norm(pt2-pt1))

        # Lines is a single object. It's the same principle as grouping particles into one object
        lines = Lines(starts, ends).lw(1).cmap('Accent', distances, on='cells')
        lines.addCellArray(np.arange(len(probes_data)), "scalars")
        lines.name = 'probes'
        self.plot.add(lines)
        return lines

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
                print("Using OSPRay")
                osprayPass= vtkOSPRayPass()
                renderer.SetPass(osprayPass)

                osprayNode=vtkOSPRayRendererNode()
                osprayNode.SetSamplesPerPixel(4,renderer)
                osprayNode.SetAmbientSamples(4,renderer)
                osprayNode.SetMaxFrames(4, renderer)
            else:
                print("Using OpenGL")
        except (ImportError, NameError):
            print("VTK is not built with OSPRay support. Using OpenGL")
        print('Renderer size', renderer.GetSize())
        
    def get_region_info(self, mesh):
        """
        Handle event
        """
        if IBLViewer.SLICE_MESH_SUFFIX in mesh.name:
            ptid = self.active_slicer.slice.closestPoint(mesh.picked3d, returnPointId=True)
            # Scalar values in volume are integers in this case
            print('Picked pt id', ptid)
            value = int(self.active_slicer.slice.getPointArray()[ptid])
            print('Picked scalar id', value)
            #print('Selected', mesh.name, 'at', mesh.picked3d, '-> id', ptid, 'with mapped scalar', value)
            #print('Raw value', self.active_slicer.slice.getPointArray()[ptid], 'with color', self.color_map[value])
        elif mesh.name == 'probes':
            return
            # WIP
            ptid = mesh.closestPoint(mesh.picked3d, returnPointId=True)
            value = int(mesh.getCellArray('scalars')[ptid])
        
        txt = None
        if mesh.name.startswith(IBLViewer.SLICE_MESH_SUFFIX):
            txt = 'Selected region [' + str(value) + ']' + ' Atlas ID: ' 
            txt += str(self.metadata.id[value]) + ' - ' + str(self.metadata.name[value])
            if isinstance(self.scalars, dict):
                #print(list(self.scalars.keys()))
                txt += '\n\nScalar value: ' + str(self.scalars.get(value, 'none found'))
        pos = mesh.points(ptid)
        
        info_point = Sphere(pos, r=20, c='white').pickable(0)
        #vig = vpt.vignette(txt, c='black', offset=(20,10)).followCamera()
        info_text = None
        if txt is not None:
            info_text = Text2D(txt, c='black', pos=[0.2, 0.92], font=self.font, s=0.75, justify='left')

        if self.region_info_text is not None:
            self.plot.remove(self.region_info_text, render=False)
            self.plot.remove(self.region_info_point, render=False)
        
        """ 
        Very slow and cannot be targetting one value only
        selected_region = self.volume_actor.isosurface(value, True)#.smoothLaplacian().lineWidth(1)
        if self.selected_region is not None:
            self.plot.remove(self.selected_region)
        self.plot.add(selected_region)
        self.selected_region = selected_region 
        """

        if info_point is not None:
            self.plot.add(info_point)
            self.region_info_point = info_point
        if info_text is not None:
            self.plot.add(info_text)
            self.region_info_text = info_text

    def get_ui_pos(self, x, y, length, horizontal=True, absolute=True):
        """
        Get relative position within UI
        """
        if horizontal:
            return np.array([[x, y], [x + length, y]])
        else:
            return np.array([[x, y], [x, y + length]])

    def add_slicer_ui(self, slicer=None):
        """
        Add slicer UI. Currently, this is made for only one slicer but it can be extended.
        """

        # Not used yet, but the goal is to add a UI panel for a given slicer
        if slicer is None:
            slicer = self.active_slicer
        
        extra_margin = 50
        rng_v = max(self.volume_dimensions) / 2 + extra_margin
        rng_v2 = max(self.volume_dimensions) + extra_margin
        
        # Init slicer
        slicer.volume_range = rng_v
        slicer.update(-rng_v)
        #print('Volume dimensions', volume_dimensions)

        def update_value(widget=None, event=None, value=-rng_v):
            if widget is not None and event is not None:
                value = widget.GetRepresentation().GetValue()
            self.active_slicer.update(value, self.normal)

        s_kwargs = {'titleSize':0.75, 'font':self.font}
        ui_pos = self.get_ui_pos(0.05, 0.065, 0.4)
        slider = self.plot.addSlider2D(update_value, -rng_v, rng_v, -rng_v, ui_pos, title='Slicing plane', **s_kwargs)
        slider.GetRepresentation().SetLabelHeight(0.022*s_kwargs['titleSize'])
        
        def update_x_normal(widget=None, event=None, value=-rng_v):
            if widget is not None and event is not None:
                value = widget.GetRepresentation().GetValue()
            normal = np.copy(self.normal)
            normal[0] = value
            self.active_slicer.update(None, normal)
            self.normal = normal

        def update_y_normal(widget=None, event=None, value=-rng_v):
            if widget is not None and event is not None:
                value = widget.GetRepresentation().GetValue()
            normal = np.copy(self.normal)
            normal[1] = value
            self.active_slicer.update(None, normal)
            self.normal = normal

        def update_z_normal(widget=None, event=None, value=-rng_v):
            if widget is not None and event is not None:
                value = widget.GetRepresentation().GetValue()
            normal = np.copy(self.normal)
            normal[2] = value
            self.active_slicer.update(None, normal)
            self.normal = normal
        
        '''
        n = self.normal
        ui_pos = self.get_ui_pos(0.05, 0.15, 0.12)
        slider = self.plot.addSlider2D(update_x_normal, -1.0, 1.0, n[0], ui_pos, title='Normal X', **s_kwargs)
        slider.GetRepresentation().SetLabelHeight(0.022*s_kwargs['titleSize'])

        ui_pos = self.get_ui_pos(0.2, 0.15, 0.12)
        slider = self.plot.addSlider2D(update_y_normal, -1.0, 1.0, n[1], ui_pos, title='Normal Y', **s_kwargs)
        slider.GetRepresentation().SetLabelHeight(0.022*s_kwargs['titleSize'])

        ui_pos = self.get_ui_pos(0.35, 0.15, 0.12)
        slider = self.plot.addSlider2D(update_z_normal, -1.0, 1.0, n[2], ui_pos, title='Normal Z', **s_kwargs)
        slider.GetRepresentation().SetLabelHeight(0.022*s_kwargs['titleSize'])
        '''

        def update_alpha(widget=None, event=None, value=-rng_v):
            if widget is not None and event is not None:
                value = widget.GetRepresentation().GetValue()
            self.volume_actor.alphaUnit(value)

        ui_pos = self.get_ui_pos(0.5, 0.065, 0.12)
        slider = self.plot.addSlider2D(update_alpha, 0.1, 10.0, 1.0, ui_pos, title='Opacity', **s_kwargs)
        slider.GetRepresentation().SetLabelHeight(0.022*s_kwargs['titleSize'])

        def update_selection(widget=None, event=None, value=-rng_v):
            if widget is not None and event is not None:
                value = widget.GetRepresentation().GetValue()
            self.reveal_region(int(value))

        """
        ui_pos = self.get_ui_pos(0.5, 0.15, 0.35)
        slider = self.plot.addSlider2D(update_selection, 0, len(self.metadata), 0, ui_pos, title='Mesh region', **s_kwargs)
        slider.GetRepresentation().SetLabelHeight(0.022*s_kwargs['titleSize'])
        """

        def update_time_series(widget=None, event=None, value=-rng_v):
            if widget is not None and event is not None:
                value = widget.GetRepresentation().GetValue()
            self.dummy_time_series(value)

        ui_pos = self.get_ui_pos(0.65, 0.065, 0.12)
        slider = self.plot.addSlider2D(update_time_series, 0, 100, 0, ui_pos, title='Time series', **s_kwargs)
        slider.GetRepresentation().SetLabelHeight(0.022*s_kwargs['titleSize'])

    def dummy_time_series(self, value):
        """
        Time series test
        """
        self.scalars_button.switch()
        mode = 'priors' if self.mode == 'atlas' else 'atlas'
        self.color_map, self.opacity_map, self.lut = self.build_color_map('priors', rand=value)
        col_map, op_map, lut = self.build_color_map('priors', rand=value, alpha_factor=0.001)
        self.volume_actor.c(col_map)
        self.volume_actor.alpha(op_map)
        self.plot.remove(self.volume_actor.scalarbar)
        self.volume_actor.addScalarBar(useAlpha=False)
        self.plot.add(self.volume_actor.scalarbar)
        self.active_slicer.color_map = self.color_map
        self.active_slicer.lut = self.lut
        self.active_slicer.slice._mapper.SetLookupTable(self.lut)
        self.mode = mode

    def add_button(self, *args, **kwargs):
        """
        Add a left-aligned button otherwise positionning it in the UI is a nightmare
        :param *args: List of arguments for addButton()
        :param **kwargs: Dictionary of arguments for addButton()
        :return: Button
        """
        button = self.plot.addButton(*args, **kwargs)
        button.actor.GetTextProperty().SetJustificationToLeft()
        button.actor.SetPosition(kwargs['pos'][0], kwargs['pos'][1])
        return button

    def add_camera_ui(self):
        """
        Add camera controls as buttons on scene
        """
        t_kwargs = {'c':["black", "#eeeeee"], 'bc':["#dddddd", "#999999"], 'font':self.font, 'size':12, 'bold':False, 'italic':False}
        kwargs = {'c':["black", "black"], 'bc':["#dddddd", "#dddddd"], 'font':self.font, 'size':12, 'bold':False, 'italic':False}

        self.coronal_button = self.add_button(self.toggle_coronal_view, pos=(0.05, 0.94), states=["Coronal", "Coronal"], **kwargs)
        self.sagittal_button = self.add_button(self.toggle_sagittal_view, pos=(0.05, 0.90), states=["Sagittal", "Sagittal"], **kwargs)
        self.axial_button = self.add_button(self.toggle_axial_view, pos=(0.05, 0.86), states=["Axial", "Axial"], **kwargs)
        self.ortho_button = self.add_button(self.toggle_orthographic_view, pos=(0.05, 0.82), states=["Orthographic", "Orthographic"], **t_kwargs)
        #self.axes_button = self.add_button(self.toggle_axes_visibility, pos=(0.05, 0.78), states=["Show axes", "Hide axes"], **t_kwargs)
        self.hollow_button = self.add_button(self.toggle_hollow_mode, pos=(0.05, 0.74), states=["Hollow mode", "Hollow mode"], **t_kwargs)
        self.atlas_button = self.add_button(self.toggle_atlas_visibility, pos=(0.05, 0.70), states=["Hide atlas", "Show atlas"], **t_kwargs)
        self.scalars_button = self.add_button(self.toggle_scalars, pos=(0.05, 0.66), states=["Demo scalars", "Atlas colors"], **kwargs)
        
        # Retrieves data about hover
        #self.plot.addCallback('HoverEvent', self.hover_slice_event)

    def toggle_axes_visibility(self):
        self.axes_button.switch()
        pass

    def toggle_atlas_visibility(self):
        self.atlas_button.switch()
        if self.atlas_visible:
            """
            self.plot.remove(self.volume_actor)
            self.plot.remove(self.volume_actor.scalarbar)
            self.plot.remove(self.active_slicer.slice)
            """
            self.volume_actor.alpha(0)
            self.plot.remove(self.volume_actor.scalarbar)
            self.plot.remove(self.active_slicer.slice)
        else:
            self.volume_actor.alpha(self.opacity_map * 0.001)
            self.volume_actor.addScalarBar(useAlpha=False)
            self.plot.add(self.volume_actor)
            self.active_slicer.update()
            #self.plot.add([self.volume_actor, self.volume_actor.scalarbar, self.active_slicer.slice])
        self.atlas_visible = not self.atlas_visible

    def toggle_scalars(self):
        """
        Toggle custom scalars data
        """
        self.scalars_button.switch()
        mode = 'priors' if self.mode == 'atlas' else 'atlas'
        self.color_map, self.opacity_map, self.lut = self.build_color_map(mode)
        col_map, op_map, lut = self.build_color_map(mode, mode == 'priors', alpha_factor=0.001)
        self.volume_actor.c(col_map)
        self.volume_actor.alpha(op_map)
        self.plot.remove(self.volume_actor.scalarbar)
        self.volume_actor.addScalarBar(useAlpha=False)
        self.plot.add(self.volume_actor.scalarbar)
        self.active_slicer.color_map = self.color_map
        self.active_slicer.lut = self.lut
        self.active_slicer.slice._mapper.SetLookupTable(self.lut)
        self.mode = mode

    def save_camera_position(self):
        """
        Save camera position in order to reset it later (useful for toggle actions below)
        """
        pass
        #self.last_camera_position = None
        #self.last_camera_focal_point = None
        #self.last_camera_distance = None
        #self.last_camera_settings = None

    def reveal_region(self, region, invert=False):
        self.plot.remove(self.current_region_surface, render=False)
        self.current_region_surface = self.volume_actor.threshold(region, region).isosurface(region).computeNormals().smoothLaplacian().alpha(0.2).c(self.get_region_color(region))
        self.plot.add(self.current_region_surface)
        '''
        alpha_map = []
        if region == 0:
            self.volume_actor.alpha(self.volume_opacity_map)
        for r_id in range(len(self.metadata)):
            alpha_value = 1.0 if r_id == region else 0.0
            alpha_map.append([r_id, alpha_value])
        self.volume_actor.alpha(alpha_map)
        '''

    def toggle_hollow_mode(self):
        self.hollow_button.switch()

        volume_property = self.volume_actor.GetProperty()
        # This is twisted to name properties Disable and then have DisableOff...
        disabled = bool(volume_property.GetDisableGradientOpacity())
        if disabled:
            volume_property.DisableGradientOpacityOff()
            alpha_gradient = vtk.vtkPiecewiseFunction()
            alpha_gradient.AddPoint(0, 0.0)
            alpha_gradient.AddPoint(1, 0.75)
            alpha_gradient.AddPoint(2, 1.0)
            volume_property.SetGradientOpacity(alpha_gradient)
        else:
            volume_property.DisableGradientOpacityOn()

    def toggle_orthographic_view(self):
        """
        Toggle orthographic/perspective views
        """
        self.ortho_button.switch()
        settings.useParallelProjection = not settings.useParallelProjection
        self.plot.camera.SetParallelScale(self.plot.camera.GetDistance() / 4)
        self.plot.camera.SetParallelProjection(settings.useParallelProjection)

    def toggle_coronal_view(self):
        """
        Toggle coronal view
        """
        if IBLViewer.REVERSED_REFERENCE:
            self.plot.camera.SetFocalPoint([5700, 6600, 4000])
            self.plot.camera.SetViewUp(*[0.0, 0.0, -1.0])
            self.plot.camera.SetPosition([5700, 6600-IBLViewer.DEFAULT_CAMERA_DISTANCE, 4000])
        else:
            self.plot.camera.SetFocalPoint([4000, 5700, 6600])#IBLViewer.DEFAULT_FOCAL_POINT)
            self.plot.camera.SetViewUp(*[-1.0, 0.0, 0.0])
            self.plot.camera.SetPosition([4000, 5700, 6600-IBLViewer.DEFAULT_CAMERA_DISTANCE])
        #*IBLViewer.DEFAULT_CAMERA_POSITION[[0, 1, 2]])
        #self.plot.camera.SetDistance(IBLViewer.DEFAULT_CAMERA_DISTANCE)
        #self.plot.renderer.ResetCamera()

    def toggle_sagittal_view(self):
        """
        Toggle sagittal view
        """
        if IBLViewer.REVERSED_REFERENCE:
            self.plot.camera.SetFocalPoint([5700, 6600, 4000])
            self.plot.camera.SetViewUp(*[0.0, 0.0, -1.0])
            self.plot.camera.SetPosition([5700+IBLViewer.DEFAULT_CAMERA_DISTANCE, 6600, 4000])
        else:
            self.plot.camera.SetFocalPoint([4000, 5700, 6600])#IBLViewer.DEFAULT_FOCAL_POINT)
            self.plot.camera.SetViewUp(*[-1.0, 0.0, 0.0])
            self.plot.camera.SetPosition([4000, 5700+IBLViewer.DEFAULT_CAMERA_DISTANCE, 6600])
        #*IBLViewer.DEFAULT_CAMERA_POSITION[[1, 2, 0]])
        #self.plot.camera.SetDistance(IBLViewer.DEFAULT_CAMERA_DISTANCE)
        #self.plot.camera.SetRoll(90)
        #self.plot.camera.OrthogonalizeViewUp()
        #self.plot.renderer.ResetCamera()

    def toggle_axial_view(self):
        """
        Toggle axial view
        """
        if IBLViewer.REVERSED_REFERENCE:
            self.plot.camera.SetFocalPoint([5700, 6600, 4000])
            self.plot.camera.SetViewUp(*[1.0, 0.0, 0.0])
            self.plot.camera.SetPosition([5700, 6600, 4000-IBLViewer.DEFAULT_CAMERA_DISTANCE])
        else:
            self.plot.camera.SetFocalPoint([4000, 5700, 6600])#IBLViewer.DEFAULT_FOCAL_POINT)
            self.plot.camera.SetViewUp(*[0.0, 0.0, 1.0])
            self.plot.camera.SetPosition([4000-IBLViewer.DEFAULT_CAMERA_DISTANCE, 5700, 6600])
        #*IBLViewer.DEFAULT_CAMERA_POSITION[[2, 1, 0]])
        self.plot.camera.SetDistance(IBLViewer.DEFAULT_CAMERA_DISTANCE)
        #self.plot.camera.SetRoll(15)
        #self.plot.renderer.ResetCamera()
        #SetModelTransformMatrix

    def initialize(self, resolution=25, mode=-1, priors_path=None, probes_path=None, use_ospray=True, verbose=True):
        """
        Initialize neuroscience viewer
        """
        self.time  = datetime.now()
        # N == number of windows/plots
        self.plot = Plotter(N=1)#axes=2)
        self.set_resolution(resolution)
        self.set_renderer()

        self.mode = mode

        #settings.useDepthPeeling = True
        settings.useFXAA = True
        settings.multiSamples = 4
        settings.useDepthPeeling = True

        """ self.origin = Sphere([0, 0, 0], r=50, c='black').pickable(0)
        self.origin.name = 'Origin'
        self.origin.pickable(True)
        self.plot.add(self.origin, render=False)
        """
        if verbose:
            print('')
            print('-- Starting visualization with volumetric resolution', self.resolution)
            print('')

        # Important to get atlas first, because volume scalar reassignment needs it
        self.atlas = atlas.AllenAtlas(resolution)#self.get_atlas()
        self.metadata = self.get_atlas()
        #self.volume = self.atlas.image

        self.clipping_planes = vtk.vtkPlanes()

        # MOCK data 1
        if priors_path is not None:
            self.test_load_priors(priors_path)

        self.color_map, self.opacity_map, self.lut = self.build_color_map(mode)
        col_map, self.volume_opacity_map, lut = self.build_color_map(mode, mode == 'priors', alpha_factor=0.001)
        if verbose:
            print('Built the transfer function:', str(time_diff(self.time)) + 's')

        # MOCK data 2
        if probes_path is not None:
            self.test_load_ibl_probes_data(probes_path)

        #self.brain_surface.pos(-np.array(IBLViewer.BREGMA))
        #self.brain_surface.pos(IBLViewer.MESH_OFFSET)
        #self.brain_surface.pos([0, 1200, -3650])
        #self.plot.add(self.brain_surface, render=False)
        
        s = self.resolution
        spacing = np.array([s, s, s])
        # 'Beryl' (IBL made-up name) or 'Allen'
        volume = self.atlas.regions.mappings['Beryl'][self.atlas.label]

        # ibllib.atlas transforms the volume to apdvml or [2, 0, 1] but we don't want that at all here...
        # otherwise all datasets that we want to visualize within this volume will have to be adapted.
        # This is a typical case of lack of consideration for UX. We fix that here.
        reoriented_volume = volume
        if IBLViewer.REVERSED_REFERENCE:
            reoriented_volume = np.transpose(volume, (2, 0, 1))#volume[[2, 0, 1]] * [1, -1, 1]
        self.volume_actor = Volume(reoriented_volume, spacing=spacing, 
        mapper='smart', c=col_map, alpha=self.volume_opacity_map)
        #self.volume_actor.cmap(vol_col_map[:, :3], on='cell')
        #self.volume_actor = Volume(self.volume, spacing=[s, s, s], mapper='opengl')
        #self.volume_actor._mapper.SetScalarRange(self.volume_scalar_min, self.volume_scalar_max)
        #self.volume_actor._mapper.SetLookupTable(lut)
        
        self.volume_actor.shade(False)
        self.volume_actor.alphaUnit(1)
        volume_axes = self.volume_actor.buildAxes(c='gray', textScale=0.0)
        self.plot.add(volume_axes)
        
        # histogram()
        #self.volume_actor.jittering(True)
        #self.volume_actor._mapper.AutoAdjustSampleDistancesOn()
        print('Volume spacing', self.volume_actor.spacing())
        #.addScalarBar3D(title='Voxel intensity', c='k')
        self.volume_actor.name = 'Allen atlas volume'
        self.volume_actor.mode(0) #0 == ugly shadows, 1 == flat but wrong render anyway
        self.volume_actor.pickable(False)
        self.volume_actor.addScalarBar(useAlpha=False)

        #self.volume_actor.pos(nudged_pos)
        
        offset = spacing / 2
        self.volume_actor.pos(offset.tolist())#self.bregma_aligned_volume_position)

        self.volume_dimensions = np.array(self.volume_actor.dimensions()).astype(np.float64) * self.resolution
        self.volume_center = np.array(self.volume_actor.pos()) + np.array(self.volume_actor.center())

        print('Vol center', np.array(self.volume_actor.center()))
        print('Vol dimensions', np.array(self.volume_actor.dimensions()) * self.resolution)
        print('Vol dimensions center', np.array(self.volume_actor.dimensions()) * self.resolution / 2)
        print('Volume abs center', self.volume_center, 'position', np.array(self.volume_actor.pos()))

        raw_volume_center = np.array(self.volume_actor.dimensions()) * self.resolution / 2
        adjusted_center = self.volume_center - raw_volume_center

        self.bregma_aligned_volume_position = np.zeros(3)
        self.bregma_aligned_volume_position[1] = self.volume_center[1] - IBLViewer.BREGMA[1]

        self.plot.add(self.volume_actor, render=False)

        #p.add_plane_widget()
        if verbose:
            print('Started visualizer:', str(time_diff(self.time))+'s')
            print('')
            print('')

        #self.init_slicer()
        #self.update_slice()
        #self.plot.camera.clippingRange = [0.01, 100000]

        self.plot.mouseLeftClickFunction = self.get_region_info
        # E.g.: KeyPressEvent, RightButtonPressEvent, MouseMoveEvent, ..etc
        #self.plot.interactor.RemoveAllObservers()

        axes_opts = dict(
            textScale=0.75,       # make all text 30% bigger
            labelFont=self.font,
        )

        self.add_slicer()
        self.add_slicer_ui()
        self.add_camera_ui()

        #h0 = histogram(g0, xtitle=n0, c=0)
        self.plot.camera.SetClippingRange([0.1, 100000.0])
        self.plot.resetcam = False
        
        if IBLViewer.REVERSED_REFERENCE:
            bregma = Sphere(IBLViewer.BREGMA, r=100, c='#eac112')
        else:
            bregma = Sphere(IBLViewer.BREGMA[[2, 0, 1]], r=100, c='#eac112')
        print('Bregma origin', IBLViewer.BREGMA)
        self.plot.add(bregma)

        self.toggle_sagittal_view()
        #self.camera.SetPosition(self.camera_positions[idx])
        #self.toggle_orthographic_view()

        self.plot.show(self.plot.actors, at=0, interactive=False) #__doc__, axes=2, interactive=False) # at=0, interactorStyle=6
        #self.plot.show(applications.Slicer2d(self.volume_actor), at=1)
        interactive()
        #self.volume_actor = self.init_volume_cutter()


if __name__ == '__main__':
    viewer = IBLViewer()
    # Put your path here
    # If paths are not given to the function below, then no custom data is loaded
    base_path = 'your_data_path/'
    priors_path = base_path + 'completefits_2020-11-09.p'
    probes_path = base_path + 'ephys_aligned_session_insertions.p'
    viewer.initialize(25, 'atlas', priors_path, probes_path)

# Disclaimer, this is a prototype, nothing clean here. Refactoring in progress on another branch.