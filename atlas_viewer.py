from vedo import load, datadir
from vedo.applications import RayCaster, Slicer

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

"""
Project: IBL Viewer
Description: Interactive volumetric and surface data visualizer that 
integrates with the Python ecosystem that is widely used nowadays

Copyright: 2021 Nicolas Antille, International Brain Laboratory
License: MIT
"""

#pv.rcParams['use_ipyvtk'] = True
base = '/Users/antyler/workspace/projects/ibl_atlas_viewer'


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

    def __init__(self, plot, volume_actor, resolution, color_map=None, lut=None, surface=None):
        """
        Constructor
        """
        self.plot = plot
        self.slice = None
        self.normal = np.array([1.0, 0.0, 0.0])
        self.volume_actor = volume_actor
        self.color_map = color_map
        self.resolution = resolution
        self.lut = lut

        self.volume_range = 0

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
        reslice.SetAutoCropOutput(True)
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

    def update(self, raw_value=None, normal=None, plot=None):
        """
        Update slicer with given value
        """
        scale = self.resolution

        normal_changed = normal is not None and (normal != self.normal).any()
        normal = normal if normal_changed else self.normal

        last_value = self.value

        if raw_value is None:
            if self.value is None:
                value = -self.volume_range
                self.value = value / self.resolution if value != 0 else 0
        else:
            value = raw_value
            self.value = value / self.resolution if value != 0 else 0

        in_volume_position = np.array(self.volume_actor.center())
        in_volume_position += self.normal * self.value
        current_slice = self.reslice(self.volume_actor, in_volume_position, normal)#self.volume_actor.slicePlane(origin=in_volume_position, normal=normal)
        current_slice.pickable(True)
        #current_slice.UseBoundsOff() # avoid resetting the cam
        current_slice.name = 'Atlas slice 1'
        current_slice.lighting('off')
        
        current_slice._mapper.SetScalarVisibility(1)
        #current_slice._mapper.SetScalarRange(self.volume_scalar_min, self.volume_scalar_max)
        current_slice._mapper.SetLookupTable(self.lut)
        # TODO: check the below method
        #current_slice.mapPointsToCells()
        
        slice_position = np.array([0.0, 0.0, 0.0])
        slice_position[:] = IBLViewer.BREGMA[[1, 2, 0]] - IBLViewer.MESH_OFFSET
        #slice_position[2] += self.value * scale + self.normal[2]
        slice_position += self.normal * self.value * scale
        current_slice.pos(slice_position)
        current_slice.scale([scale, scale, -scale])

        clipping_planes = vtk.vtkPlaneCollection()
        slice_plane = vtk.vtkPlane()
        slice_plane.SetOrigin(slice_position)
        slice_plane.SetNormal(normal)
        clipping_planes.AddItem(slice_plane)
        self.volume_actor.mapper().SetClippingPlanes(clipping_planes)
        
        cut_brain_surface = None
        if self.brain_surface is not None:
            slice_center = current_slice.centerOfMass()
            #if self.cut_brain_surface is None or self.value > last_value or normal_changed:
            # Got to take a step back
            cut_brain_surface = self.brain_surface.clone()
                
            cut_brain_surface.cutWithPlane(origin=slice_center, normal=normal)
            cut_brain_surface.name = 'Brain surface [cut]'
            #cut_brain_surface#.cap().computeNormals()

            slice_position[2] -= self.normal[2] * self.resolution
            current_slice.pos(slice_position)

            # Remove and add objects on scene
            #if self.value <= last_value:
            try:
                self.plot.remove(self.cut_brain_surface, render=False)
            except Exception as e:
                print(e)
                pass

        try:
            self.plot.remove(self.slice, render=False)
        except Exception as e:
            print(e)
            pass

        if cut_brain_surface is not None:
            self.plot.add([cut_brain_surface, current_slice])
            self.cut_brain_surface = cut_brain_surface
            self.cut_brain_surface.pickable(False)
        else:
            self.plot.add([current_slice])

        self.slice = current_slice
        
    def update_wip(self, widget=None, event=None, origin=None, value=None, normal=None):
        """
        Update slicer with given value
        """
        #last_value = self.value

        #normal_changed = normal is not None and (normal != self.normal).any()
        #normal = normal if normal_changed else self.normal

        """ if widget is not None and event is not None:
            value = widget.GetRepresentation().GetValue()
            self.value = value / scale if value != 0 else 0
        if value is None:
            # Case where for instance you update the slices with only normal change
            value = last_value """

        origin = np.array(origin)
        normal = np.array(normal)
        last_slice = self.slice
        
        in_volume_origin = np.array(self.volume_actor.center())
        dist = np.linalg.norm(IBLViewer.BREGMA - origin) / self.resolution
        print('Slicing distance', dist, normal)
        in_volume_origin += self.normal * dist

        current_slice = self.volume_actor.slicePlane(origin=in_volume_origin, normal=normal)
        current_slice.pickable(True)
        #self.slice.UseBoundsOff() # avoid resetting the cam
        current_slice.name = 'Atlas slice 1'
        current_slice.lighting('off')
        
        current_slice._mapper.SetScalarVisibility(1)
        #current_slice._mapper.SetScalarRange(self.volume_scalar_min, self.volume_scalar_max)
        #current_slice._mapper.SetLookupTable(self.lut)
        current_slice.cmap(self.color_map[:, :3], alpha=self.color_map[:, 3])
        # TODO: check the below method
        #current_slice.mapPointsToCells()
        
        """ slice_position = np.array([0.0, 0.0, 0.0])
        slice_position[:] = IBLViewer.BREGMA[[1, 2, 0]] - IBLViewer.MESH_OFFSET
        #slice_position[2] += self.value * scale + self.normal[2]
        slice_position += self.normal * self.value * scale """
        #current_slice.pos(origin/2)

        #current_slice.pos(origin)
        scale = self.resolution
        current_slice.scale([scale, scale, -scale])
        #current_slice.alignToBoundingBox(self.volume_actor, rigid=False)

        """ slice_position[2] -= self.normal[2] * self.resolution
        current_slice.pos(slice_position) """

        self.slice = current_slice


class IBLViewer():
    
    VIZ_VOLUME_SUFFIX = '_m'
    ALLEN_ATLAS_RESOLUTIONS = [10, 25, 50, 100]
    MESH_OFFSET = np.array([-1200.0, -3650.0, 0.0])
    BREGMA = np.array([5739.0, 5400.0, 332.0]) # Bregma is NEGATIVE Y and Z !

    def __init__(self):
        """
        Constructor
        """
        self.plot = None
        self.origin = None
        
        self.atlas = None
        self.resolution = 0
        self.lut = None
        self.volume_unique_scalars = []
        self.volume_dimensions = None
        #self.volume_scalar_min = 0.0
        #self.volume_scalar_max = 1.0
        
        self.volume = None
        self.scalars = None

        self.color_map = None
        self.volume_actor = None
        
        self.slicers = []
        self.active_slicer = None
        self.normal = np.array([1.0, 0.0, 0.0])

        self.region_info_text = None
        self.region_info_point = None

        self.brain_surface = None
        self.cut_brain_surface = None

        self.time = 0
        self.font = 'Source Sans Pro'

        self.per_region_priors_median_dsquare_test_data = None
        self.probes_insertions_test_data = None

    def get_region_and_row_id(self, acronym):
        region_data = self.atlas[self.atlas.acronym == acronym]
        if region_data is None or len(region_data) < 1:
            return
        return region_data.id.to_numpy()[0], region_data.index[0]

    def get_atlas(self, file_path='./data/allen/allen_structure_tree.csv'):
        return pd.read_csv(file_path)

    def import_volume(self, file_path):
        # Forcing index order to C to run faster values reassignment than with default Fortran order
        volume, header = nrrd.read(file_path, index_order='C')
        return volume, header

    """ import numba as nb
    @nb.vectorize(target="cpu")
    def nb_vf(x):
        return x+2*x*x+4*x*x*x

    def f(self, x):
        return self.atlas.index[self.atlas.id == x].to_list()[0] """

    def set_resolution(self, resolution):
        if resolution not in IBLViewer.ALLEN_ATLAS_RESOLUTIONS:
            resolution = IBLViewer.ALLEN_ATLAS_RESOLUTIONS[-1]
        self.resolution = resolution

    def load_volume(self, verbose):
        if self.atlas is None:
            self.atlas = self.get_atlas()

        base_volume_path = './data/allen/volumes/annotation_' + str(self.resolution) + '.nrrd'
        reassigned_volume_path = change_file_name(base_volume_path, None, None, IBLViewer.VIZ_VOLUME_SUFFIX)
        
        # -------------

        # TEST
        
        #self.volume, header = self.import_volume(base_volume_path)
        #return

        # --------------

        if os.path.exists(reassigned_volume_path):
            volume, header = self.import_volume(reassigned_volume_path)
        else:
            volume, header = self.import_volume(base_volume_path)
            volume = self.reassign_scalars(volume, reassigned_volume_path)
            if verbose:
                print('Reassigned scalar values in volume:', str(time_diff(self.time)) + 's')
        
        if verbose and volume is not None:
            print('Opened atlas', reassigned_volume_path, str(time_diff(self.time)) + 's')
            min_volume_value, max_volume_value = np.amin(volume), np.amax(volume)
            print('Min max scalar values in volume', min_volume_value, max_volume_value)
        else:
            print('Failed to open atlas', reassigned_volume_path)

        return volume, header

    def reassign_scalars(self, volume, write_path=None, verbose=False):
        """
        Reassign scalar values to something that makes more sense
        Scalar values in original annotation_xxx.nrrd (where xxx is the resolution) are
        not set in a clever way. Brain atlas regions (over a thousand) feature indices from 0 to
        ... 607344834!! Applying a transfer function where most values are useless is suboptimal.
        So we rewrite Allen Atlas region ids to row ids (from 0 to 1000+).
        """
        print('Building appropriate volume from Allen data source...')
        linear_ids = self.atlas.index
        region_ids = self.atlas.id

        #volume = np.vectorize(self.f)(volume)

        """ colors = self.atlas.color_hex_triplet
        print('Volume colors', colors.size, pd.unique(colors), len(pd.unique(colors)))
        return None """

        """ regions = np.arange(self.atlas.id.size)
        regions[:] = self.atlas.id.to_numpy()
        print('Len regions', len(regions), 'vs', self.atlas.id.size)
        for iter_id in range(region_ids.size):
            if verbose:
                print('Reassigning scalar', self.atlas.id[iter_id], 'to', linear_ids[iter_id])
            if len(volume[volume[:, :, :] == self.atlas.id[iter_id]]) <= 0:
                print('Could not find a scalar corresponding to region', self.atlas.id[iter_id])
            else:
                volume[volume[:, :, :] == self.atlas.id[iter_id]] = linear_ids[iter_id] """

        region_ids = np.unique(volume)
        print('Num regions labeled in volume', len(region_ids), 'from', self.atlas.id.size, 'in atlas')
        for iter_id in range(len(region_ids)):
            region_id = region_ids[iter_id]
            row_id = self.atlas.index[self.atlas.id == region_id].to_list()[0]
            volume[volume == region_id] = row_id
        
        if write_path is not None:
            print('Saving volume data under', write_path)
            nrrd.write(write_path, volume, index_order='C')
        return volume

    def get_region_info(self, mesh):
        """
        Handle event
        """
        if not 'slice' in mesh.name:
            return
        ptid = mesh.closestPoint(mesh.picked3d, returnPointId=True)
        # Scalar values in volume are integers in this case
        value = int(self.active_slicer.slice.getPointArray()[ptid])
        
        txt = 'Selected region [' + str(value) + ']' + ' Atlas ID: ' 
        txt += str(self.atlas.id[value]) + ' - ' + str(self.atlas.name[value])
        if isinstance(self.scalars, dict):
            txt += '\nScalar value: ' + str(self.scalars.get(value, 'none found'))  
        info_point = Sphere(mesh.points(ptid), r=50, c='white').pickable(0)
        #vig = vpt.vignette(txt, c='black', offset=(20,10)).followCamera()
        info_text = Text2D(txt, c='black', pos=[0.05, 0.9], font=self.font, s=0.75)

        if self.region_info_text is not None:
            self.plot.remove(self.region_info_text, render=False)
            self.plot.remove(self.region_info_point, render=False)
            
        self.plot.add([info_text, info_point])
        self.region_info_text = info_text
        self.region_info_point = info_point

    def on_event(self, iren, event):
        # TODO: WIP, use this for hover or other events
        printc(event, 'happened at position', iren.GetEventPosition())
        x, y = iren.GetEventPosition()
        # print('_mouseleft mouse at', x, y)

        renderer = iren.FindPokedRenderer(x, y)
        picker = vtk.vtkPropPicker()
        picker.PickProp(x, y, renderer)

        actor = picker.GetActor()
        print('trying actor')
        if not actor:
            actor = picker.GetAssembly()
            print('trying assembly')

        if not actor:
            actor = picker.GetProp3D()
            print('trying prop3d')

        if not hasattr(actor, "GetPickable"):
            print(actor, 'has no pickable attribute') 
        if actor is not None and not actor.GetPickable():
            print(actor, 'is not pickable')

        picked3d = picker.GetPickPosition()
        picked2d = np.array([x,y])
        if actor is not None and hasattr(actor, 'name'):
            print('Actor', actor.name, 'at', picked3d)
        elif actor is not None:
            print('Got actor but no name')

    def get_ui_pos(self, x, y, length, horizontal=True, absolute=True):
        """
        Get relative position within UI
        """
        if horizontal:
            return np.array([[x, y], [x + length, y]])
        else:
            return np.array([[x, y], [x, y + length]])

    def add_slicer(self, normal=None):
        """
        Initialize the UI for slicing
        """
        slicer = Slicer(self.plot, self.volume_actor, self.resolution, self.color_map, self.lut, self.brain_surface)
        self.slicers.append(slicer)
        self.active_slicer = self.slicers[-1]
        return slicer

    def add_slicer_ui(self, slicer=None):
        """
        Add slicer UI. Currently, this is made for only one slicer but it can be extended.
        """

        # Not used yet, but the goal is to add a UI panel for a given slicer
        if slicer is None:
            slicer = self.active_slicer
        
        extra_margin = 50
        rng_v = max(self.volume_dimensions) / 2 + extra_margin
        
        # Init slicer
        slicer.volume_range = rng_v
        slicer.update(-rng_v)
        #print('Volume dimensions', volume_dimensions)

        def update_value(widget=None, event=None, value=-rng_v):
            if widget is not None and event is not None:
                value = widget.GetRepresentation().GetValue()
            self.active_slicer.update(value, self.normal)

        s_kwargs = {'titleSize':0.75, 'font':self.font}
        ui_pos = self.get_ui_pos(0.05, 0.1, 0.4)
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
        
        n = self.normal
        ui_pos = self.get_ui_pos(0.05, 0.2, 0.12)
        slider = self.plot.addSlider2D(update_x_normal, -1.0, 1.0, n[0], ui_pos, title='Normal X', **s_kwargs)
        slider.GetRepresentation().SetLabelHeight(0.022*s_kwargs['titleSize'])

        ui_pos = self.get_ui_pos(0.2, 0.2, 0.12)
        slider = self.plot.addSlider2D(update_y_normal, -1.0, 1.0, n[1], ui_pos, title='Normal Y', **s_kwargs)
        slider.GetRepresentation().SetLabelHeight(0.022*s_kwargs['titleSize'])

        ui_pos = self.get_ui_pos(0.35, 0.2, 0.12)
        slider = self.plot.addSlider2D(update_z_normal, -1.0, 1.0, n[2], ui_pos, title='Normal Z', **s_kwargs)
        slider.GetRepresentation().SetLabelHeight(0.022*s_kwargs['titleSize'])

        # Retrieves data about hover
        #self.plot.addCallback('HoverEvent', self.hover_slice_event)

    def build_color_map(self, mode=-1, only_given_region_scalars=False):
        """
        Build a color map for the atlas volume and slices
        """
        if mode == 'atlas':
            # Modified from ibl regions_from_allen_csv()
            c = np.uint32(self.atlas.color_hex_triplet.apply(lambda x: int(x, 16) if isinstance(x, str) else 0))
            c = np.flip(np.reshape(c.view(np.uint8), (self.atlas.id.size, 4))[:, :3], 1) / 255
            # RGBA
            #c[(c[:, 0] == 0) & (c[:, 1] == 0) & (c[:, 2] == 0)] = 0.0
            c = np.c_[c.astype(np.float32), np.ones(self.atlas.id.size)]
            #print('Building color map with len colors', self.atlas.color_hex_triplet.size, 'versus len regions', self.atlas.id.size)
            
        elif mode == 'priors':
            print('Assigning color map wrt priors values')
            df = self.per_region_priors_median_dsquare_test_data
            values = df['value']
            min_p = float(np.amin(values, axis=0))
            max_p = float(np.amax(values, axis=0))
            rng_p = max_p - min_p

            # Init all to clear gray (90% white)
            c = np.ones((self.atlas.id.size, 4)).astype(np.float32) * 0.9
            c[:, -1] = 0.0 if only_given_region_scalars else 1.0
            print('Assigning', values.size, 'to atlas ids', self.atlas.id.size)
            self.scalars = {}
            for acronym, value in df.iterrows():
                region_id, row_id = self.get_region_and_row_id(acronym)
                #region = scene.add_brain_region(acronym, silhouette=True)
                value = float(value.to_numpy()[0])
                #r = (float(value.to_numpy()[0]) - min_p) / rng_p
                c[row_id] = list(colorMap(value, "jet", min_p, max_p)) + [1.0]
                self.scalars[row_id] = value
        elif isinstance(mode, int) and mode > 0:
            c = np.zeros((self.atlas.id.size, 4))
            c[mode, :] = [1.0, 0.2, 0.2, 1.0]
        else:
            c = np.random.rand(self.atlas.id.size, 3)
            c = np.c_[c.astype(np.float32), np.ones(self.atlas.id.size)/2]

        # First region in atlas is not a region (void)
        c[0, :] = [1.0, 1.0, 1.0, 0.0]

        lut = vtk.vtkLookupTable()
        num_regions = len(c)
        lut.SetNumberOfTableValues(num_regions - 1)
        lut.SetRange(0, num_regions - 1)
        for r_id in range(num_regions):
            lut.SetTableValue(r_id, *c[r_id])
        lut.Build()

        return c, lut
    
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
        self.active_slicer.update_mk2(origin=points[0], normal=normals[0])

        self.plot.add([self.active_slicer.slice])

    def init_volume_cutter(self):
        """ 
        if not self.plot.renderer:
            save_int = plt.interactive
            self.plot.show(interactive=0)
            self.plot.interactive = save_int 
        """
        volume = self.volume_actor

        boxWidget = vtk.vtkBoxWidget()
        boxWidget.SetInteractor(self.plot.interactor)
        boxWidget.SetPlaceFactor(1.0)
        boxWidget.SetHandleSize(0.0025)
        self.plot.cutterWidget = boxWidget
        #plt.renderer.AddVolume(vol)
        boxWidget.SetInputData(volume.inputdata())
        
        boxWidget.OutlineCursorWiresOn()
        boxWidget.GetSelectedOutlineProperty().SetColor(1, 0, 1)
        boxWidget.GetOutlineProperty().SetColor(0.2, 0.2, 0.2)
        boxWidget.GetOutlineProperty().SetOpacity(0.7)
        
        self.clipping_planes.SetBounds(volume.GetBounds())
        boxWidget.PlaceWidget(volume.GetBounds())
        boxWidget.InsideOutOn()
        boxWidget.AddObserver("InteractionEvent", self.clip_volume)

        self.plot.interactor.Render()
        boxWidget.On()

        self.plot.interactor.Start()
        boxWidget.Off()
        self.plot.widgets.append(boxWidget)

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
            print(per_region_median_dsquare)
            print('Min', np.amin(self.per_region_priors_median_dsquare_test_data, axis=0))
            print('Max', np.amax(self.per_region_priors_median_dsquare_test_data, axis=0))
            print('')
        return self.per_region_priors_median_dsquare_test_data

    def add_probe(self, x, y, z, depth=1, theta=0, phi=0, color=[0.5, 0.2, 0.1], alpha=0.5, name=None, bregma_offset=True):
            start_point = np.array([x, y, z,]) #np.array([-y, -z, x])
            if bregma_offset:
                start_point += IBLViewer.BREGMA#[[1, 2, 0]]
            # spher2cart(1, theta, phi) * depth
            end_point = start_point + spher2cart(depth, theta, phi)#spher2cart(1, theta, phi) * depth#[0, 5000, 0]#np.array(atlas.sph2cart(1, theta, phi)) * r
            print('probe', end_point - start_point, 'depth', depth)
            return Cylinder(pos=[start_point[[1, 2, 0]], end_point[[1, 2, 0]]], c=color, r=7, alpha=alpha)

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

        probes = []
        for single_probe_data in probes_data:
            name = 'Cylinder' +  str(len(probes))
            probe = self.add_probe(*single_probe_data)
            self.plot.add(probe)
            probes.append(probe)

        #self.probes = probes
        return probes


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

    def initialize(self, resolution=25, mode=-1, volume_only=False, use_ospray=True, verbose=True):
        """
        Initialize neuroscience viewer
        """
        self.time  = datetime.now()
        self.plot = Plotter()#axes=True)
        self.set_resolution(resolution)
        self.set_renderer()

        settings.useDepthPeeling = True
        settings.useFXAA = True

        self.origin = Sphere([0, 0, 0], r=50, c='black').pickable(0)
        self.origin.name = 'Origin'
        self.origin.pickable(True)
        self.plot.add(self.origin, render=False)

        if verbose:
            print('')
            print('-- Starting visualization with volumetric resolution', self.resolution)
            print('')

        # Important to get atlas first, because volume scalar reassignment needs it
        self.atlas = self.get_atlas()
        self.volume, header = self.load_volume(verbose)

        self.clipping_planes = vtk.vtkPlanes()

        # MOCK data
        self.test_load_priors()
        #self.test_load_ibl_probes_data()

        self.color_map, self.lut = self.build_color_map(mode)
        priors_volume_map, priors_lut = self.build_color_map(mode, mode == 'priors')
        if verbose:
            print('Built the transfer function:', str(time_diff(self.time)) + 's')
        
        self.brain_surface = self.add_region_surface()
        self.brain_surface.color([0.9, 0.9, 0.9])
        self.brain_surface.name = 'Allen CCF v3 mouse surface'
        self.brain_surface.pickable(False)
        #self.brain_surface.alpha(0.15)
        #self.brain_surface.pos(-np.array(IBLViewer.BREGMA))
        #self.brain_surface.pos(IBLViewer.MESH_OFFSET)
        #self.brain_surface.pos([0, 1200, -3650])
        #self.plot.add(self.brain_surface)
        
        self.volume_actor = Volume(self.volume, c=priors_volume_map[:, :3], alpha=priors_volume_map[:, 3], mapper='gpu')
        #.addScalarBar3D(title='Voxel intensity', c='k')
        self.volume_actor.name = 'Allen atlas volume'
        self.volume_actor.mode(0) #0 == ugly shadows, 1 == flat but wrong render anyway
        #self._RenderWindow.SetAlphaBitPlanes(1)
        #self.volume_actor._mapper.SetBlendModeToAverageIntensitye()
        #self.volume_actor._mapper.SetSampleDistance(100)
        """ volumeProperty = vtk.vtkVolumeProperty()
        #volumeProperty.SetColor(volumeColor)
        #volumeProperty.SetScalarOpacity(volumeScalarOpacity)
        #volumeProperty.SetGradientOpacity(volumeGradientOpacity)
        volumeProperty.SetInterpolationTypeToLinear()
        volumeProperty.ShadeOn()
        volumeProperty.SetAmbient(0.4)
        volumeProperty.SetDiffuse(0.6)
        volumeProperty.SetSpecular(0.2)
        volumeProperty.SetScalarOpacityUnitDistance(1)
        self.volume_actor.SetProperty(volumeProperty) """
        self.volume_actor.scale([self.resolution, self.resolution, -self.resolution])
        self.volume_actor.alignToBoundingBox(self.brain_surface, rigid=True)
        nudged_pos = np.array(self.volume_actor.GetPosition())
        nudged_pos[1] += 100
        self.volume_actor.pos(nudged_pos)
        self.volume_dimensions = np.array(self.volume_actor.dimensions()).astype(np.float64) * self.resolution
        self.volume_center = np.array(self.volume_actor.center())
        #self.plot.add(self.volume_actor, render=False) #axes=3,

        print('Saving memory, deleting volume in memory at it is already stored in VTK')
        del self.volume
        
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
        self.plot.show(self.plot.actors, interactive=False) # at=0, interactorStyle=6
        self.add_slicer()
        self.add_slicer_ui()

        self.plot.camera.SetPosition( [8000, 6000, 35000] )
        self.plot.camera.SetFocalPoint( IBLViewer.BREGMA ) # [6571, 3959, 2873] )
        self.plot.camera.SetDistance( 35000 )
        self.plot.camera.SetClippingRange( [100000.0, 0.1] )
        self.plot.camera.SetViewUp([0.0, -1.0, 0.0])
        self.plot.resetcam = False
        #self.init_volume_cutter()
        #self.volume_actor = self._addVolumeCutterTool(self.volume_actor, self.plot)

    def test_py_vista(self):
        pass
        '''
        if self.plot is None:
            self.plot = pv.Plotter()
        if color_map is not None:
            color_map = ListedColormap(color_map)
        self.plot.add_volume(volume, cmap=color_map, opacity="sigmoid")
        
        vtk_vol = p.add_volume(vol, cmap="viridis", opacity="sigmoid")
        print('Plotting volume')
        single_slice = vtk_vol.slice(normal=[1, 1, 0])
        cmap = "viridis"

        p = pv.Plotter()
        p.add_mesh(vtk_vol.outline(), color="k")
        p.add_mesh(single_slice, cmap=cmap)
        p.show()
        '''


viewer = IBLViewer()
viewer.initialize(10, 'priors', False)#38, False)
interactive()