import numpy as np
import logging

import vtk
from vedo import *
from vedo.addons import *

import iblviewer.utils as utils


class SlicerView():

    slices = []

    def __init__(self, plot, volume_view, slicer_model, atlas_model):
        """
        Constructor
        :param plot: Plot instance
        :param volume_viewer: VolumeViewer instance
        """
        SlicerView.slices.append(self)

        self.plot = plot
        self.volume_view = volume_view

        self.model = slicer_model
        self.atlas_model = atlas_model

        self.actor = None
        self.interactor = None

    def reslice(self, volume, origin, normal):
        """
        Slice a volume with a plane oriented by the given normal.
        This allows slicing in all directions.
        :param volume: Volume actor
        :param origin: Origin of the slicing plane
        :param normal: Normal of the slicing plane
        :return: Mesh object with the slice as an image texture
        """
        reslice = vtk.vtkImageReslice()
        #reslice.SetInputData(image)
        reslice.SetOutputDimensionality(2)
        reslice.SetAutoCropOutput(False)
        #reslice.SetInterpolator(interpolateMethod)
        reslice.SetInputData(self.volume_view.actor._data)

        M, T = utils.get_transformation_matrix(origin, normal)
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
    
    def slice_on_normal(self, value=None, normal=None):
        """
        Slice with the given normal plane
        :param raw_value: Raw slice value (amount multiplied by the normal)
        :param normal: Slice normal
        :return: Mesh actor
        """
        """
        #normal_changed = normal is not None and (normal != self.normal).any()
        #normal = normal if normal_changed else self.normal

        if raw_value is None:
            if self.value is None:
                self.value = -self.volume_range / self.resolution if raw_value != 0 else 0
        else:
            self.value = raw_value / self.resolution if raw_value != 0 else 0
        """
        resolution = self.volume_view.model.resolution
        #volume_center = self.volume_view.model.center
        volume_dimensions = self.volume_view.model.dimensions
        volume_actor = self.volume_view.actor

        in_volume_position = np.array([0.0, 0.0, 0.0])
        #in_volume_position[:] = np.array(self.volume_view.actor.center()) #AtlasViewer.ORIGIN
        in_volume_position += normal * value # * resolution # - AtlasViewer.ORIGIN

        new_slice = self.reslice(volume_actor, in_volume_position, normal)
        #new_slice = self.volume_actor.slicePlane(origin=in_volume_position, normal=normal)
        
        slice_position = np.array([0.0, 0.0, 0.0])
        #slice_position[:] = np.array(self.volume_actor.center())#AtlasViewer.ORIGIN
        #slice_position[2] += self.value * scale + normal[2]
        slice_position += normal * value * resolution # arbitrary slicing
        #slice_position += np.array([1.0, 0.0, 0.0]) * self.value * self.resolution # x slicing
        new_slice.pos(slice_position)
        return new_slice

    def slice_on_axis(self, value=None, normal=None, axis=None):
        """
        Slice on standard X, Y or Z axis
        :param value: Value on the given axis
        :param normal: Axis normal, can be either +1.0 or -10. along that axis
        :param axis: Axis integer, 0 for X, 1 for Y, 2 for Z
        :return: Mesh actor
        """
        resolution = self.volume_view.model.resolution
        #volume_center = self.volume_view.model.center
        volume_dimensions = self.volume_view.model.dimensions
        volume_actor = self.volume_view.actor

        if normal[axis] < 0:
            if value > 0:
                # Make value consistent with given normal.
                value *= normal[axis]
            value = volume_dimensions[axis] + value
        if axis == 0:
            #axis_center = int(volume_dimensions[0] / 2)
            in_volume_slice = int(value) // resolution
            new_slice = volume_actor.xSlice(in_volume_slice)
        elif axis == 2:
            #axis_center = int(volume_dimensions[2] / 2)
            in_volume_slice = int(value) // resolution
            new_slice = volume_actor.zSlice(in_volume_slice)
        else:
            #axis_center = int(volume_dimensions[1] / 2)
            in_volume_slice = int(value) // resolution
            new_slice = volume_actor.ySlice(in_volume_slice)
        return new_slice

    def update(self, value=None, normal=None, axis=None, clipping_planes=None):
        """
        Update slicer
        """
        value = self.model.value if value is None else value
        normal = self.model.normal if normal is None else normal
        axis = self.model.axis if axis is None else axis
        clipping_planes = self.model.clipping_planes if clipping_planes is None else clipping_planes

        if isinstance(axis, int) and 0 <= axis <= 2:
            new_slice = self.slice_on_axis(value, normal, axis)
        else:
            new_slice = self.slice_on_normal(value, normal)

        self._update(new_slice, clipping_planes)

    def apply_lut(self, lut=None, actor=None):
        """
        Apply transfer function with a look-up table
        :param lut: vtkLookupTable
        :param actor: The actor to receive this 
        """
        if (actor is None and self.actor is None):
            return
        actor = self.actor if actor is None else actor
        actor._mapper.SetLookupTable(lut)

        # cmap works for the volume but not for the slice so we build 
        # our own lut beforehand and use it as above
        #self.actor.cmap(tf.color_map, alpha=tf.opacity_map)
        
    def _update(self, new_slice, clipping_planes=None):
        """
        Internal update method to refresh the plot with the new slice mesh
        :param new_slice: Mesh actor
        :param clipping_planes: A set of vtkClippingPlanes, optional
        """
        if new_slice is None:
            return

        new_slice.pickable(True)
        #new_slice.UseBoundsOff() # avoid resetting the cam
        new_slice.lighting('off')
        
        new_slice._mapper.SetScalarVisibility(1)
        # Without setting scalar range, the mapping will be off
        new_slice._mapper.SetScalarRange(0, len(self.atlas_model.metadata))
        new_slice._mapper.SetColorModeToMapScalars()
        new_slice._mapper.SetScalarModeToUsePointData()
        # As per a bug in VTK 9 that I found while using vedo that makes pickable fail when
        # there is transparency as per https://github.com/marcomusy/vedo/issues/291
        # Force opaque fix should be gone with the next update of VTK (hopefully)
        new_slice.ForceOpaqueOn()
        new_slice.pickable(True)
        new_slice.name = self.model.name

        self.plot.remove(self.actor, render=False)
        self.plot.remove(self.interactor, render=False)

        if clipping_planes is not None:
            new_slice.mapper().SetClippingPlanes(clipping_planes)

        self.actor = new_slice
        self.apply_lut(self.atlas_model.transfer_function.lut)
        self.plot.add([new_slice])

        """ 
        slice_center = new_slice.centerOfMass()
        interactor_plane = Plane(pos=slice_center, normal=normal, sx=10000).alpha(0.2)
        interactor_plane.c('white')
        interactor_plane.ForceOpaqueOn()
        interactor_plane.pickable(True) 
        """
        #self.interactor = interactor_plane