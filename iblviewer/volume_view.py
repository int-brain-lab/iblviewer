import numpy as np
import logging

import vtk
from vedo import *
from vedo.addons import *

import iblviewer.utils as utils

class VolumeView():

    #slicing_plane_ids = {'x+':0, 'x-':1}

    def __init__(self, plot, model, atlas_model):
        """
        Constructor
        :param plot: Plot instance
        :param model: VolumeModel instance
        :param atlas_model: AtlasModel instance
        """
        self.plot = plot
        self.model = model
        self.atlas_model = atlas_model

        self.actor = None
        self.alpha_factor = 0.001 #* self.model.volume.resolution

        self.clipping_planes = None
        self.clipping_axes = []
        self.scalar_bar = None
        self.dummy_actor = Cross3D([0, 0, 0], s=0.0, c='black')
        self.dummy_actor.pickable(0).lighting('off')
        self.plot.add(self.dummy_actor, render=False)

        # Init phase
        self.build_actor()
        self.set_color_map()
        
        #msg = 'Volume abs center', self.volume_center, 'position', np.array(self.volume_actor.pos())
        #logging.info(msg)

    def build_actor(self):
        """
        Set the volume actor for visualization in VTK
        """
        spacing = np.array([self.model.resolution]*3)
        self.actor = Volume(self.model.volume, spacing=spacing, mapper='smart')
        self.actor.name = self.model.name
        self.actor.shade(False)
        self.actor.mode(0)
        self.actor.pickable(False)
        # Apparently, setting a custom spacing results in a misaligned volume
        # by exactly half a voxel. This is fixed here.
        self.actor.pos(spacing / 2)

        # TODO: Compute this in the model and check here when creating the actor that size matches?
        self.model.dimensions = np.array(self.actor.dimensions()).astype(np.float64) * self.model.resolution
        # center() is also wrong on the volume when spacing is used as it's not exactly dimensions() / 2
        self.model.center = np.array(self.actor.pos()) + np.array(self.actor.center())

        self.bounding_planes = []
        self.init_bounding_planes()
        self.init_clipping_planes()
        self.plot.add(self.actor, render=False)


        #self.actor.alphaUnit(1)
        #self.actor.jittering(True)
        #self.actor._mapper.AutoAdjustSampleDistancesOn()
        #self.actor._mapper.SetBlendModeToAverageIntensitye()
        #self.actor._mapper.SetSampleDistance(100)
        
    def init_bounding_planes(self):
        """
        Bounding planes initialization
        """
        axes = [0, 1, 2]
        for axis in axes:
            plane_origin = self.model.center + self.model.dimensions[axis]
            self.bounding_planes.append(plane_origin)

    def set_alpha_map(self, alpha_map, alpha_factor=None):
        """
        Set alpha map to the volume view
        :param alpha_map: 2D list of scalar values and alpha values
        """
        if alpha_map is None:
            alpha_map = self.atlas_model.transfer_function.alpha_map
        if alpha_factor is None:
            alpha_factor = self.alpha_factor
        volume_alpha_map = np.ones_like(alpha_map).astype(np.float)
        volume_alpha_map[:] = alpha_map[:]
        volume_alpha_map[:, 1] *= alpha_factor
        self.actor.alpha(volume_alpha_map)

    def set_color_map(self, color_map=None, alpha_map=None):
        """
        Update the color map
        :param color_map: 4D list of scalar values and rgb colors
        :param alpha_map: 2D list of scalar values and alpha values
        """
        tf = self.atlas_model.transfer_function
        if color_map is None:
            color_map = tf.color_map
        if alpha_map is None:
            alpha_map = tf.alpha_map

        if color_map is not None:
            self.actor.cmap(color_map)#['black', 'white'])
        if alpha_map is not None:# and self.segmentation_mode():
            self.set_alpha_map(alpha_map)

        #self.actor.addScalarBar(pos=(0.85,0.05), useAlpha=False)
        #self.plot.add([self.actor.scalarbar]) #, self.dummy_actor.scalarbar])
        self.plot.remove(self.scalar_bar)
        self.scalar_bar = utils.add_scalar_bar(tf.scalar_lut, pos=(0.8,0.05))
        self.plot.add([self.scalar_bar])

    def enable_shading(self):
        volumeProperty = self.actor.GetProperty()
        #volumeProperty.SetColor(volumeColor)
        #volumeProperty.SetScalarOpacity(volumeScalarOpacity)
        volumeProperty.SetInterpolationTypeToLinear()
        volumeProperty.ShadeOn()
        volumeProperty.SetAmbient(0.6)
        volumeProperty.SetDiffuse(0.8)
        volumeProperty.SetSpecular(0.9)
        volumeProperty.SetScalarOpacityUnitDistance(1)
        self.actor.SetProperty(volumeProperty)

    def init_clipping_planes(self, axes=[0, 1, 2], custom=None):
        """
        Initialize X, Y and Z clipping planes with two planes per axis 
        for positive and negative slicing and making slabs
        :param axes: List of axes
        :param custom: Custom axis normal
        """
        self.clipping_planes = vtk.vtkPlaneCollection() #vtk.vtkPlanes()
        for axis in axes:
            p_plane = vtk.vtkPlane()
            n_plane = vtk.vtkPlane()
            '''
            normal = np.zeros(3).astype(float)
            normal[axis] = 1.0
            p_plane.SetNormal(normal)
            n_plane.SetNormal(-normal)
            '''
            self.clipping_planes.AddItem(p_plane)
            self.clipping_planes.AddItem(n_plane)
        self.clipping_axes = axes
        self.actor.mapper().SetClippingPlanes(self.clipping_planes)
        self.reset_clipping_planes(axes)

    def get_clipping_planes(self, except_axis=None):
        """
        Get the current clipping planes except the ones on the given axis
        :param except_axis: Axis id to ignore. If None, all clipping planes will be returned
        :return: vtkPlaneCollection
        """
        planes = vtk.vtkPlaneCollection()
        for axis in self.clipping_axes:
            if isinstance(except_axis, int) and except_axis == axis:
                continue
            double_axis_ref = axis * 2
            p_plane = self.clipping_planes.GetItem(double_axis_ref)
            n_plane = self.clipping_planes.GetItem(double_axis_ref + 1)
            planes.AddItem(p_plane)
            planes.AddItem(n_plane)
        return planes

    def reset_clipping_planes(self, axes=[0, 1, 2]):
        """
        Reset clipping planes
        :param axes: Axes to be reset
        """
        for axis in axes:
            double_axis_ref = axis * 2
            p_plane = self.clipping_planes.GetItem(double_axis_ref)
            n_plane = self.clipping_planes.GetItem(double_axis_ref + 1)
            normal = np.zeros(3)
            normal[axis] = 1.0
            position = self.bounding_planes[axis]
            p_plane.SetNormal(normal)
            p_plane.SetOrigin(-position)
            n_plane.SetNormal(-normal)
            n_plane.SetOrigin(position)

    def clip_on_axis(self, position=None, axis=None, normal=None):
        """
        Apply clipping on a single axis
        :param position: Position
        :param axis: Clipping axis, defauls to 0 (X)
        :param thickness: Whether a thickness (so two clipping planes) are applied
        """
        factor = 1
        axis_offset = 0
        # This should already be sorted in the model but in case it isn't, we double check here
        if normal is not None and normal[axis] < 0:
            # This means that the given axis has two 
            # clipping planes and we take the negative one
            axis_offset += 1
            position = self.model.dimensions - position
        axis_storage_id = axis * 2 + axis_offset
        plane = self.clipping_planes.GetItem(axis_storage_id)
        plane.SetOrigin(position)
        plane.SetNormal(normal)

    def build_surface_mesh(self, region):
        """
        Build a surface mesh with marching cubes algorithm
        """
        # TODO: export labeled regions as surfaces
        isosurface = self.actor.threshold(region, region).isosurface(region)
        isosurface.computeNormals().smoothLaplacian().alpha(0.2)
        isosurface.color(self.atlas_model.get_region_color(region))
        #self.current_region_surface
        #for region_id in regions:
            #isosurface, laplacian smooth and export
        """
        import numpy as np
        import vtk
        from vtk.util.numpy_support import numpy_to_vtk

        from .volume import numpy_to_volume
        from .volume import volume_to_numpy

        def _marching_cubes(label, index=None, normal=True, gradient=True, force_close=True):
        """

        # Refactored code from yuta-hi/volume-renderer/pyvr/data/surface.py
        '''
        if not isinstance(label, np.ndarray):
            label, spacing, origin = volume_to_numpy(label)
        else:
            spacing, origin = [1,1,1], [0,0,0]

        if force_close: # NOTE: make the closed surface
            _pad_width = 10
            label = np.pad(label, pad_width=_pad_width, mode='constant', constant_values=0)
        else:
            _pad_width = 0

        origin -= _pad_width * np.array(spacing)
        label = numpy_to_volume(label, spacing, origin)

        surface = vtk.vtkDiscreteMarchingCubes()
        surface.SetInputData(label)

        if index is None:
            n_label = int(label.GetScalarRange()[1]) + 1
            surface.GenerateValues(n_label, 1, n_label)
        else:
            surface.GenerateValues(1, index, index)

        surface.ComputeNormalsOn()
        surface.ComputeGradientsOn()
        surface.Update()

        return surface.GetOutput()
        '''

    #def label_to_surface(self, volume, index=None, force_close=True):
        #return _marching_cubes(label=volume, index=index, force_close=force_close)


    # ------------------------------------------------------------------------- TEST CODE BELOW ONLY
    '''
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

    def init_volume_cutter(self):
        """ 
        if not self.plot.renderer:
            save_int = plt.interactive
            self.plot.show(interactive=0)
            self.plot.interactive = save_int 
        """
        volume = self.volume_actor

        widget = vtk.vtkPlaneWidget()
        widget.SetInteractor(self.plot.interactor)
        widget.SetPlaceFactor(1.0)
        widget.SetHandleSize(0.0025)
        self.plot.cutterWidget = widget
        #plt.renderer.AddVolume(vol)
        widget.SetInputData(volume.inputdata())
        
        # Only valid for boxWidget
        """ widget.OutlineCursorWiresOn()
        widget.GetSelectedOutlineProperty().SetColor(1, 0, 1)
        widget.GetOutlineProperty().SetColor(0.2, 0.2, 0.2)
        widget.GetOutlineProperty().SetOpacity(0.7) """

        widget.SetRepresentationToOutline()
        
        self.clipping_planes.SetBounds(volume.GetBounds())
        widget.PlaceWidget(volume.GetBounds())
        
        #Only boxWidget
        #widget.InsideOutOn()
        
        #widget.GenerateClippedOuputOff()
        widget.AddObserver("InteractionEvent", self.clip_volume)

        self.plot.interactor.Render()
        widget.On()

        self.plot.interactor.Start()
        widget.Off()
        self.plot.widgets.append(widget)
    '''