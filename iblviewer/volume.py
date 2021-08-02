from dataclasses import dataclass, field
from typing import Mapping, List, Any
from datetime import datetime
import logging
import pandas as pd
import glob
import numpy as np
import logging
import os
from collections import OrderedDict

import nrrd
import vtk
import vedo
from vtk.util.numpy_support import numpy_to_vtk

from iblviewer.collection import Collection
import iblviewer.objects as obj
import iblviewer.utils as utils


@dataclass
class VolumeModel:
    RAW = 'raw'
    SEGMENTED = 'segmented'
    
    NORMALIZED_SUFFIX = '_norm'
    DATA_TYPE = {RAW:0, SEGMENTED:1}
    PREFIX = 'Volume'
    __count = 0

    def unique_name():
        VolumeModel.__count += 1 
        return f'{VolumeModel.PREFIX}_{VolumeModel.__count}'

    name: str = field(default_factory=unique_name)
    file_path: str = None

    scalars: Collection = field(default_factory=Collection)
    axes: List = field(default_factory=lambda: [1, 1, 1])
    data_min: float = None
    data_max: float = None
    data_map_step: float = 1.0
    data: np.ndarray = None
    data_type: str = RAW
    resolution: int = 1
    # Default units are microns.
    units: float = 1e-06
    base_color_map: Any = None

    # At IBL, volume mappings are used from ibllib: ibllib.atlas.regions.mappings
    mapping_name: str = None
    lateralized: bool = False
    # Mapping function. If None, the volume will be given as it is.
    mapping: Any = None

    luts: Collection = field(default_factory=Collection)
    slicers: Collection = field(default_factory=Collection)
    isosurfaces: Collection = field(default_factory=Collection)

    interactive_subsampling: bool = True
    volume_visible: bool = True
    slices_visible: bool = True

    transpose_shape: Any = None
    dimensions: np.ndarray = np.zeros(3).astype(float)
    center: np.ndarray = np.zeros(3).astype(float)

    def compute_size(self):
        """
        Compute volume size
        """
        if self.data is None:
            return
        self.dimensions = np.array(self.data.shape)[:3]
        if self.resolution is None:
            return
        self.resolution = int(self.resolution) # TODO: move this to constructor or init
        self.dimensions *= self.resolution
        self.center = np.ones(3) * self.resolution / 2 + self.dimensions / 2

    def compute_range(self, force=False):
        """
        Compute min and max range in the volume
        :return: Min and max values
        """
        if self.data_min is not None and self.data_max is not None and not force:
            return self.data_min, self.data_max
        self.data_min = np.min(self.data)
        self.data_max = np.max(self.data)
        #print('Volume min-max', self.data_min, self.data_max)
        return self.data_min, self.data_max

    def guess_volume_type(self):
        """
        Infer the volume type when it was not specified by the user.
        We assume here that typical values between -1 and 1 are raw volumes.
        """
        if self.data_type is None:
            if self.data_min is None or self.data_max is None:
                self.compute_range()
            if self.data_min >= -1 and self.data_max <= 1:
                guess = VolumeModel.RAW
            else:
                guess = VolumeModel.SEGMENTED
            self.data_type = guess

    def is_segmented(self, auto_guess=True):
        """
        Get whether current volume/image is segmented
        :return: Boolean
        """
        if self.data_type is None and auto_guess:
            self.guess_volume_type()
        return self.data_type == VolumeModel.SEGMENTED

    def read_volume(self, file_path):
        """
        Read local volume. Downloads the file first if it's remote.
        :param file_path: Volume path
        :return: 3D array
        """
        if file_path.startswith('http') or file_path.startswith('ftp'):
            downloaded_temp_file_path = vedo.download(file_path, verbose=False)
            if file_path.endswith('nrrd'):
                data, header = nrrd.read(downloaded_temp_file_path)
            else:
                data = vedo.loadImageData(downloaded_temp_file_path)
        else:
            if file_path.endswith('nrrd'):
                data, header = nrrd.read(file_path, index_order='C')
            else:
                data = vedo.loadImageData(file_path)
        return data

    def load_volume(self, file_path, remap_scalars=False, mapping=None, make_current=True):
        """
        Load a volume data file. Supports NRRD and many other formats thanks to vedo/VTK
        :param file_path: Volume file path. Could support other file types easily.
        :param remap_scalars: Whether scalar values in the volume are replaced by 
            their row id from a mapping that stores. This is necessary in the case of segmented
            volumes with regions that have a discontinuous id.
        :param mapping: Pandas Series or a Dictionary
        :param make_current: Set the volume data as the current one
        :return: 3D array
        """
        data = None
        if not remap_scalars or mapping is None:
            data = self.import_volume(file_path)
        else:
            time = datetime.now()
            new_file_path = utils.change_file_name(file_path, None, None, VolumeModel.NORMALIZED_SUFFIX)
            if os.path.exists(new_file_path):
                data = self.import_volume(new_file_path)
            else:
                data = self.import_volume(file_path)
                data, mapping = self.remap_slow(data, mapping, new_file_path)
                logging.info('Remapped scalar values in: ' + str(utils.time_diff(time)) + 's')
            '''
            if volume is not None:
                logging.info('Opened atlas ' + new_file_path + ' in ' + str(utils.time_diff(time)) + 's')
                min_value, max_value = np.amin(data), np.amax(data)
                logging.info('Min max scalar values in volume ' + str(min_value) + ' -> ' + str(max_value))
            else:
                logging.error('Failed to open atlas ' + new_file_path)
            '''
        if make_current and data is not None:
            self.data = data
        return data, mapping

    def transpose(self, shape=None):
        """
        Transpose the volume for visualization in VTK
        :param shape: The new shape. If None, will default to self.transpose_shape
        """
        if shape is None:
            shape = self.transpose_shape
        if shape is None:
            return
        self.data = np.transpose(self.data, shape)

    def remap_slow(self, data, mapping=None, write_path=None):
        """
        Reassign volume values (slow on large volumes!) so that they're continuous
        :param data: Volume ndarray
        :param write_path: Where the modified volume will be stored 
            (to spare going through this method next time)
        :param mapping: Pandas Series or a Dictionary that maps raw volume scalars to new ones
        :return: Modified volume data
        """
        logging.info('\nBuilding appropriate volume from Allen data source...')
        #volume = np.vectorize(self.f)(data)
        labels = np.sort(np.unique(data))
        num_labels = len(labels)
        if mapping is None:
            mapping = pd.Series(labels)
        logging.info('Num regions labeled in volume ' + str(num_labels) + ' from ' + str(mapping.size) + ' in atlas')
        logging.info('Reassigning ' + str(num_labels) + ' scalar values...')
        for iter_id in range(num_labels):
            label = labels[iter_id]
            ids = mapping.index[mapping == label].to_list()
            if len(ids) < 1:
                continue
            # On a large volume, this takes a long time
            data[data == label] = ids[0]
            if num_labels > 10000 and iter_id % 10 == 0:
                logging.info('  Progress: ' + str(int(iter_id/num_labels)*100) + '%')
        
        if write_path is not None:
            logging.info('Saving volume data under ' + write_path)
            nrrd.write(write_path, data, index_order='C')
        return data, mapping
        
    def build_lut(self, scalar_map=None, scalar_range=None, color_map=None, 
                    alpha_map=None, zero_is_transparent=True, 
                    noise_amount=0.0, nan_rgba=None, make_active=True):
        """
        Build a look-up table (LUT, sometimes known as transfer function) for the volume
        :param scalar_map: A 2D list with values in first column from the volume itself and values from
            the second column being your scalar values that correspond to such region
        :param scalar_range: Min and max values in a list
        :param color_map: Color map name to apply
        :param alpha_map: Alpha map, either None or a list of values the same length as scalar_map, that
            says how transparent a scalar value should be
        :param zero_is_transparent: Whether zero values are made transparent, True by default
        :param noise_amount: Whether a noise value is applied on the colors
        :param nan_rgba: Color and transparency (RGBA) to assign to invalid (out of range or None) scalar values
        :param make_active: Whether this one is made active (you still have to update the views after that)
        :return: LUTModel
        """
        lut_model = LUTModel()
        lut_model.build(scalar_map, scalar_range, color_map, alpha_map, 
                        zero_is_transparent, noise_amount, nan_rgba)
        self.luts.store(lut_model, set_current=make_active)
        return lut_model


def blend_maps(map1, map2, time, total_time):
    """
    Blend color maps
    """
    weight1 = max(0.0, total_time - time)
    weight2 = max(0.0, time)
    return map1 * weight1 + map2 * weight2


class Volume(vedo.Volume):
    """
    Overwriting of vedo.Volume constructor that is ill-designed as
    it transposes the given numpy array without us knowing about it,
    not giving us the option to choose about that.
    """

    def __init__(self, 
                 inputobj=None,
                 c='RdBu_r',
                 alpha=(0.0, 0.0, 0.2, 0.4, 0.8, 1.0),
                 alphaGradient=None,
                 alphaUnit=1,
                 mode=0,
                 shade=False,
                 spacing=None,
                 dims=None,
                 origin=None,
                 mapper='smart'):

        vtk.vtkVolume.__init__(self)
        vedo.BaseGrid.__init__(self)

        self.axes = [1, 1, 1]

        ###################
        if isinstance(inputobj, str):

            if "https://" in inputobj:
                from vedo.io import download
                inputobj = download(inputobj, verbose=False) # fpath
            elif os.path.isfile(inputobj):
                pass
            else:
                inputobj = sorted(glob.glob(inputobj))

        ###################
        if 'gpu' in mapper:
            self._mapper = vtk.vtkGPUVolumeRayCastMapper()
        elif 'opengl_gpu' in mapper:
            self._mapper = vtk.vtkOpenGLGPUVolumeRayCastMapper()
        elif 'smart' in mapper:
            self._mapper = vtk.vtkSmartVolumeMapper()
        elif 'fixed' in mapper:
            self._mapper = vtk.vtkFixedPointVolumeRayCastMapper()
        elif isinstance(mapper, vtk.vtkMapper):
            self._mapper = mapper
        else:
            print("Error unknown mapper type", [mapper])
            raise RuntimeError()
        self.SetMapper(self._mapper)

        ###################
        inputtype = str(type(inputobj))
        #colors.printc('Volume inputtype', inputtype)

        if inputobj is None:
            img = vtk.vtkImageData()

        elif vedo.utils.isSequence(inputobj):

            if isinstance(inputobj[0], str): # scan sequence of BMP files
                ima = vtk.vtkImageAppend()
                ima.SetAppendAxis(2)
                pb = vedo.utils.ProgressBar(0, len(inputobj))
                for i in pb.range():
                    f = inputobj[i]
                    picr = vtk.vtkBMPReader()
                    picr.SetFileName(f)
                    picr.Update()
                    mgf = vtk.vtkImageMagnitude()
                    mgf.SetInputData(picr.GetOutput())
                    mgf.Update()
                    ima.AddInputData(mgf.GetOutput())
                    pb.print('loading...')
                ima.Update()
                img = ima.GetOutput()

            else:
                if "ndarray" not in inputtype:
                    inputobj = np.array(inputobj)

                if len(inputobj.shape)==1:
                    varr = vedo.numpy2vtk(inputobj, dtype=np.float)
                else:
                    # ------------------------------ Nasty lines commented here
                    #if len(inputobj.shape)>2:
                        #inputobj = np.transpose(inputobj, axes=[2, 1, 0])
                    varr = vedo.numpy2vtk(inputobj.ravel(order='F'), dtype=np.float)
                varr.SetName('input_scalars')

                img = vtk.vtkImageData()
                if dims is not None:
                    img.SetDimensions(dims)
                else:
                    if len(inputobj.shape)==1:
                        vedo.colors.printc("Error: must set dimensions (dims keyword) in Volume.", c='r')
                        raise RuntimeError()
                    img.SetDimensions(inputobj.shape)
                img.GetPointData().SetScalars(varr)

                #to convert rgb to numpy
                #        img_scalar = data.GetPointData().GetScalars()
                #        dims = data.GetDimensions()
                #        n_comp = img_scalar.GetNumberOfComponents()
                #        temp = utils.vtk2numpy(img_scalar)
                #        numpy_data = temp.reshape(dims[1],dims[0],n_comp)
                #        numpy_data = numpy_data.transpose(0,1,2)
                #        numpy_data = np.flipud(numpy_data)

        elif "ImageData" in inputtype:
            img = inputobj

        elif isinstance(inputobj, vedo.Volume):
            img = inputobj.GetMapper().GetInput()

        elif "UniformGrid" in inputtype:
            img = inputobj

        elif hasattr(inputobj, "GetOutput"): # passing vtk object, try extract imagdedata
            if hasattr(inputobj, "Update"):
                inputobj.Update()
            img = inputobj.GetOutput()

        elif isinstance(inputobj, str):
            from vedo.io import loadImageData, download
            if "https://" in inputobj:
                inputobj = download(inputobj, verbose=False)
            img = loadImageData(inputobj)

        else:
            vedo.colors.printc("Volume(): cannot understand input type:\n", inputtype, c='r')
            return

        if dims is not None:
            img.SetDimensions(dims)

        if origin is not None:
            img.SetOrigin(origin) ### DIFFERENT from volume.origin()!

        if spacing is not None:
            img.SetSpacing(spacing)

        self._data = img
        self._mapper.SetInputData(img)
        self.mode(mode).color(c).alpha(alpha).alphaGradient(alphaGradient)
        self.GetProperty().SetShade(True)
        self.GetProperty().SetInterpolationType(1)
        self.GetProperty().SetScalarOpacityUnitDistance(alphaUnit)

        # remember stuff:
        self._mode = mode
        self._color = c
        self._alpha = alpha
        self._alphaGrad = alphaGradient
        self._alphaUnit = alphaUnit


@dataclass
class LUTModel:
    """
    This class might look slightly convoluted but it's actually simple.

    We use double mapping here in order to enable live/interactive visualization
    of volumetric data. Instead of replacing values in a 3D volume, we only replace
    the colors in the 1D LUT list.
    
    The point is that it's too slow to update a given data, like a segmented
    volume with custom values. Instead, we map such custom values to a 1D
    array (our LUT) that maps colors to raw volume values.
    
    This is much faster in terms of rendering and it enables interactive visualization.
    The scalar_lut is the original LUT for the given scalars (custom values)
    and the mapped_lut is the LUT assigned to the surfaces (like slices)
    that have copied data from the volume. The volume is given color_map 
    and alpha_map through vedo methods.

    You might say "ok for double mapping, it's the only way for interactive
    rendering of a volume, but what about color_map and mapped_lut? Aren't
    they the same?". The answer is: they're the same but VTK does not accept
    a vtkLookupTable for a volume. Instead, it wants a vtkColorTransferFunction
    and a vtkPiecewiseFunction for alpha. There's no way around it.
    The color_map will be computed as a vtkColorTransferFunction and
    the alpha_map as the vtkPiecewiseFunction.
    """

    name: str = NotImplementedError

    color_map_function: Any = None
    scalar_map: np.ndarray = None
    scalar_min: float = 0.0
    scalar_max: float = 1.0
    scalar_lut: vtk.vtkLookupTable = None
    mapped_lut: vtk.vtkLookupTable = None
    
    color_map: np.ndarray = None
    alpha_map: np.ndarray = None
    base_color_map: np.ndarray = None


    def build(self, scalar_map=None, scalar_range=None, color_map=None, 
                    alpha_map=None, zero_is_transparent=True, 
                    noise_amount=0.0, nan_rgba=None):
        """
        Build several look-up tables (LUT, sometimes known as transfer function) for the volume.
        This is where double-mapping occurs for segmented volumes that have values from 0 to n where
        each value defines a sub-volume or region. If we want to assign values (say from another model)
        to these regions, we'd have to change the volume values and it would be too slow iterating over
        each voxel in 3D. Instead we define colors that represent these values and assign them to 
        segmented regions in a 1D list.
        :param scalar_map: A 2D list with values in first column from the volume itself and values from
            the second column being your scalar values that correspond to such region
        :param scalar_range: Min and max values in a list
        :param color_map: Color map name to apply
        :param alpha_map: Alpha map, either None or a list of values the same length as scalar_map, that
            says how transparent a scalar value should be
        :param zero_is_transparent: Whether zero values are made transparent, True by default
        :param noise_amount: Whether a noise value is applied on the colors
        :param nan_rgba: Color and alpha values to assign to invalid (out of range or None) scalar values
        :return: LUTModel
        """
        if color_map is None:
            return
        if nan_rgba is None:
            nan_rgba = [0.0, 0.0, 0.0, 0.0]
        if self.base_color_map is None:
            self.base_color_map = color_map
        
        colors = []
        alphas = []
        lut = vtk.vtkLookupTable()
        scalar_lut = vtk.vtkLookupTable()

        # Use the number of values in the volume
        num_steps = len(self.base_color_map) if self.base_color_map is not None else len(color_map)
        num_steps = 2655
        s_min = 0
        s_max = num_steps
        if scalar_map is None:
            if color_map is None and self.base_color_map is not None:
                color_map = self.base_color_map
                
        loop = range(num_steps)
        noise = None
        if isinstance(noise_amount, float) and noise_amount > 0:
            noise = np.random.rand(num_steps) * noise_amount - noise_amount / 2

        # Vedo works with nested lists: 
        # [region_id, [r, g, b]] for color, and [region_id, a] for alpha

        if scalar_map is None:
            # Standard volume that is not segmented
            lut.SetRange(s_min, s_max)
            lut.SetNumberOfTableValues(num_steps)
            scalar_lut.SetRange(s_min, s_max)
            scalar_lut.SetNumberOfTableValues(num_steps)
            for r_id in loop:
                color = vedo.colors.getColor(color_map[r_id])
                color = np.array(color)
                if noise is not None:
                    color = color + noise[r_id]
                    color = np.maximum(color, 0.0)
                    color = np.minimum(color, 1.0)
                colors.append([r_id, color])
                alpha = 1.0 if alpha_map is None else alpha_map[r_id]
                if r_id == 0 and zero_is_transparent:
                    alpha = 0.0
                alphas.append([r_id, alpha])
                lut.SetTableValue(r_id, *color, alpha)
                scalar_lut.SetTableValue(r_id, *color, alpha)
                #scalar_map[r_id] = color_map[r_id]
        else:
            # Segmented volume
            s_min, s_max = scalar_range
            lut.SetRange(0, num_steps)
            lut.SetNumberOfTableValues(num_steps)
            color = None
            for r_id in range(num_steps):
                try:
                    value = scalar_map[r_id]
                except Exception:
                    value = None
                if value is None:# or s_min > value or s_max < value:
                    color = nan_rgba[:3]
                    alpha = nan_rgba[3]
                else:
                    color = vedo.colorMap(value, color_map, s_min, s_max)
                    alpha = 1.0 if alpha_map is None else alpha_map[r_id]
                    if value == 0 and zero_is_transparent:
                        alpha = 0.0
                colors.append([r_id, color])
                alphas.append([r_id, alpha])
                lut.SetTableValue(r_id, *color, alpha)
                
            # Real scalar LUT, mainly as a reference for the user
            # Here the colors resulting from the given scalar min to max
            # are assigned to segmented values in the volume
            mock_values = np.linspace(s_min, s_max, num_steps)
            scalar_lut.SetRange(s_min, s_max)
            scalar_lut.SetNumberOfTableValues(len(mock_values))
            for r_id in range(len(mock_values)):
                color = list(vedo.colorMap(mock_values[r_id], color_map, s_min, s_max))
                alpha = 0.0 if mock_values[r_id] == 0 and zero_is_transparent else 1.0
                scalar_lut.SetTableValue(r_id, *color, 1.0)

        lut.Build()
        scalar_lut.Build()

        # Just to avoid confusion: the user can give a string as a color map, like 'viridis'
        # but the real color map object is stored in self.color_map. The name 'viridis'
        # is stored under self.color_map_function (if needed later on)
        self.color_map_function = color_map
        self.color_map = colors
        self.alpha_map = alphas
        self.scalar_map = scalar_map
        self.mapped_lut = lut
        self.scalar_lut = scalar_lut

    def get_sorted_scalars(self):
        """
        Get a numpy 2D array of key-value pairs sorted by value
        :return: 2D array
        """
        sorted_scalars = np.zeros((len(self.scalar_map), 2))
        values = list(self.scalar_map.values())
        keys = list(self.scalar_map.keys())
        sorted_scalars[:, 0] = keys
        sorted_scalars[:, 1] = values
        sorted_mask = sorted_scalars[:, 1].argsort()
        sorted_scalars = sorted_scalars[sorted_mask]
        return sorted_scalars



class VolumeController():
    """
    Wrapper class that handles both the volume and its slices
    """

    def __init__(self, plot, model, initialize=True, clipping=True, slicer_box=True, 
                center_on_edges=False, alpha_unit_upper_offset=0.0, add_to_scene=True):
        """
        Constructor
        :param plot: Plot instance
        :param model: VolumeModel instance
        :param initialize: Whether the initalization
        :param clipping: Whether clipping is enabled at init time
        :param slicer_box: Whether the slicer box is enabled at init
        :param center_on_edges: Whether the volume is offest by half a voxel or not
        :param alpha_unit_upper_offset: The offset to apply to alpha unit computation.
            If greater than 0, the volume will be less opaque
        :param add_to_scene: Whether the volume is added to scene after init
        """
        self.plot = plot
        self.model = model

        self.actor = None
        self.picker = None
        self.scalars = None
        self.mask = None
        self.bounding_mesh = None
        self.alpha_unit_upper_offset = alpha_unit_upper_offset
        self.alpha_factor = 0.001 # * self.model.resolution

        self.clipping_planes = None
        self.enable_volume_clipping = True
        self.clipping_axes = []
        self.slicers = OrderedDict()
        self.slicers_selectable = False
        self.scalar_bar = None
        
        if initialize:
            self.initialize(clipping, slicer_box, center_on_edges, add_to_scene)
        #msg = 'Volume abs center', self.volume_center, 'position', np.array(self.volume_actor.pos())
        #logging.info(msg)

    def get_related_actors(self):
        """
        Get all 3D actors related to this view (for registering it in the application)
        :return: List of VTK objects
        """
        actors = []
        for slicer_id in self.slicers:
            actor = self.slicers[slicer_id].actor
            if actor is not None:
                actors.append(actor)
        for iso_id in self.model.isosurfaces:
            actors.append(self.model.isosurfaces[iso_id])
        actors.append(self.actor)
        return actors

    def initialize(self, clipping=True, slicer_box=True, center_on_edges=False, add_to_scene=True):
        """
        Set the volume actor for visualization in VTK
        :param clipping: Whether clipping is enabled
        :param slicer_box: Whether the slicer box mode is enabled (6 clipping planes)
        :param center_on_edges: Whether the volume's center is aligned to its edges 
            rather than the voxel center
        :param add_to_scene: Whether the object is added to the scene
        """
        self.build_actor(center_on_edges, add_to_scene)
        self.initialize_picker()

        if slicer_box:
            self.initialize_slicer_box()
        self.initialize_clipping_planes()
        self.set_volume_clipping(clipping)
        self.set_color_map()

        '''
        if use_mask:
            self.mask = self.actor.clone()
            self.mask.threshold(1, replace=1, replaceOut=0)
            self.actor.mapper().SetMaskTypeToBinary()
            self.actor.mapper().SetMaskInput(self.mask)
        '''

    def set_volume_visibility(self, on=True):
        """
        Set volume visibility
        :param on: Visibility boolean
        """
        if self.actor is not None:
            self.actor.SetVisibility(on)

    def set_slices_visibility(self, on=True):
        """
        Set the visibility of slices
        :param on: Visibility boolean
        """
        for slicer_id in self.slicers:
            slicer_view = self.slicers.get(slicer_id)
            slicer_view.actor.SetVisibility(on)

    def get_slices_opacity(self):
        """
        Get the opacity of slices (should be the same value for all slices)
        A mean calculation is performed on all slices alpha, just in case
        :return: Alpha value
        """
        value = 0
        num_values = 0
        for slicer_id in self.slicers:
            slicer = self.slicers[slicer_id]
            if slicer.actor is not None:
                slice_alpha = slicer.actor.GetProperty().GetOpacity()
                if slice_alpha is None:
                    continue
                value += slice_alpha
                num_values += 1
        if num_values == 0 or value == 0:
            return None
        return value / num_values

    def set_slices_opacity(self, value):
        """
        Set the opacity of slices
        :param value: Alpha value
        """
        for slicer_id in self.slicers:
            slicer = self.slicers[slicer_id]
            if slicer.actor is not None:
                slicer.actor.alpha(value)

    def get_opacity(self):
        """
        Get the relative opacity unit
        :return: Float
        """
        return self.get_relative_opacity_unit()

    def get_relative_opacity_unit(self):
        """
        Get the alpha unit relative value
        :return: Float
        """
        alpha_unit = self.actor.alphaUnit()
        r = self.model.resolution
        # Inverse function of set_opacity_unit()
        value = 1.1 - (alpha_unit / r)**0.5
        return value

    def set_opacity(self, value):
        """
        Set the opacity of the volume like in set_opacity_unit()
        :param value: Opacity value between 0.0 and 1.0
        :return: Resulting alpha unit
        """
        self.set_opacity_unit(value)

    def set_opacity_unit(self, value):
        """
        Set the opacity of the volume by modifying its alpha unit (a VTK thing).
        The alpha unit defines how much a voxel is transparent to incoming ray.
        This method normalizes the range between 0.0 and 1.0 as it depends
        on the resolution of the volume
        :param value: Opacity value between 0.0 and 1.0
        :return: Resulting alpha unit
        """
        r = self.model.resolution
        # 1 is chosen and not 1.0 because when value == 1.0, that would
        # mean that the volume is fully opaque and this yields artifacts with VTK
        alpha_unit = (1 + self.alpha_unit_upper_offset - value)**2 * r
        # vedo calls it "alpha" unit, vtk "opacity" unit. same-same!
        self.actor.alphaUnit(alpha_unit)
        return alpha_unit

    def get_spacing(self):
        """
        Get the spacing/resolution of the volume
        """
        res = self.model.resolution
        spacing = None
        if isinstance(res, int) or isinstance(res, float):
            spacing = np.array([res]*3)
        elif len(res) == 3:
            spacing = res
        else:
            raise ValueError(f'Given volume resolution {self.model.resolution} is invalid')
        return spacing

    def build_actor(self, center_on_edges=False, add_to_scene=True): #[1, 2]
        """
        Set the volume actor for visualization in VTK
        :param center_on_edges: Whether alignment by one voxel is applied
        :param add_to_scene: Whether the object is added to the scene
        """
        spacing = self.get_spacing()
        self.actor = Volume(self.model.data, spacing=spacing, mapper='smart')
        self.scalars = self.actor._data.GetPointData().GetScalars()
        self.actor.name = self.model.name
        self.actor.shade(False)
        self.actor.mode(0)
        self.actor.pickable(True)
        self.set_interactive_subsampling(False)

        if center_on_edges:
            # Moving the volume by one voxel. This is possibly due the use of custom spacing.
            self.actor.pos(self.actor.pos() + spacing)
            center = np.array(self.actor.pos()) + self.actor.center()
            if np.linalg.norm(center - self.model.center) > 0:
                #print('Adjusting volume center from', self.model.center, 'to', center)
                self.model.center = center
        
        self.set_opacity_unit(0.9)
        self.actor.jittering(True)
        #self.actor._mapper.AutoAdjustSampleDistancesOn()
        #self.actor._mapper.SetBlendModeToAverageIntensity()
        #self.actor._mapper.SetSampleDistance(100)

        if add_to_scene:
            self.plot.add(self.actor, render=False)

    def set_position(self, position):
        """
        Set the position of the volume
        """
        self.actor.pos(position)
        # TODO: we're entering in unstable things when we move the volume
        # because there is not yet a guaranteed support for updating the slices 
        # with the correct position
        self.reset_clipping_planes()

    def mirror_volume(self, axes):
        """
        Mirror the volume on given axes
        :param mirror_axes: A list of axes (either 0, 1, 2 or 'x', 'y', 'z') on which
            the volume will be mirrored. Optional
        """
        if axes is None or self.actor is None:
            return
        axes_str = ['x', 'y', 'z']
        for axis in axes:
            if isinstance(axis, int) and 0 <= axis <= 2:
                axis = axes_str[axis]
            if isinstance(axis, str) and len(axis) == 1:
                self.actor.mirror(axis=axis.lower())

    def initialize_picker(self, opacity_iso_value=0.0001):
        """
        Initialize the volume picker
        :param opacity_iso_value: Threshold that defines at what accumulated
            opacity the picker hits the volume. In the case of a segmented volume,
            you want to keep this value very low as the default one.
        """
        # As per C++ doc https://vtk.org/Wiki/VTK/Examples/Cxx/VTKConcepts/Scalars
        # https://stackoverflow.com/questions/35378796/vtk-value-at-x-y-z-point 
        picker = vtk.vtkVolumePicker()
        picker.PickCroppingPlanesOn()
        picker.UseVolumeGradientOpacityOff()
        picker.SetTolerance(opacity_iso_value)
        # A low OpacityIsoValue is necessary in the case of segmented volumes
        picker.SetVolumeOpacityIsovalue(opacity_iso_value)
        picker.AddPickList(self.actor)
        picker.PickFromListOn()
        self.picker = picker
    
    def initialize_slicer_box(self):
        """
        Initialize 6 slicing planes as a box.
        """
        for axis_id in range(6):
            slicer_model = SlicerModel(axis=axis_id)
            slicer_model.align_to_axis(axis_id, self.model.dimensions)
            self.model.slicers.store(slicer_model)
            # It's important in this case to have standalone=False
            self.slicers[axis_id] = SlicerView(self.plot, self, slicer_model, standalone=False)
        
    def update_slicer(self, slicer_id, value=None, normal=None):
        """
        Update a given slicer with the given value
        :param slicer_id: SlicerView id
        :param value: Value or 3D point
        :param normal: Normal
        """
        slicer_view = self.slicers.get(slicer_id)
        if slicer_view is None:
            return

        # This is an important part where the slicing plane is itself sliced by other planes
        slicer_model = slicer_view.model
        slicer_model.clipping_planes = self.get_clipping_planes(slicer_model.axis)
        
        # Use given value (or point) and normal to guide the below code
        result = slicer_model.update(value, normal)
        if not result:
            return

        # Update slicing image
        slicer_view.update()

    def initialize_clipping_planes(self):
        """
        Initialize X, Y and Z clipping planes with two planes per axis 
        for positive and negative slicing
        """
        self.clipping_planes = vtk.vtkPlaneCollection()
        slicer_models = self.model.slicers
        for slicer_id in slicer_models:
            self.clipping_planes.AddItem(vtk.vtkPlane())
        self.reset_clipping_planes()
        return

    def get_clipping_planes(self, except_axis=None):
        """
        Get the current clipping planes except the ones on the given axis
        :param except_axis: Axis id to ignore. If None, all clipping planes will be returned
        :return: vtkPlaneCollection
        """
        if not isinstance(except_axis, int):
            return self.clipping_planes
        exceptions = [except_axis * 2, except_axis * 2 + 1]
        planes = vtk.vtkPlaneCollection()
        for plane_id in range(self.clipping_planes.GetNumberOfItems()):
            if plane_id in exceptions:
                continue
            plane = self.clipping_planes.GetItem(plane_id)
            planes.AddItem(plane)
        return planes

    def reset_clipping_planes(self):
        """
        Reset clipping planes
        """
        slicer_models = self.model.slicers
        for slicer_id in slicer_models:
            slicer_model = slicer_models[slicer_id]
            plane_id = slicer_model.get_box_plane_id()
            plane = self.clipping_planes.GetItem(plane_id)
            plane.SetOrigin(slicer_model.origin + self.actor.pos())
            plane.SetNormal(slicer_model.normal)

    def clip_on_axis(self, position=None, axis=None, normal=None):
        """
        Apply clipping on a single axis
        :param position: Position
        :param axis: Clipping axis, defaults to 0 (X axis)
        :param thickness: Whether a thickness (so two clipping planes) are applied
        """
        axis_offset = 0
        # This should already be sorted in the model but in case it isn't, we double check here
        if normal is not None and normal[axis] < 0:
            # This means that the given axis has two 
            # clipping planes and we take the negative one
            axis_offset += 1
            #position = self.model.dimensions - position
        axis_storage_id = axis * 2 + axis_offset
        plane = self.clipping_planes.GetItem(axis_storage_id)
        plane.SetOrigin(position)
        plane.SetNormal(normal)

    def set_volume_clipping(self, on=None):
        """
        Set volume clipping on or off.
        :param on: Whether clipping is enabled or disabled. If None, then
        the state is toggled.
        """
        if on is None:
            self.enable_volume_clipping = not self.enable_volume_clipping
        else:
            self.enable_volume_clipping = on
        if self.enable_volume_clipping:
            self.actor.mapper().SetClippingPlanes(self.clipping_planes)
        else:
            self.actor.mapper().SetClippingPlanes(None)

    def clip_to_bounds(self, bounds):
        """
        Clip the volume and move the slicing planes according to 6 boundary points
        :param bounds: Six values in a list (xmin, xmax, ymin, ymax, zmin, zmax)
        """
        planes = vtk.vtkPlanes()
        planes.SetBounds(bounds)
        # Normals are reversed with the above code
        # so we fix that here with flip_normals=True
        self.set_clipping_planes(planes, flip_normals=True)

    def box_widget_update(self, widget=None, event=None):
        """
        Clip the volume with the current box widget
        :param widget: vtkBoxCutter
        :param event: vtkEvent
        """
        if widget is None:
            return
        planes = vtk.vtkPlanes()
        widget.GetPlanes(planes)
        self.set_clipping_planes(planes)

    def set_clipping_planes(self, planes, flip_normals=False):
        """
        Clip the volume and move the slicing planes according the given planes
        :param planes: vtkPlanes
        """
        vtk_n = planes.GetNormals()
        vtk_pts = planes.GetPoints()
        num_pts = vtk_pts.GetNumberOfPoints()
        for plane_id in range(num_pts):
            normal = vtk_n.GetTuple(plane_id)
            origin = vtk_pts.GetPoint(plane_id)
            plane = self.clipping_planes.GetItem(plane_id)
            current_origin = np.array(plane.GetOrigin())
            # We don't need to check the normal because
            # we prevent box cutter rotation in our case
            if np.linalg.norm(current_origin - origin) < 0.1:
                continue
            plane.SetOrigin(origin)
            if flip_normals:
                normal = np.array(normal)*-1
            plane.SetNormal(normal)
            self.update_slicer(plane_id, origin, normal)

    def set_alpha_map(self, alpha_map, alpha_factor=None):
        """
        Set alpha map to the volume view
        :param alpha_map: 2D list of scalar values and alpha values
        :param alpha_factor: Alpha factor
        """
        if alpha_map is None:
            if self.model.luts.current is None:
                return
            alpha_map = self.model.luts.current.alpha_map
        if alpha_factor is None:
            alpha_factor = self.alpha_factor
        if len(np.array(alpha_map).shape) > 1:
            volume_alpha_map = np.ones_like(alpha_map).astype(float)
            volume_alpha_map[:] = alpha_map[:]
            volume_alpha_map[:, 1] *= alpha_factor
            self.actor.alpha(volume_alpha_map)
        else:
            self.actor.alpha(np.array(alpha_map) * alpha_factor)

    def set_color_map(self, color_map=None, alpha_map=None):
        """
        Set the color and alpha map to the view objects
        :param color_map: Nested list of scalar values and rgb colors
            like [[0, [0.0, 0.0, 0.0]], [8, [0.5, 0.8, 0.3]], ...]
        :param alpha_map: 2D list of scalar values and alpha values
        """
        lut = self.model.luts.current
        if color_map is None and lut is not None:
            color_map = lut.color_map
        if alpha_map is None and lut is not None:
            alpha_map = lut.alpha_map
        if color_map is None:
            return
        self.actor.cmap(color_map)
        self.set_alpha_map(alpha_map)
        
        if lut is not None:
            for surface in self.model.isosurfaces:
                surface._mapper.SetLookupTable(lut.opaque_lut)
            for slicer_id in self.slicers:
                slicer = self.slicers[slicer_id]
                slicer.apply_lut(lut.mapped_lut)
        else:
            for slicer_id in self.slicers:
                slicer = self.slicers[slicer_id]
                slicer.set_color_map(color_map, alpha_map)

    def disable_shading(self):
        """
        Disable volume shading
        """
        volumeProperty = self.actor.GetProperty()
        volumeProperty.ShadeOff()
        self.actor.SetProperty(volumeProperty)

    def enable_shading(self, ambient=0.6, diffuse=0.8, specular=0.9):
        """
        Enable volume shading
        TODO: See if this method is useful
        """
        volumeProperty = self.actor.GetProperty()
        volumeProperty.SetInterpolationTypeToLinear()
        volumeProperty.ShadeOn()
        volumeProperty.SetAmbient(ambient)
        volumeProperty.SetDiffuse(diffuse)
        volumeProperty.SetSpecular(specular)
        volumeProperty.SetScalarOpacityUnitDistance(1)
        self.actor.SetProperty(volumeProperty)

    def toggle_slices_visibility(self):
        """
        Toggle slices visibility
        """
        self.model.slices_visible = not self.model.slices_visible
        for slicer_id in self.slicers:
            slicer = self.slicers[slicer_id]
            self.update_slicer(slicer)
            if slicer.actor is not None:
                slicer.actor.SetVisibility(self.model.slices_visible)

    def toggle_hollow(self):
        """
        Toggle hollow mode for volume rendering. This is intended 
        to work only on segmented (annotated) volumes.
        """
        volume_property = self.actor.GetProperty()
        # Shout at VTK devs: it's twisted to name properties Disable and then have DisableOff...
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
        return not disabled

    def get_value_from_xyz(self, position, normal_step=None, avoid_values=0, cast_to_int=True, none_as_zero=False):
        """
        Get a scalar value from the volume with respect to XYZ coordinates and a optionally a normal step,
        that is the normal on which to probe multiplied by the distance you want to travel further into
        the volume to pick a correct value. Often the "surface point" on a volume with non uniform transparency
        is at the boundary between transparent (let's say a 0 value is transparent) and more opaque parts.
        So you need to go further into the "cloud" so to speak, in order to find the values you want.
        :param position: 3D array
        :param normal_step: A vector normal multiplied by the lookup distance, in case the raw position yields
            bad or unwanted results
        :param avoid_values: Try and find other values than this
        :param cast_to_int: Whether the value should be cast to integer
        :return: Scalar value
        """
        if isinstance(avoid_values, int) or isinstance(avoid_values, float):
            avoid_values = [avoid_values]
        # TODO: see if this is faster? To be tested
        # ijk_result = [0.0, 0.0, 0.0]
        # volume_actor._data.TransformPhysicalPointToContinuousIndex(xyz, ijk_result)
        # volume_actor._data.GetPoint(ijk_result)
        pt_id = self.actor._data.FindPoint(*position)
        valid_id = 0 < pt_id < self.scalars.GetNumberOfValues()
        value = self.scalars.GetValue(pt_id) if valid_id else None
        if not valid_id or (value in avoid_values):
            if normal_step is not None:
                position += normal_step
                pt_id = self.actor._data.FindPoint(*position)
                valid_id = 0 < pt_id < self.scalars.GetNumberOfValues()
                value = self.scalars.GetValue(pt_id) if valid_id else None
        if cast_to_int and value is not None:
            value = int(value)
        if value is None and none_as_zero:
            value = 0
        return value

    def raycast(self, origin, screen_position):
        """
        Shorthand for pick() method
        """
        return self.pick(origin, screen_position)

    def pick(self, origin, screen_position):
        """
        Find the nearest intersection – even on sliced volume – with the ray formed
        by an origin and a screen-space position (given by VTK when you click on an actor)
        :param origin: Origin of the vector
        :param screen_position: 2D position on screen. This is given by vtk events like MouseRelease
        :return: The nearest position and its related value queried in the volume image
        """
        self.picker.Pick(*screen_position[:2], 0, self.plot.renderer)
        position = np.array(self.picker.GetPickPosition())
        ray = position - origin
        distance = np.linalg.norm(ray)
        normal = ray / distance

        # Go half a voxel further to make sure we don't hit "void"
        vol_position = position # + normal * self.model.resolution / 2
        probe_position = position + normal * self.model.resolution * 10

        closest_dist = distance
        slice_position = None
        # See if the line hits any of the slicers (that are image planes)
        for slicer_id in self.slicers:
            slicer = self.slicers[slicer_id]
            if slicer.got_slice:
                hits = slicer.actor.intersectWithLine(origin, probe_position)
                if len(hits) != 1:
                    continue
                new_dist = np.linalg.norm(position - hits[0])
                if new_dist < closest_dist and new_dist < self.model.resolution * 2:
                    closest_dist = new_dist
                    slice_position = hits[0]

        if slice_position is None:
            position = vol_position
        else:
            position = slice_position
        value = self.get_value_from_xyz(position, normal * self.model.resolution * 4)
        return position, value

    def add_probe(self, origin, destination, resolution=40, radius=10, color_map=None, 
                screen_space=True, min_v=None, max_v=None, add_to_scene=True):
        """
        Add a series of points along a line probe
        :param origin: Probe origin
        :param destination: Probe destination point
        :param resolution: Number of (equidistant) points that will be probed along that line
        :param radius: Radius of the points
        :param color_map: Scalars color map
        :param screen_space: Whether the points are screen space or spheres
        :param min_v: Min scalar value
        :param max_v: Max scalar value
        :param add_to_scene: Whether the new probe is added to scene
        :return: Points
        """
        if color_map is None:
            color_map = self.model.luts.current.color_map
        
        positions, values = self.probe(origin, destination, resolution)

        points_obj = obj.Points(positions, values=values, radius=radius, screen_space=screen_space,
                                color_map=color_map, min_v=min_v, max_v=max_v)
        
        points_obj.origin = origin
        points_obj.destination = destination
        # Dynamic properties assignment
        points_obj.target = self.actor
        points_obj.target_controller = self

        if add_to_scene:
            self.plot.add(points_obj)
        return points_obj

    def update_probe(self, origin, destination, points_obj):
        """
        Update a probe with given start and end points
        :param origin: Start point
        :param destination: End point
        :param points_obj: Points object
        """
        resolution = points_obj._polydata.GetPoints().GetNumberOfPoints()
        positions, values = self.probe(origin, destination, resolution)
        points_obj.update_data(positions, values)

    def probe(self, origin, destination, resolution=40):
        """
        Probe a volume with a line
        :param origin: Origin of the line probe
        :param destination: Destination of the line probe
        :param resolution: Number of point samples along the probe
        :return: Positions and values
        """
        origin = np.array(origin)
        destination = np.array(destination)
        distance = np.linalg.norm(destination - origin)
        ray = destination - origin
        ray_norm = ray / distance
        step = distance / resolution
        positions = [origin + ray_norm * p_id * step for p_id in range(resolution)]
        values = np.array([self.get_value_from_xyz(point, none_as_zero=True) for point in positions])
        return positions, values

    def set_interactive_subsampling(self, on=False):
        """
        Set volume subsampling on or off. 
        This is enabled by default in VTK and we disable it by default in IBLViewer
        :param on: Whether volume subsampling in interactive mode is on or off
        """
        #self.plot.window.SetDesiredUpdateRate(0)
        #self.actor._mapper.SetInteractiveUpdateRate(0)
        self.model.interactive_subsampling = on
        self.actor._mapper.SetAutoAdjustSampleDistances(on)
        if on:
            self.actor._mapper.InteractiveAdjustSampleDistancesOn()
        else:
            self.actor._mapper.InteractiveAdjustSampleDistancesOff()

    def isosurface(self, label, exceptions=[0], force_rebuild=False, set_current=True, to_int=True, split_meshes=True):
        """
        Creates a surface mesh (isosurface) of a segmented/labelled volume for the given value.
        Unlike general isosurfacing, this method extracts only the surface mesh of the 
        desired region/label/segmentation, not of all values from 0 to label.
        :param label: Label (scalar) value found in the volume
        :param exceptions: If the label is found in the exceptions list, isosurfacing will not occur
        :param force_rebuild: Whether rebuilding is forced in case we find an existing mesh for the given label
        :param set_current: Whether the label is set as the current one in the model
        :param to_int: Whether the label is cast to integer
        :param split_meshes: Whether we split meshes when multiple ones are found
        :return: A list of all manifold meshes for the given label
        """
        if label is None or label in exceptions:
            return
        if to_int:
            label = int(label)
        existing_meshes = self.model.isosurfaces.get(label)
        if existing_meshes is not None and not force_rebuild:
            return existing_meshes

        lut = self.model.luts.current
        simple_lut = vtk.vtkLookupTable()
        simple_lut.SetNumberOfColors(1)
        simple_lut.SetTableRange(0, 1)
        simple_lut.SetScaleToLinear()
        simple_lut.SetTableValue(0, 0, 0, 0, 0)
        simple_lut.SetTableValue(1, *lut.mapped_lut.GetTableValue(label))
        simple_lut.Build()

        # Generate object boundaries from labelled volume 
        discrete = vtk.vtkDiscreteMarchingCubes()
        discrete.SetInputData(self.actor.imagedata())
        discrete.GenerateValues(1, label, label)

        smoothing_iterations = 15
        pass_band = 0.001
        feature_angle = 120.0

        smoother = vtk.vtkWindowedSincPolyDataFilter()
        smoother.SetInputConnection(discrete.GetOutputPort())
        smoother.SetNumberOfIterations(smoothing_iterations)
        smoother.BoundarySmoothingOff()
        smoother.FeatureEdgeSmoothingOff()
        smoother.SetFeatureAngle(feature_angle)
        smoother.SetPassBand(pass_band)
        smoother.NonManifoldSmoothingOn()
        smoother.NormalizeCoordinatesOn()
        smoother.Update()
        
        self.model.isosurfaces[label] = []
        #splitter = vtk.vtkExtractPolyDataGeometry()
        if split_meshes:
            splitter = vtk.vtkPolyDataConnectivityFilter()
            splitter.SetInputConnection(smoother.GetOutputPort())
            splitter.SetExtractionModeToAllRegions()
            splitter.ColorRegionsOn()
            splitter.Update()

            for region_id in range(splitter.GetNumberOfExtractedRegions()):
                #splitter.AddSpecifiedRegion(region_id)
                #splitter.Update()
                
                #poly = vtk.vtkPolyData()
                #poly.ShallowCopy(splitter.GetOutput())

                threshold = vtk.vtkThreshold()
                threshold.SetInputConnection(splitter.GetOutputPort())
                threshold.ThresholdBetween(region_id, region_id)
                threshold.Update()
                actor = vedo.Mesh(threshold.GetOutput())
                #actor._mapper.SetScalarRange(min_value, lut.scalar_max)
                #actor._mapper.SetUseLookupTableScalarRange(True)
                actor._mapper.SetLookupTable(simple_lut)
                actor._mapper.ScalarVisibilityOn()
                actor.name = 'Isosurface_' + str(label)
                self.model.isosurfaces[label].append(actor)
                #actor.cmap(lut.scalar_lut, np.ones(poly.GetNumberOfVerts())*label)
        else:
            poly = smoother.GetOutput()
            actor = vedo.Mesh(poly)
            actor._mapper.SetLookupTable(simple_lut)
            actor._mapper.ScalarVisibilityOn()
            actor.name = 'Isosurface_' + str(label)
            self.model.isosurfaces[label].append(actor)

        '''
        pdnorm = vtk.vtkPolyDataNormals()
        pdnorm.SetInputData(smoother.GetOutput())
        pdnorm.ComputePointNormalsOn()
        pdnorm.ComputeCellNormalsOn()
        pdnorm.FlipNormalsOff()
        pdnorm.ConsistencyOn()
        pdnorm.Update()
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(smoother.GetOutputPort())
        mapper.SetLookupTable(lut.scalar_lut)
        mapper.SetScalarRange(min_value, lut.scalar_max)
        '''
        if set_current:
            self.model.isosurfaces.set_current(label)
        return self.model.isosurfaces[label]


@dataclass
class SlicerModel:
    PREFIX = '[Slicer]_'
    MIN_SLAB_THICKNESS = 1.0 #um

    __count = 0

    def unique_name():
        SlicerModel.__count += 1
        return f'{SlicerModel.PREFIX}_{SlicerModel.__count}'

    name: str = field(default_factory=unique_name)
    # 0, 1 or 2. See the normal for axis orientation
    axis: int = None
    value: float = 0.0
    bounds: np.ndarray = None
    #thickness: float = 0.0
    origin: np.ndarray = np.array([0.0, 0.0, 0.0])
    normal: np.ndarray = np.array([1.0, 0.0, 0.0])
    clipping_planes: vtk.vtkPlaneCollection = None

    def get_box_plane_id(self):
        """
        Get the plane id
        :return: Int
        """
        if self.axis is None:
            return
        offset = 0 if self.normal[self.axis] < 0 else 1
        return self.axis * 2 + offset

    def get_axis_aligned_info(self, vtk_axis):
        """
        VTK stores box clipping planes in the order:
        -X to +X: 0, 1
        -Y to +Y: 2, 3
        -Z to +Z: 4, 5
        This method retrieves what is the XYZ axis (0, 1 or 2)
        and its orientation sign
        :return: Int axis and float orientation
        """
        orientation = -1.0 if vtk_axis % 2 == 0 else 1.0
        axis = (vtk_axis - vtk_axis % 2) // 2
        return axis, orientation

    def align_to_axis(self, axis, dimensions=None):
        """
        Set the axis of the slicer
        :param axis: See parameter vtk_axis in SlicerModel.get_axis_aligned_info()
        :param dimensions: Dimensions of the volume
        """
        if not isinstance(axis, int):
            return
        normal = np.zeros(3).astype(float)
        xyz_axis, orientation = self.get_axis_aligned_info(axis)
        normal[xyz_axis] = orientation
        self.axis = xyz_axis
        if dimensions is not None and orientation < 0:
            self.origin = np.zeros(3)
            self.origin[xyz_axis] = dimensions[xyz_axis]
        self.normal = normal

    def flip_normal(self):
        """
        Flip the normal of the slicer
        """
        self.normal *= -1.0
        self.check_normal()
        if isinstance(self.axis, int):
            self.axis *= -1

    def check_normal(self):
        """
        Check if the normal is axis-aligned.
        If not, the axis is set to None.
        """
        zeros = self.normal == 0
        if len(self.normal[zeros]) >= 2:
            self.axis = 0

    def update(self, value=None, normal=None, axis=None):
        """
        Update slicer
        :param value: Origin of the slicing plane
        :param normal: Normal of the slicing plane
        :param axis: Axis, if the plane is axis-aligned
        :return: True if model changed, False if it didn't
        """
        if not(isinstance(value, int) or isinstance(value, float)):
            if normal is None:
                normal = self.normal
            if normal is None:
                return False
            if normal[1] == 0 and normal[2] == 0:
                axis = 0 #if normal[0] > 0 else 1
            elif normal[0] == 0 and normal[2] == 0:
                axis = 1 #if normal[1] > 0 else 1
            elif normal[0] == 0 and normal[1] == 0:
                axis = 2 #if normal[2] > 0 else 1
            if axis is not None:
                value = value[axis]
        if axis is None:
            axis = self.axis
        if self.value == value:
            return False
        if axis is not None:
            self.value = value
            self.origin = np.array(normal) * value
        else:
            self.value = None
            self.origin = value
        self.normal = normal
        self.axis = axis
        return True


class SlicerView():

    slices = {}

    def __init__(self, plot, volume_view, slicer_model, standalone=True):
        """
        Constructor
        :param plot: Plot instance
        :param volume_view: VolumeView instance
        :param slicer_model: SlicerModel instance
        :param standalone: Whether the slice is a standalone actor that
        can be clicked. Set this to False if you want to use transparency,
        at the expense that because of a VTK bug, you won't be able to
        click on it anymore, requiring you to code another way of detecting
        where the user clicked. See more in initialize_mapper()
        """
        self.plot = plot
        self.volume_view = volume_view
        self.model = slicer_model
        self.actor = None
        self.filter = None
        self.filter = None
        self.actor = None
        self.reslice = None
        self.slice_type = -1
        self.depth_peeling_enabled = None
        self.standalone = standalone
        self.got_slice = False
        self.color_map = None
        self.alpha_map = None
        self.initialize()

    def initialize(self, render=False):
        """
        Initialize the slicer object
        """
        if self.filter is None:
            self.filter = vtk.vtkImageDataGeometryFilter()
        
        if self.actor is None:
            self.actor = vedo.Mesh(self.filter.GetOutput())
            # Adding empty actor so that it's updated later on
            self.plot.add(self.actor, render=render)
            self.actor.lighting('off')
            self.actor.name = self.model.name

        self.initialize_mapper()

    def initialize_mapper(self):
        """
        Initialize the object mapper
        """
        mapper = self.actor._mapper
        mapper.SetScalarModeToUsePointData() #SetScalarModeToUsePointFieldData
        mapper.SetColorModeToMapScalars()
        mapper.ScalarVisibilityOn()
        # We operate on static volumes thanks to the double LUT mapping implemented here
        mapper.SetStatic(True)

        # Without using scalar range, the mapping will be off
        mapper.SetUseLookupTableScalarRange(True)
        
        # We prevent this actor from being pickable as a result of the bug described below
        # when we want to use transparency on the slice.
        self.actor.pickable(self.standalone)
        if self.standalone:
            # There is a bug in VTK 9 that prevents clicking on transparent objects
            # as reported on vedo's tracker https://github.com/marcomusy/vedo/issues/291
            # The "Force opaque fix" below should be gone with the next VTK update hopefully.
            # In the meantime, we use this.
            # TODO: remove this when this bug is fixed in VTK
            self.actor.ForceOpaqueOn()
        else:
            # We bypass the transparent selection bug when a VolumeView has multiple slicers 
            # like in box mode because the click detection occurs on the volume and we perform 
            # an additional test to see if a slicer yields a nearby result. If it does, 
            # the result is like clicking on the slice and we get transparency for free.
            pass

        # Make sure we have depth peeling activated, otherwise transparency with volumes
        # will look weird and in the wrong order
        self.plot.renderer.UseDepthPeelingOn()
        self.plot.renderer.UseDepthPeelingForVolumesOn()
        
        segmented = self.volume_view.model.is_segmented()
        if segmented:
            # This very line below will mess up the entire slice coloring if:
            # - you have a segmented volume and this is set to True
            # - you have a non-segmented (like raw MRI, CT) volume and this is set to False
            mapper.SetInterpolateScalarsBeforeMapping(not segmented)
        mapper.Update()

    def set_color_map(self, color_map, alpha_map=None):
        """
        Set a color map to the slice
        :param color_map: Color map, can be a string, a list of colors or more. 
            See vedo documentation.
        """
        self.color_map = color_map
        if alpha_map is not None:
            self.alpha_map = alpha_map
        if self.got_slice and color_map is not None:
            self.actor.cmap(self.color_map, alpha=self.alpha_map)

    def set_slice_type(self, slice_type):
        """
        Set the slice type. 0 for axial, 1 for free slicing
        :param slice_type: Int value
        """
        if slice_type == 0 and self.slice_type != slice_type:
            self.slice_type = slice_type
            self.filter.SetInputData(self.volume_view.actor.imagedata())
        elif slice_type == 1 and self.slice_type != slice_type:
            self.slice_type = slice_type
            self.filter.SetInputData(self.reslice.GetOutput())

    def slice_on_normal(self, origin, normal):
        """
        Slice a volume with a plane oriented by the given normal.
        This allows slicing in all directions.
        :param origin: Origin of the slicing plane
        :param normal: Normal of the slicing plane
        :return: Mesh object with the slice as an image texture
        """
        '''
        mapper = vtk.vtkImageResliceMapper()
        mapper.SetInputData(self.volume_view.actor._data)
        mapper.SliceFacesCameraOff()
        mapper.SliceAtFocalPointOff()
        mapper.JumpToNearestSliceOn()
        mapper.SetImageSampleFactor(2)
        mapper.BorderOn()
        mapper.BackgroundOff()
        mapper.UpdateInformation()
        mapper.GetSlicePlane().SetOrigin(*origin)
        mapper.GetSlicePlane().SetNormal(*normal)
        mapper.GetSlicePlane().Modified()
        mapper.Modified()
        mapper.Update()

        self.actor = vtk.vtkImageSlice()
        self.actor.SetMapper(mapper)
        prop = vtk.vtkImageProperty()
        if True:
            prop.SetInterpolationTypeToLinear()
        else:
            prop.SetInterpolationTypeToNearest()
        self.actor.SetProperty(prop)
        return
        '''
        if self.reslice is None:
            reslice = vtk.vtkImageReslice()
            reslice.SetInputData(self.volume_view.actor._data)
            #reslice.SetInputData(image)
            reslice.SetOutputDimensionality(2)
            reslice.SetAutoCropOutput(False)
            #reslice.SetInterpolationModeToLinear()
            reslice.SetInterpolationModeToNearestNeighbor()
            reslice.SetSlabNumberOfSlices(1)
            reslice.SetOutputSpacing(self.volume_view.get_spacing())
            reslice.ReleaseDataFlagOn()
            self.reslice = reslice
        
        self.set_slice_type(1)
        M, T = utils.get_transformation_matrix(origin, normal)
        self.reslice.SetResliceAxes(M)
        self.reslice.Update()
        self.filter.Update()

        if self.actor is None:
            self.actor = vedo.Mesh(self.filter.GetOutput())
            self.initialize_mapper()
        else:
            self.actor._update(self.filter.GetOutput())

        self.initialize_mapper()

        self.actor.SetOrientation(T.GetOrientation())
        self.actor.SetPosition(origin)
        self.got_slice = True
        return self.actor

    def x_slice(self, i):
        """
        Extract the slice at index `i` of volume along x-axis.
        :param i: I index
        """
        self.set_slice_type(0)
        nx, ny, nz = self.volume_view.actor.GetMapper().GetInput().GetDimensions()
        if i <= 1 or i > nx - 1:
            return False
        self.filter.SetExtent(i, i, 0, ny, 0, nz)
        self.filter.Update()
        if self.actor is not None:
            self.actor._update(self.filter.GetOutput())
        else:
            self.actor = vedo.Mesh(self.filter.GetOutput())
            self.initialize_mapper()
        self.got_slice = True
        return True

    def y_slice(self, j):
        """
        Extract the slice at index `j` of volume along y-axis.
        :param j: J index
        """
        self.set_slice_type(0)
        #nx, ny, nz = self.volume_view.model.dimensions / resolution
        nx, ny, nz = self.volume_view.actor.GetMapper().GetInput().GetDimensions()
        if j <= 1 or j > ny - 1:
            return False
        self.filter.SetExtent(0, nx, j, j, 0, nz)
        self.filter.Update()
        if self.actor is not None:
            self.actor._update(self.filter.GetOutput())
        else:
            self.actor = vedo.Mesh(self.filter.GetOutput())
            self.initialize_mapper()
        self.got_slice = True
        return True

    def z_slice(self, k):
        """
        Extract the slice at index `k` of volume along z-axis.
        :param k: K index
        """
        self.set_slice_type(0)
        nx, ny, nz = self.volume_view.actor.GetMapper().GetInput().GetDimensions()
        if k <= 1 or k > nz - 1:
            return False
        self.filter.SetExtent(0, nx, 0, ny, k, k)
        self.filter.Update()
        if self.actor is not None:
            self.actor._update(self.filter.GetOutput())
        else:
            self.actor = vedo.Mesh(self.filter.GetOutput())
            self.initialize_mapper()
        self.got_slice = True
        return True

    def slice_on_axis(self, value=None, normal=None, axis=None, use_reslice=False):
        """
        Slice on standard X, Y or Z axis
        :param value: Value on the given axis
        :param normal: Axis normal, can be either +1.0 or -1.0 along that axis
        :param axis: Axis integer, 0 for X, 1 for Y, 2 for Z
        :param use_reslice: if True, this enables vtkImageReslice which is useful when
            the normal is not aligned to either X, Y or Z. If you use it on an axis-aligned
            normal, some color inaccuracies will appear if you don't tweak the vtkImageResliceMapper.
            This is why the default is False.
        :return: Result boolean, whether slice occured or not
        """
        resolution = self.volume_view.model.resolution
        volume_dimensions = self.volume_view.model.dimensions
        '''
        if normal[axis] < 0:
            if value > 0:
                # Make value consistent with given normal.
                value *= normal[axis]
            value = volume_dimensions[axis] + value
        '''
        in_volume_slice = int(value) // resolution

        if use_reslice:
            self.slice_on_normal(normal * value, normal)
            return

        if axis == 0:
            result = self.x_slice(in_volume_slice)
        elif axis == 1:
            result = self.y_slice(in_volume_slice)
        elif axis == 2:
            result = self.z_slice(in_volume_slice)
        return result

    def update(self):
        """
        Update slice object according to data in the model
        """
        had_slice = self.got_slice
        result = True
        if isinstance(self.model.axis, int) and 0 <= self.model.axis <= 2:
            result = self.slice_on_axis(self.model.value, self.model.normal, self.model.axis)
        else:
            self.slice_on_normal(self.model.origin, self.model.normal)

        if not result:
            self.plot.remove(self.actor)
            self.got_slice = False
            return

        #self.actor.pos(*(self.volume_view.actor.pos()-self.actor.pos()))
        lut = self.volume_view.model.luts.current
        if lut is not None:
            '''
            This is VTK for you...a mesh can use a vtkLookupTable for RGBA mapping
            BUT volumes require vtkColorTransferFunction (RGB) and vtkPiecewiseFunction (alpha)
            So we have to put a color map, alpha map and a vtkLookupTable 
            built from both maps in a LUTModel.
            Alternatively, we could update the LUT with alpha values but it's a pain.
            
            ctf = self.volume_view.actor.GetProperty().GetRGBTransferFunction()
            lut = vedo.utils.ctf2lut(self.volume_view.actor)
            otf = self.volume_view.actor.GetProperty().GetScalarOpacity

            # using "ctf" would work only for colors, not for transparency!
            self.apply_lut(ctf)
            '''
            self.apply_lut(lut.mapped_lut)
        else:
            if self.alpha_map is None:
                self.actor.cmap(self.color_map)
            else:
                self.actor.cmap(self.color_map, alpha=self.alpha_map)
            
        if self.model.clipping_planes is not None:
            self.actor.mapper().SetClippingPlanes(self.model.clipping_planes)

        if not had_slice:
            self.plot.add(self.actor, render=True)

    def apply_lut(self, lut=None):
        """
        Apply a LUT to the volume
        :param lut: vtkLookupTable
        :param actor: The actor to receive this
        """
        if self.actor is None or lut is None:
            return
        mapper = self.actor._mapper
        mapper.SetLookupTable(lut)