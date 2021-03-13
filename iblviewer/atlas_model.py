
from dataclasses import dataclass, field
from typing import Mapping, List, Any
from datetime import datetime
import os
import numpy as np
import pandas as pd

import vtk
from vedo import *
import ibllib.atlas

import logging

from iblviewer.transfer_function_model import TransferFunctionModel
from iblviewer.volume_model import VolumeModel
from iblviewer.slicer_model import SlicerModel
import iblviewer.utils as utils


BASE_PATH = utils.split_path(os.path.realpath(__file__))[0]
REASSIGNED_VOLUME_SUFFIX = '_row_scalars'


# A camera model does not necessarily reflect the view. It's a state that can be set
# to a view. It's also a state that can be retrieved from the view.
@dataclass
class CameraModel:
    name: str = '[Camera]'
    # Could be a vtk matrix or a numpy 4x4 array (or a list of lists)
    #matrix: 'typing.Any'
    # -1 means custom normal then 0, 1, 2 represent x, y and z
    axis: int = 0
    normal: np.ndarray = np.array([1.0, 0.0, 0.0])
    distance: float = 0.0
    orthographic: bool = False # Vedo stores this as useParallelProjection in vedo.settings

    focal_point: np.ndarray = np.array([0.0, 0.0, 0.0])
    target: Any = None
    up_axis: np.ndarray = np.array([0.0, 1.0, 0.0])
    distance_factor: int = 3
    orthographic_scale_factor: int = 1

    def is_orthographic(self):
        return self.orthographic == True


@dataclass
class AtlasUIModel:
    CONTEXTS = ['default', 'slicing', 'visiblity_groups', 'time_series', 'measuring', 'movie_export']

    font = 'Source Sans Pro'
    font_scale = 0.75
    toggle_config = {'c':["black", "#eeeeee"], 'bc':["#dddddd", "#999999"], 'font':font, 'size':14, 'bold':False, 'italic':False}
    button_config = {'c':["black", "black"], 'bc':["#dddddd", "#dddddd"], 'font':font, 'size':14, 'bold':False, 'italic':False}
    slider_config = {'titleSize':font_scale, 'font':font}

    atlas_visible: bool = True
    context: str = CONTEXTS[0]
    jupyter: bool =  False
    embed: bool = True

    def set_context(self, context):
        """
        Set the context
        :param context: A context from CONTEXTS
        """
        if context is None:
            return
        if context in AtlasUIModel.CONTEXTS:
            self.context = context
        else:
            logging.error('Context ' + str(context) + ' is invalid. Ignoring it.')


@dataclass
class AtlasModel:
    # Default origin is "bregma", an origin defined at the center of the XY axes (not on Z)
    # For reference, bregma is np.array([5739.0, 5400.0, 332.0])
    # And for some reason, it's not exactly in the middle of X axis (should be 5400.0)...
    IBL_BREGMA_ORIGIN = ibllib.atlas.ALLEN_CCF_LANDMARKS_MLAPDV_UM['bregma']
    ALLEN_ATLAS_RESOLUTIONS = [10, 25, 50, 100]
    LINES_PREFIX = '[Lines]'
    POINTS_PREFIX = '[Points]'

    runtime: datetime = datetime.now()
    # State vars
    camera: CameraModel = None
    atlas_transfer_function: TransferFunctionModel = None
    transfer_function: TransferFunctionModel = None
    transfer_function_id: int = 0
    volume: VolumeModel = None
    slicer: SlicerModel = None
    selection: Any = None

    animation_playing: bool = False
    animation_function: Any = None
    time: float = 0.0

    origin: np.ndarray = IBL_BREGMA_ORIGIN
    # Atlas wrapper, in the case of IBL: ibllib.atlas.AllenAtlas
    atlas: Any = None
    atlas_mapping_ids: list = None

    cameras: dict = field(default_factory=dict)

    volumes: dict = field(default_factory=dict)
    transfer_functions: dict = field(default_factory=dict)
    slicers: dict = field(default_factory=dict)

    points: dict = field(default_factory=dict)
    lines: dict = field(default_factory=dict)
    settings: dict = field(default_factory=dict)

    def get_num_scalars(self):
        return self.atlas.regions.id.size

    def find_model(self, substring, models):
        """
        Find a model name within a dictionary
        :param substring: Substring to find
        :param models: Dictionary of models
        :return: Model instance found, otherwise None
        """
        for key in models:
            if substring in key:
                return models.get(key)

    def add_model(self, model_name, models, model_class):
        """
        Create and store a model
        :param model_name: A model name 
        :param models: Dictionary of models
        :param model_class: The class of the model
        :return: A model of type model_class
        """
        return self.get_model(model_name, models, model_class)

    def get_model(self, model, models, model_class):
        """
        Get a model from a dictionary of models or create it if none found.
        This method works only with Models that have a name (str) property.
        :param model: Either a model name or the model itself.
        If a string is given and no such named model is found, 
        a new one is created with the model_type (the class of the model)
        :param models: Dictionary of models
        :param model_class: The class of the model
        :return: A model of type model_class
        """
        if isinstance(model, str):
            model_inst = models.get(model)
            if model_inst is None:
                model_inst = model_class(name=model)
            models[model] = model_inst
            return model_inst
        elif hasattr(model, 'name') and models.get(model.name) == model:
            return model

    def store_model(self, model, storage_property):
        """
        Store a model in a dictionary, the key being its name property
        :param model: Model instance
        :param storage_property: A dict that will store this model
        """
        if not hasattr(model, 'name') or model.name == None:
            logging.error('Model has no name and cannot be registered: ' + str(model))
            return
        storage_property[model.name] = model

    def remap(self, ids, source='Allen', dest='Beryl'):
        """
        Remap ids/scalar values from source to destination
        Function by Olivier Winter
        """
        #from ibllib.atlas import BrainRegions as br
        from brainbox.numerical import ismember
        _, inds = ismember(ids, self.atlas.regions.mappings[source])
        return self.atlas.regions.mappings[dest][inds]

    def initialize(self, resolution=None):
        """
        Get Allen Atlas metadata from their CSV file embed in IBL module
        """
        if resolution not in AtlasModel.ALLEN_ATLAS_RESOLUTIONS:
            resolution = AtlasModel.ALLEN_ATLAS_RESOLUTIONS[-1]

        logging.info('[AtlasModel] origin: ' + str(self.origin))
        logging.info('[AtlasModel] resolution: ' + str(resolution))

        default_name = 'Atlas'
        self.atlas = ibllib.atlas.AllenAtlas(resolution)
        self.atlas_mapping_ids = list(self.atlas.regions.mappings.keys())

        self.volume = self.get_model('Mouse atlas CCF v3', self.volumes, VolumeModel)
        self.volume.resolution = resolution
        self.transfer_function = self.build_atlas_transfer_function(default_name)
        self.atlas_transfer_function = self.transfer_function

        self.camera = self.add_model('Camera', self.cameras, CameraModel)
        self.ui = AtlasUIModel()

    def get_name(self, *args):
        """
        Get full name for a model, separated by underscores
        """
        return '_'.join(args)

    def initialize_slicers(self):
        """
        Initialize slicer models
        """
        # By default, we create x, y, z and custom-normal slicers. These are just models.
        # If you want to use them, you have to create SlicerView instances too.

        default_name = 'Atlas'
        pn = SlicerModel.NAME_XYZ_POSITIVE
        prefix = SlicerModel.NAME_PREFIX
        x_slicer = self.add_model(SlicerModel.NAME_PREFIX + pn[0], self.slicers, SlicerModel)
        x_slicer.set_axis(0)
        y_slicer = self.add_model(SlicerModel.NAME_PREFIX + pn[1], self.slicers, SlicerModel)
        y_slicer.set_axis(1)
        z_slicer = self.add_model(SlicerModel.NAME_PREFIX + pn[2], self.slicers, SlicerModel)
        z_slicer.set_axis(2)
        # Opposite side of the same axes
        nn = SlicerModel.NAME_XYZ_NEGATIVE
        x_slicer = self.add_model(SlicerModel.NAME_PREFIX + nn[0], self.slicers, SlicerModel)
        x_slicer.set_axis(0, True)
        y_slicer = self.add_model(SlicerModel.NAME_PREFIX + nn[1], self.slicers, SlicerModel)
        y_slicer.set_axis(1, True)
        z_slicer = self.add_model(SlicerModel.NAME_PREFIX + nn[2], self.slicers, SlicerModel)
        z_slicer.set_axis(2, True)
        
        n_slicer = self.add_model('N slicer', self.slicers, SlicerModel)
        n_slicer.set_axis([0.0, 1.0, 0.0])
        n_slicer = self.add_model('-N slicer', self.slicers, SlicerModel)
        n_slicer.set_axis([0.0, -1.0, 0.0])
        n_slicer.flip_normal()
        #self.store_model(self.transfer_function, self.transfer_functions)

    def load_allen_volume(self, mapping=None, mode=None):
        """
        Load the volume to visualize
        :param mapping: Mapping, optional. This will process the volume and reassign values depending
        on the chosen mapping.
        """
        self.volume.mapping = mapping
        if mode == 0 or mode == 'dwi':
            self.set_allen_dwi_volume()
        else:
            self.set_allen_segmented_volume(self.volume.mapping)

    def set_allen_segmented_volume(self, atlas_mapping=None):
        """
        Set the segmented (labelled) volume, aka taxonomy, with the given mapping
        :param atlas_mapping: Mapping, either a string for the name of the mapping or an integer. 
        because such level of detail is not useful for analysis (2021).
        """
        self.volume.volume_type = VolumeModel.VOLUME_TYPES.get(VolumeModel.SEGMENTED)
        self.volume.volume = self.get_mapped_volume(self.atlas.label, atlas_mapping, True)

    def set_allen_dwi_volume(self, atlas_mapping=None):
        """
        Set the diffusion weighted average imaging with the given mapping
        :param atlas_mapping: Mapping, either a string for the name of the mapping or an integer.
        because such level of detail is not useful for analysis (2021).
        """
        self.volume.volume_type = VolumeModel.VOLUME_TYPES.get(VolumeModel.DWI)
        self.volume.volume = self.get_mapped_volume(self.atlas.image, atlas_mapping, True)

    def untranspose(self, volume):
        """
        Procedure to reverse the (wrong) transposition from volume loaded with pynrrd by ibllib.atlas.AllenAtlas
        :param volume: Given volume
        """
        return np.transpose(volume, (2, 0, 1))

    def get_mapped_volume(self, volume, atlas_mapping=None, ibl_back_end=True):
        """
        Set the volume data according to a mapping
        :param volume: Given volume to display
        :param atlas_mapping: Mapping, either a string for the name of the mapping or an integer. 
        :param ibl_back_end: If you are not using ibllib and want to load your own volume, set this to False
        so that there will be no transposition of the volume (needed for the ones from IBL)
        :return: Volume nd array
        """
        volume_data = None
        if ibl_back_end:
            if isinstance(atlas_mapping, int):
                if atlas_mapping > len(self.atlas_mapping_ids) - 1:
                    logging.error('[AtlasModel.get_mapped_volume()] could not find atlas mapping with id ' + str(atlas_mapping) + '. Returning raw volume...')
                    return volume
                map_id = self.atlas_mapping_ids[atlas_mapping]
            elif atlas_mapping is None:
                map_id = self.atlas_mapping_ids[0]
            else:
                map_id = atlas_mapping

            # This mapping actually changes the order of the axes in the volume...
            volume_data = self.atlas.regions.mappings[map_id][volume]
            # Undoing something wrong done in IBL back-end when reading the nrrd file or when applying the mapping
            volume_data = self.untranspose(volume_data)
            logging.info('Loaded atlas volume with ' + map_id + ' mapping')
        else:
            volume_data = volume
        return volume_data

    def get_region_and_row_id(self, acronym):
        """
        Get region and row id given an acronym
        :param acronym: Acronym of a brain region
        :return: Region id and row id
        """
        ind = np.where(self.atlas.regions.acronym == 'PRC')[0]
        if ind.size < 1:
            return
        return self.atlas.regions.id[ind], ind

    def get_transfer_function_and_id(self, index):
        """
        Get a transfer function based on its index
        :param index: Integer value, it will be bounded by the number of stored transfer functions
        :return: Index (int) and TransferFunctionModel
        """
        if not isinstance(index, int):
            return
        ids = list(self.transfer_functions.keys())
        num_ids = len(ids)
        transfer_function_id = min(num_ids-1, max(0, index))
        return transfer_function_id, self.transfer_functions[ids[transfer_function_id]]

    def set_transfer_function(self, index):
        """
        Set the current transfer function given its storage key id (0, 1, 2, ..., n)
        :param index: Integer value, it will be bounded by the number of stored transfer functions
        """
        if not isinstance(index, int):
            return
        ids = list(self.transfer_functions.keys())
        num_ids = len(ids)
        self.transfer_function_id = min(num_ids-1, max(0, index))
        self.transfer_function = self.transfer_functions[ids[self.transfer_function_id]]

    # TODO: see if we need a transfer function builder or we leave it to the user to make it (might be too specific to every case)
    # :param scalar_map: Map of custom data to the scalar values of the volume
    # def build_transfer_function(scalar_map=None, color_function=None, unmapped_color=None, force_rebuild=False):

    def build_atlas_transfer_function(self, model,  make_active=True):
        """
        Build a transfer function for the atlas volume
        :param model: Either a TransferFunction model or a name to get or create a new one
        :param make_active: Whether this one is made active (you still have to update the views after that)
        :return: TransferFunctionModel
        """
        model = self.add_model(model, self.transfer_functions, TransferFunctionModel)
        nregions = self.atlas.regions.id.size
        scalar_map = {}
        for ir in range(nregions):
            scalar_map[ir] = ir

        rgb = []
        alpha = []
        # Vedo works with nested lists with rows: [region_id, [r, g, b]] for color, and [region_id, a] for alpha
        for r_id in range(nregions):
            rgb.append([r_id, colors.getColor(self.atlas.regions.rgb[r_id])])
            # First region in atlas is not a region (void) so we make it transparent
            alpha.append([r_id, 0.0 if r_id <= 0 else 1.0])
        rgb = np.array(rgb, dtype=object)
        alpha = np.array(alpha)
        model.set_data(scalar_map, rgb, alpha)
        if make_active:
            self.transfer_function = model
        return model

    def get_regions_mask(self, region_ids, alpha_map=None):
        """
        Build an alpha map that reveals only the given region ids
        :param region_ids: List or numpy array of region ids
        :param alpha_map: Optional alpha map that will be modified. If None provided, the method will attempt
        to use the current active alpha map
        :return: 2D numpy array with region ids and corresponding alpha values
        """
        if alpha_map is None:
            alpha_map = self.transfer_function.alpha_map
        if alpha_map is None:
            logging.error('[Method build_regions_alpha_map()] requires that an alpha map is created by build_regions_alpha_map()')
            return
        new_alpha_map = np.zeros_like(alpha_map).astype(float)
        new_alpha_map[:, 0] = alpha_map[:, 0]
        new_alpha_map[region_ids, 1] = alpha_map[region_ids, 1]
        self.transfer_function.sec_alpha_map = new_alpha_map
        return new_alpha_map
