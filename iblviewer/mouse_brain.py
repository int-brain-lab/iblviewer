from dataclasses import dataclass
import os
import logging
import numpy as np
import requests
from pathlib import Path
import textwrap

import vedo
import nrrd
import ibllib.atlas
from ibllib.atlas.regions import BrainRegions
from iblutil.numerical import ismember

from iblviewer.application import Viewer
from iblviewer.volume import VolumeController, VolumeModel, LUTModel
import iblviewer.utils as utils

ALLEN_ATLAS_RESOLUTIONS = [10, 25, 50, 100]
ALLEN_ATLASES = {'base_url': 'http://download.alleninstitute.org/informatics-archive',
                'mouse_ccf_folder': '/current-release/mouse_ccf',
                'atlas_folder': '/annotation/ccf_2017', 'atlas_prefix': '/annotation_',
                'dwi_folder': '/average_template', 'dwi_prefix': '/average_template_',
                'volume_extension': '.nrrd'}

# Default origin is "bregma", an origin defined at the center of the XY axes (not on Z)
# For reference, bregma is np.array([5739.0, 5400.0, 332.0])
# And for some reason, it's not exactly in the middle of X axis (should be 5400.0)...
IBL_BREGMA_ORIGIN = ibllib.atlas.ALLEN_CCF_LANDMARKS_MLAPDV_UM['bregma']
BASE_PATH = utils.split_path(os.path.realpath(__file__))[0]
REMAPPED_VOLUME_SUFFIX = '_remapped'
_logger = logging.getLogger('ibllib')
LUT_VERSION = 'v01' # version 01 is the lateralized version


class AllenAtlasExt(ibllib.atlas.AllenAtlas):
    """
    This overwrites the constructor of AllenAtlas that is not designed to be used for the
    public, that is people outside of IBL. Typically, you'd want to display the Allen volume
    data in this viewer and perform additional tasks (such as loading your own extra data)
    with other libraries. Dev note: I'm forced to copy and modify the whole constructor in this case.

    Instantiates an atlas.BrainAtlas corresponding to the Allen CCF at the given resolution
    using the IBL Bregma and coordinate system
    """
    @staticmethod
    def _read_volume(file_volume):
        if file_volume.suffix == '.nrrd':
            volume, _ = nrrd.read(file_volume, index_order='C')  # ml, dv, ap
            # we want the coronal slice to be the most contiguous
            volume = np.transpose(volume, (2, 0, 1))  # image[iap, iml, idv]
        elif file_volume.suffix == '.npz':
            volume = np.load(str(file_volume), allow_pickle=True)['arr_0']
        return volume

    def __init__(self, res_um=25, brainmap='Allen', scaling=np.array([1, 1, 1]),
                image_file_path=None, label_file_path=None):
        """
        :param res_um: 10, 25 or 50 um
        :param brainmap: defaults to 'Allen', see ibllib.atlas.BrainRegion for re-mappings
        :param scaling: scale factor along ml, ap, dv for squeeze and stretch ([1, 1, 1])
        :return: atlas.AllenAtlasGP
        """
        xyz2dims = np.array([1, 0, 2]) # this is the c-contiguous ordering
        dims2xyz = np.array([1, 0, 2])
        # we use Bregma as the origin
        self.res_um = res_um
        dxyz = self.res_um * 1e-6 * np.array([1, -1, -1]) * scaling

        if label_file_path is None:
            # No point in going further
            return

        regions = BrainRegions()
        ibregma = (ibllib.atlas.ALLEN_CCF_LANDMARKS_MLAPDV_UM['bregma'] / self.res_um)
        #image_path = atlas_path.joinpath(f'average_template_{res_um}.nrrd')
        # It is really unfortunate that users are forced to have in memory two volumes: the atlas and the DWI!
        image = AllenAtlasExt._read_volume(Path(image_file_path))
        label = self.remap_atlas(label_file_path, regions, ibregma)

        # This calls BrainAtlas, the mother class of AllenAtlas because we want to overwrite it
        super(ibllib.atlas.AllenAtlas, self).__init__(image, label, dxyz, regions,
        ibregma, dims2xyz=dims2xyz, xyz2dims=xyz2dims)

    def remap_atlas(self, local_file_path, regions=None, ibregma=None):
        """
        Remap the atlas label into a usable volume
        """
        file_path = Path(local_file_path)
        parent_folder = file_path.parent.absolute()
        npz_name = f'{file_path.stem}_lut_{LUT_VERSION}.npz'
        remapped_file_path = parent_folder.joinpath(npz_name)
        if not remapped_file_path.exists():
            volume = self._read_volume(file_path)
            if regions is None:
                # Encapsulating the lateralization, that should be a given
                volume = self.lateralize(volume, regions, ibregma)
            _logger.info(f"saving {remapped_file_path} ...")
            np.savez_compressed(str(remapped_file_path), volume)
        else:
            volume = np.load(str(remapped_file_path))['arr_0']
        return volume

    def lateralize(self, label, regions=None, ibregma=None):
        """
        Breaks the symmetry in regions labels in the Allen Mouse Atlas where the id
        of a region in the left hemisphere is the same as the same region in the right
        hemisphere. But if we want to map recordings to the brain, we need separate ids.
        :param label: Segmented volume
        :return: Modified volume
        """
        _logger.info("computing brain atlas annotations lookup table")
        if regions is None:
            regions = BrainRegions()
        if ibregma is None:
            ibregma = (ibllib.atlas.ALLEN_CCF_LANDMARKS_MLAPDV_UM['bregma'] / self.res_um)
        xyz2dims = np.array([1, 0, 2])  # this is the c-contiguous ordering
        # lateralize atlas: for this the regions of the left hemisphere have primary
        # keys opposite to to the normal ones
        lateral = np.zeros(label.shape[xyz2dims[0]])
        lateral[int(np.floor(ibregma[0]))] = 1
        lateral = np.sign(np.cumsum(lateral)[np.newaxis, :, np.newaxis] - 0.5)
        label = label * lateral
        _, im = ismember(label, regions.id)
        label = np.reshape(im.astype(np.uint16), label.shape)
        return label



@dataclass
class IBLAtlasModel():
    """
    IBL Atlas is a wrapper for the Allen Atlas with added features.
    The volume is also modified such that it fits functional needs.
    """
    origin: np.ndarray = IBL_BREGMA_ORIGIN
    atlas: ibllib.atlas.AllenAtlas = None
    atlas_lut: LUTModel = None
    atlas_mapping_ids: list = None
    ibl_back_end: bool = True

    IBL_TRANSPOSE = (1, 0, 2)

    atlas_volume: VolumeModel = None
    dwi_volume: VolumeModel = None

    def initialize(self, resolution=None, ibl_back_end=True):
        """
        Get Allen Atlas metadata (.csv file that is copied into ibllib.atlas) and volume files
        :param resolution: Volume resolution in microns
        """
        if resolution not in ALLEN_ATLAS_RESOLUTIONS:
            resolution = ALLEN_ATLAS_RESOLUTIONS[-1]
        if resolution == ALLEN_ATLAS_RESOLUTIONS[0]:
            print('Warning: the Allen Atlas at 10um resolution will require over 10GB of RAM')
        self.resolution = resolution
        self.use_ibl_back_end = ibl_back_end

        try:
            self.atlas = ibllib.atlas.AllenAtlas(resolution)
        except Exception as e:
            atlas_volume_url = self.get_allen_volume_url(resolution)
            atlas_volume_name = self.get_allen_volume_file_name(resolution)
            atlas_file = requests.get(atlas_volume_url)
            atlas_volume_path = './' + atlas_volume_name
            open(atlas_volume_path, 'wb').write(atlas_file.content)

            dwi_volume_url = self.get_allen_volume_url(resolution, True)
            dwi_volume_name = self.get_allen_volume_file_name(resolution, True)
            dwi_file = requests.get(dwi_volume_url)
            dwi_volume_path = './' + dwi_volume_name
            open(dwi_volume_path, 'wb').write(dwi_file.content)
            self.atlas = AllenAtlasExt(resolution, image_file_path=dwi_volume_path, label_file_path=atlas_volume_path)

        self.atlas_mapping_ids = list(self.atlas.regions.mappings.keys())
        # Further than initialize(), you will need to either call get_atlas_model() or get_dwi_model()

    def get_atlas_model(self, atlas_mapping=None):
        """
        Get a VolumeModel instance that represents the atlas (segmented) volume
        :param atlas_mapping: IBL Mapping that we want to use on this atlas.
            See ibllib.atlas.AllenAtlas.regions.mappings.keys()
        :return: VolumeModel
        """
        if self.atlas is None:
            print('No atlas!')
            return
        if self.atlas_volume is not None:
            return self.atlas_volume
        self.atlas_volume = VolumeModel(resolution=self.resolution, base_color_map=self.atlas.regions.rgb)
        self.atlas_volume.data = self.get_mapped_volume(self.atlas.label, atlas_mapping)
        self.atlas_volume.mapping = atlas_mapping
        self.atlas_volume.data_type = VolumeModel.SEGMENTED
        self.atlas_volume.name = 'Allen Mouse CCF v3 Atlas volume'
        # IBL convention
        self.atlas_volume.lateralized = atlas_mapping is not None and '-lr' in atlas_mapping
        if self.ibl_back_end:
            # The IBL back-end uses a different convention for memory representation
            # so we are forced to untranspose the volume...
            self.atlas_volume.transpose(IBLAtlasModel.IBL_TRANSPOSE)
        self.atlas_volume.build_lut(color_map=self.atlas.regions.rgb, make_active=True)
        self.atlas_volume.compute_size()
        return self.atlas_volume

    def get_dwi_model(self):
        """
        Get a VolumeModel instance of the DWI volume image
        :return: VolumeModel
        """
        if self.atlas is None:
            return
        if self.dwi_volume is not None:
            return self.dwi_volume
        self.dwi_volume = VolumeModel(resolution=self.resolution)
        self.dwi_volume.data = self.atlas.image
        self.dwi_volume.data_type = VolumeModel.RAW
        self.dwi_volume.name = 'Allen Mouse CCF v3 DWI volume'
        if self.ibl_back_end:
            self.dwi_volume.transpose(IBLAtlasModel.IBL_TRANSPOSE)
        self.dwi_volume.compute_size()
        return self.dwi_volume

    def get_num_regions(self):
        """
        Get how many regions are labelled
        """
        return self.atlas.regions.id.size

    def get_value_from_scalar_map(self, scalar):
        """
        Reverse look-up in array to find a corresponding value
        :param scalar: Scalar value
        :return: Raw volume value
        """
        scalar_map = self.atlas_volume.luts.current.scalar_map
        if scalar_map is None:
            return
        for value in range(len(scalar_map)):
            if scalar_map[value] is None:
                continue
            #print(scalar - scalar_map[value])
            if scalar_map[value] == scalar:
                return value

    def get_mapped_data(self, value):
        """
        Given a value from the segmented volume, we retrieve useful info
        :param value: Value from the volume
        :return: Dictionary of corresponding data
        """
        if value is None:
            return
        value = int(value)
        region_id = self.atlas.regions.id[value]
        region_data = self.atlas.regions.get(region_id)

        data_lut = self.atlas_volume.luts.current
        region_name = region_data.name[0].title()
        color = data_lut.color_map[value][1]
        alpha = 1.0
        if data_lut.alpha_map is not None:
            alpha = data_lut.alpha_map[value][1]
        scalar = None
        scalar_map = data_lut.scalar_map
        if scalar_map is not None:
            if isinstance(scalar_map, dict):
                scalar = scalar_map.get(value)
            else:
                scalar = scalar_map[value]
        return {'scalar':scalar, 'region_id':region_id, 'color':color, 'alpha':alpha,
                'region_name':region_name, 'region_data':region_data}

    def remap(self, ids, source='Allen', dest='Beryl'):
        """
        Remap ids/scalar values from source to destination
        Function by Olivier Winter
        :param ids: List of ids
        :param source: Source mapping
        :param dest: Destination mapping
        """
        #from ibllib.atlas import BrainRegions as br
        _, inds = ismember(ids, self.atlas.regions.mappings[source])
        return self.atlas.regions.mappings[dest][inds]

    def get_allen_volume_file_name(self, resolution, raw_image=False):
        """
        Get the Allen volume file name given its resolution
        :param resolution: Resolution of the volume
        :param raw_image: Whether we want the raw volume file name or the segmented one
        :return: String
        """
        file_name = ALLEN_ATLASES['dwi_prefix'] if raw_image else ALLEN_ATLASES['atlas_prefix']
        file_name += str(resolution) + ALLEN_ATLASES['volume_extension']
        return file_name

    def get_allen_volume_url(self, resolution, raw_image=False):
        """
        Construct a URL with which we can download data sets
        :param resolution: Volume resolution, either 10, 25, 50 or 100 (um)
        :param raw_image: Whether the volume is the segmented one (aka the atlas) or the DWI
        :return: String
        """
        url = ALLEN_ATLASES['base_url'] + ALLEN_ATLASES['mouse_ccf_folder']
        url += ALLEN_ATLASES['dwi_folder'] if raw_image else ALLEN_ATLASES['atlas_folder']
        url += self.get_allen_volume_file_name(resolution, raw_image)
        return url

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
        return super().load_volume(file_path, remap_scalars, self.atlas.regions.id, make_current)

    def get_name(self, *args):
        """
        Get full name for a model, separated by underscores
        :return: String
        """
        return '_'.join(args)

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
                    #logging.error('[AtlasModel.get_mapped_volume()] could not find atlas mapping with id ' + str(atlas_mapping) + '. Returning raw volume...')
                    return volume
                map_id = self.atlas_mapping_ids[atlas_mapping]
            elif atlas_mapping is None:
                map_id = self.atlas_mapping_ids[0]
            else:
                map_id = atlas_mapping

            # This mapping actually changes the order of the axes in the volume...
            volume_data = self.atlas.regions.mappings[map_id][volume]
            #logging.info('Loaded atlas volume with ' + map_id + ' mapping')
        else:
            volume_data = volume
        return volume_data

    def get_region_and_row_id(self, acronym):
        """
        Get region and row id given an acronym
        :param acronym: Acronym of a brain region
        :return: Region id and row id
        """
        ind = np.where(self.atlas.regions.acronym == acronym)[0]
        if ind.size < 1:
            return None, None
        return self.atlas.regions.id[ind], ind

    def get_regions_mask(self, region_ids, alpha_map=None):
        """
        Build an alpha map that reveals only the given region ids
        :param region_ids: List or numpy array of region ids
        :param alpha_map: Optional alpha map that will be modified. If None provided, the method will attempt
            to use the current active alpha map
        :return: 2D numpy array with region ids and corresponding alpha values
        """
        if alpha_map is None:
            alpha_map = self.lut.alpha_map
        if alpha_map is None:
            #logging.error('[Method build_regions_alpha_map()] requires that an alpha map is created by build_regions_alpha_map()')
            return
        new_alpha_map = np.zeros_like(alpha_map).astype(float)
        new_alpha_map[:, 0] = alpha_map[:, 0]
        new_alpha_map[region_ids, 1] = alpha_map[region_ids, 1]
        self.lut.sec_alpha_map = new_alpha_map
        return new_alpha_map


class MouseBrainViewer(Viewer):
    """
    This is your entry point to International Brain Laboratory data visualization
    """

    def __init__(self):
        """
        Constructor
        """
        super().__init__()
        self.bounding_mesh = None
        self.atlas_controller = None
        self.dwi_controller = None
        self.ibl_model = None
        # Shortcut for users
        self.ibl_transpose = IBLAtlasModel.IBL_TRANSPOSE

    def initialize(self, resolution=25, mapping='Beryl', add_atlas=True, add_dwi=False,
                    dwi_color_map='viridis', dwi_alpha_map=None, local_allen_volumes_path=None,
                    offscreen=False, jupyter=False, embed_ui=False, embed_font_size=15,
                    plot=None, plot_window_id=0, num_windows=1, render=False, dark_mode=False):
        """
        Initialize the controller, main entry point to the viewer
        :param resolution: Resolution of the atlas volume.
            Possible values are 10 (requires a lot of RAM), 25, 50, 100. Units are in microns
        :param mapping: Optional mapping value. In the context of IBL, there is 'Allen' for the standard Allen map
            and 'Beryl' (random name) which aggregates cortical layers as one.
        :param add_atlas: Whether the Atlas is included in the viewer
        :param add_dwi: Whether the diffusion weighted imaging is loaded in the viewer (same boundaries as atlas)
        :param context: Context of the visualization
        :param embed_ui: Whether the UI is embed within the VTK window
        :param embed_font_size: Embed font size. Defaults to 16 points. You might need larger values
            in case you have a small screen with high dpi (but VTK methods fail to detect that).
        :param jupyter: Whether we're running from a jupyter notebook or not
        :param plot: A vedo Plotter instance. You can either create it by yourself before hand, in case you want
            to have multiple windows with other stats or let the controller create a new one
        :param plot_window_id: Sub-window id where the 3D visualization will be displayed
        :param num_windows: Number of subwindows, in case you want to display your own stuff later
        :param dark_mode: Whether the viewer is in dark mode
        """
        self.model.title = 'IBL Viewer'

        # ibllib works with two volumes at the same time: the segmented volume (called 'label')
        # and the DWI volume (called 'image')
        self.ibl_model = IBLAtlasModel()
        self.ibl_model.initialize(resolution, local_allen_volumes_path)
        self.model.origin = self.ibl_model.origin

        # Neuroscience specific
        self.model.probe_initial_point1 = self.model.origin
        self.model.probe_initial_point2 = self.model.origin + [0, 0, 8000]

        super().initialize(offscreen, jupyter, embed_ui, embed_font_size,
                            plot, plot_window_id, num_windows, dark_mode)

        # A VolumeController has a unique volume as target so if we want to visualize both volumes, we create two views
        if add_atlas:
            self.ibl_model.get_atlas_model(mapping)
            self.add_atlas_segmentation()
        if add_dwi:
            self.ibl_model.get_dwi_model()
            if dwi_alpha_map is None:
                #dwi_alpha_map = np.ones(516)
                #dwi_alpha_map[:140] = 0
                dwi_alpha_map = [0.0, 1.0, 1.0]
            self.add_atlas_dwi(dwi_color_map, dwi_alpha_map)

        try:
            self.load_bounding_mesh()
        except Exception:
            pass

        #light = vedo.Light(self.model.IBL_BREGMA_ORIGIN - [0, 0, 1000], c='w', intensity=0.2)
        #self.plot.add(light)

        # By default, the atlas volume is our target
        if self.atlas_controller is not None:
            self.select(self.atlas_controller.actor)
        elif self.dwi_controller is not None:
            self.select(self.dwi_controller.actor)

        if self.model.selection is not None:
            #self.add_origin()
            self.handle_lut_update()
            self.set_left_view()

    def add_atlas_segmentation(self):
        """
        Add the Allen Atlas segmented volume (aka label)
        """
        if isinstance(self.atlas_controller, VolumeController):
            return
        self.atlas_controller = VolumeController(self.plot, self.ibl_model.atlas_volume,
                                                alpha_unit_upper_offset=0.1, center_on_edges=True)
        self.register_controller(self.atlas_controller, self.atlas_controller.get_related_actors())

    def add_atlas_dwi(self, color_map, alpha_map):
        """
        Add the Allen Atlas diffusion weighted image
        :param color_map: Color map for the volume
        :param alpha_map: Alpha map for the volume
        """
        if isinstance(self.dwi_controller, VolumeController):
            return
        self.dwi_controller = VolumeController(self.plot, self.ibl_model.dwi_volume,
                                                center_on_edges=True)
        self.dwi_controller.set_color_map(color_map, alpha_map)
        self.register_controller(self.dwi_controller, self.dwi_controller.get_related_actors())

    def load_bounding_mesh(self, add_to_scene=False, alpha_on_scene=0.3):
        """
        Load the bounding mesh of the mouse brain that represents its approximate pial limit
        """
        self.bounding_mesh = utils.load_surface_mesh('997')
        if add_to_scene:
            self.bounding_mesh.alpha(alpha_on_scene)
            self.plot.add(self.bounding_mesh)

        # An offset is applied to the volume in build_actor, so we have to apply it here
        self.bounding_mesh.pos(np.array([-100+self.ibl_model.resolution, 0.0, 0.0]))
        #self.bounding_mesh.mapper().SetClippingPlanes(self.clipping_planes)

    def find_region(self, term):
        """
        Find a region with a substring
        :param term: Search term
        :return: List of matching entries and the corresponding mask
        """
        # ibl_model.atlas.regions is a BrainRegion object that puts a pandas dataframe
        # into separate numpy arrays (like 'name') so we work on a numpy array here
        mask = np.flatnonzero(np.char.find(self.ibl_model.atlas.regions.name.astype(str), term) != -1)
        return mask

    def get_region_names(self):
        """
        Get the region names
        :return: List
        """
        return self.ibl_model.atlas.regions.name.tolist()

    def _select(self, actor=None, controller=None, event=None,
                    camera_position=None, position=None, value=None):
        """
        Define the current object selected
        :param actor: a vtkActor
        :param controller: Controller of the given actor (optional)
        :param event: a vedo event from which we use picked3d and picked2d (we could directly use vtk)
        :param camera_position: position of the camera (optional) at selection time
        :param position: The final position computed on the volume or mesh or point or line.
            If not given, this will be automatically calculated
        .param value: The value corresponding to the point on the object. If not given, this will
            be automatically retrieved
        """
        super()._select(actor, controller, event, camera_position, position, value)
        extra_data = self.ibl_model.get_mapped_data(self.model.selection_value)
        if extra_data is None:
            return
        # This is where we retrieve neuroscience specific data about our selection
        selection = self.model.selection
        if selection == self.atlas_controller.actor or self.is_probe(selection):
            self.model.selection_related_name = extra_data.get('region_name')
            self.model.selection_related_value = extra_data.get('scalar')

    def get_selection_info(self, line_length=40, precision=5):
        """
        Get information about the current selection
        :param line_length: Region name line length after what it's word-wrapped
        :param precision: Scalar value floating precision displayed
        :return: Preformatted multiline text and a dictionary of extra data
        """
        text, data = super().get_selection_info()
        if self.atlas_controller is None:
            return text, data
        region_name = self.model.selection_related_name
        scalar = self.model.selection_related_value
        if region_name is not None:
            if isinstance(line_length, int):
                lines = textwrap.wrap(region_name, line_length, break_long_words=True)
                region_name = '\n'.join(lines)
            text += f'\nRegion: {region_name}'
        if scalar:
            text += f'\n\nScalar value: {scalar:.{precision}f}'
        return text, data

    def add_origin(self):
        """
        Add the origin on scene
        """
        self.atlas_origin = utils.Cross3DExt(self.model.origin, thickness=2, size=500)
        self.atlas_origin.lighting('off')
        self.plot.add(self.atlas_origin)
        #text_pos = self.atlas_origin.pos()+[0, 100, 0]
        #font_size = self.model.ui.font_size * 10
        #self.atlas_origin_label = vedo.Text3D('Bregma origin', pos=text_pos, s=font_size, c='k')
        #self.atlas_origin_label.followCamera()
        #self.plot.add([self.atlas_origin, self.atlas_origin_label])

    def add_many_points_test(self, positions, point_radius=2, auto_xy_rotate=True, add_to_scene=True):
        """
        Test method that validates that VTK is fast enough for displaying 10 million points interactively (and it is :)
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
        self.register_object(point_cloud)
        if add_to_scene:
            self.plot.add(point_cloud)
        return point_cloud

    def add_spheres(self, positions, radius=10, values=None, color_map='Accent', name='Spheres',
                    use_origin=True, add_to_scene=True, noise_amount=0, trim_outliers=True,
                    bounding_mesh=None, ibl_flip_yz=True, **kwargs):
        """
        Add new spheres
        :param positions: 3D array of coordinates
        :param radius: List same length as positions of radii. The default size is 5um, or 5 pixels
            in case as_spheres is False.
        :param values: 1D array of values, one per neuron or a time series of such 1D arrays (numpy format)
        :param color_map: A color map, it can be a color map built by IBLViewer or
            a color map name (see vedo documentation), or a list of values, etc.
        :param name: All point neurons are grouped into one object, you can give it a custom name
        :param use_origin: Whether the origin is added as offset to the given positions
        :param add_to_scene: Whether the new lines are added to scene/plot and rendered
        :param noise_amount: Amount of 3D random noise applied to each point. Defaults to 0
        :param trim_outliers: If bounding_mesh param is given, then the spheres will be trimmed,
            only the ones inside the bounding mesh will be kept
        :param bounding_mesh: A closed manifold surface mesh used for trimming segments. If None,
            the current self.bounding_mesh is used (if it exists)
        :param ibl_flip_yz: If you have an IBL data set, its 3D coordinates will be multiplied by -1
            on Y and Z axes in order to match Allen Brain Atlas volume and how it's stored by IBL.
        :return: objects.Points
        """
        axes = [1, 1, 1]
        if ibl_flip_yz:
            axes = [1, -1, -1]
            positions = np.array(positions) * [axes]
        if noise_amount is not None:
            positions += np.random.rand(len(positions), 3) * noise_amount
        link = True if add_to_scene and not trim_outliers else False
        spheres = super().add_spheres(positions, radius, values, color_map,
                                    name, use_origin, link, **kwargs)
        spheres.axes = axes
        if bounding_mesh is None:
            bounding_mesh = self.bounding_mesh
        if trim_outliers and bounding_mesh is not None:
            spheres.cutWithMesh(bounding_mesh)
            spheres.mapper().SetScalarVisibility(True)
            if add_to_scene:
                self.plot.add(spheres)
        return spheres

    def add_points(self, positions, radius=10, values=None, color_map='Accent', name='Points', screen_space=False,
                    use_origin=True, add_to_scene=True, noise_amount=0, trim_outliers=True, bounding_mesh=None,
                    ibl_flip_yz=True, **kwargs):
        """
        Add new points
        :param positions: 3D array of coordinates
        :param radius: List same length as positions of radii. The default size is 5um, or 5 pixels
            in case as_spheres is False.
        :param values: 1D array of values, one per neuron or a time series of such 1D arrays (numpy format)
        :param color_map: A color map, it can be a color map built by IBLViewer or
            a color map name (see vedo documentation), or a list of values, etc.
        :param name: All point neurons are grouped into one object, you can give it a custom name
        :param screen_space: Type of point, if True then the points are static screen-space points.
            If False, then the points are made to scale in 3D, ie you see them larger when you
            zoom closer to them, while this is not the case with screen-space points. Defaults to False.
        :param use_origin: Whether the origin is added as offset to the given positions
        :param add_to_scene: Whether the new lines are added to scene/plot and rendered
        :param noise_amount: Amount of 3D random noise applied to each point. Defaults to 0
        :param trim_outliers: If bounding_mesh param is given, then the spheres will be trimmed,
            only the ones inside the bounding mesh will be kept
        :param bounding_mesh: A closed manifold surface mesh used for trimming segments. If None,
            the current self.bounding_mesh is used (if it exists)
        :param ibl_flip_yz: If you have an IBL data set, its 3D coordinates will be multiplied by -1
            on Y and Z axes in order to match Allen Brain Atlas volume and how it's stored by IBL.
        :return: objects.Points
        """
        axes = [1, 1, 1]
        if ibl_flip_yz:
            axes = [1, -1, -1]
            positions = np.array(positions) * [axes]
        if noise_amount is not None:
            positions += np.random.rand(len(positions), 3) * noise_amount
        link = True if add_to_scene and not trim_outliers else False
        points = super().add_points(positions, radius, values, color_map, name,
                                    screen_space, use_origin, link, **kwargs)
        points.axes = axes
        if bounding_mesh is None:
            bounding_mesh = self.bounding_mesh
        if trim_outliers and bounding_mesh is not None:
            points.cutWithMesh(bounding_mesh)
            points.mapper().SetScalarVisibility(True)
            if add_to_scene:
                self.plot.add(points)
        return points

    def add_segments(self, points, end_points=None, line_width=2, values=None, color_map='Accent',
                    name='Segments', use_origin=True, add_to_scene=True, relative_end_points=False,
                    spherical_angles=None, radians=True, trim_outliers=True, bounding_mesh=None,
                    ibl_flip_yz=True):
        """
        Add a set of segments
        :param points: 3D numpy array of points of length n
        :param end_points: 3D numpy array of points of length n
        :param line_width: Line width, defaults to 2px
        :param values: 1D list of length n, for one scalar value per line
        :param color_map: A color map, it can be a color map built by IBLViewer or
            a color map name (see vedo documentation), or a list of values, etc.
        :param name: Name to give to the object
        :param use_origin: Whether the current origin (not necessarily absolute 0) is used as offset
        :param add_to_scene: Whether the new lines are added to scene/plot and rendered
        :param relative_end_points: Whether the given end point is relative to the start point. False by default,
            except is spherical coordinates are given
        :param spherical_angles: 3D numpy array of spherical angle data of length n
            In case end_points is None, this replaces end_points by finding the relative
            coordinate to each start point with the given radius/depth, theta and phi
        :param radians: Whether the given spherical angle data is in radians or in degrees
        :param trim_outliers: Whether segments are cropped by the bounding mesh
        :param bounding_mesh: A closed manifold surface mesh used for trimming segments. If None,
            the current self.bounding_mesh is used (if it exists)
        :param ibl_flip_yz: If you have an IBL data set, its 3D coordinates will be multiplied by -1
            on Y and Z axes in order to match Allen Brain Atlas volume and how it's stored by IBL.
        :return: objects.Lines
        """
        axes = [1, 1, 1]
        if ibl_flip_yz:
            axes = [1, -1, -1]
            points = np.array(points) * axes
            if end_points is not None:
                end_points = np.array(end_points) * axes
        pre_add = True if add_to_scene and not trim_outliers else False

        #lines = super().add_segments(points, end_points, line_width, values, color_map, name, use_origin,
                                    #pre_add, relative_end_points, spherical_angles, radians)
        '''
        Crazy python stuff here [WARNING]
        If the above line with super() is called, then the scope of self within super() is this one,
        which is wrong. When we are in super, self should be the parent class.
        '''
        # Copy-paste from application.Viewer.add_segments, due to above reason
        if end_points is None and spherical_angles is not None:
            relative_end_points = True
            spherical_angles = np.array(spherical_angles)
            if radians:
                end_points = spherical_angles.apply(vedo.spher2cart)
            else:
                end_points = spherical_angles.apply(utils.spherical_degree_angles_to_xyz)
            if relative_end_points:
                end_points += points
            points = np.c_[points, end_points].reshape(-1, 2, 3)
        elif end_points is not None and len(points) != len(end_points):
            n = min(len(points), len(end_points))
            logging.error(f'[add_segments() error] Mismatch between start and end points length. Only {n} segments shown.')
            points = np.c_[points[n], end_points[n]].reshape(-1, 2, 3)
        lines = super().add_lines(points, line_width, values, color_map, name, use_origin, pre_add)

        lines.axes = axes
        if bounding_mesh is None:
            bounding_mesh = self.bounding_mesh
        if trim_outliers and bounding_mesh is not None:
            lines.cutWithMesh(bounding_mesh)
            lines.mapper().SetScalarVisibility(True)
            if add_to_scene:
                self.plot.add(lines)
        return lines

    def add_lines(self, points, line_width=2, values=None, color_map='Accent', name='Lines',
                use_origin=True, add_to_scene=True, trim_outliers=True, bounding_mesh=None, ibl_flip_yz=True):
        """
        Create a set of lines with given point sets
        :param points: List of lists of 3D coordinates
        :param line_width: Line width, defaults to 2px
        :param values: 1D list of length n, for one scalar value per line
        :param color_map: A color map, it can be a color map built by IBLViewer or
            a color map name (see vedo documentation), or a list of values, etc.
        :param name: Name to give to the object
        :param use_origin: Whether the current origin (not necessarily absolute 0) is used as offset
        :param add_to_scene: Whether the new lines are added to scene/plot and rendered
        :param trim_outliers: Whether segments that are out of the brain envelope are trimmed or not. True by default
        :param bounding_mesh: A closed manifold surface mesh used for trimming lines. If None,
            the current self.bounding_mesh is used (if it exists)
        :param ibl_flip_yz: If you have an IBL data set, its 3D coordinates will be multiplied by -1
            on Y and Z axes in order to match Allen Brain Atlas volume and how it's stored by IBL.
        :return: objects.Lines
        """
        axes = [1, 1, 1]
        if ibl_flip_yz:
            axes = [1, -1, -1]
        #target =  list(points.keys()) if isinstance(points, dict) else range(len(points))
        if not isinstance(points, np.ndarray):
            '''
            This part below is to handle the numpy error that does not allow a numpy array
            to contain lists with unequal lengths.
            -> VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences 
            (which is a list-or-tuple of lists-or-tuples-or ndarrays with different 
            lengths or shapes) is deprecated
            '''
            all_points = []
            indices = []
            line_id = 0
            # Possible speed improvement: use map or np.apply_along_axis
            for index in range(len(points)):
                point_set = points[index]
                point_set = np.array(point_set).astype(float)
                point_set = point_set * [axes]
                if use_origin:
                    point_set = point_set + self.model.origin
                all_points.append(point_set)
                indices.append(line_id)
                line_id += 1
            points = all_points
            if values is None:
                values = indices

        pre_add = True if add_to_scene and not trim_outliers else False
        lines = super().add_lines(points, line_width, values, color_map, name, False, pre_add)

        if bounding_mesh is None:
            bounding_mesh = self.bounding_mesh
        if trim_outliers and bounding_mesh is not None:
            lines.cutWithMesh(bounding_mesh)
            # This is the sort of thing that kills me. I was loosing all scalar colors
            # and it took me a while to see it's because of vedo's cutWithMesh that disables
            # scalar visibility!
            lines.mapper().SetScalarVisibility(True)
            if add_to_scene:
                self.plot.add(lines)
        return lines

    def add_volume(self, data=None, resolution=None, file_path=None, color_map='viridis',
                    alpha_map=None, select=False, add_to_scene=True, transpose=None):
        """
        Add a volume to the viewer with box clipping and slicing enabled by default
        :param data: Volume image data or a file_path
        :param resolution: Resoluton of the volume
        :param file_path: File path of the volume. If you don't provide an image volume data,
            then the file_path will be used to load the volume data
        :param color_map: Color map for the volume
        :param alpha_map: Alpha map for the volume. If None, it will assume that 0 values
            are transparent and maximum values are opaque
        :param select: Whether the volume is selected
        :param add_to_scene: Whether the volume is added to scene
        :param transpose: Transposition parameter. If None. nothing happens. If True,
            then the default IBL transposition is applied. You can provide your own, that is,
            a list of 3 elements to reorder the volume as desired.
        :return: VolumeController
        """
        if transpose == True:
            transpose = self.ibl_transpose
        return super().add_volume(data, resolution, file_path, color_map,
                                alpha_map, select, add_to_scene, transpose)

    def set_left_view(self):
        """
        Set left sagittal view
        """
        self.update_camera([1.0, 0.0, 0.0], self.model.Z_DOWN)

    def set_right_view(self):
        """
        Set right sagittal view
        """
        self.update_camera([-1.0, 0.0, 0.0], self.model.Z_DOWN)

    def set_anterior_view(self):
        """
        Set anterior coronal view
        """
        self.update_camera([0.0, 1.0, 0.0], self.model.Z_DOWN)

    def set_posterior_view(self):
        """
        Set posterior coronal view
        """
        self.update_camera([0.0, -1.0, 0.0], self.model.Z_DOWN)

    def set_dorsal_view(self):
        """
        Set dorsal axial view
        """
        self.update_camera([0.0, 0.0, 1.0], self.model.X_UP)

    def set_ventral_view(self):
        """
        Set ventral axial view
        """
        self.update_camera([0.0, 0.0, -1.0], self.model.X_UP)
