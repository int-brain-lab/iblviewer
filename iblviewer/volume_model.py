from dataclasses import dataclass, field
from typing import Mapping, List, Any
from datetime import datetime
import numpy as np
import os
import logging

import nrrd
import vedo

import iblviewer.utils as utils

@dataclass
class VolumeModel:
    DWI = 'dwi'
    SEGMENTED = 'segmented'
    NORMALIZED_VOLUME_SUFFIX = '_norm'
    VOLUME_TYPES = {DWI:0, SEGMENTED:1}
    VOLUME_PREFIX = '[Volume]'

    name: str = VOLUME_PREFIX
    file_path: str = None
    # In early 2021, either 'Beryl', a made-up name for grouping 
    # some cortical layers and the default 'Atlas'. 
    # See ibllib.atlas.regions.mappings
    mapping_name: str = None
    lateralized: bool = False
    # Mapping function. If None, the volume will be given as it is.
    mapping: Any = 'Allen'
    resolution: int = 25
    volume: np.ndarray = None
    volume_type: str = None
    interactive_subsampling: bool = True
    dimensions: np.ndarray = np.zeros(3).astype(np.float64)
    center: np.ndarray = np.zeros(3).astype(np.float64)

    def is_segmented_volume(self):
        """
        Get whether current volume is segmented or rather a weighted diffusion imaging
        """
        return self.volume_type == VolumeModel.VOLUME_TYPES.get(VolumeModel.SEGMENTED)

    def import_volume(self, file_path):
        """
        Import volume routine
        :param file_path: Volume path
        """
        if "https://" in file_path:
            downloaded_temp_file_path = vedo.download(file_path, verbose=False)
            if file_path.endswith('nrrd'):
                volume, header = nrrd.read(downloaded_temp_file_path)
            else:
                volume = vedo.loadImageData(downloaded_temp_file_path)
        else:
            if file_path.endswith('nrrd'):
                volume, header = nrrd.read(file_path, index_order='C')
            else:
                volume = vedo.loadImageData(file_path)
        return volume

    def load_volume(self, file_path, reassign_scalars=False, set_as_current_volume=True):
        """
        Load a custom volume data. Only useful if you use volume data that does not come from IBL back-end.
        :param file_path: Volume file path. Could support other file types easily.
        :param reassign_scalars: Whether scalar values in the volume are replaced by their row id from a mapping that stores
        unique scalar values. Useful only for segmented volumes.
        """
        if not reassign_scalars:
            return self.import_volume(file_path)

        time = datetime.now()

        mod_file_path = utils.change_file_name(file_path, None, None, VolumeModel.NORMALIZED_VOLUME_SUFFIX)
        volume = None
        if os.path.exists(mod_file_path):
            volume = self.import_volume(mod_file_path)
        else:
            volume = self.import_volume(file_path)
            volume = self.change_labels(volume, mod_file_path)
            logging.info('Reassigned scalar values in volume: ' + str(utils.time_diff(time)) + 's')
        
        if volume is not None:
            logging.info('Opened atlas ' + mod_file_path + ' in ' + str(utils.time_diff(time)) + 's')
            min_volume_value, max_volume_value = np.amin(volume), np.amax(volume)
            logging.info('Min max scalar values in volume ' + str(min_volume_value) + ' -> ' + str(max_volume_value))
        else:
            logging.error('Failed to open atlas ' + mod_file_path)

        if set_as_current_volume:
            self.volume = volume
        return volume

    def change_labels(self, volume, df_column_map, write_path=None):
        """
        Reassign scalar values to something that makes more sense.
        There are two issues we fix here:

        # 1 Weird scalar value range
        Scalar values in original annotation_xxx.nrrd (where xxx is the resolution) are
        not set in a clever way. Brain atlas regions (over a thousand) feature indices from 0 to
        ... 607344834! Applying a transfer function interactively where most values are useless 
        is at best suboptimal, not to mention difficulties in plotting scalars
        So we rewrite Allen Atlas region ids to row ids (from 0 to 1000+) because these should
        have been the actual unique ids. This is stored to disk so that we don't recompute
        this all the time.

        # 2 Hemisphere symmetry
        All regions are labelled with the same labels on both hemispheres, which is a problem
        if we want to assign different values to, say, the left hippocampus CA1 vs right hippo CA1.
        
        :param volume: Volume ndarray
        :param write_path: Where the modified volume will be stored (to spare going through this method next time)
        :param df_column_map: Pandas DataFrame column that maps to the scalars for each unique label of the segmented volume. 
        :return: Modified volume data
        """
        logging.info('\nBuilding appropriate volume from Allen data source...')
        #volume = np.vectorize(self.f)(volume)
        labels = np.unique(volume)
        num_labels = len(labels)
        logging.info('Num regions labeled in volume ' + str(num_labels) + ' from ' + str(df_column_map.size) + ' in atlas')
        logging.info('Reassigning ' + str(num_labels) + ' scalar values...')
        # On a large volume, this can take a long time
        for iter_id in range(num_labels):
            label = labels[iter_id]
            row_id = df_column_map.index[df_column_map == label].to_list()[0]
            volume[volume == label] = row_id
            if num_labels > 10000 and iter_id % 10 == 0:
                logging.info('  Progress: ' + str(int(iter_id/num_labels)*100) + '%')
        
        if write_path is not None:
            logging.info('Saving volume data under ' + write_path)
            nrrd.write(write_path, volume, index_order='C')
        return volume