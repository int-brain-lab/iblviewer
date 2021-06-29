import numpy as np
import pandas as pd
import pickle
import random
import vedo
import argparse

from iblviewer.mouse_brain import MouseBrainViewer
from iblviewer import utils

#pLeft_iti_scores_n_gt_50
# Data given by Berk Gerçek, International Brain Laboratory
def process_df(viewer, file_path=None, aggregator='median', grouper='acronym'):
    """
    Process priors data and get color map and scalar values
    """
    if file_path is None:
        test_data = './stimonR_top10_rawpoints.p'
        file_path = str(utils.EXAMPLES_DATA_FOLDER.joinpath(test_data))
    
    df = pickle.load(open(file_path, 'rb'))
    
    # For testing data given by Guido
    # df['r_over_chance'] = df['r_prior'] - df['r_prior_null']
    # filtered_df = df.groupby('region').median()['r_over_chance']

    raw_df = df['rawpoints']
    filtered_df = raw_df.groupby(grouper).agg({'value': aggregator})
    filtered_df.dropna(inplace=True)
    min_value = float(np.amin(filtered_df, axis=0).to_numpy()[0])
    max_value = float(np.amax(filtered_df, axis=0).to_numpy()[0])
    print('Min prior value ' + str(min_value))
    print('Max prior value ' + str(max_value))
    
    #return filtered_df, min_value, max_value

    scalars_map = [None]*viewer.ibl_model.get_num_regions()

    # This code is to be modified if you have split data for left and right hemispheres
    # The concept is pretty simple: scalars_map is a 1D list that maps to brain regions.
    # With the lateralized brain mapping, the standard region id in Allen CCF is negated
    # on the right hemisphere. 
    # Currently this code uses standard acronym lookup, which yields a region on both
    # hemispheres. The value you assign to an acronym will thus be mirrored.

    # Or for i in range(0, len(df)): which preserves data types
    for acronym, row in filtered_df.iterrows():
        value = row.iloc[0]
        if value is None:
            continue
        region_ids, row_ids = viewer.ibl_model.get_region_and_row_id(acronym)
        if region_ids is None:
            print('Acronym', acronym, 'was not found in Atlas')
            continue
        for r_id in range(len(region_ids)):
            region_id = region_ids[r_id]
            row_id = row_ids[r_id]
            if region_id is None:
                print('Error, could not find acronym', acronym, '...ignoring it)')
                continue
            if row_id == 0: #or value.isnull().values.any():
                # We ignore void acronym (which is equal to row_id 0) on Allen Mouse CCF v3
                continue
            #print('Setting row', int(row_id), 'with value', value)
            scalars_map[int(row_id)] = value
    return scalars_map, [min_value, max_value]
    
def load_priors(viewer, file_path=None, aggregator='median', 
                color_map_func='viridis', nan_color=[0.0, 0.0, 0.0, 0.0]):
    """
    Load priors into the viewer
    """
    scalars_map, scalar_range = process_df(viewer, file_path=file_path, aggregator=aggregator)
    # Ensure we have the right selection, here 0 is the first registered object: the atlas
    viewer.select(-1)
    viewer.assign_scalars(scalars_map, scalar_range, color_map_func)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data mapping on the Allen CCF volume atlas')
    parser.add_argument('-r', dest='resolution', type=int, default=25, 
    help='Volume resolution. Possible values: 100, 50, 25, and 10. Units are in microns. The 10um volume takes a lot of RAM (and some time to load)')
    parser.add_argument('-m', dest='mapping', type=str, default='Allen', 
    help='Volume mapping name. Either Allen (default value) or Beryl (IBL specific simplified mapping).')
    parser.add_argument('-f', dest='file_path', type=str, default=None, 
    help='File path to your data. None will load a default example.')
    parser.add_argument('-a', dest='aggregator', type=str, default='median', 
    help='Aggregator type. Defaults to median.')
    parser.add_argument('-cm', dest='color_map', type=str, default='viridis', 
    help='Color map for the given values')
    parser.add_argument('-nc', dest='nan_color', type=float, default=0.65, 
    help='Gray color (between 0 and 1) for regions that have no assigned value')
    parser.add_argument('-na', dest='nan_alpha', type=float, default=0.5, 
    help='Alpha (opacity) value for regions that have no assigned value')

    args = parser.parse_args()
    viewer = MouseBrainViewer()
    viewer.initialize(resolution=args.resolution, mapping=args.mapping, embed_ui=True, jupyter=False, dark_mode=False)
    load_priors(viewer, 
                file_path=args.file_path, 
                aggregator=args.aggregator,
                color_map_func=args.color_map,
                nan_color=[args.nan_color]*3+[args.nan_alpha])
    viewer.show().close()