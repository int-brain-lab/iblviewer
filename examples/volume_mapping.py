import numpy as np
import pandas as pd
import random
import vedo
import argparse
from iblviewer import atlas_controller
from iblviewer import utils


# Data given by Berk Gerçek, International Brain Laboratory
def process_df(controller, file_path=None, aggregator='median', grouper='acronym'):
    """
    Process priors data and get color map and scalar values
    """
    if file_path is None:
        test_data = './stimonR_top10_rawpoints.p'
        file_path = str(utils.EXAMPLES_DATA_FOLDER.joinpath(test_data))
    
    df = np.load(file_path, allow_pickle=True)
    
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
    
    scalars_map = {}

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
        region_ids, row_ids = controller.model.get_region_and_row_id(acronym)
        if region_ids is None:
            print('Acronym', acronym, 'was not found in Atlas')
            continue
        for r_id in range(len(region_ids)):
            region_id = region_ids[r_id]
            row_id = row_ids[r_id]
            if region_id is None:
                print('Error, could not find acronym (ignoring it)', acronym)
                continue
            if row_id == 0: #or value.isnull().values.any():
                # We ignore void acronym and nan values
                continue
            scalars_map[int(row_id)] = value
    return scalars_map
    
def get_color_map(controller, scalar_map, color_map_func='viridis', 
                    nan_color=[0.0, 0.0, 0.0], nan_alpha=0.0, seed=None):
    """
    Generate a color map with the scalar map.
    Simply put, we assign colors to the values.
    """
    if seed is not None:
        random.seed(seed)
    rgb = []
    alpha = []
    for r_id in range(controller.model.atlas.regions.id.size):
        rand_val = np.random.uniform(0, 0.35)
        rgb.append([r_id, np.array([rand_val]*3) + nan_color])
        a = nan_alpha if r_id > 0 else 0.0
        alpha.append([r_id, a])
    
    values = sorted(scalar_map.values())

    min_p = min(values)
    max_p = max(values)
    rng_p = max_p - min_p
    #cmap = vedo.colorMap(values, cmap_name, min_p, max_p)
    for row_id in scalar_map:
        value = scalar_map[row_id]
        if seed is not None and seed > 0:
            value = value + random.random() * rng_p / 2
        #rgb[row_id] = [row_id, list(vedo.colorMap(value, cmap_name, min_p, max_p))]
        if isinstance(color_map_func, str):
            rgb[row_id][1] = list(vedo.colorMap(value, color_map_func, min_p, max_p))
        else:
            # Here we assume you provided a function that is called with these values
            rgb[row_id][1] = color_map_func(value, min_p, max_p)

        alpha[row_id] = [row_id, 1.0]
    return rgb, alpha
    
def load_priors(controller, file_path=None, aggregator='median', 
                color_map_func='viridis', nan_color=[0.0, 0.0, 0.0], nan_alpha=0.0):
    """
    Load priors into the viewer, faking a time series from there
    """
    scalar_map = process_df(controller, file_path=file_path, aggregator=aggregator)
    rgb_map, alpha_map = get_color_map(controller, scalar_map, color_map_func, nan_color, nan_alpha)
    controller.add_transfer_function(scalar_map, rgb_map, alpha_map, color_map_func, make_current=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data mapping on the Allen CCF volume atlas')
    parser.add_argument('--r', dest='resolution', type=int, default=25, 
    help='Volume resolution. Possible values: 100, 50, 25, and 10. Units are in microns. The 10um volume takes a lot of RAM (and some time to load)')
    parser.add_argument('--m', dest='mapping', type=str, default='Allen', 
    help='Volume mapping name. Either Allen (default value) or Beryl (IBL specific simplified mapping).')
    parser.add_argument('--f', dest='file_path', type=str, default=None, 
    help='File path to your data. None will load a default example.')
    parser.add_argument('--a', dest='aggregator', type=str, default='median', 
    help='Aggregator type. Defaults to median.')
    parser.add_argument('--cm', dest='color_map', type=str, default='viridis', 
    help='Color map for the given values')
    parser.add_argument('--nc', dest='nan_color', type=float, default=0.65, 
    help='Gray color (between 0 and 1) for regions that have no assigned value')
    parser.add_argument('--na', dest='nan_alpha', type=float, default=0.8, 
    help='Alpha (opacity) value for regions that have no assigned value')

    args = parser.parse_args()
    controller = atlas_controller.AtlasController()
    controller.initialize(resolution=args.resolution, mapping=args.mapping, embed_ui=True, jupyter=False)
    load_priors(controller, 
                file_path=args.file_path, 
                aggregator=args.aggregator,
                color_map_func=args.color_map,
                nan_color=[args.nan_color]*3, 
                nan_alpha=args.nan_alpha)
    controller.render()