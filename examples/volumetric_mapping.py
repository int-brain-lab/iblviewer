import sys
import os
import numpy as np
import pandas as pd
import random
import vedo
from iblviewer import atlas_controller

# Data given by Berk
def process_df(controller, file_path='./examples/data/completefits_2020-11-09.p',
               aggregator='median', grouper='acronym'):
    """
    Process priors data and get color map and scalar values
    """
    df = np.load(file_path, allow_pickle=True)
    
    df['r_over_chance'] = df['r_prior'] - df['r_prior_null']
    filtered_df = df.groupby('region').median()['r_over_chance']
    '''
    raw_df = df['rawpoints']
    filtered_df = raw_df.groupby(grouper).agg({'value': aggregator})
    min_value = float(np.amin(filtered_df, axis=0).to_numpy()[0])
    max_value = float(np.amax(filtered_df, axis=0).to_numpy()[0])
    print('Min prior value ' + str(min_value))
    print('Max prior value ' + str(max_value))
    '''
    scalars_map = {}
    for acronym, value in filtered_df.items():
        region_ids, row_ids = controller.model.get_region_and_row_id(acronym)
        if region_ids is None:
            print('Acronym', acronym, 'was not found in Atlas')
            continue
        #region_id = region_ids[0]
        #row_id = row_ids[0]
        for r_id in range(len(region_ids)):
            region_id = region_ids[r_id]
            row_id = row_ids[r_id]
            if region_id is None:
                print('Error, could not find acronym (ignoring it)', acronym)#, 'in', list(controller.model.metadata.acronym))
                continue
            if row_id == 0: #or value.isnull().values.any():
                # We ignore void acronym and nan values
                continue
            scalars_map[int(row_id)] = value #float(value.to_numpy()[0])
        #scalars_map[int(row_id)] = value #float(value.to_numpy()[0])
    return scalars_map
    
def get_color_map(controller, scalar_map, color_map_func='viridis', nan_color=[0.0, 0.0, 0.0], nan_alpha=0.0, seed=None):
    """
    Get a color map
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
    
def load_priors_in_viewer(controller, file_path, nan_color=[0.0, 0.0, 0.0], color_map_func='viridis', nan_alpha=0.0, fake_time_series_steps=0):
    """
    Load priors into the viewer, faking a time series from there
    """
    scalar_map = process_df(controller, file_path=file_path, aggregator=sys.argv[2])
    for index in range(fake_time_series_steps + 1):
        rgb_map, alpha_map = get_color_map(controller, scalar_map, color_map_func, nan_color, nan_alpha, index)
        controller.add_transfer_function(scalar_map, rgb_map, alpha_map, color_map_func, make_current=False)


if __name__ == '__main__':
    controller = atlas_controller.AtlasController()
    controller.initialize(resolution=25, mapping='Allen-lr', embed_ui=True, jupyter=False)
    load_priors_in_viewer(controller, file_path=sys.argv[1], color_map_func='viridis', nan_color=[0.65, 0.65, 0.65], nan_alpha=1.0, fake_time_series_steps=10)
    controller.render()