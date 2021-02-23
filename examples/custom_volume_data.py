# BASE CASE: load the viewer with desired options
from vedo import *
from iblviewer import *

import os
import numpy as np
import pandas as pd
import pickle

import random
import vedo


# Data given by Berk Gerçek, International Brain Laboratory
def process_priors(controller, file_path='./examples/data/completefits_2020-11-09.p', randomize=None):
    """
    Process priors data and get color map and scalar values
    """
    pickles = []
    with (open(os.path.abspath(file_path), 'rb')) as openfile:
        while True:
            try:
                pickles.append(pickle.load(openfile))
            except EOFError:
                break
    df = pickles[0]
    filtered_df = df['rawpoints'].groupby('acronym').agg({'value':'median'})
    
    min_value = float(np.amin(filtered_df, axis=0).to_numpy()[0])
    max_value = float(np.amax(filtered_df, axis=0).to_numpy()[0])
    print('Min prior value ' + str(min_value))
    print('Max prior value ' + str(max_value))

    scalars_map = {}
    for acronym, value in filtered_df.iterrows():
        region_id, row_id = controller.model.get_region_and_row_id(acronym)
        if row_id == 0:
            # We ignore void acronym
            continue
        scalars_map[int(row_id)] = float(value.to_numpy()[0])

    return scalars_map


def get_color_map(controller, scalar_map, nan_color=[0.0, 0.0, 0.0], nan_alpha=1.0, seed=None):
    """
    Get a color map
    :param controller: IBLViewer.AtlasController
    :param scalar_map: Dictionary that maps scalar values in the dictionary to your custom values
    :param nan_color: Default color for unspecified values
    :param nan_alpha: Default alpha for unspecified values
    :param seed: Random seed to fake a time series
    :return: Color map and alpha map
    """
    if seed is not None:
        random.seed(seed)
    rgb = []
    alpha = []

    # Init all to clear gray (90% white)
    #c = np.ones((self.metadata.id.size, 4)).astype(np.float32) * 0.9
    #c[:, -1] = 0.0 if only_custom_data else alpha_factor
    #print('Assigning', values.size, 'to atlas ids', self.metadata.id.size)

    for r_id in range(len(controller.model.metadata)):
        rgb.append([r_id, nan_color])
        a = nan_alpha if r_id > 0 else 0.0
        alpha.append([r_id, a])

    values = scalar_map.values()
    min_p = min(values)
    max_p = max(values)
    rng_p = max_p - min_p

    """
    # Another way to compute colors with matplotlib cm. You will need to import matplotlib
    norm = matplotlib.colors.Normalize(vmin=min_p, vmax=max_p, clip=True)
    mapper = matplotlib.cm.ScalarMappable(norm=norm, cmap=cm.viridis)
    """
    
    for row_id in scalar_map:
        value = scalar_map[row_id]
        if seed is not None and seed > 0:
            value = value + random.random() * rng_p / 2
        """
        # Another way to compute colors with matplotlib cm 
        # (yields the same result as vedo, except you get alpha on top)
        r, g, b, a = mapper.to_rgba(value)
        rgb[row_id] = [row_id, [r, g, b]]
        """
        rgb[row_id] = [row_id, list(vedo.colorMap(value, 'viridis', min_p, max_p))]
        alpha[row_id] = [row_id, 1.0]

    return rgb, alpha


def load_priors_in_viewer(controller, nan_color=[0.0, 0.0, 0.0], nan_alpha=1.0, fake_time_series_steps=0):
    """
    Load priors into the viewer, faking a time series from there
    :param controller: IBLViewer.AtlasController
    :param nan_color: Default color for unspecified values
    :param nan_alpha: Default alpha for unspecified values
    :param fake_time_series_steps: Number of time series fake steps to demonstrate time series in the demo
    """
    scalar_map = process_priors(controller)
    for index in range(fake_time_series_steps + 1):
        rgb, alpha = get_color_map(controller, scalar_map, nan_color, nan_alpha, index)
        controller.add_transfer_function(scalar_map, rgb, alpha, make_current=False)



if __name__ == '__main__':
    resolution = 25 # units = um
    mapping = 'Allen'
    controller = atlas_controller.AtlasController()
    controller.initialize(resolution, mapping, embed_ui=True, jupyter=False)

    load_priors_in_viewer(controller, nan_color=[0.75, 0.75, 0.75], nan_alpha=0.5, fake_time_series_steps=100)
    controller.render()