
# DEMO 2: add point neurons from conenctivity data
#from oneibl import one
from oneibl.one import ONE
import numpy as np
from vedo import *
from iblviewer import *

import os
import numpy as np
import pandas as pd
import pickle

import random
import vedo

'''
# Invoke your own code with the lines below but you should wrap it in
# properly encapsulated code...
%%bash
git clone https://github.com/int-brain-lab/friendly_neurons.git
%load friendly_neurons/data_analysis.py
'''

def add_point_neurons(controller, data, with_labels=False):
    """
    Add point neurons
    :param controller: IBLViewer atlas controller
    :param data: At least 5D array of x, y, z, region, custom data. 
    All the columns after 'region' are be stored as a time series
    :param with_labels: Whether labels are added to the points
    """
    if isinstance(data, str):
        pickles = []
        data_path = data
        with (open(os.path.abspath(data_path), 'rb')) as openfile:
            while True:
                try:
                    pickles.append(pickle.load(openfile))
                except EOFError:
                    break
        data = pickles[0]

    # Structure of 'data': x | y | z | region | pre-time allegiance | during-time allegiance | post-time allegiance 
    positions = []
    regions = []
    timings = []
    # Cleaning weird data and extracting what we need. When Alessandro fixes his code, we can get rid of this cleaning.
    for weird_data in data:
        try:
            positions.append([weird_data[0][0], weird_data[1][0], weird_data[2][0]])
        except TypeError:
            continue
            #positions.append(bad_stuff[:3])
        regions.append(weird_data[3])
        timings.append(weird_data[4:])
    positions = np.array(positions).astype(np.float)

    timings = np.array(timings)
    regions = np.array(regions)
    # 16um is a good compromise for visibility from afar. So we make somata roughly 2-3 times larger than they are in reality
    points = controller.view.new_points(positions, radius=16, values=timings, as_spheres=True, noise_amount=100)
    
    actors = [points]
    if with_labels:
        # Using functions from vedo (easy-to-use wrapper on top of VTK)
        labels = points.labels('id', cells=True)
        actors.append(labels)
    controller.plot.add(actors)
    return points



if __name__ == '__main__':

    #one = ONE(base_url="https://alyx.internationalbrainlab.org")
    resolution = 25 # units = um
    mapping = 'Beryl'
    controller = atlas_controller.AtlasController()
    controller.initialize(resolution, mapping, embed_ui=True, jupyter=False)
    
    data = ['./examples/data/exp2_db4df448-e449-4a6f-a0e7-288711e7a75a_both', 
    './examples/data/exp3_3dd347df-f14e-40d5-9ff2-9c49f84d2157_both', 
    './examples/data/exp4_3c851386-e92d-4533-8d55-89a46f0e7384_both', 
    './examples/data/exp5_158d5d35-a2ab-4a76-87b0-51048c5d5283_both']
    for data_set in data:
        add_point_neurons(controller, data_set)
    
    controller.render()