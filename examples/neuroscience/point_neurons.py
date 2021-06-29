# Add point neurons from connectivity data
from pathlib import Path

import os
import numpy as np
import pickle

from iblviewer.mouse_brain import MouseBrainViewer
from iblviewer import utils

'''
# Invoke your own code with the lines below but you should wrap it in
# properly encapsulated code...
%%bash
git clone https://github.com/int-brain-lab/friendly_neurons.git
%load friendly_neurons/data_analysis.py
'''

# After that, we try loading experimental data, here point neurons.
# Once this cell is run, click on the viewer to see the update

def process_point_neurons(data):
    """
    Pre process point neurons data and find the common min and max values
    :param data: At least 5D array of x, y, z, region, custom data. 
    All the columns after 'region' are be stored as a time series
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
    positions = np.array(positions).astype(float)

    timings = np.array(timings)
    regions = np.array(regions)
    min_v = np.min(timings)
    max_v = np.max(timings)
    return positions, timings, min_v, max_v


if __name__ == '__main__':
    viewer = MouseBrainViewer()
    viewer.initialize(resolution=50, mapping='Allen', add_atlas=False, add_dwi=True, 
                    dwi_color_map='Greys_r', embed_ui=True)

    # Now add point neurons
    data = ['./exp2_db4df448-e449-4a6f-a0e7-288711e7a75a_both', 
    './exp3_3dd347df-f14e-40d5-9ff2-9c49f84d2157_both', 
    './exp4_3c851386-e92d-4533-8d55-89a46f0e7384_both', 
    './exp5_158d5d35-a2ab-4a76-87b0-51048c5d5283_both']
    data = [str(utils.EXAMPLES_DATA_FOLDER.joinpath(d)) for d in data]

    min_value = None
    max_value = None
    all_positions = []
    all_timings = []
    # A first loop to find out min and max values among timings
    for data_set in data:
        positions, timings, min_v, max_v = process_point_neurons(data_set)
        all_positions.append(positions)
        all_timings.append(timings)
        min_value = min(min_v, min_value) if min_value is not None else min_v
        max_value = max(max_v, max_value) if max_value is not None else max_v
        
    point_actors = []

    # Now visualize the points. Remove existing ones (if any) first and load new ones.
    viewer.plot.remove(point_actors, render=False)
    # We keep newly added objects in memory so that if you rerun this cell and change parameters, 
    # the previous points get replaced by the updated ones. Try changing the radius and run it again.
    point_actors = []
    for d_id in range(len(data)):
        # 16um is a good compromise for visibility from afar. So we make somata roughly 2-3 times larger than they are in reality
        points = viewer.add_points(all_positions[d_id], radius=16, values=all_timings[d_id], screen_space=False, 
                                    noise_amount=100, min_v=min_value, max_v=max_value)
        point_actors.append(points)
    viewer.plot.add(point_actors)

    viewer.show().close()