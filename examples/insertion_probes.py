# DEMO 1: add insertion probes data from IBL database (DataJoints)
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


def get_bwm_ins_alyx(one):
    """
    Return insertions that match criteria :
    - project code
    - session QC not critical (TODO may need to add probe insertion QC)
    - at least 1 alignment
    - behavior pass
    :return:
    ins: dict containing the full details on insertion as per the alyx rest query
    ins_id: list of insertions eids
    sess_id: list of (unique) sessions eids
    """
    print('Querying alyx...')
    ins = one.alyx.rest('insertions', 'list',
                        provenance='Ephys aligned histology track',
                        django='session__project__name__icontains,ibl_neuropixel_brainwide_01,'
                               'session__qc__lt,50,'
                               'json__extended_qc__alignment_count__gt,0,'
                               'session__extended_qc__behavior,1')
    
    ins_id = [item['id'] for item in ins]
    sess_id = [item['session_info']['id'] for item in ins]
    '''
    for item in ins:
        print(item['json'].keys())
    return None, None, None
    '''
    # Here's what's in 'json':
    # dict_keys(['qc', 'n_units', 'xyz_picks', 'extended_qc', 'drift_rms_um', 'firing_rate_max', 'n_units_qc_pass', 'amplitude_max_uV', 'firing_rate_median', 'amplitude_median_uV', 'whitening_matrix_conditioning'])
    xyz_picks = {}
    for item in ins:
        ins_id = item['id']
        picks = np.array(item['json'].get('xyz_picks', []))
        xyz_picks[ins_id] = picks
    sess_id = np.unique(sess_id)
    #ins, ins_id, sess_idins, ins_id, sess_id = get_bwm_ins_alyx(one)
    return xyz_picks


def get_picks_mean_vectors(xyz_picks):
    vectors = []
    ids = []
    # Mean between first and last three picks
    for ins_id in xyz_picks:
        raw_picks = xyz_picks[ins_id]
        end_pt = np.mean(raw_picks[-3:], axis=0)
        start_pt = np.mean(raw_picks[:3], axis=0)
        vectors.append([start_pt, end_pt])
        ids.append(ins_id)
    return vectors, ids


def add_insertion_probes(controller, one, reduce=True, with_labels=True):
    xyz_picks = get_bwm_ins_alyx(one)

    vectors, ids = get_picks_mean_vectors(xyz_picks)

    points = df[['x', 'y', 'z']].to_numpy()
    angles = df[['depth', 'theta', 'phi']].to_numpy()
    probes_data = df[['x', 'y', 'z', 'depth', 'theta', 'phi']].to_numpy()

    lines = viewer.add_lines(vectors[:, 0], vectors[:, 1], values=ids, add_to_scene=False)
    actors = [lines]
    if with_labels:
        labels = lines.labels('id', cells=True)
        actors.append(labels)
    controller.plot.add(actors)
    return lines


if __name__ == '__main__':
    one = ONE(base_url="https://alyx.internationalbrainlab.org")
    resolution = 25 # units = um
    mapping = 'Allen'
    controller = atlas_controller.AtlasController()
    controller.initialize(resolution, mapping, embed_ui=True, jupyter=False)

    add_insertion_probes(controller, one)
    controller.render()