from one.api import ONE
import os
import pickle
import numpy as np

import ibllib.atlas as atlas
from brainbox.io.one import load_channels_from_insertion
import alf.io

from iblviewer.mouse_brain import MouseBrainViewer
from iblviewer import utils


def get_valid_insertions(force_query=False, local_file_path=None):
    """
    Get all valid insertions from the data base
    """
    insertions = None
    if local_file_path is None:
        local_file_path = str(utils.EXAMPLES_DATA_FOLDER.joinpath('valid_insertions.p'))
    if not force_query and os.path.exists(os.path.abspath(local_file_path)):
            insertions = pickle.load(open(local_file_path, 'rb'))
            print('Using local file for insertions', local_file_path)
    else:
            dq = 'session__project__name__icontains,ibl_neuropixel_brainwide_01,session__qc__lt,50,'
            dq += '~json__qc,CRITICAL,session__extended_qc__behavior,1,json__extended_qc__tracing_exists,True,'
            dq += '~session__extended_qc___task_stimOn_goCue_delays__lt,0.9,'
            dq += '~session__extended_qc___task_response_feedback_delays__lt,0.9,'
            dq += '~session__extended_qc___task_wheel_move_before_feedback__lt,0.9,'
            dq += '~session__extended_qc___task_wheel_freeze_during_quiescence__lt,0.9,'
            dq += '~session__extended_qc___task_error_trial_event_sequence__lt,0.9,'
            dq += '~session__extended_qc___task_correct_trial_event_sequence__lt,0.9,'
            dq += '~session__extended_qc___task_reward_volumes__lt,0.9,'
            dq += '~session__extended_qc___task_reward_volume_set__lt,0.9,'
            dq += '~session__extended_qc___task_stimulus_move_before_goCue__lt,0.9,'
            dq += '~session__extended_qc___task_audio_pre_trial__lt,0.9'
            query_result = one.alyx.rest('insertions', 'list', django=dq)
            #insertions = one.alyx.rest('insertions', 'list', django=query)
            insertions = []
            for insertion in query_result:
                    insertions.append(insertion)
            pickle.dump(insertions, open(local_file_path, 'wb'))
            #points_query = one.alyx.rest('')
    return insertions


def get_insertions_data(one=None, ba=None, insertions=None, start=None, end=None, 
                        exceptions=None, force_query=False, local_file_path=None):
    """
    Get all valid insertions and related data from the DB.

    """
    insertions_data = None
    if local_file_path is None:
        local_file_path = str(utils.EXAMPLES_DATA_FOLDER.joinpath('valid_insertions_data.p'))
    if not force_query and os.path.exists(os.path.abspath(local_file_path)):
            insertions_data = pickle.load(open(local_file_path, 'rb'))
            print('Using local file for insertions data', local_file_path)
    else:
        if one is None:
            one = ONE()
        if ba is None:
            ba = atlas.AllenAtlas()
        if insertions is None:
            insertions = get_valid_insertions()
        num_insertions = len(insertions)

        print('')
        print('')
        print('Got', len(insertions), 'valid insertions (that do not necessarily have data)')

        # Retrieving XYZ point neuron coordinates goes like this: 
        # spikes.clusters -> clusters.channels -> channels.xyz 
        # but in many cases, channels.xyz (a file called mlapdv.npy) does not exist
        # so you have to compute XYZ coordinates given spikes depths and insertion XYZ
        dtypes = [
            'spikes.depths',
            'spikes.amps',
            'spikes.times',
            'spikes.clusters',
            'clusters.channels',
            'clusters.mlapdv'
        ]

        # Filter insertions and retain only valid data because even 
        # the insertions deemed valid are sometimes not what we want
        insertions_data = {}

        if not isinstance(start, int):
            start = 0
        if not isinstance(end, int):
            end = num_insertions
        start = max(0, start)
        end = min(num_insertions, end)
        print('Getting data for insertions', start, 'to', end, 'out of', num_insertions)
        for loop_id in range(start, end):
            insertion = insertions[loop_id]
            probe_id = insertion['id']
            if exceptions is not None and probe_id in exceptions:
                print('Probe', loop_id, 'with id', probe_id, 'is in exclusion list (probably because of invalid data)')
                continue
            data_sets = one.alyx.rest('datasets', 'list', probe_insertion=probe_id, django=f'dataset_type__name__in,{dtypes}')
            if data_sets is None or len(data_sets) < 1:
                print('Probe', loop_id, ': no dataset found (!) for id', probe_id)
                continue
            print('Probe', loop_id, 'using', len(data_sets), 'data sets for id', probe_id)
            insertions_data[probe_id] = data_sets
        
        print(len(insertions_data), 'valid insertions found on', num_insertions)
        print('')
        pickle.dump(insertions_data, open(local_file_path, 'wb'))
    return insertions_data


def get_point_neurons_data(one=None, ba=None, insertions=None, start=0, end=None, 
                            recompute_positions=True, probe_ids=None, 
                            exceptions=['825ba9b8-ce03-49b7-b1a8-4d85ae2185af'], 
                            local_file_path=None):
    """
    Get all point neurons data sets that IBL has measured/recorded.
    Warning, this can result in veeery long download time initially.
    I had to download 85GB of data (including relevant spike data) in May 2021. 
    And the best part is that the position of the neurons in the end
    is put in a pickle file less than 1MB. :)
    :param one: ONE connection object
    :param ba: BrainAtlas object
    :param insertions: 
    :param start: 
    :param end: 
    :param recompute_positions: 
    :param probe_ids: 
    :param exceptions: 
    :param local_file_path: 
    """
    if local_file_path is None:
        local_file_path = str(utils.EXAMPLES_DATA_FOLDER.joinpath('ibl_point_neurons.npz'))
    
    use_local_data = not recompute_positions and os.path.exists(os.path.abspath(local_file_path))
    if use_local_data:
        result = np.load(local_file_path, allow_pickle=True)
        print('Point neurons data loaded (using local npz storage)', local_file_path)
        insertion_ids = result['insertion_ids']
        xyz_positions = result['xyz_positions']
        xyz_resolved_mask = result['xyz_resolved_mask']
        data_mask = result['data_mask']
        return insertion_ids, xyz_positions, xyz_resolved_mask, data_mask

    if one is None:
        one = ONE()
    if ba is None:
        ba = atlas.AllenAtlas()
    if insertions is None:
        insertions = get_valid_insertions()
    num_insertions = len(insertions)
    insertions_data = get_insertions_data(one, ba, insertions, start, end, exceptions)

    insertion_ids = [''] * num_insertions
    xyz_positions = [[]] * num_insertions
    xyz_resolved_mask = [False] * num_insertions
    data_mask = [False] * num_insertions

    loop_id = -1
    got_ids_filter = isinstance(probe_ids, dict) or isinstance(probe_ids, list)
    
    # Lots of try catch ahead! This is to handle all the cases where things might be missing or failing
    for probe_id in insertions_data:
        loop_id += 1
        data_sets = insertions_data[probe_id]
        if got_ids_filter and probe_id not in probe_ids:
            continue
        try:
            local_files = one.download_datasets(data_sets)
        except Exception as e:
            print('Error downloading dataset', e)
            continue
        
        if local_files is None:
            print('Local files not found for', probe_id)
            continue
        '''
        TODO: this is where you start using spike data.
        try:
            spikes = alf.io.load_object(local_files[0].parent, 'spikes')
        except AttributeError:
            continue
        '''
        try:
            clusters = alf.io.load_object(local_files[0].parent, 'clusters')
        except AttributeError:
            continue
        #channels = clusters['channels']

        print('Processing insertion', loop_id+1, 'on', len(insertions_data), '- Probe id', probe_id)
        # When mlapdv dataset does not exist, it means we have to 
        # estimate xyz with the given spike depths and insertion xyz
        if 'mlapdv' in clusters:
            xyz_positions[loop_id] = clusters['mlapdv']
        else:
            '''
            Exemple of an insertion that throws a KeyError:
            0393f34c-a2bd-4c01-99c9-f6b4ec6e786d
            KeyError: ibl.brainbox.io '2021-04-20T15:49:05_anup.khanal'
            So we have an id but something went wrong in the data pipeline and it got invalid
            '''
            insertion = one.alyx.rest('insertions', 'read', id=probe_id)
            #print('Is insertion equal to itself???', insertion==insertion2, insertion['id'], insertion2['id'])
            try:
                channel_data = load_channels_from_insertion(insertion, one=one, ba=ba)
            except Exception as e:
                print(e)
                continue
            # Units are in meters so we make that consistent with mlapdv data sets that are in microns
            xyz_positions[loop_id] = channel_data * 1000000

        xyz_resolved_mask[loop_id] = 'mlapdv' in clusters
        data_mask[loop_id] = True
        insertion_ids[loop_id] = probe_id
        
    #if not isinstance(xyz_positions, np.ndarray):
    xyz_positions = np.array(xyz_positions)
    if local_file_path is not None:
        print('Saving processed data under', local_file_path)
        np.savez_compressed(local_file_path, insertion_ids=insertion_ids, xyz_positions=xyz_positions, 
        xyz_resolved_mask=xyz_resolved_mask, data_mask=data_mask)
    
    print('Point neurons data downloaded')
    return insertion_ids, xyz_positions, xyz_resolved_mask, data_mask


#def get_points_inside_brain(viewer, positions):


if __name__ == '__main__':

    # Test probe ids. The first one has MLAPDV data, the others have their channels computed instead.
    probe_ids = ['00a824c0-e060-495f-9ebc-79c82fef4c67',
                '0143d3fe-79c2-4922-8332-62c3e4e0ba85',
                '4762e8ed-4d94-4fd7-9522-e927f5ffca74',
                '4755877d-fd14-42b3-bc15-e1996d153015']

    # If you want to run the test with only the selection from above, then comment the below line
    probe_ids = None

    '''
    There is an error for this data:
    Spike data not found: 56f2a378-78d2-4132-b3c8-8c1ba82be598
    '''
    data = get_point_neurons_data(probe_ids=probe_ids, recompute_positions=False)
    insertion_ids, xyz_positions, xyz_resolved_mask, data_mask = data
    positions = []
    values = []
    loop_id = 0
    # xyz_positions is a sparse array so we take only valid data from it
    insertions = xyz_positions[data_mask]

    for channels in insertions:
        channels_positions = channels if isinstance(channels, list) else channels.tolist()
        #inside_brain_positions = get_points_inside_brain(viewer, channels_positions)
        positions += channels_positions
        mock_ids = np.ones(len(channels_positions))*loop_id
        values += mock_ids.tolist()
        loop_id += 1
    positions = np.array(positions)
    values = np.array(values)
    
    av = np.copy(values) * np.random.random(len(values))
    bv = np.copy(values) * np.random.random(len(values))
    cv = np.copy(values) * np.random.random(len(values))
    values = np.c_[values, av, bv, cv]

    viewer = MouseBrainViewer()
    print('Data report: using', len(insertions), 'insertions and', len(positions), 'channels (point neurons)')
    print('-> Bounds:', np.amin(positions, axis=0), np.amax(positions, axis=0))
    # This starts the viewer with two volumes: the segmented atlas and the DWI of the mouse brain
    viewer.initialize(resolution=50, add_dwi=True, dwi_color_map='viridis', 
                        add_atlas=True, mapping='Allen', dark_mode=True, embed_ui=True)

    # Below are a few optimizations you could use if your computer 
    # is too slow to visualize the result:

    # 1. We add all the points together as one object. 
    # The fewer objects on your 3D scene, the faster it will render.

    # 2. Activate screen space mode so that the points are in 2D.
    # This can literally change the game if you have a low-end computer

    # 3. Do not trim outliers, because the computation to know if each
    # and every point is within the brain is quite demanding.
    points = viewer.add_points(positions, radius=8, values=values, screen_space=True, 
                                noise_amount=100, trim_outliers=False, add_to_scene=True)

    # You will need some CPU/GPU power to handle volume 
    # transparency and all those semi-transparent points
    points.alpha(0.5)

    # Finally, before showing everything, we select the latest added object (the points)
    viewer.select(-1)
    viewer.show()