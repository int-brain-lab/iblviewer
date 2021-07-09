# DEMO: add insertion probes data from IBL database (DataJoints)
from one.api import ONE
import numpy as np

from iblviewer.mouse_brain import MouseBrainViewer


def get_bwm_ins_alyx(one):
    """
    Return insertions that match criteria :
    - project code
    - session QC not critical (TODO may need to add probe insertion QC)
    - at least 1 alignment
    - behavior pass
    :param one: "one" connection handler
    :return:
    ins: dict containing the full details on insertion as per the alyx rest query
    ins_id: list of insertions eids
    sess_id: list of (unique) sessions eids
    """
    ins = one.alyx.rest('insertions', 'list',
                        provenance='Ephys aligned histology track',
                        django='session__project__name__icontains,ibl_neuropixel_brainwide_01,'
                               'session__qc__lt,50,'
                               'json__extended_qc__alignment_count__gt,0,'
                               'session__extended_qc__behavior,1')
    
    ins_ids = [item['id'] for item in ins]
    sess_id = [item['session_info']['id'] for item in ins]
    # Here's what's in 'json':
    # dict_keys(['qc', 'n_units', 'xyz_picks', 'extended_qc', 'drift_rms_um', 'firing_rate_max', 'n_units_qc_pass', 
    # 'amplitude_max_uV', 'firing_rate_median', 'amplitude_median_uV', 'whitening_matrix_conditioning'])
    positions = []
    for item in ins:
        picks = item['json'].get('xyz_picks', [])
        positions.append(picks)
    sess_id = np.unique(sess_id)
    return positions, ins_ids


def get_picks_mean_vectors(xyz_picks, extent=3):
    """
    Get a mean vector from picks coordinates
    :param xyz_picks: Dictionary xyz picks, the key being the identifier for that data set
    :param extent: Number of points to take from start and end for mean computation of end points
    :return: List of varying lists of 3D points and a list of line ids
    """
    vectors = []
    ids = []
    # Mean between first and last three picks
    for ins_id in range(len(xyz_picks)):
        raw_picks = xyz_picks[ins_id]
        end_pt = np.mean(raw_picks[-extent:], axis=0)
        start_pt = np.mean(raw_picks[:extent], axis=0)
        vectors.append([start_pt, end_pt])
        ids.append(ins_id)
    return vectors, ids


def add_insertion_probes(viewer, one_connection, as_segments=False, line_width=2, trim_outliers=True):
    """
    Add insertion probe vectors
    :param viewer: The IBLViewer controller
    :param one_connection: The "one" connection to IBL server
    :param as_segments: Whether insertion probes should be reduced to straight lines (segments)
    :param trim_outliers: Whether you want the lines to be cut when they're out of the brain
    :param with_labels: Whether labels should be added to the lines
    """
    lines_data, line_ids = get_bwm_ins_alyx(one_connection)
    if as_segments:
        segments_data, segment_ids = get_picks_mean_vectors(lines_data)
        line_ids = np.array(line_ids)
        segment_ids = line_ids[segment_ids]
        lines = viewer.add_segments(segments_data, line_width=line_width, 
                                    add_to_scene=True, trim_outliers=trim_outliers)
    else:
        lines = viewer.add_lines(lines_data, line_width=line_width,
                                add_to_scene=True, trim_outliers=trim_outliers)
    return lines


if __name__ == '__main__':

    one_connection = ONE(base_url="https://alyx.internationalbrainlab.org")
    viewer = MouseBrainViewer()
    viewer.initialize(resolution=25, embed_ui=True)
    add_insertion_probes(viewer, one_connection, as_segments=False, line_width=5)
    viewer.show()
