import time
import os

import numpy as np
import pandas as pd
from scipy.ndimage import convolve

from ibllib.dsp import fcn_cosine
from ibllib.atlas import AllenAtlas

from iblviewer.mouse_brain import MouseBrainViewer
from iblviewer import utils


DIST_FCN = np.array([100, 150]) / 1e6


def compute_coverage_volume(ba=None):
    file_channels = next(utils.EXAMPLES_DATA_FOLDER.glob('channels.*.pqt'))

    print('Computing coverage volume...')
    # read the channels files and create a volume with one in each voxel containing a spike
    channels = pd.read_parquet(file_channels)
    if ba is None:
        ba = AllenAtlas(25)
    cvol = np.zeros(ba.image.shape, dtype=float)
    xyz = np.c_[channels['ml'].to_numpy(), channels['ap'].to_numpy(), channels['dv'].to_numpy()] / 1e6
    iii = ba.bc.xyz2i(xyz)
    cvol[np.unravel_index(ba._lookup(xyz), cvol.shape)] = 1

    # create the convolution kernel, 3D cosine function decreasing from 1 to 0 between bounds in DISC_FCN
    dx = ba.bc.dx
    template = np.arange(- np.max(DIST_FCN) - dx, np.max(DIST_FCN) + 2 * dx, dx) ** 2
    kernel = sum(np.meshgrid(template, template, template))
    kernel = 1 - fcn_cosine(DIST_FCN)(np.sqrt(kernel))

    # and convolve (NB: 3D FFT would probably go much faster here)
    start = time.time()
    cvol = convolve(cvol, kernel)
    print('Done in', time.time() - start, 'seconds')
    return cvol

'''
def plot_agg(ccov, axis, ax=None, ba=None, **kwargs):
    if not ax:
        ax = plt.gca()
        ax.axis('equal')
    hm = np.sum(ccov, axis=axis)
    hm[np.all(ccov == 0, axis=axis)] = np.nan
    # hm[np.all(ccov == 0, axis=axis)] = np.nan
    if axis == 0:
        axextent = 1  # coronal
        hm = hm.T
    elif axis == 1:
        axextent = 0  # sagittal
        hm = hm.T
    elif axis == 2:
        axextent = 2  # horizontal
    ax.imshow(hm, extent=ba.extent(axis=axextent), **kwargs)
    return hm
'''

viewer = MouseBrainViewer()
resolution = 50
#viewer.initialize(resolution=resolution, mapping='Allen-lr', embed_ui=True, dark_mode=False, add_atlas=False, add_dwi=True, dwi_color_map='viridis')
viewer.initialize(resolution=resolution, mapping='Allen-lr', embed_ui=True, dark_mode=False)

ba = viewer.ibl_model.atlas
file_path = utils.EXAMPLES_DATA_FOLDER.joinpath(f'./ncov_{resolution}.npz')
if os.path.exists(str(file_path)):
    ncov = np.load(str(file_path))['arr_0']
else:
    cvol = compute_coverage_volume(ba)
    ncov = cvol.copy()
    ncov[ncov < 0] = -1
    ncov += 1
    print('Done computing volume with range', np.min(ncov), np.max(ncov))
    np.savez_compressed(str(file_path), ncov)

cov_vol = viewer.add_volume(ncov, resolution, color_map='viridis', transpose=True, select=True) #, alpha_map=[0, 0.5, 1]
cov_vol.set_opacity(1.0)
viewer.show().close()
'''
ncov[ncov == -1] = np.nan

plt.figure(), plot_agg(ncov, 2, ax=ba.plot_hslice(-.002), ba=ba, alpha=0.5)
plt.figure(), plot_agg(ncov, 1, ax=ba.plot_sslice(0), ba=ba, alpha=0.5)
plt.figure(), plot_agg(ncov, 0, ax=ba.plot_cslice(0), ba=ba, alpha=0.5)
'''
