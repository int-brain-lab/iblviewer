import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import convolve
import matplotlib.pyplot as plt

from ibllib.dsp import fcn_cosine
from ibllib.atlas import AllenAtlas
from iblviewer.atlas_view import AtlasView, VolumeView
from iblviewer import atlas_controller

DIST_FCN = np.array([100, 150]) / 1e6


def compute_coverage_volume():
    file_channels = next(Path(atlas_controller.__file__).parents[1].joinpath(
        'examples', 'data').glob('channels.*.pqt'))

    # read the channels files and create a volume with one in each voxel containing a spike
    channels = pd.read_parquet(file_channels)
    ba = AllenAtlas(25)
    cvol = np.zeros(ba.image.shape, dtype=np.float)
    xyz = np.c_[channels['ml'].to_numpy(), channels['ap'].to_numpy(), channels['dv'].to_numpy()] / 1e6
    iii = ba.bc.xyz2i(xyz)
    cvol[np.unravel_index(ba._lookup(xyz), cvol.shape)] = 1
    # np.savez_compressed(file_coverage, cvol)

    # create the convolution kernel, 3D cosine function decreasing from 1 to 0 between bounds in DISC_FCN
    dx = ba.bc.dx
    template = np.arange(- np.max(DIST_FCN) - dx, np.max(DIST_FCN) + 2 * dx, dx) ** 2
    kernel = sum(np.meshgrid(template, template, template))
    kernel = 1 - fcn_cosine(DIST_FCN)(np.sqrt(kernel))

    # and convolve (NB: 3D FFT would probably go much faster here)
    start = time.time()
    cvol = convolve(cvol, kernel)
    print(time.time() - start)
    # file_coverage_conv = Path('/datadisk/FlatIron/tables').joinpath('coverage_conv_500.npz')
    # np.savez_compressed(file_coverage_conv, cvol)
    return cvol


def add_vol(controller, vol):
    """
    This is a complete hack just to get a display for prototype.
    We need to build a proper primitive to add other full volume channels (ie. not going through
    mapping) for the visualizer
    """
    controller.model.volume.volume = np.transpose((vol).astype(np.int), (2, 0, 1))
    controller.view = AtlasView(controller.plot, controller.model)
    controller.view.initialize()

    controller.volume_view = VolumeView(controller.plot, controller.model.volume, controller.model)
    controller.view.volume_view = controller.volume_view
    controller.render()


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


controller = atlas_controller.AtlasController()
controller.initialize(resolution=25, mapping='Allen-lr', embed_ui=True, jupyter=False)
ba = controller.model.atlas

cvol = compute_coverage_volume()
ncov = cvol.copy()
ncov[ncov > 0] = -1
ncov += 1
add_vol(controller, ncov)

ncov[ncov == -1] = np.nan


plt.figure(), plot_agg(ncov, 2, ax=ba.plot_hslice(-.002), ba=ba, alpha=0.5)
plt.figure(), plot_agg(ncov, 1, ax=ba.plot_sslice(0), ba=ba, alpha=0.5)
plt.figure(), plot_agg(ncov, 0, ax=ba.plot_cslice(0), ba=ba, alpha=0.5)
