import argparse
from cloudvolume import CloudVolume
from iblviewer.launcher import IBLViewer


class HumanBrainData():

    def __init__(self):
        self.volume = None
        self.cloud_volume = None
        self.atlas = 'classif'
        self.lod = 3
        self.resolution = None
        self.color_map = None

    def load_volume(self, atlas='classif', lod=3):
        url = 'precomputed://https://neuroglancer.humanbrainproject.eu/'
        url += 'precomputed/BigBrainRelease.2015/' + atlas
        print('About to load precomputed volume', url)
        volume = CloudVolume(url, mip=lod)
        print('Metadata for lod', lod, ':', volume.scale)
        image = volume[:, :, :]
        self.volume = image.flatten().reshape(volume.scale['size'])
        self.cloud_volume = volume
        self.resolution = volume.scale['resolution'][0]

    def on_viewer_initialized(self, viewer):
        viewer.add_volume(self.volume, self.resolution, None, self.color_map, select=True)
        viewer.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Human Brain Atlas')
    parser.add_argument('-l', dest='lod', type=int, default=3, 
    help='LOD value')
    parser.add_argument('-v', dest='volume', type=str, default='classif', 
    help='Volume type, either 8bit or classif')
    parser.add_argument('-c', dest='color_map', type=str, default='viridis', 
    help='Color map')

    iblviewer = IBLViewer()
    args = iblviewer.parse_args(parser)

    hb = HumanBrainData()
    hb.color_map = args.color_map
    hb.load_volume(args.volume, args.lod)

    iblviewer.launch(hb.on_viewer_initialized)