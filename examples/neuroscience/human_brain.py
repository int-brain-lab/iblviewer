import argparse
from cloudvolume import CloudVolume
from iblviewer.application import Viewer


def load_human_brain_volume(atlas='classif', lod=3):
    url = 'precomputed://https://neuroglancer.humanbrainproject.eu/'
    url += 'precomputed/BigBrainRelease.2015/' + atlas
    print('About to load precomputed volume', url)
    volume = CloudVolume(url, mip=lod)
    print('Metadata for lod', lod, ':', volume.scale)
    image = volume[:, :, :]
    return image.flatten().reshape(volume.scale['size']), volume

def view_volume(image, resolution, color_map=None):
    viewer = Viewer()
    viewer.initialize(embed_ui=True)
    viewer.add_volume(image, resolution, None, color_map, select=True)
    viewer.show()

#volume, cloud_volume = load_human_brain_volume()
#view_volume(volume, cloud_volume.scale['resolution'][0])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Human Brain Atlas')
    parser.add_argument('-l', dest='lod', type=int, default=3, 
    help='LOD value')
    parser.add_argument('-a', dest='atlas', type=str, default='classif', 
    help='Atlas type, either 8bit or classif')
    parser.add_argument('-c', dest='color_map', type=str, default='viridis', 
    help='Color map')
    args = parser.parse_args()
    volume, cloud_volume = load_human_brain_volume(args.atlas, args.lod)
    resolution = cloud_volume.scale['resolution'][0]
    view_volume(volume, resolution, args.color_map)