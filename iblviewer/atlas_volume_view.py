from iblviewer.volume_view import VolumeView
import iblviewer.utils as utils


class AtlasVolumeView(VolumeView):

    def build_surface_mesh(self, region=None):
        """
        Build a surface mesh with marching cubes algorithm
        """
        if not self.model.is_segmented_volume():
            #logging.error('[VolumeView.build_surface_mesh()] cannot work without the segmented Allen volume')
            return

        self.bounding_mesh = utils.load_surface_mesh('997')
        #mesh = vedo.utils.vedo2trimesh(actor)
        '''
        self.surface_actor.alpha(0)
        self.surface_actor.name = self.model.name + '_surface'
        self.surface_actor.mapper().SetClippingPlanes(self.clipping_planes)
        self.surface_actor.pickable(0)
        #self.surface_actor.ForceOpaqueOn()
        '''