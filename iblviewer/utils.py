from datetime import datetime
import numpy as np
import os

import vtk
from vedo import *


def spherical_degree_angles_to_xyz(radius, theta, phi):
    return spher2cart(radius, theta / 180 * math.pi, phi / 180 * math.pi) 


def change_file_name(file_path, prefix=None, name=None, suffix=None):
    """
    Change the file name from the given file path
    :param file_path: Input file path
    :param prefix: Prefix to the file name
    :param name: Whether a new name is set instead of the current name.
    If None, the current file name is used.
    :param suffix: Suffix to the file name
    :return: New file path
    """
    path, file_name, extension = split_path(file_path)
    if prefix is None:
        prefix = ''
    if suffix is None:
        suffix = ''
    if name is None or name == '' or not isinstance(name, str):
        name = file_name
    return os.path.join(path, prefix + name + suffix + extension)


def split_path(path):
    """
    Split any given file path to folder path, file name and extension
    :return: Folder path, file name and extension
    """
    base_name = os.path.basename(path)
    file_name, extension = os.path.splitext(base_name)
    return path[:-len(base_name)], file_name, extension


def time_diff(t):
    now  = datetime.now()
    duration = now - t
    return duration.total_seconds()


def get_actor_center(actor):
    """
    Get center position of an actor
    :param actor: VTK actor
    :return: 3d array
    """
    if isinstance(actor, Volume):
        return actor.center() + actor.pos()
    else:
        return actor.centerOfMass() + actor.pos() # TODO: check that this is necessary (adding pos)


def get_actor_dimensions(actor):
    """
    Get dimensions of an actor
    :param actor: VTK actor
    :return: 3d array
    """
    if isinstance(actor, Volume):
        return actor.dimensions() * actor.spacing()# equivalent to self.model.resolution
    else:
        xmin, xmax, ymin, ymax, zmin, zmax = actor.bounds()
        return np.array([xmax - xmin, ymax - ymin, zmax - zmin])


def get_transformation_matrix(origin, normal):
    """
    Get transformation matrix for a plane given by its origin and normal
    :param origin: Origin 3D vector
    :param normal: Normal 3D vector
    :return: Matrix and Translation
    """
    newaxis = utils.versor(normal)
    initaxis = (0, 0, 1)
    crossvec = np.cross(initaxis, newaxis)
    angle = np.arccos(np.dot(initaxis, newaxis))
    T = vtk.vtkTransform()
    T.PostMultiply()
    T.RotateWXYZ(np.rad2deg(angle), crossvec)
    T.Translate(np.array(origin))
    M = T.GetMatrix()
    return M, T


def add_scalar_bar(lut, pos=(0.8, 0.05), font_color=[0, 0, 0], title="", titleYOffset=15, titleFontSize=12,
                    size=(None,None), nlabels=None, horizontal=False, useAlpha=True):
    """
    Add a 2D scalar bar for the specified obj. Modified method from vedo.addons.addScalarBar
    :param list pos: fractional x and y position in the 2D window
    :param list size: size of the scalarbar in pixel units (width, heigth)
    :param int nlabels: number of numeric labels to be shown
    :param bool useAlpha: retain trasparency in scalarbar
    :param bool horizontal: show in horizontal layout
    """
    sb = vtk.vtkScalarBarActor()
    #sb.SetLabelFormat('%-#6.3g')
    #print(sb.GetLabelFormat())
    sb.SetLookupTable(lut)
    sb.SetUseOpacity(useAlpha)
    sb.SetDrawFrame(0)
    sb.SetDrawBackground(0)
    if lut.GetUseBelowRangeColor():
        sb.DrawBelowRangeSwatchOn()
        sb.SetBelowRangeAnnotation('')
    if lut.GetUseAboveRangeColor():
        sb.DrawAboveRangeSwatchOn()
        sb.SetAboveRangeAnnotation('')
    if lut.GetNanColor() != (0.5, 0.0, 0.0, 1.0):
        sb.DrawNanAnnotationOn()
        sb.SetNanAnnotation('nan')

    if title:
        if "\\" in repr(title):
            for r in shapes._reps:
                title = title.replace(r[0], r[1])
        titprop = sb.GetTitleTextProperty()
        titprop.BoldOn()
        titprop.ItalicOff()
        titprop.ShadowOff()
        titprop.SetColor(font_color)
        titprop.SetVerticalJustificationToTop()
        titprop.SetFontSize(titleFontSize)
        titprop.SetFontFamily(vtk.VTK_FONT_FILE)
        titprop.SetFontFile(settings.fonts_path + settings.defaultFont +'.ttf')
        sb.SetTitle(title)
        sb.SetVerticalTitleSeparation(titleYOffset)
        sb.SetTitleTextProperty(titprop)

    sb.UnconstrainedFontSizeOn()
    sb.DrawAnnotationsOn()
    sb.DrawTickLabelsOn()
    sb.SetMaximumNumberOfColors(256)

    if horizontal:
        sb.SetOrientationToHorizontal()
        sb.SetNumberOfLabels(3)
        sb.SetTextPositionToSucceedScalarBar()
        sb.SetPosition(pos)
        sb.SetMaximumWidthInPixels(1000)
        sb.SetMaximumHeightInPixels(50)
    else:
        sb.SetNumberOfLabels(7)
        sb.SetTextPositionToPrecedeScalarBar()
        sb.SetPosition(pos[0]+0.09, pos[1])
        sb.SetMaximumWidthInPixels(60)
        sb.SetMaximumHeightInPixels(250)

    if size[0] is not None: sb.SetMaximumWidthInPixels(size[0])
    if size[1] is not None: sb.SetMaximumHeightInPixels(size[1])

    if nlabels is not None:
        sb.SetNumberOfLabels(nlabels)

    sctxt = sb.GetLabelTextProperty()
    sctxt.SetFontFamily(vtk.VTK_FONT_FILE)
    sctxt.SetFontFile(settings.fonts_path + settings.defaultFont +'.ttf')
    sctxt.SetColor(font_color)
    sctxt.ItalicOff()
    sctxt.SetShadow(0)
    sctxt.SetFontSize(titleFontSize-2)
    sb.SetAnnotationTextProperty(sctxt)
    sb.PickableOff()
    return sb

# ------------------------------ WORK IN PROGRESS, NOT SURE THE BELOW METHODS WILL STAY HERE
def add_region_surface(region_id=997, meshes_path='./data/allen/structure/structure_meshes/clean_ply/', ext='ply'):
    #if region_id == 997:
        #region_id = str(region_id) + 'm'
    region_mesh_path = meshes_path + str(region_id) + '.' + ext
    if os.path.exists(region_mesh_path):
        return load(region_mesh_path)

def read_surfaces(glb_path='./data/surfaces/brain_regions_leaves_one_up.glb'):
    """
    Read surface data from a binary GLTF scene
    """
    loader = vtk.vtkGLTFDocumentLoader()
    reader = vtk.vtkGLTFReader()
    reader.SetFileName(glb_path)
    reader.Update() 
    #reader.Read()

    mb = reader.GetOutput()
    iterator = mb.NewIterator()

    vtk_polyobjects = []
    while not iterator.IsDoneWithTraversal():
        item = iterator.GetCurrentDataObject()
        vtk_polyobjects.append(item)
        iterator.GoToNextItem()

    print('Read', len(vtk_polyobjects), 'objects from glb')
    """ 
    output_port = reader.GetOutputPort()

    mapper = vtkPolyDataMapper()
    mapper.SetInputConnection(output_port)

    actor = vtkActor()
    actor.SetMapper(mapper) 
    """
    return vtk_polyobjects