from datetime import datetime
import numpy as np
import os
from pathlib import Path

from vtk.util.numpy_support import numpy_to_vtk
import vtk
import vedo
import math
import trimesh


ROOT_FOLDER = Path(__file__).parent.parent
DATA_FOLDER = ROOT_FOLDER.joinpath('./data')
EXAMPLES_FOLDER = ROOT_FOLDER.joinpath('./examples')
EXAMPLES_DATA_FOLDER = ROOT_FOLDER.joinpath('./examples/data')


def Cross3DExt(pos=(0,0,0), size=1.0, thickness=0.25, color="b", alpha=1, res=4, lines_mode=True):
    """
    Build a 3D cross shape, mainly useful as a 3D marker.
    """
    if lines_mode:
        x1 = np.array([1.0, 0.0, 0.0]) * size / 2
        x2 = np.array([-1.0, 0.0, 0.0]) * size / 2
        c1 = vedo.Line(x1, x2, lw=thickness)

        y1 = np.array([0.0, 1.0, 0.0]) * size / 2
        y2 = np.array([0.0, -1.0, 0.0]) * size / 2
        c2 = vedo.Line(y1, y2, lw=thickness)

        z1 = np.array([0.0, 0.0, 1.0]) * size / 2
        z2 = np.array([0.0, 0.0, -1.0]) * size / 2
        c3 = vedo.Line(z1, z2, lw=thickness)
    else:
        c1 = vedo.Cylinder(r=thickness, height=size, res=res)
        c2 = vedo.Cylinder(r=thickness, height=size, res=res).rotateX(90)
        c3 = vedo.Cylinder(r=thickness, height=size, res=res).rotateY(90)
    cross = vedo.merge(c1,c2,c3).color(color).alpha(alpha)
    cross.SetPosition(pos)
    cross.name = "[Marker]"
    return cross


class LinesExt(vedo.Line):
    """
    Improved Lines class from vedo. 
    This one accepts point sets of varying lengths
    """
    def __init__(self, point_sets, c='gray', alpha=1, lw=1, dotted=False):
        
        polylns = vtk.vtkAppendPolyData()
        for point_set in point_sets:
            # This part taken from class Line, which accepts n points
            ppoints = vtk.vtkPoints()  # Generate the polyline
            ppoints.SetData(numpy_to_vtk(np.ascontiguousarray(point_set), deep=True))
            lines = vtk.vtkCellArray()
            npt = len(point_set)
            lines.InsertNextCell(npt)
            for i in range(npt):
                lines.InsertCellPoint(i)
            poly = vtk.vtkPolyData()
            poly.SetPoints(ppoints)
            poly.SetLines(lines)
            polylns.AddInputData(poly)
        polylns.Update()

        vedo.Mesh.__init__(self, polylns.GetOutput(), c, alpha)
        self.lw(lw).lighting('off')
        if dotted:
            self.GetProperty().SetLineStipplePattern(0xF0F0)
            self.GetProperty().SetLineStippleRepeatFactor(1)
        self.name = "[Lines]"


class SpheresExt(vedo.Mesh):
    """
    Build a set of spheres at `centers` of radius `r`.
    Either `c` or `r` can be a list of RGB colors or radii.
    """
    def __init__(self, centers, r=1, c="r", alpha=1, res=8):

        if isinstance(centers, vedo.Points):
            centers = centers.points()

        cisseq = False
        if vedo.utils.isSequence(c):
            cisseq = True

        if cisseq:
            if len(centers) > len(c):
                vedo.printc("\times Mismatch in Spheres() colors", len(centers), len(c), c='r')
                raise RuntimeError()
            if len(centers) != len(c):
                vedo.printc("\lightningWarning: mismatch in Spheres() colors", len(centers), len(c))

        risseq = False
        if vedo.utils.isSequence(r):
            risseq = True

        if risseq:
            if len(centers) > len(r):
                vedo.printc("times Mismatch in Spheres() radius", len(centers), len(r), c='r')
                raise RuntimeError()
            if len(centers) != len(r):
                vedo.printc("\lightning Warning: mismatch in Spheres() radius", len(centers), len(r))
        if cisseq and risseq:
            vedo.printc("\noentry Limitation: c and r cannot be both sequences.", c='r')
            raise RuntimeError()

        src = vtk.vtkSphereSource()
        if not risseq:
            src.SetRadius(r)
        if vedo.utils.isSequence(res):
            res_t, res_phi = res
        else:
            res_t, res_phi = 2*res, res

        src.SetThetaResolution(res_t)
        src.SetPhiResolution(res_phi)
        src.Update()

        psrc = vtk.vtkPointSource()
        psrc.SetNumberOfPoints(len(centers))
        psrc.Update()
        pd = psrc.GetOutput()
        vpts = pd.GetPoints()

        glyph = vtk.vtkGlyph3D()
        glyph.SetSourceConnection(src.GetOutputPort())

        if cisseq:
            glyph.SetColorModeToColorByScalar()
            ucols = vtk.vtkUnsignedCharArray()
            ucols.SetNumberOfComponents(3)
            ucols.SetName("colors")
            #for i, p in enumerate(centers):
            for cx, cy, cz in c:
                #cx, cy, cz = getColor(acol)
                ucols.InsertNextTuple3(cx * 255, cy * 255, cz * 255)
            pd.GetPointData().SetScalars(ucols)
            glyph.ScalingOff()
        elif risseq:
            glyph.SetScaleModeToScaleByScalar()
            urads = numpy_to_vtk(np.ascontiguousarray(2*r).astype(float), deep=True)
            urads.SetName("radii")
            pd.GetPointData().SetScalars(urads)

        vpts.SetData(numpy_to_vtk(np.ascontiguousarray(centers), deep=True))

        glyph.SetInputData(pd)
        glyph.Update()

        vedo.Mesh.__init__(self, glyph.GetOutput(), alpha=alpha)
        self.phong()

        self._polydata = pd

        if cisseq:
            self.mapper().ScalarVisibilityOn()
        else:
            self.mapper().ScalarVisibilityOff()
            self.GetProperty().SetColor(vedo.getColor(c))
        self.name = "[Spheres]"


def spherical_degree_angles_to_xyz(radius, theta, phi):
    return vedo.spher2cart(radius, theta / 180 * math.pi, phi / 180 * math.pi) 


def add_callback(plot, event_name, func, priority=0.0):
    """
    Modified function from vedo. The issue is that the way vedo (and pyvista for that matter)
    is structured is that it helps using vtk but sometimes hinders using it with code that makes
    assumptions we don't want.

    Add a function to be executed while show() is active.
    Information about the event can be acquired with method ``getEvent()``.
    Return a unique id for the callback.
    The callback function (see example below) exposes a dictionary
    Frequently used events are:
        - KeyPress, KeyRelease: listen to keyboard events
        - LeftButtonPress, LeftButtonRelease: listen to mouse clicks
        - MiddleButtonPress, MiddleButtonRelease
        - RightButtonPress, RightButtonRelease
        - MouseMove: listen to mouse pointer changing position
        - MouseWheelForward, MouseWheelBackward
        - Enter, Leave: listen to mouse entering or leaving the window
        - Pick, StartPick, EndPick: listen to object picking
        - ResetCamera, ResetCameraClippingRange
        - Error, Warning
        - Char
        - Timer
    Check the complete list of events here:
        https://vtk.org/doc/nightly/html/classvtkCommand.html
    """
    if not plot.interactor:
        return None

    # Processing names is removed from original function

    def _func_wrap(iren, ename):
        x, y = plot.interactor.GetEventPosition()
        plot.renderer = plot.interactor.FindPokedRenderer(x, y)
        if not plot.picker:
            plot.picker = vtk.vtkPropPicker()
        plot.picker.PickProp(x, y, plot.renderer)
        plot.picked2d = (x,y)
        xp, yp = plot.interactor.GetLastEventPosition()
        actor = plot.picker.GetProp3D()
        delta3d = np.array([0,0,0])
        if actor:
            picked3d = np.array(plot.picker.GetPickPosition())
            if actor.picked3d is not None:
                delta3d = picked3d - actor.picked3d
            actor.picked3d = picked3d
        else:
            picked3d = None

        dx, dy = x-xp, y-yp

        event_dict = vedo.utils.dotdict({
            "name": ename,
            "id": cid,
            "priority": priority,
            "at": plot.renderers.index(plot.renderer),
            "actor": actor,
            "picked3d": picked3d,
            "keyPressed": plot.interactor.GetKeySym(),
            "picked2d": (x,y),
            "delta2d": (dx, dy),
            "angle2d": np.arctan2(dy,dx),
            "speed2d": np.sqrt(dx*dx+dy*dy),
            "delta3d": delta3d,
            "speed3d": np.sqrt(np.dot(delta3d,delta3d)),
            "isPoints":   isinstance(actor, vedo.Points),
            "isMesh":     isinstance(actor, vedo.Mesh),
            "isAssembly": isinstance(actor, vedo.Assembly),
            "isVolume":   isinstance(actor, vedo.Volume),
            "isPicture":  isinstance(actor, vedo.Picture),
        })
        func(event_dict)
        return

    cid = plot.interactor.AddObserver(event_name, _func_wrap, priority)
    return cid


def get_file_name(file_name, extension):
    """
    Get full file name
    :param file_name: File name without extension
    :param extension: File extension
    :return: File name with extension
    """
    if str(file_name).endswith(extension):
        full_file_name = str(file_name)
    else:
        full_file_name = str(file_name) + '.' + str(extension)
    return full_file_name


def get_local_data_file_path(file_name, extension):
    """
    Get data path
    :param file_name: File name without extension
    :param extension: File extension
    :return: File path
    """
    return DATA_FOLDER.joinpath('./surfaces/' + get_file_name(file_name, extension))


def get_surface_mesh_path(file_name, meshes_path=None, extension='ply'):
    """
    Get a surface mesh file path
    :param file_name: File name without extension
    :param meshes_path: Folder path. If None given, this method will look into the data folder of iblviewer
    :param extension: File extension
    :return: Full mesh file path.
    """
    if meshes_path is None:
        region_mesh_path = str(get_local_data_file_path(file_name, extension))
        if not os.path.exists(region_mesh_path):
            region_mesh_path = 'https://raw.github.com/int-brain-lab/iblviewer/main/data/surfaces/'
            region_mesh_path += get_file_name(file_name, extension)
    else:
        region_mesh_path = str(os.path.join(meshes_path, get_file_name(file_name, extension)))
    return region_mesh_path


def load_surface_mesh(file_name, meshes_path=None, extension='ply'):
    """
    Load a surface mesh with vedo.
    :param file_name: File name without extension
    :param meshes_path: Folder path. If None given, this method will look into the data folder of iblviewer
    :param extension: File extension
    :return: Mesh or None if path is invalid
    """
    file_path = get_surface_mesh_path(file_name, meshes_path, extension)
    if file_path.startswith('https') or os.path.exists(file_path):
        return vedo.load(file_path)


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
    """
    Get a time difference in seconds
    :param t: Time
    return: Number of seconds
    """
    now  = datetime.now()
    duration = now - t
    return duration.total_seconds()


def get_actor_center(actor):
    """
    Get center position of an actor
    :param actor: VTK actor
    :return: 3d array
    """
    try:
        if isinstance(actor, vedo.Volume):
            return actor.center() + actor.pos()
        else:
            return actor.centerOfMass() + actor.pos() # TODO: check that this is necessary (adding pos)
    except Exception as e:
        raise e


def get_actor_dimensions(actor):
    """
    Get dimensions of an actor
    :param actor: VTK actor
    :return: 3d array
    """
    try:
        if isinstance(actor, vedo.Volume):
            return actor.dimensions() * actor.spacing()# equivalent to self.model.resolution
        else:
            xmin, xmax, ymin, ymax, zmin, zmax = actor.bounds()
            return np.array([xmax - xmin, ymax - ymin, zmax - zmin])
    except Exception as e:
        raise e


def get_transformation_matrix(origin, normal):
    """
    Get transformation matrix for a plane given by its origin and normal
    :param origin: Origin 3D vector
    :param normal: Normal 3D vector
    :return: Matrix and Translation
    """
    newaxis = vedo.utils.versor(normal)
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
                    size=(None,None), nlabels=None, horizontal=False, useAlpha=False):
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
            for r in vedo.shapes._reps:
                title = title.replace(r[0], r[1])
        titprop = sb.GetTitleTextProperty()
        titprop.BoldOn()
        titprop.ItalicOff()
        titprop.ShadowOff()
        titprop.SetColor(font_color)
        titprop.SetVerticalJustificationToTop()
        titprop.SetFontSize(titleFontSize)
        titprop.SetFontFamily(vtk.VTK_FONT_FILE)
        titprop.SetFontFile(vedo.settings.fonts_path + vedo.settings.defaultFont +'.ttf')
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
    sctxt.SetFontFile(vedo.settings.fonts_path + vedo.settings.defaultFont +'.ttf')
    sctxt.SetColor(font_color)
    sctxt.ItalicOff()
    sctxt.SetShadow(0)
    sctxt.SetFontSize(titleFontSize-2)
    sb.SetAnnotationTextProperty(sctxt)
    sb.PickableOff()
    return sb

def add_caption_symbol(point, size=0.5, color='red', alpha=1.0, overlay_2d=True):
    """
    Add a 2D or 3D overlay (aka caption in VTK).
    Modified from vedo caption() method
    """
    #c = np.array(self.GetProperty().GetColor())/2
    color = vedo.colors.getColor(color)

    """
    if point is None:
        x0,x1,y0,y1,z0,z1 = self.GetBounds()
        pt = [(x0+x1)/2, (y0+y1)/2, z1]
        point = self.closestPoint(pt)
    """

    caption = vtk.vtkCaptionActor2D()
    caption.SetAttachmentPoint(point)
    caption.SetBorder(False)
    caption.SetLeader(True)
    sph = vtk.vtkSphereSource()
    sph.Update()
    caption.SetLeaderGlyphData(sph.GetOutput())
    caption.SetLeaderGlyphSize(5)
    caption.SetMaximumLeaderGlyphSize(5)
    #capt.SetPadding(pad)
    #capt.SetCaption(txt)
    #capt.SetWidth(width)
    #capt.SetHeight(height)
    caption.SetThreeDimensionalLeader(not overlay_2d)

    prop = caption.GetProperty()
    prop.SetColor(color)
    prop.SetOpacity(alpha)
    """
    pr = caption.GetCaptionTextProperty()
    pr.SetFontFamily(vtk.VTK_FONT_FILE)
    if 'LogoType' in font: # special case of big file
        fl = vedo.io.download("https://vedo.embl.es/fonts/LogoType.ttf")
    else:
        fl = settings.fonts_path + font + '.ttf'
    if not os.path.isfile(fl):
        fl = font
    pr.SetFontFile(fl)
    pr.ShadowOff()
    pr.BoldOff()
    pr.FrameOff()
    pr.SetColor(c)
    pr.SetOpacity(alpha)
    pr.SetJustificationToLeft()
    if "top" in justify:
        pr.SetVerticalJustificationToTop()
    if "bottom" in justify:
        pr.SetVerticalJustificationToBottom()
    if "cent" in justify:
        pr.SetVerticalJustificationToCentered()
        pr.SetJustificationToCentered()
    if "left" in justify:
        pr.SetJustificationToLeft()
    if "right" in justify:
        pr.SetJustificationToRight()
    pr.SetLineSpacing(vspacing)
    self._caption = capt
    """
    return caption


def load_gltf_mesh(file_path='./data//brain_regions.glb'):
    """
    Read surface data from a binary GLTF scene
    """
    loader = vtk.vtkGLTFDocumentLoader()
    reader = vtk.vtkGLTFReader()
    reader.SetFileName(file_path)
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