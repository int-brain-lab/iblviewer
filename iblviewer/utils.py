from datetime import datetime
import numpy as np
import os
import glob
from pathlib import Path

from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy, numpy_to_vtkIdTypeArray
import vtk
import vedo
import math
#import trimesh


ROOT_FOLDER = Path(__file__).parent.parent
ASSETS_FOLDER = ROOT_FOLDER.joinpath('./iblviewer/assets')
FONTS_FOLDER = ASSETS_FOLDER.joinpath('./fonts')
EXAMPLES_FOLDER = ROOT_FOLDER.joinpath('./iblviewer_examples')
EXAMPLES_DATA_FOLDER = ROOT_FOLDER.joinpath('./iblviewer_examples/data')


def get_type(element):
    """
    Get the type of object as a string
    :return: String
    """
    return str(element.__class__.__name__).lower()


def get_unique_name(collection, name, spacer='_'):
    """
    Create a unique key for a collection by appending numbers when entries exist
    :param collection: A list, collection, array, ...
    :param name: Name (for instance 'Points')
    :param spacer: Spacer char
    :return: New name, for instance 'Points_4'
    """
    similar_ones = []
    max_value = 0
    if name not in collection:
        return name
    for key in collection:
        if name in key:
            similar_ones.append(key)
            if spacer in key:
                value = key.split(spacer)[1]
                max_value = max(int(value), max_value)
    value = max(len(similar_ones), max_value)
    return f'{name}{spacer}{value}'
    

def numpy2vtk(arr, dtype=None, deep=True, name=""):
    """
    Convert a numpy array into a vtkDataArray
    :param arr: Array
    :param dtype: Data type. Allows to set a specific data type to the VTK array
    :param deep: Whether a deep copy is made. Defaults to True
    :param name: Name of the array
    """
    if arr is None:
        return None
    arr = np.ascontiguousarray(arr)
    if dtype is not None and dtype!='id':
        arr = arr.astype(dtype)

    if dtype and dtype=='id':
        varr = numpy_to_vtkIdTypeArray(arr.astype(np.int64), deep=deep)
    else:
        varr = numpy_to_vtk(arr, deep=deep)
    if name:
        varr.SetName(name)
    return varr


def spherical_degree_angles_to_xyz(radius, theta, phi):
    """
    Convert spherical degree angles to XYZ coordinates
    :param radius: Radius
    :param theta: Theta angle value in degrees
    :param phi: Phi angle value in degrees
    :return: List of 3 coordinates
    """
    return vedo.spher2cart(radius, theta / 180 * math.pi, phi / 180 * math.pi) 


def pick_object(plot, event_name=None, priority=None, cid=None):
    """
    Pick an object
    """
    x, y = plot.interactor.GetEventPosition()
    plot.renderer = plot.interactor.FindPokedRenderer(x, y)
    if not plot.picker:
        plot.picker = vtk.vtkPropPicker()
    plot.picker.PickProp(x, y, plot.renderer)
    plot.picked2d = (x,y)
    xp, yp = plot.interactor.GetLastEventPosition()
    actor = plot.picker.GetProp3D()
    delta3d = np.array([0,0,0])
    picked3d = None
    picker = plot.picker
    if actor is None:
        # Ok, this is tricky. I found out that vtkPropPicker, even
        # if it optimized, can fail at detecting a simple mesh
        # so we use the vtkPicker as fall back plan
        picker = vtk.vtkPicker()
        picker.Pick(x, y, 0.0, plot.renderer)
        actor = picker.GetProp3D()
    if actor is not None:
        picked3d = np.array(picker.GetPickPosition())
        if isinstance(actor, vedo.Mesh):
            # There is a bug with transparent objects or objects that do not have ForceOpaqueOn()
            # which prevents picked3d from being valid so we have to use another picking method
            cell_picker = vtk.vtkCellPicker()
            cell_picker.Pick(x, y, 0.0, plot.renderer)
            if cell_picker.GetProp3D() == actor:
                picked3d = np.array(cell_picker.GetPickPosition())
        try:
            if actor.picked3d is not None:
                delta3d = picked3d - actor.picked3d
            actor.picked3d = picked3d
        except AttributeError:
            return
    else:
        actor = plot.picker.GetActor2D()

    dx, dy = x-xp, y-yp

    event_dict = vedo.utils.dotdict({
        "name": event_name,
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
        "isActor2D":  isinstance(actor, vtk.vtkActor2D)
    })
    return event_dict


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

    # Processing names is removed from vedo function
    # Also the whole thing is refactored with improved picking
    def wrapper(iren=None, event_name=None):
        event_dict = pick_object(plot, event_name, priority, cid)
        func(event_dict)

    cid = plot.interactor.AddObserver(event_name, wrapper, priority)
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


def get_local_data_file_path(file_name, extension, sub_folder=''):
    """
    Get data path
    :param file_name: File name without extension
    :param extension: File extension
    :return: File path
    """
    return ASSETS_FOLDER.joinpath(sub_folder, get_file_name(file_name, extension))


def get_surface_mesh_path(file_name, meshes_path=None, extension='ply', default_meshes_path=None):
    """
    Get a surface mesh file path
    :param file_name: File name without extension
    :param meshes_path: Folder path. If None given, this method will look into the data folder of iblviewer
    :param extension: File extension
    :param default_meshes_path: Fallback local or remote path
    :return: Full mesh file path
    """
    if meshes_path is None:
        region_mesh_path = str(get_local_data_file_path(file_name, extension, 'surfaces'))
        if not os.path.exists(region_mesh_path):
            if default_meshes_path is not None:
                region_mesh_path = default_meshes_path
            else:
                region_mesh_path = 'https://raw.github.com/int-brain-lab/iblviewer/main/assets/surfaces/'
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
    :return: Number of seconds
    """
    now  = datetime.now()
    duration = now - t
    return duration.total_seconds()


def recompute_normals(target):
    pdnorm = vtk.vtkPolyDataNormals()
    pdnorm.SetInputData(target)
    pdnorm.ComputePointNormalsOn()
    pdnorm.ComputeCellNormalsOn()
    pdnorm.FlipNormalsOff()
    pdnorm.ConsistencyOn()
    pdnorm.Update()
    return pdnorm.GetOutput() #self._data


def get_actor_center(actor):
    """
    Get the absolute center position of an actor
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


def get_actor_bounds(actor):
    """
    Get the bounds of an actor as xmin, xmax, ymin, ymax, zmin, zmax
    :param actor: VTK actor
    :return: Array with 6 values
    """
    if actor is None:
        return
    try:
        if isinstance(actor, vedo.Volume):
            d = actor.dimensions() * actor.spacing()
            c = get_actor_center(actor)
            return c[0] - d[0], c[0] + d[0], c[1] - d[1], c[1] + d[1], c[2] - d[2], c[2] + d[2]
        else:
            return actor.bounds()
    except Exception as e:
        raise e


def get_actor_dimensions(actor):
    """
    Get the dimensions of an actor
    :param actor: VTK actor
    :return: 3d array
    """
    if actor is None:
        return
    try:
        if isinstance(actor, vedo.Volume):
            return actor.dimensions() * actor.spacing()# equivalent to self.model.resolution
        else:
            xmin, xmax, ymin, ymax, zmin, zmax = actor.bounds()
            return np.array([xmax - xmin, ymax - ymin, zmax - zmin])
    except Exception as e:
        raise e


def get_bounding_planes(actor):
    """
    Get bounding planes for an actor
    :param actor: VTK actor
    :return: vtkPlanes
    """
    planes = vtk.vtkPlanes()
    planes.SetBounds(actor.GetBounds())
    return planes


def get_planes_bounds(planes):
    """
    Get the bounding box coordinates of a series of planes.
    [WARNING] Only works for six planes (box mode) at the moment
    :param planes: vtkPlaneCollection
    :return: 6 values
    """
    origins = list()
    for p_id in range(planes.GetNumberOfItems()):
        plane = planes.GetItem(p_id)
        origin = np.array(plane.GetOrigin())
        origins.append(origin)
    # We don't want zeros to be accounted for so we select planes of interest
    # First x planes, then y planes, then z ones. To be improved/generalized.
    origins = np.array(origins)
    mi_x = np.min(origins[:2], axis=0).tolist()
    ma_x = np.max(origins[:2], axis=0).tolist()
    mi_y = np.min(origins[2:4], axis=0).tolist()
    ma_y = np.max(origins[2:4], axis=0).tolist()
    mi_z = np.min(origins[4:6], axis=0).tolist()
    ma_z = np.max(origins[4:6], axis=0).tolist()
    return mi_x[0], ma_x[0], mi_y[1], ma_y[1], mi_z[2], ma_z[2]


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


def set_clipping_planes(target, planes, flip_normals=False):
    """
    Clip the volume and move the slicing planes according the given planes
    :param target: vedo.Mesh (can be iblviewer.objects.Points or similar)
    :param planes: vtkPlanes
    :param flip_normals: Whether clipping normals are flipped
    """
    if not isinstance(planes, vtk.vtkPlanes):
        bounds = planes
        planes = vtk.vtkPlanes()
        planes.SetBounds(bounds)
    if flip_normals:
        for plane_id in range(planes.GetNumberOfPlanes()):
            plane = planes.GetPlane(plane_id)
            plane.SetNormal(np.array(plane.GetNormal())*-1)
    target.GetMapper().SetClippingPlanes(planes)


def probe(plot, target, widget=None, interaction_callback=None, point1=None, point2=None, 
            place_factor=1, handle_size=0.005, color=None):
    """
    Initializes a line widget on the given target
    :param plot: vtk plot
    :param target: Target object
    :param widget: Existing widget. In case a valid vtkLineWidget is given,
    it will be used and modified directly
    :param interaction_callback: Function that will be called every
    time there is an interaction with the widget. That's where
    you set the clipping planes to the object for instance
    :param point1: Initial position of point 1
    :param point2: Initial position of point 2
    :param place_factor: see vtkBoxWidget.setPlaceFactor()
    :param handle_size: set the relative handle size, see vtkBoxWidget.SetHandleSize()
    :param color: Color of the line
    :return: vtkLineWidget
    """
    existing = isinstance(widget, vtk.vtkLineWidget)
    if not existing:
        widget = vtk.vtkLineWidget()
        widget.SetInteractor(plot.interactor)
        widget.SetPlaceFactor(place_factor)
        widget.SetHandleSize(handle_size)
    widget.SetInputData(target.GetMapper().GetInput())
    
    if color is None:
        color = [0.5, 0.5, 0.5]
    widget.GetSelectedLineProperty().SetColor(*color)
    #widget.GetSelectedLineProperty().SetOpacity(0.7)
    if point1 is not None and point2 is not None:
        widget.PlaceWidget()
        widget.SetPoint1(*point1)
        widget.SetPoint2(*point2)
    else:
        widget.PlaceWidget(target.GetBounds())

    if interaction_callback is not None:
        widget.RemoveObservers('InteractionEvent')
        widget.AddObserver('InteractionEvent', interaction_callback)

    plot.interactor.Render()
    widget.On()
    plot.widgets.append(widget)
    return widget


def box_widget(plot, target, interaction_callback=None, place_factor=1, 
                handle_size=0.005, outline_color=None):
    """
    Initializes a box widget on the given target
    :param plot: vtk plot
    :param target: Target object
    :param interaction_callback: Function that will be called every
        time there is an interaction with the widget. That's where
        you set the clipping planes to the object for instance
    :param place_factor: see vtkBoxWidget.setPlaceFactor()
    :param handle_size: set the relative handle size, see vtkBoxWidget.SetHandleSize()
    :param outline_color: Color of the outline
    :return: vtkBoxWidget
    """
    widget = vtk.vtkBoxWidget()
    widget.SetInteractor(plot.interactor)
    widget.SetPlaceFactor(place_factor)
    widget.SetHandleSize(handle_size)
    # TODO: handle the update event in volumes in order to choose
    # the best method, either axis-aligned slicing when normals
    # are axis-aligned, or slicing with vtkImageReslice. 
    # Both functionalities are available in VolumeView already.
    widget.RotationEnabledOff()
    widget.ScalingEnabledOn()
    widget.TranslationEnabledOn()
    widget.SetInputData(target.GetMapper().GetInput())
    plot.cutterWidget = widget
    
    # Only valid for vtkBoxWidget
    widget.OutlineCursorWiresOn()
    widget.InsideOutOn()
    widget.GetSelectedOutlineProperty().SetColor(1, 0, 1)
    if outline_color is None:
        outline_color = [0.5, 0.5, 0.5]
    widget.GetOutlineProperty().SetColor(*outline_color)
    #widget.GetOutlineProperty().SetOpacity(0.7)

    #widget.SetRepresentationToOutline()
    existing_planes = target.GetMapper().GetClippingPlanes()
    if existing_planes is not None:
        try:
            bounds = get_planes_bounds(existing_planes)
            widget.PlaceWidget(bounds)
        except Exception:
            msg = '[Warning] Object ' + target.name + ' does not have six clipping planes.'
            msg += 'Placing widget is not supported atm in this configuration.'
            print(msg)
            widget.PlaceWidget(target.GetBounds())
    else:
        widget.PlaceWidget(target.GetBounds())
    
    def clip_target(widget=None, event=None):
        """
        Clip the target with the current box widget
        """
        if widget is None:
            return
        clipping_planes = vtk.vtkPlanes()
        widget.GetPlanes(clipping_planes)
        target.GetMapper().SetClippingPlanes(clipping_planes)

    if interaction_callback is None:
        interaction_callback = clip_target
    
    widget.AddObserver('InteractionEvent', interaction_callback)
    plot.interactor.Render()
    widget.On()
    plot.widgets.append(widget)
    return widget


def update_scalar_bar(sb, lut, use_alpha=False, nan_color=None):
    """
    Update a scalar bar with a new LUT
    :param sb: vtkScalarBarActor
    :param lut: vtkLookupTable
    :param use_alpha: whether alpha is used in the scalar bar
    """
    if sb.GetLookupTable() == lut:
        return
    sb.SetLookupTable(lut)
    sb.SetUseOpacity(use_alpha)
    sb.SetDrawFrame(0)
    sb.SetDrawBackground(0)
    if lut.GetUseBelowRangeColor():
        sb.DrawBelowRangeSwatchOn()
        sb.SetBelowRangeAnnotation('')
    if lut.GetUseAboveRangeColor():
        sb.DrawAboveRangeSwatchOn()
        sb.SetAboveRangeAnnotation('')
    if nan_color is not None:#lut.GetNanColor() != (0.5, 0.0, 0.0, 1.0):
        lut.SetNanColor(*nan_color)
        sb.DrawNanAnnotationOn()
        sb.SetNanAnnotation('nan')


def add_scalar_bar(lut, pos=(0.8, 0.05), font_color=[0, 0, 0], title="", titleYOffset=15, titleFontSize=12,
                    size=(None,None), nlabels=None, horizontal=False, use_alpha=False):
    """
    Create a new 2D scalar bar. This is a modified method from vedo.addons.addScalarBar
    :param lut: Color map LUT
    :param list pos: fractional x and y position in the 2D window
    :param list size: size of the scalarbar in pixel units (width, heigth)
    :param int nlabels: number of numeric labels to be shown
    :param bool use_alpha: retain trasparency in scalarbar
    :param bool horizontal: show in horizontal layout
    """
    if isinstance(font_color, str):
        font_color = vedo.getColor(font_color)
    sb = vtk.vtkScalarBarActor()
    sb.SetLabelFormat('%-#6.4g')
    #print(sb.GetLabelFormat())
    sb.SetLookupTable(lut)
    sb.SetUseOpacity(use_alpha)
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