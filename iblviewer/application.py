from dataclasses import dataclass, field
from typing import Any
from datetime import datetime
from collections import OrderedDict
import logging
import math
import numpy as np
from vedo.colors import colorMap

import vtk
import vedo

from iblviewer.volume import Volume, VolumeModel, VolumeController
from iblviewer.collection import Collection
import iblviewer.objects as obj
import iblviewer.utils as utils


# A camera model does not necessarily reflect the view. It's a state that can be set
# to a view. It's also a state that can be retrieved from the view.
@dataclass
class CameraModel:
    camera: vtk.vtkCamera = None
    autofocus: bool = True
    target: Any = None
    distance_factor: int = 3
    orthographic_scale_factor: int = 1


@dataclass
class UIModel:
    DEFAULT = 'default'
    CAMERA = 'camera'
    SCENE = 'scene'
    OBJECT = 'object'
    DATA = 'data'
    EXPORT = 'export'

    visible: bool = True
    embed: bool = False
    embed_menu_x: int = 40
    embed_menu_y: int = 50
    embed_submenu_x: int = 200
    embed_submenu_y: int = 50
    embed_slider_width: int = 250

    color: Any = '#000000'
    background_color: Any = '#ffffff'

    contexts = {DEFAULT: {}, CAMERA: {}, SCENE: {}, OBJECT: {}, DATA: {}, EXPORT: {}}

    # Font license is under https://raw.github.com/int-brain-lab/iblviewer/main/assets/fonts
    font_path = str(utils.FONTS_FOLDER)
    font = 'SourceSansPro-Regular'
    #font = 'https://raw.github.com/int-brain-lab/iblviewer/main/assets/fonts/SourceSansPro-Regular.ttf'

    # VTK is a nightmare for having a basic UI working...
    # and we need one when running in standalone or jupyter modes.
    font_size = 15
    toggle_config = {'c':['black', '#bbbbbb'], 
                    'bc':['#dddddd', '#ffffff'], 
                    'font':font, 'size':font_size, 
                    'bold':False, 'italic':False}
    button_config = {'c':['black', 'black'], 
                    'bc':['#dddddd', '#dddddd'], 
                    'font':font, 'size':font_size, 
                    'bold':False, 'italic':False}
    slider_config = {'font':font, 'c':'black'}

    context: str = DEFAULT
    context_changed: bool = False
    last_context: str = None
    main_context: str = None
    dark_mode: bool = False

    def set_dark_mode(self, on=True):
        self.toggle_config['c'] = ['#eeeeee', '#eeeeee'] if on else ['#111111', '#111111']
        self.toggle_config['bc'] = ['#444444', '#777777'] if on else ['#cfcfcf', '#dfdfdf']
        self.button_config['c'] = ['#eeeeee', '#eeeeee'] if on else ['#111111', '#111111']
        self.button_config['bc'] = ['#444444', '#777777'] if on else ['#cfcfcf', '#dfdfdf']
        self.slider_config['c'] = 'white' if on else 'black'
        self.color = '#eeeeee' if on else '#222222'
        self.background_color = '#2d2d2d' if on else '#eeeeee'
        self.dark_mode = on

    def all_elements(self):
        elements = dict()
        for context_id in self.contexts:
            elements.update(self.get_elements(context_id))
        return elements

    def get_elements(self, context=None):
        if context is None:
            context = self.context
        return self.contexts.get(context)

    def get_element(self, name, context=None):
        return self.get_elements(context).get(name)

    def is_camera_context(self):
        return self.context == UIModel.CAMERA

    def is_scene_context(self):
        return self.context == UIModel.SCENE

    def is_object_context(self):
        return self.context == UIModel.OBJECT

    def is_data_context(self):
        return self.context == UIModel.DATA

    def is_export_context(self):
        return self.context == UIModel.EXPORT

    def is_valid_context(self, context):
        return context in self.contexts

    def toggle_context(self, context=None):
        """
        Toggles between current context and given one
        :param context: A context string
        """
        if context is None or self.main_context is not None:
            if self.main_context == context:
                self.context_changed = False
                return
            self.context = self.main_context
            self.main_context = None
            self.context_changed = True
        else:
            if self.context == context or not self.is_valid_context(context):
                self.context_changed = False
                return
            self.main_context = self.context
            self.context = context
            self.context_changed = True

    def set_context(self, context):
        """
        Set the context
        :param context: A context from CONTEXTS
        :return: True if context was changed
        """
        if self.is_valid_context(context) and context != self.context:
            self.last_context = self.context
            self.context = context
            self.context_changed = True
            return True
        else:
            #logging.error('Context ' + str(context) + ' is invalid. Ignoring it.')
            self.context_changed = False
            return False

    def register(self, element, context=None):
        """
        Register UI components (or anything else really) to the given context
        """
        if context is None:
            context = self.context
        if not self.is_valid_context(context):
            return
        if isinstance(element, list):
            for item in element:
                self.contexts[context][item.name] = item
        else:
            self.contexts[context][element.name] = element
        

@dataclass
class ViewerModel:
    POINT = 'point'
    LINE = 'line'
    SURFACE = 'surface'
    VOLUME = 'volume'
    UI = 'ui'
    OBJECTS_TYPE = {POINT:0, LINE:1, SURFACE:2, VOLUME:3, UI:4}

    # Global application camera presets
    X_UP = np.array([1.0, 0.0, 0.0])
    Y_UP = np.array([0.0, 1.0, 0.0])
    Z_UP = np.array([0.0, 0.0, 1.0])
    X_DOWN = np.array([-1.0, 0.0, 0.0])
    Y_DOWN = np.array([0.0, -1.0, 0.0])
    Z_DOWN = np.array([0.0, 0.0, -1.0])

    title: str = 'VTK Viewer'
    window_size: np.ndarray = np.array([1280, 720])
    web_window_size: np.ndarray = np.array([960, 600])
    web_ui: bool =  False
    offscreen: bool = False

    ui: UIModel = field(default_factory=UIModel)

    origin: np.ndarray = np.array([0.0, 0.0, 0.0])

    cameras: Collection = field(default_factory=Collection)
    # VolumeModel instances have their own SlicerModel instances (ie. box slicing)
    volumes: Collection = field(default_factory=Collection)
    # This set is used only by independent SlicerModel instances
    slicers: Collection = field(default_factory=Collection)
    points: Collection = field(default_factory=Collection)
    lines: Collection = field(default_factory=Collection)
    surfaces: Collection = field(default_factory=Collection)
    probes: Collection = field(default_factory=Collection)
    luts: Collection = field(default_factory=Collection)

    # Optimization for faster interactivity with volumes
    interactive_volume_subsampling: bool = False

    runtime: datetime = datetime.now()
    timer_id: Any = None
    playback_speed: int = 20 #ms
    animation_playing: bool = False
    animation_function: Any = None
    time: float = 0.0

    selection_changed: bool = False
    selection_point_changed: bool = False
    selection: Any = None
    selection_type: Any = None
    selection_point: Any = None # Point relative to absolute origin in VTK
    selection_relative_point: Any = None # Point relative to the origin you define
    selection_controller: Any = None
    selection_value: float = None
    selection_related_value: float = None # Custom value that you assign to a volume region (optional)
    selection_related_name: str = None # Related name to the custom value
    selection_camera_position: Any = None

    selection_marker_visible: bool = True
    selection_text_visible: bool = True
    selection_outline_visible: bool = True
    selection_color_bar_visible: bool = True

    probe_initial_point1: Any = None
    probe_initial_point2: Any = None
    
    outline_opacity: float = 1.0
    outline_color: list = field(default_factory=lambda:[0.5, 0.5, 0.5])
    outline_width: int = 3

    pickings: list = field(default_factory=list)
    first_picked_position: np.ndarray = None
    last_picked_position: np.ndarray = None

    video_start_angle: float = 0.0
    video_end_angle: float = 360.0
    video_duration: float = 8.0

    global_slicer: vtk.vtkPlaneCollection = None

    def initialize(self):
        """
        Initialize the model
        """
        # We always want to have at least a camera
        self.cameras.store(CameraModel())

    def get_actor_type(self, actor):
        """
        Get the type of the given actor
        Vedo does a similar thing for some of its events:
        "isPoints":   isinstance(actor, vedo.Points),
        "isMesh":     isinstance(actor, vedo.Mesh),
        "isAssembly": isinstance(actor, vedo.Assembly),
        "isVolume":   isinstance(actor, vedo.Volume),
        "isPicture":  isinstance(actor, vedo.Picture),
        "isActor2D":  isinstance(actor, vtk.vtkActor2D),
        :param actor: vtkActor
        :return: String
        """
        qualified_type = str(type(actor)).lower()
        obj_type = None
        if isinstance(actor, vedo.Mesh) and ViewerModel.LINE in qualified_type:
            obj_type = ViewerModel.LINE
        elif isinstance(actor, vedo.Mesh):
            obj_type = ViewerModel.SURFACE
        elif isinstance(actor, vedo.Points) or isinstance(actor, vedo.Spheres) or isinstance(actor, obj.Points):
            obj_type = ViewerModel.POINT
        elif isinstance(actor, Volume) or isinstance(actor, vedo.Volume):
            obj_type = ViewerModel.VOLUME
        return obj_type

    def got_selection(self):
        """
        Whether a selection is set
        :return: Boolean
        """
        return self.selection is not None

    def got_new_selection(self):
        """
        Whether a the selection target changed
        :return: Boolean
        """
        return self.selection_changed and self.got_selection()
        
    def is_selection_point(self):
        """
        Whether the type of selection is a point cloud
        :return: Boolean
        """
        return self.selection_type == ViewerModel.POINT

    def is_selection_line(self):
        """
        Whether the type of selection is a set of lines
        :return: Boolean
        """
        return self.selection_type == ViewerModel.LINE

    def is_selection_surface(self):
        """
        Whether the type of selection is a surface mesh
        :return: Boolean
        """
        return self.selection_type == ViewerModel.SURFACE

    def is_selection_volume(self):
        """
        Whether the type of selection is a volume
        :return: Boolean
        """
        return self.selection_type == ViewerModel.VOLUME

    def define_view_axis(self, normal):
        if abs(normal[0]) < 0.05:
            return ViewerModel.X_UP
        elif abs(normal[1]) < 0.05:
            return ViewerModel.Y_UP
        elif abs(normal[2]) < 0.05:
            return ViewerModel.Z_UP


class Viewer():
    """
    This class is the container and context for having 
    an interactive VTK viewer with a controllable scene
    and objects.
    """

    def __init__(self):
        """
        Constructor
        """
        #logging.basicConfig(level=logging.DEBUG)
        self.plot = None
        self.plot_window_id = 0
        self.num_windows = 1
        self.model = ViewerModel()
        self.model.initialize()
        #self.views = Collection()
        self.objects = OrderedDict()
        self.controllers_map = OrderedDict()

        self.selection_marker = None
        self.selection_info = None
        self.color_bar = None

        self.last_mouse_press_position = None
        self.last_mouse_release_position = None
        self.left_mouse_down = False
        self.last_mouse_time = datetime.now()
        self.last_time_series_value = None

        # UI related
        self.outline_actor = None
        self.axes_assembly = None
        self.widgets_reflection = dict()
        self.widgets = dict()
        self.box_widget = None
        self.line_widget = None

        self.depth_picking = OrderedDict()
        self.depth_peeling_enabled = True

        vedo.settings.useDepthPeeling = True
        vedo.settings.useFXAA = True
        vedo.settings.multiSamples = 0
        vedo.settings.alphaBitPlane = 1
        vedo.settings.immediateRendering = False
        # This line is necessary as for some reason, 
        # vedo's allowInteraction kills the app when we listen to TimerEvents!
        vedo.settings.allowInteraction = False
        vedo.settings.fonts_path = self.model.ui.font_path
        vedo.settings.defaultFont = self.model.ui.font
        font_params = {'islocal':True}
        if vedo.settings.font_parameters is None:
            vedo.settings.font_parameters = {}
        vedo.settings.font_parameters[self.model.ui.font] = font_params
        vedo.settings.enableDefaultKeyboardCallbacks = False

    def initialize(self, offscreen=False, jupyter=False, embed_ui=False, embed_font_size=15, 
                    plot=None, plot_window_id=0, num_windows=1, dark_mode=False, silent=False):
        """
        Initialize the controller, main entry point to the viewer
        :param context: Context of the visualization
        :param embed_ui: Whether the UI is embed within the VTK window
        :param embed_font_size: Embed font size. Defaults to 16 points. You might need larger values
            in case you have a small screen with high dpi (but VTK methods fail to detect that).
        :param jupyter: Whether we're running from a jupyter notebook or not
        :param offscreen: Whether the rendering is done offscreen (headless) or not
        :param plot: A vedo Plotter instance. You can either create it by yourself before hand, in case 
            you want to have multiple windows with other stats or let the controller create a new one
        :param plot_window_id: Sub-window id where the 3D visualization will be displayed
        :param num_windows: Number of subwindows, in case you want to display your own stuff later
        :param render: Whether rendering occurs at the end of the initialization or not. Defaults to False
        :param dark_mode: Whether the viewer is in dark mode
        :param auto_select_first_object: Auto select the first object displayed
        :param silent: Whether printing to console is disabled or not
        """
        self.silent = silent
        if not silent:
            print('IBL Viewer...')

        self.model.ui.font_size = embed_font_size
        self.model.web_ui = jupyter
        if self.model.web_ui:
            vedo.embedWindow('ipyvtk')

        self.plot = plot
        window_size = self.model.web_window_size if self.model.web_ui else self.model.window_size
        if plot is None:
            self.plot = vedo.Plotter(N=num_windows, size=window_size, 
                                    title=self.model.title, 
                                    bg=self.model.ui.background_color,
                                    offscreen=offscreen or jupyter)
        self.plot_window_id = plot_window_id
        self.plot.window.SetSize(*window_size)

        self.set_renderer()
        self.set_dark_mode(dark_mode, False)
        '''
        if jupyter:
            self.plot.offscreen = offscreen if offscreen is not None else offscreen is None
            self.plot.window.SetOffScreenRendering(self.plot.offscreen)
        '''
        try:
            # In cases where we are headless, this will fail because there is no interactor
            # but in cases where a headless window is embed into an app like Qt, this will work
            # So everything's fine like this
            self.initialize_window_interactions()
        except Exception:
            pass
        
        self.model.ui.embed = embed_ui
        if self.model.ui.embed:
            self.model.ui.visible = True
            self.model.ui.set_context(UIModel.DEFAULT)
            self.initialize_vtk_ui(self.model.ui.embed_menu_x, self.model.ui.embed_menu_y)
            self.update_ui()
        
        # Generic UI part: prepares the cursor and draws a box outline on the selected object
        self.initialize_selection_ui()

        self.initialized()
        #logging.info('Initialization complete.')
    
    def initialize_window_interactions(self):
        """
        Initialize window interactions on the VTK window
        """
        # LeftButtonReleaseEvent does not work. You have to use EndInteractionEvent instead (go figure...)
        # see https://stackoverflow.com/questions/52209534
        self.add_callback('LeftButtonPressEvent', self.handle_left_mouse_press)
        self.add_callback('EndInteractionEvent', self.handle_left_mouse_release)
        
        self.plot.interactor.AddObserver('TimerEvent', self.handle_timer)
        self.plot.interactor.AddObserver('KeyPressEvent', self.handle_key_press)

    def initialize_selection_ui(self):
        """
        Initialize selection marker and text
        """
        #if self.model.ui.embed:
        self.selection_info = self.add_text('selection_info', '', [0.02, 0.95], 
                                            color=self.model.ui.color)
        self.plot.add(self.selection_info, render=False)
        self.set_selection_marker()

    def handle_key_press(self, iren, event):
        """
        Handle key press events
        :param iren: vtk iren
        :param event: vtk event
        """
        key = iren.GetKeySym().lower()
        if 'esc' in key:
            iren.ExitCallback()
        elif 'space' in key:
            self.clear_line_widget()
            self.clear_box_widget()

    def exit_interactive_mode(self):
        """
        Exit/leave interactive mode
        """
        self.plot.window.ExitCallback()

    def toggle_dark_mode(self):
        """
        Toggle dark/light mode
        """
        self.set_dark_mode(not self.model.ui.dark_mode)

    def set_dark_mode(self, on=True, update_ui=True):
        """
        Set dark mode on or off
        """
        self.model.ui.set_dark_mode(on)
        if self.plot is not None:
            self.plot.background(self.model.ui.background_color)
        if self.color_bar is not None:
            self.update_element_color(self.color_bar)
        if not update_ui:
            return
        if self.model.ui.embed:
            all_elements = self.model.ui.all_elements()
            for key in all_elements:
                element = all_elements[key]
                self.update_element_color(element)

    def set_renderer(self):
        """
        Set VTK renderer, attempts to use OSPRay, if available
        OSPRay is not supported (2021) by default and there is no
        pip wheel for it with vtk, or paraview or any vtk-based tool.
        So you can only rely on OSPRay if you compile it alongside VTK.
        """
        renderer = self.plot.renderer
        try:
            ospray_pass= vtk.vtkOSPRayPass()
            renderer.SetPass(ospray_pass)

            node = vtk.vtkOSPRayRendererNode()
            node.SetSamplesPerPixel(4,renderer)
            node.SetAmbientSamples(4,renderer)
            node.SetMaxFrames(4, renderer)
        except (AttributeError, ImportError, NameError):
            pass

        # For some reason, depth peeling is sometimes not activated
        # so it's further made active in volume.SlicerView.initialize_mapper()
        # If you don't know about depth peeling: https://vtk.org/Wiki/VTK/Depth_Peeling
        for renderer in self.plot.renderers:
            renderer.UseDepthPeelingForVolumesOn()
            renderer.SetOcclusionRatio(0.001)
            renderer.SetMaximumNumberOfPeels(100)

    def register_object(self, vtk_object, name=None, selectable=True):
        """
        Register an object as selectable by the user in the UI
        :param vtk_object: VTK object
        :param name: Name or given id. IF None, the name of the
        object is used
        """
        if name is None:
            name = vtk_object.name
        existing_obj = self.objects.get(name)
        if existing_obj != vtk_object and name in self.objects:
            # Then we have two same names for two different objects
            # let's change that
            name = self.get_unique_object_name(name)
        self.objects[name] = vtk_object
        if selectable:
            vtk_object.SetPickable(True)
        # We overwrite the vtk object's name with the new one
        vtk_object.name = name
        self.update_selection_slider()
        self.objects_changed()

    def get_unique_object_name(self, name, spacer='_'):
        """
        Get a unique key/name for selectable objects with similar names
        :param name: Name (for instance 'Points')
        :param spacer: Spacer char
        :return: New name, for instance 'Points_4'
        """
        return utils.get_unique_name(self.objects, name, spacer)

    def unregister_object(self, name):
        """
        Unregister an object from the selectable objects list
        :param name: Object name or given id or int or the object itself
        """
        if isinstance(name, int):
            keys = list(self.objects.keys())
            try:
                name = keys[name]
            except Exception:
                pass
        elif not isinstance(name, str):
            for key in self.objects:
                if name == self.objects[key]:
                    name = key
                    break
        del self.objects[name]
        self.update_selection_slider()

    def update_selection_slider(self, max_value=None):
        """
        Update the selection slider max value
        :param max_value: Max value. If None, the maximum value
        is the length of the self.objects
        """
        if max_value is None:
            max_value = len(self.objects)-1
        slider = self.widgets.get('selection')
        if slider is not None:
            slider.GetRepresentation().SetMinimumValue(0)
            slider.GetRepresentation().SetMaximumValue(max_value)

    def register_controller(self, controller, vtk_objects):
        """
        Register/map VTK objects to a view
        :param controller: A controller instance (like VolumeController)
        :param vtk_objects: All objects directly related to this view
        """
        self.register_object(controller.actor)
        if not isinstance(vtk_objects, list):
            vtk_objects = [vtk_objects]
        for obj in vtk_objects:
            self.controllers_map[obj] = controller

    def get_view_objects(self):
        """
        Get all view objects registered in this model
        :return: List
        """
        return list(self.controllers_map.keys())

    def get_view_objects_names(self):
        """
        Get view objects names
        :return: Dict
        """
        names = {}
        for obj in self.controllers_map:
            names[obj.name] = obj
        return names

    def get_window(self):
        """
        Get the plot window object. This is useful for displaying this window 
        in Jupyter notebooks for instance
        :return: iren window object
        """
        return self.plot.window

    def render(self, save_to_file=None, width=None, height=None, scale=1):
        """
        Render the current state of the viewer, optionally to a file.
        Supported formats are jpg, png, pdf, svg, eps
        :param save_to_file: File path
        :param width: Width of the rendered image
        :param height: Height of the rendered image
        :param scale: Rendering scale factor. Defaults to 1
        """
        self.plot.render()
        if save_to_file is not None:
            custom_size = isinstance(width, int) and isinstance(height, int)
            if custom_size:
                current_width, current_height = self.plot.window.GetSize()
                # Set the desired size
                self.plot.window.SetSize(width, height)
                self.plot.show()
            if scale > 1:
                # Apparently it's better to enable the setting below when we use
                # scaling for larger size rendering.
                # There's a typo in the variable name and even the variable name
                # is inappropriate. This isn't a screenshot but a render. (@marcomusy)
                # So I put that in a try catch just in case it changes in the future...
                try:
                    vedo.settings.screeshotLargeImage = True
                except Exception:
                    pass
            vedo.screenshot(save_to_file, scale)
            if custom_size:
                # Now reset the size as it was
                self.plot.window.SetSize(current_width, current_height)
                self.plot.show()

    def show(self, interactive=True, actors=None, at=0, **kwargs):
        """
        Render the plot and let the user interact with it
        :param interactive: Whether we render and make the window interactive
        :param actors: List of actors to show. Use this parameter only if you know what you're doing.
        :param at: Which VTK window to use. Defaults to 0
        """
        if actors is not None:
            actors_to_show = actors
        else:
            actors_to_show = self.plot.actors

        if self.model.selection is None:
            self.select(-1)
            #self.view_selected()
        
        if not interactive:
            self.plot.render()

        if self.model.web_ui:
            logging.info(f'\nVisualizer started in Web UI mode: ' + str(utils.time_diff(self.model.runtime)) + 's\n')
            return self.plot.show(actors_to_show, at=at, resetcam=False, interactive=interactive, **kwargs)
            #return ViewInteractiveWidget(self.plot.window)
        else:
            logging.info('\nVisualizer started: ' + str(utils.time_diff(self.model.runtime)) + 's\n')
            #self.plot.window.SetWindowName()
            return self.plot.show(actors_to_show, at=at, resetcam=False, interactive=interactive, **kwargs)

    def close(self):
        """
        Close the current plot
        """
        self.plot.close()

    def add_callback(self, event_name, func, priority=0.0):
        """
        Add an event listener (aka callback method)
        :param event_name: A VTK event name
        :param func: Listener function
        :param priority: Priority in event queue
        :return: Callback id
        """
        return utils.add_callback(self.plot, event_name, func, priority)

    def remove_callback(self, callback_id):
        """
        Add an event listener (aka callback method)
        :param callback_id_or_event_name: A VTK event name
        """
        self.plot.interactor.RemoveObserver(callback_id)

    def handle_timer(self, iren, event, **kwargs):
        """
        Handle vtk timer event
        :param iren: Render window
        :param event: TimerEvent
        """
        if self.model.animation_playing:
            valid = self.next_time_series()
            if not valid:
                self.plot.interactor.DestroyTimer(self.model.timer_id)

    def handle_left_mouse_press(self, event):
        """
        Handle left mouse down event. The event is triggered for any "pickable" object
        :param mesh: Selected object
        """
        if event is None:
            return
        self.last_mouse_press_position = np.array(event.picked2d)
        self.left_mouse_down = True

        # 1. Make sure the click occured on a vtk actor
        actor = event.actor
        return actor is not None

    def clear_depth_picking(self, return_element=None):
        """
        Clear depth picking, making all objects put in depth queue pickable again
        :param return_element: An element to be returned by its index of insertion
        :return: object and picked 3d coordinates (only if return_element is valid)
        """
        picked3d = None
        iter_id = 0
        for obj in self.depth_picking:
            if return_element == iter_id:
                picked3d = self.depth_picking[obj]
            obj.SetPickable(True)
            iter_id += 1
        self.depth_picking = OrderedDict()
        if return_element:
            return obj, picked3d

    def get_actor_type(self, actor):
        """
        Get what type is the given actor
        :param actor: vtkActor
        :return: String
        """
        return self.get_actor_type(actor)

    def handle_left_mouse_release(self, event):
        """
        Handle left mouse up event.
        :param event: vtkEvent
        :return: Int value. 0 for invalid click, 1 for actor click, 
        2 for double click on actor
        """
        state = 0
        # 1. Make sure it's left mouse context
        if not self.left_mouse_down:
            return state

        # Handle mouse click release with a workaround for bad 
        # VTK event mgmt (LeftButtonReleaseEvent does not work)
        self.left_mouse_down = False

        if event is None:
            return

        drag_delta = np.linalg.norm(self.last_mouse_press_position - event.picked2d)
        # 2. If delta is larger than a few pixels, it means there was a click and drag
        # which is not considered as a single click
        actor = event.actor
        if drag_delta > 4:
            self.clear_depth_picking()
            self.handle_drag_release(state, actor, event)
            return state

        if event.isActor2D:
            self.handle_actor2d_click(state, actor, event)
            return
        
        picked3d = event.picked3d
        if self.last_mouse_release_position is not None:
            release_delta = np.linalg.norm(self.last_mouse_release_position - event.picked2d)
            if release_delta > 2:
                self.clear_depth_picking()
        self.last_mouse_release_position = np.array(event.picked2d)

        # 3. Make sure the click occured on a vtk actor
        if actor is None:
            self.handle_void_click(state, event)
            return state

        # 4. If we click on the same spot, maybe we want to select another object behind that one
        camera_position = self.plot.camera.GetPosition()
        actor_type = self.model.get_actor_type(actor)
        if actor_type == ViewerModel.VOLUME:
            controller = self.controllers_map.get(actor)
            if controller is not None:
                # Volume picking often results in a wrong coordinate so we compute it properly here
                picked3d, value = controller.pick(camera_position, np.array(event.picked2d))
        if actor in self.depth_picking:
            # First disable temporarily the selected actor
            actor.SetPickable(False)

            # This gets needlessly complicated but there's no other way with VTK...
            # So, the standard vtk.vtkPropPicker() that is hardware accelerated works
            # for points, lines and partly for volumes (which is why we have the pick()
            # method in VolumeController to readjust that).
            # Now, vtkPropPicker fails to work on meshes that do not have ForceOpaqueOn()
            # and obviously we want to keep transparency enabled so if we want to pick
            # a transparent mesh, we have to use vtkCellPicker and then compare results...
            cell_picker = vtk.vtkCellPicker()
            picker = self.plot.picker # this is a vtk.vtkPropPicker()
            
            x, y = event.picked2d
            picker.Pick(x, y, 0.0, self.plot.renderer)
            cell_picker.Pick(x, y, 0.0, self.plot.renderer)
            valid_result = True
            if picker.GetProp3D() is None and cell_picker.GetProp3D() is None:
                valid_result = False
            elif picker.GetProp3D() is not None and picker.GetProp3D() != actor:
                actor = picker.GetProp3D()
            elif cell_picker.GetProp3D() is not None:
                actor = cell_picker.GetProp3D()
                picker = cell_picker
            else:
                valid_result = False

            if not valid_result:
                result = self.clear_depth_picking(0)
                if result is None:
                    actor.SetPickable(True)
                else:
                    actor, picked3d = result

            last = picked3d
            picked3d = np.array(picker.GetPickPosition())
            if np.linalg.norm(last - picked3d) > 0:
                self.clear_depth_picking()
            try:
                actor.picked3d = picked3d
            except AttributeError:
                pass
            event['picked3d'] = picked3d
            delta3d = np.array([0,0,0])
            try:
                if actor.picked3d is not None:
                    delta3d = picked3d - actor.picked3d
                actor.picked3d = picked3d
            except AttributeError:
                return
            event['delta3d'] = delta3d
            event['speed3d'] = np.sqrt(np.dot(delta3d,delta3d))
            event['isPoints'] = isinstance(actor, vedo.Points)
            event['isMesh'] = isinstance(actor, vedo.Mesh)
            event['isAssembly'] = isinstance(actor, vedo.Assembly)
            event['isVolume'] = isinstance(actor, vedo.Volume)
            event['isPicture'] = isinstance(actor, vedo.Picture)
            event['isActor2D'] = isinstance(actor, vtk.vtkActor2D)

        self.depth_picking[actor] = event.picked3d

        state = 1

        # 5. Handle double click case
        if utils.time_diff(self.last_mouse_time) < 0.5:
            state = 2
        self.last_mouse_time = datetime.now()

        # We have a valid click event
        self.handle_object_click(actor, state, event)
        return state

    def handle_actor2d_click(self, state, actor, event):
        """
        Handle click on a 2D actor
        """
        pass

    def handle_void_click(self, state=None, event=None):
        """
        Handle click and drag
        """
        # Selection is None here
        self.clear_box_widget()
        self.clear_line_widget()
        self._select()
        self.update_selection_info()
        self.set_outline_visibility(False)
        self.set_color_bar_visibility(False)
        # Event-like function that you may customize
        self.selection_changed()
        #self.update_ui()

    def handle_drag_release(self, state, actor, event):
        """
        Handle click and drag
        """
        pass

    def handle_object_click(self, actor, state=1, event=None):
        """
        Handle valid object selection
        :param actor: Selected actor
        :param state: Number of mouse clicks
        :param event: Event given by vedo
        """
        # We ignore double clicks by default
        if state != 1:
            return
        camera = self.plot.renderer.GetActiveCamera()
        camera_position = np.array(camera.GetPosition())
        # If a view container is found by reverse mapping, we get it here
        view = self.controllers_map.get(actor)
        self._select(actor, view, event, camera_position)
        if self.model.selection_changed:
            self.set_outline_visibility()
            self.clear_line_widget()
            self.clear_box_widget()
            self.selection_changed()
        elif self.model.selection_point_changed:
            self.sub_selection_changed()
        self.update_ui()

    def objects_changed(self):
        """
        Function called when a new object is registered
        """
        pass

    def initialized(self):
        """
        Function called the viewer is initialized
        """
        pass

    def selection_changed(self):
        """
        Function called when another object is selected.
        """
        pass

    def sub_selection_changed(self):
        """
        Function called when for the same object, a new data point is selected
        """
        pass

    def get_object_names(self):
        return list(self.objects.keys())

    def get_selectable_key_by_id(self, value=0):
        """
        Get an object by the id of the key (not by the key/name itself)
        This is useful for sliders
        :return: String
        """
        object_ids = self.get_object_names()
        key = None
        try:
            key = object_ids[int(value)]
        except IndexError:
            pass
        return key

    def _select(self, actor=None, controller=None, event=None, 
                camera_position=None, position=None, value=None):
        """
        Set the current selected object in the model
        :param actor: a vtkActor
        :param controller: Controller of the given actor (optional)
        :param event: a vedo event from which we use picked3d and picked2d (we could directly use vtk)
        :param camera_position: position of the camera (optional) at selection time
        :param position: The final position computed on the volume or mesh or point or line.
            If not given, this will be automatically calculated
        .param value: The value corresponding to the point on the object. If not given, this will
            be automatically retrieved
        """
        mdl = self.model
        mdl.selection_changed = actor != mdl.selection
        if actor is None:
            mdl.selection_point_changed = False
            mdl.selection = None
            mdl.selection_controller = None
            mdl.selection_type = None
            mdl.selection_point = None
            mdl.selection_relative_point = None
            mdl.selection_value = None
            mdl.selection_camera_position = None
            return

        if mdl.selection_changed:
            # We assume that, even in the case where the point is the same
            # as before, the selection point changed because it's a different object
            mdl.selection_point_changed = True
            mdl.selection_value = None

        actor_type = mdl.get_actor_type(actor)
        if event is None:
            mdl.selection = actor
            mdl.selection_controller = controller
            mdl.selection_type = actor_type
            return

        if position is None or value is None:
            # We're interested in the vedo class type like <class 'vedo.shapes.Line'>
            if actor_type == ViewerModel.LINE:
                mesh_id = actor.closestPoint(actor.picked3d, returnCellId=True)
                value = actor.polydata().GetCellData().GetScalars().GetValue(mesh_id)
                position = actor.picked3d

            elif actor_type == ViewerModel.SURFACE:
                point_id = actor.closestPoint(event.picked3d, returnPointId=True)
                # Scalar values in volume are integers in this case
                arr = actor.getPointArray()
                if arr is not None:
                    value = arr[point_id]
                position = event.picked3d# actor.points(point_id)

            elif actor_type == ViewerModel.POINT:
                # If on a sphere glyph, we will not get the center of it here
                point_id = actor.closestPoint(actor.picked3d, returnPointId=True)
                poly = actor.polydata()
                scalars = poly.GetPointData().GetScalars()
                if scalars is not None and 0 < point_id < scalars.GetNumberOfValues():
                    value = scalars.GetValue(point_id)
                '''
                [Optional] TODO: WIP below to have the position in the center of the glyph when using
                non screen-space points. 
                # So we do another query to have the center position which
                # correctly represents the given data
                if actor.point_locator is None:
                    actor.point_locator = vtk.vtkPointLocator()
                    actor.point_locator.SetDataSet(poly)
                    actor.point_locator.BuildLocator()
                pid = point_id
                point_id = actor.point_locator.FindClosestPoint(actor.picked3d)
                '''
                position = poly.GetPoints().GetPoint(point_id)

            elif actor_type == ViewerModel.VOLUME:
                position, value = controller.pick(camera_position, np.array(event.picked2d))
                if position is None or value is None:
                    return
                if controller.slicers_selectable:
                    override_selection = False
        
        if camera_position is None:
            camera_position = mdl.cameras.current.GetPosition()
        
        if 'volume' in str(type(controller)).lower():
            actor = controller.actor

        if mdl.selection_point is None and position is not None:
            mdl.selection_point_changed = True
        else:
            p1 = np.array(mdl.selection_point)
            mdl.selection_point_changed = np.linalg.norm(p1 - position) > 0.1

        mdl.selection = actor
        mdl.selection_controller = controller
        mdl.selection_type = actor_type
        mdl.selection_point = np.array(position)
        mdl.selection_relative_point = np.array(position) - mdl.origin
        mdl.selection_value = value
        mdl.selection_camera_position = camera_position

    def select(self, target=None, allow_none=False):
        """
        Select an object, either by its name (or subname) or by
        reference to the vtk object. 
        :param target: Either the vtk actor or its name 
            or its index (an int from object.keys())
        :param allow_none: Whether an invalid target is accepted
            as clicking on nothing (to cancel the current selection)
        """
        if isinstance(target, str):
            key = target
            target = self.objects.get(target)
            if not allow_none and target is None:
                keys = list(self.objects.keys())
                if not self.silent:
                    print('Could not select object with given key', 
                            key, 'Please use a name from this list', keys)
        elif isinstance(target, int) or isinstance(target, float):
            target = int(target)
            key = self.get_selectable_key_by_id(target)
            try:
                target = self.objects.get(key, None)
            except Exception:
                pass
        if target is None:
            if allow_none:
                self.handle_void_click()
            else:
                return
        else:
            # Case where an object was removed and the user wants to add it back
            if target not in self.plot.actors:
                self.plot.add(target)
            self.handle_object_click(target)
            if self.model.cameras.current.autofocus:
                self.view_selected()
        self.update_ui()

    def get_selection_info(self):
        """
        Get information about the current selection
        :return: Preformatted multiline text and a dictionary of extra data
        """
        if self.model.selection is None:
            text = ''
        else:
            text = f'{self.model.selection.name}'
            data_type = self.model.selection_type.title()
            if self.model.selection_point is not None:
                # We display a point relative to the origin set by the user
                relative_point = self.model.selection_relative_point
                text += f'\n\nX: {relative_point[0]:0.2f}'
                text += f'\nY: {relative_point[1]:0.2f}'
                text += f'\nZ: {relative_point[2]:0.2f}'
            if self.model.selection_value is not None:
                text += f'\n\n{data_type} value: {self.model.selection_value}'
        return text, {}

    def update_selection_info(self):
        """
        Update text and point information for the current selection
        """
        text, data = self.get_selection_info()
        if self.selection_marker is not None:
            if self.model.selection is None:
                self.selection_marker.SetVisibility(False)
            else:
                cond = self.model.selection_point is not None and self.model.selection_marker_visible
                self.selection_marker.SetVisibility(cond)
                self.selection_marker.pos(self.model.selection_point)
                self.selection_marker.color(self.model.ui.color)
        if self.selection_info is not None:
            self.selection_info.GetMapper().SetInput(text)
            self.selection_info.SetVisibility(self.model.selection_text_visible)

    def draw_axes(self):
        """
        Draw axes around selection
        """
        if self.model.selection is None:
            return
        if self.axes_assembly is not None:
            self.plot.remove(self.axes_assembly, render=False)
        xr = [0, 0, -180]
        yr = [-180, 0, -180]
        zr = [-180, -45, 0]
        font = self.model.ui.font
        self.axes_assembly = vedo.addons.Axes(self.model.selection, hTitleRotation=xr, 
        xLabelRotation=xr, yLabelRotation=yr, zLabelRotation=zr, 
        xTitleRotation=xr, yTitleRotation=yr, zTitleRotation=zr,
        labelFont=font, titleFont=font, hTitleFont=font,
        xyGrid=False, yzGrid=False, zxGrid=False) #buildRulerAxes(actor)
        self.plot.add(self.axes_assembly)

    def set_outline_visibility(self, on=None):
        """
        Show/hide the outline
        :param on: Visibility boolean. If None, the value from the model will be used
        """
        if on is None:
            on = self.model.selection_outline_visible
        if self.outline_actor is not None:
            self.outline_actor.SetVisibility(on)

    def draw_outline(self, auto_hide_if_no_selection=True):
        """
        Draw the bounding box of the current selection.
        :param auto_hide_if_no_selection: Whether we hide the outline when there is no selection
        """
        if self.model.selection is None:
            if auto_hide_if_no_selection:
                self.set_outline_visibility(False)
            return

        if 'volume' in utils.get_type(self.model.selection):
            outline = vtk.vtkVolumeOutlineSource()
            outline.GenerateOutlineOn()
            outline.SetColor(self.model.outline_color)
            outline.SetVolumeMapper(self.model.selection.mapper())
            outline.Update()
        else:        
            outline = vtk.vtkOutlineFilter()
            #outline.GenerateOutlineOn()
            outline.SetInputData(self.model.selection.polydata())
            outline.Update()

        if self.outline_actor is not None:
            # Update existing outline
            pdm = self.outline_actor.mapper()
            pdm.SetInputConnection(outline.GetOutputPort())
            pdm.Update()
            self.set_outline_visibility()
        else:
            # Create a new outline object
            pdm = vtk.vtkPolyDataMapper()
            pdm.SetInputConnection(outline.GetOutputPort())
            pdm.Update()

            outline_actor = vedo.Mesh(pdm.GetInput())
            if 'volume' in utils.get_type(self.model.selection):
                # vtkVolumeOutlineSource doesn't account for the position of the object
                # so we have to move the outline here
                outline_actor.pos(self.model.selection.pos())
            outline_actor.SetPickable(False)
            outline_actor.GetProperty().SetOpacity(self.model.outline_opacity)
            outline_actor.GetProperty().SetColor(self.model.outline_color)
            outline_actor.GetProperty().SetLineWidth(self.model.outline_width)
            self.outline_actor = outline_actor
            self.plot.add(outline_actor)

    def add_segments(self, points, end_points=None, line_width=2, values=None, color_map='Accent', 
                    name='Segments', use_origin=True, add_to_scene=True,
                    relative_end_points=False, spherical_angles=None, radians=True):
        """
        Add a set of segments (lines made of two points). The difference with add_lines() 
        is that you have more options like giving spherical angles and setting relative end points.
        :param points: 3D numpy array of points of length n
        :param end_points: 3D numpy array of points of length n
        :param line_width: Line width, defaults to 2px
        :param values: 1D list of length n, for one scalar value per line
        :param color_map: A color map, it can be a color map built by IBLViewer or 
            a color map name (see vedo documentation), or a list of values, etc.
        :param name: Name to give to the object
        :param use_origin: Whether the current origin (not necessarily absolute 0) is used as offset
        :param add_to_scene: Whether the new lines are added to scene/plot and rendered
        :param relative_end_points: Whether the given end point is relative to the start point. False by default,
            except is spherical coordinates are given
        :param spherical_angles: 3D numpy array of spherical angle data of length n 
            In case end_points is None, this replaces end_points by finding the relative
            coordinate to each start point with the given radius/depth, theta and phi
        :param radians: Whether the given spherical angle data is in radians or in degrees
        :return: Lines
        """
        # The base assumption is that points is a 2D array, each row having a start and end point
        # but you may also pass a 1D list of positions for points and another 1D list of 
        # size N positions for end_points. Alternatively, you may pass spherical angles for end points.
        if end_points is None and spherical_angles is not None:
            relative_end_points = True
            spherical_angles = np.array(spherical_angles)
            if radians:
                end_points = spherical_angles.apply(vedo.spher2cart)
            else:
                end_points = spherical_angles.apply(utils.spherical_degree_angles_to_xyz)
            if relative_end_points:
                end_points += points
            points = np.c_[points, end_points].reshape(-1, 2, 3)
        elif end_points is not None and len(points) != len(end_points):
            n = min(len(points), len(end_points))
            logging.error(f'[add_segments() error] Mismatch between start and end points length. Only {n} segments shown.')
            points = np.c_[points[n], end_points[n]].reshape(-1, 2, 3)
        
        return self.add_lines(points, line_width, values, color_map, name, use_origin, add_to_scene)

    def add_lines(self, points, line_width=2, values=None, color_map='Accent', 
                    name='Lines', use_origin=True, add_to_scene=True):
        """
        Create a set of lines with given point sets
        :param points: List or 2D array of 3D coordinates
        :param line_width: Line width, defaults to 2px
        :param values: 1D list of length n, for one scalar value per line
        :param color_map: A color map, it can be a color map built by IBLViewer or 
            a color map name (see vedo documentation), or a list of values, etc.
        :param name: Name to give to the object
        :param use_origin: Whether the current origin (not necessarily absolute 0) is used as offset
        :param add_to_scene: Whether the new lines are added to scene/plot and rendered
        :return: objects.Lines
        """
        if not isinstance(points, np.ndarray):
            points = np.array(points, dtype=object)
        if use_origin:
            points += self.model.origin

        if values is None:
            values = np.arange(len(points))
        
        lines = obj.Lines(points, lw=line_width)
        lines.lighting(0)
        lines.cmap(color_map, values, on='cells', arrayName='data')

        # This method renames the object with a suffix if a conflicting name is found
        self.register_object(lines, name)
        if add_to_scene:
            self.plot.add(lines)
        return lines
        
    def add_spheres(self, positions, radius=10, values=None, color_map='Accent', name='Points', 
                    use_origin=True, add_to_scene=True, **kwargs):
        """
        Add new spheres. This is a shortcut for add_points(screen_space=False)
        :param positions: 3D array of coordinates
        :param radius: List same length as positions of radii. The default size is 5um, or 5 pixels
            in case as_spheres is False.
        :param values: 1D array of values, one per neuron or a time series of such 1D arrays (numpy format)
        :param color_map: A color map, it can be a color map built by IBLViewer or 
            a color map name (see vedo documentation), or a list of values, etc.
        :param name: All point neurons are grouped into one object, you can give it a custom name
        :param use_origin: Whether the current origin (not necessarily absolute 0) is used as offset
        :param add_to_scene: Whether the new lines are added to scene/plot and rendered
        :return: objects.Points
        """
        # If you use obj.Spheres, you have to do it this way
        #spheres = obj.Spheres(positions, radius, color_map, **kwargs)
        #spheres.cmap(color_map, values)
        return self.add_points(positions, radius, values, color_map, name, False, 
                use_origin, add_to_scene, **kwargs)
        
    def add_points(self, positions, radius=10, values=None, color_map='Accent', name='Points', 
                    screen_space=True, use_origin=True, add_to_scene=True, **kwargs):
        """
        Add new points, either as screen-space dots or as 3D spheres (see screen_space param)
        :param positions: 3D array of coordinates
        :param radius: List same length as positions of radii. The default size is 5um, or 5 pixels
            in case as_spheres is False.
        :param values: 1D array of values, one per neuron or a time series of such 1D arrays (numpy format)
        :param color_map: A color map, it can be a color map built by IBLViewer or 
            a color map name (see vedo documentation), or a list of values, etc.
        :param name: All point neurons are grouped into one object, you can give it a custom name
        :param screen_space: Type of point, Defaults to True. If True then the points always occupy
            the same amount of pixels defined by the radius, this is the fastest method to display
            large amount of points. If False, then points are 3D spheres and you see them larger 
            when you zoom closer to them with the camera.
        :param use_origin: Whether the current origin (not necessarily absolute 0) is used as offset
        :param add_to_scene: Whether the new lines are added to scene/plot and rendered
        :return: objects.Points
        """
        if use_origin:
            positions = np.array(positions) + self.model.origin
        if values is None:
            values = np.arange(len(positions))
        
        # You cannot easily set a time series to vedo.Spheres()
        # so this one of the reasons why objects.Points exists
        points = obj.Points(positions, radius, values, color_map, screen_space, **kwargs)
        points.lighting('off')

        # This method renames the object with a suffix if a conflicting name is found
        self.register_object(points, name)
        if add_to_scene:
            self.plot.add(points)
        return points

    def add_mesh(self, file_path, mesh_name=None, use_origin=True, add_to_scene=True):
        """
        Add a surface mesh to the scene
        :param file_path: Mesh file path (any kind of file supported by vedo)
        :param mesh_name: Name of the mesh. If None, the file name will be used.
        :param use_origin: Whether the origin is used in positioning the mesh
        :param add_to_scene: Whether the new lines are added to scene/plot and rendered
        :return: Mesh
        """
        return self.add_surface_mesh(file_path, mesh_name, use_origin, add_to_scene)

    def add_surface_mesh(self, file_path, mesh_name=None, use_origin=True, add_to_scene=True):
        """
        Add a surface mesh to the scene
        :param file_path: Mesh file path (any kind of file supported by vedo)
        :param mesh_name: Name of the mesh. If None, the file name will be used.
        :param use_origin: Whether the origin is used in positioning the mesh
        :param add_to_scene: Whether the new lines are added to scene/plot and rendered
        :return: Mesh
        """
        mesh = vedo.load(file_path)
        if use_origin:
            mesh.pos(self.model.origin)
        name = mesh_name if mesh_name is not None else utils.split_path(file_path)[1]
        # This method renames the object with a suffix if a conflicting name is found
        self.register_object(mesh, name)
        # There is a bug in VTK 9 that prevents clicking on transparent objects
        # as reported on vedo's tracker https://github.com/marcomusy/vedo/issues/291
        # The "Force opaque fix" below should be gone with the next VTK update hopefully.
        # In the meantime, we use this.
        # TODO: remove this when this bug is fixed in VTK
        #mesh.ForceOpaqueOn()
        if add_to_scene:
            self.plot.add(mesh)
        return mesh

    def add_volume(self, data=None, resolution=None, file_path=None, color_map='viridis', 
                    alpha_map=None, select=False, add_to_scene=True, transpose=None):
        """
        Add a volume to the viewer with box clipping and slicing enabled by default
        :param data: Volume image data or a file_path
        :param resolution: Resoluton of the volume
        :param file_path: File path of the volume. If you don't provide an image volume data,
            then the file_path will be used to load the volume data
        :param color_map: Color map for the volume
        :param alpha_map: Alpha map for the volume. If None, it will assume that 0 values
            are transparent and maximum values are opaque
        :param select: Whether the volume is selected
        :param add_to_scene: Whether the volume is added to scene
        :param transpose: Transposition parameter. If None. nothing happens. If True, 
            then the default IBL transposition is applied. You can provide your own, that is,
            a list of 3 elements to reorder the volume as desired.
        :return: VolumeController
        """
        if isinstance(data, str) and file_path is None:
            file_path = data
            data = None
        model = VolumeModel(resolution=resolution, file_path=file_path, data=data)
        if data is None:
            model.load_volume(file_path)
        if transpose is not None:
            model.transpose(transpose)
        model.compute_size()
        if alpha_map is None:
            alpha_map = np.linspace(0.0, 1.0, 10)
            #alpha_map = np.flip(alpha_map)
        controller = VolumeController(self.plot, model, add_to_scene=add_to_scene)
        controller.set_color_map(color_map, alpha_map)
        # register_object is automatically called on view.actor within register_controller
        self.register_controller(controller, controller.get_related_actors())
        if select:
            self.select(controller.actor)
            self.view_selected()
        return controller

    def remove_object(self, target=None, unregister=False):
        """
        Remove an object from the plot
        :param target: Target object
        :param unregister: Whether the object is also unregistered, which
            means that it's not in the list of objects anymore and cannot
            be recovered using select(object_name)
        """
        deselect = False
        if target is None:
            target = self.model.selection
            deselect = True
        if target is None:
            return
        if deselect:
            self.handle_void_click()
        self.plot.remove(target)
        if unregister:
            self.unregister_object(target)
        self.update_ui()

    def toggle_menu(self, name=None, context=None):
        """
        Toggle menu and activate only one button in the set
        """
        if context is None:
            context = UIModel.DEFAULT
        elements = self.model.ui.get_elements(context)
        for el_name in elements:
            element = elements[el_name]
            if el_name == name:
                self.set_button_state(element, True)
                continue
            self.set_button_state(element, False)

    def set_menu_context(self, button_name, context):
        """
        Set the context from the main menu buttons
        :param button_name: Which button is clicked
        :param context: What context to activate
        """
        if self.model.ui.context == context:
            self.model.ui.context = UIModel.DEFAULT
            self.toggle_menu()
            self.update_ui()
            return
        self.toggle_menu(button_name)
        self.model.ui.set_context(context)
        self.update_ui()

    def camera_context(self):
        """
        Set camera context
        """
        self.set_menu_context('camera', UIModel.CAMERA)

    def scene_context(self):
        """
        Set scene context
        """
        self.set_menu_context('scene', UIModel.SCENE)

    def data_context(self):
        """
        Set data context
        """
        self.set_menu_context('data', UIModel.DATA)

    def object_context(self):
        """
        Set object context
        """
        self.set_menu_context('object', UIModel.OBJECT)

    def time_series_context(self):
        """
        Set time series context
        """
        self.set_menu_context('time_series', UIModel.TIME_SERIES)

    def export_context(self):
        """
        Set export context
        """
        self.set_menu_context('export', UIModel.EXPORT)

    def toggle_marker_type(self):
        """
        Toggle marker type (between cross and sphere)
        """
        if isinstance(self.selection_marker, obj.Cross3D):
            self.set_selection_marker('sphere')
        else:
            self.set_selection_marker('cross')

    def set_selection_marker(self, marker_type='cross', size=500, cross_thickness=3):
        """
        Set the selection marker type. Accepted values currently are 'cross' or 'sphere'
        :param marker_type: Marker type
        """
        if self.selection_marker is not None:
            #position = self.selection_marker.pos()
            self.plot.remove(self.selection_marker, render=False)
        if marker_type.lower() == 'cross':
            self.selection_marker = obj.Cross3D(size=size, thickness=cross_thickness, color=self.model.ui.color)
        else:
            self.selection_marker = vedo.Sphere(r=size / 2, c=self.model.ui.color)
        self.selection_marker.pickable(False)
        if self.model.selection_point is not None:
            self.selection_marker.pos(self.model.selection_point)
        self.plot.add(self.selection_marker, render=False)
        self.selection_marker.lighting('off')
        #self.selection_marker.GetProperty().SetOpacity(0.5)

    def initialize_vtk_ui(self, x=40, y=50, sx=None, sy=None, sw=None):
        """
        Initialize a per-context VTK UI.
        :param x: Base screen-space x coordinate
        :param y: Base screen-space y coordinate (starts from the bottom in VTK)
        """
        if sx is None:
            sx = self.model.ui.embed_submenu_x
        if sy is None:
            sy = self.model.ui.embed_submenu_y
        if sw is None:
            sw = self.model.ui.embed_slider_width

        self.add_button('camera', self.camera_context, [x, y+200], 'Camera', toggle=True)
        self.add_button('scene', self.scene_context, [x, y+150], 'Scene', toggle=True)
        self.add_button('object', self.object_context, [x, y+100], 'Object', toggle=True)
        self.add_button('data', self.data_context, [x, y+50], 'Data', toggle=True)
        self.add_button('export', self.export_context, [x, y], 'Export', toggle=True)
        self.add_camera_embed_ui(sx, sy, sw)
        self.add_scene_embed_ui(sx, sy, sw)
        self.add_object_embed_ui(sx, sy, sw)
        self.add_data_embed_ui(sx, sy, sw)
        self.add_export_embed_ui(sx, sy, sw)

    def add_camera_embed_ui(self, x, y, sw, nc=150):
        """
        Add camera-context UI
        :param x: Base X position
        :param y: Base Y position (0 is at the bottom of the screen)
        :param sw: Slider width
        """
        self.model.ui.toggle_context(UIModel.CAMERA)
        self.add_button('anterior', self.set_anterior_view, [x, y+120], 'Anterior')
        self.add_button('posterior', self.set_posterior_view, [x+nc, y+120], 'Posterior')

        self.add_button('dorsal', self.set_dorsal_view, [x, y+80], 'Dorsal')
        self.add_button('ventral', self.set_ventral_view, [x+nc, y+80], 'Ventral')

        self.add_button('left', self.set_left_view, [x, y+40], 'Left')
        self.add_button('right', self.set_right_view, [x + nc, y + 40], 'Right')

        self.add_button('ortho', self.toggle_orthographic_view, [x, y], 'Orthographic', toggle=True)
        
        # or better yet, group these buttons into a labelled collection? If this is any useful later on?
        #self.model.ui.register([b1, b2, b3, b4, b5, b6, b7], UIModel.CAMERA)
        self.model.ui.toggle_context(UIModel.CAMERA)

    def add_scene_embed_ui(self, x, y,  sw):
        """
        Add scene-context UI
        :param x: Base X position
        :param y: Base Y position (0 is at the bottom of the screen)
        :param sw: Slider width
        """
        self.model.ui.toggle_context(UIModel.SCENE)
        s_kw = self.model.ui.slider_config
        #self.add_button('move_pivot', self.move_camera_pivot, [x, y + 90], 'Move camera pivot')
        self.add_button('previous_selection', self.select_previous_object, [x, y-5], '<')
        self.add_slider('selection', self.select_object_by_id, 0, 0, 0, [x+55, y, sw], precision=0, **s_kw)
        self.add_button('next_selection', self.select_next_object, [x+sw+80, y-5], '>')
        self.add_button('autofocus', self.toggle_autofocus, [x, y+40], 
                        ['Autofocus: Off', 'Autofocus: On'], toggle=True, state=1)
        self.add_button('view_selected', self.view_selected, [x, y+80], 'Focus on selection')

        self.add_button('dark_mode', self.toggle_dark_mode, [x+400, y+80], 
                        ['Dark UI', 'Light UI'], toggle=True, state=0 if self.model.ui.dark_mode else 1)
        self.add_button('marker', self.toggle_marker, [x+400, y+40], 
                        ['Marker: Off', 'Marker: On'], toggle=True, state=1)
        #self.add_button('clickable', self.toggle_pickable, [x, y], ['Clickable: On', 'Clickable: Off'], toggle=True)
        self.add_button('toggle_info', self.toggle_info, [x+200, y+40], 
                        ['Info overlay: Off', 'Info overlay: On'], toggle=True, state=1)
        self.add_button('toggle_outline', self.toggle_outline, [x+200, y+80], 
                        ['Bounding box: Off', 'Bounding box: On'], toggle=True, state=1)
        #self.add_button('view_slices', self.toggle_view_slices, [x, y], ["Hide slices", "Show slices"], True)
        #self.axes_button = self.add_button(self.toggle_axes_visibility, pos=(50, 0.78), ["Show axes", "Hide axes"], True)
        self.model.ui.toggle_context(UIModel.SCENE)

    def add_object_embed_ui(self, x, y, sw):
        """
        Add object-context UI
        :param x: Base X position
        :param y: Base Y position (0 is at the bottom of the screen)
        :param sw: Slider width
        """
        s_kw = self.model.ui.slider_config
        # TODO: add option to use same slicer for all objects
        self.model.ui.toggle_context(UIModel.OBJECT)
        
        #self.add_button('global_slicer', self.toggle_global_slicer, [x, y+145], ['Global slicer: OFF', 'Global slicer: ON'], toggle=True)
        self.add_button('box_widget', self.add_box_widget, [x, y+40], 'Cutter/Slicer')
        self.add_button('remove_object', self.remove_object, [x+130, y+40])
        self.add_button('isosurface', self.isosurface, [x+280, y+40])
        #self.add_slider('isosurface', self.update_isosurface, 0, 1200, 200, [x+2*sw+80, y-5, sw], precision=2, **s_kw)
        self.add_button('hollow_volume', self.toggle_hollow_mode, [x+380, y+40], 'Hollow regions', toggle=True)
        self.add_slider('opacity', self.update_opacity, 0.0, 1.0, 0.9, [x, y-5, sw], precision=2, **s_kw)
        self.add_slider('slices_opacity', self.update_slices_opacity, 0, 1.0, 0.75, [x+sw+40, y-5, sw], precision=2, **s_kw)
        #self.add_button('slices_visibility', self.toggle_slices_visibility, [x, yb + 130], ["Hide slices", "Show slices"])

        '''
        # Code for adding six sliders for box slicing.
        self.add_slider('-x', self.update_nx_slicer, -1, 0, 0, [x+sw+40, y+70, sw], oid=0, **s_kw)
        self.add_slider('+x', self.update_px_slicer, 0, 1, 0, [x, y+70, sw], oid=1, **s_kw)

        self.add_slider('-y', self.update_ny_slicer, -1, 0, 0, [x+sw+40, y+35, sw], oid=2, **s_kw)
        self.add_slider('+y', self.update_py_slicer, 0, 1, 0, [x, y+35, sw], oid=3, **s_kw)

        self.add_slider('-z', self.update_nz_slicer, -1, 0, 0, [x+sw+40, y, sw], oid=4, **s_kw)
        self.add_slider('+z', self.update_pz_slicer, 0, 1, 0, [x, y, sw], oid=5, **s_kw)
        '''
        self.model.ui.toggle_context(UIModel.OBJECT)

    def add_data_embed_ui(self, x, y, sw):
        """
        Add data-context UI
        :param x: Base X position
        :param y: Base Y position (0 is at the bottom of the screen)
        :param sw: Slider width
        """
        self.model.ui.toggle_context(UIModel.DATA)
        s_kw = self.model.ui.slider_config
        self.add_slider('data_time_series', self.update_time_series, 0, 1, 0, [x, y-5, sw], **s_kw)
        self.add_button('play_time_series', self.play_time_series, [x, y+40], 
                        ['Play time series', 'Pause time series'], toggle=True)
        self.add_button('new_probe', self.add_probe, [x+180, y+40])
        self.add_button('move_probe', self.edit_probe, [x+180, y+40])
        self.model.ui.toggle_context(UIModel.DATA)
        
    def add_export_embed_ui(self, x, y, sw):
        """
        Add export-context UI
        :param x: Base X position
        :param y: Base Y position (0 is at the bottom of the screen)
        :param sw: Slider width
        """
        s_kw = self.model.ui.slider_config
        self.model.ui.toggle_context(UIModel.EXPORT)
        self.add_button('export_image', self.export_image, [x, y], 'Export image')
        self.add_button('export_turntable_video', self.export_turntable_video, [x + 150, y], 'Export 360 video')
        self.add_slider('video_duration', self.update_video_duration, 0, 60, 
                        self.model.video_duration, [x + 150, y+120, sw], 'Video duration (s)', **s_kw)
        self.add_slider('start_angle', self.update_video_start_angle, 0, 360, 0, [x + 150, y+80, sw], **s_kw)
        self.add_slider('end_angle', self.update_video_end_angle, 0, 360, 360, [x + 150, y+40, sw], **s_kw)
        self.model.ui.toggle_context(UIModel.EXPORT)

    def update_scene_ui(self, context_elements):
        """
        Update the scene-related UI
        :param context_elements: Current UI context elements
        """
        if not self.model.ui.is_scene_context():
            return

        slider = self.model.ui.get_element('selection')
        value = 0
        got_value = False
        for key in self.objects:
            if self.objects[key] == self.model.selection:
                got_value = True
                break
            value += 1
        if got_value:
            slider.update(widget=slider, value=value)

    def is_volume(self, target=None):
        """
        Get whether the target object is a volume
        :param target: vtkActor. If None, the current selected object is used.
        :return: Boolean
        """
        if target is None:
            target = self.model.selection
        if target is None:
            return
        return isinstance(target, vedo.Volume)

    def get_selection_opacity(self):
        """
        Get the opacity of the current selected object
        :return: Float or None
        """
        opacity_value = None
        try:
            opacity_value = self.model.selection.GetProperty().GetOpacity()
        except Exception:
            if isinstance(self.model.selection_controller, VolumeController):
                opacity_value = self.model.selection_controller.get_opacity()
        return opacity_value

    def get_selection_slices_opacity(self):
        """
        Get the opacity of the current volume slices
        :return: Float or None
        """
        slices_opacity_value = None
        if isinstance(self.model.selection_controller, VolumeController):
            slices_opacity_value = self.model.selection_controller.get_slices_opacity()
        return slices_opacity_value

    def update_object_ui(self, context_elements):
        """
        Update the object mode UI
        :param context_elements: Current UI context elements
        """
        if not self.model.ui.is_object_context():
            return

        volume_mode = isinstance(self.model.selection_controller, VolumeController)
        self.set_element_visibility(self.model.ui.get_element('hollow_volume'), volume_mode)
        self.set_element_visibility(self.model.ui.get_element('view_region'), volume_mode)

        slider = self.widgets.get('opacity')#self.model.ui.get_element('opacity')
        opacity_value = self.get_selection_opacity()
        if opacity_value is not None:
            slider.update(widget=slider, value=opacity_value)

        slider = self.widgets.get('slices_opacity')#model.ui.get_element('slices_opacity')
        slices_opacity_value = self.get_selection_slices_opacity()
        if slices_opacity_value is not None:
            slider.update(widget=slider, value=slices_opacity_value)
        '''
        # Below code not necessary anymore, we keep it here as a reference
        # if you want to update sliders for slicing instead of using vtk widgets
        dimensions = utils.get_actor_dimensions(self.model.selection)
        for element_name in context_elements:
            element = context_elements[element_name]
            axis = None
            slicers = None
            try:
                axis = element.oid
                slicers = self.model.selection_controller.slicers
            except Exception:
                continue
            model = slicers.get(axis).model
            mod = axis % 2
            orientation = -1 if mod == 0 else 1
            if orientation == 1:
                min_v = 0
                max_v = dimensions[model.axis] * orientation
            else:
                min_v = dimensions[model.axis] * orientation
                max_v = 0
            rep = element.GetRepresentation()
            rep.SetMinimumValue(min_v)
            rep.SetMaximumValue(max_v)
            if model.value is not None:
                element.update(widget=element, value=model.value)
        '''

    def update_data_ui(self, context_elements):
        """
        Update data UI
        :param context_elements: Current UI context elements
        """
        if not self.model.ui.is_data_context():
            return

        volume_mode = isinstance(self.model.selection_controller, VolumeController)
        self.set_element_visibility(self.model.ui.get_element('new_probe'), volume_mode)

        probe_mode = self.is_probe(self.model.selection)
        self.set_element_visibility(self.model.ui.get_element('move_probe'), probe_mode)

        slider = self.widgets.get('data_time_series')
        current_value, max_value = self.get_time_series_info()
        if slider is not None:
            if current_value is not None:
                slider.update(widget=slider, value=current_value)
            slider.GetRepresentation().SetMinimumValue(0)
            slider.GetRepresentation().SetMaximumValue(max_value)
            self.draw_color_bar()

    def get_time_series_info(self, target=None):
        """
        Get the number of time series and id of the current one for a target.
        If None, then the current selection is used
        :param target: a vtk object
        :return: Int, Int
        """
        if target is None:
            target = self.model.selection
        if target is None:
            return None, None
        # Only volume object has a controller in iblviewer
        controller = self.controllers_map.get(target) #self.model.selection_controller
        max_value = 0
        current_value = None
        if isinstance(controller, VolumeController):
            max_value = len(controller.model.luts)-1
            current_value = controller.model.luts.current_key_id
        else:
            try:
                point_data = target.polydata().GetPointData()
                #name = target.polydata().GetPointData().GetArrayName()
                if isinstance(target, obj.Points):
                    # We use this method to ignore unwanted arrays that are automatically
                    # created by VTK
                    max_value = target.get_number_of_arrays() - 1
                else:
                    max_value = point_data.GetNumberOfArrays() - 1
                polydata = target.polydata()
                current_name = polydata.GetPointData().GetScalars().GetName()
                # Forced to make a loop here in order to get the current id...
                for array_id in range(max_value):
                    if point_data.GetArray(array_id).GetName() == current_name:
                        current_value = array_id
                        break
                #current_value = point_data.GetAbstractArray(current_name)
            except Exception:
                return None, None
        return current_value, max_value

    def update_ui(self):
        """
        Update the UI
        """
        self.update_selection_info()

        if self.model.got_selection():
            self.draw_outline()

        if self.model.selection_changed or self.model.selection_point_changed:
            self.draw_color_bar()
            self.set_color_bar_visibility()

        if self.model.ui.embed:
            context_elements = self.model.ui.get_elements()
            if self.model.ui.context_changed:
                default_elements = self.model.ui.get_elements(UIModel.DEFAULT)
                context_elements.update(default_elements) # concatenates dictionaries
                self.set_ui_visibility(False, context_elements)
            self.update_scene_ui(context_elements)
            self.update_object_ui(context_elements)
            self.update_data_ui(context_elements)

    def set_color_bar_visibility(self, on=None):
        """
        Show or hide the color bar
        :param on: Visibility boolean
        """
        if self.color_bar is not None:
            if on is None:
                on = self.model.selection_color_bar_visible
            self.color_bar.SetVisibility(on)

    def draw_color_bar(self):
        """
        Draw a color bar for the current selection
        """
        ui = self.model.ui
        selection = self.model.selection
        controller = self.model.selection_controller
        if selection is None:
            return
        lut = None
        if self.is_probe(selection):
            controller = selection.target_controller
        if controller is not None and hasattr(controller.model, 'luts'):
            if controller.model.luts.current is not None:
                lut = controller.model.luts.current.scalar_lut
        if lut is None and isinstance(controller, VolumeController):
            lut = vedo.utils.ctf2lut(controller.actor)
        elif lut is None:
            # Default way to get a LUT for the scalar/color bar
            lut = selection.mapper().GetLookupTable()
        if lut is None:
            return
        #if self.color_bar is not None:
            #self.set_color_bar_visibility(True)
            #self.plot.remove(self.color_bar, render=False)
        if self.color_bar is not None:
            utils.update_scalar_bar(self.color_bar, lut)
        else:
            self.color_bar = utils.add_scalar_bar(lut, pos=(0.85,0.04), 
                                    font_color=ui.color, titleFontSize=ui.font_size)
            self.plot.add([self.color_bar])

    def clear_color_bar(self):
        """
        Clear the color bar
        """
        self.plot.remove(self.color_bar)

    def set_button_state(self, element, state):
        """
        Set the state of a button. For a toggle, True or False.
        """
        if 'button' in utils.get_type(element):
            element.status(state)

    def get_button_state(self, element):
        """
        Get the state of a button
        """
        if 'button' in utils.get_type(element):
            return element.status()

    def set_element_visibility(self, element, visible=True):
        """
        Set a UI element's visibility
        :param element: a vtk2DActor or something like that (changes for sliders, texts, etc.)
        :param visible: Visible or not
        """
        t = utils.get_type(element)
        if 'slider' in t:
            element.GetRepresentation().SetVisibility(visible)
            element.GetRepresentation().SetPickable(visible)
            if visible:
                element.EnabledOn()
            else:
                element.EnabledOff()
            #element.GetRepresentation().PickingManagedOn()
        if 'text' in t:
            element.SetVisibility(visible)
        elif 'button' in t:
            element.actor.SetVisibility(visible)
            element.actor.SetPickable(visible)

    def update_element_color(self, element):
        """
        Set a UI element's color
        :param element: a vtk2DActor or something like that (changes for sliders, texts, etc.)
        """
        t = utils.get_type(element)
        if 'slider' in t:
            config = self.model.ui.slider_config
            color = vedo.getColor(config['c'])
            rep = element.GetRepresentation()
            rep.GetLabelProperty().SetColor(color)
            rep.GetTubeProperty().SetColor(color)
            #rep.GetTubeProperty().SetOpacity(0.6)
            rep.GetSliderProperty().SetColor(color)
            rep.GetSelectedProperty().SetColor(np.sqrt(np.array(color)))
            rep.GetCapProperty().SetColor(color)
            rep.GetTitleProperty().SetColor(color)
            element.label.property.SetColor(color)
        if 'text' in t:
            config = self.model.ui.slider_config
            color = vedo.getColor(config['c'])
            element.property.SetColor(color)
        elif 'button' in t:
            if element.toggle:
                config = self.model.ui.toggle_config
            else:
                config = self.model.ui.button_config
            colors = config['c']
            background_colors = config['bc']
            status = element.statusIdx
            element.colors = colors
            element.bcolors = background_colors
            element.textproperty.SetColor(vedo.getColor(element.colors[status]))
            bcc = np.array(vedo.getColor(element.bcolors[status]))
            element.textproperty.SetBackgroundColor(bcc)
            if element.showframe:
                #element.textproperty.FrameOn()
                element.textproperty.SetFrameColor(np.sqrt(bcc))
        elif 'scalarbar' in t:
            color = vedo.getColor(self.model.ui.color)
            element.GetTitleTextProperty().SetColor(color)
            element.GetLabelTextProperty().SetColor(color)

    def set_ui_visibility(self, visible, exceptions=None, ui_button_visible=True):
        """
        Set the UI visibility
        :param visible: Whether the UI is visible or not
        :param exceptions: Dictionary of objects that have to be the inverse of
            the given state
        :param ui_button_visible: Whether the UI button is still visible or not,
            so that the user can restore the UI by clicking on it.
            This is not necessary in case of Jupyter NB
        """
        self.model.ui.visible = visible

        all_elements = self.model.ui.all_elements()
        for key in all_elements:
            self.set_element_visibility(all_elements[key], visible)

        if exceptions is None:
            return
        for element_id in exceptions:
            element = exceptions[element_id]
            self.set_element_visibility(element, not visible)
    
    def add_text(self, name, text, pos, color=None, justify='top-left', context=None, **kwargs):
        """
        Add a 2D text on scene
        :param name: Name of the object
        :param text: Text string
        :param pos: Position on screen
        :param color: Color of the font
        :param justify: Type of justification
        :param context: Context to register this object to
        :return: Text2D
        """
        if color is None:
            color = self.model.ui.color
        text = vedo.Text2D(text, c=color, pos=pos, font=self.model.ui.font, justify=justify, **kwargs)
        text.property.SetFontSize(self.model.ui.font_size)
        # Assume absolute coordinates in pixels
        if pos[0] > 1.0 or pos[1] > 1.0:
            text.GetPositionCoordinate().SetCoordinateSystemToDisplay()
            text.SetPosition(pos)
        text.name = name
        self.model.ui.register(text, context)
        self.set_element_visibility(text, False)
        return text

    def add_button(self, name, event_handler, pos, label=None, toggle=False, state=0, context=None):
        """
        Add a left-aligned button
        :param name: Button name
        :param event_handler: Event handler function
        :param pos: Screen-space position
        :param label: Button text label
        :param toggle: Whether it's a toggle button or not
        :return: Button
        """
        kwargs = self.model.ui.button_config
        if toggle:
            kwargs = self.model.ui.toggle_config
        if label is None:
            label = name.replace('_', ' ').title()
        if isinstance(label, str):
            label = [label, label] if toggle else [label]
        button = self.plot.addButton(event_handler, states=label, pos=pos, **kwargs)
        button.status(state)
        if toggle:
            # Decorator that automatically changes the button state on click
            def wrapped_event_handler():
                button.switch()
                event_handler()
            button.function = wrapped_event_handler
        #button.actor.SetTextScaleModeToViewport()
        button.actor.GetTextProperty().SetJustificationToLeft()
        # Assume absolute coordinates in pixels
        if pos[0] > 1.0 or pos[1] > 1.0:
            button.actor.GetActualPositionCoordinate().SetCoordinateSystemToDisplay()
            button.actor.SetPosition(pos)
        button.toggle = toggle
        button.name = name
        self.model.ui.register(button, context)
        self.set_element_visibility(button, False)
        return button

    def get_slider_bounds(self, x, y, length, horizontal=True):
        """
        Get slider coordinates, releative to lower left corner of the window
        :param x: X position
        :param y: Y position
        :param length: Length of the slider
        :param horizontal: Whether the length is horizontal or vertical
        :return: np 2d array with min and max coordinates
        """
        if horizontal:
            return np.array([[x, y], [x + length, y]])
        else:
            return np.array([[x, y], [x, y + length]])

    def add_slider(self, name, event_handler, min_value, max_value, value, pos, 
                    title=None, oid=None, precision=0, context=None, **kwargs):
        """
        Add a slider to the UI
        :param event_handler: Event handler function called when the slider is updated
        :param min_value: Min slider value
        :param max_value: Max slider value
        :param value: Default value
        :param size: Font size
        :param pos: Position in the UI as a tuple or list, see get_slider_bounds for details
        :param kwargs: Dictionary of kwargs, see vedo.addons.addSlider2D() for details
        :return: 2D Slider
        """
        position = self.get_slider_bounds(*pos)
        
        if kwargs is None:
            kwargs = {}
        kwargs['title'] = False
        kwargs['showValue'] = False
        
        # Decorator for the event handler
        def update_slider(widget=None, event=None, value=None, *args, **kwargs):
            if precision == 0 and value is not None:
                value = int(value)
            if widget is not None:
                if value is not None:
                    widget.GetRepresentation().SetValue(value)
                else:
                    value = widget.GetRepresentation().GetValue()
                if precision == 0 and value is not None:
                    value = int(value)
                widget.text = f'{slider.label_prefix}: {value:.{precision}f}'
                widget.label.GetMapper().SetInput(widget.text)
                if widget.last_value == value:
                    return
                widget.last_value = value
            if event is not None:
                event_handler(widget, event, *args, **kwargs)
        
        slider = self.plot.addSlider2D(update_slider, min_value, max_value, value, position, **kwargs)
        if precision == 0:
            slider.SetAnimationModeToJump()
        rep = slider.GetRepresentation()
        '''
        rep.SetLabelHeight(0.003 * size)
        rep.GetLabelProperty().SetFontSize(size)
        rep.SetTitleHeight(0.003 * size)
        rep.GetTitleProperty().SetFontSize(size)
        '''
        label_position = [position[0, 0], position[0, 1] + 0.11]
        # Assume absolute coordinates in pixels
        if pos[0] > 1.0 or pos[1] > 1.0:
            rep.GetPoint1Coordinate().SetCoordinateSystemToDisplay()
            rep.GetPoint2Coordinate().SetCoordinateSystemToDisplay()
            rep.GetPoint1Coordinate().SetValue(position[0, 0], position[0, 1])
            rep.GetPoint2Coordinate().SetValue(position[1, 0], position[1, 1])
            label_position = [position[0, 0], position[0, 1] + 27]
            
        rep.SetSliderLength(0.012)
        rep.SetSliderWidth(0.01)
        rep.SetTubeWidth(0.0025)

        if oid is not None:
            slider.oid = oid

        slider.update = update_slider
        slider.name = name
        prefix = title.title() if isinstance(title, str) and title != '' else name.title()
        prefix = prefix.replace('_', ' ')
        slider.label_prefix = prefix
        slider.text = f'{slider.label_prefix}: {rep.GetValue():.{precision}f}'
        slider.label = self.add_text('selection_name', slider.text, label_position)
        slider.label.name = name + '_label'
        slider.last_value = None
        self.plot.add(slider.label)

        self.set_element_visibility(slider, False)
        self.set_element_visibility(slider.label, False)
        self.widgets_reflection[slider.GetRepresentation()] = slider
        self.widgets[name] = slider
        self.model.ui.register([slider, slider.label], context)
        return slider

    def slider_value(self, element_id, value=None, widget=None):
        """
        Abstract method to get or set a value on a VTK slider element
        :param element_id: Element id (found in self.model.ui.get_elements().keys())
        :param value: Value of Z slice, defaults to 0.0
        :param widget: Widget instance (given by the event caller)
        :return: Float value
        """
        if widget is not None:
            value = widget.GetRepresentation().GetValue()
        elif value is not None:
            widget = self.model.ui.context_element(element_id)
            if widget is not None:
                widget.GetRepresentation().SetValue(value)
        return value

    def clip_to_bounds(self, bounds_obj, target=None):
        """
        Clip a target object with the bounding box of another object
        :param bounds_obj: a vtkActor
        :param target: Target object. If None, the current selection is used
        """
        if target is None:
            target = self.model.selection
        if target is None:
            return
        bounds = utils.get_actor_bounds(bounds_obj)
        if isinstance(target, Volume):
            controller = self.controllers_map.get(target)
            if controller is not None:
                controller.clip_to_bounds(bounds)
        else:
            utils.set_clipping_planes(target, bounds)

    def isosurface(self, value=None, volume_controller=None, split_meshes=False, remove_existing=True):
        """
        Generate mesh(es) for a given value in a volume
        :param value: Value/label for which we want to generate an isosurface mesh
        :param volume_controller: Volume controller. If None, the current one is used.
        If the current one is None, then nothing happens.
        :param split_meshes: Whether manifold meshes are split if there are multiple ones
        :param remove_existing: Whether existing meshes are removed from the plot
        :return: List of meshes computed
        """
        if volume_controller is None:
            volume_controller = self.model.selection_controller
        if volume_controller is None:
            return
        if value is None:
            value = self.model.selection_value
        if value is None:
            return
        if remove_existing:
            current_meshes = volume_controller.model.isosurfaces.current
            self.plot.remove(current_meshes)
        if value is None:
            value = self.model.selection_value
        if value is None:
            return
        meshes = volume_controller.isosurface(value, split_meshes=split_meshes)
        if meshes is None:
            return
        for mesh in meshes:
            self.register_object(mesh)
            self.plot.add(mesh)
        return meshes

    def set_probe_position(self, pt1=None, pt2=None):
        """
        Set the position of the current probe
        """
        if self.line_widget is not None and pt1 is not None:
            self.line_widget.SetPoint1(pt1)
        if self.line_widget is not None and pt2 is not None:
            self.line_widget.SetPoint2(pt2)
            
    def add_probe(self):
        """
        Add a new probe widget
        """
        self.clear_line_widget()
        event_handler = None
        controller = self.model.selection_controller
        if isinstance(controller, VolumeController):
            event_handler = self.update_current_probe
            self.line_widget = utils.probe(self.plot, self.model.selection, 
                                            self.line_widget, event_handler,
                                            self.model.probe_initial_point1,
                                            self.model.probe_initial_point2)
            pt1 = self.line_widget.GetPoint1()
            pt2 = self.line_widget.GetPoint2()
            probe = controller.add_probe(pt1, pt2)
            self.register_object(probe)
            self.model.probes.store(probe, probe.name, replace_existing=False, set_current=True)
        '''
        # If later interested in this, a generic approach to surface mesh data probing
        // Get the ID of the point that is closest to the query position
        vtkIdType id = locator->FindClosestPoint(pt);
        // Retrieve the first attribute value from this point
        double value = polyData->GetPointData()->GetScalars()->GetTuple(id, 0);
        '''

    def is_probe(self, probe=None):
        """
        Check if an object is (the result of) a probe
        :param probe: Probe object
        :return: Boolean
        """
        if probe is None:
            probe = self.model.selection
        if probe is None:
            return False
        target = hasattr(probe, 'target_controller') and hasattr(probe, 'target')
        points = hasattr(probe, 'origin') and hasattr(probe, 'destination')
        return target and points

    def update_current_probe(self, widget=None, event=None, point1=None, point2=None):
        """
        Update the current probe
        :param widget: vtkLineWidget
        :param event: vtkEvent
        :param point1: Point 1, this will override the widget's point 1
        :param point2: Point 2, this will override the widget's point 2
        """
        if widget is None:
            widget = self.line_widget
        if widget is None and point1 is None and point2 is None:
            return
        if widget is not None:
            point1 = widget.GetPoint1()
            point2 = widget.GetPoint2()
        obj = self.model.probes.current
        volume_controller = obj.target_controller
        # Probe method is within VolumeController
        volume_controller.update_probe(point1, point2, obj)

    def edit_probe(self, probe=None):
        """
        Edit/move an existing probe
        """
        if probe is None:
            probe = self.model.selection
        if probe is None:
            return
        if self.is_probe(probe):
            # The event handler in this case will refer to the active probe in
            # self.model.probes.current so we have to set it
            self.model.probes.set_current(probe) # or probe.name works too
            event_handler = self.update_current_probe
            self.line_widget = utils.probe(self.plot, probe, self.line_widget, event_handler, 
                                            probe.origin, probe.destination)

    def clear_line_widget(self):
        """
        Clear the active line widget
        """
        if self.line_widget is not None:
            self.line_widget.Off()
            #self.line_widget.RemoveObservers('InteractionEvent')
            #self.plot.widgets.remove(self.line_widget)

    def toggle_box_widget(self):
        """
        Toggle box widget
        """
        if self.box_widget is not None:
            self.clear_box_widget()
        else:
            self.add_box_widget()
            
    def add_box_widget(self):
        """
        Add a box widget for clipping the selected object
        """
        self.clear_box_widget()
        if self.model.selection is not None:
            event_handler = None
            if isinstance(self.model.selection_controller, VolumeController):
                event_handler = self.model.selection_controller.box_widget_update
            self.box_widget = utils.box_widget(self.plot, self.model.selection, event_handler)

    def clear_box_widget(self):
        """
        Clear the active box widget
        """
        if self.box_widget is not None:
            self.box_widget.Off()
            self.box_widget.RemoveObservers('InteractionEvent')
            self.plot.widgets.remove(self.box_widget)
            self.plot.interactor.Render()
            self.box_widget = None
        
    def update_px_slicer(self, widget=None, event=None, value=0.0):
        """
        Event handler for Slicer on +X axis
        :param widget: Widget instance (given by the event caller)
        :param event: Event (given by the event caller)
        :param value: Value of X slice, defaults to 0.0
        """
        value = self.slider_value('+x', value, widget)
        try:
            self.model.selection_controller.update_slicer(1, value)
        except AttributeError as e:
            pass
    
    def update_nx_slicer(self, widget=None, event=None, value=0.0):
        """
        Event handler for Slicer on -X axis
        :param widget: Widget instance (given by the event caller)
        :param event: Event (given by the event caller)
        :param value: Value of X slice, defaults to 0.0
        """
        value = self.slider_value('-x', value, widget)
        try:
            self.model.selection_controller.update_slicer(0, value)
        except AttributeError:
            pass

    def update_py_slicer(self, widget=None, event=None, value=0.0):
        """
        Event handler for Slicer on +Y axis
        :param widget: Widget instance (given by the event caller)
        :param event: Event (given by the event caller)
        :param value: Value of Y slice, defaults to 0.0
        """
        value = self.slider_value('+y', value, widget)
        try:
            self.model.selection_controller.update_slicer(3, value)
        except AttributeError:
            pass

    def update_ny_slicer(self, widget=None, event=None, value=0.0):
        """
        Event handler for Slicer on -Y axis
        :param widget: Widget instance (given by the event caller)
        :param event: Event (given by the event caller)
        :param value: Value of Y slice, defaults to 0.0
        """
        value = self.slider_value('-y', value, widget)
        try:
            self.model.selection_controller.update_slicer(2, value)
        except AttributeError:
            pass

    def update_pz_slicer(self, widget=None, event=None, value=0.0):
        """
        Event handler for Slicer on +Z axis
        :param widget: Widget instance (given by the event caller)
        :param event: Event (given by the event caller)
        :param value: Value of Z slice, defaults to 0.0
        """
        value = self.slider_value('+z', value, widget)
        try:
            self.model.selection_controller.update_slicer(5, value)
        except AttributeError:
            pass

    def update_nz_slicer(self, widget=None, event=None, value=0.0):
        """
        Event handler for Slicer on -Z axis
        :param widget: Widget instance (given by the event caller)
        :param event: Event (given by the event caller)
        :param value: Value of Z slice, defaults to 0.0
        """
        value = self.slider_value('-z', value, widget)
        try:
            self.model.selection_controller.update_slicer(4, value)
        except AttributeError:
            pass

    def toggle_slices(self, event=None):
        """
        Toggle slicers visibility
        :param widget: Widget instance (given by the event caller)
        :param event: Event (given by the event caller)
        """
        self.model.slices_visible = not self.model.slices_visible
        try:
            self.model.selection_controller.toggle_slices_visibility()
        except Exception:
            pass
    
    def toggle_global_slicer(self, event=None):
        """
        Toggle global slicer mode
        :param widget: Widget instance (given by the event caller)
        :param event: Event (given by the event caller)
        """
        raise NotImplementedError
        '''
        self.model.global_slicer = current_slicer
        try:
            for key in self.objects:
                obj = self.objects[key]
                obj.mapper().SetClippingPlanes(self.model.global_slicer)
        except Exception:
            pass
        '''

    def set_camera(position, quat, camera_distance=10000):
        """
        TODO: make it possible to store and assign a camera transformation matrix
        """
        raise NotImplementedError
        camera = vtk.vtkCamera()
        # Define the quaternion in vtk, note the swapped order
        # w,x,y,z instead of x,y,z,w
        quat_vtk = vtk.vtkQuaterniond(quat[3], quat[0], quat[1], quat[2])

        M = np.zeros((3, 3), dtype=np.float32)
        quat_vtk.ToMatrix3x3(M)
        up = [0, 1, 0]
        pos = [0, 0, camera_distance]

        camera.SetViewUp(*np.dot(M, up))
        camera.SetPosition(*np.dot(M, pos))

        p = camera.GetPosition()
        p_new = np.array(p) + position
        camera.SetPosition(*p_new)
        camera.SetFocalPoint(*position)
        return camera

    def update_camera(self, normal=None, view_up=None, scale_factor=1.5, min_distance=10000):
        """
        Update the camera frustrum
        :param normal: View normal
        :param view_up: Up axis normal
        :param scale_factor: Scale factor to make the camera closer to the target
        Smaller values than 1 make the target closer to the camera.
        """
        camera_model = self.model.cameras.current
        if camera_model is None or camera_model.target is None:
            return

        if view_up is not None:
            camera_model.view_up = view_up
            self.plot.camera.SetViewUp(*camera_model.view_up)

        focal_point = None
        try:
            focal_point = utils.get_actor_center(camera_model.target)
        except Exception:
            try:
                if len(camera_model.target) >= 3:   
                    focal_point = camera_model.target[:3]
            except TypeError:
                pass
        if focal_point is None:
            focal_point = np.zeros(3)
        
        try:
            target_dimensions = utils.get_actor_dimensions(camera_model.target)
        except Exception:
            target_dimensions = np.ones(3).astype(float) * min_distance
        max_dim = max(target_dimensions)

        if normal is None:
            normal = self.plot.camera.GetViewPlaneNormal()
            normal = np.array(normal) * -1.0

        if not self.plot.camera.GetParallelProjection():
            distance = max_dim * camera_model.distance_factor * 1 / scale_factor
        else:
            distance = max_dim * 1 / scale_factor

        # Update orthographic scale too so that it's synced with perspective
        self.plot.camera.SetParallelScale(distance / scale_factor)
        
        #self.plot.camera.SetDistance(0)
        self.plot.camera.SetFocalPoint(focal_point)
        camera_position = focal_point - distance * np.array(normal)
        self.plot.camera.SetPosition(camera_position)
        self.plot.camera.SetClippingRange([0.1, abs(distance)*4])

    def toggle_orthographic_view(self):
        """
        Toggle orthographic/perspective views
        """
        is_ortho = self.plot.camera.GetParallelProjection()
        vedo.settings.useParallelProjection = not is_ortho
        self.plot.camera.SetParallelProjection(not is_ortho)
        self.update_camera()

    def set_left_view(self):
        """
        Set left sagittal view
        """
        self.update_camera([1.0, 0.0, 0.0], self.model.Z_UP)

    def set_right_view(self):
        """
        Set right sagittal view
        """
        self.update_camera([-1.0, 0.0, 0.0], self.model.Z_UP)

    def set_anterior_view(self):
        """
        Set anterior coronal view
        """
        self.update_camera([0.0, 1.0, 0.0], self.model.Z_UP)

    def set_posterior_view(self):
        """
        Set posterior coronal view
        """
        self.update_camera([0.0, -1.0, 0.0], self.model.Z_UP)

    def set_dorsal_view(self):
        """
        Set dorsal axial view
        """
        self.update_camera([0.0, 0.0, -1.0], self.model.Y_UP)

    def set_ventral_view(self):
        """
        Set ventral axial view
        """
        self.update_camera([0.0, 0.0, 1.0], self.model.Y_UP)

    def toggle_autofocus(self):
        """
        Reset the camera target to the selected object
        """
        self.model.cameras.current.autofocus = not self.model.cameras.current.autofocus

    def view_selected(self):
        """
        Reset the camera target to the selected object
        """
        camera_model = self.model.cameras.current
        if camera_model is None or self.model.selection is None:
            return
        camera_model.target = self.model.selection
        self.update_camera()

    def move_camera_pivot(self, point=None):
        """
        Focus the camera on the current selection point
        :param point: Point to focus on
        """
        camera_model = self.model.cameras.current
        if camera_model is None:
            return
        camera_model.target = self.model.selection
        self.update_camera()
    
    def toggle_ui(self):
        """
        Toggle user interface
        """
        self.set_ui_visibility(not self.model.ui.visible)

    def toggle_hollow_mode(self):
        """
        Toggle hollow volume visualization
        """
        try:
            self.model.selection_controller.toggle_hollow()
        except Exception:
            pass

    def toggle_info(self):
        """
        Toggle visibility of information about current selection
        """
        common_value = not self.model.selection_outline_visible
        self.model.selection_outline_visible = common_value
        self.model.selection_marker_visible = common_value
        self.model.selection_color_bar_visible = common_value

        self.outline_actor.SetVisibility(self.model.selection_outline_visible)
        self.selection_marker.SetVisibility(self.model.selection_marker_visible)
        self.color_bar.SetVisibility(self.model.selection_color_bar_visible)

    def toggle_outline(self, event=None):
        """
        Toggle outline visibility
        :param event: Event (given by the event caller)
        """
        self.set_outline_visibility(not self.model.selection_outline_visible)

    def set_outline_visibility(self, on=None):
        """
        Set outline visibility
        :param on: Visibility boolean
        """
        if on is None:
            on = self.model.selection_outline_visible
        self.model.selection_outline_visible = on
        if self.outline_actor is not None:
            self.outline_actor.SetVisibility(on)

    def toggle_marker(self):
        """
        Toggle visibility of the marker
        """
        self.set_marker_visibility(not self.model.selection_marker_visible)

    def set_marker_visibility(self, on=True):
        """
        Set marker visibility
        :param on: Visibility boolean
        """
        self.model.selection_marker_visible = on
        if self.selection_marker is not None:
            self.selection_marker.SetVisibility(on)

    def toggle_info_text(self):
        """
        Toggle visibility of the marker
        """
        self.set_info_text_visibility(not self.model.selection_text_visible)

    def set_info_text_visibility(self, on=True):
        """
        Set info text visibility
        :param on: Visibility boolean
        """
        self.model.selection_text_visible = on
        if self.selection_info is not None:
            self.selection_info.SetVisibility(on)

    def toggle_color_bar(self):
        """
        Togggle the color bar visibility
        """
        self.model.selection_color_bar_visible = not self.model.selection_color_bar_visible
        if self.model.selection_color_bar_visible:
            self.draw_color_bar()
        self.set_color_bar_visibility()

    def set_info_visibility(self, visible=True, actors=None, update_color_bar=True):
        """
        Set the visibility of info data
        :param visible: Whether info data is visible or not
        :param actors: Any desired actors to be made visible/invisible
        :param update_color_bar: Whether this method also affects the color bar
        """
        self.model.selection_info_visible = visible
        if update_color_bar:
            self.model.selection_color_bar_visible = visible
            if self.model.selection_color_bar_visible:
                self.draw_color_bar()
            self.set_color_bar_visibility()
        if actors is None:
            actors = []
        if self.selection_info is not None:
            actors += [self.selection_info]
        actors += [self.selection_marker]
        for actor in actors:
            if actor is not None:
                actor.SetVisibility(visible)

    def select_object_by_name(self, name):
        raise NotImplementedError

    def select_previous_object(self):
        """
        Select the next object in the list of selectable objects
        """
        widget = self.widgets.get('selection')
        repr = widget.GetRepresentation()
        value = repr.GetValue()
        min_value = repr.GetMinimumValue()
        previous_value = max(value - 1, min_value)
        self.select(previous_value)

    def select_object_by_id(self, widget=None, event=None, value=1.0):
        """
        Select object by id (order in which objects were registered in this app)
        :param widget: Widget instance (given by the event caller)
        :param event: Event (given by the event caller)
        :param value: Object id
        """
        if widget is not None and event is not None:
            value = widget.GetRepresentation().GetValue()
        self.select(value)

    def select_next_object(self):
        """
        Select the next object in the list of selectable objects
        """
        widget = self.widgets.get('selection')
        repr = widget.GetRepresentation()
        value = repr.GetValue()
        max_value = repr.GetMaximumValue()
        next_value = min(value + 1, max_value)
        self.select(next_value)

    def toggle_pickable(self, widget=None, event=None, value=1.0):
        """
        Toggle the clickable state of an object
        :param widget: Widget instance (given by the event caller)
        :param event: Event (given by the event caller)
        :param value: 0 for disabled, 1 for enabled
        """       
        if self.model.selection is None:
            return
        try:
            pickable = self.model.selection.pickable()
            self.model.selection.pickable(not pickable)
            actors = self.model.selection_controller.get_related_actors()
            for actor in actors:
                actor.pickable(not pickable)
        except Exception as e:
            if not self.silent:
                print(e)

    def update_opacity(self, widget=None, event=None, value=1.0):
        """
        Update the alpha unit of the current volume, making it more or less transparent
        :param widget: Widget instance (given by the event caller)
        :param event: Event (given by the event caller)
        :param value: alpha unit value. If none given by the event, the value defaults to 1.0
        """
        if widget is not None and event is not None:
            value = widget.GetRepresentation().GetValue()
        if self.model.selection is None:
            return
        self.model.selection.SetVisibility(value > 0)
        if isinstance(self.model.selection_controller, VolumeController):
            self.model.selection_controller.set_opacity(value)
        else:
            self.model.selection.alpha(value)

    def update_slices_opacity(self, widget=None, event=None, value=1.0):
        """
        Update the opacity of the current volume slices
        :param widget: Widget instance (given by the event caller)
        :param event: Event (given by the event caller)
        :param value: alpha unit value. If none given by the event, the value defaults to 1.0
        """
        if widget is not None and event is not None:
            value = widget.GetRepresentation().GetValue()
        try:
            self.model.selection_controller.set_slices_opacity(value)
        except Exception:
            pass

    def play_time_series(self, widget=None, event=None):
        """
        Play/pause time series
        :param widget: Widget instance (given by the event caller)
        :param event: Event (given by the event caller)
        """
        self.model.animation_playing = not self.model.animation_playing
        if self.model.timer_id is not None:
            self.plot.interactor.DestroyTimer(self.model.timer_id)
        if self.model.animation_playing:
            self.model.timer_id = self.plot.interactor.CreateRepeatingTimer(self.model.playback_speed)

    def previous_time_series(self, offset=1, loop=True):
        """
        Previous time series
        :param offset: Offset integer. If negative, then it's like using next_time_series()
        :param loop: Whether next() goes to 0 when it reached the end of the time series or not
        :return: Returns whether the next time series is valid (within range of the time series)
        """
        if self.model.selection is None:
            return
        # TODO: handle the case where the last value could be invalid if selection changes
        self.set_time_series(self.last_time_series_value - offset)

    def next_time_series(self, offset=1, loop=True):
        """
        Next time series
        :param offset: Offset integer, can be negative to go backwards
        :param loop: Whether next() goes to 0 when it reached the end of the time series or not
        :return: Returns whether the next time series is valid (within range of the time series)
        """
        if self.model.selection is None:
            return
        self.set_time_series(self.last_time_series_value + offset)

        # TODO: below code would require that the slider min-max is set on object selection
        return
        slider = self.widgets.get('time_series')
        value = int(slider.GetRepresentation().GetValue())
        min_value = int(slider.GetRepresentation().GetMinimumValue())
        max_value = int(slider.GetRepresentation().GetMaximumValue())
        new_value = value + offset
        if new_value > max_value and loop:
            # Reached the end, playing from 0
            new_value = min_value
        elif new_value < min_value and loop:
            # Playing backwards
            new_value = max_value

        slider.GetRepresentation().SetValue(new_value)
        self.update_time_series(value=new_value)
        return loop or (min_value <= new_value <= max_value)

    def update_time_series(self, widget=None, event=None, value=None):
        """
        Update the time series
        :param widget: The slider widget (optional)
        :param event: The slider event (optional)
        :param value: The value to set. If None, then the slider value
            is used, if it's given as widget param
        """
        if widget is not None and event is not None:
            value = int(widget.GetRepresentation().GetValue())
        self.set_time_series(value)
        """
        TODO: See if it's worth implementing transfer functions tweening in order to let
        the user see a gradual change between steps in a time series instead of a jump. Probably
        quite far down in the list of features.
        if value is None:
            lut = self.model.lut
            index, next_lut = self.model.get_lut_and_id(self.model.lut_id+1)
            tweened_tf = LUTModel()
            tweened_rgb = utils.blend_maps(lut.color_map, next_lut.color_map, self.model.time)
            tweened_alpha = utils.blend_maps(lut.alpha_map, next_lut.alpha_map, self.model.time)
            tweened_tf.set_color_and_alpha(tweened_rgb, tweened_alpha)
        """

    def set_time_series(self, value, force_update=False):
        """
        Set the time series for the current selected object
        :param value: The given time step
        :param force_update: Whether an update is forced even if the
            given value is the same as the existing time step
        """
        if value is None:
            return
        if self.last_time_series_value == value and not force_update:
            return
        self.last_time_series_value = value
        actor = self.model.selection
        controller = self.model.selection_controller
        # TODO: handle case where value == string
        if isinstance(actor, vedo.Lines):
            cell_data = actor.polydata().GetCellData()
            if hasattr(cell_data, 'GetArrayNames'):
                names = cell_data.GetArrayNames()
                #name = actor.scalars_prefix + str(value)
                #point_data.SetActiveScalars(names[value])
                # Necessary for utils.Points in sphere mode (non screen-space)
                actor.mapper().SelectColorArray(names[value])
        elif isinstance(actor, vedo.Points):
            name = actor.scalars_prefix + str(value)
            actor.polydata().GetPointData().SetActiveScalars(name)
            # Necessary for utils.Points in sphere mode (non screen-space)
            actor.mapper().SelectColorArray(name)
        elif isinstance(actor, vedo.Mesh):
            name = actor.scalars_prefix + str(value)
            actor.polydata().GetPointData().SetActiveScalars(value)
        elif isinstance(actor, vedo.Volume):
            controller.model.luts.set_current(value)
            self.handle_lut_update()
            self.draw_color_bar()
        #elif isinstance(actor, vedo.Assembly):
        #elif isinstance(actor, vedo.Picture),
        #elif isinstance(actor, vtk.vtkActor2D)

    def assign_scalars(self, scalar_map=None, scalar_range=None, color_map='viridis', make_active=True):
        """
        Assign a new data set and a color map to a target
        min_value=None, max_value=None
        """
        selected_view = self.model.selection_controller
        if selected_view is None:
            return
        if isinstance(selected_view, VolumeController):
            volume_model = selected_view.model
            lut_model = volume_model.build_lut(scalar_map, scalar_range, color_map, make_active=make_active)
            selected_view.set_color_map()
            if make_active:
                self.handle_lut_update()
            self.update_ui()
            return lut_model

    def handle_lut_update(self):
        """
        Update the view with the given or current transfer function
        :param lut_model: A LUTModel (whose table property is a vtkLookupTable) 
            to set on the current volume. If None, the current one will be used.
        """
        view = self.model.selection_controller
        if view is None or not isinstance(view, VolumeController):
            return
        if view.model.luts.current is None:
            return
        view.set_color_map()
    
    def animation_callback(self, progress):
        """
        You may override this function according to your needs
        :param progress: ratio of total animation (from 0.0 to 1.0)
        """
        pass

    def update_video_duration(self, widget=None, event=None, value=None):
        """
        Update the video duration
        :param widget: The slider widget (optional)
        :param event: The slider event (optional)
        :param value: The value to set. If None, then the slider value
            is used, if it's given as widget param
        """
        if widget is not None and event is not None:
            value = widget.GetRepresentation().GetValue()
        if value is None:
            return
        self.model.video_duration = int(value)

    def update_video_start_angle(self, widget=None, event=None, value=None):
        """
        Update the video start angle
        :param widget: The slider widget (optional)
        :param event: The slider event (optional)
        :param value: The value to set. If None, then the slider value
        is used, if it's given as widget param
        """
        if widget is not None and event is not None:
            value = widget.GetRepresentation().GetValue()
        if value is None:
            return
        self.model.video_start_angle = int(value)

    def update_video_end_angle(self, widget=None, event=None, value=None):
        """
        Update the video end angle
        :param widget: The slider widget (optional)
        :param event: The slider event (optional)
        :param value: The value to set. If None, then the slider value
            is used, if it's given as widget param
        """
        if widget is not None and event is not None:
            value = widget.GetRepresentation().GetValue()
        if value is None:
            return
        self.model.video_end_angle = int(value)

    def toggle_volumetric_lod(self):
        """
        Toggle volumetric LOD (subsampling)
        """
        self.volumetric_lod(not self.model.interactive_volume_subsampling)

    def volumetric_lod(self, on=False):
        """
        Enable or disable volumetric LOD subsampling.
        Turn this on when you want faster interactive visualizations
        :param on: Whether volumetric subsampling is activated
        """
        if self.model.interactive_volume_subsampling == on:
            return
        self.model.interactive_volume_subsampling = on
        for target in self.controllers_map:
            controller = self.controllers_map.get(target)
            if isinstance(controller, VolumeController):
                controller.set_interactive_subsampling(on)

    def export_image(self, file_name='iblviewer.png', width=None, height=None, scale=2):
        """
        Export the current image to a file
        :param file_name: File name with extension. PNG by default
        :param width: Width in pixels. If None, the width of the window is used.
        :param height: Height in pixels. If None, the height of the window is used.
        :param scale: Scale factor to make the image larger. Defaults to 2.
        """
        ui_visibility = self.model.ui.visible
        self.set_ui_visibility(False, ui_button_visible=False)
        self.render(file_name, width, height, scale)
        self.set_ui_visibility(ui_visibility)

    def export_turntable_video(self, file_name='iblviewer.mp4', start_angle=0, 
                                end_angle=360, duration=None, fps=25):
        """
        Export a sagittal turntable video of the viewer.
        :param file_name: File name
        :param start_angle: Start angle
        :param end_angle: End angle. If it's the same value as start angle, then 360 is added
        :param duration: Duration of the video
        :param fps: Frames per second, defaults to 25
        """
        if self.model.ui.embed:
            start_angle = self.widgets.get('start_angle').GetRepresentation().GetValue()
            end_angle = self.widgets.get('end_angle').GetRepresentation().GetValue()
            duration = self.widgets.get('video_duration').GetRepresentation().GetValue()
        #if start_angle is None or end_angle is None or duration is None or fps is None:
            #return
        if start_angle == end_angle:
            end_angle += 360

        video = vedo.Video(file_name, duration=duration, backend='ffmpeg', fps=fps)
        
        # Disable volumetric LOD for video-making
        self.volumetric_lod(False)
        
        ui_visibility = self.model.ui.visible
        self.set_ui_visibility(False, ui_button_visible=False)

        start = 0
        end = int(duration * fps)
        angle_step = (end_angle - start_angle) / end
        #axes = [0, 1, 2]
        for step in range(start, end):
            normal = np.zeros(3) * 1.0
            t = step * angle_step / 180 * math.pi
            normal[0] = math.cos(t)
            normal[1] = math.sin(t)
            self.update_camera(normal, self.model.Z_DOWN)
            self.animation_callback(step / (end - start))
            self.render()
            video.addFrame()
        
        # Disable volumetric LOD for video-making
        self.volumetric_lod(self.model.interactive_volume_subsampling)
        self.set_ui_visibility(ui_visibility)
        video.close()

    '''
    TODO: record a video from offscreen buffer
    def record_video(self):
        video = Video(video_file_path, fps=fps, duration=duration)
        self.show(interactive=False, resetcam=False)
        video.addFrame()
        video.close()
    '''
