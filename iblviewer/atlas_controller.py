
from datetime import datetime
import logging
import math
import numpy as np
import os
import pandas as pd
import sched
import time

import vtk
import vedo
from vedo import *
from vedo.addons import *
import ibllib.atlas as atlas

from iblviewer.transfer_function_model import TransferFunctionModel
from iblviewer.atlas_model import AtlasModel, AtlasUIModel, CameraModel
from iblviewer.volume_model import VolumeModel
from iblviewer.slicer_model import SlicerModel

from iblviewer.atlas_view import AtlasView
from iblviewer.volume_view import VolumeView
from iblviewer.slicer_view import SlicerView
import iblviewer.utils as utils


class AtlasController():

    def __init__(self):
        """
        Constructor
        """
        logging.basicConfig(level=logging.DEBUG)
        self.plot = None
        self.plot_window_id = 0
        self.num_windows = 1
        self.model = AtlasModel()
        self.view = None
        self.volume_view = None

        self.scheduler = sched.scheduler(time.time, time.sleep)

        # UI related
        self.buttons = {}
        self.sliders = {}
        self.texts = {}
        self.selection_text = None
        self.selection_point = None

    def initialize(self, resolution=25, mapping='Beryl', context=None, embed_ui=False, jupyter=False, plot=None, plot_window_id=0, num_windows=1,):
        """
        Initialize the controller, main entry point to the viewer
        :param resolution: Resolution of the atlas volume.
        Possible values are 10 (requires a lot of RAM), 25, 50, 100. Units are in microns
        :param mapping: Optional mapping value. In the context of IBL, there is 'Allen' for the standard Allen map
        and 'Beryl' (random name) which aggregates cortical layers as one.
        :param context: Context of the visualization
        :param embed_ui: Whether the UI is embed within the VTK window
        :param jupyter: Whether we're running from a jupyter notebook or not
        :param plot: A vedo Plotter instance. You can either create it by yourself before hand, in case you want
        to have multiple windows with other stats or let the controller create a new one
        :param plot_window_id: Sub-window id where the 3D visualization will be displayed
        :param num_windows: Number of subwindows, in case you want to display your own stuff later
        """
        logging.info('Starting IBL Viewer')

        settings.notebookBackend = jupyter
        embedWindow(False) # or k3d, itk, panel or False
        self.plot = Plotter(N=num_windows) if plot is None else plot
        self.plot_window_id = plot_window_id

        self.model = AtlasModel()
        self.model.initialize(resolution)
        self.model.load_volume(mapping)
        self.model.initialize_slicers()

        self.model.ui.set_context(context)
        self.model.ui.jupyter = jupyter
        self.model.ui.embed = embed_ui

        self.view = AtlasView(self.plot, self.model)
        self.view.initialize()
        self.volume_view = VolumeView(self.plot, self.model.volume, self.model)
        self.view.volume_view = self.volume_view
        
        pn = SlicerModel.NAME_XYZ_POSITIVE
        nn = SlicerModel.NAME_XYZ_NEGATIVE

        # Positive slicers
        pxs_model = self.model.find_model(pn[0], self.model.slicers)
        self.px_slicer = SlicerView(self.plot, self.volume_view, pxs_model, self.model)
        
        pys_model = self.model.find_model(pn[1], self.model.slicers)
        self.py_slicer = SlicerView(self.plot, self.volume_view, pys_model, self.model)
        
        pzs_model = self.model.find_model(pn[2], self.model.slicers)
        self.pz_slicer = SlicerView(self.plot, self.volume_view, pzs_model, self.model)
        
        # Negative slicers
        nxs_model = self.model.find_model(nn[0], self.model.slicers)
        self.nx_slicer = SlicerView(self.plot, self.volume_view, nxs_model, self.model)
        
        nys_model = self.model.find_model(nn[1], self.model.slicers)
        self.ny_slicer = SlicerView(self.plot, self.volume_view, nys_model, self.model)
        
        nzs_model = self.model.find_model(nn[2], self.model.slicers)
        self.nz_slicer = SlicerView(self.plot, self.volume_view, nzs_model, self.model)

        self.slicers = [self.px_slicer, self.py_slicer, self.pz_slicer, self.nx_slicer, self.ny_slicer, self.nz_slicer]
        """
        for slicer_name in self.model.slicers:
            slicer_model = self.model.slicers.get(slicer_name)
            slicer_view = SlicerView(self.plot, self.volume_view, slicer_model, self.model)
            #self.view.slicers.append(slicer_view)
        """

        # By default, the atlas volume is our target
        self.model.camera.target = self.volume_view.actor
        self.toggle_left_view()

        if self.model.ui.embed:
            settings.defaultFont = self.model.ui.font
            self.initialize_embed_ui(slicer_target=self.volume_view)

    def render(self):
        """
        Render the plot and let the user interact with it
        """
        logging.info('\nVisualizer started: ' + str(utils.time_diff(self.model.runtime)) + 's\n\n')
        self.plot.resetcam = False
        jupyter = self.model.ui.jupyter
        if jupyter:
            from ipyvtk_simple.viewer import ViewInteractiveWidget
            '''
            # TODO: see with Marco how we can get jupyter notebook integration working perfectly.
            # Currently this only works within Visual Studio, with a minor caveat and slower UI.
            '''
            self.plot.show(self.plot.actors, at=self.plot_window_id, resetcam=False, interactive=jupyter)
            ViewInteractiveWidget(self.plot.window)
        else:
            self.plot.show(self.plot.actors, at=self.plot_window_id, resetcam=False, interactive=True)
            exit()
            #self.plot.show(applications.Slicer2d(self.volume_actor), at=1) #p.add_plane_widget()

    def add_2d_text(self, name, text, position, color='black', justify='top-left', **kwargs):
        """
        Add a 2D text on scene
        """
        text_obj = Text2D(text, c=color, pos=position, font=self.model.ui.font, s=self.model.ui.font_scale, justify=justify, **kwargs)
        text_obj.name = name
        self.texts[name] = text_obj
        return text_obj

    def add_button(self, name, event, *args, **kwargs):
        """
        Add a left-aligned button otherwise positionning it in the UI is a nightmare
        :param *args: List of arguments for addButton()
        :param **kwargs: Dictionary of arguments for addButton()
        :return: Button
        """
        button = self.plot.addButton(event, *args, **kwargs)
        button.actor.GetTextProperty().SetJustificationToLeft()
        button.actor.SetPosition(kwargs['pos'][0], kwargs['pos'][1])
        button.name = name
        self.buttons[name] = button
        return button

    def get_slider_bounds(self, x, y, length, horizontal=True):
        """
        Get slider relative bounds to lower left corner of the window
        :param x: X position
        :param y: Y position
        :param length: Length of the slider
        :parma horizontal: Whether the length is horizontal or vertical
        :return: np 2d array with min and max coordinates
        """
        if horizontal:
            return np.array([[x, y], [x + length, y]])
        else:
            return np.array([[x, y], [x, y + length]])

    def add_slider(self, name, event_handler, min_value, max_value, value, pos, **kwargs):
        """
        Add a slider to the UI
        :param event_handler: Event handler function called when the slider is updated
        :param min_value: Min slider value
        :param max_value: Max slider value
        :param value: Default value
        :param pos: Position in the UI as a tuple or list, see get_slider_bounds for details
        :param kwargs: Dictionary of kwargs, see vedo.addons.addSlider2D() for details
        :return: 2D Slider
        """
        slider = self.plot.addSlider2D(event_handler, min_value, max_value, value, self.get_slider_bounds(*pos), **kwargs)
        slider.GetRepresentation().SetLabelHeight(0.022*kwargs['titleSize'])
        slider.name = name
        self.sliders[name] = slider
        return slider

    def update_px_slicer(self, widget=None, event=None, value=0.0):
        """
        Event handler for Slicer on X axis
        :param widget: Widget instance (given by the event caller)
        :param event: Event (given by the event caller)
        :param value: Value of X slice, defaults to 0.0
        """
        if widget is not None and event is not None:
            value = widget.GetRepresentation().GetValue()
        self.update_slicer(self.px_slicer, value)

    def update_py_slicer(self, widget=None, event=None, value=0.0):
        """
        Event handler for Slicer on Y axis
        :param widget: Widget instance (given by the event caller)
        :param event: Event (given by the event caller)
        :param value: Value of Y slice, defaults to 0.0
        """
        if widget is not None and event is not None:
            value = widget.GetRepresentation().GetValue()
        self.update_slicer(self.py_slicer, value)

    def update_pz_slicer(self, widget=None, event=None, value=0.0):
        """
        Event handler for Slicer on Z axis
        :param widget: Widget instance (given by the event caller)
        :param event: Event (given by the event caller)
        :param value: Value of Z slice, defaults to 0.0
        """
        if widget is not None and event is not None:
            value = widget.GetRepresentation().GetValue()
        self.update_slicer(self.pz_slicer, value)
    
    def update_nx_slicer(self, widget=None, event=None, value=0.0):
        """
        Event handler for Slicer on X axis
        :param widget: Widget instance (given by the event caller)
        :param event: Event (given by the event caller)
        :param value: Value of X slice, defaults to 0.0
        """
        if widget is not None and event is not None:
            value = widget.GetRepresentation().GetValue()
        self.update_slicer(self.nx_slicer, value)

    def update_ny_slicer(self, widget=None, event=None, value=0.0):
        """
        Event handler for Slicer on Y axis
        :param widget: Widget instance (given by the event caller)
        :param event: Event (given by the event caller)
        :param value: Value of Y slice, defaults to 0.0
        """
        if widget is not None and event is not None:
            value = widget.GetRepresentation().GetValue()
        self.update_slicer(self.ny_slicer, value)

    def update_nz_slicer(self, widget=None, event=None, value=0.0):
        """
        Event handler for Slicer on Z axis
        :param widget: Widget instance (given by the event caller)
        :param event: Event (given by the event caller)
        :param value: Value of Z slice, defaults to 0.0
        """
        if widget is not None and event is not None:
            value = widget.GetRepresentation().GetValue()
        self.update_slicer(self.nz_slicer, value)

    def update_slicer(self, slicer_view, value):
        """
        Update a given slicer with the given value
        :param slicer_view: SlicerView instance
        :param value: Value 
        """
        volume = self.volume_view
        model = slicer_view.model
        model.set_value(value)
        model.clipping_planes = volume.get_clipping_planes(model.axis)
        slicer_view.update()
        volume.clip_on_axis(model.origin, model.axis, model.normal)

    def update_reveal_regions(self, widget=None, event=None, value=0.0):
        """
        Update a given slicer with the given value
        :param widget: Widget instance (given by the event caller)
        :param event: Event (given by the event caller)
        :param value: Value of Z slice, defaults to 0.0
        """
        if widget is not None and event is not None:
            value = widget.GetRepresentation().GetValue()
        self.reveal_regions(value)
    
    """
    def toggle_slicers_visibility(self, widget=None, event=None, value=0):
        if widget is not None and event is not None:
            value = widget.GetRepresentation().GetValue()
        #self.x_slicer.update()
    """

    def toggle_selection_visibility(self):
        """
        Toggle visibility of selection
        """
        if self.selection_point is not None:
            self.selection_point.alpha(0 if self.selection_point.alpha > 0 else 1)
        if self.selection_text is not None:
            self.selection_text.alpha(0 if self.selection_text.alpha > 0 else 1)

    def toggle_transfer_function(self, widget=None, event=None):
        """
        Show the next transfer function. Currently this is a simple iterator on all transfer_functions stored in the atlas_model
        """
        self.model.transfer_function = self.model.next_transfer_function()
        self.handle_transfer_function_update()

    def update_alpha_unit(self, widget=None, event=None, value=1.0):
        if widget is not None and event is not None:
            value = widget.GetRepresentation().GetValue()
        self.volume_view.actor.alphaUnit(value)

    def add_callback(self, event_name, func, priority=0.0):
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
        if not self.plot.interactor:
            return None

        # Processing names is removed from original function

        def _func_wrap(iren, ename):
            x, y = self.plot.interactor.GetEventPosition()
            self.plot.renderer = self.plot.interactor.FindPokedRenderer(x, y)
            if not self.plot.picker:
                self.plot.picker = vtk.vtkPropPicker()
            self.plot.picker.PickProp(x, y, self.plot.renderer)
            self.plot.picked2d = (x,y)
            xp, yp = self.plot.interactor.GetLastEventPosition()
            actor = self.plot.picker.GetProp3D()
            delta3d = np.array([0,0,0])
            if actor:
                picked3d = np.array(self.plot.picker.GetPickPosition())
                if actor.picked3d is not None:
                    delta3d = picked3d - actor.picked3d
                actor.picked3d = picked3d
            else:
                picked3d = None

            dx, dy = x-xp, y-yp

            event_dict = utils.dotdict({
                "name": ename,
                "id": cid,
                "priority": priority,
                "at": self.plot.renderers.index(self.plot.renderer),
                "actor": actor,
                "picked3d": picked3d,
                "keyPressed": self.plot.interactor.GetKeySym(),
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

        cid = self.plot.interactor.AddObserver(event_name, _func_wrap, priority)
        return cid

    def handle_left_mouse_release(self, event):
        """
        Handle left click event. The event is triggered for any "pickable" object
        :param mesh: Selected object
        """
        actor = event.actor
        if actor is None:
            return

        pos = None
        txt = ''
        data_color = np.zeros(3).astype(float)
        if actor.name.startswith(SlicerModel.NAME_PREFIX):
            ptid = actor.closestPoint(event.picked3d, returnPointId=True)
            # Scalar values in volume are integers in this case
            value = int(actor.getPointArray()[ptid])
            txt = 'Atlas ID: ' + str(self.model.metadata.id[value]) + ' - ' + str(self.model.metadata.name[value])
            data_color = self.model.transfer_function.color_map[value, 1]
            scalar = self.model.transfer_function.scalar_map.get(value)
            if scalar is not None:
                txt += '\n\nScalar value: ' + str(scalar)
            pos = actor.points(ptid)
        """
        elif:
            neuron_id = int(mesh.name.split('n_')[1])
            region_info = self.regions_data[neuron_id]
            txt = 'Selected neuron [' + str(neuron_id) + '] in region ' + str(region_info)
            pos = mesh.pos()
        """

        #vig = vpt.vignette(txt, c='black', offset=(20,10)).followCamera()
        actors_to_add = []
        self.plot.remove(self.selection_point, render=False)
        #if self.selection_point is not None:
            #self.plot.remove(self.selection_point._caption, render=False) 
        if pos is not None:
            #inv_color = np.array([1.0, 1.0, 1.0]) - data_color
            color = np.ones(3).astype(float) if np.mean(data_color) < 0.5 else np.zeros(3).astype(float)
            self.selection_point = Cross3D(pos, s=200, thickness=0.1, c=color).pickable(0).lighting('off') # to replace with KREUZ
            #self.selection_point.caption(txt, size=(0.3,0.05))
            actors_to_add.append(self.selection_point)
        
        if self.selection_text is None:
            self.selection_text = self.add_2d_text('selection', txt, [0.2, 0.95])
            actors_to_add.append(self.selection_text)
        else:
            self.selection_text.GetMapper().SetInput(txt)

        self.plot.add(actors_to_add)

    def initialize_embed_ui(self, slicer_target=None):
        """
        Initialize VTK UI
        :param slicer_target: Slicer target # not used yet
        """
        btn_kw = self.model.ui.button_config
        tog_kw = self.model.ui.toggle_config

        self.add_callback('EndInteractionEvent', self.handle_left_mouse_release)
        # LeftButtonReleaseEvent does not work. You have to use EndInteractionEvent instead (go figure...)
        # see https://stackoverflow.com/questions/52209534/vtk-mayavi-bug-all-buttonreleaseevents-fail-yet-all-buttonpressevents-wor
        
        #self.add_button('atlas_visible', self.toggle_atlas_visibility, pos=(0.05, 0.70), states=["Atlas", "Atlas"], **tog_kw)
        self.add_button('hollow', self.toggle_hollow_mode, pos=(0.65, 0.06), states=["Hollow mode", "Hollow mode"], **tog_kw)
        #self.add_button('time_series', self.play_time_series, pos=(0.65, 0.15), states=["Play", "Pause"], **tog_kw)

        self.add_button('anterior', self.toggle_anterior_view, pos=(0.05, 0.94), states=["Anterior"], **btn_kw)
        self.add_button('left', self.toggle_left_view, pos=(0.05, 0.90), states=["Left"], **btn_kw)
        self.add_button('dorsal', self.toggle_dorsal_view, pos=(0.05, 0.86), states=["Dorsal"], **btn_kw)

        self.add_button('posterior', self.toggle_posterior_view, pos=(0.12, 0.94), states=["Posterior"], **btn_kw)
        self.add_button('right', self.toggle_right_view, pos=(0.12, 0.90), states=["Right"], **btn_kw)
        self.add_button('ventral', self.toggle_ventral_view, pos=(0.12, 0.86), states=["Ventral"], **btn_kw)
        self.add_button('ortho', self.toggle_orthographic_view, pos=(0.05, 0.82), states=["Orthographic", "Orthographic"], **tog_kw)
        #elf.add_button('selection', self.toggle_selection_visibility, pos=(0.05, 0.78), states=["Selection", "Selection"], **tog_kw)
        #self.axes_button = self.add_button(self.toggle_axes_visibility, pos=(0.05, 0.78), states=["Show axes", "Hide axes"], **tog_kw)

        # Retrieves data about hover
        #self.plot.addCallback('HoverEvent', self.hover_slice_event)
        
        s_kw = self.model.ui.slider_config

        self.add_slider('time_series', self.update_time_series, 0, 1, 0, (0.5, 0.15, 0.12), title='Time series', **s_kw)
        self.add_slider('alpha_unit', self.update_alpha_unit, 0.1, 5.0, 1.0, (0.5, 0.065, 0.12), title='Transparency', **s_kw)
        
        d = self.volume_view.model.dimensions
        if d is None:
            return
        
        self.add_slider('px', self.update_px_slicer, 0, int(d[0]), 0, pos=(0.05, 0.065, 0.12), title='+X', **s_kw)
        self.add_slider('py', self.update_py_slicer, 0, int(d[1]), 0, pos=(0.2, 0.065, 0.12), title='+Y', **s_kw)
        self.add_slider('pz', self.update_pz_slicer, 0, int(d[2]), 0, pos=(0.35, 0.065, 0.12), title='+Z', **s_kw)

        self.add_slider('nx', self.update_nx_slicer, int(-d[0]), 0, 0, pos=(0.05, 0.15, 0.12), title='-X', **s_kw)
        self.add_slider('ny', self.update_ny_slicer, int(-d[1]), 0, 0, pos=(0.2, 0.15, 0.12), title='-Y', **s_kw)
        self.add_slider('nz', self.update_nz_slicer, int(-d[2]), 0, 0, pos=(0.35, 0.15, 0.12), title='-Z', **s_kw)

    def update_ui(self):
        """
        Update the UI
        """
        # TODO: add more code to handle more cases
        tfb = self.buttons.get('transfer_function')
        btn_kw = self.model.ui.button_config
        num_tfs = len(self.model.transfer_functions)
        #if num_tfs > 1 and tfb is None:
            #self.add_button('transfer_function', self.toggle_transfer_function, pos=(0.05, 0.62), states=["Next volume data", "Next volume data"], **btn_kw)
        
        slider = self.sliders.get('time_series')
        if slider is not None:
            slider.GetRepresentation().SetMaximumValue(num_tfs-1)

    def interpolate_transfer_functions(self):
        self.model.time += 0.01
        self.update_time_series()

    def play_time_series(self, widget=None, event=None):
        """
        Play/pause time series
        """
        button = self.buttons.get('time_series')
        if button is not None:
            button.switch()
            self.model.animation_playing = not self.model.animation_playing

        # Work in progres
        return

        num_tfs = len(self.model.transfer_functions)
        for i in range(num_tfs):
            self.next_time_series()
            time.sleep(0.5)

        return
        '''
        if self.model.animation_playing:
            self.handle_next_time_series(self.next_time_series)
            self.scheduler.run()
        else:
            #if self.schedule and self.event:
            self.scheduler.cancel(self.model.animation_function)
        '''
        '''
        #self.model.animation_function = sched.scheduler(time.time, time.sleep)

        if self.model.animation_playing:
            def do_something(sc):
                self.model.animation_function.enter(50, 1, self.next_time_series, (sc,))

            #self.model.animation_function.enter(50, 1, do_something, (s,))
            self.model.animation_function.run()
        else:
            self.model.animation_function.cancel()
        '''

    '''
    TODO: record a video from offscreen buffer
    def record_video(self):
        video = Video(video_file_path, fps=fps, duration=duration)
        self.show(interactive=False, resetcam=False)
        video.addFrame()
        video.close()
    '''

    def handle_next_time_series(self, action, actionargs=()):
        if self.model.animation_playing:
            self.animation_function = self.scheduler.enter(1.0, 1, self.handle_next_time_series, (action, actionargs))
            action(*actionargs)

    def next_time_series(self, offset=1):
        """
        Next time series
        :param offset: Offset integer, can be negative to go backwards
        """
        slider = self.sliders.get('time_series')
        value = int(slider.GetRepresentation().GetValue())
        print('Going to the next value', value)
        self.update_time_series(value + offset)
        slider.GetRepresentation().SetValue(value + offset)
        self.plot.show()

    def update_time_series(self, widget=None, event=None, value=None):
        """
        Update the time series
        """
        if widget is not None and event is not None:
            value = int(widget.GetRepresentation().GetValue())

        """
        WORK IN PROGRESS: THE IDEA IS TO TWEEN TRANSFER FUNCTIONS TO HAVE SMOOTH ANIMATIONS
        if value is None:
            this_tf = self.model.transfer_function

            index, next_tf = self.model.get_transfer_function_and_id(self.model.transfer_function_id+1)

            tweened_tf = TransferFunctionModel()
            tweened_rgb = utils.blend_maps(this_tf.color_map, next_tf.color_map, self.model.time)
            tweened_alpha = utils.blend_maps(this_tf.alpha_map, next_tf.alpha_map, self.model.time)
            tweened_tf.set_color_and_alpha(tweened_rgb, tweened_alpha)
        """

        #self.model.transfer_function.set_time_series_index(value)
        self.model.set_transfer_function(value)
        self.handle_transfer_function_update()

    def add_transfer_function(self, scalar_map, color_map, alpha_map=None, make_current=True):
        """
        Add a new transfer function to the model and optionally make it current and refresh the view
        :param scalar_map: Dictionary of key-value pairs. The key is the label (the value of the scalar in the volume)
        and the value is whatever custom data you want to assign to it.
        :param make_current: Whether it becomes the current transfer function and the view is refreshed
        """
        tf_model = TransferFunctionModel()
        tf_model.name = 'Transfer function ' + str(len(self.model.transfer_functions))
        tf_model.set_data(scalar_map, color_map, alpha_map)
        
        self.model.store_model(tf_model, self.model.transfer_functions)
        self.update_ui()
        if make_current:
            self.model.transfer_function = tf_model
            self.handle_transfer_function_update()

    def handle_transfer_function_update(self, transfer_function=None):
        """
        Update the view with the given or current transfer function
        :param transfer_function: Transfer function to set on the view. If None, the current one will be used.
        """
        tf = transfer_function if transfer_function is not None else self.model.transfer_function
        if tf is None:
            logging.error('[AtlasController.handle_transfer_function_update()] failed to retrieve a transfer function and cannot update view.')
            return
        #logging.info('[AtlasController] updating the view with transfer function ' + tf.name)
        # Cannot easily set the LUT to the volume so we use the color map and alpha map and let vedo build appropriate stuff
        self.volume_view.set_color_map(tf.color_map, tf.alpha_map)
        for slicer in self.slicers:
            slicer.apply_lut(tf.lut)

    def reveal_regions(self, regions): #invert=False
        """
        Reveal some labelled regions in a segmented volume
        :param regions: Scalar values representing each label. Can be a single value too.
        """
        if regions is None or regions == 0:
            self.view.volume_view.set_alpha_map(None)
            return
        if isinstance(regions, int) or isinstance(regions, float):
            regions = [int(regions)]
        regions = np.array(regions).astype(int)
        alpha_map = self.model.get_regions_mask(regions)
        self.view.volume_view.set_alpha_map(alpha_map, 1.0)

    def toggle_hollow_mode(self):
        """
        Toggle hollow volume visualization
        """
        button = self.buttons.get('hollow')
        if button is not None:
            button.switch()

        volume_property = self.view.volume_view.actor.GetProperty()
        # Shout at VTK devs: it's twisted to name properties Disable and then have DisableOff...
        disabled = bool(volume_property.GetDisableGradientOpacity())
        if disabled:
            volume_property.DisableGradientOpacityOff()
            alpha_gradient = vtk.vtkPiecewiseFunction()
            alpha_gradient.AddPoint(0, 0.0)
            alpha_gradient.AddPoint(1, 0.75)
            alpha_gradient.AddPoint(2, 1.0)
            volume_property.SetGradientOpacity(alpha_gradient)
        else:
            volume_property.DisableGradientOpacityOn()

    def toggle_orthographic_view(self):
        """
        Toggle orthographic/perspective views
        """
        self.model.camera.orthographic = not self.model.camera.orthographic
        settings.useParallelProjection = not settings.useParallelProjection

        button = self.buttons.get('ortho')
        if button is not None:
            button.switch()
        self.plot.camera.SetParallelProjection(settings.useParallelProjection)
        self.update_camera()

    def update_camera(self, normal=None, view_up=None, scale_factor=0.75, min_distance=1000):
        """
        Update the camera frustrum
        :param normal: View normal
        :param view_up: Up axis normal
        :param scale_factor: Scale factor. 
        Smaller values than 1 make the target closer to the camera.
        """
        camera_model = self.model.camera

        if view_up is not None:
            camera_model.view_up = view_up
        focal_point = utils.get_actor_center(camera_model.target)
        if focal_point is None:
            focal_point = np.zeros(3)
        
        target_dimensions = utils.get_actor_dimensions(camera_model.target)
        max_dim = max(target_dimensions)
        # Update orthographic scale too so that it's synced with perspective
        self.plot.camera.SetParallelScale(max_dim * scale_factor)

        if normal is None:
            normal = self.plot.camera.GetViewPlaneNormal()
            normal = np.array(normal) * -1.0

        if not camera_model.is_orthographic():
            distance = max_dim * camera_model.distance_factor * scale_factor
        else:
            distance = max_dim * scale_factor
        
        camera_model.focal_point = focal_point
        camera_model.distance = distance
        
        self.plot.camera.SetViewUp(*camera_model.view_up)
        #self.plot.camera.SetDistance(0)
        self.plot.camera.SetFocalPoint(camera_model.focal_point)
        camera_position = camera_model.focal_point - distance * np.array(normal)
        self.plot.camera.SetPosition(camera_position)
        self.plot.camera.SetClippingRange([0.1, abs(distance)*4])

    def toggle_focus(self):
        """
        Toggle coronal view
        """
        button = self.buttons.get('focus')
        if button is not None:
            button.switch()

        if self.selection_point is None:
            return
        """
        camera_position = self.plot.camera.GetPosition()
        camera_position = np.array(camera_position)
        camera_position += self.model.camera.target.pos() - self.selection_point.pos()
        """
        #self.model.camera.focal_point = utils.get_actor_center(self.selection_point)
        #self.plot.camera.SetFocalPoint(self.model.camera.focal_point)
        '''
        if self.model.camera.target != self.volume_view.actor:
            self.model.camera.target = self.volume_view.actor
        else:
            self.model.camera.target = self.selection_point
        self.update_camera()
        '''

    def toggle_anterior_view(self):
        """
        Toggle anterior view
        """
        self.update_camera([0.0, 1.0, 0.0], [0.0, 0.0, -1.0])

    def toggle_dorsal_view(self):
        """
        Toggle dorsal view
        """
        self.update_camera([0.0, 0.0, 1.0], [1.0, 0.0, 0.0])

    def toggle_left_view(self):
        """
        Toggle left view
        """
        self.update_camera([1.0, 0.0, 0.0], [0.0, 0.0, -1.0])

    def toggle_posterior_view(self):
        """
        Toggle coronal view
        """
        self.update_camera([0.0, -1.0, 0.0], [0.0, 0.0, -1.0])

    def toggle_ventral_view(self):
        """
        Toggle ventral view
        """
        self.update_camera([0.0, 0.0, -1.0], [1.0, 0.0, 0.0])

    def toggle_right_view(self):
        """
        Toggle right view
        """
        self.update_camera([-1.0, 0.0, 0.0], [0.0, 0.0, -1.0])


    '''
    # TO BE REFACTORED

    def on_event(self, iren, event):
        # TODO: WIP, use this for hover or other events
        printc(event, 'happened at position', iren.GetEventPosition())
        x, y = iren.GetEventPosition()
        # print('_mouseleft mouse at', x, y)

        renderer = iren.FindPokedRenderer(x, y)
        picker = vtk.vtkPropPicker()
        picker.PickProp(x, y, renderer)

        actor = picker.GetActor()
        print('trying actor')
        if not actor:
            actor = picker.GetAssembly()
            print('trying assembly')

        if not actor:
            actor = picker.GetProp3D()
            print('trying prop3d')

        if not hasattr(actor, "GetPickable"):
            print(actor, 'has no pickable attribute') 
        if actor is not None and not actor.GetPickable():
            print(actor, 'is not pickable')

        picked3d = picker.GetPickPosition()
        picked2d = np.array([x,y])
        if actor is not None and hasattr(actor, 'name'):
            print('Actor', actor.name, 'at', picked3d)
        elif actor is not None:
            print('Got actor but no name') 

    def toggle_axes_visibility(self):
        self.axes_button.switch()
        pass

    def toggle_atlas_visibility(self):
        self.atlas_button.switch()
        if self.atlas_visible:
            """
            self.plot.remove(self.volume_actor)
            self.plot.remove(self.volume_actor.scalarbar)
            self.plot.remove(self.active_slicer.slice)
            """
            self.volume_actor.alpha(0)
            self.plot.remove(self.volume_actor.scalarbar)
            self.plot.remove(self.active_slicer.slice)
        else:
            self.volume_actor.alpha(self.opacity_map * 0.001)
            self.volume_actor.addScalarBar(useAlpha=False)
            self.plot.add(self.volume_actor)
            self.plot.add(self.volume_actor.scalarbar)
            self.active_slicer.update()
            #self.plot.add([self.volume_actor, self.volume_actor.scalarbar, self.active_slicer.slice])
        self.atlas_visible = not self.atlas_visible
    '''