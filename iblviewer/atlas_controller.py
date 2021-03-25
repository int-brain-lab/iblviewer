
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
import ibllib.atlas as atlas

from iblviewer.transfer_function_model import TransferFunctionModel
from iblviewer.atlas_model import AtlasModel, AtlasUIModel, CameraModel
from iblviewer.volume_model import VolumeModel
from iblviewer.slicer_model import SlicerModel

from iblviewer.atlas_view import AtlasView
from iblviewer.volume_view import VolumeView
from iblviewer.slicer_view import SlicerView
import iblviewer.utils as utils

from ipyvtk_simple.viewer import ViewInteractiveWidget


class AtlasController():

    def __init__(self):
        """
        Constructor
        """
        #logging.basicConfig(level=logging.DEBUG)
        self.plot = None
        self.plot_window_id = 0
        self.num_windows = 1
        self.model = AtlasModel()
        self.view = None

        self.volumes = []
        self.surfaces = []
        self.slicers = []

        #self.scheduler = sched.scheduler(time.time, time.sleep)
        # This block of vars will be moved to the model later on
        self.last_mouse_position = None
        self.left_mouse_down = False
        self.last_mouse_time = datetime.now()
        self.hover_id = None
        self.line = None

        # UI related
        self.buttons = {}
        self.sliders = {}
        self.texts = {}
        self.info_text = None
        self.selection_point = None

    def initialize(self, resolution=25, mapping='Beryl', volume_mode=None, context=None, embed_ui=False, jupyter=False, plot=None, plot_window_id=0, num_windows=1, render=False):
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
        :param render: Whether rendering occurs at the end of the initialization or not. Defaults to False
        """
        logging.info('Starting IBL Viewer...')

        # This line is necessary as for some reason, vedo's allowInteraction kills the app when we listen to TimerEvents!
        vedo.settings.allowInteraction = False
        if jupyter:
            vedo.embedWindow('ipyvtk')
        self.plot = vedo.Plotter(N=num_windows) if plot is None else plot
        self.plot_window_id = plot_window_id
        if jupyter:
            self.plot.offscreen = True
            self.plot.window.SetOffScreenRendering(1)
            #self.plot.window.SetSize(1000, 800)

        print('IBL Viewer...')

        self.model = AtlasModel()
        self.model.initialize(resolution)
        self.model.load_allen_volume(mapping, volume_mode)
        self.model.initialize_slicers()
        
        self.model.ui.set_context(context)
        self.model.ui.jupyter = jupyter
        self.model.ui.embed = embed_ui

        self.view = AtlasView(self.plot, self.model)
        self.view.initialize()
        self.view.volume = VolumeView(self.plot, self.model.volume, self.model)
        
        pn = SlicerModel.NAME_XYZ_POSITIVE
        nn = SlicerModel.NAME_XYZ_NEGATIVE

        # Positive slicers
        pxs_model = self.model.find_model(pn[0], self.model.slicers)
        self.px_slicer = SlicerView(self.plot, self.view.volume, pxs_model, self.model)
        pys_model = self.model.find_model(pn[1], self.model.slicers)
        self.py_slicer = SlicerView(self.plot, self.view.volume, pys_model, self.model)
        pzs_model = self.model.find_model(pn[2], self.model.slicers)
        self.pz_slicer = SlicerView(self.plot, self.view.volume, pzs_model, self.model)
        
        # Negative slicers
        nxs_model = self.model.find_model(nn[0], self.model.slicers)
        self.nx_slicer = SlicerView(self.plot, self.view.volume, nxs_model, self.model)
        nys_model = self.model.find_model(nn[1], self.model.slicers)
        self.ny_slicer = SlicerView(self.plot, self.view.volume, nys_model, self.model)
        nzs_model = self.model.find_model(nn[2], self.model.slicers)
        self.nz_slicer = SlicerView(self.plot, self.view.volume, nzs_model, self.model)

        self.slicers = [self.px_slicer, self.py_slicer, self.pz_slicer, self.nx_slicer, self.ny_slicer, self.nz_slicer]

        self.handle_transfer_function_update()
        # By default, the atlas volume is our target
        self.model.camera.target = self.view.volume.actor
        # We start with a sagittal view
        self.set_left_view()

        if self.model.ui.embed:
            vedo.settings.defaultFont = self.model.ui.font
            self.initialize_embed_ui(slicer_target=self.view.volume)

        if render:
            logging.info('Initialization complete. Rendering...')
            return self.render()
        else:
            logging.info('Initialization complete.')

    def add_lines(self):
        # TODO: this method will keep the added lines created in the view
        raise NotImplementedError

    def add_points(self):
        # TODO
        raise NotImplementedError

    def add_surface(self):
        # TODO
        raise NotImplementedError

    def add_volume(self):
        # TODO
        raise NotImplementedError

    def get_window(self):
        """
        Get the plot window object. Useful for displaying this window in Jupyter notebooks for instance
        :return: iren window object
        """
        return self.plot.window

    def render(self, interactive_window=True):
        """
        Render the plot and let the user interact with it
        :param interactive_window: Whether we render and make the window interactive
        """
        self.plot.resetcam = False
        if not interactive_window:
            self.plot.render()
            return

        if self.model.ui.jupyter:
            logging.info('\nVisualizer started in Jupyter mode: ' + str(utils.time_diff(self.model.runtime)) + 's\n')
            return self.plot.show(self.plot.actors, resetcam=False, interactive=False)
            #return ViewInteractiveWidget(self.plot.window)
        else:
            logging.info('\nVisualizer started: ' + str(utils.time_diff(self.model.runtime)) + 's\n')
            self.plot.show(self.plot.actors, at=self.plot_window_id, resetcam=False, interactive=True)
            try:
                # In Ipython, this seems unecessary and yields an error
                exit()
            except Exception:
                pass

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

    def add_2d_text(self, name, text, position, color='black', justify='top-left', **kwargs):
        """
        Add a 2D text on scene
        """
        text_obj = vedo.Text2D(text, c=color, pos=position, font=self.model.ui.font, s=0.7, justify=justify, **kwargs)
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
        button.actor.SetTextScaleModeToViewport()
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

    def add_slider(self, name, event_handler, min_value, max_value, value, pos, size=14, **kwargs):
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
        slider = self.plot.addSlider2D(event_handler, min_value, max_value, value, self.get_slider_bounds(*pos), **kwargs)
        slider.GetRepresentation().SetLabelHeight(0.003 * size)
        slider.GetRepresentation().GetLabelProperty().SetFontSize(size)
        slider.GetRepresentation().SetTitleHeight(0.003 * size)
        slider.GetRepresentation().GetTitleProperty().SetFontSize(size)
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
        volume = self.view.volume
        model = slicer_view.model
        model.set_value(value)
        model.clipping_planes = volume.get_clipping_planes(model.axis)
        slicer_view.update(add_to_scene=self.model.slices_visible)
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

    def toggle_transfer_function(self, widget=None, event=None):
        """
        Show the next transfer function. Currently this is a simple iterator on all transfer_functions stored in the atlas_model
        """
        self.model.transfer_function = self.model.next_transfer_function()
        self.handle_transfer_function_update()

    def update_alpha_unit(self, widget=None, event=None, value=1.0):
        """
        Update the alpha unit of the current volume
        :param widget: Widget instance (given by the event caller)
        :param event: Event (given by the event caller)
        :param value: alpha unit value. If none given by the event, the value defaults to 1.0
        """
        if widget is not None and event is not None:
            value = widget.GetRepresentation().GetValue()
        self.view.volume.actor.alphaUnit(value)

    def toggle_atlas_visibility(self):
        """
        Toggle the visibility of the atlas
        """
        #self.atlas_button.switch()
        self.model.atlas_visible = not self.model.atlas_visible
        self.view.volume.actor._mapper.SetVisibility(self.model.atlas_visible)
        if self.view.volume.actor.scalarbar is not None:
            self.view.volume.actor.scalarbar.SetVisibility(self.model.atlas_visible)
        if self.view.volume.scalar_bar is not None:
            self.view.volume.scalar_bar.SetVisibility(self.model.atlas_visible)

    def toggle_slices_visibility(self, event=None):
        """
        Toggle slicers visibility
        :param widget: Widget instance (given by the event caller)
        :param event: Event (given by the event caller)
        """
        self.model.slices_visible = not self.model.slices_visible
        button = self.buttons.get('slices_visibility')
        if button is not None:
            button.switch()
        for slicer in self.slicers:
            if slicer.actor is not None:
                slicer.actor.SetVisibility(self.model.slices_visible)

    def toggle_info_visibility(self):
        """
        Toggle visibility of information about current selection
        """
        button = self.buttons.get('info_visibility')
        if button is not None:
            button.switch()
        self.set_info_visibility(not self.model.info_visible)

    def set_info_visibility(self, visible=True):
        """
        Set the visibility of info data
        :parma visible: Whether info data is visible or not
        """
        self.model.info_visible = visible
        actors = [self.info_text, self.selection_point, self.view.volume.scalar_bar]
        for actor in actors:
            if actor is not None:
                actor.SetVisibility(visible)

    def toggle_ui(self):
        """
        Toggle user interface
        """
        button = self.buttons.get('ui')
        if button is not None:
            button.switch()
        self.set_ui_visibility(not self.model.ui.visible)

    def set_ui_visibility(self, visible, ui_button_visible=True, info_visible=True):
        """
        Set the UI visibility
        :param visible: Whether the UI is visible or not
        :param ui_button_visible: Whether the UI button is still visible or not,
        so that the user can restore the UI by clicking on it. 
        This is not necessary in case of Jupyter NB
        :param info_visible: Whether info data is visible, usually we leave it to True
        """
        self.model.ui.visible = visible
        props = self.plot.renderer.GetViewProps()
        props.InitTraversal()
        for prop_id in range(props.GetNumberOfItems()):
            prop = props.GetNextProp()
            if prop is None:
                continue
            if prop.IsA('vtkActor2D') or prop.IsA('vtkSliderRepresentation2D'):
                #print('prop name', prop.name)
                prop.SetVisibility(visible)
        
        self.buttons.get('ui').actor.SetVisibility(ui_button_visible)
        self.set_info_visibility(info_visible)


    def handle_left_mouse_press(self, event):
        """
        Handle left mouse down event. The event is triggered for any "pickable" object
        :param mesh: Selected object
        """
        self.last_mouse_position = np.array(event.picked2d)
        self.left_mouse_down = True

        # 1. Make sure the click occured on a vtk actor
        actor = event.actor
        if actor is None:
            return

        # 2. Handle click and drag a vector along a plane
        #if actor.name.startswith(VolumeModel.VOLUME_PREFIX) and self.model.ui.is_context_probes():
        if self.model.ui.first_picked_position is None:
            return
            # WIP: we have to first disable the interactor so that we can observe a MouseMoveEvent
            #print('Starting line...')
            self.plot.interactor.GetInteractorStyle().EnabledOff()
            self.plot.interactor.GetInteractorStyle().Off()
            self.model.ui.first_picked_position = np.array(event.picked3d) # Maybe we want to find the nearest point on the nearest plane...
            # Register mouse move event here
            self.hover_id = self.add_callback('MouseMoveEvent', self.handle_mouse_hover, priority=10.0)
            
    def handle_mouse_hover(self, event):
        """
        Handle mouse hover event
        """
        # WIP
        self.model.ui.last_picked_position = np.array(event.picked3d)
        # Remove-add line?? Ideally I want to keep the same object and update it
        print('Position', event.picked3d)
        print('Num actors', self.plot.actors)
        self.plot.remove(self.line, render=False)
        print('Num actors rem', self.plot.actors)
        self.line = vedo.Line(self.model.ui.first_picked_position, self.model.ui.last_picked_position).c('black').lw(1)
        self.plot.add(self.line)

    def handle_left_mouse_release(self, event):
        """
        Handle left mouse up event.
        :param mesh: Selected object
        """
        # 1. Make sure it's left mouse context
        if not self.left_mouse_down:
            return
        self.left_mouse_down = False

        # 2. Make sure the click occured on a vtk actor
        actor = event.actor
        if actor is None:
            return

        self.remove_callback(self.hover_id)
        self.plot.interactor.GetInteractorStyle().EnabledOn()
        self.plot.interactor.GetInteractorStyle().On()

        # 3. Handle double click case
        detla_norm_2d_click = np.linalg.norm(self.last_mouse_position - np.array(event.picked2d))
        if utils.time_diff(self.last_mouse_time) < 0.5 and detla_norm_2d_click < 4:
            # Double click/tap means we focus the camera on the selection
            #logging.info('Double-click at coordinate ' + str(event.picked3d))
            self.model.camera.target = event.picked3d
            self.update_camera()

        self.last_mouse_time = datetime.now()

        # 4. Handle mouse click release with a workaround for bad VTK event mgmt (LeftButtonReleaseEvent does not work)
        if self.last_mouse_position is None or detla_norm_2d_click > 10:
            return

        # 5. Do nothing else if info should not be shown (and not computed either)
        if not self.model.info_visible:
            return

        # 6. Now handle various cases for showing what was clicked depending on the type of object selected
        position = None
        data_color = np.zeros(3).astype(float)
        atlas_pick = np.array(event.picked3d) - self.model.origin
        actors_to_add = []
        text = f'{actor.name}'
        if event.isMesh and actor.name.startswith(SlicerModel.NAME_PREFIX):
            ptid = actor.closestPoint(event.picked3d, returnPointId=True)
            # Scalar values in volume are integers in this case
            value = int(actor.getPointArray()[ptid])

            atlas_id = self.model.atlas.regions.id[value]
            region = self.model.atlas.regions.get(atlas_id)
            data_color = self.model.transfer_function.color_map[value, 1]
            scalar = self.model.transfer_function.scalar_map.get(value)

            # Here we overwrite the default object selection name
            text = f'{region.name[0].title()}\nAtlas id: {atlas_id}'
            if scalar is not None:
                text += ', value: ' + str(scalar)
            position = actor.points(ptid)

        elif event.isMesh and actor.name.startswith(AtlasModel.LINES_PREFIX):
            mesh_id = actor.closestPoint(actor.picked3d, returnCellId=True)
            data_color = np.ones(3)
            value = actor._polydata.GetCellData().GetScalars().GetValue(mesh_id)
            position = actor.picked3d
            text = f'\nElement id: {mesh_id}, value: {value}'
            #print('Selected cell', line_id, 'at', precision(mesh.picked3d,3))

        elif event.isPoints: # and actor.name.startswith(AtlasModel.POINTS_PREFIX):
            """
            # This is what you do in VTK
            point_locator = vtk.vtkPointLocator()
            point_locator.SetDataSet(poly)
            point_locator.BuildLocator()
            point_locator.FindClosestPoint(event.picked3d)
            """
            point_id = actor.closestPoint(actor.picked3d, returnPointId=True)
            data_color = np.ones(3) 
            value = actor.getPointArray(0)[point_id]
            position = actor.closestPoint(actor.picked3d)
            text += f'\nPoint: {point_id}, value: {value}'
            #np.mean(np.array(actor.closestPoint(actor.picked3d, N=1)), axis=0)
            #actor.picked3d # not the good one if spheres

        elif event.isVolume: #and actor.name.startswith(VolumeModel.VOLUME_PREFIX)
            if self.model.ui.is_context_probes() and self.model.ui.first_picked_position is not None:
                self.model.ui.first_picked_position = np.array(event.picked3d)
                self.add_probe(self.model.ui.first_picked_position, self.model.ui.last_picked_position, self.view.volume.actor)
            else:
                origin = np.array(self.plot.camera.GetPosition())
                position, value = self.find_nearest_volume_intersection_data(origin, np.array(event.picked3d))
                
                value = int(value)
                atlas_id = self.model.atlas.regions.id[value]
                region = self.model.atlas.regions.get(atlas_id)
                data_color = self.model.transfer_function.color_map[value, 1]
                scalar = self.model.transfer_function.scalar_map.get(value)

                # Here we overwrite the default object selection name
                text = f'{region.name[0].title()}\nAtlas id: {atlas_id}'
                if scalar is not None:
                    text += ', value: ' + str(scalar)
        else:
            return
        
        text += '\nPosition: ' + "{0:.2f}, {1:.2f}, {2:.2f}".format(*atlas_pick)

        #vig = vpt.vignette(txt, c='black', offset=(20,10)).followCamera()
        self.plot.remove(self.selection_point, render=False)
        #if self.selection_point is not None:
            #self.plot.remove(self.selection_point._caption, render=False) 
        if position is not None:
            if len(position) >= 3:
                #inv_color = np.array([1.0, 1.0, 1.0]) - data_color
                color = np.ones(3).astype(float) * 0.75 if np.mean(data_color) < 0.5 else np.zeros(3).astype(float)
                self.selection_point = utils.Cross3DExt(position, size=1000, thickness=2, color=color, alpha=1).pickable(0).lighting('off')
                #self.selection_point = utils.add_caption_symbol(pos)
                #self.selection_point.caption('x', size=(0.4,0.3), c='black')
                #self.selection_point.caption(txt, size=(0.3,0.05))
                actors_to_add.append(self.selection_point)
        
        if self.info_text is None:
            self.info_text = self.add_2d_text('selection', text, [0.3, 0.96])
            actors_to_add.append(self.info_text)
        else:
            self.info_text.GetMapper().SetInput(text)

        self.plot.add(actors_to_add)

    def find_nearest_volume_intersection_data(self, origin, direction_point, max_distance=-1):
        """
        Find the nearest volume intersection given a vector formed by two coordinates.
        This function relies on a trick, using surface meshes to get the proper first intersection
        and this works with slicers as well.
        :param: Origin of the vector
        :param direction_point: Second point that forms the ray/vector to intersect the volume with
        :return: The nearest position and its related value queried in the volume image
        """

        '''
        # As per C++ doc https://vtk.org/Wiki/VTK/Examples/Cxx/VTKConcepts/Scalars
        # https://stackoverflow.com/questions/35378796/vtk-value-at-x-y-z-point 
        # Unfortunately, this code picks positions "too far" in the volume 
        # but there's maybe a way it work...
        picker = vtk.vtkVolumePicker()
        picker.SetTolerance(0.001)
        picker.PickCroppingPlanesOn()
        picker.UseVolumeGradientOpacityOff()
        picker.SetVolumeOpacityIsovalue(0.1)
        picker.AddPickList(actor)
        picker.PickFromListOn()
        picker.Pick(*event.picked2d, 0, self.plot.renderer)
        return picker.GetPickPosition()
        #camera_position = np.array(self.plot.camera.GetPosition())
        #quat = self.plot.camera.GetOrientationWXYZ()
        #transform = self.plot.camera.GetProjectionTransformMatrix(self.plot.renderer)
        #quat = vtk.vtkMath.Matrix3x3ToQuaternion(transform)
        #picker.IntersectVolumeWithLine(np.array(event.picked3d) - origin)
        '''

        """
        Direct volume picking is hard to control so we use a surface mesh to make sure
        about the boundaries. [WIP]
        """
        # See how to generalize this trick to other volumes (maybe with dynamic isosurfacing)
        if max_distance == -1:
            max_distance = max(self.model.volume.dimensions)*2
        origin = np.array(origin)
        direction_point = np.array(direction_point)
        ray = direction_point - origin
        ray_norm = ray / np.linalg.norm(ray)
        direction_point = direction_point + ray_norm * max_distance
        positions = self.view.volume.surface_actor.intersectWithLine(origin, direction_point)

        distances = []
        for p_id in range(len(positions)):
            position = positions[p_id]
            dist = np.linalg.norm(position - origin)
            distances.append([p_id, dist])
        
        distances = np.array(distances)
        if len(distances) > 1:
            ordering = distances[:, 1].argsort()
            sorted_distances = distances[ordering]
            nearest_outer_atlas_position = positions[int(sorted_distances[0, 0])]
        elif len(distances) == 1:
            nearest_outer_atlas_position = positions[0]
        else:
            return None, None

        nearest_slice_position = None
        shortest_slice_distance = 10000000000
        for slicer in self.slicers:
            if slicer.actor is None:
                continue
            position = slicer.actor.intersectWithLine(origin, direction_point)
            if position is None or len(position) < 1:
                continue
            # When intersecting with a plane, we should only have one point
            position = position[0]
            dist = np.linalg.norm(position - origin)
            if dist < shortest_slice_distance:
                nearest_slice_position = position

        nearest_position = nearest_outer_atlas_position
        if nearest_slice_position is not None:
            dist = np.linalg.norm(nearest_slice_position - origin)
            a_dist = np.linalg.norm(nearest_outer_atlas_position - origin)
            if dist > a_dist:
                nearest_position = nearest_slice_position

        # Go "inside" the volume a little bit to have a valid point id
        position = nearest_position + ray_norm * self.model.volume.resolution * 3
        pt_id = self.view.volume.actor._data.FindPoint(*position)
        scalar_data = self.view.volume.actor._data.GetPointData().GetScalars()
        valid_id = 0 < pt_id < scalar_data.GetNumberOfValues()
        value = int(scalar_data.GetValue(pt_id)) if valid_id else None

        return nearest_position, value

    def add_probe(self, origin, destination, volume_actor):
        """
        Add a probe
        """
        raise NotImplementedError
        actors_to_add = []

        position, value = self.find_nearest_volume_intersection_data(origin, destination)
        '''
        #if np.linalg.norm(origin - positions[0]) < np.linalg.norm()
        pts = vedo.Points(positions, r=10).c('#dddddd')
        line = vedo.Line(origin, destination).c('black').lw(1)
        #print('Line length', np.linalg.norm(destination - origin))
        #print('Origin', origin, 'destination', destination, 'dest', destination)
        #actors_to_add.append(pts)
        actors_to_add.append(line)

        trajectory = atlas.Trajectory.fit(positions)
        # Now see what to do with a trajectory
        
        num_samples = 40
        inside_volume_distance = abs(sorted_distances[0, 1] - sorted_distances[-1, 1])
        step = inside_volume_distance / num_samples
        samples = [position + ray_norm * p_id * step for p_id in range(num_samples)]
        colors = []
        values = []
        validated_samples = []
        color_map = self.model.transfer_function.color_map
        for sample in samples:
            # Maybe use this if it is faster? To be tested
            # ijk_result = [0.0, 0.0, 0.0]
            # volume_actor._data.TransformPhysicalPointToContinuousIndex(xyz, ijk_result)
            # volume_actor._data.GetPoint(ijk_result)
            pt_id = volume_actor._data.FindPoint(*sample)
            if 0 < pt_id < scalar_data.GetNumberOfValues():
                value = int(scalar_data.GetValue(pt_id))
                if value > len(color_map):
                    value = value // 2
                colors.append(color_map[value, 1])
                values.append(color_map[value, 0])
                validated_samples.append(sample)
                #print('Sample region', self.model.metadata.id[value])
        
        pts = utils.SpheresExt(validated_samples, r=50, c=colors) 
        actors_to_add.append(pts)
        
        if 0 < pt_id < scalar_data.GetNumberOfValues():
            value = scalar_data.GetValue(pt_id)
            allen_id = self.model.atlas.regions.id[value]
            region = self.model.atlas.regions.get(allen_id)
            text = f'Atlas ID: {allen_id} - {region.name[0]}'

        self.plot.add(actors_to_add)
        '''

        '''
        // Get the ID of the point that is closest to the query position
        vtkIdType id = locator->FindClosestPoint(pt);
        // Retrieve the first attribute value from this point
        double value = polyData->GetPointData()->GetScalars()->GetTuple(id, 0);
        '''

    def initialize_embed_ui(self, slicer_target=None):
        """
        Initialize VTK UI
        :param slicer_target: Slicer target # not used yet
        """
        btn_kw = self.model.ui.button_config
        tog_kw = self.model.ui.toggle_config

        # LeftButtonReleaseEvent does not work. You have to use EndInteractionEvent instead (go figure...)
        # see https://stackoverflow.com/questions/52209534/vtk-mayavi-bug-all-buttonreleaseevents-fail-yet-all-buttonpressevents-wor
        self.add_callback('LeftButtonPressEvent', self.handle_left_mouse_press)
        self.add_callback('EndInteractionEvent', self.handle_left_mouse_release)
        
        self.plot.interactor.AddObserver('TimerEvent', self.handle_timer)
        #self.plot.addCallback('HoverEvent', self.hover_slice_event)

        self.add_button('ui', self.toggle_ui, pos=(0.95, 0.94), states=["-", "+"], **tog_kw)
        
        self.add_button('hollow', self.toggle_hollow_mode, pos=(0.05, 0.74), states=["Hollow volume", "Hollow volume"], **tog_kw)
        self.add_button('time_series', self.play_time_series, pos=(0.65, 0.14), states=["Play time series", "Pause time series"], **tog_kw)
        self.add_button('turntable_video', self.export_turntable_video, pos=(0.65, 0.08), states=["Export turntable video"], **btn_kw)

        self.add_button('anterior', self.set_anterior_view, pos=(0.05, 0.94), states=["Anterior"], **btn_kw)
        self.add_button('dorsal', self.set_dorsal_view, pos=(0.05, 0.90), states=["Dorsal"], **btn_kw)
        self.add_button('left', self.set_left_view, pos=(0.05, 0.86), states=["Left"], **btn_kw)

        self.add_button('posterior', self.set_posterior_view, pos=(0.15, 0.94), states=["Posterior"], **btn_kw)
        self.add_button('ventral', self.set_ventral_view, pos=(0.15, 0.90), states=["Ventral"], **btn_kw)
        self.add_button('right', self.set_right_view, pos=(0.15, 0.86), states=["Right"], **btn_kw)
        self.add_button('ortho', self.toggle_orthographic_view, pos=(0.05, 0.82), states=["Orthographic", "Orthographic"], **tog_kw)
        self.add_button('atlas_view', self.reset_camera_target, pos=(0.05, 0.78), states=["Reset view"], **btn_kw) # TODO: make a toggle

        #self.axes_button = self.add_button(self.toggle_axes_visibility, pos=(0.05, 0.78), states=["Show axes", "Hide axes"], **tog_kw)
        self.add_button('slices_visibility', self.toggle_slices_visibility, pos=(0.05, 0.66), states=["Slices visible", "Slices hidden"], **tog_kw)
        self.add_button('info_visibility', self.toggle_info_visibility, pos=(0.05, 0.62), states=["Info visible", "Info hidden"], **tog_kw)

        
        s_kw = self.model.ui.slider_config
        self.add_slider('time_series', self.update_time_series, 0, 1, 0, (0.5, 0.15, 0.12), title='Time series', **s_kw)
        self.add_slider('alpha_unit', self.update_alpha_unit, 0.0, 20.0, 1.0, (0.5, 0.065, 0.12), title='Transparency', **s_kw)
        
        d = self.view.volume.model.dimensions
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
    '''
    TODO: record a video from offscreen buffer
    def record_video(self):
        video = Video(video_file_path, fps=fps, duration=duration)
        self.show(interactive=False, resetcam=False)
        video.addFrame()
        video.close()
    '''

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

        if self.model.timer_id is not None:
            self.plot.interactor.DestroyTimer(self.model.timer_id)
        if self.model.animation_playing:
            self.model.timer_id = self.plot.interactor.CreateRepeatingTimer(self.model.playback_speed)

    def handle_timer(self, iren, event, **kwargs):
        """
        Handle vtk timer event
        :param iren: Render window
        :param event: TimerEvent
        """
        if self.model.animation_playing:
            #num_tfs = len(self.model.transfer_functions)
            #for i in range(num_tfs):
            valid = self.next_time_series()
            if not valid:
                self.plot.interactor.DestroyTimer(self.model.timer_id)

    def next_time_series(self, offset=1, loop=True):
        """
        Next time series
        :param offset: Offset integer, can be negative to go backwards
        :param loop: Whether next() goes to 0 when it reached the end of the time series or not
        :return: Returns whether the next time series is valid (within range of the time series)
        """
        slider = self.sliders.get('time_series')
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
        self.update_time_series(new_value)
        slider.GetRepresentation().SetValue(new_value)
        self.update_time_series(value=new_value)
        return loop or (min_value <= new_value <= max_value)

    def update_time_series(self, widget=None, event=None, value=None):
        """
        Update the time series
        """
        if widget is not None and event is not None:
            value = int(widget.GetRepresentation().GetValue())

        """
        WORK IN PROGRESS: THE IDEA IS TO TWEEN TRANSFER FUNCTIONS

        Possible hint from 10 years ago: http://vtk.1045678.n5.nabble.com/Basic-Animation-Code-td1250309.html
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

    def add_transfer_function(self, scalar_map, color_map, alpha_map=None, func=None, make_current=True):
        """
        Add a new transfer function to the model and optionally make it current and refresh the view
        :param scalar_map: Dictionary of key-value pairs. The key is the label (the value of the scalar in the volume)
        and the value is whatever custom data you want to assign to it.
        :param color_map: Color map
        :param alpha_map: Alpha map
        :param func: The function that generated the color and alpha map (aka the "color map"...function)
        :param make_current: Whether it becomes the current transfer function and the view is refreshed
        """
        tf_model = TransferFunctionModel()
        tf_model.name = 'Transfer function ' + str(len(self.model.transfer_functions))
        tf_model.set_data(scalar_map, color_map, alpha_map, func)

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
            #logging.error('[AtlasController.handle_transfer_function_update()] failed to retrieve a transfer function and cannot update view.')
            return
        #logging.info('[AtlasController] updating the view with transfer function ' + tf.name)
        # Cannot easily set the LUT to the volume so we use the color map and alpha map and let vedo build appropriate stuff
        if not self.view.volume.model.is_segmented_volume():
            return
        self.view.volume.set_color_map(tf.color_map, tf.alpha_map)
        for slicer in self.slicers:
            slicer.apply_lut(tf.lut)

    def reveal_regions(self, regions): #invert=False
        """
        Reveal some labelled regions in a segmented volume
        :param regions: Scalar values representing each label. Can be a single value too.
        """
        if regions is None or regions == 0:
            self.view.volume.set_alpha_map(None)
            return
        if isinstance(regions, int) or isinstance(regions, float):
            regions = [int(regions)]
        regions = np.array(regions).astype(int)
        alpha_map = self.model.get_regions_mask(regions)
        self.view.volume.set_alpha_map(alpha_map, 1.0)

    def toggle_hollow_mode(self):
        """
        Toggle hollow volume visualization
        """
        button = self.buttons.get('hollow')
        if button is not None:
            button.switch()

        volume_property = self.view.volume.actor.GetProperty()
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

    def update_camera(self, normal=None, view_up=None, scale_factor=1.5, min_distance=1000):
        """
        Update the camera frustrum
        :param normal: View normal
        :param view_up: Up axis normal
        :param scale_factor: Scale factor to make the camera closer to the target
        Smaller values than 1 make the target closer to the camera.
        """
        camera_model = self.model.camera

        if view_up is not None:
            camera_model.view_up = view_up
            self.plot.camera.SetViewUp(*camera_model.view_up)

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
        # Update orthographic scale too so that it's synced with perspective
        self.plot.camera.SetParallelScale(max_dim * 1 / scale_factor)

        if normal is None:
            normal = self.plot.camera.GetViewPlaneNormal()
            normal = np.array(normal) * -1.0

        if not camera_model.is_orthographic():
            distance = max_dim * camera_model.distance_factor * 1 / scale_factor
        else:
            distance = max_dim * 1 / scale_factor
        
        camera_model.focal_point = focal_point
        camera_model.distance = max(min_distance, distance)
        
        #self.plot.camera.SetDistance(0)
        self.plot.camera.SetFocalPoint(camera_model.focal_point)
        camera_position = camera_model.focal_point - distance * np.array(normal)
        self.plot.camera.SetPosition(camera_position)
        self.plot.camera.SetClippingRange([0.1, abs(distance)*4])

    def toggle_orthographic_view(self):
        """
        Toggle orthographic/perspective views
        """
        self.model.camera.orthographic = not self.model.camera.orthographic
        vedo.settings.useParallelProjection = not vedo.settings.useParallelProjection

        button = self.buttons.get('ortho')
        if button is not None:
            button.switch()
        self.plot.camera.SetParallelProjection(vedo.settings.useParallelProjection)
        self.update_camera()

    def set_left_view(self):
        """
        Set left sagittal view
        """
        self.update_camera([1.0, 0.0, 0.0], self.model.camera.z_down)

    def set_right_view(self):
        """
        Set right sagittal view
        """
        self.update_camera([-1.0, 0.0, 0.0], self.model.camera.z_down)

    def set_anterior_view(self):
        """
        Set anterior coronal view
        """
        self.update_camera([0.0, 1.0, 0.0], self.model.camera.z_down)

    def set_posterior_view(self):
        """
        Set posterior coronal view
        """
        self.update_camera([0.0, -1.0, 0.0], self.model.camera.z_down)

    def set_dorsal_view(self):
        """
        Set dorsal axial view
        """
        self.update_camera([0.0, 0.0, 1.0], self.model.camera.x_up)

    def set_ventral_view(self):
        """
        Set ventral axial view
        """
        self.update_camera([0.0, 0.0, -1.0], self.model.camera.x_up)

    def reset_camera_target(self):
        """
        Reset the camera target to the current atlas volume
        """
        self.model.camera.target = self.view.volume.actor
        self.update_camera()

    def animation_callback(self, progress):
        """
        You may rewrite this function to your needs
        """
        pass

    def export_turntable_video(self, file_name='iblviewer.mp4', start_angle=0, end_angle=360, duration=8, fps=25):
        """
        Export a sagittal turntable video of the viewer.
        TODO: implement more advanced support and options
        """
        video = vedo.Video(file_name, duration=duration, backend='ffmpeg', fps=fps) # backend='opencv'
        
        # Disable volumetric LOD for video-making
        subsampling = {}
        # TODO: see how the refactoring for multiple volumes goes
        volumes = [self.view.volume]
        for volume_view in volumes:
            subsampling[volume_view] = volume_view.model.interactive_subsampling
            volume_view.set_interactive_subsampling(False)

        ui_visibility = self.model.ui.visible
        info_visibility = self.model.info_visible
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
            self.update_camera(normal, self.model.camera.z_down)
            self.animation_callback(step / (end - start))
            video.addFrame()
        
        for volume_view in volumes:
            if subsampling.get(volume_view, False):
                volume_view.set_interactive_subsampling(True)

        self.set_ui_visibility(ui_visibility, info_visible=info_visibility)
        video.close()