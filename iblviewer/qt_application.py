from enum import auto
import sys
from PyQt5 import Qt
from PyQt5 import QtCore
from PyQt5 import QtWidgets
from ipywebrtc.webrtc import VideoStream

# You may need to uncomment these lines on some systems:
#import vtk.qt
#vtk.qt.QVTKRWIBase = "QGLWidget"
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib as mpl

from iblviewer.application import Viewer
from iblviewer.mouse_brain import MouseBrainViewer
from vedo import Plotter
import numpy as np

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from threading import *


class MplCanvas(FigureCanvasQTAgg): #or simply FigureCanvas?
    """
    Matplotlib statistics
    """
    def __init__(self, parent=None, width=5, height=4, dpi=100, background_color='white'):
        fig = Figure(figsize=(width, height), dpi=dpi, facecolor=background_color)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)


class ViewerWindow(Qt.QMainWindow):
    """
    This is the main window container that holds the UI, the 3D viewer and statistics
    """
    
    def __init__(self, parent=None):
        """
        Constructor
        """
        Qt.QMainWindow.__init__(self, parent)
        self.frame = Qt.QFrame()

        self.ui_layout = Qt.QVBoxLayout()
        self.main_layout = Qt.QHBoxLayout()

        self.frame.setLayout(self.main_layout)
        self.setCentralWidget(self.frame)

        self.left = 10
        self.top = 10
        self.title = 'International Brain Laboratory Viewer'
        self.width = 1920
        self.height = 1280

        self.viewer = None
        self.viewer_function = None
        self.statistics_function = None
        self.dark_mode = False
        self.kwargs = {}

        self.auto_complete_data_changed = False
        self.regions_search_names = None

        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        # Overwrite some plot properties to integrate it within Qt
        self.vtkWidget = QVTKRenderWindowInteractor(self.frame)
        self.plot = Plotter(qtWidget=self.vtkWidget)
        self.viewer_initialized = False
        self.neuroscience_context = False
        self.statistics_visible = True

    def initialize(self, viewer=None, callable=None, stats_callable=None, dark_mode=True, **kwargs):
        """
        Initialize the viewer app
        :param viewer: Either iblviewer.application.Viewer or 
            iblviewer.mouse_brain.MouseBrainViewer
        :param callable: A function that you pass to this method. You must write
            that function with a required parameter (the viewer). This allows
            you to perform actions with the viewer before content is shown in QT UI.
            It would also be possible to add a python console in QT in order to run
            Python code live like in Jupyter notebooks but this isn't implemented yet.
        :param stats_callable: A function that will be executed every time the plot is updated
            for instance when a new object or sub selecton is made
        :param dark_mode: Whether the app is in dark mode or not
        """
        self.viewer_function = callable
        self.statistics_function = stats_callable
        self.dark_mode = dark_mode
        if isinstance(kwargs, dict):
            self.kwargs = kwargs
        if isinstance(viewer, MouseBrainViewer) or isinstance(viewer, Viewer):
            self.viewer = viewer
        if self.viewer is None:
            self.viewer = MouseBrainViewer()
        self.neuroscience_context = isinstance(self.viewer, MouseBrainViewer)

        self.initialize_ui()

        # It's important to start the viewer in another thread so that the QT UI
        # doesn't freeze when interacting with it.
        # In case you need to go further with this, look at QThread with a good summary here:
        # https://realpython.com/python-pyqt-qthread/#using-qthread-to-prevent-freezing-guis
        thread = Thread(target=self._initialize_viewer)
        thread.start()

    def _initialize_viewer(self):
        """
        Initialize the viewer
        """
        #if viewer.plot is not None:
            #viewer.plot.close()
        if 'embed_ui' in self.kwargs:
            self.kwargs['embed_ui'] = False
        #if 'offscreen' in self.kwargs:
            #self.kwargs['offscreen'] = True
        if self.viewer is None:
            self.viewer = MouseBrainViewer()

        self.viewer.initialize(plot=self.plot, dark_mode=self.dark_mode, **self.kwargs)
        if self.viewer_function is not None:
            try:
                # Allows users to add data to be visualized
                self.viewer_function(self.viewer)
            except Exception as e:
                print(e)
        
        # Assign functions (to mimick event callbacks)
        # Have a look at Qt signals if you want to do something more advanced
        # Basic example : https://stackoverflow.com/questions/28793440/pyqt5-focusin-out-events
        self.viewer.selection_changed = self.onSelectionChanged
        self.viewer.sub_selection_changed = self.onSelectionChanged
        self.viewer.objects_changed = self.onObjectsChanged

        self.viewer_initialized = True
        self.show_viewer()

    def initialize_ui(self):
        """
        Initialize the QT UI
        """
        self.background_color = 'white'
        mpl.rcParams['axes.spines.left'] = True
        mpl.rcParams['axes.spines.right'] = False
        mpl.rcParams['axes.spines.top'] = False
        mpl.rcParams['axes.spines.bottom'] = True
        self.set_dark_mode(self.dark_mode)
        
        self.main_layout.addLayout(self.ui_layout, 1)

        self.menu_tabs = Qt.QTabWidget()
        self.camera_menu = Qt.QWidget()
        self.tools_menu = Qt.QWidget()
        self.object_menu = Qt.QWidget()
        #self.data_menu = Qt.QWidget()
        self.export_menu = Qt.QWidget()
        self.menu_tabs.addTab(self.camera_menu, 'Camera')
        self.menu_tabs.addTab(self.tools_menu, 'Tools')
        self.menu_tabs.addTab(self.object_menu, 'Object')
        #self.menu_tabs.addTab(self.data_menu, 'Data')
        self.menu_tabs.addTab(self.export_menu, 'Export')

        #camera_group = QtWidgets.QGroupBox('Camera presets')
        self.camera_menu.layout = Qt.QVBoxLayout()
        hbox = Qt.QHBoxLayout()
        self.add_button('Left', self.onLeftCameraPreset, hbox)
        self.add_button('Right', self.onRightCameraPreset, hbox)
        self.camera_menu.layout.addLayout(hbox)
        hbox = Qt.QHBoxLayout()
        self.add_button('Dorsal', self.onDorsalCameraPreset, hbox)
        self.add_button('Ventral', self.onVentralCameraPreset, hbox)
        self.camera_menu.layout.addLayout(hbox)
        hbox = Qt.QHBoxLayout()
        self.add_button('Anterior', self.onAnteriorCameraPreset, hbox)
        self.add_button('Posterior', self.onPosteriorCameraPreset, hbox)
        self.camera_menu.layout.addLayout(hbox)
        hbox = Qt.QHBoxLayout()
        self.add_checkbox('Autofocus', self.onToggleAutofocus, hbox, set_checked=True)
        self.add_checkbox('Orthographic', self.onToggleOrthoCamera, hbox)
        hbox.addStretch(1)
        self.camera_menu.layout.addLayout(hbox)
        hbox = Qt.QHBoxLayout()
        self.add_button('View selected', self.onViewSelection, hbox)
        self.camera_menu.layout.addLayout(hbox)
        self.camera_menu.layout.addStretch(1)
        self.camera_menu.setLayout(self.camera_menu.layout)
        #self.ui_layout.addWidget(camera_group)

        scene_group = QtWidgets.QGroupBox('Scene')
        vbox = Qt.QVBoxLayout()
        self.selection_combo = self.add_combo('Select an object', [], self.onChangeSelection, vbox)
        scene_group.setLayout(vbox)

        view_group = QtWidgets.QGroupBox('View options')
        vbox = Qt.QVBoxLayout()
        hbox = Qt.QHBoxLayout()
        self.add_checkbox('Outline', self.onToggleOutline, hbox, set_checked=True)
        self.add_checkbox('Marker', self.onToggleMarker, hbox, set_checked=True)
        self.add_checkbox('Color bar', self.onToggleColorBar, hbox, set_checked=True)
        vbox.addLayout(hbox)
        hbox = Qt.QHBoxLayout()
        self.add_checkbox('Information text', self.onToggleInfoText, hbox, set_checked=True)
        self.add_checkbox('Dark background', self.onToggleDarkBackground, hbox, set_checked=True)
        vbox.addLayout(hbox)
        hbox = Qt.QHBoxLayout()
        self.add_checkbox('Fast volumes', self.onToggleVolumesLOD, hbox)
        vbox.addLayout(hbox)
        view_group.setLayout(vbox)

        self.ui_layout.addWidget(scene_group)
        self.ui_layout.addWidget(view_group)

        if self.statistics_function is not None:
            stats_group = QtWidgets.QGroupBox('Statistics')
            hbox = Qt.QHBoxLayout()
            self.add_checkbox('Statistics panel', self.onToggleStatistics, hbox, auto_render=False)
            stats_group.setLayout(hbox)
            self.ui_layout.addWidget(stats_group)

        #object_group = QtWidgets.QGroupBox('Object settings')
        self.object_menu.layout = Qt.QVBoxLayout()
        self.opacity_slider = self.add_slider('Opacity', 1.0, 0.0, 1.0, 0.05, self.onOpacityChange, self.object_menu.layout)
        self.slices_opacity_slider = self.add_slider('Slices opacity', 1.0, 0.0, 1.0, 0.05, 
                                                    self.onSliceOpacityChange, self.object_menu.layout)
        self.hollow_checkbox = self.add_checkbox('Hollow regions', self.onToggleHollow, self.object_menu.layout)
        self.time_series_slider = self.add_slider('Time series', 1, 0, 10, 1, self.onTimeSeriesChange, self.object_menu.layout)
        self.add_button('Remove object', self.onRemoveObject, self.object_menu.layout)
        self.object_menu.layout.addStretch(1)
        self.object_menu.setLayout(self.object_menu.layout)
        #self.ui_layout.addWidget(object_group)
        #self.object_menu.layout.addWidget(object_group)

        self.tools_menu.layout = Qt.QVBoxLayout()
        self.new_probe_button = self.add_button('Add new probe', self.onNewProbe, self.tools_menu.layout)
        self.edit_probe_button = self.add_button('Edit probe', self.onEditProbe, self.tools_menu.layout)
        self.slicer_button = self.add_button('Cutter/Slicer mode', self.onSlicerToggle, self.tools_menu.layout, toggle=True)

        if self.neuroscience_context:
            self.search_input = self.add_input('Search an atlas region', None, self.tools_menu.layout, autocomplete=True)
            #self.isosurface_checkbox = self.add_checkbox('Show regions surface', None, vbox, set_checked=True)
            self.search_button = self.add_button('Get region', self.onSearch, self.tools_menu.layout)

        self.clipping_combo = self.add_combo('Select a clipping object', [], None, self.tools_menu.layout, auto_render=False)
        self.clip_to_bounds = self.add_button('Clip to object bounds', self.onClipToBounds, self.tools_menu.layout)
        self.tools_menu.layout.addStretch(1)
        self.tools_menu.setLayout(self.tools_menu.layout)

        #group = QtWidgets.QGroupBox('Data settings')
        #self.data_menu.layout = Qt.QVBoxLayout()
        #self.selection_label = QtWidgets.QLabel()
        #self.data_menu.layout.addWidget(self.selection_label)
        #self.data_menu.layout.addStretch(1)
        #self.data_menu.setLayout(self.data_menu.layout)
        #self.ui_layout.addWidget(data_group)

        #video_group = QtWidgets.QGroupBox('Video export')
        self.export_menu.layout = Qt.QVBoxLayout()
        video_group = QtWidgets.QGroupBox('Export video presets')
        vbox = Qt.QVBoxLayout()
        self.duration_slider = self.add_slider('Duration', 8, 0, 60, 1, None, vbox)
        self.start_angle_slider = self.add_slider('Start angle', 0, 0, 360, 1, None, vbox)
        self.end_angle_slider = self.add_slider('End angle', 360, 0, 360, 1, None, vbox)
        self.end_angle_slider.setTracking(True)
        self.end_angle_slider.setTickPosition(QtWidgets.QSlider.TicksAbove)
        self.export_button = self.add_button('Export video...', self.onExportVideo, vbox)
        video_group.setLayout(vbox)
        #self.video_menu.layout.addStretch(1)
        self.export_menu.layout.addWidget(video_group)

        image_group = QtWidgets.QGroupBox('Export image presets')
        vbox = Qt.QVBoxLayout()
        self.magnification_scale = self.add_slider('Magnification scale', 2, 1, 10, 1, None, vbox)
        self.export_button = self.add_button('Export image...', self.onExportImage, vbox)
        image_group.setLayout(vbox)
        self.export_menu.layout.addStretch(1)
        self.export_menu.layout.addWidget(image_group)

        self.export_menu.setLayout(self.export_menu.layout)
        #self.ui_layout.addWidget(video_group)
        
        # Once we're done with preparing the whole menu with tabs, add that to the UI
        self.ui_layout.addWidget(self.menu_tabs)
        self.ui_layout.addStretch(1)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        splitter.addWidget(self.vtkWidget)
        
        self.statistics = MplCanvas(self, 5, 4, 100, self.background_color)
        self.statistics.setStyleSheet("background-color:transparent;")
        self.update_statistics()

        # Create toolbar, passing statistics as first parament, parent (self, the MainWindow) as second.
        #toolbar = NavigationToolbar(self.statistics, self)
        #layout = Qt.QVBoxLayout()
        #layout.addWidget(toolbar)
        #layout.addWidget(self.statistics)
        #self.main_vbox.addLayout(layout)

        self.statistics_widget = Qt.QWidget()
        self.statistics_layout = Qt.QHBoxLayout(self.statistics_widget)
        self.statistics_layout.addWidget(self.statistics)
        # Don't show stats initially
        self.onToggleStatistics()

        splitter.addWidget(self.statistics_widget)
        #splitter.setStretchFactor(3, 1)
        self.main_layout.addWidget(splitter, 4)
        self.main_splitter = splitter
        #splitter.setSizes([125, 150])

        self.vtkWidget.update()
        self.show()

    def onViewerInitialized(self):
        regions_data = self.viewer.get_region_names()
        self.search_input.completer_model.setStringList(regions_data)

    def show_viewer(self):
        """
        Show the viewer (when it's initialized)
        """
        self.viewer.show()
        self.vtkWidget.update()
        self.onObjectsChanged()
        self.onViewerInitialized()
        self.update_ui()

    def set_light_mode(self):
        """
        Set light mode to viewer and stats
        """
        self.set_dark_mode(False)

    def set_dark_mode(self, on=True):
        """
        Set dark mode to viewer and stats
        :param on: Whether dark mode is on
        """
        if on:
            self.background_color = '#2d2d2d'
            plt.style.use('dark_background')
            mpl.rcParams['axes.facecolor'] = self.background_color
        else:
            self.background_color = '#dddddd'
            plt.style.use('default')
            mpl.rcParams['axes.facecolor'] = self.background_color

        if self.viewer is not None:
            self.viewer.set_dark_mode(on)
    
    def update_ui(self):
        """
        Update the QT UI
        """
        got_selection = self.viewer.model.got_selection()
        is_volume = self.viewer.is_volume()
        is_probe = self.viewer.is_probe()

        self.new_probe_button.setEnabled(got_selection and is_volume)
        self.edit_probe_button.setEnabled(got_selection and is_probe)
        #self.isosurface_checkbox.setEnabled(got_selection and is_volume)
        self.hollow_checkbox.setEnabled(got_selection and is_volume)

        self.clipping_combo.setEnabled(got_selection)
        self.clip_to_bounds.setEnabled(got_selection)
        self.slicer_button.setEnabled(got_selection)

        self.slicer_button.setChecked(self.viewer.box_widget is not None)
        
        if got_selection:
            '''
            text, data = self.viewer.get_selection_info(line_length=32)
            if text is not None:
                self.selection_label.setText(text+'\n')
            else:
                self.selection_label.setText('')
            '''
            index = self.selection_combo.findText(self.viewer.model.selection.name)
            if index != -1:
                self.selection_combo.setCurrentIndex(index)
        else:
            self.selection_combo.setCurrentIndex(0)
            #self.selection_label.setText('')
            
        opacity = self.viewer.get_selection_opacity()
        self.opacity_slider.setEnabled(opacity is not None)
        self.opacity_slider.label.setEnabled(opacity is not None)
        '''
        if opacity is None:
            self.opacity_slider.label.hide()
        else:
            self.opacity_slider.show()
            self.opacity_slider.label.show()
        '''
        if opacity is not None:
            self.opacity_slider.setValue(int(opacity / self.opacity_slider.step))

        slices_opacity = self.viewer.get_selection_slices_opacity()
        self.slices_opacity_slider.setEnabled(slices_opacity is not None)
        self.slices_opacity_slider.label.setEnabled(slices_opacity is not None)
        if slices_opacity is not None:
            self.slices_opacity_slider.setValue(int(slices_opacity / self.slices_opacity_slider.step))
        
        current_value, max_value = self.viewer.get_time_series_info()
        self.time_series_slider.setEnabled(current_value is not None)
        if isinstance(current_value, int):
            self.time_series_slider.setValue(current_value)
        if isinstance(max_value, int):
            self.time_series_slider.setRange(0, max_value)

    def update_statistics(self):
        """
        Update statistics
        """
        if not self.statistics_visible:
            return
        if self.statistics_function is not None:
            try:
                plot = self.statistics_function(self.statistics, self.viewer)
                if plot is not None:
                    self.statistics = plot
            except Exception as e:
                e_type = type(e).__name__
                msg = 'Your custom statistics function failed with error type ' + e_type
                if 'TypeError' in e_type:
                    msg += '\nMake sure your functions accepts two arguments (the statistics plot and the 3d viewer).'
                print(msg)
                print(e)
        else:
            self.statistics.axes.clear()
            self.statistics.axes.plot(np.arange(20), np.random.random(20)/2)
            #self.statistics.setStyleSheet("background-color:#eeeeee;")
            self.statistics.draw()

    def onObjectsChanged(self):
        """
        Event triggered when the dictionary of 3D objects has been updated
        """
        names = self.viewer.get_object_names()
        current_id = None
        for n_id in range(len(names)):
            if names[n_id] == self.selection_combo.currentText():
                current_id = n_id
        names = ['None'] + names
        self.selection_combo.clear()
        self.selection_combo.addItems(names)
        self.clipping_combo.clear()
        self.clipping_combo.addItems(names)
        if current_id is not None:
            self.selection_combo.setCurrentIndex(current_id)

    def onChangeSelection(self, value):
        """
        Event triggered by QT to change the viewer's selection
        """
        index = self.selection_combo.findText(value)
        if index != -1:
            self.selection_combo.setCurrentIndex(index)
        self.viewer.select(value, allow_none=True)

    def onSelectionChanged(self):
        """
        Event triggered by the viewer when a new valid selection is made
        """
        self.update_ui()
        self.update_statistics()

    def onSearch(self):
        search_term = self.search_input.text()
        if search_term == '':
            return
        result = self.viewer.find_region(search_term)
        if len(result) < 1:
            return
        # TODO: improve this and give user the choice (left or right hemisphere)
        region_id = result[0]
        #if self.isosurface_checkbox.isChecked():
        self.viewer.isosurface(region_id, split_meshes=False)

    def onOpacityChange(self, value):
        self.viewer.update_opacity(value=value)

    def onSliceOpacityChange(self, value):
        self.viewer.update_slices_opacity(value=value)

    def onTimeSeriesChange(self, value):
        self.viewer.set_time_series(value)

    @Qt.pyqtSlot()
    def onToggleStatistics(self):
        if self.statistics_visible:
            self.statistics_visible = False
            self.statistics_widget.hide()
        else:
            self.statistics_visible = True

            # Using show on the maptlotlib widget will segfault with an error
            # in mpl backend wrt to QT. So it's better to start with a new plot
            self.statistics_layout.removeWidget(self.statistics)
            self.statistics = MplCanvas(self, 5, 4, 100, self.background_color)
            self.statistics.setStyleSheet("background-color:transparent;")
            self.update_statistics()
            self.statistics_layout.addWidget(self.statistics)
            self.statistics_widget.show()

    @Qt.pyqtSlot()
    def onToggleMarker(self):
        self.viewer.toggle_marker()

    @Qt.pyqtSlot()
    def onToggleOutline(self):
        self.viewer.toggle_outline()

    @Qt.pyqtSlot()
    def onToggleColorBar(self):
        self.viewer.toggle_color_bar()

    @Qt.pyqtSlot()
    def onToggleInfoText(self):
        self.viewer.toggle_info_text()

    @Qt.pyqtSlot()
    def onToggleDarkBackground(self):
        self.viewer.toggle_dark_mode()

    @Qt.pyqtSlot()
    def onToggleVolumesLOD(self):
        self.viewer.toggle_volumetric_lod()

    @Qt.pyqtSlot()
    def onToggleMarkerType(self):
        self.viewer.toggle_marker_type()

    @Qt.pyqtSlot()
    def onToggleHollow(self):
        self.viewer.toggle_hollow_mode()

    @Qt.pyqtSlot()
    def onRemoveObject(self):
        self.viewer.remove_object()
        self.onObjectsChanged()
            
    @Qt.pyqtSlot()
    def onExportImage(self):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'QFileDialog.getSaveFileName()', 
                                                'iblviewer.jpg', 'All Files (*);;Images (*.png *.jpg);', 
                                                options=options)
        if file_path:
            self.viewer.render(file_path)
            Qt.QMessageBox.about(self, 'Image rendering complete', f'File was saved under {file_path}')
            
    @Qt.pyqtSlot()
    def onExportVideo(self):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'QFileDialog.getSaveFileName()', 
                                                'iblviewer.mp4', 'All Files (*);;Videos (*.mp4);', 
                                                options=options)
        if file_path:
            start_angle = self.start_angle_slider.value()
            end_angle = self.end_angle_slider.value()
            duration = self.duration_slider.value()
            self.viewer.export_turntable_video(file_path, start_angle, end_angle, duration)
            Qt.QMessageBox.about(self, 'Video rendering complete', f'File was saved under {file_path}')
    
    @Qt.pyqtSlot()
    def onClipToBounds(self):
        bounds_obj = self.viewer.objects.get(self.clipping_combo.currentText())
        self.viewer.clip_to_bounds(bounds_obj)

    @Qt.pyqtSlot()
    def onIsosurface(self):
        self.viewer.isosurface(split_meshes=True)

    @Qt.pyqtSlot()
    def onSlicerToggle(self):
        self.viewer.toggle_box_widget()

    @Qt.pyqtSlot()
    def onNewProbe(self):
        self.viewer.add_probe()

    @Qt.pyqtSlot()
    def onEditProbe(self):
        self.viewer.edit_probe()

    @Qt.pyqtSlot()
    def onToggleOrthoCamera(self):
        self.viewer.toggle_orthographic_view()

    @Qt.pyqtSlot()
    def onViewSelection(self):
        self.viewer.view_selected()

    @Qt.pyqtSlot()
    def onToggleAutofocus(self):
        self.viewer.toggle_autofocus()

    @Qt.pyqtSlot()
    def onLeftCameraPreset(self):
        self.viewer.set_left_view()

    @Qt.pyqtSlot()
    def onRightCameraPreset(self):
        self.viewer.set_right_view()

    @Qt.pyqtSlot()
    def onDorsalCameraPreset(self):
        self.viewer.set_dorsal_view()

    @Qt.pyqtSlot()
    def onVentralCameraPreset(self):
        self.viewer.set_ventral_view()

    @Qt.pyqtSlot()
    def onAnteriorCameraPreset(self):
        self.viewer.set_anterior_view()

    @Qt.pyqtSlot()
    def onPosteriorCameraPreset(self):
        self.viewer.set_posterior_view()

    @Qt.pyqtSlot()
    def onClose(self):
        #Disable the interactor before closing to prevent it
        #from trying to act on already deleted items
        self.vtkWidget.close()

    # Below are utility functions 

    def add_input(self, label_text, change_function=None, layout=None, autocomplete=False):
        """
        Add an input line with a label
        """
        label = QtWidgets.QLabel(self)
        label.setText(label_text)
        input = QtWidgets.QLineEdit(self)
        input.label = label
        if change_function is not None:
            input.editingFinished.connect(change_function)
        if autocomplete:
            completer = QtWidgets.QCompleter()
            model = Qt.QStringListModel()
            completer.setModel(model)
            completer.setCaseSensitivity(QtCore.Qt.CaseInsensitive)
            completer.setCompletionMode(QtWidgets.QCompleter.InlineCompletion)
            input.setCompleter(completer)
            # Store properties in order to update the model with a list later on
            input.completer_model = model
            input.completer = completer
            # If autocomplete is a list of strings, we use it as the completion list
            if isinstance(autocomplete, list):
                try:
                    input.completer_model.setStringList(autocomplete)
                except Exception:
                    pass
        if layout is not None:
            layout.addWidget(label)
            layout.addWidget(input)
        return input

    def add_button(self, label, click_function, layout=None, tooltip=None, 
                    auto_render=True, toggle=False, set_checked=False):
        """
        Add a new button to a layout
        """
        button = QtWidgets.QPushButton(label, self)
        if isinstance(tooltip, str):
            button.setToolTip(tooltip)
        def click_handler(value):
            if click_function is not None:
                click_function()
            if auto_render:
                self.viewer.render()
        button.setCheckable(toggle)
        if toggle and set_checked:
            button.setChecked(set_checked)
        button.clicked.connect(click_handler)
        if layout is not None:
            layout.addWidget(button)
        return button

    def add_checkbox(self, label, click_function, layout=None, tooltip=None, 
                    auto_render=True, set_checked=False):
        """
        Add a new checkbox to a layout
        """
        checkbox = QtWidgets.QCheckBox(label, self)
        checkbox.move(20, 0)
        if isinstance(tooltip, str):
            checkbox.setToolTip(tooltip)
        def change_handler(value):
            if click_function is not None:
                click_function()
            if auto_render:
                self.viewer.render()
        checkbox.setChecked(set_checked)
        checkbox.stateChanged.connect(change_handler)
        if layout is not None:
            layout.addWidget(checkbox)
        return checkbox

    def add_combo(self, text, values=None, change_function=None, layout=None, auto_render=True):
        """
        Add a new combobox with a label to a layout
        """
        label = QtWidgets.QLabel(self)
        label.setText(text)

        combo = QtWidgets.QComboBox(self)
        combo.label = label
        combo.last_value = None
        if values is not None:
            combo.addItems(values)
        
        def update_combo(value):
            #label.setText(f'{value}')
            if change_function is not None:
                change_function(value)
            if auto_render:
                self.viewer.render()
            combo.last_value = value

        if change_function is not None:
            combo.activated[str].connect(update_combo)
        #combo.currentIndexChanged['QString'].connect(update_combo)
        #combo.valueChanged.connect(update_combo)
        
        if layout is not None:
            layout.addWidget(label)
            layout.addWidget(combo)
        return combo

    def add_slider(self, text, value, min_value=0, max_value=10, step=1, change_function=None, 
                    layout=None, ticks=None, label_precision=2, auto_render=True):
        """
        Add a new slider with a label to a layout
        """
        label = QtWidgets.QLabel(self)
        label.setText(text)
        #label.setPixmap(QPixmap('mute.png'))

        slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
        #slider.setGeometry(50,50, 200, 50)
        #slider.move(0, 30)
        # Sliders in Qt can only accept integers (!)
        # so we have to work around that
        if max_value <= 1.0 and step < 1.0:
            min_value /= step
            max_value /= step
            value /= step
        # Custom dynamic properties
        slider.label = label
        slider.step = step
        slider.last_value = None

        slider.setMinimum(int(min_value))
        slider.setMaximum(int(max_value))
        slider.setValue(int(value))
        slider.setMinimumWidth(200)

        real_value = value
        if slider.step < 1.0:
            real_value = value * step
        if isinstance(real_value, float):
            label.setText(f'{text}: {real_value:.{label_precision}}')
        else:
            label.setText(f'{text}: {real_value}')

        if ticks is None:
            ticks = {'interval':2, 'position':'below'}
        #slider.setFocusPolicy(Qt.StrongFocus)
        slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        if ticks.get('interval') is not None:
            slider.setTickInterval(ticks.get('interval'))
        if step is not None:
            slider.setSingleStep(max(1, step))
        
        def update_slider_label(value):
            real_value = value
            if slider.step < 1.0:
                real_value = value * step
            if isinstance(real_value, float):
                label.setText(f'{text}: {real_value:.{label_precision}}')
            else:
                label.setText(f'{text}: {real_value}')
            if change_function is not None:
                change_function(real_value)
            if auto_render:
                self.viewer.render()
            slider.last_value = real_value
        
        slider.valueChanged.connect(update_slider_label)
        
        if layout is not None:
            layout.addWidget(label)
            layout.addWidget(slider)
        return slider


class ViewerApp(Qt.QApplication):
    """
    This is the main entry point to start a Qt application.
    """
    def __init__(self):
        super().__init__(sys.argv)
        self.window = ViewerWindow()
        self.aboutToQuit.connect(self.window.onClose)

    def initialize(self, viewer=None, callable=None, stats_callable=None, dark_mode=True, **kwargs):
        if dark_mode:
            # Other dark-style Qt stylesheets exist but few get it right
            from darktheme.widget_template import DarkPalette
            self.setStyle('Fusion')
            self.setPalette(DarkPalette())
            # Handling disabled states with the custom stylesheet below
            self.setStyleSheet("QToolTip { color: #ffffff; background-color: grey; border: 1px solid white; }"
            "QCheckBox:disabled {color:#999999;}"
            "QSlider::sub-page:disabled {background:#999999;}"
            "QRadioButton:disabled {color:#999999;}"
            "QWidget:disabled {color:#999999;}")
        Qt.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        self.window.initialize(viewer, callable, stats_callable, dark_mode, **kwargs)
        Qt.QApplication.restoreOverrideCursor()
        self.exec_()

def main():
    app = ViewerApp(dark_mode=False)

if __name__ == "__main__":
    main()