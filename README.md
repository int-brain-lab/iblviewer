# IBL Viewer
A lightweight and fast brain atlas visualization tool based on VTK that uses (GPU) accelerated volume and surface rendering. This runs on python 3.8+ and there is work in progress to have it fully integrated in Jupyter notebooks.

## Requirements
python3.8+, numpy, pandas, vtk, pynrrd, matplotlib, ibllib, vedo.

If you install and use ibllib, you should only need to install vedo.

## Installation
This is the current (February 2021) installation procedure with pip.
Later a wheel will be published so that you may "pip install iblviewer".

If you use a conda environment like iblenv, make sure you've got it activated.
```bash
conda activate iblenv
```
Then:
```bash
pip install git+https://github.com/marcomusy/vedo.git
pip install git+https://github.com/int-brain-lab/iblviewer.git
```

## Troubleshooting
If at some point it complains about failing to uninstall vtk, it's likely that vtk is already installed within your conda environment and that it causes troubles (even if it's the proper version).
Run the following:
```bash
conda uninstall vtk
pip install vtk
```
This will uninstall vtk and reinstall it (version 9+) with pip.

## Updating
If you have installed IBLViewer (see below) and you to update to the latest version, run:
```bash
pip install -U git+https://github.com/int-brain-lab/iblviewer.git
```
This works on conda environments as well.

In rare cases like observed on Windows once, updating fails and the user doesn't know about it. Reinstall iblviewer:
```bash
pip uninstall iblviewer
pip install git+https://github.com/int-brain-lab/iblviewer.git
```

## Examples
Insertion probes, or how to display lines made of an heterogeneous amounf of points.
![Demo 1](preview/insertion_probes.gif?raw=true)

Time series of values assigned to brain regions
![Demo 2](preview/volume_scalars.gif?raw=true)

Point neurons and connectivity data
![Demo 3](preview/point_neurons.gif?raw=true)

## Architecture
This application relies on the well-known pattern MVC (Model-View-Controller).
By decoupling these elements, it is easy to extend the application and customize it for your needs.
Do not forget to check a similar tool for surface rendering, called brainrender.

## Examples 
Sample code to run the viewer
```python
from iblviewer import atlas_controller

resolution = 50 # units = um
mapping = 'Allen'
controller = atlas_controller.AtlasController()
controller.initialize(resolution, mapping, embed_ui=True, jupyter=False)
# put your code here to load your data into the viewer
controller.render()
```

Have a look at the examples folder which contains several cases for loading dynamic (database) or static (pickle) data. Since this tool is built on top of vedo, a wrapper for vtk that makes it easy to use, you have endless possibilities for plotting and visualizing your data. See https://github.com/marcomusy/vedo for (lots of) examples.

## Issues and feature request
Feel free to request features and raise issues.

## Author
Nicolas Antille, International Brain Laboratory, 2021

## Special thanks
Thanks Marco Musy and Federico Claudi for their support in using vedo. Check out the tool that Federico made, called brainrender, a tool that leverages surface rendering to create great scientific visualizations and figures.
From International Brain Laboratory:
Thanks professor Alexandre Pouget, Berk Gerçek, Leenoy Meshulam and Alessandro Santos for their constructive feedbacks and guidance.
Thanks Olivier Winter, Gaelle Chapuis and Shan Shen for their support on the back-end side.
