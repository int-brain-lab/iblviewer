# IBL Viewer
Lightweight and fast brain atlas visualization tool that uses accelerated GPU volume rendering and many features that VTK provides. This runs on python 3.8+ and there is work in progress to have it fully working in Jupyter notebooks.

## Examples [early WIP]
Example screenshot of single plane slicing with colours from Allen Brain Atlas CCF v3. Multiple planes possible, part of WIP.
![Demo 1](preview/viewer_atlas_colouring.jpg?raw=true)

Example with custom data such as priors scalar values per region. Regions without given values are light gray.
![Demo 2](preview/viewer_custom_data_sample.jpg?raw=true)

## Architecture
This application relies on the well-known pattern MVC (Model-View-Controller).
By decoupling these elements, it is easy to extend the application and customize it for your needs.

## Requirements
python3.8+, numpy, pandas, vtk, pynrrd, matplotlib, ibllib, vedo
If you install ibllib, you should only need to install vedo

## Installation
This is the current (February 2021) installation procedure with pip.
Later a wheel will be published so that you may pip install iblviewer.

If you use a conda environment like iblenv, make sure you've got it activated.
```bash
conda activate iblenv
```
Then:
```bash
pip install git+https://github.com/marcomusy/vedo.git
pip install git+https://github.com/int-brain-lab/iblviewer.git
```

If at some point it complains about failing to uninstall vtk, it's possible that the vtk installed within the conda environment causes troubles (even if it's the proper version).
Run the following:
```bash
conda uninstall vtk
pip install git+https://github.com/int-brain-lab/iblviewer.git
```
This will uninstall vtk and reinstall it (version 9+) through pip.

## Examples 
Sample code to run the viewer
```python
resolution = 50 # units = um
mapping = 'Allen'
controller = atlas_controller.AtlasController()
controller.initialize(resolution, mapping, embed_ui=True, jupyter=False)
# put your code here to load your data into the viewer
controller.render()
```

Have a look at the examples folder which contains several examples.
Since this tool is built on top of vedo, a wrapper for vtk that makes it easy to use, you have endless possibilities for plotting and visualizing your data. See https://github.com/marcomusy/vedo for examples

## Issues and feature request
Since this is an early version, feel free to request features and raise issues that you might stumble upon.