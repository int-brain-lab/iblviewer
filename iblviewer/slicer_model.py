from dataclasses import dataclass, field
from typing import Mapping, List, Any
import numpy as np

import vtk

@dataclass
class SlicerModel:
    NAME_XYZ_POSITIVE = ['px', 'py', 'pz']
    NAME_XYZ_NEGATIVE = ['nx', 'ny', 'nz']
    NAME_PREFIX = '[Slicer]'
    MIN_SLAB_THICKNESS = 1.0 #um

    name: str = NAME_PREFIX
    axis: int = 0
    value: float = 0.0
    bounds: np.ndarray = None
    #thickness: float = 0.0
    origin: np.ndarray = np.array([0.0, 0.0, 0.0])
    normal: np.ndarray = np.array([1.0, 0.0, 0.0])
    clipping_planes: vtk.vtkPlaneCollection = None

    def set_axis(self, axis, flip=False):
        is_xyz_axis = isinstance(axis, int) and 0 <= axis <= 2
        self.axis = axis if is_xyz_axis else -1
        if self.axis == -1:
            self.normal = np.array([1.0, 0.0, 0.0])
        elif is_xyz_axis:
            new_normal = np.zeros(3).astype(float)
            new_normal[axis] = 1.0
            self.normal = new_normal
        else:
            # The given axis is actually a normal
            self.normal = axis
        if flip:
            self.flip_normal()

    def flip_normal(self):
        self.normal *= -1.0

    def set_value(self, value):
        #if self.normal is not None and self.normal[self.axis] < 0:
            #self.value = self.model.dimensions - self.value
        self.value = value
        self.origin = self.normal * self.value