
from dataclasses import dataclass, field
from typing import Mapping, List, Any
import numpy as np
import pandas as pd
import logging

import vtk
import vedo


def blend_maps(map1, map2, time, total_time):
    """
    Blend color maps
    """
    weight1 = max(0.0, total_time - time)
    weight2 = max(0.0, time)
    return map1 * weight1 + map2 * weight2


@dataclass
class TransferFunctionModel:
    name: str = None

    function: Any = None
    scalar_map: np.ndarray = None
    scalar_min: float = 0.0
    scalar_max: float = 1.0
    
    slice_color_map: np.ndarray = None
    color_map: np.ndarray = None
    alpha_map: np.ndarray = None

    sec_alpha_map: np.ndarray = None
    lut: vtk.vtkLookupTable = None
    opaque_lut: vtk.vtkLookupTable = None
    scalar_lut: vtk.vtkLookupTable = None

    def set_data(self, scalar_map, color_map, alpha_map=None, function=None):
        """
        Get a look-up table from the current color and opacity/alpha map
        :parma scalar_map: Dictionary with key-value pairs in the form volume_scalar_value -> your custom value
        :param color_map: Color map, a nested list, each row containing [volume_scalar_value, [r, g, b]]
        :param alpha_map: Alpha map, optional, a 2D list with each row as [volume_scalar_value, alpha]
        """
        self.scalar_map = scalar_map
        self.function = function

        values = scalar_map.values()
        self.scalar_min = min(values)
        self.scalar_max = max(values)

        self.color_map = np.array(color_map, dtype=object)

        slice_cmap = []
        for entry_id in range(len(self.color_map)):
            slice_cmap.append(self.color_map[entry_id][1])
        self.slice_color_map = np.array(slice_cmap, dtype=object)
        if alpha_map is not None:
            self.alpha_map = np.array(alpha_map)
        self.build_luts()

    def get_sorted_scalars(self):
        """
        Get a numpy 2D array of key-value pairs sorted by value
        :return: 2D array
        """
        sorted_scalars = np.zeros((len(self.scalar_map), 2))
        values = list(self.scalar_map.values())
        keys = list(self.scalar_map.keys())
        sorted_scalars[:, 0] = keys
        sorted_scalars[:, 1] = values
        sorted_mask = sorted_scalars[:, 1].argsort()
        sorted_scalars = sorted_scalars[sorted_mask]
        return sorted_scalars

    def build_luts(self, slice_alpha=1.0):
        """
        Build a look-up tables
        """
        # First LUT: the one that maps to scalar values in the volume
        num_values = len(self.color_map)
        lut = vtk.vtkLookupTable()
        lut.SetRange(0, num_values)
        lut.SetNumberOfTableValues(num_values)
        
        # TODO: note that the alpha_map will work if it's not same length as rgb map. Only it has 
        # to be 2D (i.e. mapping n volume scalar values to n alpha values
        for iter_id in range(num_values):
            value, rgb = self.color_map[iter_id]
            a = 1.0 if self.alpha_map is None else self.alpha_map[iter_id, 1]
            lut.SetTableValue(value, *rgb, a)
        lut.Build()
        self.lut = lut
        
        # Second LUT: same as #1 but with fixed alpha, used for slicing
        lut = vtk.vtkLookupTable()
        lut.SetRange(0, num_values)
        lut.SetNumberOfTableValues(num_values)
        for iter_id in range(num_values):
            value, rgb = self.color_map[iter_id]
            if self.alpha_map[iter_id, 1] > 0:
                a = 1.0 if self.alpha_map is None else max(self.alpha_map[iter_id, 1], slice_alpha)
            else:
                a = 0.0
            lut.SetTableValue(value, *rgb, a)
        lut.Build()
        self.opaque_lut = lut
        
        # Third LUT: ordered LUT that goes from min to max scalar value, used for displaying scale bars
        # it's actually how the user would expect the color bar to be as one doesn't know about the
        # way we use static labels and remap them with LUTs to have a fast visualization
        lut = vtk.vtkLookupTable()
        lut.SetRange(self.scalar_min, self.scalar_max)
        sorted_scalars = self.get_sorted_scalars()
        #print('')
        #print('->Bla', self.scalar_min, self.scalar_max, min(sorted_scalars[:, 1]), max(sorted_scalars[:, 1]))
        lut.SetNumberOfTableValues(len(sorted_scalars))
        iter_id = 0
        for data_id in range(len(sorted_scalars)):
            key = int(sorted_scalars[data_id, 0])
            value = float(sorted_scalars[data_id, 1])
            rgb = self.color_map[self.color_map[:, 0] == key][0][1]
            #rgb = list(vedo.colorMap(value, self.function, self.scalar_min, self.scalar_max))
            #rgb = list(vedo.colorMap(value, self.function, self.scalar_min, self.scalar_max))
            lut.SetTableValue(iter_id, *rgb, 1.0)
            iter_id += 1
        '''
        lin_range = np.linspace(self.scalar_min, self.scalar_max, num_values)
        colors = vedo.colorMap(lin_range[iter_id], self.function, self.scalar_min, self.scalar_max)
        print(lin_range)
        for iter_id in range(len(lin_range)):
            lut.SetTableValue(iter_id, *colors[iter_id], 1.0)
        lut.Build()
        '''
        self.scalar_lut = lut