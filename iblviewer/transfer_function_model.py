
from dataclasses import dataclass, field
from typing import Mapping, List, Any
import numpy as np
import pandas as pd
import logging

import vtk


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

    scalar_map: np.ndarray = None
    scalar_min: float = 0.0
    scalar_max: float = 1.0
    
    color_map: np.ndarray = None
    alpha_map: np.ndarray = None

    sec_alpha_map: np.ndarray = None
    lut: vtk.vtkLookupTable = None
    scalar_lut: vtk.vtkLookupTable = None

    def set_data(self, scalar_map, color_map, alpha_map=None):
        """
        Get a look-up table from the current color and opacity/alpha map
        :parma scalar_map: Dictionary with key-value pairs in the form volume_scalar_value -> your custom value
        :param color_map: Color map, a nested list, each row containing [volume_scalar_value, [r, g, b]]
        :param alpha_map: Alpha map, optional, a 2D list with each row as [volume_scalar_value, alpha]
        """
        self.scalar_map = scalar_map

        values = scalar_map.values()
        self.scalar_min = min(values)
        self.scalar_max = max(values)

        self.color_map = np.array(color_map, dtype=object)
        if alpha_map is not None:
            self.alpha_map = np.array(alpha_map)
        self.build_luts()

    def build_luts(self):
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
        
        # Second LUT: "Reversed" reference yields the real scalar LUT with respect to colors
        num_values = len(self.scalar_map)
        scalar_range = np.linspace(self.scalar_min, self.scalar_max, num=100)

        lut = vtk.vtkLookupTable()
        lut.SetRange(self.scalar_min, self.scalar_max)
        lut.SetNumberOfTableValues(num_values)

        sorted_references = sorted(self.scalar_map, key=self.scalar_map.get)
        iter_id = 0
        for table_value in sorted_references:
            value, rgb = self.color_map[table_value]
            lut.SetTableValue(iter_id, *rgb, 1.0)
            iter_id += 1
        lut.Build()
        self.scalar_lut = lut