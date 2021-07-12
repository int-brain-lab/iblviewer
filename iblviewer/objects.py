import numpy as np
from vtk.util.numpy_support import numpy_to_vtk
import vtk
import vedo

"""
The classes are modifications of existing vedo classes that were either doing 
things in unsatisfying manner or lacking features.
"""


def Cross3D(pos=(0,0,0), size=1.0, thickness=0.5, color='black', alpha=1, res=4, lines_mode=True):
    """
    Build a 3D cross shape, mainly useful as a 3D marker.
    :param pos: Position of the cross
    :param size: Size of the cross
    :param thickness: Thickness in pixels (remains constant on screen)
    :param color: Color of the cross
    :param alpha: Alpha/opacity of the cross
    :param res: Resolution of the cylinders if not used in line_mode
    :param lines_mode: Whether lines are used or cylinders. The difference is in how they look like
    when you zoom in on the cross. If you use lines, their thickness is constant on screen, whether
    close or far. If you use cylinders, they will get thicker the closer you approach the camera
    (in perspective mode of course)
    :return: vedo.Mesh
    """
    if lines_mode:
        x1 = np.array([1.0, 0.0, 0.0]) * size / 2
        x2 = np.array([-1.0, 0.0, 0.0]) * size / 2
        c1 = vedo.Line(x1, x2, lw=thickness)

        y1 = np.array([0.0, 1.0, 0.0]) * size / 2
        y2 = np.array([0.0, -1.0, 0.0]) * size / 2
        c2 = vedo.Line(y1, y2, lw=thickness)

        z1 = np.array([0.0, 0.0, 1.0]) * size / 2
        z2 = np.array([0.0, 0.0, -1.0]) * size / 2
        c3 = vedo.Line(z1, z2, lw=thickness)
    else:
        c1 = vedo.Cylinder(r=thickness, height=size, res=res)
        c2 = vedo.Cylinder(r=thickness, height=size, res=res).rotateX(90)
        c3 = vedo.Cylinder(r=thickness, height=size, res=res).rotateY(90)
    cross = vedo.merge(c1,c2,c3).color(color).alpha(alpha)
    cross.SetPosition(pos)
    cross.name = 'Marker'
    return cross


class Lines(vedo.Lines):
    """
    Improved Lines class that supports point sets of varying lengths
    """
    def __init__(self, points, end_points=None, c='gray', alpha=1, lw=1, dotted=False):
        """
        Constructor
        parameters are the same as vedo.Line
        """
        self.axes = [1, 1, 1]
        #if not isinstance(point_sets, np.ndarray):
            #point_set = np.array(point_sets, dtype=object)
        if len(points.shape) > 1 and points.shape[1] == 2:
            super().__init__(points, end_points, c=c, alpha=alpha, lw=lw, dotted=dotted)
        else:
            polylns = vtk.vtkAppendPolyData()
            for point_set in points:
                positions = point_set
                if not isinstance(point_set, np.ndarray):
                    point_set = np.array(point_set)
                # numpy_to_vtk is unhappy if dtype is not to its liking
                point_set = point_set.astype(float)
                positions = numpy_to_vtk(np.ascontiguousarray(point_set), deep=True)

                # This part taken from class Line, which accepts n points
                vtk_points = vtk.vtkPoints()
                vtk_points.SetData(positions)
                
                lines = vtk.vtkCellArray()
                num_pts = len(point_set)
                lines.InsertNextCell(num_pts)
                for i in range(num_pts):
                    lines.InsertCellPoint(i)

                poly = vtk.vtkPolyData()
                poly.SetPoints(vtk_points)
                poly.SetLines(lines)
                polylns.AddInputData(poly)
            polylns.Update()

            vedo.Mesh.__init__(self, polylns.GetOutput())
            self.lw(lw).lighting('off')
            if dotted:
                self.GetProperty().SetLineStipplePattern(0xF0F0)
                self.GetProperty().SetLineStippleRepeatFactor(1)

        self.name = 'Lines'


class Points(vedo.Points):
    """
    Improved Points class that supports time series and screen-space mode
    """

    def __init__(self, positions, radius=1, values=None, color_map='viridis', screen_space=False, 
    alpha=1, res=6, min_v=None, max_v=None, scalars_prefix=None):
        """
        Constructor
        :param positions: 3D positions
        :param radius: Radius of the points
        :param values: Custom scalar values. You may pass a list the same length 
        as the number of points. If values is a 2D array, then we assume these are time series.
        You will then need to call:
        actor.polydata().GetPointData().SetActiveScalars(name)
        actor.mapper().SelectColorArray(name)
        where name is the array name that starts with the given scalars_prefix.
        If you have three steps in your time series, you will have by default
        Scalars_0, Scalars_1, Scalars_2.
        :param color_map: Color map, either a list of values and corresponding colors
        or a color map name (see vedo documentation with color maps that follow matplotlib)
        :param screen_space: Whether the points are rendered as screen-space points or as
        spheres. The main difference is that screen-space is very fast and can display millions
        of points whereas sphere mode allows you to zoom in on a point and you will see it
        bigger up-close with a perspective camera.
        :param alpha: Alpha/opacity value
        :param res: Resolution of the point if screen-space is disabled
        :param min_v: Minimum value for the given values (will be computed if not given)
        :param max_v: Maximum value for the given values (will be computed if not given)
        :param scalars_prefix: Scalar array name prefix. The rest of the name is _id
        where id starts at 0.
        """
        self.scalars_prefix = 'Scalars_' if scalars_prefix is None else scalars_prefix
        self.axes = [1, 1, 1]

        # Multi component (ndimensional) arrays in vtk:
        # https://vtk.org/doc/nightly/html/classvtkAbstractArray.html#a528de7a4879a219e7f82a82130186dc8
        polydata = vtk.vtkPolyData()
        points = vtk.vtkPoints()
        num_points = len(positions)
        points.SetNumberOfPoints(num_points)
        #for p_id in range(num_points):
            #points.SetPoint(p_id, positions[p_id])
        positions = positions.astype(float)
        points_data = numpy_to_vtk(np.ascontiguousarray(positions), deep=True)
        points.SetData(points_data)
        polydata.SetPoints(points)

        # We have to set scalar values after the object is created because VTK automatically
        # creates some array values initially and we want to ignore them later on
        scalars = []
        if values is not None and len(values) > 0:
            if not isinstance(values, np.ndarray):
                values = np.array(values)
            if len(values.shape) > 1 and values.shape[1] > 1:
                # Safeguard
                if num_points == values.shape[0]:
                    values.reshape(num_points, -1)
                all_values = values.ravel()
                if min_v is None:
                    min_v = min(all_values)
                if max_v is None:
                    max_v = max(all_values)
                for loop_id in range(values.shape[1]):
                    scalars.append(self.add_scalar_data(polydata, values[:, loop_id]))
            else:
                if min_v is None:
                    min_v = min(values)
                if max_v is None:
                    max_v = max(values)
                scalars.append(self.add_scalar_data(polydata, values))

        # The following sets the "ActiveScalar", i.e. default scalar used by
        # VTK (and is must for setting radius of each sphere by corresponding
        # value in radii array). But what tells VTK to use it for scaling?
        ctf = None
        values_range = None
        if len(scalars) > 0:
            polydata.GetPointData().SetActiveScalars(scalars[0].GetName())
            if isinstance(color_map, vtk.vtkColorTransferFunction):
                ctf = color_map
            else:
                ctf = vtk.vtkColorTransferFunction()
                values_range = np.linspace(min_v, max_v, 20)
                for v_id in range(len(values_range)):
                    value = values_range[v_id]
                    ctf.AddRGBPoint(value, *vedo.colorMap(value, color_map, min_v, max_v))

        if screen_space:
            glyph = vtk.vtkVertexGlyphFilter()
            glyph.SetInputData(polydata)
        else:
            sphere = vtk.vtkSphereSource()
            sphere.SetRadius(radius)
            sphere.SetThetaResolution(res)
            sphere.SetPhiResolution(res)
            sphere.Update()

            glyph = vtk.vtkGlyph3D()
            glyph.SetSourceConnection(sphere.GetOutputPort())
            glyph.SetInputData(polydata)
            
            #glyph.SetVectorModeToUseVector()
            #glyph.OrientOn()

            glyph.ClampingOn()
            if len(scalars) > 0:
                # If a value == min_v, then the point has radius 0
                glyph.SetRange(min_v - 1, max_v)
            
            glyph.ScalingOn()
            glyph.SetScaleFactor(1)
            glyph.SetScaleModeToScaleByScalar()
            
            glyph.SetScaleMode(3)

            '''
            For further work:

            # Tell glyph which attribute arrays to use for what
            glyph.SetInputArrayToProcess(0, 0, 0, 0, 'Elevation')		# scalars
            #glyph.SetInputArrayToProcess(1,0,0,0,'RTDataGradient')		# vectors
            # glyph.SetInputArrayToProcess(2,0,0,0,'nothing')		# normals
            glyph.SetInputArrayToProcess(3, 0, 0, 0, 'RTData')		# colors

            # Calling update because I'm going to use the scalar range to set the color map range
            glyph.Update()

            coloring_by = 'RTData'
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(glyph.GetOutputPort())
            mapper.SetScalarModeToUsePointFieldData()
            mapper.SetColorModeToMapScalars()
            mapper.ScalarVisibilityOn()
            mapper.SetScalarRange(glyph.GetOutputDataObject(0).GetPointData().GetArray(coloring_by).GetRange())
            mapper.SelectColorArray(coloring_by)
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            '''

        glyph.Update()
        vedo.Points.__init__(self, glyph.GetOutput(), alpha=alpha)
        mapper = self._mapper

        if screen_space:
            #self.GetProperty().SetColor(*color)
            self.GetProperty().SetPointSize(radius)
            self.GetProperty().SetRenderPointsAsSpheres(True)
        else:
            mapper.SetScalarModeToUsePointFieldData()
            if len(scalars) > 0:
                mapper.SetScalarRange(min_v, max_v)
            mapper.SetColorModeToMapScalars()

        if len(scalars) > 0:
            mapper.SelectColorArray(scalars[0].GetName())
        if ctf is not None:
            mapper.SetLookupTable(ctf)
        mapper.Update()

        self._polydata = polydata
        self.glyph = glyph
        self.name = 'Points'

    def get_number_of_arrays(self, ignore=['GlyphScale', 'Normals']):
        """
        """
        num_scalar_arrays = 0
        point_data = self._polydata.GetPointData()
        for i in range(point_data.GetNumberOfArrays()):
            if point_data.GetArrayName(i) in ignore:
                pass
            num_scalar_arrays += 1
        return num_scalar_arrays

    def add_scalar_data(self, polydata, values, step_id=None):
        """
        Add scalar data to a VTK polydata
        :param polydata: vtkPolyData
        :param values: Numpy 1D array or list
        :param step_id: ID for naming the scalar with Scalars_#ID
        :return: vtkFloatArray
        """
        scalars = numpy_to_vtk(np.ascontiguousarray(values), deep=True)
        num_existing_ones = polydata.GetPointData().GetNumberOfArrays()
        if step_id is None:
            step_id = num_existing_ones
        scalars.SetName(self.scalars_prefix + str(step_id))
        polydata.GetPointData().AddArray(scalars)
        return scalars


class Spheres(vedo.Mesh):
    """
    Reimplementation of vedo.Spheres that was not handling things properly
    when it comes to set time series and visualise them with colors.

    This class isn't used at the moment. Points is preferred as it acts either as
    vedo.Spheres or as screen space Points depending on screen_space param.

    In general, vedo uses "c" and "r" short variable names which are a hindrance...
    utils.Spheres is deprecated and you should favor utils.Points instead.
    """
    def __init__(self, centers, r=1, c="r", alpha=1, res=8):
        """
        Constructor.
        Parameters are the same as vedo.Spheres
        """
        self.axes = [1, 1, 1]
        if isinstance(centers, vedo.Points):
            centers = centers.points()

        cisseq = False
        if vedo.utils.isSequence(c):
            cisseq = True

        if cisseq:
            if len(centers) > len(c):
                vedo.printc("\times Mismatch in Spheres() colors", len(centers), len(c), c='r')
                raise RuntimeError()
            if len(centers) != len(c):
                vedo.printc("\lightningWarning: mismatch in Spheres() colors", len(centers), len(c))

        risseq = False
        if vedo.utils.isSequence(r):
            risseq = True

        if risseq:
            if len(centers) > len(r):
                vedo.printc("times Mismatch in Spheres() radius", len(centers), len(r), c='r')
                raise RuntimeError()
            if len(centers) != len(r):
                vedo.printc("\lightning Warning: mismatch in Spheres() radius", len(centers), len(r))
        if cisseq and risseq:
            vedo.printc("\noentry Limitation: c and r cannot be both sequences.", c='r')
            raise RuntimeError()

        src = vtk.vtkSphereSource()
        if not risseq:
            src.SetRadius(r)
        if vedo.utils.isSequence(res):
            res_t, res_phi = res
        else:
            res_t, res_phi = 2*res, res

        src.SetThetaResolution(res_t)
        src.SetPhiResolution(res_phi)
        src.Update()

        psrc = vtk.vtkPointSource()
        psrc.SetNumberOfPoints(len(centers))
        psrc.Update()
        pd = psrc.GetOutput()
        vpts = pd.GetPoints()

        glyph = vtk.vtkGlyph3D()
        glyph.SetSourceConnection(src.GetOutputPort())

        if cisseq:
            glyph.SetColorModeToColorByScalar()
            ucols = vtk.vtkUnsignedCharArray()
            ucols.SetNumberOfComponents(3)
            ucols.SetName("colors")
            #for i, p in enumerate(centers):
            for cx, cy, cz in c:
                #cx, cy, cz = getColor(acol)
                ucols.InsertNextTuple3(cx * 255, cy * 255, cz * 255)
            pd.GetPointData().SetScalars(ucols)
            glyph.ScalingOff()
        elif risseq:
            glyph.SetScaleModeToScaleByScalar()
            urads = numpy_to_vtk(np.ascontiguousarray(2*r).astype(float), deep=True)
            urads.SetName("radii")
            pd.GetPointData().SetScalars(urads)

        vpts.SetData(numpy_to_vtk(np.ascontiguousarray(centers), deep=True))

        glyph.SetInputData(pd)
        glyph.Update()

        vedo.Mesh.__init__(self, glyph.GetOutput(), alpha=alpha)
        self.phong()

        self._polydata = pd

        if cisseq:
            self.mapper().ScalarVisibilityOn()
        else:
            self.mapper().ScalarVisibilityOff()
            self.GetProperty().SetColor(vedo.getColor(c))
        self.name = 'Spheres'
