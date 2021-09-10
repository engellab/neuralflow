# -*- coding: utf-8 -*-
from scipy.interpolate import CubicSpline, PPoly
import numpy as np


def add_anchor_point(x, y):
    """Auxilliary function to create stepping potential used by energy_model.peq_models.jump_spline2 module
       Find additional anchoring point (x_add, y_add), such that x[0] < x_add < x[1], and y_add = y_spline (x_add_mirror),
       where y_spline is spline between x[1] and x[2], and x_add_mirror is a mirror point of x_add w.r.t. x[1], e.g. |x_add-x[1]|=|x[1]-x_add_mirror|.
       This additional point will force the barriers to have symmetrical shape.

       Params:
           x: x - values at three right or three left boundary points, e.g. interp_x[0:3], or interp_x[-3:][::-1] (reverse order on the right boundary)
           y: corresponding y values
       Returns:
           x_add, y_add - additional point in between x[0] < x_add < x[1] that can be used for spline interpolation
    """
    # Start with the middle point
    x_add_mirror = 0.5 * (x[1] + x[2])
    not_found = True
    cpoly = PPoly(CubicSpline(np.sort(x[1:3]), np.sort(
        y[1:3]), bc_type=[(1, 0), (1, 0)]).c, np.sort([x[1], x[2]]))
    first_peak_is_maximum = True if y[0] <= y[1] else False
    while not_found:
        # Check if y-value at anchor point exceeds value at the boundary and that x-value is within an interval
        if first_peak_is_maximum:
            if cpoly(x_add_mirror) > y[0] and np.abs(x_add_mirror - x[1]) < np.abs(x[1] - x[0]):
                x_add = 2 * x[1] - x_add_mirror
                if x[1] > x[0]:
                    poly2 = PPoly(CubicSpline([x[0], x_add, x[1]], [y[0], cpoly(
                        x_add_mirror), y[1]], bc_type=[(1, 0), (1, 0)]).c, [x[0], x_add, x[1]])
                else:
                    poly2 = PPoly(CubicSpline([x[1], x_add, x[0]], [y[1], cpoly(
                        x_add_mirror), y[0]], bc_type=[(1, 0), (1, 0)]).c, [x[1], x_add, x[0]])
                x_dense = np.linspace(x[0], x[1], 100)
                if all(poly2(x_dense) <= y[1]):
                    not_found = False
        else:
            if cpoly(x_add_mirror) < y[0] and np.abs(x_add_mirror - x[1]) < np.abs(x[1] - x[0]):
                x_add = 2 * x[1] - x_add_mirror
                if x[1] > x[0]:
                    poly2 = PPoly(CubicSpline([x[0], x_add, x[1]], [y[0], cpoly(
                        x_add_mirror), y[1]], bc_type=[(1, 0), (1, 0)]).c, [x[0], x_add, x[1]])
                else:
                    poly2 = PPoly(CubicSpline([x[1], x_add, x[0]], [y[1], cpoly(
                        x_add_mirror), y[0]], bc_type=[(1, 0), (1, 0)]).c, [x[1], x_add, x[0]])
                x_dense = np.linspace(x[0], x[1], 100)
                if all(poly2(x_dense) >= y[1]):
                    not_found = False
        x_add_mirror = 0.5 * (x[1] + x_add_mirror)
    return x_add, cpoly(2 * x[1] - x_add)
