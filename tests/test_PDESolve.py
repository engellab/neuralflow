#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test PDE solver
"""

import unittest
from neuralflow import PDE_Solve
import numpy as np
from testhelper_PDE_Solve import PerformTest
from pkg_resources import working_set

GPU_support = any([pkg.key.startswith('cupy') for pkg in working_set])


class TestPDESolve(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Parameters for testing EV problem.

        # NUmber of elements
        cls.Ne = 128
        # Number of points per element.
        cls.Np = 8
        # Number of EV/EVects to compare with analytical solution
        cls.Nv = 10
        # Name of the problem. see EigenValueProblemTests.py
        cls.EVTestNames = ['Sine', 'Sine2', 'Cosine', 'MixedCosineSine',
                           'SquarePeqMixed', 'Bessel']

        # Test derivative and antiderivative matrices
        # Define (f(x),f'(x), and intergral(f(x)))
        # antiderivative is zero at x=xbegin
        cls.test_functions = [lambda x, x0: (x**2, 2*x, (x**3-x0**3)/3),
                              lambda x, x0: (
                                  np.sin(x), np.cos(x), np.cos(x0) - np.cos(x)
        )
        ]

    def testInputs(self):
        """"A few test for input parameters
        """
        with self.assertRaises(ValueError):
            xbegin, xend = -1, -1
            _ = PDE_Solve.PDESolve(xbegin, xend)
        with self.assertRaises(ValueError):
            Np = 1
            _ = PDE_Solve.PDESolve(Np=Np)
        with self.assertRaises(ValueError):
            Ne = 0
            _ = PDE_Solve.PDESolve(Ne=Ne)
        with self.assertRaises(ValueError):
            BoundCond = {'leftB': 'Neumann', 'rightB': 'Neuman'}
            _ = PDE_Solve.PDESolve(BoundCond=BoundCond)

    def testEVsolver(self):
        """Solve EV problem and compare numerical solution with analytical for
        six problems with known analytical solution.
        """
        for name in self.EVTestNames:
            _, ErrVec, ErrVal = PerformTest(
                self.Ne, self.Nv, self.Np, name, False
            )
            self.assertTrue(
                all(ErrVec < 10**-6) and all(ErrVal < 10**-6),
                f'{name} error is lager than 10^-6'
            )

    @unittest.skipIf(not GPU_support, 'cupy not installed')
    def testEVsolverGPU(self):
        """Solve EV problem and compare numerical solution with analytical for
        six problems with known analytical solution.
        """
        for name in self.EVTestNames:
            _, ErrVec, ErrVal = PerformTest(
                self.Ne, self.Nv, self.Np, name, True
            )
            self.assertTrue(
                all(ErrVec < 10**-6) and all(ErrVal < 10**-6),
                f'{name} error is lager than 10^-6'
            )

    def testIntegrate(self):
        """Test antiderivative and integration functions"""

        # Initialize class instance
        xbegin, xend = -2, 2
        solver = PDE_Solve.PDESolve(xbegin, xend, self.Np, self.Ne)
        for f in self.test_functions:
            res = f(solver.grid.x_d, xbegin)
            # Compute derivative
            f_derivative = solver.grid.Differentiate(res[0])
            # Compute antiderivative
            f_integral = solver.grid.Integrate(res[0])
            self.assertTrue(np.allclose(f_derivative, res[1]))
            self.assertTrue(np.allclose(f_integral, res[2]))

    @unittest.skipIf(not GPU_support, 'cupy not installed')
    def testIntegrateGPU(self):
        """Test antiderivative and integration functions"""

        import cupy as cp
        # Initialize class instance
        xbegin, xend = -2, 2
        solver = PDE_Solve.PDESolve(
            xbegin, xend, self.Np, self.Ne, with_cuda=True
        )
        for f in self.test_functions:
            res = list(f(solver.grid.cuda_var.x_d, xbegin))
            # Compute derivative
            f_derivative = solver.grid.Differentiate(res[0], device='GPU')
            # Compute antiderivative
            f_integral = solver.grid.Integrate(res[0], device='GPU')
            self.assertTrue(cp.allclose(f_derivative, res[1]))
            self.assertTrue(cp.allclose(f_integral, res[2]))


if __name__ == '__main__':
    unittest.main()
