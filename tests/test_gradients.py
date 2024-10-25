# -*- coding: utf-8 -*-

"""Unittests for the gradients. It tests the gradients
of loglikelihood w.r.t. each of the parameters (F,D,F0,Fr,C) and compares the
result with finite difference approximation.
"""

import unittest
from neuralflow.model import model
from neuralflow.data_generation import SyntheticData
from neuralflow.gradients import Grads
from neuralflow.spike_data import SpikeData
from neuralflow.grid import GLLgrid
from neuralflow import peq_models
from neuralflow import firing_rate_models
import numpy as np
from itertools import product
from pkg_resources import working_set

GPU_support = any([pkg.key.startswith('cupy') for pkg in working_set])


def ConvexSum(Left, Right, theta): return Left*theta + Right*(1-theta)


class TestGradients(unittest.TestCase):
    """Computes gradient numerically and compares to finite diffrence
    result. Takes two different models on the left end and the right ends (F1
    and F2), and defines a family of models
    F(theta) = theta * M1 + (1 - theta) * M2. Calculates the
    gradient at theta = 0.5 and compares to finite difference approximation.
    """
    @classmethod
    def setUpClass(cls):
        """Define the model and some unittest hyperparameters
        """

        cls.boundary_modes = ['absorbing', 'reflecting']
        cls.pde_solve_params = {'xbegin': -1, 'xend': 1, 'Np': 8, 'Ne': 32}

        # Initialize grid with and without GPU support (so that there is no
        # error on platforms without cuda-enabled GPU)
        cls.grid = GLLgrid(**cls.pde_solve_params)
        if GPU_support:
            cls.grid_with_cuda = GLLgrid(
                **cls.pde_solve_params, with_cuda=True
            )

        # Ground-truth model for data generation
        cls.em = model.new_model(
            peq_model={'model': 'linear_pot', 'params': {'slope': 5}},
            p0_model={'model': 'single_well', 'params': {'miu': 100}},
            D=0.1,
            fr_model=[
                {"model": "linear", "params": {"slope": 50, "bias": 60}},
                {"model": "linear", "params": {"slope": -50, "bias": 60}}
            ],
            grid=cls.grid,
        )
        cls.num_neurons = cls.em.fr.shape[2]

        # Generate data for absorbing and reflecting modes
        cls.data = []
        for bm in cls.boundary_modes:
            dg = SyntheticData(cls.em, bm)
            data, _, _ = dg.generate_data(0, 1, 10, 0)
            cls.data.append(data)

        # Theta is a 1D variable that runs from 0 to 1. Gradients will be
        # calculate at 0.5
        cls.theta = 0.5

        # Epsilon for finite diffrence and two values of theta where logliks
        # will be calculated
        eps = 1e-3
        cls.theta1, cls.theta2 = cls.theta + eps, cls.theta - eps

        # Maximum allowed relative difference for grad vs. finite difference
        # estimate
        cls.tol = 1e-04

    def FD_grad_testing(self, grad, em_c, em_1, em_2, boundary_mode,
                        with_cuda):
        """ Compute gradient along the line at em_c and approximate it with
        the difference in logliks.
        """

        # Each boundary mode has its own data
        data = SpikeData(
            self.data[self.boundary_modes.index(boundary_mode)],
            'ISIs',
            with_cuda=with_cuda
        )

        if with_cuda:
            data.to_GPU()
            data = data.cuda_var.data
        else:
            data = data.data

        # Calculate gradient at theta
        gr = grad.get_grad_data(data, em_c, mode='gradient')

        # Calculate logliks at theta1 and theta2
        ll1 = grad.get_grad_data(data, em_1, mode='loglik')
        ll2 = grad.get_grad_data(data, em_2, mode='loglik')

        # Finite difference estimate
        fd_approx = (ll2 - ll1) / (self.theta2 - self.theta1)
        return gr, fd_approx

    def ForceTesting(self, with_cuda, boundary_mode, non_equilibrium):

        device = 'GPU' if with_cuda else 'CPU'

        # Left end: peq and force
        peqLeft = peq_models.linear_pot(self.grid.x_d, self.grid.w_d, 5)
        FLeft = self.em.force_from_peq(peqLeft)

        # Right end
        peqRight = peq_models.double_well(
            self.grid.x_d, self.grid.w_d, 0, 0.7, 2.0, 0
        )
        FRight = self.em.force_from_peq(peqRight)

        # Calculate forces at theta, theta1, and theta2
        # In general, F(theta) = Fleft*theta+ FRight*(1-theta)
        F = ConvexSum(FLeft, FRight, self.theta)
        F1 = ConvexSum(FLeft, FRight, self.theta1)
        F2 = ConvexSum(FLeft, FRight, self.theta2)

        # Calculate the corresponding peqs
        peq = self.em.peq_from_force(F)
        peq1 = self.em.peq_from_force(F1)
        peq2 = self.em.peq_from_force(F2)

        if with_cuda:
            grid = self.grid_with_cuda
        else:
            grid = self.grid

        if non_equilibrium:
            p0 = self.em.p0
        else:
            p0 = None

        em_c = model.new_model(
            peq, p0, self.em.D, self.em.fr, grid, with_cuda=with_cuda
        )
        em_1 = model.new_model(
            peq1, p0, self.em.D, self.em.fr, grid, with_cuda=with_cuda
        )
        em_2 = model.new_model(
            peq2, p0, self.em.D, self.em.fr, grid, with_cuda=with_cuda
        )

        grad = Grads(
            self.pde_solve_params, boundary_mode, ['F'], self.num_neurons,
            device=device
        )

        # Compute gradient and finite difference approximation
        gr, fd_approx = self.FD_grad_testing(
            grad, em_c, em_1, em_2, boundary_mode, with_cuda
        )

        # Gradient along theta direction by chain rule
        if with_cuda:
            gr['F'] = grad.cuda.cp.asnumpy(gr['F'])
        dLdth_grad = np.sum(gr['F'] * (FLeft-FRight) * self.grid.w_d)
        # Relative difference
        reldiff = np.abs((dLdth_grad - fd_approx)/(fd_approx))
        return reldiff

    def DTesting(self, with_cuda, boundary_mode):

        device = 'GPU' if with_cuda else 'CPU'

        # Left/right end D values
        DLeft = 5
        DRight = 15

        # Calculate D at theta, theta1, theta2
        D = ConvexSum(DLeft, DRight, self.theta)
        D1 = ConvexSum(DLeft, DRight, self.theta1)
        D2 = ConvexSum(DLeft, DRight, self.theta2)

        if with_cuda:
            grid = self.grid_with_cuda
        else:
            grid = self.grid

        em_c = model.new_model(
            self.em.peq, self.em.p0, D, self.em.fr, grid, with_cuda=with_cuda
        )
        em_1 = model.new_model(
            self.em.peq, self.em.p0, D1, self.em.fr, grid, with_cuda=with_cuda
        )
        em_2 = model.new_model(
            self.em.peq, self.em.p0, D2, self.em.fr, grid, with_cuda=with_cuda
        )

        grad = Grads(
            self.pde_solve_params, boundary_mode, ['D'], self.num_neurons,
            device=device
        )

        # Compute gradient and finite difference approximation
        gr, fd_approx = self.FD_grad_testing(
            grad, em_c, em_1, em_2, boundary_mode, with_cuda
        )

        # dLikelihood/dtheta with gradient by chain rule:
        dLdth_grad = gr['D'] * (DLeft-DRight)

        # Relative difference
        reldiff = np.abs((dLdth_grad - fd_approx)/(fd_approx))
        return reldiff

    def F0Testing(self, with_cuda, boundary_mode, non_equilibrium):

        device = 'GPU' if with_cuda else 'CPU'

        # Left end: p0 and F0
        p0Left = peq_models.linear_pot(self.grid.x_d, self.grid.w_d, 5)
        F0Left = self.em.force_from_peq(p0Left)

        # Right end: p0
        p0Right = peq_models.double_well(
            self.grid.x_d, self.grid.w_d, 0, 0.7, 2.0, 0
        )
        F0Right = self.em.force_from_peq(p0Right)

        # Calculate F0 at theta, theta1, and theta2
        # In general, F0(theta) = F0left*theta+ F0Right*(1-theta)
        F0 = ConvexSum(F0Left, F0Right, self.theta)
        F01 = ConvexSum(F0Left, F0Right, self.theta1)
        F02 = ConvexSum(F0Left, F0Right, self.theta2)

        p0 = self.em.peq_from_force(F0)
        p0_1 = self.em.peq_from_force(F01)
        p0_2 = self.em.peq_from_force(F02)

        if with_cuda:
            grid = self.grid_with_cuda
        else:
            grid = self.grid

        em_c = model.new_model(
            self.em.peq, p0, self.em.D, self.em.fr, grid, with_cuda=with_cuda
        )
        em_1 = model.new_model(
            self.em.peq, p0_1, self.em.D, self.em.fr, grid,
            with_cuda=with_cuda
        )
        em_2 = model.new_model(
            self.em.peq, p0_2, self.em.D, self.em.fr, grid,
            with_cuda=with_cuda
        )

        grad = Grads(
            self.pde_solve_params, boundary_mode, ['F0'], self.num_neurons,
            device=device
        )

        # Compute gradient and finite difference approximation
        gr, fd_approx = self.FD_grad_testing(
            grad, em_c, em_1, em_2, boundary_mode, with_cuda
        )
        if device == 'GPU':
            gr['F0'] = grad.cuda.cp.asnumpy(gr['F0'])

        # dLikelihood/dtheta by chain rule:
        dLdth_grad = np.sum(gr['F0'] * (F0Left-F0Right) * self.grid.w_d)

        # Relative difference
        reldiff = np.abs((dLdth_grad - fd_approx)/(fd_approx))

        return reldiff

    def FrTesting(self, with_cuda, boundary_mode):

        device = 'GPU' if with_cuda else 'CPU'
        # Left end: fr and Fr
        frLeft = np.stack([
            firing_rate_models.linear(self.grid.x_d, 50, 60),
            firing_rate_models.sinus(self.grid.x_d, 50, 30)
        ]).T
        # Ensure constant C is 1 by dividing by the fr(xbegin)
        frLeft /= frLeft[0, :]
        FrLeft = self.em.Fr_from_fr(frLeft)

        # Right end: fr and Fr
        frRight = np.stack([
            firing_rate_models.linear(self.grid.x_d, -50, 60),
            firing_rate_models.sinus(self.grid.x_d, 40, 20)
        ]).T
        frRight /= frRight[0, :]
        FrRight = self.em.Fr_from_fr(frRight)

        # Calculate Fr at theta, theta1, and theta2
        # In general, Fr(theta) = Frleft*theta+ FrRight*(1-theta)
        Fr = ConvexSum(FrLeft, FrRight, self.theta)
        Fr1 = ConvexSum(FrLeft, FrRight, self.theta1)
        Fr2 = ConvexSum(FrLeft, FrRight, self.theta2)

        # Calculate the corresponding frs
        fr = self.em.fr_from_Fr(Fr)
        fr_1 = self.em.fr_from_Fr(Fr1)
        fr_2 = self.em.fr_from_Fr(Fr2)

        if with_cuda:
            grid = self.grid_with_cuda
        else:
            grid = self.grid

        em_c = model.new_model(
            self.em.peq, self.em.p0, self.em.D, fr, grid, with_cuda=with_cuda
        )
        em_1 = model.new_model(
            self.em.peq, self.em.p0, self.em.D, fr_1, grid,
            with_cuda=with_cuda
        )
        em_2 = model.new_model(
            self.em.peq, self.em.p0, self.em.D, fr_2, grid,
            with_cuda=with_cuda
        )

        grad = Grads(
            self.pde_solve_params, boundary_mode, ['Fr'], self.num_neurons,
            device=device
        )

        # Compute gradient and finite difference approximation
        gr, fd_approx = self.FD_grad_testing(
            grad, em_c, em_1, em_2, boundary_mode, with_cuda
        )
        if device == 'GPU':
            gr['Fr'] = grad.cuda.cp.asnumpy(gr['Fr'])

        # dLikelihood/dtheta by chain rule:
        dLdth_grad = np.sum(
            gr['Fr']*(FrLeft-FrRight)*self.grid.w_d[:, np.newaxis]
        )

        # Relative difference
        reldiff = np.abs((dLdth_grad - fd_approx)/(fd_approx))

        return reldiff

    # @profile
    def CTesting(self, with_cuda, boundary_mode):

        device = 'GPU' if with_cuda else 'CPU'

        # Left/right C values
        CLeft = np.array([1, 2])
        CRight = np.array([5, 3])

        # Calculate C at theta, theta1, theta2
        C = ConvexSum(CLeft, CRight, self.theta)
        C1 = ConvexSum(CLeft, CRight, self.theta1)
        C2 = ConvexSum(CLeft, CRight, self.theta2)

        # Calculate the corresponding firing rates:
        fr = self.em.fr[0]/self.em.fr[0][0, :]*C
        fr_1 = self.em.fr[0]/self.em.fr[0][0, :]*C1
        fr_2 = self.em.fr[0]/self.em.fr[0][0, :]*C2

        if with_cuda:
            grid = self.grid_with_cuda
        else:
            grid = self.grid

        em_c = model.new_model(
            self.em.peq, self.em.p0, self.em.D, fr, grid, with_cuda=with_cuda
        )
        em_1 = model.new_model(
            self.em.peq, self.em.p0, self.em.D, fr_1, grid,
            with_cuda=with_cuda
        )
        em_2 = model.new_model(
            self.em.peq, self.em.p0, self.em.D, fr_2, grid,
            with_cuda=with_cuda
        )

        grad = Grads(
            self.pde_solve_params, boundary_mode, ['C'], self.num_neurons,
            device=device
        )

        # Compute gradient and finite difference approximation
        gr, fd_approx = self.FD_grad_testing(
            grad, em_c, em_1, em_2, boundary_mode, with_cuda
        )
        if device == 'GPU':
            gr['C'] = grad.cuda.cp.asnumpy(gr['C'])

        # dLikelihood/dtheta with gradient by chain rule:
        dLdth_grad = np.sum(gr['C']*(CLeft-CRight))

        # Relative difference
        reldiff = np.abs((dLdth_grad - fd_approx)/(fd_approx))

        return reldiff

    def testFGrad_CPU(self):
        for bm, eq_mode in product(self.boundary_modes, [True]):  # , False]):
            test_name = f'Test grad F, CPU, {bm}, non-equilibrium = {eq_mode}'
            with self.subTest(test_name, bm=bm, eq_mode=eq_mode):
                reldiff = self.ForceTesting(False, bm, eq_mode)
                print(f'CPU, {bm}, non-equilibrium = {eq_mode}, F-grad '
                      f' reldiff: {reldiff :.12f}')
                self.assertTrue(reldiff < self.tol, 'Grad F test failed')

    def testDGrad_CPU(self):
        for bm in self.boundary_modes:
            test_name = f'Test grad D, CPU, {bm}'
            with self.subTest(test_name, bm=bm):
                reldiff = self.DTesting(False, bm)
                print(f'CPU, {bm}, D-grad reldiff: {reldiff :.12f}')
                self.assertTrue(reldiff < self.tol, 'Grad D test failed')

    def testF0Grad_CPU(self):
        for bm, eq_mode in product(self.boundary_modes, [True, False]):
            test_name = f'Test grad F0, CPU, {bm}, non-equilibrium = {eq_mode}'
            with self.subTest(test_name, bm=bm, eq_mode=eq_mode):
                reldiff = self.F0Testing(False, bm, eq_mode)
                print(f'CPU, {bm}, non-equilibrium = {eq_mode}, F0-grad '
                      f' reldiff: {reldiff :.12f}')
                self.assertTrue(reldiff < self.tol, 'Grad F0 test failed')

    def testFrGrad_CPU(self):
        for bm in self.boundary_modes:
            test_name = f'Test grad Fr, CPU, {bm}'
            with self.subTest(test_name, bm=bm):
                reldiff = self.FrTesting(False, bm)
                print(f'CPU, {bm}, Fr-grad reldiff: {reldiff :.12f}')
                self.assertTrue(reldiff < self.tol, 'Grad Fr test failed')

    def testCGrad_CPU(self):
        for bm in self.boundary_modes:
            test_name = f'Test grad C, CPU, {bm}'
            with self.subTest(test_name, bm=bm):
                reldiff = self.CTesting(False, bm)
                print(f'CPU, {bm}, C-grad reldiff: {reldiff :.12f}')
                self.assertTrue(reldiff < self.tol, 'Grad C test failed')

    @unittest.skipIf(not GPU_support, 'cupy not installed')
    def testFGrad_GPU(self):
        for bm, eq_mode in product(self.boundary_modes, [True, False]):
            test_name = f'Test grad F, GPU, {bm}, non-equilibrium = {eq_mode}'
            with self.subTest(test_name, bm=bm, eq_mode=eq_mode):
                reldiff = self.ForceTesting(True, bm, eq_mode)
                print(f'GPU, {bm}, non-equilibrium = {eq_mode}, F-grad '
                      f' reldiff: {reldiff :.12f}')
                self.assertTrue(reldiff < self.tol, 'Grad F test failed')

    @unittest.skipIf(not GPU_support, 'cupy not installed')
    def testDGrad_GPU(self):
        for bm in self.boundary_modes:
            test_name = f'Test grad D, GPU, {bm}'
            with self.subTest(test_name, bm=bm):
                reldiff = self.DTesting(True, bm)
                print(f'GPU, {bm}, D-grad reldiff: {reldiff :.12f}')
                self.assertTrue(reldiff < self.tol, 'Grad D test failed')

    @unittest.skipIf(not GPU_support, 'cupy not installed')
    def testF0Grad_GPU(self):
        for bm, eq_mode in product(self.boundary_modes, [True, False]):
            test_name = f'Test grad F0, GPU, {bm}, non-equilibrium = {eq_mode}'
            with self.subTest(test_name, bm=bm, eq_mode=eq_mode):
                reldiff = self.F0Testing(True, bm, eq_mode)
                print(f'GPU, {bm}, non-equilibrium = {eq_mode}, F0-grad '
                      f' reldiff: {reldiff :.12f}')
                self.assertTrue(reldiff < self.tol, 'Grad F0 test failed')

    @unittest.skipIf(not GPU_support, 'cupy not installed')
    def testFrGrad_GPU(self):
        for bm in self.boundary_modes:
            test_name = f'Test grad Fr, GPU, {bm}'
            with self.subTest(test_name, bm=bm):
                reldiff = self.FrTesting(True, bm)
                print(f'GPU, {bm}, Fr-grad reldiff: {reldiff :.12f}')
                self.assertTrue(reldiff < self.tol, 'Grad Fr test failed')

    @unittest.skipIf(not GPU_support, 'cupy not installed')
    def testCGrad_GPU(self):
        for bm in self.boundary_modes:
            test_name = f'Test grad C, GPU, {bm}'
            with self.subTest(test_name, bm=bm):
                reldiff = self.CTesting(True, bm)
                print(f'GPU, {bm}, C-grad reldiff: {reldiff :.12f}')
                self.assertTrue(reldiff < self.tol, 'Grad C test failed')


if __name__ == '__main__':
    unittest.main()
