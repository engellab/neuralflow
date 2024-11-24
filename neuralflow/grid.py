#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base class for computing the GLL grid, the assosiated weights, differentiation
matrix and antiderivative matrix (asuming Lagrnage polynomials as basis funcs)
"""

import numpy as np
import numpy.matlib
from numpy.polynomial import legendre
from functools import reduce
from itertools import combinations
from operator import mul


class GLLgrid():
    """Grid class that calculates Gauss-Lobatto-Legendre grid and implements
    integration and differentiation on the grid.

    Parameters
    ----------
    xbegin : float
        The left boundary of the latent state. The default is -1.
    xend : float
        The right boundary of the latent state. The default is 1.
    Np : int
        The degree of Langrange interpolation polynomial, also the
        number of grid points at each element. The default is 8.
    Ne : int
        Number of SEM elements. The default is 64.
    with_cuda : bool, optional
        Whether to include GPU support. For GPU optimization, the platform
        has to be cuda-enabled, and cupy package has to be installed. The
        default is False.

    """

    def __init__(self, xbegin=-1, xend=1, Np=8, Ne=64, with_cuda=False):

        self.xbegin = xbegin
        self.xend = xend
        self.Np = Np
        self.Ne = Ne

        # Check inputs
        self._check_inputs()

        # cuda
        self.with_cuda = with_cuda
        if with_cuda:
            import neuralflow.base_cuda as cuda
            self.cuda = cuda
            self.cuda_var = cuda.var()

        # Set N - total number of points
        self.N = (Np-1) * Ne + 1

        # Compute grid, weights, and differention matrix
        self._get_grid()

        # Calculate antiderivative matrix
        self._set_AD_mat()

    def __repr__(self):
        return (
            f'Grid(xbegin={self.xbegin}, xend={self.xend}, Np={self.Np}, '
            f'Ne={self.Ne}, {"with cuda" if self.with_cuda else "cpu only"})'
            )

    def __deepcopy__(self, memo):
        """ Note: deepcopy won't work if grid initializaed with with_cuda=True
        since self.cuda references a module (which are singletons in python)
        Simply create a new instance with the same parameters for deepcopy
        """
        return GLLgrid(
            self.xbegin, self.xend, self.Np, self.Ne, self.with_cuda
        )

    def Differentiate(self, f, result=None, device='CPU'):
        """ Take a derivative of function f


        Parameters
        ----------
        f : numpy array, dtype=float
            Function values evaluated on the grid
        result : numpy array, dtype=float
            A container for the results (to avoid additional allocation).
            If not provided, will return a result. The default is None.
        device : str, optional
            ENUM('CPU', 'GPU'). The default is 'CPU'.

        Returns
        -------
        numpy array
            If the result is not provided at the input, it will be returned.

        """
        if device not in ['CPU', 'GPU']:
            raise ValueError(f'Unknown device {device}')
        elif device == 'GPU' and not self.with_cuda:
            raise ValueError('Initialize the class variable with with_cuda = '
                             'True to support GPU computations')

        diff_mat = self.dmat_d if device == 'CPU' else self.cuda_var.dmat_d

        if result is None:
            return diff_mat.dot(f)
        else:
            diff_mat.dot(f, out=result)

    def Integrate(self, f, result=None, device='CPU'):
        """Indefinite integral of a function f using integration matrix.


        Parameters
        ----------
        f : numpy array, dtype=float
            Function values evaluated on the grid
        result : numpy array, dtype=float
            A container for the results (to avoid additional allocation).
            If not provided, will return a result. The default is None.

        Returns
        -------
        numpy array
            If the result is not provided at the input, it will be returned.

        """

        if device not in ['CPU', 'GPU']:
            raise ValueError(f'Unknown device {device}')
        elif device == 'GPU' and not self.with_cuda:
            raise ValueError('Initialize the class variable with with_cuda = '
                             'True to support GPU computations')

        int_mat = self.AD_d if device == 'CPU' else self.cuda_var.AD_d

        if result is None:
            return int_mat.dot(f)
        else:
            int_mat.dot(f, out=result)

    def _get_grid(self):
        """Calculate grid nodes, corresponding weights and differentiation
        matrix (with SEM method)


        Sets
        ----
        x_d, w_d, dmat_d, dx, x_, w_, dmat_, ele_scale_
        """

        # Scaling factor:
        self.ele_scale_ = (self.xend - self.xbegin) / (2 * self.Ne)

        # Calculate local grid, weights, differentiation matrix
        self._get_single_element()

        # Now patch locals to get globals
        self.x_d = np.zeros(self.N)
        self.w_d = np.zeros(self.N)
        self.dmat_d = np.zeros((self.N, self.N), dtype='float64')

        for i in range(self.Ne):
            patch = np.arange(i * (self.Np - 1), i *
                              (self.Np - 1) + self.Np)

            # Patch as described in SEM documentation
            self.x_d[patch] = self.x_ + (2 * i + 1) * self.ele_scale_
            self.w_d[patch] += self.w_
            self.dmat_d[np.ix_(patch, patch)] += self.dmat_

        self.x_d += self.xbegin

        # Divide rows that correspond to primary nodes by 2
        # This is because at primay nodes the derivative is a half-sum of
        # derivative from the left and derivative from the right
        for i in range(self.Ne - 1):
            self.dmat_d[i * (self.Np - 1) + self.Np - 1, :] /= 2.0

        # Copy the results to GPU memory
        if self.with_cuda:
            self.cuda_var.x_ = self.cuda.cp.asarray(self.x_)
            self.cuda_var.w_ = self.cuda.cp.asarray(self.w_)
            self.cuda_var.dmat_ = self.cuda.cp.asarray(self.dmat_)
            self.cuda_var.x_d = self.cuda.cp.asarray(self.x_d)
            self.cuda_var.w_d = self.cuda.cp.asarray(self.w_d)
            self.cuda_var.dmat_d = self.cuda.cp.asarray(self.dmat_d)

    def _get_single_element(self):
        """Calculate local grid nodes, corresponding weights and
        differentiation matrix using numpy.polynomial.legendre module


        Sets
        ----
        x_, w_, dmat_

        """
        # Interested in Legendre polynomial #(Np-1):
        coefs = np.append(np.zeros(self.Np - 1), 1)

        # Calculate grid points:
        self.x_ = np.append(
            np.append(-1, legendre.Legendre(coefs).deriv().roots()), 1)

        # Need legendre polynomial at grid points:
        Ln = legendre.legval(self.x_, coefs)

        # Calculate weights:
        self.w_ = 2 / ((self.Np - 1) * self.Np * Ln**2)

        # Calculate differentiation matrix:
        self.dmat_ = np.zeros((len(Ln), len(Ln)))
        for i in range(self.Np):
            for j in range(self.Np):
                if i != j:
                    self.dmat_[i][j] = (
                        Ln[i] / (Ln[j] * (self.x_[i] - self.x_[j]))
                    )
                else:
                    self.dmat_[i][i] = 0
        self.dmat_[0, 0] = -(self.Np - 1) * (self.Np) / 4
        self.dmat_[-1, -1] = (self.Np - 1) * (self.Np) / 4

        # Scale locals:
        self.x_ *= self.ele_scale_
        self.w_ *= self.ele_scale_
        self.dmat_ /= self.ele_scale_

    def _set_AD_mat(self):
        """Calculates Integration Matrix that can be used to calculate
        antiderivative
        """

        # Define local grid at a single element xi \in [-1;1]
        x_local = self.x_ / self.ele_scale_

        # Allocate local and global antiderivative matrix
        self.AD_ = np.zeros((self.Np, self.Np))
        self.AD_d = np.zeros((self.N, self.N))

        # Construct local matrix first:
        # integration coefficients of x, x^2, x^3, ... of Lagrange
        # interpolation polynomials
        coefs = np.zeros(self.Np)
        coefs[-1] = 1 / self.Np
        # Matrix with columns x, x^2, ..., x^N
        x_mat = (
            np.transpose(np.matlib.repmat(x_local, self.Np, 1))
        )**np.arange(1, self.Np + 1)
        for i in range(self.Np):
            # take of all but current grid points:
            inds = np.append(np.arange(i), np.arange(i + 1, self.Np))
            x_crop = x_local[inds]

            # Calculate integration coefficients and common denominator using
            # sums of all single, pairwise, triplewise, etc. combinations
            Combinations = [
                sum(reduce(mul, c)
                    for c in combinations(x_crop, i + 1))
                for i in range(self.Np - 1)
            ]
            coefs[:-1] = (
                (-1)**np.arange(1 - self.Np % 2, self.Np - self.Np % 2)
            ) * Combinations[::-1] / np.arange(1, self.Np)
            denominator = np.prod(np.ones(self.Np - 1) * x_local[i] - x_crop)

            # Choose integration constant c0 such that F(-1)=0
            c0 = -np.sum((-1)**np.arange(1, self.Np + 1) * coefs)

            # Calculate differentiation matrix
            self.AD_[:, i] = (x_mat.dot(coefs) + c0) / denominator

        # Set first row to zero and scale
        self.AD_[0, :] = 0
        self.AD_ *= self.ele_scale_

        # Now calculate global AD matrix:
        for i in range(self.Ne):
            patch = np.arange(i * (self.Np - 1), i *
                              (self.Np - 1) + self.Np)
            self.AD_d[np.ix_(patch, patch)] += self.AD_
            self.AD_d[np.ix_(
                np.arange(i * (self.Np - 1) + self.Np, self.N), patch
            )] += self.AD_[-1, :]
        if self.with_cuda:
            self.cuda_var.AD_d = self.cuda.cp.asarray(self.AD_d)

    def _check_inputs(self):
        if self.xend <= self.xbegin:
            raise ValueError('x interval length is <= 0')
        if not isinstance(self.Np, (int, np.integer)) or self.Np < 3:
            raise ValueError('Np is not int or less than 3')
        if not isinstance(self.Ne, (int, np.integer)) or self.Ne < 1:
            raise ValueError('Ne is not int or less than 1')
