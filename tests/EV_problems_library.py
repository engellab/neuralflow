# -*- coding: utf-8 -*-
"""
This file contains tests for the Sturm–Liouville eigenvalue problems with known
analytical solution. All the eigenvectors normilizaed by there max(abs()) value

Input:
    Nv : int
        Number of EigenValues and EigenVectors to output.
    xgrid : np.array or None
        Array of x-values where the exact solution will be evaluated.
    mode : ENUM('domain_and_bc', 'exact_solution').
        Either return xbegin, xend, and boundary conditions, or the exact
        solution.
Output:
    if mode == 'domain_and_bc':
        BC : a dictionary that specify boundary conditions
        xbeing : float
            x-position of the domain's left end
        xend : float
            x-position of the domain's right end
    else:
        p(x) - function p calculated at xgrid
        q(x) - function q calculated at xgrid
        w(x) - function w calculated at xgrid
        BC - boundary conditions
        Evals - array of size Nv with analytical solutions for the eigenvalues
        Evects - array (len(xgrid),Nv) with analytical solutions for the
            eigenvectors
"""
import numpy as np
import numpy.matlib
import scipy.special as specfuns

# Sine test: -u''(x)=lambda*u(x); u(0)=u(1)=0


def SineTest(Nv, xgrid, mode):
    xbegin, xend = 0, 1
    if mode == 'domain_and_bc':
        BC = {'leftB': 'Dirichlet', 'rightB': 'Dirichlet'}
        return BC, xbegin, xend
    else:
        assert (
            np.abs(xgrid[0]-xbegin) < 10**-
            5 and np.abs(xgrid[-1]-xend) < 10**-5
        ), 'Incorrect x-interval'
        EVals = [((n+1)*np.pi)**2 for n in range(0, Nv)]
        EVects = np.sin(np.outer(xgrid, np.sqrt(EVals)))
        EVects /= np.max(np.abs(EVects), 0)
        p = -np.ones(len(xgrid))
        q = None
        w = np.ones(len(xgrid))
        return p, q, w, EVals, EVects

# Cosine test: -u''(x)=lambda*u(x); u'(0)=u'(1)=0


def CosineTest(Nv, xgrid, mode):
    xbegin, xend = 0, 1
    if mode == 'domain_and_bc':
        BC = {'leftB': 'Neumann', 'rightB': 'Neumann'}
        return BC, xbegin, xend
    else:
        assert (
            np.abs(xgrid[0]-xbegin) < 10**-
            5 and np.abs(xgrid[-1]-xend) < 10**-5
        ), 'Incorrect x-interval'
        EVals = [(n*np.pi)**2 for n in range(0, Nv)]
        EVects = np.cos(np.outer(xgrid, np.sqrt(EVals)))
        EVects /= np.max(np.abs(EVects), 0)
        p = -np.ones(len(xgrid))
        q = None
        w = np.ones(len(xgrid))
        return p, q, w, EVals, EVects

# Sine Test2: -u''(x)=lambda*u(x), u(0)=0, u'(1)=0


def Sine2Test(Nv, xgrid, mode):
    xbegin, xend = 0, 1
    if mode == 'domain_and_bc':
        BC = {'leftB': 'Dirichlet', 'rightB': 'Neumann'}
        return BC, xbegin, xend
    else:
        assert (
            np.abs(xgrid[0]-xbegin) < 10**-
            5 and np.abs(xgrid[-1]-xend) < 10**-5
        ), 'Incorrect x-interval'
        EVals = [((n+1/2)*np.pi)**2 for n in range(0, Nv)]
        EVects = np.sin(np.outer(xgrid, np.sqrt(EVals)))
        EVects /= np.max(np.abs(EVects), 0)
        p = -np.ones(len(xgrid))
        q = None
        w = np.ones(len(xgrid))
        return p, q, w, EVals, EVects

# Mixed test: -u''(x)=lambda*u(x); u(0)+u'(0)=u(1)+u'(1)=0


def MixedCosineSineTest(Nv, xgrid, mode):
    xbegin, xend = 0, 1
    if mode == 'domain_and_bc':
        BC = {
            'leftB': 'Robin', 'leftBCoeff': {'c1': 1, 'c2': 1},
            'rightB': 'Robin', 'rightBCoeff': {'c1': 1, 'c2': 1}
        }
        return BC, xbegin, xend
    else:
        assert (
            np.abs(xgrid[0]-xbegin) < 10**-
            5 and np.abs(xgrid[-1]-xend) < 10**-5
        ), 'Incorrect x-interval'
        EVals = [(n*np.pi)**2 for n in range(1, Nv)]
        EVects = (
            np.sin(np.outer(xgrid, np.sqrt(EVals))) -
            np.cos(np.outer(xgrid, np.sqrt(EVals)))*np.sqrt(EVals)
        )
        EVals = np.append(-1, EVals)
        EVects = np.append(
            np.reshape(np.exp(-xgrid), (len(xgrid), 1)), EVects, axis=1
        )
        EVects /= np.max(np.abs(EVects), 0)
        p = -np.ones(len(xgrid))
        q = None
        w = np.ones(len(xgrid))
        return p, q, w, EVals, EVects

# -x^2y''(x)-2xy(x)=lambda*y, y(1)+2y'(1)=0, y(2)+4y'(2)=0


def SquarePeqMixedTest(Nv, xgrid, mode):
    xbegin, xend = 1, 2
    if mode == 'domain_and_bc':
        BC = {
            'leftB': 'Robin', 'leftBCoeff': {'c1': 1, 'c2': 2},
            'rightB': 'Robin', 'rightBCoeff': {'c1': 1, 'c2': 4}
        }
        return BC, xbegin, xend
    else:
        assert (
            np.abs(xgrid[0]-xbegin) < 10**-
            5 and np.abs(xgrid[-1]-xend) < 10**-5
        ), 'Incorrect x-interval'
        EVals = [1/4 + (n*np.pi/np.log(2))**2 for n in range(0, Nv)]
        EVects = (
            np.cos(np.outer(
                np.log(xgrid), [np.pi*n/np.log(2) for n in range(0, Nv)]
            )) /
            np.matlib.repmat(np.sqrt(xgrid), Nv, 1).transpose()
        )
        EVects /= np.max(np.abs(EVects), 0)
        p = -xgrid**2
        q = None
        w = np.ones(len(xgrid))
        return p, q, w, EVals, EVects

# -(xy')'+y/x=lambda*x*y, y(0)=0, y'(1)=0


def BesselTest(Nv, xgrid, mode):
    xbegin, xend = 0, 1
    if mode == 'domain_and_bc':
        BC = {'leftB': 'Dirichlet', 'rightB': 'Neumann'}
        return BC, xbegin, xend
    else:
        assert (
            np.abs(xgrid[0]-xbegin) < 10**-
            5 and np.abs(xgrid[-1]-xend) < 10**-5
        ), 'Incorrect x-interval'
        EVals = specfuns.jnp_zeros(1, Nv)**2
        EVects = specfuns.j1(np.outer(xgrid, np.sqrt(EVals)))
        EVects /= np.max(np.abs(EVects), 0)
        p = -xgrid
        # Replace Inf with 10**10 as Inf is not supported by the solver
        q = np.append(10**10, 1/xgrid[1:])
        w = xgrid
        return p, q, w, EVals, EVects
