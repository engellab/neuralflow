    # -*- coding: utf-8 -*-

# This is a auxilliary class of neuralflow package

"""This is a part of neuralflow package/EnergyModel class.

Solves Stourm-Liouville problem:
    (D*p(x)y')'+q(x)y=lambda w(x) y
    with specified boundary conditions. 
Also performs additional EVP solving to find the eigenvalues and eigenvectors of H operator.
"""

import numpy as np, numpy.matlib
import numbers
from scipy import sparse, linalg
from numpy.polynomial import legendre
from .rank_nullspace import nullspace

from functools import reduce
from itertools import combinations
from operator import mul


MACHINE_EPSILON = np.finfo(np.double).eps


class PDESolve:
    """Numerical solution of Stourm-Liouville problem  

    Parameters
    ----------
    xbegin : float
        The left boundary of the latent state. The default is -1.
    xend : float
        The right boundary of the latent state. The default is 1.
    method : dictionary
        A dictionary that contains 2 keys: 
            name : string
                Specifies the method for the numerical solution of EV problem, can be either 'FD' or 'SEM' (forward differences or spectral elements method). 
            gridsize : dictionary 
                Specifies number of grid size points N for 'FD' method, or Np and Ne for 'SEM' method (Ne is the number of elements, Np is the number of grid points in each element).
        The default is {'name': 'SEM', 'gridsize': {'Np': 8, 'Ne': 256}}.
    BoundCond : dictionary
        A dictionary that specifies boundary conditions (Dirichlet, Neumann or Robin).
        The default is {'leftB': 'Neumann', 'rightB': 'Neumann'}
    grid_mode_calc : str
        Specify how to calculate SEM grid collocation points.
        Availiable options:
        'built_in': using numpy.polynomial module
        'newton': using Newton's method to calculate zeros of Legendre polinomial for the GLL grid
        The default is 'newton'. 
    BC_method : str
        Specify the method of boundary condition handling when transforming the EV problem into linear system of equations. 
        Availiable options:
        'projection': use projection operator.
        'bound_subst': use boundary condition substitution into the first and the last equations of the associated linear system.
        The default is 'projection'
    int_mode : str
        Specify the integration mode.
        Availiable options:
            'full' - use full integration matrix.
            'sparse' - use sparse integration matrix with bias.
        The default is 'full'. See Supplementary Materials 2.3 from M. Genkin, T. A. Engel, Nat Mach Intell 2, 674–683 (2020) for details.

    Attributes
    ----------
    AD_d : numpy array (N,N), dtype=float
        Integration matrix (only for SEM method).    
    dmat_d : numpy array (N,N), dtype=float
        Differentiation matrix (only for SEM method).
    dx : float
        Uniform grid step size (only for FD method).
    N : int
        A total number of the grid points.
    Np : int 
        Degree of each element (number of grid points in each element, only for SEM method).
    Ne : int
        A number of SEM elements (only for SEM method).
    w_d : numpy array (N,), dtype=float
        Weights of the nodes (on the global grid). 
    x_d: numpy array (N,), dtype=float
        Domain grid points.

    Hidden attributes
    -----------------
    AD_ : numpy array (Np,Np), dtype=float
        Integration matrix on a single element in local frame (only for SEM method).
    dmat_ : numpy array (Np,Np), dtype=float
        Differentiation matrix on a single element in local frame (only for SEM method).
    BC_ : numpy array (1,4), dtype=float
        Representation of boundary condition with four values consistent with to:
            BC_[0]*y[xbegin]+BC_[1]*y'[xbegin]=0
            BC_[2]*y[xend]+BC_[3]*y'[xend]=0
    ele_scale_ : float
        Scaling coefficient for each element (only for SEM method)
    massmat_full_ : numpy array (N,N), dtype=float
        Full mass matrix
    massmat_red_ : numpy array, dtype=float
        Reduced mass matrix of the same size as stiffmat_red_
    stiffmat_full_ : numpy array (N,N), dtype=float
        Full stiffness matrix    
    stiffmat_red_ : numpy array, dtype=float
        preallocated stiffness matrix with possibly reduced size,
        due to throughing away of some of the equations, or projection onto Nullspace of Boundary operator
    x_ : numpy array (Np,), dtype=float
        Grid on a single element in local frame (only for SEM method)
    w_ : numpy array (Np,), dtype=float   
        Weights on a single element in local frame (only for SEM method)        
    Null_M_ :  numpy array
        Nullspace of boundary operator (only for SEM method and 'projection' BC_method)

    Methods:
    --------
    solve_EV : solves the eigenvalue problem for specified
            functions peq(x), q(x), w(x), D in a chosen mode


    Hidden functions and methods
    ----------------------------
    _check_and_set_params : checks input parameters and sets grid dimensions
        called upon initialization
    _get_grid : calculates grid.
        called upon initialization    
    _get_single_element : calculates local grid with numpy.polinomial functions
        called by _get_grid
    __get_single_element_numerics : calculates local grid with Newton method
        called by _get_grid
    _get_matrices : preallocates full and reduced stiffness and mass matrices 
    _set_AD_mat : calculate antiderivative matrix
    _setmat : calculates stiffnes and mass matrices
        called by solve_EV

    """

    # List of availible methods
    _MethodList = ['FD', 'SEM']
    _grid_mode_calcList = ['built_in', 'newton']
    _BC_methodList = ['projection', 'bound_subst']

    def __init__(self, xbegin=-1.0, xend=1.0, method={'name': 'SEM', 'gridsize': {'Np': 8, 'Ne': 256}},
                 BoundCond={'leftB': 'Neumann', 'rightB': 'Neumann'}, grid_mode_calc='newton',
                 BC_method='projection', int_mode='full'):

        self.xbegin = xbegin
        self.xend = xend
        self.method = method
        self.BoundCond = BoundCond
        self.grid_mode_calc = grid_mode_calc
        self.BC_method = BC_method
        self.int_mode = int_mode

        # Assert inputs and set grid parameters: N, (Np, Ne)
        self._check_and_set_params()

        # Convert given boundary condition into a vector BC_ of size (1,4)
        self._get_BC()

        # Calculate grid, weights and (differentiation matrix)
        self._get_grid()

        # PreAllocate stiffness and mass matrices
        self._get_matrices()

        # Get the Nullspace
        self._get_Nullspace()

        # Calculate antiderivative matrix
        self._set_AD_mat()

    def Integrate(self, f, result=None):
        """Takes an indefinite integral of a function f using integration matrix (and a cumulative correction for 'sparse' int_mode).     


        Parameters
        ----------
        f : numpy array, dtype=float
            Function values evaluated on the grid
        result : numpy array, dtype=float
            A container for the results (to avoid additional allocation). If not provided, will return a result. The default is None.

        Returns
        -------
        numpy array
            If the result is not provided at the input, it will be returned.

        """
        if result is None:
            if self.int_mode == 'full':
                return self.AD_d.dot(f)
            elif self.int_mode == 'sparse':
                return self.AD_d.dot(f) + np.append([0], np.repeat(np.cumsum(self.AD_d.dot(f)[0:-1:self.Np - 1]), self.Np - 1))
        else:
            if self.int_mode == 'full':
                self.AD_d.dot(f, out=result)
            elif self.int_mode == 'sparse':
                self.AD_d.dot(f, out=result)
                result += np.append([0], np.repeat(
                    np.cumsum(result[0:-1:self.Np - 1]), self.Np - 1))

    def set_BoundCond(self, BoundCond):
        """Set new boundary conditions for the Stourm-Liouville probelem


        Parameters
        ----------
        BoundCond : dictionary
            Specify boundary conditions 
            keys : 'leftB', 'rightB', (optionally: 'leftBCoeff', 'rightBCoeff')
            values : 'Dirichlet' 'Neumann' or 'Robin'. If 'Robin', addionally specify 
            coefficients as a dictionary with two keys: [c1,c2], consistent with the boundary condition 
            of the form: c1*y(B)+c2*y'(B)=0
            Example: {'leftB':'Robin','leftBCoeff':{'c1'=1, 'c2'=2}, 'rightB':'Robin','rightBCoeff':{'c1'=3, 'c2'=4}}
            The default is  {'leftB': 'Neumann', 'rightB': 'Neumann, 'leftBCoeff': {'c1': 1, 'c2': 2} }

        """
        # Check parameters, set new boundary conditions and calculate new Nullspace projector
        self.BoundCond = BoundCond
        self._check_and_set_params()
        self._get_BC()
        self._get_Nullspace()

    def solve_EV(self, peq=None, D=1, q=None, w=None, mode='hdark', fr=None, Nv=64):
        """Solve the Sturm-Liouville eigenvalue-eigenvector problem. 
        The problem can be specified either by peq, q and w functions or by the precalculated stiffmat and massmat


        Parameters
        ----------
        peq : numpy array, dtype=float 
            Equilibirum probabilioty distribution that determines potential Phi(x), see Suplementary Note 1.1. 1D array.     
        D : float
            Noise magnitude.         
        q : numpy array, dtype=float 
            A function q(x) in the S-L problem. The default value is None, in this case q(x)=0
        w : numpy array, dtype=float
            A function w(x) in the S-L problem (non-negative). The default is None, in this case w(x)=1 
        mode : str
            Specify mode. Availiable modes:
                'normal': solve Sturm-Liouville problem, ignore D and fr.
                'h0': solve for eigenvalues and vectors of FP operator H0.
                'hdark': solve for eigenvalues and vector of FP and H operator
            The default is 'hdark'.    
        fr : numpy array
            The firing rate function (required for 'hdark' mode). 
            This firing rate function is an elementwise sum of the firing rate functions of all the neuronal responses. 
            The default is None.
        Nv : int
            A number of eigenvectors/eigenvalues returned. The default is 64.

        Returns
        -------    
        lQ : numpy array (Nv,), dtype=float 
            The least Nv eigenvalues for the eigenvalue problem of H0 operator.
        QxOrig : numpy array (Nv,Nv), dtype=float 
            The corresponding scaled eigenvectors
        Qx : numpy array (Nv,Nv), dtype=float
            The eigenvectors of EV problem of H0 operator (only for 'h0' and 'hdark' modes).
        lQd: numpy array (Nv,), dtype=float
            The eigenvalues of H operator (only for 'hdark' mode).
        Qd: numpy array (Nv,Nv), dtype=float
            The corresponding eigenvectors in H0 basis (only for 'hdark' mode).

        """

        assert(mode in {'normal', 'h0', 'hdark'}), 'Incorrect mode!'
        # Fill peq and w with ones if needed
        if peq is None:
            peq = np.ones(self.N)
        if w is None:
            w = np.ones(self.N)

        # If mode is normal do not use D. Otherwise, multiply peq by D and flip sign
        if mode == 'normal':
            self._setmat(peq, q, w)
        else:
            self._setmat(-D * peq, q, w)

        # Solve eigenvalue problem with symmetric matrices
        if self.method['name'] == 'FD':
            lQ, QxOrig = linalg.eigh(
                self.stiffmat_full_.A, self.massmat_full_.A, eigvals=(0, Nv - 1))
        elif self.method['name'] == 'SEM':
            lQ, QxOrig = linalg.eigh(
                self.stiffmat_red_.A, self.massmat_red_.A, eigvals=(0, Nv - 1))

        # Define solution at the domain boundaries if needed:
        if self.method['name'] == 'FD':
            c1 = self.BC_[1] / (self.BC_[1] - self.BC_[0] * self.dx)
            c2 = self.BC_[3] / (self.BC_[3] + self.BC_[2] * self.dx)
            QxOrig = np.concatenate([np.reshape(QxOrig[1, :] * c1, (1, Nv)), QxOrig,
                                     np.reshape(QxOrig[-2, :] * c2, (1, Nv))])
        elif self.method['name'] == 'SEM':
            if self.BC_method == 'bound_subst':
                if self.BC_[1] == 0:
                    QxOrig = np.concatenate(
                        (np.reshape(np.zeros(Nv), (1, Nv)), QxOrig), axis=0)
                if self.BC_[3] == 0:
                    QxOrig = np.concatenate(
                        (QxOrig, np.reshape(np.zeros(Nv), (1, Nv))), axis=0)
            elif self.BC_method == 'projection':
                QxOrig = self.NullM_.dot(QxOrig)

        # Rescale eigenvectors by sqrt(peq) to obtain original eigenvectors of FP operator
        if mode == 'h0' or mode == 'hdark':
            Qx = sparse.diags(np.sqrt(peq), 0).dot(QxOrig)

        assert(all(np.abs(lQ[i]) <= np.abs(lQ[i + 1]) for i in range(len(lQ) - 1))), \
            'Error! Returned eigenvalues are not sorted'

        # Perform additional computations for 'hdark' mode
        if mode == 'hdark':

            # Eigenvalue/vectors of dark operator
            Kd = np.diag(lQ) + Qx.T.dot(sparse.diags(self.w_d * fr, 0).dot(Qx))
            lQd, Qd = linalg.eigh(Kd)

            assert(all(lQd[i] <= lQd[i + 1] for i in range(len(lQd) - 1))
                   ), 'Error! Returned EVVd not sorted'

        # return:
        if mode == 'normal':
            return lQ, QxOrig
        elif mode == 'h0':
            return lQ, QxOrig, Qx
        elif mode == 'hdark':
            return lQ, QxOrig, Qx, lQd, Qd

    def _setmat(self, p, q, w):
        """Calculate stiffness and mass matrices. 
        Sets stiffmat_full_, massmat_full_, stiffmat_red_, massmat_red_ matrices


        Parameters
        ----------
        p : numpy array, dtype=float 
            function p(x) in S-L problem

        q : numpy array, dtype=float 
            function q(x) in S-L problem

        w : numpy array, dtype=float 
            function w(x) in S-L problem


        """

        if self.method['name'] == 'FD':

            # Need p(x) between grid points
            p_bp = 0.5 * (p[0:-1] + p[1:])

            # Use central differences and strong formulation to set stiffmat
            self.stiffmat_full_ = sparse.diags(p_bp[1:-1], 1) + sparse.diags(p_bp[1:-1], -1) - \
            sparse.diags(((p_bp + np.roll(p_bp, -1))[0:-1]), 0)

            # Take care of boundary conditions
            # By substituting y[0] and y[-1] into the reduced (N-2,N-2) system
            c1 = self.BC_[1] / (self.BC_[1] - self.BC_[0] * self.dx)
            c2 = self.BC_[3] / (self.BC_[3] + self.BC_[2] * self.dx)
            self.stiffmat_full_[0, 0] += c1 * \
                p_bp[0]  # Take care of BC[xbegin]
            self.stiffmat_full_[-1, -1] += c2 * \
                p_bp[-1]  # Take care of BC[xend]

            # Scale
            self.stiffmat_full_ /= self.dx**2

            # Add diagonal part proportional to q(x)
            if q is not None:
                self.stiffmat_full_ += sparse.diags(q[1:-1], 0)

            # calculate mass matrix
            self.massmat_full_ = sparse.diags(w[1:-1], 0)

        elif self.method['name'] == 'SEM':

            # Patch stiffness matrix
            # for i in range(0, self.Ne):
            #    idx_s, idx_e = i * (self.Np - 1), i * (self.Np - 1) + self.Np
            #    self.stiffmat_full_[idx_s:idx_e, idx_s:idx_e] -= self.dmat_.T.dot(
            #           np.diag(self.w_ * p[idx_s:idx_e])).dot(self.dmat_)

            # Different way of patching stiffness matrix:
            # temporary store a value at stiching point and add it up when needed
            pr_node_temp = 0
            for i in range(0, self.Ne):
                idx_s, idx_e = i * (self.Np - 1), i * (self.Np - 1) + self.Np
                self.stiffmat_full_[idx_s:idx_e, idx_s:idx_e] = - self.dmat_.T.dot(
                    np.diag(self.w_ * p[idx_s:idx_e])).dot(self.dmat_)
                self.stiffmat_full_[idx_s, idx_s] += pr_node_temp
                pr_node_temp = self.stiffmat_full_[idx_e - 1, idx_e - 1]

            # Add diagonal part proportional to q(x)
            if q is not None:
                self.stiffmat_full_ += sparse.diags(q * self.w_d, 0)

            if self.BC_method == 'bound_subst':

                # Take care of boundary conditions:
                if self.BC_[1] != 0:
                    self.stiffmat_full_[0, 0] += p[0] * \
                        self.BC_[0] / self.BC_[1]
                if self.BC_[3] != 0:
                    self.stiffmat_full_[-1, -1] -= p[self.N -
                                                     1] * self.BC_[2] / self.BC_[3]

                # Reduce matrix sizes if needed
                idx_s, idx_e = (self.BC_[1] == 0), self.N - (self.BC_[3] == 0)
                self.stiffmat_red_ = self.stiffmat_full_[
                    idx_s:idx_e, idx_s:idx_e]
                self.massmat_red_ = sparse.diags(
                    self.w_d[idx_s:idx_e] * w[idx_s:idx_e], 0)

            elif self.BC_method == 'projection':

                # Take care of terms from integration by parts:
                self.stiffmat_full_[0, :] -= p[0] * self.dmat_d[0, :]
                self.stiffmat_full_[-1, :] += p[-1] * self.dmat_d[-1, :]
                self.massmat_full_ = sparse.diags(self.w_d * w, 0)

                self.stiffmat_full_ = sparse.csr_matrix(self.stiffmat_full_)
                self.massmat_full_ = sparse.csr_matrix(self.massmat_full_)

                # Project onto nullspace:
                self.massmat_red_ = self.NullM_.T.dot(
                    self.massmat_full_.dot(self.NullM_))
                self.stiffmat_red_ = self.NullM_.T.dot(
                    self.stiffmat_full_.dot(self.NullM_))

                self.stiffmat_full_ = sparse.lil_matrix(self.stiffmat_full_)
                self.massmat_full_ = sparse.lil_matrix(self.massmat_full_)

    def _get_grid(self):
        """Calculate grid nodes, corresponding weights and differentiation matrix (with SEM method)


        Sets
        ----
        x_d, w_d, dmat_d, dx, x_, w_, dmat_, ele_scale_
        """

        if self.method['name'] == 'FD':

            # In this case grid and weights are uniform. No differentiation matrix required
            self.x_d = np.linspace(self.xbegin, self.xend, self.N)
            self.dx = (self.xend - self.xbegin) / (self.N - 1)
            self.w_d = self.dx * np.ones(self.x_d.shape)
            self.w_d[0] /= 2
            self.w_d[-1] /= 2

        elif self.method['name'] == 'SEM':

            # Scaling factor:
            self.ele_scale_ = (self.xend - self.xbegin) / (2 * self.Ne)

            # Calculate local grid, weights, differentiation matrix
            if self.grid_mode_calc == 'built_in':
                self._get_single_element()
            elif self.grid_mode_calc == 'newton':
                self._get_single_element_numerics()

            # Now patch locals to get globals:
            self.x_d = np.zeros(self.N)
            self.w_d = np.zeros(self.N)
            self.dmat_d = sparse.lil_matrix((self.N, self.N), dtype=np.float64)

            for i in range(self.Ne):
                patch = np.arange(i * (self.Np - 1), i *
                                  (self.Np - 1) + self.Np)

                # Patch as described in SEM documentation
                self.x_d[patch] = self.x_ + (2 * i + 1) * self.ele_scale_
                self.w_d[patch] += self.w_
                self.dmat_d[np.ix_(patch, patch)] += self.dmat_

            self.x_d += self.xbegin

            # Divide rows that correspond to primary nodes by 2:
            for i in range(self.Ne - 1):
                self.dmat_d[i * (self.Np - 1) + self.Np - 1, :] /= 2.0

            self.dmat_d = self.dmat_d.tocsr()

    def _get_single_element(self):
        """Calculate local grid nodes, corresponding weights and differentiation matrix
        using numpy.polynomial.legendre module


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
                    self.dmat_[i][j] = Ln[i] / \
                        (Ln[j] * (self.x_[i] - self.x_[j]))
                else:
                    self.dmat_[i][i] = 0
        self.dmat_[0, 0] = -(self.Np - 1) * (self.Np) / 4
        self.dmat_[-1, -1] = (self.Np - 1) * (self.Np) / 4

        # Scale locals:
        self.x_ *= self.ele_scale_
        self.w_ *= self.ele_scale_
        self.dmat_ /= self.ele_scale_

    def _get_single_element_numerics(self):
        """Calculate local grid nodes, corresponding weights and differentiation matrix
        using Newton method for the root finding of Legendre polynomials


        Sets:
        -----
        x_, w_, dmat_

        """

        L = np.float64(np.arange(self.Np)) / (self.Np - 1)

        # First guess:
        self.x_ = np.cos(np.pi * L)

        P = np.zeros((self.Np, self.Np))

        # Far initialization for Newton-Raphson method.
        xold = 2
        while max(abs(self.x_ - xold)) > MACHINE_EPSILON:
            xold = self.x_
            P[:, 0] = 1
            P[:, 1] = self.x_

            # Use recursive definition of Legendre polynomials
            for k in range(2, self.Np):
                P[:, k] = ((2 * k - 1) * self.x_ * P[:, k - 1] -
                           (k - 1) * P[:, k - 2]) / k

            self.x_ = xold - \
                (self.x_ * P[:, self.Np - 1] - P[:, self.Np - 2]
                 ) / (self.Np * P[:, self.Np - 1])

        # calculate weights
        self.w_ = 2 / ((self.Np - 1) * self.Np * P[:, self.Np - 1]**2)

        # Flip grid and weights
        self.x_ = np.flipud(self.x_)
        self.w_ = np.flipud(self.w_)

        # Takes the last polynomial (L_N(x_i)) evaluated at the nodes and flips the values.
        Pend = np.flipud(P[:, -1])

        self.dmat_ = np.zeros(np.shape(P))

        # Gets the derivatives of the Lagrange polynomials in the space.
        for i in range(self.Np):
            for j in range(self.Np):
                if i != j:
                    self.dmat_[i][j] = Pend[i] / \
                        (Pend[j] * (self.x_[i] - self.x_[j]))
                else:
                    self.dmat_[i][i] = 0

        self.dmat_[0, 0] = -self.Np * (self.Np - 1) / 4
        self.dmat_[-1, -1] = self.Np * (self.Np - 1) / 4

        # Scaling
        self.x_ *= self.ele_scale_
        self.w_ *= self.ele_scale_
        self.dmat_ /= self.ele_scale_

    def _get_BC(self):
        """Create _BC array that contains boundary condition coefficients that is
        consistent with the following representation: 
           BC_[0]*y(xbegin)+BC_[1]*y'(xbegin)=0
           BC_[2]*y(xend)+BC_[3]*y'(xend)=0   
        """

        if isinstance(self.BoundCond, np.ndarray):
            self.BC_ = self.BoundCond
        else:
            self.BC_ = np.zeros(4)
            if self.BoundCond['leftB'] == 'Robin':
                self.BC_[0] = self.BoundCond['leftBCoeff']['c1']
                self.BC_[1] = self.BoundCond['leftBCoeff']['c2']
            else:
                self.BC_[:2] = {'Dirichlet': np.array([1, 0]), 'Neumann': np.array(
                    [0, 1])}.get(self.BoundCond['leftB'])

            if self.BoundCond['rightB'] == 'Robin':
                self.BC_[2] = self.BoundCond['rightBCoeff']['c1']
                self.BC_[3] = self.BoundCond['rightBCoeff']['c2']
            else:
                self.BC_[2:] = {'Dirichlet': np.array([1, 0]), 'Neumann': np.array(
                    [0, 1])}.get(self.BoundCond['rightB'])

    def _get_Nullspace(self):
        """Calculates Nullspace of a projection on boundary conditions operator
        """
        if self.BC_method == 'projection':
            BCmat = np.zeros((2, self.N))
            BCmat[0, :] = np.append(self.BC_[0], np.zeros(
                (1, self.N - 1))) + self.BC_[1] * self.dmat_d[0, :]
            BCmat[1, :] = np.append(
                np.zeros((1, self.N - 1)), self.BC_[2]) + self.BC_[3] * self.dmat_d[-1, :]
            self.NullM_ = nullspace(BCmat)
            self.NullM_ = sparse.csr_matrix(self.NullM_)

    def _set_AD_mat(self):
        """Calculates Integration Matrix that can be used to calculate antiderivative
        Options:
            "full" - full antiderivative matrix, 
            "sparse" - Sparse antiderivative matrix. 
        """

        # Define local grid at a single element xi \in [-1;1]
        x_local = self.x_ / self.ele_scale_

        # Allocate local and global antiderivative matrix
        self.AD_ = np.zeros((self.Np, self.Np))
        self.AD_d = np.zeros((self.N, self.N))

        # Construct local matrix first:
        # integration coefficients of x, x^2, x^3, ... of Lagrange interpolation polynomials
        coefs = np.zeros(self.Np)
        coefs[-1] = 1 / self.Np
        # Matrix with columns x, x^2, ..., x^N
        x_mat = (np.transpose(np.matlib.repmat(x_local, self.Np, 1))
                 )**np.arange(1, self.Np + 1)
        for i in range(self.Np):
            # take of all but current grid points:
            inds = np.append(np.arange(i), np.arange(i + 1, self.Np))
            x_crop = x_local[inds]

            # Calculate integration coefficients and common denominator using sums of all single, pairwise, triplewise, etc. combinations
            Combinations = [sum(reduce(mul, c) for c in combinations(
                x_crop, i + 1)) for i in range(self.Np - 1)]
            coefs[:-1] = ((-1)**np.arange(1 - self.Np%2, self.Np - self.Np %
                          2)) * Combinations[::-1] / np.arange(1, self.Np)
            denominator = np.prod(np.ones(self.Np - 1) * x_local[i] - x_crop)

            # Choose integration constant c0 such that F(-1)=0
            c0 = -np.sum((-1)**np.arange(1, self.Np + 1) * coefs)

            # Calculate differentiation matrix
            self.AD_[:, i] = (x_mat.dot(coefs) + c0) / denominator

        # Set first row to zero and scale
        self.AD_[0, :] = 0
        self.AD_ *= self.ele_scale_

        # Now calculate global AD matrix:
        if self.int_mode == 'full':
            for i in range(self.Ne):
                patch = np.arange(i * (self.Np - 1), i *
                                  (self.Np - 1) + self.Np)
                self.AD_d[np.ix_(patch, patch)] += self.AD_
                self.AD_d[np.ix_(
                    np.arange(i * (self.Np - 1) + self.Np, self.N), patch)] += self.AD_[-1, :]
        elif self.int_mode == 'sparse':
            for i in range(self.Ne):
                patch = np.arange(i * (self.Np - 1), i *
                                  (self.Np - 1) + self.Np)
                self.AD_d[np.ix_(patch, patch)] += self.AD_

    def _get_matrices(self):
        """ Allocate full and reduced stiffness and mass matrices and Nullspace of boundary operator
        """

        # With Finite difference use only full matrices. Size is (N-2)x(N-2)
        if self.method['name'] == 'FD':
            self.stiffmat_full_ = sparse.lil_matrix(
                (self.N - 2, self.N - 2), dtype=np.float64)
            self.massmat_full_ = sparse.lil_matrix(
                (self.N - 2, self.N - 2), dtype=np.float64)

        elif self.method['name'] == 'SEM':

            # Full matrices are of size NxN
            self.stiffmat_full_ = sparse.lil_matrix(
                (self.N, self.N), dtype=np.float64)
            self.massmat_full_ = sparse.lil_matrix(
                (self.N, self.N), dtype=np.float64)

            # Indices to switch from full to reduced matrices
            idx_s, idx_e = (self.BC_[1] == 0), self.N - (self.BC_[3] == 0)

            # Allocate reduced matrices
            if self.BC_method == 'bound_subst':
                self.stiffmat_red_ = sparse.csr_matrix(
                    (idx_e - idx_s, idx_e - idx_s), dtype=np.float64)
                self.massmat_red_ = sparse.csr_matrix(
                    (idx_e - idx_s, idx_e - idx_s), dtype=np.float64)
            elif self.BC_method == 'projection':
                self.stiffmat_red_ = sparse.csr_matrix(
                    (self.N - 2, self.N - 2), dtype=np.float64)
                self.massmat_red_ = sparse.csr_matrix(
                    (self.N - 2, self.N - 2), dtype=np.float64)

    def _check_and_set_params(self):
        """Check the initialized parameters. Set self.N, self.Np, self.Ne
        """

        assert (isinstance(self.xbegin, numbers.Number) and isinstance(self.xend, numbers.Number)), \
        'xbegin and xend must be a numbers'
        assert (self.xend >= self.xbegin), 'x interval length is <= 0'
        assert (self.method['name'] in self._MethodList), 'Unidentified method'
        assert (all(isinstance(item, int) for item in list(self.method['gridsize'].values()))), \
        'N, Np or Ne are not integers'
        assert (self.grid_mode_calc in {
                'built_in', 'newton'}), 'Unknown grid_mode_clc'
        assert (self.int_mode in {'full', 'sparse'}
                ), 'Unknown integration mode'
        if self.method['name'] == 'FD':
            self.N = self.method['gridsize']['N']
            assert(self.N > 0), 'N is incorrect'
        elif self.method['name'] == 'SEM':
            self.N = self.method['gridsize']['Ne'] * \
                (self.method['gridsize']['Np'] - 1) + 1
            self.Np = self.method['gridsize']['Np']
            self.Ne = self.method['gridsize']['Ne']
            assert (self.Np > 0 and self.Ne > 0 and self.N >
                    0), 'Total number of grid points incorrect'
        if isinstance(self.BoundCond, np.ndarray):
            assert (len(self.BoundCond) == 4 and np.abs(self.BoundCond[0]) + np.abs(self.BoundCond[1]) > 0 and
                    np.abs(self.BoundCond[2]) + np.abs(self.BoundCond[3]) > 0), 'Incorrect Boundary conditions'
        else:
            assert (
                'leftB' in self.BoundCond and 'rightB' in self.BoundCond), 'Incorrect Boundary Conditions'
            assert (self.BoundCond['leftB'] in [
                    'Dirichlet', 'Neumann', 'Robin']), 'Unknown left boundary condition'
            assert (self.BoundCond['rightB'] in [
                    'Dirichlet', 'Neumann', 'Robin']), 'Unknown right boundary condition'
            if self.BoundCond['leftB'] == 'Robin':
                assert ('leftBCoeff' in self.BoundCond.keys()
                        ), 'leftBCoeff entry is missing in BoundCond'
                assert ('c1' in self.BoundCond['leftBCoeff'].keys() and 'c2' in self.BoundCond['leftBCoeff'].keys(
                )), 'values for Robin left boundary condition unspecifyed'
                assert (isinstance(self.BoundCond['leftBCoeff']['c1'], numbers.Number) and isinstance(
                    self.BoundCond['leftBCoeff']['c2'], numbers.Number)), 'values for Robin left boundary condition are incorrect'
            if self.BoundCond['rightB'] == 'Robin':
                assert ('rightBCoeff' in self.BoundCond.keys()
                        ), 'rightBCoeff entry is missing in BoundCond'
                assert ('c1' in self.BoundCond['rightBCoeff'] and 'c2' in self.BoundCond['rightBCoeff']
                        ), 'values for Robin right boundary condition unspecifyed'
                assert (isinstance(self.BoundCond['rightBCoeff']['c1'], numbers.Number) and isinstance(
                    self.BoundCond['rightBCoeff']['c2'], numbers.Number)), 'values for Robin right boundary condition are incorrect'
