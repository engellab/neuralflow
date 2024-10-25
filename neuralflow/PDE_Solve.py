# -*- coding: utf-8 -*-

"""This class solves the following eigenvector-eigenvalue problems:
    1) Fokker-Planck equation
    2) Modified Fokker-Planck equation
    3) Stourm-Liouville problem
"""

from neuralflow.grid import GLLgrid
from neuralflow.utilities.rank_nullspace import nullspace
import numpy as np
import numpy.matlib
import numbers
import scipy
from scipy import linalg
from copy import deepcopy


class PDESolve:
    """Numerical solution of Stourm-Liouville problem and Fokker-Planck.


    Parameters
    ----------
    xbegin : float, optional
        The left boundary of the latent state. The default is -1.
    xend : float
        The right boundary of the latent state. The default is 1.
    Np : int
        The degree of Langrange interpolation polynomial, also the
        number of grid points at each element. The default is 8.
    Ne : int
        Number of SEM elements. The default is 64.
   BoundCond : dict
       A dictionary that specifies boundary conditions (Dirichlet, Neumann
       or Robin). The default is {'leftB': 'Dirichlet',
                                  'rightB': 'Dirichlet'}.
       In case of Robin, the dictionary must also contain 'leftBCoeff'
       and/or 'rightBCoeff', each are dictionaries with two entries
       'c1' and 'c2' that specify BCs coefficients in the following format:
       c1*y[xbegin]+c2*y'[xbegin]=0 (left boundary)
       c1*y[xend]+c2*y'[xend]=0 (right boundary)
    Nv : int, optional
        Number of retained eigenvalues and eigenvectors of the operator H.
        If set to None, will be equal to grid.N-2, which is the maximum
        possible value. If Dirichlet BCs are used, it is stongly
        recommended to set this to None to avoid spurious high-freq
        oscillations in the fitted functions. The default is None.
    with_cuda : bool, optional
        Whether to include GPU support. For GPU optimization, the platform
        has to be cuda-enabled, and cupy package has to be installed. The
        default is False.
    """

    # List of availible methods
    _availible_BCs = ['Dirichlet', 'Neumann', 'Robin']

    def __init__(self, xbegin=-1.0, xend=1.0, Np=8, Ne=64,
                 BoundCond={'leftB': 'Dirichlet', 'rightB': 'Dirichlet'},
                 Nv=None, with_cuda=False
                 ):
        """


        Public methods
        ------
        set_BoundCond, solve_EV

        """

        self.BoundCond = deepcopy(BoundCond)
        self.with_cuda = with_cuda

        if with_cuda:
            import neuralflow.base_cuda as cuda
            self.cuda = cuda
            self.cuda_var = cuda.var()

        # Check inputs
        self._check_inputs()

        # Convert given boundary condition into a vector BC_ of size (1,4)
        self._get_BC()

        # get grid
        self.grid = GLLgrid(xbegin, xend, Np, Ne, with_cuda)

        self.Nv = Nv
        if Nv is None:
            self.Nv = self.grid.N-2

        # Get the Nullspace
        self._get_Nullspace()

    @classmethod
    def init_from_grid(cls, grid, boundary_mode=None,  BoundCond=None):
        """Init from grid object


        Parameters
        ----------
        grid : neuralflow.grid
            Grid object.
        boundary_mode : str, optional
            Absorbing or reflecting. If None, BoundCond will be used for
            boundary conditions. The default is None.
        BoundCond : dict
            A dictionary that specifies boundary conditions (Dirichlet, Neumann
            or Robin). The default is {'leftB': 'Dirichlet',
                                       'rightB': 'Dirichlet'}.
            In case of Robin, the dictionary must also contain 'leftBCoeff'
            and/or 'rightBCoeff', each are dictionaries with two entries
            'c1' and 'c2' that specify BCs coefficients in the following
            format:
            c1*y[xbegin]+c2*y'[xbegin]=0 (left boundary)
            c1*y[xend]+c2*y'[xend]=0 (right boundary)

        Returns
        -------
        self
            Initialized PDESolve object.

        """
        pde_solve_params = {
            'xbegin': grid.xbegin, 'xend': grid.xend, 'Np': grid.Np,
            'Ne': grid.Ne, 'with_cuda': grid.with_cuda}
        if boundary_mode is not None:
            if boundary_mode == 'absorbing':
                pde_solve_params['BoundCond'] = {
                    'leftB': 'Dirichlet', 'rightB': 'Dirichlet'
                }
            elif boundary_mode == 'reflecting':
                pde_solve_params['BoundCond'] = {
                    'leftB': 'Neumann', 'rightB': 'Neumann'
                }
        else:
            pde_solve_params['BoundCond'] = BoundCond
        return cls(**pde_solve_params)

    def set_BoundCond(self, BoundCond):
        """Set new boundary conditions for the Stourm-Liouville probelem


        Parameters
        ----------
        BoundCond : dictionary or str
            If str, can be 'absorbing' or 'reflecting'.
            Alternatively, specify boundary conditions as a dict:
            keys : 'leftB', 'rightB', (optionally: 'leftBCoeff', 'rightBCoeff')
            values : 'Dirichlet' 'Neumann' or 'Robin'. If 'Robin', addionally
            specify coefficients as a dictionary with two keys: [c1,c2],
            consistent with the boundary condition of the form:
            c1*y(B)+c2*y'(B)=0.
            Example: {'leftB':'Robin','leftBCoeff':{'c1'=1, 'c2'=2},
                      'rightB':'Robin','rightBCoeff':{'c1'=3, 'c2'=4}}

        """
        # Check parameters, set new boundary conditions and calculate new
        # Nullspace projector
        if type(BoundCond) is str:
            if BoundCond == 'absorbing':
                bc = {'leftB': 'Dirichlet', 'rightB': 'Dirichlet'}
            elif BoundCond == 'reflecting':
                bc = {'leftB': 'Neumann', 'rightB': 'Neumann'}
            else:
                raise ValueError('Unknown boudnary conditions/mode')
        else:
            bc = BoundCond

        self.BoundCond = bc
        self._check_inputs()
        self._get_BC()
        self._get_Nullspace()

    def solve_EV(self, peq=None, D=1, q=None, w=None, mode='hdark', fr=None,
                 device='CPU'):
        """Solve the Sturm-Liouville/FP/FPE eigenvalue-eigenvector problem.


        Parameters
        ----------
        peq : numpy array, dtype=float
            Equilibirum probabilioty distribution that determines potential
            Phi(x), or a function p(x) in S-L problem.
        D : float
            Noise magnitude.
        q : numpy array, dtype=float
            A function q(x) in the S-L problem. The default value is None, in
            this case q(x)=0.
        w : numpy array, dtype=float
            A function w(x) in the S-L problem (non-negative). The default is
            None, in this case w(x)=1.
        mode : str
            Specify mode. Availiable modes:
                'normal': solve Sturm-Liouville problem, ignore D and fr.
                'h0': solve for eigenvalues and vectors of FP operator H0.
                'hdark': solve for eigenvalues and vector of FP operator H.
            The default is 'hdark'.
        fr : numpy array
            The firing rate function (required for 'hdark' mode).
            This firing rate function is an elementwise sum of the firing rate
            functions of all the neuronal responses. The default is None.
       device : str
           Can be 'CPU' or 'GPU'.

        Returns
        -------
        lQ : numpy array (Nv,), dtype=float
            The least Nv eigenvalues for the eigenvalue problem of H0 operator.
        QxOrig : numpy array (Nv,Nv), dtype=float
            The corresponding scaled eigenvectors
        Qx : numpy array (Nv,Nv), dtype=float
            The eigenvectors of EV problem of H0 operator (only for 'h0' and
            'hdark' modes).
        lQd: numpy array (Nv,), dtype=float
            The eigenvalues of H operator (only for 'hdark' mode).
        Qd: numpy array (Nv,Nv), dtype=float
            The corresponding eigenvectors in H0 basis (only for 'hdark' mode).

        """

        assert (mode in {'normal', 'h0', 'hdark'}), 'Incorrect mode!'

        if device not in ['CPU', 'GPU']:
            raise ValueError(f'Unknown device {device}')
        elif device == 'GPU' and not self.with_cuda:
            raise ValueError('Initialize the class variable with with_cuda = '
                             'True to support GPU computations')

        lib = self.cuda.cp if device == 'GPU' else np

        # Fill peq and w with ones if needed
        if peq is None:
            peq = lib.ones(self.grid.N, dtype='float64')
        if w is None:
            w = lib.ones(self.grid.N, dtype='float64')
        Nv = self.Nv
        if Nv is None:
            Nv = self.grid.N - 2

        # If mode is normal do not use D. Otherwise, multiply peq by D and
        # flip sign
        if mode == 'normal':
            self._setmat(peq, q, w, device)
        else:
            self._setmat(-D * peq, q, w, device)

        if device == 'GPU':
            eigh, cpx = self.cuda.cp.linalg.eigh, self.cuda.cupyx
            stiffmat = self.cuda_var.stiffmat_
            massmat = self.cuda_var.massmat_
            dmassmat = self.cuda_var.dmassmat_
            NullM_ = self.cuda_var.NullM_
            w_d = self.grid.cuda_var.w_d
        else:
            eigh = linalg.eigh
            stiffmat, massmat = self.stiffmat_, self.massmat_
            NullM_ = self.NullM_
            w_d = self.grid.w_d

        # Solve EV
        if device == 'CPU':
            # In scipy >= 1.14.0 argument eigvals chaged to subset_by_index
            major, minor = [int(el) for el in scipy.__version__.split('.')[:2]]
            if major >= 1 and minor >= 14:
                lQ, QxOrig = eigh(
                    stiffmat, massmat, subset_by_index=(0, Nv - 1)
                )
            else:
                lQ, QxOrig = eigh(stiffmat, massmat, eigvals=(0, Nv - 1))
        else:
            # in cupy generalized EV is not supported. Thus convert it to conv
            # EV problem by multiplying both sides by inverted mass matrix
            if self.BC_[1] == 0 and self.BC_[3] == 0:
                # With Dirichlet BCs the projected massmat is symmetric
                temp1 = cpx.rsqrt(dmassmat)
                lQ, QxOrig = eigh(temp1[:, None]*stiffmat*temp1)
            else:
                # Invert and take Cholesky to find W^(-1/2)
                temp1 = lib.linalg.cholesky(lib.linalg.inv(massmat))
                lQ, QxOrig = eigh(temp1.T.dot(stiffmat.dot(temp1)))
            lQ = lQ[0:Nv]
            QxOrig = QxOrig[:, 0:Nv]
            if self.BC_[1] == 0 and self.BC_[3] == 0:
                QxOrig = QxOrig*temp1[:, None]
            else:
                QxOrig = temp1.dot(QxOrig)

        # Transfor back to N-dimensional basis
        QxOrig = NullM_.dot(QxOrig)

        # Rescale eigenvectors by sqrt(peq) to obtain original eigenvectors of
        # FP operator
        if mode == 'h0' or mode == 'hdark':
            if device == 'CPU':
                Qx = np.diag(lib.sqrt(peq), 0).dot(QxOrig)
            else:
                Qx = lib.sqrt(peq)[:, None]*QxOrig

        # Perform additional computations for 'hdark' mode
        if mode == 'hdark':

            # Eigenvalue/vectors of dark operator
            if device == 'CPU':
                Kd = np.diag(lQ) + Qx.T.dot(np.diag(w_d*fr, 0).dot(Qx))
            else:
                Kd = lib.diag(lQ) + (Qx.T*(w_d*fr)).dot(Qx)
            lQd, Qd = eigh(Kd)
            Qd = lib.asarray(Qd, order='C')

            if lib.diff(lQd).min() < 0:
                raise ValueError('Error! Returned EVVd not sorted')
            # assert(all(lQd[i] <= lQd[i + 1] for i in range(len(lQd) - 1))
            #         ), 'Error! Returned EVVd not sorted'

        # return
        if mode == 'normal':
            return lQ, QxOrig
        elif mode == 'h0':
            return lQ, QxOrig, Qx
        elif mode == 'hdark':
            return lQ, QxOrig, Qx, lQd, Qd

    def _setmat(self, p, q, w, device):
        """Calculate stiffness and mass matrices.
        Sets stiffmat_full_, massmat_full_, stiffmat_red_, massmat_red_
        matrices
        """

        if device == 'GPU':
            lib = self.cuda.cp
            dmat_ = self.grid.cuda_var.dmat_
            dmat_d = self.grid.cuda_var.dmat_d
            NullM_ = self.cuda_var.NullM_
            w_d = self.grid.cuda_var.w_d
            w_ = self.grid.cuda_var.w_
        else:
            lib = np
            dmat_ = self.grid.dmat_
            dmat_d = self.grid.dmat_d
            NullM_ = self.NullM_
            w_d = self.grid.w_d
            w_ = self.grid.w_

        N, Np, Ne = self.grid.N, self.grid.Np, self.grid.Ne
        stiffmat = lib.zeros((N, N), dtype='float64')
        # Patch stiffness matrix
        pr_node_temp = 0
        for i in range(0, Ne):
            idx_s, idx_e = i * (Np - 1), i * (Np - 1) + Np
            stiffmat[idx_s:idx_e, idx_s:idx_e] = - dmat_.T.dot(
                lib.diag(w_ * p[idx_s:idx_e])).dot(dmat_)
            stiffmat[idx_s, idx_s] = stiffmat[idx_s, idx_s]+pr_node_temp
            pr_node_temp = stiffmat[idx_e-1, idx_e-1].copy()

        # Add diagonal part proportional to q(x)
        if q is not None:
            stiffmat += lib.diag(q*w_d, 0)

        # Take care of terms from integration by parts:
        stiffmat[0, :] -= p[0]*dmat_d[0, :]
        stiffmat[-1, :] += p[-1]*dmat_d[-1, :]

        # Massmat
        massmat = lib.diag(w_d*w, 0)

        # Project into potent space of operator that preserve BCs
        massmat = NullM_.T.dot(massmat.dot(NullM_))
        stiffmat = NullM_.T.dot(stiffmat.dot(NullM_))

        if device == 'GPU':
            self.cuda_var.stiffmat_ = stiffmat
            self.cuda_var.massmat_ = massmat
            self.cuda_var.dmassmat_ = lib.diag(massmat)
        else:
            self.stiffmat_ = stiffmat
            self.massmat_ = massmat

    def _get_BC(self):
        """Create _BC array that contains boundary condition coefficients that
        is consistent with the following representation:
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
                self.BC_[:2] = {
                    'Dirichlet': np.array([1, 0]),
                    'Neumann': np.array([0, 1])
                }.get(self.BoundCond['leftB'])

            if self.BoundCond['rightB'] == 'Robin':
                self.BC_[2] = self.BoundCond['rightBCoeff']['c1']
                self.BC_[3] = self.BoundCond['rightBCoeff']['c2']
            else:
                self.BC_[2:] = {
                    'Dirichlet': np.array([1, 0]),
                    'Neumann': np.array([0, 1])
                }.get(self.BoundCond['rightB'])

    def _get_Nullspace(self):
        """Calculates Nullspace of a projection on boundary conditions operator
        """

        BCmat = np.zeros((2, self.grid.N))
        BCmat[0, :] = (
            np.append(self.BC_[0], np.zeros((1, self.grid.N - 1))) +
            self.BC_[1] * self.grid.dmat_d[0, :]
        )
        BCmat[1, :] = (
            np.append(np.zeros((1, self.grid.N - 1)), self.BC_[2]) +
            self.BC_[3] * self.grid.dmat_d[-1, :]
        )
        self.NullM_ = nullspace(BCmat)
        if self.with_cuda:
            self.cuda_var.NullM_ = self.cuda.cp.asarray(self.NullM_)

    def _check_inputs(self):
        """Check the initialized parameters.
        """

        # BoundCond set as array
        if isinstance(self.BoundCond, np.ndarray):
            if (
                len(self.BoundCond) != 4 or
                np.abs(self.BoundCond[0]) + np.abs(self.BoundCond[1]) == 0 or
                np.abs(self.BoundCond[2]) + np.abs(self.BoundCond[3]) == 0
            ):
                raise ValueError('Incorrect Boundary conditions')
        else:
            for bc in ['leftB', 'rightB']:
                if (
                    bc not in self.BoundCond or
                    self.BoundCond[bc] not in PDESolve._availible_BCs
                ):
                    raise ValueError('Incorrect Boundary Conditions')

            if self.BoundCond[bc] == 'Robin':
                if (
                    f'{bc}Coeff' not in self.BoundCond.keys() or
                    'c1' not in self.BoundCond[f'{bc}Coeff'].keys() or
                    'c2' not in self.BoundCond[f'{bc}Coeff'].keys() or
                    not isinstance(
                        self.BoundCond[f'{bc}Coeff']['c1'], numbers.Number) or
                    not isinstance(
                        self.BoundCond[f'{bc}Coeff']['c2'], numbers.Number)
                ):
                    raise ValueError('f{bc} dictionary is incorrect')

        if type(self.with_cuda) is not bool:
            try:
                self.with_cuda = bool(self.with_cuda)
            except Exception as e:
                raise TypeError(f'with_cuda must be boolean. Exception: {e}')
