#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Langeving model parameters and related functions
"""

import numpy as np
from copy import deepcopy
import numbers
from neuralflow.peq_models import peq_model_types_
from neuralflow.firing_rate_models import firing_model_types_
import logging
from neuralflow.settings import MINIMUM_PEQ

logger = logging.getLogger(__name__)


class model:
    """Model class supports one or multiple Langevin dynamics defined by
    peq(x), p0(x), D, fr(x). For multiple dynamics, some of the parameters
    can be shared which is defined by params_size dictionary. For example,
    if we want to model 4 conditions, we might want a model with four
    Langevin dynamics.


    Parameters
    ----------
    peq : numpy array, (num_models, grid.N)
        peq function evaluated on a grid.
    p0 : numpy array, (num_models, grid.N) or None
        p0 function evaluated on a grid. If set to None, assume equilibirum
        model.
    D : numpy array, (num_models,)
        Diffusion, or noise magnitude.
    fr : numpy array, (num_models, grid.N, num_neuron)
        Firing rate function for each neuron.
    params_size : dict
        Number of distinct functions in the model. For each of the
        parameters (peq, p0, D, fr), it specifies whether the parameter is
        shared across the dynamics (in which case the value is 1), or
        non-shared (in which case the value is equal to num_models). For
        example, if there are 4 Langevin dynamics with distinct potentials,
        but shared p0, D, and fr, then params_size = {'peq': 4, 'p0': 1,
        'D': 1, 'fr': 1}.
    grid : grid.GLLgrid object
        Initialized GLL grid. Should be initiatialized with with_cuda=True
        for GPU support.
    peq_model : dict, optional
        Specify peq model that was used to initialize peq. This is only
        needed to keep a record. The default is None.
    p0_model : dict, optional
        Specify p0 model that was used to initialize p0. This is only
        needed to keep a record. The default is None.
    fr_model : list, optional
        For each neuron this specifies a fr model. This is only
        needed to keep a record (and also can be used for data generation).
        The default is None.
    with_cuda : bool, optional
       Whether to include GPU support. For GPU optimization, the platform
       has to be cuda-enabled, and cupy package has to be installed. The
       default is False.
    """

    # Default initialization is from predefined models
    def __init__(self, peq, p0, D, fr, params_size, grid,
                 peq_model=None, p0_model=None, fr_model=None,
                 with_cuda=False):
        """
        Public methods
        ------
        new_model, get_params, get_params_for_grad, get_fr_lambda, sync_model,
        compute_rho0, peq_from_force, force_from_peq, Fr_from_fr, fr_from_Fr,
        FeatureComplexity, FeatureComplexityFderiv

        """
        self.grid = grid
        self.peq_model = peq_model
        self.p0_model = p0_model
        self.fr_model = fr_model

        # Equilibrium model: p0 = peq. Non-equilibirum model: p0 can differ
        if p0 is None:
            p0 = peq.copy()
            self.non_equilibrium = False
        else:
            self.non_equilibrium = True

        # Expand dimensions if needed. The first dimension is the model number
        if len(peq.shape) == 1:
            self.peq = peq[np.newaxis, :]
        else:
            self.peq = peq
        self.peq = np.maximum(self.peq, MINIMUM_PEQ)
        if len(p0.shape) == 1:
            self.p0 = p0[np.newaxis, :]
        else:
            self.p0 = p0
        # number of datasamples x N x number of neurons
        if len(fr.shape) == 1:
            self.fr = fr[np.newaxis, :, np.newaxis]
        elif len(fr.shape) == 2:
            self.fr = fr[np.newaxis, ...]
        else:
            self.fr = fr

        self.D = D
        self.num_neuron = self.fr.shape[-1]

        # Clone models if the number of datasamples in param_size exceed the
        # first dimension of the model arrays
        if self.peq.ndim < 2 or self.peq.shape[0] == 1:
            self.peq = np.repeat(self.peq, params_size['peq'], 0).astype(
                'float64'
            )
        if self.p0.ndim < 2 or self.p0.shape[0] == 1:
            self.p0 = np.repeat(self.p0, params_size['p0'], 0).astype(
                'float64'
            )
        if not isinstance(self.D, np.ndarray) or self.D.shape[0] == 1:
            self.D = np.repeat(self.D, params_size['D'], 0).astype('float64')
        if self.fr.ndim < 3 or self.fr.shape[0] == 1:
            self.fr = np.repeat(self.fr, params_size['fr'], 0).astype(
                'float64'
            )

        # define rho0
        params_size['rho0'] = max(params_size['peq'], params_size['p0'])
        if params_size['rho0'] == params_size['peq']:
            self.rho0 = np.zeros_like(self.peq)
        else:
            self.rho0 = np.zeros_like(self.p0)
        for model_num in range(params_size['rho0']):
            self.rho0[model_num, :] = (
                self.p0[min(model_num, self.p0.shape[0]-1), :] /
                np.maximum(
                    np.sqrt(self.peq[min(model_num, self.peq.shape[0]-1), :]),
                    10**-20
                )
            )

        # Compute C - needed for optimization. C is the value of fr at
        # x = xbegin
        self.C = self.fr[:, 0, :]

        self.params_size = params_size
        self.num_models = max(params_size.values())
        self.num_neuron = self.fr.shape[-1]

        self.with_cuda = with_cuda
        if self.with_cuda:
            # Make sure grid is initialized with cuda
            if not grid.with_cuda:
                raise ValueError(
                    'Grid has to be initialized with_cuda for GPU support'
                )
            import neuralflow.base_cuda as cuda
            self.cuda = cuda
            self.cuda_var = cuda.var()
            self.cuda_var.peq = self.cuda.cp.asarray(self.peq, dtype='float64')
            self.cuda_var.p0 = self.cuda.cp.asarray(self.p0, dtype='float64')
            self.cuda_var.rho0 = self.cuda.cp.asarray(
                self.rho0, dtype='float64'
            )
            self.cuda_var.D = self.D
            self.cuda_var.fr = self.cuda.cp.asarray(self.fr, dtype='float64')
            self.cuda_var.C = self.cuda.cp.asarray(self.C, dtype='float64')

    def __deepcopy__(self, memo):
        """ When with_cuda = True, self.cuda should be shared across instances
        """
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k != 'cuda':
                setattr(result, k, deepcopy(v, memo))
            else:
                setattr(result, k, v)
        return result

    @classmethod
    def new_model(cls, peq_model, p0_model, D, fr_model, grid,
                  params_size=None, with_cuda=False):
        """Initialize a model


        Parameters
        ----------
        peq_model : callable or dict or np.array
            peq function. Can be either callable lambda, or dictionary that
            specify peq from peq_models.py, or an array of peq values evaluated
            on the grid.
        p0_model : callable or dict or np.array
            p0 function. Can be either callable lambda, or dictionary that
            specify peq from peq_models.py, or an array of peq values evaluated
            on the grid.
        D : np.array or float
            Diffusion coefficint. Either an array that contains a value for
            each of the models, or a float.
        fr_model : list or np.array
            Firing rate functions. Either a list of callables/dict (one entry
            for each neuron), or np.array.
        grid : grid.GLLgrid object
            Initialized GLL grid. Should be initiatialized with with_cuda=True
            for GPU support.
        params_size : dict, optional
            Number of distinct functions in the model. For each of the
            parameters (peq, p0, D, fr), it specifies whether the parameter is
            shared across the dynamics (in which case the value is 1), or
            non-shared (in which case the value is equal to num_models). For
            example, if there are 4 Langevin dynamics with distinct potentials,
            but shared p0, D, and fr, then params_size = {'peq': 4, 'p0': 1,
            'D': 1, 'fr': 1}.
        with_cuda : bool, optional
           Whether to include GPU support. For GPU optimization, the platform
           has to be cuda-enabled, and cupy package has to be installed. The
           default is False.

        Returns
        -------
        model
            Initialized model.

        """

        if (callable(peq_model)):
            # Initialize by custom function
            peq = peq_model(grid.x_d, grid.w_d)
            peq_model_ = peq_model
        elif type(peq_model) is dict:
            # initialize by a default function
            if peq_model['model'] in peq_model_types_:
                if 'params' not in peq_model.keys():
                    peq_model['params'] = {}
                peq_model_ = peq_model
                peq = peq_model_types_[peq_model['model']](
                    grid.x_d, grid.w_d, **peq_model['params'])
            else:
                raise ValueError(f'Unknown model {peq_model}')
        else:
            # peq provided directly as array. Make sure to copy it
            peq = np.array(peq_model)
            peq_model_ = None

        if (callable(p0_model)):
            p0 = p0_model(grid.x_d, grid.w_d)
            p0_model_ = p0_model
        elif type(p0_model) is dict:
            # initialize by a default function
            if p0_model['model'] in peq_model_types_:
                if 'params' not in p0_model.keys():
                    p0_model['params'] = {}
                p0_model_ = p0_model
                p0 = peq_model_types_[p0_model['model']](
                    grid.x_d, grid.w_d, **p0_model['params'])
            else:
                raise ValueError(f'Unknown model {p0_model}')
        elif p0_model is None:
            p0 = None
            p0_model_ = None
        else:
            p0 = np.array(p0_model)
            p0_model_ = None

        if isinstance(fr_model, dict):
            fr_model = [fr_model]
        if type(fr_model) != np.ndarray:
            fr = np.empty((grid.N, len(fr_model)), dtype=np.float64)
            fr_model_ = []
            for i in range(len(fr_model)):
                if callable(fr_model[i]):
                    fr[:, i] = fr_model[i](grid.x_d)
                    fr_model_.append(fr_model)
                elif type(fr_model[i]) is dict:
                    if fr_model[i]['model'] in firing_model_types_:
                        if 'params' not in fr_model[i].keys():
                            fr_model[i]['params'] = {}
                        fr_model_.append(fr_model[i])
                        fr[:, i] = firing_model_types_[fr_model[i]['model']](
                            grid.x_d, **fr_model[i]['params'])
                    else:
                        raise ValueError(f'Unknown model {fr_model}')
        else:
            # fr provided directly
            fr = np.array(fr_model)
            fr_model_ = None

        if type(D) != np.ndarray:
            if type(D) is list:
                D = np.array(D)
            else:
                D = np.array([D])

        # Default param size
        if params_size is None:
            params_size = {
                'peq': peq.shape[0] if len(peq.shape) > 1 else 1,
                'p0': p0.shape[0] if p0 is not None and len(p0.shape) > 1
                else 1,
                'fr': fr.shape[0] if len(fr.shape) > 2 else 1,
                'D': D.size,
            }
        else:
            params_size = deepcopy(params_size)
        grid = deepcopy(grid)
        model._check_inputs(peq, p0, D, fr, params_size, grid)
        return cls(peq, p0, D, fr, params_size, grid, peq_model_, p0_model_,
                   fr_model_, with_cuda)

    def get_params(self, model_num, device='CPU'):
        """Extract peq, p0, D, fr
        """
        if device == 'CPU':
            var = self
        else:
            var = self.cuda_var
        peq = var.peq[min(model_num, self.params_size['peq'] - 1), ...]
        p0 = var.p0[min(model_num, self.params_size['p0'] - 1), ...]
        D = var.D[min(model_num, self.params_size['D'] - 1)]
        fr = var.fr[min(model_num, self.params_size['fr'] - 1), ...]
        return peq, p0, D, fr

    def get_params_for_grad(self, model_num, device='CPU'):
        """Extract peq, rho0, D, fr
        """
        if device == 'CPU':
            var = self
        else:
            var = self.cuda_var
        peq = var.peq[min(model_num, self.params_size['peq'] - 1), ...]
        rho0 = var.rho0[min(model_num, self.params_size['rho0'] - 1), ...]
        D = var.D[min(model_num, self.params_size['D'] - 1)]
        fr = var.fr[min(model_num, self.params_size['fr'] - 1), ...]
        return peq, rho0, D, fr

    def get_fr_lambda(self):
        """Extract firing rate function
        """
        if type(self.fr_model) is list:
            fr_model_lambdas = []
            for fr in self.fr_model:
                if type(fr) is dict:
                    fr_model_lambdas.append(
                        lambda x, fr=fr: firing_model_types_[fr['model']](
                            x, **fr['params']
                        )
                    )
                else:
                    fr_model_lambdas.append(fr)
            return fr_model_lambdas
        return None

    def sync_model(self, mode='GPU_to_CPU'):
        """Sync GPU and CPU by coping a model over.
        """
        if not self.with_cuda:
            self.logeer.warning('Model does not support GPU. Aborting')
            return
        if mode == 'GPU_to_CPU':
            self.peq = self.cuda.cp.asnumpy(self.cuda_var.peq)
            self.p0 = self.cuda.cp.asnumpy(self.cuda_var.p0)
            self.rho0 = self.cuda.cp.asnumpy(self.cuda_var.rho0)
            self.D = self.cuda_var.D
            self.fr = self.cuda.cp.asnumpy(self.cuda_var.fr)
            self.C = self.cuda.cp.asnumpy(self.cuda_var.C)
        elif mode == 'CPU_to_GPU':
            self.cuda_var.peq = self.cuda.cp.asarray(
                self.peq, dtype='float64'
            )
            self.cuda_var.p0 = self.cuda.cp.asarray(
                self.p0, dtype='float64'
            )
            self.cuda_var.rho0 = self.cuda.cp.asarray(
                self.rho0, dtype='float64'
            )
            self.cuda_var.D = self.D
            self.cuda_var.fr = self.cuda.cp.asarray(self.fr, dtype='float64')
            self.cuda_var.C = self.cuda.cp.asarray(self.C, dtype='float64')
        else:
            raise ValueError('Unknown mode')

    def compute_rho0(self, p0, peq, device='CPU', min_peq=10**-20):
        """Compute rho0 from p0 and peq


        Parameters
        ----------
        p0 : numpy array
            p0(x) distribution.
        peq : numpy array
            peq distribution.
        device : str, optional
            "CPU" or "GPU". The default is None, which is "CPU".
        min_peq : float, optional
            Min peq to avoid division by zero. The default is 10**-20.

        Returns
        -------
        rho0 : numpy array
            rho0 distribution.

        """
        if device == 'CPU' or device is None:
            rho0 = p0 / np.maximum(np.sqrt(peq), min_peq)
        else:
            rho0 = p0 / self.cuda.cp.maximum(self.cuda.cp.sqrt(peq), min_peq)
        return rho0

    def peq_from_force(self, F, device=None):
        """Calculates normalized peq from a given driving Force


        Parameters
        ----------
        F : numpy array
            Driving force.

        Returns
        -------
        peq : numpy array
            peq.

        """
        if device == 'CPU' or device is None:
            peq = np.exp(self.grid.AD_d.dot(F))
            peq /= np.sum(peq*self.grid.w_d)
            peq = np.maximum(peq, MINIMUM_PEQ)
        else:
            peq = self.cuda.cp.exp(self.grid.cuda_var.AD_d.dot(F))
            peq /= self.cuda.cp.sum(peq*self.grid.cuda_var.w_d)
            peq = self.cuda.cp.maximum(peq, MINIMUM_PEQ)
        return peq

    def force_from_peq(self, peq, device=None):
        """Calculates driving force F from peq


        Parameters
        ----------
        peq : numpy array
            peq.

        Returns
        -------
        F : numpy array
            Driving force.

        """

        if device == 'CPU' or device is None:
            return self.grid.dmat_d.dot(np.log(peq))
        return self.grid.cuda_var.dmat_d.dot(self.cuda.cp.log(peq))

    def Fr_from_fr(self, fr, device=None):
        """Calculates Fr (an auxiliary function for tuning function
        optimization) from fr (tuning function)


        Parameters
        ----------
        fr : numpy array
            An array that represents tuning function f(x).

        Returns
        -------
        Fr : numpy array
            An array that represents an auxiliary function Fr(x) for tuning
            function optimization.

        """
        if device == 'CPU' or device is None:
            return self.grid.dmat_d.dot(np.log(fr))
        else:
            return self.grid.cuda_var.dmat_d.dot(self.cuda.cp.log(fr))

    def fr_from_Fr(self, Fr, C=1, device=None):
        """Calculates fr (tuning function) from Fr (an auxiliary function for
        tuning function optimization) and C (a scaling constant)


        Parameters
        ----------
        Fr : numpy array
            Fr reprsented on the grid.
        C : float, optional
            Constant C. The default is 1.
        device : str, optional
            "CPU" or "GPU". The default is None, which is CPU.

        Returns
        -------
        numpy array
            Tuning function fr.

        """
        if device == 'CPU' or device is None:
            return C*np.exp(self.grid.AD_d.dot(Fr))
        else:
            return C*self.cuda.cp.exp(self.grid.cuda_var.AD_d.dot(Fr))

    def FeatureComplexity(self, model=None, model_num=0, pde_solver=None,
                          device='CPU'):
        """Calculate feature complexity
        """
        if model is None:
            model = self
        lib = np if device == 'CPU' else self.cuda.cp
        grid = self.grid if device == 'CPU' else self.grid.cuda_var

        peq, p0, D, _ = model.get_params(model_num, device)
        if model.non_equilibrium:
            if pde_solver is None:
                raise ValueError(
                    'pde_solver have to be initialized to compute feature '
                    'complexity of non-equilibrium model'
                )
            Seq = lib.sum(p0 * lib.log(p0) * grid.w_d) + lib.log(2)
            Force = model.force_from_peq(peq, device)
            Fs = D / 4 * Force ** 2
            lQ, _, Qx = pde_solver.solve_EV(
                peq, D, None, peq, 'h0', None, device
            )
            Fd = Qx.T.dot(lib.diag(grid.w_d)).dot(Fs * lib.sqrt(peq))
            rho0d = Qx.T.dot(lib.diag(grid.w_d)).dot(p0 / lib.sqrt(peq))
            S2 = lib.sum(Fd * rho0d / lQ)
            out = Seq + S2
        else:
            result = (grid.dmat_d.dot(lib.sqrt(peq))) ** 2
            out = lib.sum(4 * result * grid.w_d)
        return out

    def FeatureComplexityFderiv(self, peq, device='CPU'):
        """Calculate variational derivative of equilirium model complexity
        w.r.t. force F. Used in Genkin, Engel 2020 for regularization.
        """
        if self.non_equilibrium:
            raise ValueError(
                'This feature not supported for non-equilibrium model'
            )
        if device == 'CPU':
            term1 = -4 * (self.grid.dmat_d.dot(np.sqrt(peq)))**2
            term2 = 2 * self.grid.dmat_d.dot(self.grid.dmat_d.dot(peq))
        else:
            term1 = -4 * \
                (self.grid.cuda_var.dmat_d.dot(self.cuda.cp.sqrt(peq)))**2
            term2 = 2 * \
                self.grid.cuda_var.dmat_d.dot(
                    self.grid.cuda_var.dmat_d.dot(peq))
        term3 = self.FeatureComplexity(peq, device) * peq
        return self.grid.Integrate(term1+term2+term3, device=device)

    @staticmethod
    def _check_inputs(peq, p0, D, fr, params_size, grid):

        # Check fr
        if fr.min() < 0:
            raise ValueError('Negative firing rate function')

        # check peq
        if len(peq.shape) > 1:
            for model_num in range(peq.shape[0]):
                if np.abs(sum(peq[model_num, :]*grid.w_d)-1) > 10**-5:
                    raise ValueError(
                        f'For model_num {model_num} peq is not normalized'
                    )
        else:
            if np.abs(sum(peq*grid.w_d)-1) > 10**-5:
                raise ValueError('Peq is not normalized')

        # check p0
        if p0 is not None:
            if len(p0.shape) > 1:
                for model_num in range(p0.shape[0]):
                    if np.abs(sum(p0[model_num, :]*grid.w_d)-1) > 10**-5:
                        raise ValueError(
                            f'For model_num {model_num} p0 is not normalized'
                        )
            else:
                if np.abs(sum(p0*grid.w_d)-1) > 10**-5:
                    raise ValueError('P0 is not normalized')

        # check D
        if (isinstance(D, numbers.Number) and D < 10**-8) or min(D) < 10**-8:
            raise ValueError('D is smaller than 10^-8')

        # Params size
        if len(peq.shape) == 1:
            peq_size = 1
        else:
            peq_size = peq.shape[0]
        if peq_size > params_size['peq']:
            raise ValueError('Provided peq contains more datasamples than '
                             'expected in params_size')
        if p0 is not None:
            if len(p0.shape) == 1:
                p0_size = 1
            else:
                p0_size = p0.shape[0]
            if p0_size > params_size['p0']:
                raise ValueError('Provided p0 contains more datasamples than '
                                 'expected in params_size')
        if len(fr.shape) == 2:
            fr_size = 1
        else:
            fr_size = fr.shape[0]
        if fr_size > params_size['fr']:
            raise ValueError('Provided fr contains more models than '
                             'expected in params_size')
        if D.shape[0] > params_size['D']:
            raise ValueError('Provided D contains more params than '
                             'expected in params_size')
