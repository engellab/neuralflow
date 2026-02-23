#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Langevin model parameters and related functions
"""

import numpy as np
from copy import deepcopy
import numbers
from neuralflow.peq_models import peq_model_types_
from neuralflow.firing_rate_models import firing_model_types_
import logging
from neuralflow.settings import MINIMUM_PEQ

logger = logging.getLogger(__name__)


def _convert_params_size_to_condition_mapping(params_size, num_models, num_neuron):
    """Convert old params_size format to new condition_mapping format.
    
    This function provides backward compatibility with the old params_size API.
    
    Parameters
    ----------
    params_size : dict
        Old format dictionary with keys 'peq', 'p0', 'D', 'fr' specifying
        the number of distinct parameter sets for each.
    num_models : int
        Number of conditions/models.
    num_neuron : int
        Number of neurons.
    
    Returns
    -------
    dict
        New condition_mapping format.
    """
    return {
        'num_conditions': num_models,
        'num_peq': params_size['peq'],
        'num_p0': params_size['p0'],
        'num_D': params_size['D'],
        'num_fr': params_size['fr'] * num_neuron,
        'peq_indices': [min(i, params_size['peq']-1) for i in range(num_models)],
        'p0_indices': [min(i, params_size['p0']-1) for i in range(num_models)],
        'D_indices': [min(i, params_size['D']-1) for i in range(num_models)],
        'fr_indices': [
            [j * params_size['fr'] + min(i, params_size['fr']-1) 
             for j in range(num_neuron)]
            for i in range(num_models)
        ]
    }


def _validate_condition_mapping(condition_mapping, peq_shape, p0_shape, D_shape, fr_shape):
    """Validate condition_mapping dictionary.
    
    Parameters
    ----------
    condition_mapping : dict
        The condition mapping to validate.
    peq_shape : tuple
        Shape of peq array.
    p0_shape : tuple
        Shape of p0 array.
    D_shape : tuple
        Shape of D array.
    fr_shape : tuple
        Shape of fr array.
    
    Raises
    ------
    ValueError
        If condition_mapping is invalid.
    """
    required_keys = ['num_conditions', 'num_peq', 'num_p0', 'num_D', 'num_fr',
                     'peq_indices', 'p0_indices', 'D_indices', 'fr_indices']
    
    for key in required_keys:
        if key not in condition_mapping:
            raise ValueError(f"condition_mapping missing required key: {key}")
    
    num_cond = condition_mapping['num_conditions']
    
    # Check indices lengths
    for key in ['peq_indices', 'p0_indices', 'D_indices', 'fr_indices']:
        if len(condition_mapping[key]) != num_cond:
            raise ValueError(f"{key} length ({len(condition_mapping[key])}) must equal num_conditions ({num_cond})")
    
    # Check indices bounds
    for i, idx in enumerate(condition_mapping['peq_indices']):
        if idx < 0 or idx >= condition_mapping['num_peq']:
            raise ValueError(f"peq_indices[{i}] = {idx} is out of bounds [0, {condition_mapping['num_peq']-1}]")
    
    for i, idx in enumerate(condition_mapping['p0_indices']):
        if idx < 0 or idx >= condition_mapping['num_p0']:
            raise ValueError(f"p0_indices[{i}] = {idx} is out of bounds [0, {condition_mapping['num_p0']-1}]")
    
    for i, idx in enumerate(condition_mapping['D_indices']):
        if idx < 0 or idx >= condition_mapping['num_D']:
            raise ValueError(f"D_indices[{i}] = {idx} is out of bounds [0, {condition_mapping['num_D']-1}]")
    
    for cond_idx, fr_idx_list in enumerate(condition_mapping['fr_indices']):
        for fr_idx in fr_idx_list:
            if fr_idx < 0 or fr_idx >= condition_mapping['num_fr']:
                raise ValueError(f"fr_indices[{cond_idx}] contains {fr_idx} which is out of bounds [0, {condition_mapping['num_fr']-1}]")
    
    # Check array sizes match
    if peq_shape[0] != condition_mapping['num_peq']:
        raise ValueError(f"peq first dimension ({peq_shape[0]}) != num_peq ({condition_mapping['num_peq']})")
    if p0_shape[0] != condition_mapping['num_p0']:
        raise ValueError(f"p0 first dimension ({p0_shape[0]}) != num_p0 ({condition_mapping['num_p0']})")
    if D_shape[0] != condition_mapping['num_D']:
        raise ValueError(f"D first dimension ({D_shape[0]}) != num_D ({condition_mapping['num_D']})")
    if fr_shape[0] != condition_mapping['num_fr']:
        raise ValueError(f"fr first dimension ({fr_shape[0]}) != num_fr ({condition_mapping['num_fr']})")


class model:
    """Model class supports one or multiple Langevin dynamics defined by
    peq(x), p0(x), D, fr(x). 
    
    The new condition-based mapping system allows flexible assignment of
    parameters to experimental conditions. Each condition can have:
    - One peq model (equilibrium distribution)
    - One p0 model (initial distribution) 
    - One D value (diffusion coefficient)
    - Multiple fr models (firing rates)
    
    Parameters can be shared across conditions by specifying the same index
    in the condition_mapping.

    Parameters
    ----------
    peq : numpy array, (num_peq, grid.N)
        peq function evaluated on a grid. Each row is a distinct peq model.
    p0 : numpy array, (num_p0, grid.N) or None
        p0 function evaluated on a grid. If set to None, assume equilibrium
        model (p0 = peq).
    D : numpy array, (num_D,)
        Diffusion coefficient values.
    fr : numpy array, (num_fr, grid.N)
        Firing rate functions. Each row is a firing rate for one neuron.
    condition_mapping : dict
        Dictionary specifying how parameters map to conditions:
            num_conditions : int
                Number of experimental conditions.
            num_peq, num_p0, num_D, num_fr : int
                Number of distinct parameter models for each type.
            peq_indices, p0_indices, D_indices : list of int
                Index of the parameter model to use for each condition.
            fr_indices : list of list of int
                For each condition, list of fr model indices for each neuron.
    grid : grid.GLLgrid object
        Initialized GLL grid. Should be initialized with with_cuda=True
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
        
    Examples
    --------
    Create a model with 2 conditions sharing peq but with different neurons:
    
    >>> condition_mapping = {
    ...     'num_conditions': 2,
    ...     'num_peq': 1, 'num_p0': 2, 'num_D': 2, 'num_fr': 15,
    ...     'peq_indices': [0, 0],
    ...     'p0_indices': [0, 1],
    ...     'D_indices': [0, 1],
    ...     'fr_indices': [
    ...         [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],      # condition 0: 10 neurons
    ...         [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]  # condition 1: 11 neurons
    ...     ]
    ... }
    >>> model = Model(peq, p0, D, fr, condition_mapping, grid)
    """

    # Default initialization is from predefined models
    def __init__(self, peq, p0, D, fr, condition_mapping, grid,
                 peq_model=None, p0_model=None, fr_model=None,
                 with_cuda=False):
        """
        Public methods
        ------
        new_model, get_params, get_params_for_grad, get_fr_lambda, sync_model,
        compute_rho0, peq_from_force, force_from_peq, Fr_from_fr, fr_from_Fr,
        FeatureComplexity, FeatureComplexityFderiv, get_num_neurons

        """
        self.grid = grid
        self.peq_model = peq_model
        self.p0_model = p0_model
        self.fr_model = fr_model

        # Equilibrium model: p0 = peq. Non-equilibrium model: p0 can differ
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
        self.peq = np.maximum(self.peq, MINIMUM_PEQ).astype('float64')
        
        if len(p0.shape) == 1:
            self.p0 = p0[np.newaxis, :]
        else:
            self.p0 = p0
            
        # fr shape: (num_fr, grid.N) - each row is one neuron's firing rate
        if len(fr.shape) == 1:
            self.fr = fr[np.newaxis, :]
        else:
            self.fr = fr

        self.fr=self.fr.astype('float64')
        self.D = D.astype('float64')
        
        # Store condition_mapping
        self.condition_mapping = deepcopy(condition_mapping)
        
        # Compute rho0 for each unique (peq, p0) combination
        self._compute_rho0()
        
        # For backward compatibility, also store params_size
        self._update_params_size_from_mapping()
        
        # Get number of neurons per condition
        self._num_neurons_per_condition = [
            len(fr_idx) for fr_idx in condition_mapping['fr_indices']
        ]
        
        # Maximum number of neurons across all conditions (for backward compat)
        # self.num_neuron = max(self._num_neurons_per_condition)
        self.num_neuron = max([
            max(fr_idx) for fr_idx in condition_mapping['fr_indices']
            ]) + 1
        self.num_models = condition_mapping['num_conditions']

        # Validate inputs
        self._check_inputs(self.peq, self.p0 if p0 is not None else None, 
                          self.D, self.fr, self.condition_mapping, self.grid)
        
        # Compute C - needed for optimization. C is the value of fr at x = xbegin
        # Now C has shape (num_fr,)
        self.C = self.fr[:, 0]

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
    
    def _update_params_size_from_mapping(self):
        """Update params_size from condition_mapping for backward compatibility."""
        cm = self.condition_mapping
        self.params_size = {
            'peq': cm['num_peq'],
            'p0': cm['num_p0'],
            'D': cm['num_D'],
            'fr': cm['num_fr'],
            'rho0': cm['num_rho0']
        }

    def _compute_rho0(self):
        """Compute rho0 for all unique (peq, p0) combinations.
        
        rho0 = p0 / sqrt(peq), so the number of rho0 models equals the
        number of unique (peq_idx, p0_idx) combinations.
        
        For example:
        - peq_indices = [0, 1, 0, 1]
        - p0_indices = [1, 0, 0, 1]
        - Unique combinations: (0,1), (1,0), (0,0), (1,1) = 4 rho0 models
        """
        cm = self.condition_mapping
        
        # Find all unique (peq_idx, p0_idx) pairs and assign each a unique index
        pair_to_rho0_idx = {}
        rho0_indices = []
        
        for cond_idx in range(cm['num_conditions']):
            peq_idx = cm['peq_indices'][cond_idx]
            p0_idx = cm['p0_indices'][cond_idx]
            pair = (peq_idx, p0_idx)
            
            if pair not in pair_to_rho0_idx:
                # Assign new rho0 index for this unique pair
                pair_to_rho0_idx[pair] = len(pair_to_rho0_idx)
            
            rho0_indices.append(pair_to_rho0_idx[pair])
        
        # Number of unique rho0 models
        num_rho0 = len(pair_to_rho0_idx)
        cm['num_rho0'] = num_rho0
        cm['rho0_indices'] = rho0_indices
        
        # Create rho0 array and compute for each unique pair
        self.rho0 = np.zeros((num_rho0, self.peq.shape[1]), dtype='float64')
        
        self.pair_to_rho0_idx = pair_to_rho0_idx
        for (peq_idx, p0_idx), rho0_idx in pair_to_rho0_idx.items():
            self.rho0[rho0_idx, :] = (
                self.p0[p0_idx, :] /
                np.maximum(np.sqrt(self.peq[peq_idx, :]), 10**-20)
            )

    def __deepcopy__(self, memo):
        """When with_cuda = True, self.cuda should be shared across instances"""
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
                  condition_mapping=None, params_size=None, with_cuda=False):
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
            Diffusion coefficient. Either an array that contains a value for
            each of the models, or a float.
        fr_model : list or np.array
            Firing rate functions. Either a list of callables/dict (one entry
            for each neuron), or np.array.
        grid : grid.GLLgrid object
            Initialized GLL grid. Should be initialized with with_cuda=True
            for GPU support.
        condition_mapping : dict, optional
            New format for specifying parameter-to-condition mapping. See class
            docstring for details.
        params_size : dict, optional
            **Deprecated** Old format for specifying parameter sharing. Use
            condition_mapping instead.
        with_cuda : bool, optional
           Whether to include GPU support. For GPU optimization, the platform
           has to be cuda-enabled, and cupy package has to be installed. The
           default is False.

        Returns
        -------
        model
            Initialized model.

        """

        # Handle peq_model - can be callable, dict, list of dicts, or array
        if callable(peq_model):
            # Initialize by custom function
            peq = peq_model(grid.x_d, grid.w_d)
            peq_model_ = peq_model
        elif isinstance(peq_model, dict):
            # Initialize by a single dict (one peq model)
            if peq_model['model'] in peq_model_types_:
                if 'params' not in peq_model.keys():
                    peq_model['params'] = {}
                peq_model_ = peq_model
                peq = peq_model_types_[peq_model['model']](
                    grid.x_d, grid.w_d, **peq_model['params'])
            else:
                raise ValueError(f'Unknown model {peq_model}')
        elif isinstance(peq_model, list):
            # Initialize by a list of dicts (multiple peq models)
            peq_list = []
            peq_model_ = []
            for i, pm in enumerate(peq_model):
                if callable(pm):
                    peq_list.append(pm(grid.x_d, grid.w_d))
                    peq_model_.append(pm)
                elif isinstance(pm, dict):
                    if pm['model'] in peq_model_types_:
                        if 'params' not in pm.keys():
                            pm['params'] = {}
                        peq_list.append(peq_model_types_[pm['model']](
                            grid.x_d, grid.w_d, **pm['params']))
                        peq_model_.append(pm)
                    else:
                        raise ValueError(f'Unknown peq model at index {i}: {pm}')
                else:
                    raise ValueError(f'Invalid peq_model at index {i}: must be dict or callable')
            peq = np.vstack(peq_list)  # shape: (num_peq, N)
        else:
            # peq provided directly as array. Make sure to copy it
            peq = np.array(peq_model)
            peq_model_ = None

        # Handle p0_model - can be callable, dict, list of dicts, array, or None
        if callable(p0_model):
            p0 = p0_model(grid.x_d, grid.w_d)
            p0_model_ = p0_model
        elif isinstance(p0_model, dict):
            # Initialize by a single dict (one p0 model)
            if p0_model['model'] in peq_model_types_:
                if 'params' not in p0_model.keys():
                    p0_model['params'] = {}
                p0_model_ = p0_model
                p0 = peq_model_types_[p0_model['model']](
                    grid.x_d, grid.w_d, **p0_model['params'])
            else:
                raise ValueError(f'Unknown model {p0_model}')
        elif isinstance(p0_model, list):
            # Initialize by a list of dicts (multiple p0 models)
            p0_list = []
            p0_model_ = []
            for i, pm in enumerate(p0_model):
                if callable(pm):
                    p0_list.append(pm(grid.x_d, grid.w_d))
                    p0_model_.append(pm)
                elif isinstance(pm, dict):
                    if pm['model'] in peq_model_types_:
                        if 'params' not in pm.keys():
                            pm['params'] = {}
                        p0_list.append(peq_model_types_[pm['model']](
                            grid.x_d, grid.w_d, **pm['params']))
                        p0_model_.append(pm)
                    else:
                        raise ValueError(f'Unknown p0 model at index {i}: {pm}')
                else:
                    raise ValueError(f'Invalid p0_model at index {i}: must be dict or callable')
            p0 = np.vstack(p0_list)  # shape: (num_p0, N)
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

        # Expand dimensions if needed
        if len(peq.shape) == 1:
            peq = peq[np.newaxis, :]
        if p0 is not None and len(p0.shape) == 1:
            p0 = p0[np.newaxis, :]
        # Transpose fr to (num_fr, grid.N) format
        if len(fr.shape) == 2 and fr.shape[0] == grid.N:
            fr = fr.T  # Now shape is (num_neuron, grid.N)
        
        # Handle condition_mapping vs params_size
        if condition_mapping is None and params_size is None:
            # Default: all conditions have same number of neurons
            num_neuron = fr.shape[0]
            condition_mapping = {
                'num_conditions': 1,
                'num_peq': peq.shape[0],
                'num_p0': p0.shape[0] if p0 is not None else 1,
                'num_D': D.size,
                'num_fr': num_neuron,
                'peq_indices': [0],
                'p0_indices': [0],
                'D_indices': [0],
                'fr_indices': [list(range(num_neuron))]
            }
        elif condition_mapping is None and params_size is not None:
            # Convert old params_size to new condition_mapping
            num_neuron = fr.shape[0]
            num_models = max(params_size.values())
            condition_mapping = _convert_params_size_to_condition_mapping(
                params_size, num_models, num_neuron
            )
        
        condition_mapping = deepcopy(condition_mapping)
        grid = deepcopy(grid)
        
        model._check_inputs(peq, p0, D, fr, condition_mapping, grid)
        return cls(peq, p0, D, fr, condition_mapping, grid, peq_model_, p0_model_,
                   fr_model_, with_cuda)

    def get_num_neurons(self, condition_num):
        """Get the number of neurons for a specific condition.
        
        Parameters
        ----------
        condition_num : int
            Condition index.
            
        Returns
        -------
        int
            Number of neurons for this condition.
        """
        return self._num_neurons_per_condition[condition_num]

    def get_params(self, condition_num, device='CPU'):
        """Extract peq, p0, D, fr for a specific condition.
        
        Parameters
        ----------
        condition_num : int
            Condition index.
        device : str, optional
            'CPU' or 'GPU'. Default is 'CPU'.
            
        Returns
        -------
        peq : array
            peq for this condition, shape (grid.N,).
        p0 : array
            p0 for this condition, shape (grid.N,).
        D : float
            Diffusion coefficient for this condition.
        fr : array
            Firing rates for this condition, shape (grid.N, num_neurons).
        """
        if device == 'CPU':
            var = self
        else:
            var = self.cuda_var
        
        cm = self.condition_mapping
        
        peq_idx = cm['peq_indices'][condition_num]
        peq = var.peq[peq_idx, ...]
        
        p0_idx = cm['p0_indices'][condition_num]
        p0 = var.p0[p0_idx, ...]
        
        D_idx = cm['D_indices'][condition_num]
        D = var.D[D_idx]
        
        fr_indices = cm['fr_indices'][condition_num]
        # fr is stored as (num_fr, grid.N), extract and transpose to (grid.N, num_neurons)
        fr = var.fr[fr_indices, ...].T  # shape: (grid.N, num_neurons)
        
        return peq, p0, D, fr

    def get_params_for_grad(self, condition_num, device='CPU'):
        """Extract peq, rho0, D, fr for gradient calculation.
        
        Parameters
        ----------
        condition_num : int
            Condition index.
        device : str, optional
            'CPU' or 'GPU'. Default is 'CPU'.
            
        Returns
        -------
        peq : array
            peq for this condition, shape (grid.N,).
        rho0 : array
            rho0 for this condition, shape (grid.N,).
        D : float
            Diffusion coefficient for this condition.
        fr : array
            Firing rates for this condition, shape (grid.N, num_neurons).
        """
        if device == 'CPU':
            var = self
        else:
            var = self.cuda_var
        
        cm = self.condition_mapping
        
        peq_idx = cm['peq_indices'][condition_num]
        peq = var.peq[peq_idx, ...]
        
        rho0_idx = cm['rho0_indices'][condition_num]
        rho0 = var.rho0[rho0_idx, ...]
        
        D_idx = cm['D_indices'][condition_num]
        D = var.D[D_idx]
        
        fr_indices = cm['fr_indices'][condition_num]
        fr = var.fr[fr_indices, ...].T  # shape: (grid.N, num_neurons)
        
        return peq, rho0, D, fr
    
    def get_param_indices(self, condition_num):
        """Get parameter indices for a specific condition.
        
        Parameters
        ----------
        condition_num : int
            Condition index.
            
        Returns
        -------
        dict
            Dictionary with peq_idx, p0_idx, D_idx, fr_indices, rho0_idx.
        """
        cm = self.condition_mapping
        return {
            'peq_idx': cm['peq_indices'][condition_num],
            'p0_idx': cm['p0_indices'][condition_num],
            'D_idx': cm['D_indices'][condition_num],
            'fr_indices': cm['fr_indices'][condition_num],
            'rho0_idx': cm['rho0_indices'][condition_num]
        }

    def get_fr_lambda(self):
        """Extract firing rate function"""
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
        """Sync GPU and CPU by copying a model over."""
        if not self.with_cuda:
            logger.warning('Model does not support GPU. Aborting')
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
            "CPU" or "GPU". The default is "CPU".
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
            Fr represented on the grid.
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

    def FeatureComplexity(self, model=None, condition_num=0, pde_solver=None,
                          device='CPU'):
        """Calculate feature complexity
        
        Parameters
        ----------
        model : model, optional
            Model to use. Default is self.
        condition_num : int, optional
            Condition number. Default is 0.
        pde_solver : PDESolve, optional
            PDE solver for non-equilibrium models.
        device : str, optional
            'CPU' or 'GPU'. Default is 'CPU'.
            
        Returns
        -------
        float
            Feature complexity value.
        """
        if model is None:
            model = self
        lib = np if device == 'CPU' else self.cuda.cp
        grid = self.grid if device == 'CPU' else self.grid.cuda_var

        peq, p0, D, _ = model.get_params(condition_num, device)
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
        """Calculate variational derivative of equilibrium model complexity
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
        term3 = self.FeatureComplexity(peq=peq, device=device) * peq
        return self.grid.Integrate(term1+term2+term3, device=device)

    @staticmethod
    def _check_inputs(peq, p0, D, fr, condition_mapping, grid):
        """Validate inputs for model initialization."""
        
        cm = condition_mapping
        num_cond = cm['num_conditions']
        
        # Check fr
        if fr.min() < 0:
            raise ValueError('Negative firing rate function')

        # Check peq normalization
        for peq_idx in range(cm['num_peq']):
            if np.abs(np.sum(peq[peq_idx, :]*grid.w_d) - 1) > 10**-5:
                raise ValueError(f'peq[{peq_idx}] is not normalized')

        # Check p0 normalization
        if p0 is not None:
            for p0_idx in range(cm['num_p0']):
                if np.abs(np.sum(p0[p0_idx, :]*grid.w_d) - 1) > 10**-5:
                    raise ValueError(f'p0[{p0_idx}] is not normalized')

        # Check D
        if (isinstance(D, numbers.Number) and D < 10**-8) or np.min(D) < 10**-8:
            raise ValueError('D is smaller than 10^-8')

        # Validate condition_mapping structure
        _validate_condition_mapping(condition_mapping, peq.shape, 
                                     p0.shape if p0 is not None else (cm['num_p0'],),
                                     D.shape, fr.shape)

    # Backward compatibility: support old model_num parameter name
    def get_params_old(self, model_num, device='CPU'):
        """Backward compatible get_params using model_num instead of condition_num."""
        return self.get_params(model_num, device)
    
    def get_params_for_grad_old(self, model_num, device='CPU'):
        """Backward compatible get_params_for_grad using model_num."""
        return self.get_params_for_grad(model_num, device)