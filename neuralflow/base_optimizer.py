#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
optimizer class contains all of the parameters needed for optimization
Classes defined in this script: optimizer (parent), ADAM(child), GD(child)
"""

from neuralflow.spike_data import SpikeData
from neuralflow.settings import (
    opt_settings, line_search_setting, implemented_bms
)
from neuralflow.gradients import Grads
import logging
import math
import numpy as np
import os

logger = logging.getLogger(__name__)


class optimizer():

    def __init__(
            self, dataTR, init_model, opt_options,
            line_search_options={}, pde_solve_params={},
            boundary_mode='absorbing', save_options={}, dataCV=None,
            device='CPU'
    ):

        # Number of datasamples will be inferred from dataTR and dataCV
        self.num_datasamples = None
        self.device = device
        if device == 'GPU':
            import neuralflow.base_cuda as cuda
            self.cuda = cuda

        # Check data
        self.dataTR = self._check_optimizer_data(dataTR)
        if dataCV is not None:
            dataCV = self._check_optimizer_data(dataCV)
        self.dataCV = dataCV

        # Check model
        if init_model.num_models != self.num_datasamples:
            raise ValueError(
                f'Number of models ({init_model.num_models}) differs from the '
                f'number of datasamples ({self.num_datasamples()}).'
            )
        if device == 'GPU' and not init_model.with_cuda:
            raise ValueError(
                'For GPU optimization init_model has to be initialized with '
                'GPU support'
            )
        self.model = init_model

        # check that number of neurons is consistent in data and model
        num_neuron = init_model.num_neuron
        for ids, ds in enumerate(self.dataTR):
            if ds.num_neuron != num_neuron:
                raise ValueError(
                    f'Init model contains {num_neuron} neurons, while training'
                    f' data sample {ids} contains {ds.num_neuron} neurons'
                )
        if dataCV is not None:
            for ids, ds in enumerate(self.dataCV):
                if ds.num_neuron != num_neuron:
                    raise ValueError(
                        f'Init model contains {num_neuron} neurons, while val'
                        f' data sample {ids} contains {ds.num_neuron} neurons'
                    )

        # Optimization options
        optimizer._check_common_optim_options(opt_options)
        self.opt_options = opt_options

        # Line search options for C and D: how often to do the line search
        optimizer._check_line_search_options(line_search_options)
        self.line_search_options = line_search_options

        # For the gradient computation
        self.pde_solve_params = pde_solve_params
        for param in ['xbegin', 'xend', 'Np', 'Ne']:
            if param not in self.pde_solve_params.keys():
                self.pde_solve_params[param] = getattr(self.model.grid, param)

        self.boundary_mode = boundary_mode
        if boundary_mode not in implemented_bms:
            raise ValueError('Unknown boundary mode')
        self.gradient = Grads(
            pde_solve_params, boundary_mode, opt_options['params_to_opt'],
            self.dataTR[0].num_neuron, self.dataTR[0].with_trial_end, device
        )

        # Save options
        optimizer._check_save_options(save_options, opt_options['max_epochs'])
        self.save_options = save_options

        # Find shared params, which are the same across all datasamples
        self.opt_model_map = {
            'F': 'peq', 'F0': 'p0', 'D': 'D', 'Fr': 'fr', 'C': 'fr'
        }
        self.shared_params = [
            k for k in opt_options['params_to_opt']
            if self.model.params_size[self.opt_model_map[k]] == 1
        ]

        # Number of trials in each datasample
        self.num_trial = [
            len(self.get_dataTR(samp)) for samp in range(self.num_datasamples)
        ]

        # Size of minibatch for each sample
        self.mini_batch_size = [
            math.ceil(tr_size / opt_options['mini_batch_number'])
            for tr_size in self.num_trial
        ]

        # Number of iterations in epoch for each datasample
        self.iter_in_epoch = [
            math.ceil(tr_size/mbatch_size) for tr_size, mbatch_size
            in zip(self.num_trial, self.mini_batch_size)
        ]

        # Last batch may contain less trials if num_trials is not divisable by
        # the mini batch size
        self.last_batch_size = [
            tr_size % mbatch_size if tr_size % mbatch_size > 0 else mbatch_size
            for tr_size, mbatch_size
            in zip(self.num_trial, self.mini_batch_size)
        ]

        # Schedule for line search of C and D
        all_iterations = np.sum(self.iter_in_epoch)
        for par in ['C', 'D']:
            n_per_epoch = line_search_options[f'{par}_opt']['nSearchPerEpoch']
            if par in self.shared_params:
                line_search_options[f'{par}_opt']['iter_schedule'] = [
                    np.arange(
                        0, all_iterations,
                        math.ceil(all_iterations / n_per_epoch)
                    )
                ]
            else:
                line_search_options[f'{par}_opt']['iter_schedule'] = [
                    np.arange(
                        0, iter_in_ep, math.ceil(iter_in_ep / n_per_epoch)
                    )
                    for iter_in_ep in self.iter_in_epoch
                ]

    def get_dataTR(self, nsample):
        """Return nsample sample of training data
        """
        if self.device == 'CPU':
            return self.dataTR[nsample].data
        return self.dataTR[nsample].cuda_var.data

    def get_dataCV(self, nsample):
        """Return nsample sample of validation data
        """
        if self.device == 'CPU':
            return self.dataCV[nsample].data
        return self.dataCV[nsample].cuda_var.data

    def _check_optimizer_data(self, data):
        if type(data) is not list:
            logger.warning('Wrapping up data into the list format')
            data = [data]

        if self.num_datasamples is None:
            self.num_datasamples = len(data)
        if self.num_datasamples != len(data):
            raise ValueError(
                'Training and val data have different number of datasamples'
            )
        for ds in data:
            if type(ds) is not SpikeData:
                raise ValueError(
                    'Each datasample must be an instance of SpikeData class'
                )
            if ds.dformat != 'ISIs':
                logger.warning('Converting data into ISI format')
                ds.change_format('ISIs')
            if self.device == 'GPU' and not hasattr(ds, 'cuda_var'):
                logger.info('Moving data to GPU')
                ds.to_GPU()
        return data

    @staticmethod
    def _check_common_optim_options(opt_options):
        # max epochs
        if 'max_epochs' not in opt_options.keys():
            def_val = opt_settings['max_epochs']
            logger.warning(f'Setting max_epochs to {def_val}')
            opt_options['max_epochs'] = def_val
        elif opt_options['max_epochs'] < 1:
            raise ValueError('max_epochs has to be greater than 0')

        # mini batch number
        if 'mini_batch_number' not in opt_options.keys():
            def_val = opt_settings['mini_batch_number']
            logger.warning(f'Setting mini_batch_number to {def_val}')
            opt_options['mini_batch_number'] = def_val
        elif opt_options['mini_batch_number'] < 1:
            raise ValueError('mini_batch_number has to be greater than 0')

        # params to opt
        if 'params_to_opt' not in opt_options.keys():
            def_val = opt_settings['params_to_opt']
            logger.warning(f'Setting params_to_opt to {def_val}')
            opt_options['params_to_opt'] = def_val
        else:
            if not all([
                    el in opt_settings['params_to_opt']
                    for el in opt_options['params_to_opt']
            ]):
                raise ValueError(
                    'params_opt can only include '
                    f'{opt_settings["params_to_opt"]}'
                )

        # etaf - regularization strength
        if 'etaf' not in opt_options.keys():
            opt_options['etaf'] = 0

    @staticmethod
    def _check_line_search_options(line_search_options):
        if type(line_search_options) is not dict:
            raise ValueError('line_search_options must be a dict')
        for param in ['C_opt', 'D_opt']:
            if param not in line_search_options.keys():
                line_search_options[param] = {}
            for el in line_search_setting.keys():
                if el not in line_search_options[param].keys():
                    line_search_options[param][el] = line_search_setting[el]
            if line_search_options[param]['max_fun_eval'] < 1:
                raise ValueError('max_fun_eval must be greater than 0')
            if line_search_options[param]['nSearchPerEpoch'] < 1:
                raise ValueError('nSearchPerEpoch must be greater than 0')
            line_search_options[param]['epoch_schedule'] = (
                optimizer._check_schedule(
                    line_search_options[param]['epoch_schedule']
                )
            )

    @staticmethod
    def _check_schedule(schedule):
        schedule = np.array(schedule).astype(int)
        if not all(schedule >= 0):
            raise ValueError('schedule contains negative entries')
        schedule.sort()
        # Remove duplicates
        if schedule.size > 0:
            schedule = schedule[
                np.insert(np.diff(schedule).astype(bool), 0, True)
            ]
        return schedule

    @staticmethod
    def _check_save_options(save_options, max_epochs):
        if 'path' not in save_options or save_options['path'] is None:
            logger.info(
                'Path is not specified. The results will only be saved in RAM'
            )
            save_options['path'] = None
        elif not os.path.exists(save_options['path']):
            raise FileNotFoundError(
                f'Location {save_options["path"]} does not exist'
            )

        if 'stride' not in save_options or save_options['stride'] is None:
            if save_options['path'] is not None:
                logger.info(
                    'Save stride not specified. No itermediate results will be'
                    'saved to persistance storage'
                )
            save_options['stride'] = None

        if (
                'sim_start' not in save_options or
                save_options['sim_start'] is None
                ):
            logger.info('sim_start not provided, setting to zero')
            save_options['sim_start'] = 0

        if 'schedule' in save_options and save_options['schedule'] is not None:
            optimizer._check_schedule(save_options['schedule'])
            if save_options['schedule'][0] < save_options['sim_start']:
                raise ValueError(
                    'Saving schedule should not contain epochs numbers less '
                    'than sim_start'
                )
            if save_options['schedule'][-1] > max_epochs:
                raise ValueError(
                    'Saving schedule should not contain epochs numbers greater'
                    ' than max_epochs'
                )
            if save_options['schedule'][0] > save_options['sim_start']:
                logger.info('Setting the first entry in schedule to sim_start')
                save_options['schedule'] = np.insert(
                    save_options['schedule'], 0, save_options['sim_start']
                )
            elif save_options['schedule'][-1] < (
                    save_options['sim_start'] + max_epochs
            ):
                logger.info('Setting the last entry in schedule to max_epoch')
                save_options['schedule'] = np.insert(
                    save_options['schedule'],
                    len(save_options['schedule']),
                    save_options['sim_start'] + max_epochs
                )
        else:
            logger.info('Setting saving schedule to default: save every epoch')
            start = save_options['sim_start']
            stop = save_options['sim_start'] + max_epochs + 1
            save_options['schedule'] = np.arange(start, stop).astype(int)


class adam_opt(optimizer):

    """ADAM optimizer
    """

    @classmethod
    def initialize(
            cls, dataTR, init_model, opt_options, line_search_options,
            pde_solve_params={}, boundary_mode='absorbing',
            save_options={}, dataCV=None, device='CPU'
    ):

        """Initialize ADAM optimizer

        Parameters
        ----------
        dataTR : list
            List that contains neuralflow.spike_data.SpikeData object for each
            datasample in the data. For each datasample a separate model will
            be trained, however, it is allowed to have shared parameters across
            datasamples (e.g. allow different potentials but the same D, p0,
            and fr). Usually the number of datasamples is the number of
            conditions in the experimental data.
        init_model : neuralflow.model.model
            A model object. model.num_models should be equal to number of data
            samples.
        opt_options : dict
            Optimization options dictionary:
                learning_rate : dict
                    ADAM learning rates:
                        alpha : the main ADAM learning rate.
                        beta1 : ADAM weight for momentum. The default is 0.9.
                        beta2 : ADAM weight for RMS prop. The default is 0.99.
                        epsilon : ADAM epsilon. The default is 10**-8.
                max_epochs : int
                    Maximum number of optimization epochs. The default is 100.
                mini_batch_number : int
                    Number of minibatches. On each epoch ADAM will update the
                    model by iterating through each of the minibatches. Thus,
                    the number of iterations per each epoch is equal to the
                    product of number of data samples and mini_batch_number.
                    Sometimes we cannot achieve the desired number of
                    minibatches, in which case the actual number of minibatches
                    will be smaller than mini_batch_number.
                params_to_opt : list
                    The parameters that will be optimized. The default is
                    ['F', 'F0', 'D', 'Fr', 'C'].
                etaf : float
                    Regularization strength for potenial. Was used in 2020
                    paper. The default is zero.
        line_search_options : dict
            Line search options dictionary:
                C_opt : dictionary
                    Options for line search of C:
                        max_fun_eval : int
                            Maximum number of function evaluation passed to
                            scipy.optimize.minimize. The default is 3.
                        epoch_schedule : numpy array, dtype = int
                            Epochs on which line search will be performed. The
                            default is np.array([]).
                        nSearchPerEpoch : int
                            Number of line searches per epoch. The default is
                            1.
                D_opt: dictionary.
                    Options for line search of D. Same entries as in C_opt.
        pde_solve_params : dict
            Parameters for solveing FP equation. By default these will be
            inherited from model.grid. Only provide if some unusual option
            is needed, e.g. using some unusual boundary conditions.
            Possible parameters:
                xbegin : float
                    The left boundary of the latent state. The default is -1.
                xend : float
                    The right boundary of the latent state. The default is 1.
                Np : int
                    The degree of Langrange interpolation polynomial, also the
                    number of grid points at each element. The default is 8.
                Ne : int
                    Number of SEM elements. The default is 64.
                BoundCond : dict
                    Boundary conditions. Only include if you want to enforce
                    unusual boundary conditions. Otherwise, use boundary_mode
                    to specify boundary conditions.
                Nv : int, optional
                    Number of retained eigenvalues and eigenvectors of the
                    operator H. If set to None, will be equal to grid.N-2,
                    which is the maximum possible value. If Dirichlet BCs are
                    used, it is stongly recommended to set this to None to
                    avoid spurious high-freq oscillations in the fitted
                    functions. The default is None.
        boundary_mode : str
            Boundary mode, can be either absorbing or reflecting. The default
            is 'absorbing'.
        save_options : dict
            Options for saving the results:
                path : str
                    Local path for saving intermediate results. The default is
                    None, in which case nothing will be saved (the results will
                    only be in RAM).
                stride : int
                    Save intermediate files every stride number of epochs.
                    The default is max_iteration (only save the final file).
                schedule : numpy array, dtype=int
                    1D array with epoch numbers on which the fitted models
                    (peq, D, p0, fr) are saved to RAM. The default is
                    np.arange(1-new_sim, max_iteration+1), which includes all
                    of the epochs.
                sim_start : int
                    Starting epoch number (0 if new simulation, otherwise int >
                    0). The default is 0. Don't change as loading the
                    intermediate results and continuing optimization is not
                    supported yet.
        dataCV : list or None
            List that contains neuralflow.spike_data.SpikeData object for each
            datasample in the data. If not provided, validation loglik will not
            be calculated. The default is None.
        device : str
            Can be 'CPU' or 'GPU'. For GPU optimization, the platform has to be
            cuda-enabled, and cupy package has to be installed. The default is
            'CPU'.

        Returns
        -------
        self
            Initialized optimizer object.
        """

        # Check learning rate parameters for Adam:
        adam_opt._check_optimization_options(opt_options)
        return cls(
            dataTR, init_model, opt_options, line_search_options,
            pde_solve_params, boundary_mode, save_options, dataCV, device
        )

    @staticmethod
    def _check_optimization_options(opt_options):
        if type(opt_options) is not dict:
            raise ValueError('opt_options has to be a dict')
        if 'learning_rate' not in opt_options.keys():
            raise ValueError('Learining rate is a mandatory parameter')
        if 'alpha' not in opt_options['learning_rate'].keys():
            raise ValueError('Learining rate has to include alpha parameter')
        if opt_options['learning_rate']['alpha'] <= 0:
            raise ValueError("Learning rate's alpha must be positive")
        for lr in ['beta1', 'beta2', 'epsilon']:
            if lr not in opt_options['learning_rate'].keys():
                logger.debug(
                    f'Learning rate {lr} not provided. Setting to '
                    f'{opt_settings[lr]}'
                )
                opt_options['learning_rate'][lr] = opt_settings[lr]
            elif opt_options['learning_rate'][lr] < 0:
                raise ValueError(
                    f"{lr} parameter in learning rate must be positive"
                )

    def __repr__(self):
        if self.dataCV is None:
            with_CV = 'without validation data'
        else:
            with_CV = 'with validation data'
        return f'Adam optimizer, {self.boundary_mode} boundary, {with_CV}'

    def _prepare_moments(self):
        """Initialize adam_counter, running average and RMS prop
        """
        self.adam_counter = {
            key: np.zeros(
                self.model.params_size[self.opt_model_map[key]], dtype=int
                ) for key in self.opt_options['params_to_opt']
        }
        self.gradients_av, self.RMS_av = {}, {}
        for key in self.opt_options['params_to_opt']:
            if key == 'C':
                gr_size = rms_size = (
                    self.model.params_size['fr'], self.model.num_neuron
                )
            elif key == 'D':
                gr_size = rms_size = (self.model.params_size['D'],)
            elif key == 'Fr':
                gr_size = (
                    self.model.params_size['fr'],
                    self.model.grid.N,
                    self.model.num_neuron
                )
                rms_size = (
                    self.model.params_size['fr'], self.model.num_neuron
                    )
            else:
                gr_size = (
                    self.model.params_size[self.opt_model_map[key]],
                    self.model.grid.N
                    )
                rms_size = (self.model.params_size[self.opt_model_map[key]],)

            if self.device == 'CPU':
                self.gradients_av[key] = np.zeros(gr_size, dtype='float64')
                self.RMS_av[key] = np.zeros(rms_size, dtype='float64')
            else:
                self.gradients_av[key] = self.cuda.cp.zeros(
                    gr_size, dtype='float64'
                )
                self.RMS_av[key] = self.cuda.cp.zeros(
                    rms_size, dtype='float64'
                )


class gd_opt(optimizer):

    """Gradient-descent optimizer
    """

    @classmethod
    def initialize(
            cls, dataTR, init_model, opt_options, line_search_options,
            pde_solve_params={}, boundary_mode='absorbing',
            save_options={}, dataCV=None, device='CPU'
    ):
        """Initialize GD optimizer


        Parameters
        ----------
        dataTR : list
            List that contains neuralflow.spike_data.SpikeData object for each
            datasample in the data. For each datasample a separate model will
            be trained, however, it is allowed to have shared parameters across
            datasamples (e.g. allow different potentials but the same D, p0,
            and fr). Usually the number of datasamples is the number of
            conditions in the experimental data.
        init_model : neuralflow.model.model
            A model object. model.num_models should be equal to number of data
            samples.
        opt_options : dict
            Optimization options dictionary:
                learning_rate : dict
                    Keys are each of the parameters to be optimized, and values
                    are the learning rates, e.g. {'F': 0.01, 'F0': 0.02}.
                max_epochs : int
                    Maximum number of optimization epochs. The default is 100.
                mini_batch_number : int
                    Number of minibatches. On each epoch ADAM will update the
                    model by iterating through each of the minibatches. Thus,
                    the number of iterations per each epoch is equal to the
                    product of number of data samples and mini_batch_number.
                params_to_opt : list
                    The parameters that will be optimized. The default is
                    ['F', 'F0', 'D', 'Fr', 'C'].
                etaf : float
                    Regularization strength for potenial. The default is zero.
        line_search_options : dict
            Line search options dictionary:
                C_opt : dictionary
                    Options for line search of C:
                        max_fun_eval : int
                            Maximum number of function evaluation passed to
                            scipy.optimize.minimize. The default is 3.
                        epoch_schedule : numpy array, dtype = int
                            Epochs on which line search will be performed. The
                            default is np.array([]).
                        nSearchPerEpoch : int
                            Number of line searches per epoch. The default is
                            1.
                D_opt: dictionary.
                    Options for line search of D. Same entries as in C_opt.
        pde_solve_params : dict
            Parameters for solveing FP equation. By default these will be
            inherited from model.grid. Only provide if some unusual option
            is needed, e.g. using some unusual boundary conditions.
            Possible parameters:
                xbegin : float
                    The left boundary of the latent state. The default is -1.
                xend : float
                    The right boundary of the latent state. The default is 1.
                Np : int
                    The degree of Langrange interpolation polynomial, also the
                    number of grid points at each element. The default is 8.
                Ne : int
                    Number of SEM elements. The default is 64.
                BoundCond : dict
                    Boundary conditions. Only include if you want to enforce
                    unusual boundary conditions. Otherwise, use boundary_mode
                    to specify boundary conditions.
                Nv : int, optional
                    Number of retained eigenvalues and eigenvectors of the
                    operator H. If set to None, will be equal to grid.N-2,
                    which is the maximum possible value. If Dirichlet BCs are
                    used, it is stongly recommended to set this to None to
                    avoid spurious high-freq oscillations in the fitted
                    functions. The default is None.
        boundary_mode : str
            Boundary mode, can be either absorbing or reflecting. The default
            is 'absorbing'.
        save_options : dict
            Options for saving the results:
                path : str
                    Local path for saving intermediate results. The default is
                    None, in which case nothing will be saved (the results will
                    only be in RAM).
                stride : int
                    Save intermediate files every stride number of epochs.
                    The default is max_iteration (only save the final file).
                schedule : numpy array, dtype=int
                    1D array with epoch numbers on which the fitted models
                    (peq, D, p0, fr) are saved to RAM. The default is
                    np.arange(1-new_sim, max_iteration+1), which includes all
                    of the epochs.
                sim_start : int
                    Starting epoch number (0 if new simulation, otherwise int >
                    0). The default is 0. Don't change as loading the
                    intermediate results and continuing optimization is not
                    supported yet.
        dataCV : list or None
            List that contains neuralflow.spike_data.SpikeData object for each
            datasample in the data. If not provided, validation loglik will not
            be calculated. The default is None.
        device : str
            Can be 'CPU' or 'GPU'. For GPU optimization, the platform has to be
            cuda-enabled, and cupy package has to be installed. The default is
            'CPU'.
        """

        # Check learning rate parameters for Adam:
        gd_opt._check_optimization_options(opt_options)
        return cls(
            dataTR, init_model, opt_options, line_search_options,
            pde_solve_params, boundary_mode, save_options, dataCV, device
        )

    def __repr__(self):
        if self.dataCV is None:
            with_CV = 'without validation data'
        else:
            with_CV = 'with validation data'
        return (f'Gradient descent optimizer, {self.boundary_mode} boundary, '
                f'{with_CV}'
                )

    @staticmethod
    def _check_optimization_options(opt_options):
        if 'learning_rate' not in opt_options.keys():
            raise ValueError('Learining rate is a mandatory parameter')
        params_to_opt = opt_options.get(
            'params_to_opt', opt_settings['params_to_opt']
        )

        for param in params_to_opt:
            if param not in opt_options['learning_rate'].keys():
                raise ValueError(f'{param} learning rate must be provided')
            elif opt_options['learning_rate'][param] < 0:
                raise ValueError(f'{param} leraning rate must be positive')
