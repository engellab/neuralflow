#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main optimization class
"""

import logging
import numpy as np
import os
import sys
import pickle
from scipy.optimize import minimize
from tqdm import tqdm
from copy import deepcopy
from neuralflow.base_optimizer import adam_opt, gd_opt
from neuralflow.settings import (
    _cmin, _cmax, _dmin, _dmax, implemented_optimizers,
    MINIMUM_D
)

logger = logging.getLogger(__name__)


class Optimization:
    """


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
    optimizer_name : str
        ENUM('ADAM', 'GD').
    opt_options : dict
        Optimization options (see init method of each of the optimizers).
    line_search_options : dict, optional
        Line search options. The default is {}.
    pde_solve_params : dict, optional
        Parameters for solving FPE. The default is {}.
    boundary_mode : str, optional
        'absorbing' or 'reflecting'. The default is 'absorbing'.
    save_options : dict, optional
        Options for saving the results. The default is {}.
    dataCV : list, optional
        Validation data. The default is None.
    device : str, optional
        'CPU' or 'GPU'. The default is 'CPU'.
    """

    def __init__(
        self, dataTR, init_model, optimizer_name, opt_options,
        line_search_options={}, pde_solve_params={},
        boundary_mode='absorbing', save_options={}, dataCV=None, device='CPU'
    ):
        """

        Public methods
        ------
        run_optimization, compute_tr_loglik, compute_val_loglik
        """

        # Need deepcopy to fix the parameters in place without modifying the
        # originals. Data will not be deepcopied as this can be a waste of
        # resources if the data is too large.
        init_model = deepcopy(init_model)
        opt_options = deepcopy(opt_options)
        line_search_options = deepcopy(line_search_options)
        pde_solve_params = deepcopy(pde_solve_params)
        save_options = deepcopy(save_options)

        if optimizer_name == 'ADAM':
            opt_fun = adam_opt
        elif optimizer_name == 'GD':
            opt_fun = gd_opt
        else:
            logger.info(f'Impl. optimizers: {implemented_optimizers}')
            raise ValueError(f'Unknown optimizer {optimizer_name}')
        self.optimizer_name = optimizer_name

        # Optimizer object - see base_optimization class
        self.optimizer = opt_fun.initialize(
            dataTR, init_model, opt_options, line_search_options,
            pde_solve_params, boundary_mode,
            save_options, dataCV, device
        )
        # Initialize ruinning average, RMS prop, and epoch counter needed for
        # ADAM algorithm.
        if optimizer_name == 'ADAM':
            self.optimizer._prepare_moments()

        self.device = device
        # Set up CUDA streams. Number of streams is
        # max(number of training trials, number of validation trials)
        if device == 'GPU':
            nstreams = max(
                [len(self.optimizer.get_dataTR(i))
                 for i in range(self.optimizer.num_datasamples)]
            )
            if dataCV is not None:
                nstreams_cv = max(
                    [len(self.optimizer.get_dataCV(i))
                     for i in range(len(self.optimizer.dataCV))]
                )
                nstreams = max(nstreams, nstreams_cv)
            self.optimizer.cuda._update_streams(nstreams)
            self.lib = self.optimizer.cuda.cp
        else:
            self.lib = np

        # Determine if this is a new simulation. Note that the option to
        # continue the simulation is under development, so always set
        # save_options['sim_start'] == 0
        self.new_sim = 1 if save_options['sim_start'] == 0 else 0

        # Number of epochs
        self.nepochs = self.optimizer.opt_options['max_epochs']

        # Initialize results dict
        self.results = self._init_results()

        # Adjust initial guess for firing rate to match the average firing in
        # the data for each neuron. Needed to accelerate optimization.
        _opt_params = self.optimizer.opt_options['params_to_opt']
        if 'Fr' in _opt_params or 'C' in _opt_params and self.new_sim:
            self._adjust_firing_rate()

        # Start and end epochs
        self.epoch_start = self.new_sim + save_options['sim_start']
        self.epoch_end = self.epoch_start + self.nepochs

    def _init_results(self):
        """Initialize results dict"""
        results = {}
        epochs_to_save = self.optimizer.save_options['schedule'].size

        # Iteration numbers on which parameters are saved
        results['iter_num'] = np.zeros(epochs_to_save, dtype='int')
        # Training logliks
        results['logliks'] = np.zeros(
            (self.optimizer.num_datasamples, self.nepochs + self.new_sim),
            dtype='float64'
        )
        if self.optimizer.dataCV is not None:
            results['logliksCV'] = np.zeros_like(results['logliks'])

        # Other params
        # Mapping from optimization parameters to the parameters that will be
        # recorded in the results dict
        param_mapping = {'F': 'peq', 'F0': 'p0',
                         'D': 'D', 'Fr': 'fr', 'C': 'fr'}
        for param in self.optimizer.opt_options['params_to_opt']:
            prim_param = param_mapping[param]
            if prim_param not in results.keys():
                results[prim_param] = []
        return results

    def _adjust_firing_rate(self):
        """Adjust initial guess for firing rate to match the average rates from
        data"""

        fr_av = [d.trial_average_fr() for d in self.optimizer.dataTR]
        model = self.optimizer.model

        # Average rate is integral of fr(x) over peq
        scaling = fr_av / np.sum(
            (model.peq * model.grid.w_d)[..., np.newaxis] * model.fr, axis=1
        )
        if 'Fr' in self.optimizer.shared_params:
            # If firing rates are shared, use sample-average scaling
            scaling = np.mean(scaling, axis=0)
            model.fr = model.fr * np.expand_dims(scaling, 0)
            model.C[0] = model.fr[0, 0, ...].copy()
        else:
            model.fr = model.fr * np.expand_dims(scaling, 1)
            model.C = model.fr[:, 0, ...].copy()
        if self.device == 'GPU':
            model.cuda_var.fr = self.lib.asarray(model.fr, dtype='float64')
            model.cuda_var.C = self.lib.asarray(model.C, dtype='float64')
        self.fr_av = fr_av

    # @profile
    def run_optimization(self):
        """Optimize the model on the data
        Inputs are defined at initialization, see __init__ docstring.

        Sets
        -------
        self.results : dict
            The optimization results. Contains the following:
                iter_num : numpy array, dtype = int
                    Epoch numbers on which the model is saved. The shape is
                    equal to save_options['schedule'].size. If 'schedule' is
                    not provided in save_options, it will be equal to
                    self.nepochs.
                logliks : numpy array
                    Training loglikelihoods, recorded on every epoch. The
                    shape is (num_datasamples, self.nepochs).
                logliksCV : numpy array
                    Validation loglikelihoods, recorded on every epoch. Only
                    present if dataCV is not None. The shape is
                    (num_datasamples, self.nepochs).
                peq : list
                    List that provides peq on the epochs number specified in
                    iter_num. The length is self.results['iter_num'].size.
                    Each element is numpy array of size (num_peq_models,
                    grid.N), where num_peq_models is the number of peq models
                    (which equals to num_datasamples if peq is non-shared, or
                    1 if peq is shared parameter, num_peq_models =
                    self.optimizer.model.params_size['peq']). Only presented if
                    'F' is optimized.
                D : list
                    List of D on the epochs number specified in iter_num. The
                    length is self.results['iter_num'].size. Each element is
                    numpy array of size self.optimizer.model.params_size['D'].
                    Only presented if 'D' is optimized.
                p0 : list
                    List of p0 on the epochs number specified in iter_num. The
                    length is self.results['iter_num'].size. Each element is
                    numpy array of size (num_p0_models, grid.N), where
                    num_p0_models is the number of p0 models (which equals to
                    num_datasamples if p0 is non-shared, or 1 if p0 is shared
                    parameter, num_p0_models =
                    self.optimizer.model.params_size['p0']). Only presented if
                    'F0' is optimized.
                fr : list
                    List of fr on the epochs number specified in iter_num. The
                    length is self.results['iter_num'].size. Each element is
                    numpy array of size (num_fr_models, grid.N, num_neurons),
                    where num_fr_models is the number of fr models (which
                    equals to num_datasamples if fr is non-shared, or 1 if fr
                    is shared parameter, num_fr_models =
                    self.optimizer.model.params_size['fr']). Only presented if
                    'Fr' or 'C' are optimized.

        If specified in save_opts, this will be also saved periodically to the
        persistance local storage.

        Returns
        -------
        None.

        """

        # For convinience, define aliases for some of the frequently used
        # variables.
        optimizer = self.optimizer
        all_iterations = np.sum(optimizer.iter_in_epoch)
        num_samples = optimizer.num_datasamples
        iter_in_epoch = optimizer.iter_in_epoch
        ls_opts = optimizer.line_search_options
        save_opts = optimizer.save_options
        opt_params = optimizer.opt_options['params_to_opt']

        # Pointer to the class varaible that holds model parameters
        if self.device == 'CPU':
            model_params = self.optimizer.model
        else:
            model_params = self.optimizer.model.cuda_var

        # For persistance storage of the intermediate results. start tracks
        # the starting index of model parameters in self.results,
        # start_ll tracks the starting index of loglik
        start = start_ll = 0

        # index of the next epoch when the results will be saved to
        # self.results
        next_save = self.new_sim

        # Save initial likelihoods and model params to the results structure
        if self.new_sim:
            for cur_samp in range(num_samples):
                # If number of batches > 1, need to compute likelihood now.
                # Otherwise it will be computed along with the gradients and
                # recorded later
                if optimizer.iter_in_epoch[cur_samp] > 1:
                    # Compute initial likelihood
                    self.compute_tr_loglik(cur_samp, 0)
                self.compute_val_loglik(cur_samp, 0)

            # Save initial params to the results
            for param in opt_params:
                # Don't record fr twice if C and Fr are optimized
                if 'Fr' in opt_params and 'C' in opt_params and param == 'C':
                    continue
                param = self.optimizer.opt_model_map[param]
                self.results[param].append(
                    getattr(model_params, param).copy()
                )

        for i, iEpoch in enumerate(
                tqdm(range(self.epoch_start, self.epoch_end))
        ):

            # Shuffle data
            shuffle_trials = [
                np.random.permutation(optimizer.num_trial[samp])
                for samp in range(num_samples)
            ]
            shuffle_datasamples = np.random.permutation(
                np.concatenate(
                    [
                        i * np.ones(iter_in_epoch[i])
                        for i in range(num_samples)
                    ],
                    axis=0
                ).astype(int))
            # Counter of iterations for each data sample
            iter_per_samp = np.zeros(num_samples, dtype=int) - 1

            for iIter in range(all_iterations):
                logger.debug(f'Epoch {iEpoch}, Iteration {iIter}')

                cur_samp = shuffle_datasamples[iIter]
                iter_per_samp[cur_samp] += 1

                # Size of the current minibatch
                if iter_per_samp[cur_samp] != iter_in_epoch[cur_samp]-1:
                    cur_mb_size = optimizer.mini_batch_size[cur_samp]
                else:
                    cur_mb_size = optimizer.last_batch_size[cur_samp]

                # Take the data for this minibatch
                start_ind = (
                    optimizer.mini_batch_size[cur_samp] *
                    iter_per_samp[cur_samp]
                )
                end_ind = start_ind + cur_mb_size
                data_mb = [
                    optimizer.get_dataTR(cur_samp)[trial]
                    for trial in shuffle_trials[cur_samp][start_ind:end_ind]
                ]

                # get gradients and extract loglikelihood
                gradients = optimizer.gradient.get_grad_data(
                    data_mb, optimizer.model, cur_samp, 'gradient'
                )

                # Update the parameters
                for param in opt_params:
                    # index of the current param
                    param_num = min(
                        cur_samp,
                        optimizer.model.params_size[
                            optimizer.opt_model_map[param]] - 1
                    )
                    if param in self.optimizer.shared_params:
                        cur_iter = iIter
                    else:
                        cur_iter = iter_per_samp[cur_samp]
                    if (
                        (param == 'C' or param == 'D') and
                        iEpoch in ls_opts[f'{param}_opt']['epoch_schedule'] and
                        cur_iter in
                        ls_opts[f'{param}_opt']['iter_schedule'][param_num]
                    ):
                        # Line search update
                        self._update_params_ls(param, cur_samp, param_num)
                    else:
                        gd_update = self._update_rule(
                            gradients, param, param_num
                        )
                        # Gradient-based update
                        self._update_param_gd(param, gd_update, param_num)

            # Update loglik
            ll_index = i + self.new_sim
            for cur_samp in range(num_samples):
                if optimizer.iter_in_epoch[cur_samp] == 1:
                    # loglik is already computed, but before the parameters
                    # were updated
                    self.results['logliks'][cur_samp, ll_index - 1] = (
                        gradients['loglik']
                    )
                else:
                    # For batched optimization, need to compute likelihood
                    # on a full training dataset
                    self.compute_tr_loglik(cur_samp, ll_index)

                # Validation loglik
                self.compute_val_loglik(cur_samp, ll_index)

            # Save model
            if iEpoch == save_opts['schedule'][next_save]:
                for param in opt_params:
                    prim_param = self.optimizer.opt_model_map[param]
                    if 'Fr' in opt_params and 'C' in opt_params:
                        if param == 'C':
                            continue
                    self.results[prim_param].append(
                        getattr(model_params, prim_param).copy()
                    )
                self.results['iter_num'][next_save] = iEpoch
                next_save += 1

            # Save intermediate results to persistence storage
            if (
                    save_opts['path'] is not None and
                    (i+1) % save_opts['stride'] == 0
            ):
                # Update loglik for GD - only for 1-batch mode because of
                # loglik lagging
                for cur_samp in range(num_samples):
                    if self.optimizer.iter_in_epoch[cur_samp] == 1:
                        self.compute_tr_loglik(cur_samp, ll_index)

                self._save_results(start, next_save, start_ll, ll_index + 1)

                # Update starting saving points
                start = next_save
                start_ll = ll_index + 1

            # flush std out
            sys.stdout.flush()

        # Calculate the last loglik (unless already calculated)
        if save_opts['stride'] is None or (i+1) % save_opts['stride'] != 0:
            for cur_samp in range(num_samples):
                if optimizer.iter_in_epoch[cur_samp] == 1:
                    self.compute_tr_loglik(cur_samp, i + self.new_sim)

        # Save the last portion of the results
        if save_opts['path'] is not None and start < next_save:
            self._save_results(start, next_save, start_ll,
                               i + 1 + self.new_sim)

        # Transfer results from GPU to CPU
        if self.device == 'GPU':
            for key in self.results.keys():
                if type(self.results[key]) is list:
                    for i in range(len(self.results[key])):
                        self.results[key][i] = self.lib.asnumpy(
                            self.results[key][i]
                        )
                else:
                    self.results[key] = self.lib.asnumpy(self.results[key])

        logger.info('Optimization completed')

    def compute_tr_loglik(self, cur_samp, epoch):
        """ Compute training negative loglik
        """
        self.results['logliks'][cur_samp, epoch] = (
            self.optimizer.gradient.get_grad_data(
                self.optimizer.get_dataTR(cur_samp),
                self.optimizer.model,
                cur_samp,
                'loglik'
            )
        )

    def compute_val_loglik(self, cur_samp, epoch):
        """ Compute validation negative loglik
        """
        if self.optimizer.dataCV is not None:
            self.results['logliksCV'][cur_samp, epoch] = (
                self.optimizer.gradient.get_grad_data(
                    self.optimizer.get_dataCV(cur_samp),
                    self.optimizer.model,
                    cur_samp,
                    'loglik'
                )
            )

    def _save_results(self, start, end, start_ll, end_ll, save_data=False):
        """ Save the results into local persistance storage
        """
        to_save = ['opt_options', 'line_search_options',
                   'pde_solve_params', 'boundary_mode', 'save_options',
                   'device']

        if save_data:
            to_save += ['dataTR', 'dataCV']
        save_res = dict.fromkeys(to_save)

        for key in to_save:
            save_res[key] = getattr(self.optimizer, key)

        save_res['with_CV'] = False if self.optimizer.dataCV is None else True
        save_res['optimizer_name'] = self.optimizer_name

        if self.device == 'GPU':
            self.optimizer.model.sync_model('GPU_to_CPU')
        save_res['model'] = self.optimizer.model

        if self.optimizer_name == 'ADAM':
            save_res['adam_counter'] = self.optimizer.adam_counter
            if self.device == 'GPU':
                save_res['gradients_av'], save_res['RMS_av'] = {}, {}
                for key in self.optimizer.gradients_av.keys():
                    save_res['gradients_av'][key] = (
                        self.lib.asnumpy(self.optimizer.gradients_av[key])
                    )
                    save_res['RMS_av'][key] = (
                        self.lib.asnumpy(self.optimizer.RMS_av[key])
                    )
            else:
                save_res['gradients_av'] = self.optimizer.gradients_av
                save_res['RMS_av'] = self.optimizer.RMS_av

        save_res['results'] = {}
        for key in self.results:
            if key == 'iter_num':
                save_res['results'][key] = self.results[key][..., start: end]
            elif key.startswith('loglik'):
                save_res['result'][key] = self.results[key][...,
                                                            start_ll: end_ll]
            elif key == 'D':
                save_res['result'][key] = self.results[key][..., start: end]
            else:
                if self.device == 'GPU':
                    save_res['result'][key] = []
                    for d in save_res['result'][key][start: end]:
                        save_res['result'][key].append(self.lib.asnumpy(d))
                else:
                    save_res['result'][key] = (
                        save_res['result'][key][start: end]
                    )
                save_res['result'][key] = np.array(save_res['result'][key])

        fullname = os.path.join(
            self.optimizer.save_options['path'],
            f'results_iterations_{start_ll}_{end_ll}.pkl'
        )
        with open(fullname, 'wb') as file:
            pickle.dump(save_res, file, protocol=pickle.HIGHEST_PROTOCOL)

    # @profile
    def _update_rule(self, gradients, param, model_num):
        """ADAM/GD update
        """
        lr = self.optimizer.opt_options['learning_rate']
        if self.optimizer_name == 'ADAM':
            # Update running average
            self.optimizer.gradients_av[param][model_num] = (
                lr['beta1'] * self.optimizer.gradients_av[param][model_num] +
                (1 - lr['beta1']) * gradients[param]
            )
            # Update RMS prop
            if param != 'C' and param != 'D':
                self.optimizer.RMS_av[param][model_num] = (
                    lr['beta2'] * self.optimizer.RMS_av[param][model_num] +
                    (1 - lr['beta2']) *
                    self.lib.power(
                        self.lib.linalg.norm(gradients[param], axis=0), 2
                    )
                )
            else:
                self.optimizer.RMS_av[param][model_num] = (
                    lr['beta2'] * self.optimizer.RMS_av[param][model_num] +
                    (1 - lr['beta2']) * self.lib.power(gradients[param], 2)
                )
            # Update epoch counter
            self.optimizer.adam_counter[param][model_num] += 1

            numerator = (
                self.optimizer.gradients_av[param][model_num] /
                (1 - lr['beta1']**self.optimizer.adam_counter[param][model_num]
                 )
            )
            denominator = (
                self.lib.sqrt(self.optimizer.RMS_av[param][model_num] /
                              (1 - lr['beta2'] **
                               self.optimizer.adam_counter[param][model_num])
                              ) + lr['epsilon']
            )
            gd_update = lr['alpha'] * numerator / denominator
        else:
            # GD update rule
            gd_update = lr[param] * gradients[param]
        return gd_update

    def _update_param_gd(self, param, gd_update, model_num):
        """Update a parameter
        """
        model = self.optimizer.model
        if self.device == 'CPU':
            model_params = self.optimizer.model
        else:
            model_params = self.optimizer.model.cuda_var

        if param == 'F':
            # Compute force from peq, update force, compute new peq
            peq = model_params.peq[model_num]
            F = model.force_from_peq(peq, self.device)
            F = F - gd_update

            # Regularization term
            if self.optimizer.opt_options['etaf'] != 0:
                F = (
                    F - self.optimizer.opt_options['etaf'] *
                    model.FeatureComplexityFderiv(peq, self.device)
                )
            peq_old = peq.copy()

            # Update peq
            model_params.peq[model_num] = model.peq_from_force(F, self.device)

            # Update rho0, since the denominator chaged
            if model.non_equilibrium:
                # This is numerically stable update
                ratio = self.lib.sqrt(peq_old / model_params.peq[model_num])
                if model.params_size['peq'] >= model.params_size['p0']:
                    model_params.rho0[model_num] = (
                        model_params.rho0[model_num] * ratio
                    )
                else:
                    # If p0 are non-shared and peq is shared, need to update
                    # all of the rho0s.
                    for i in range(model.params_size['rho0']):
                        model_params.rho0[i] = model_params.rho0[i] * ratio
        elif param == 'Fr':
            # Compute Fr from fr, update Fr, compute new fr
            Fr = model.Fr_from_fr(model_params.fr[model_num], self.device)
            Fr = Fr - gd_update
            fr = model.fr_from_Fr(Fr, model_params.C[model_num], self.device)
            model_params.fr[model_num] = fr
        elif param == 'C':
            # Compute new C, divide fr by old C and multiply by new C
            fr = model_params.fr[model_num]
            fr = fr / model_params.C[model_num]
            # Do not replace with -=, as there is an issue with __iadd__
            model_params.C[model_num] = model_params.C[model_num] - gd_update
            model_params.C[model_num] = self.lib.maximum(
                model_params.C[model_num], 0)
            fr = fr * model_params.C[model_num]
            model_params.fr[model_num] = fr
        elif param == 'D':
            # Direct update
            model_params.D[model_num] = model_params.D[model_num] - gd_update
            model_params.D[model_num] = np.maximum(
                model_params.D[model_num], MINIMUM_D)
        elif param == 'F0':
            # Compute p0 from F0, update F0, compute new p0 and rho0
            F0 = model.force_from_peq(model_params.p0[model_num], self.device)
            F0 = F0 - gd_update
            p0 = model.peq_from_force(F0, self.device)
            ratio = p0 / model_params.p0[model_num]
            if model.params_size['rho0'] == model.params_size['p0']:
                model_params.rho0[model_num] = (
                    model_params.rho0[model_num] * ratio
                )
            else:
                for i in range(model.params_size['rho0']):
                    model_params.rho0[i] = model_params.rho0[i] * ratio
            model_params.p0[model_num] = p0

    def _update_params_ls(self, param, model_num, param_num):
        """Perform line search
        """
        logger.debug(f'Performing LS of {param} for model {model_num}')
        lib = self.lib
        model = self.optimizer.model
        if self.device == 'CPU':
            model_params = self.optimizer.model
        else:
            model_params = self.optimizer.model.cuda_var

        if param == 'C':
            Cmin = lib.minimum(
                lib.min(model_params.fr[param_num], axis=0), _cmin)
            Cmax = lib.maximum(
                lib.max(model_params.fr[param_num], axis=0), _cmax)
            if self.device == 'GPU':
                Cmin, Cmax = lib.asnumpy(Cmin), lib.asnumpy(Cmax)
                C_init = lib.asnumpy(model_params.C[param_num])
            else:
                C_init = model_params.C[param_num]
            # Optimize
            minC = minimize(
                lambda chi: self.line_search_score_C(
                    self.optimizer.get_dataTR(model_num), model, model_num,
                    model_params.fr, param_num, chi
                ),
                C_init,
                # model_params.C[param_num],
                bounds=list(zip(Cmin, Cmax)),
                method='L-BFGS-B',
                options={
                    'ftol': 1e-3, 'disp': False, 'maxiter':
                        self.optimizer.line_search_options[
                            'C_opt'
                        ]['max_fun_eval']
                }
            )
            # Reset adam counter
            if self.optimizer_name == 'ADAM':
                if self.optimizer.adam_counter is not None:
                    self.optimizer.adam_counter['C'][param_num] = 0
            model_params.C[param_num] = lib.asarray(minC.x)
            model_params.fr[param_num] = (
                model_params.fr[param_num] /
                model_params.fr[param_num, 0] * model_params.C[param_num]
            )
        elif param == 'D':
            lowerDbound = _dmin
            max_runs, cur_runs, not_done = 10, 0, True
            # Crashes for small D, try gradually increasing the lower bound
            while not_done:
                cur_runs += 1
                Dinit = model_params.D.copy()
                minD = minimize(
                    lambda chi: self.line_search_score_D(
                        self.optimizer.get_dataTR(model_num), model, model_num,
                        model_params.D, param_num, chi
                    ),
                    model_params.D[param_num],
                    method='L-BFGS-B',
                    bounds=[(lowerDbound, _dmax)],
                    options={
                        'ftol': 1e-3, 'disp': False, 'maxiter':
                        self.optimizer.line_search_options[
                            'D_opt']['max_fun_eval'],
                        'maxls': 5
                    }
                )
                if (
                    minD.message != 'ABNORMAL_TERMINATION_IN_LNSRCH' or
                    cur_runs == max_runs
                ):
                    not_done = False
                else:
                    lowerDbound += 0.5 * _dmin
            if minD.message == 'ABNORMAL_TERMINATION_IN_LNSRCH':
                self.loger.warning('D line search did not succeed')
                model_params.D = Dinit
            else:
                model_params.D[param_num] = minD.x[0]
                if self.optimizer_name == 'ADAM':
                    if self.optimizer.adam_counter is not None:
                        self.optimizer.adam_counter['D'][param_num] = 0

    def line_search_score_D(self, dataTR, model, model_num, D, param_num, chi):
        """ Line search function for optimization
        """
        D[param_num] = chi[0]
        return self.optimizer.gradient.get_grad_data(dataTR, model, model_num,
                                                     'loglik').item()

    def line_search_score_C(self, dataTR, model, model_num, fr, param_num,
                            chi):
        """ Line search function for optimization
        """
        fr[param_num] = fr[param_num] / fr[param_num, 0]
        fr[param_num] = fr[param_num] * self.lib.asarray(chi)
        return self.optimizer.gradient.get_grad_data(
            dataTR, model, model_num, 'loglik'
        ).item()
