#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""This is a part of neuralflow package/EnergyModel class.
This source file contains functions for checking the input parameters and EnergyModel class initialization."""

import numpy as np
import numbers
import os


def _check_integer(param, constraint=0, error=None):
    """Checks if input is integer and greater or equal to the constraint
    """
    if not np.issubdtype(type(param), np.integer) or param < constraint:
        if error is None:
            return False
        else:
            raise ValueError(error)
    else:
        return True


def check_schedule(schedule, error):
    """Checks the schedule parameter
    """
    if type(schedule) is list:
        schedule = np.asarray(schedule, dtype=np.int)
    if not isinstance(schedule, np.ndarray) or not np.issubdtype(schedule.dtype, np.int) or not (schedule >= 0).all():
        if error is None:
            return False
        else:
            raise ValueError(error)

    # Sort schedule
    schedule.setflags(write=1)
    schedule.sort()

    # Remove duplicates
    schedule = schedule[np.insert(np.diff(schedule).astype(np.bool), 0, True)]

    return schedule


def check_em_parameters(func):
    """Decorator functions that checks EnergyModel initialization
    """
    def check_and_set_init_vars(self_var, num_neuron=1,
                                firing_model=[{"model": "linear", "params": {
                                    "r_slope": 50, "r_bias": 60}}],
                                peq_model={"model": "linear_pot",
                                           "params": {"slope": -2.65}},
                                p0_model=None,
                                boundary_mode=None,
                                D0=0.56, Nv=None, pde_solve_param={}, verbose=False):

        # Check the parameters
        _check_integer(num_neuron, 1, "Number of neurons is not integer of less than 1")

        # If firing_model consists of 1 neuron and passed as dictionary, convert into list
        if num_neuron == 1 and isinstance(firing_model, dict):
            firing_model = [firing_model]
        if not isinstance(firing_model, list):
            raise ValueError("Firing rate model: not a list")

        # Check firing_rate_model and evaluate firing rates
        self_var.firing_model_ = []
        for i, fr in enumerate(firing_model):
            if callable(fr):
                self_var.firing_model_.append(fr)
            elif 'model' in fr and fr['model'] in self_var._firing_model_types:
                self_var.firing_model_.append(lambda x, fr=fr: self_var._firing_model_types[fr['model']](x, **fr['params']))
            else:
                raise ValueError(
                    "Firing rate model #{}: unknown or not provided".format(i))

        # Check peq model
        if callable(peq_model):
            self_var.peq_model_ = self_var.peq_model
        else:
            if peq_model['model'] in self_var._peq_model_types:
                self_var.peq_model_ = lambda x, w: self_var._peq_model_types[peq_model['model']](
                    x, w, **peq_model['params'])
            else:
                raise ValueError("peq model: unknown or not provided")

        # Check p0 model
        if callable(p0_model):
            self_var.p0_model_ = p0_model
        elif p0_model is None:
            pass
        else:
            if p0_model['model'] in self_var._peq_model_types:
                self_var.p0_model_ = lambda x, w: self_var._peq_model_types[p0_model['model']](x, w, **p0_model['params'])
            else:
                raise ValueError("p0 model: unknown or not provided")

        # boundary mode
        if boundary_mode not in self_var._boundary_modes and boundary_mode is not None:
            raise ValueError("boundary mode: unknown")

        # Check D0 value
        if D0 <= 0.:
            raise ValueError("D0 must be positive.")

        # Check Nv:
        if Nv is not None:
            _check_integer(Nv, 1, "Nv is not integer of less than 1")

        # Force verbose type to bool
        verbose = bool(verbose)

        func(self_var, num_neuron, firing_model, peq_model, p0_model,
             boundary_mode, D0, Nv, pde_solve_param, verbose)
    return check_and_set_init_vars


def _check_data(self, data):
    """Checks if data is in the valid format.
    """
    if (not isinstance(data, np.ndarray)):
        raise ValueError("Data must be an array of spike-time arrays and indecies"
                         "of neurons which fired for each trial.")
    if (not len(data.shape) == 2) or (not data.shape[1] == 2):
        raise ValueError("Data must be an array with the shape (num_trial,2). "
                         "The shape of the provided array was {} instead.".format(data.shape))

    # number of sequencies in the data
    num_seq = data.shape[0]
    if (num_seq == 0):
        raise ValueError(
            "Empty list of spike-time arrays is provided, need some data to work with.")
    # now check each spike-time array
    for iSeq in range(num_seq):
        # check sequence for each trial
        data[iSeq, :] = self._check_sequence(data[iSeq, :])
    return data


def _check_sequence(self, sequence):
    """Checks if sequence (a single trial of data) is in the valid format
    """
    # check the inter-spike interval data
    seq = sequence[0]
    if (not isinstance(seq, np.ndarray)):
        raise ValueError(
            "Interspike interval data must be provided as a numpy array.")
    elif np.any(seq < 0.0):
        raise ValueError("Interspike interval data must be non-negative.")
    # check the data for index of neurons which fired
    ind_seq = sequence[1]
    if (not isinstance(ind_seq, np.ndarray)):
        raise ValueError(
            "Indecies of neurons which fired must be provided as a numpy array.")
    elif not np.issubdtype(ind_seq.dtype, np.integer) and ind_seq.size > 0:
        raise ValueError("Indecies of neurons which fired must be integers.")
    elif (np.any(ind_seq[:-1] < 0.0) or np.any(ind_seq >= self.num_neuron)):
        raise ValueError(
            "Indecies of neurons which fired must be within the range of number of neurons.")
    # check that inter-spike interval and indeces arrays are the same length
    if (not seq.shape == ind_seq.shape):
        raise ValueError("Arrays of interspike intervals and of indecies of neurons which fired "
                         "must have the same length.")
    return sequence


def _check_optimization_options(self, optimizer, options):
    """Checks optimization options
    """

    data, optimization = options['data'], options['optimization']
    save = options['save'] if 'save' in options else None
    inference = options['inference'] if 'inference' in options else {}
    if 'inference' not in options:
        options['inference'] = {'metadataTR': None, 'metadataCV': None}
    inference = options['inference']

    data_options = ['dataTR', 'dataCV']
    save_options = ['path', 'stride', 'sim_start', 'sim_id', 'schedule']
    inference_options = ['metadataTR', 'metadataCV']
    optimization_options = ['gd_type', 'gamma',
                            'max_iteration', 'loglik_tol', 'etaf']

    if optimizer not in self._optimizer_types:
        raise ValueError('Unknown optimizer')

    # Check data
    if 'dataTR' not in data:
        raise ValueError("Training data not provided")
    else:
        options['data']['dataTR'] = self._check_data(data['dataTR'])

    if 'dataCV' not in data or data['dataCV'] is None:
        options['data']['dataCV'] = None
    else:
        options['data']['dataCV'] = self._check_data(data['dataCV'])

    for entry in data.keys():
        if entry not in data_options:
            raise ValueError('Unknown data option {}'.format(entry))

    max_iteration = optimization['max_iteration'] if 'max_iteration' in optimization else 100

    # Check save options
    if save is not None:
        if 'path' not in save:
            if self.verbose:
                print('Warning: path is not specified! The results will only be saved to RAM')
            options['save']['path'] = None
        elif save['path'] is not None:
            if not isinstance(save['path'], str):
                raise ValueError('path has to be specified as string')
            elif not os.path.exists(save['path']):
                raise FileNotFoundError('the path {} does not exist'.format(save['path']))

        if 'stride' not in save or save['stride'] is None:
            if self.verbose:
                print('Warning: saveing stride not specified. Only the final result will be saved')
            options['save']['stride'] = max_iteration
        else:
            _check_integer(save['stride'], 1, 'stride has to be a positive integer')

        if 'sim_start' not in save or save['sim_start'] is None:
            if self.verbose:
                print('Warning: sim_start not provided, setting to zero')
            options['save']['sim_start'] = 0
        else:
            _check_integer(save['sim_start'], 0, 'sim_start has to be a non-negative integer')

        if 'sim_id' not in save or save['sim_id'] is None:
            if self.verbose:
                print('Warning: sim_id not provided for file saveing')
            options['save']['sim_id'] = ''
        else:
            if save['sim_id'] != '':
                _check_integer(save['sim_id'], 1, 'sim_id has to be a postive integer')

        if 'schedule' in save and save['schedule'] is not None:
            save['schedule'] = check_schedule(save['schedule'], 'save schedule must be a list or array with non-negative integers and integer dtype')

            # Needed for schedule adjustment
            new_sim = True if save['sim_start'] == 0 else False

            # Save initial parameters (zero iteration) by default if new_sim=True
            if save['schedule'][0] != 0 and new_sim:
                save['schedule'] = np.insert(save['schedule'], 0, 0)
            elif save['schedule'][0] == 0 and (not new_sim):
                save['schedule'] = save['schedule'][1:]

            # Check if last iteration to be saved coincides with max_iteration
            if save['schedule'][-1] < max_iteration + save['sim_start'] + new_sim - 1:
                if self.verbose:
                    print("Setting max_iteration to {}".format(
                        save['schedule'][-1] - save['sim_start'] + 1 - new_sim))
                optimization['max_iteration'] = int(
                    save['schedule'][-1] - save['sim_start'] + 1 - new_sim)
            elif save['schedule'][-1] > max_iteration + save['sim_start']:
                raise ValueError("Iteration #{} exceeds max_iterations".format(save['schedule'][-1]))
            options['save']['schedule'] = save['schedule']
        else:
            if self.verbose:
                print(
                    'Warning: schedule is not provided. The results will be saved on every iteration')
            options['save']['schedule'] = np.array(
                range(save['sim_start'], save['sim_start'] + max_iteration + 1)).astype(np.int)

        for entry in save.keys():
            if entry not in save_options:
                raise ValueError('Unknown save option {}'.format(entry))

    else:
        options['save'] = {}
        new_sim = True
        options['save']['path'] = None
        options['save']['sim_start'] = 0
        options['save']['stride'] = max_iteration
        options['save']['schedule'] = np.arange(0, max_iteration + new_sim)
        options['save']['sim_id'] = ''

    # check inference options
    for meta in inference_options:
        if meta in inference and inference[meta] is not None:
            if type(inference[meta]) is not dict:
                raise ValueError('{} must be a dictionary'.format(meta))
            else:
                for key, value in inference[meta].items():
                    if key not in ['last_event_is_spike', 'absorption_event']:
                        raise ValueError('Unknown {} entry {}'.format(meta, key))
                    else:
                        if value is list:
                            if not all(type(el) is bool for el in value):
                                raise ValueError('Expected boolean array/variable in {}'.format(meta))
                        elif type(value) is not bool:
                            raise ValueError('Expected boolean array/variable in {}'.format(meta))
        else:
            options['inference'][meta] = None

    if inference is not None:
        for entry in inference.keys():
            if entry not in inference_options:
                raise ValueError('Unknown inference option {}'.format(entry))

    if 'gd_type' not in optimization or optimization['gd_type'] is None:
        options['optimization']['gd_type'] = 'simultaneous_update'
    elif type(optimization['gd_type']) is not str or optimization['gd_type'] not in ["coordinate_descent", "simultaneous_update"]:
        raise ValueError('Unknown gd_type')

    if 'gamma' not in optimization:
        raise ValueError(
            'gamma must be specified in options for {} optimizier for at least one parameter'.format(optimizer))
    elif type(optimization['gamma']) is not dict:
        raise ValueError('gamma must be a dictionary')
    elif len(optimization['gamma'].keys()) == 0:
        raise ValueError('gamma must contain at least 1 key-value pair')
    else:
        if not all((isinstance(values, numbers.Number) or isinstance(values, list) or values > 0) and keys in self._parameters_to_optimize for keys, values in optimization['gamma'].items()):
            raise ValueError(
                'Error in gamma: unsupported parameter or learning rate not a number')

    if 'max_iteration' not in optimization or optimization['max_iteration'] is None:
        print('Warning: setting max_iteration to default value 100')
        options['optimization']['max_iteration'] = max_iteration
    elif not np.issubdtype(type(optimization['max_iteration']), np.integer) or optimization['max_iteration'] < 1:
        raise ValueError('Invalid max_iteration option')

    if 'loglik_tol' not in optimization or optimization['loglik_tol'] is None:
        options['optimization']['loglik_tol'] = 0
    else:
        if not isinstance(optimization['loglik_tol'], numbers.Number):
            raise ValueError('loglik_tol must be a number')

    if 'etaf' not in optimization or optimization['etaf'] is None:
        options['optimization']['etaf'] = 0
    else:
        if not isinstance(optimization['etaf'], numbers.Number):
            raise ValueError('etaf must be a positive number')

    for entry in optimization.keys():
        if entry not in optimization_options:
            raise ValueError('Unknown optimization option {}'.format(entry))
