# -*- coding: utf-8 -*-
"""This is a part of neuralflow package/EnergyModel class.
This source file contains public functions for the framework optimization."""

import numpy as np
import os.path


def score(self, data, metadata=None, peq=None, rho0=None, D=None, fr=None):
    """Evaluate negative log-likelihood for a given data and model


    Parameters
    ----------
    data : numpy array (N,2), dtype=np.ndarray.
        Spike data packed as numpy array of the size (N,2), where each elements is a 1D array.
        N is the number of trials, and for each trial the first column contains inter spike intervals (ISIs) in seconds for all neurons,
        and the second column contains the corresponding neuronal IDs (trial termination, if recorded, is indicated with -1).
        data[i][0] - 1D array, a sequence of ISIs of all neurons for the trial i. The last entry can be time interval between the last spike
        and trial termination time.
        data[i][1] - 1D array, neuronal IDs of type int64 for the trial i. The last entry is -1 if the trial termination time is recorded.
        Example: neuron 0 spiked at times 0.12, 0.15, 0.25, and neuron 1 spiked at times 0.05, 0.2. Trial 0 started at t=0 and ended at t=0.28.
        In this case data[0][0]=np.array([0.05,0.07,0.03,0.05,0.05,0.03]), and data[0][1]=np.array([1,0,0,1,0,-1]).
    metadata : dictionary
        Metadata dictionary that supports the following options:
           last_event_is_spike : bool or list/array of bools
                If true, trial termination time will be ignored (even if recorded). Otherwise, trial termination time will be used. The default is False.
                Can be specified for all trials, or for each trial separately.
            absorption_event : bool or list/array of bools
                If true, absorption operator will be applied in the end of the loglikelihood chain. The default is True.
                Can be specified for all trials, or for each trial separately.
        The default is None.
    peq : numpy array, dtype=float
        The equilibrium probability density evaluated on SEM grid. The default is self.peq_
    rho0 : numpy array, dtype=float
        Scaled initial probaiblity distribution, normalized by sqrt(peq): rho0=p0/np.sqrt(peq).
        The default is self.p0_/np.sqrt(peq), if self.p0_ is not None, otherwise np.sqrt(peq).
    D : float
        Noise intensity. The default is self.D_
    fr : numpy array, (N,num_neuron), dtype=float
        2D array that contains firing rate functions for each neuron evaluated on SEM grid. The default is self.fr_

    Returns
    -------
    likelihood : float
        Negative loglikelihood of a data given a model
    """
    # Run data checks
    self._check_data(data)

    if peq is None:
        peq = self.peq_
    if rho0 is None:
        rho0 = self.p0_ / np.sqrt(peq) if self.p0_ is not None else None
    if fr is None:
        fr = self.fr_
    if D is None:
        D = self.D_

    return self._get_loglik_data(data, metadata, peq, rho0, D, fr, 'likelihood', None, None)['loglik']


def fit(self, optimizer='GD', options=None):
    """Perform model fitting


    Parameters
    ----------
    data : numpy array (N,2), dtype=np.ndarray.
        See docstring for score function for details.
    optimizer : str
        Optimizier. Currently, the only availible option is 'GD'. The default is 'GD'.
    options : dictionary
        see self._GD_optimization for availiable options of GD optimizer. The default is None

    Returns
    -------
    em : self
        A fitted EnergyModel object.
    """

    # Run data check
    self._check_data(options['data']['dataTR'])

    # Check optimizer
    if optimizer not in self._optimizer_types:
        raise NotImplementedError("optimizer should be one of %s" % self._optimizer_types)
    if options is None:
        options = {}

    # reset self.converged_ to False
    self.converged_ = False

    # Save initial parameters:
    if not hasattr(self, 'peq_init'):
        self.peq_init = np.copy(self.peq_)
    if not hasattr(self, 'fr_init'):
        self.fr_init = np.copy(self.fr_)
    if not hasattr(self, 'D_init'):
        self.D_init = np.copy(self.D_)

    self._check_optimization_options(optimizer, options)
    self.iterations_GD_ = self._GD_optimization(**options)

    return self


def SaveResults(self, results_type='iterations', **options):
    """Save fitting results to a file


    Parameters
    ----------
    results_type : str
        Type of the results to save. Supported types: 'iterations', which saves iteration results. The default is 'iterations'
    options : dict
        Availiable options:
            results : dictionary
                dictionary with the results (mandatory)
            path : str
                path to files. The default is empty str
            name : str
                enforce a particular file name, otherwise, the default name will be generated.
            sim_id : int
                id of a simulation (appended to the name and saved inside the dictionary). The default is empty str.
            iter_start : int
                Starting iteration used for automatic filename generation. The default is iter_num[0]
            iter_end : int
                Terminal iteration used for automatic filename generation. The default is iter_num[-1]
    """

    # Insert '/' in path string if necessary
    if 'path' in options:
        if options['path'][-1] != '/' and len(options['path']) > 0:
            options['path'] += '/'
        path = options['path']
    else:
        path = ''

    # Exctract simulation ID if provided
    if 'sim_id' in options:
        sim_id = options['sim_id']
        if sim_id == '':
            print("Warning: sim_id not provided")
    else:
        print("Warning: sim_id not provided")
        sim_id = ''

    # define the name prefix and data dictionary
    if results_type == 'iterations':
        prefix = 'results'
        if 'results' in options:
            dictionary = options['results']
        else:
            raise ValueError("Please specify data_dictionary")
    else:
        raise ValueError(results_type + " is not supported")

    # Add sim_id if providied:
    if isinstance(sim_id, int):
        dictionary['sim_id'] = sim_id

    # Generate name and save
    if 'name' in options:
        fullname = path + options['name']
    else:
        if results_type == 'iterations':
            if 'iter_start' in options:
                iter_start = str(options['iter_start'])
            elif 'iter_num' in dictionary:
                iter_start = str(dictionary['iter_num'][0])
            else:
                iter_start = '0'
            iter_start += '-'
            if 'iter_end' in options:
                iter_end = str(options['iter_end'])
            elif 'iter_num' in dictionary:
                iter_end = str(dictionary['iter_num'][-1])
            else:
                iter_end = str(self.iterations_GD_['logliks'].size)
            postfix = '_iterations_'
        fullname = path + prefix + \
            str(sim_id) + postfix + iter_start + iter_end + '.npz'
    if os.path.isfile(fullname):
        print('Error: file ' + fullname + ' already exists. Aborting...')
        return
    else:
        np.savez(fullname, **dictionary)
        print('file ' + fullname + ' saved.')


def calc_peq(self, F):
    """Calculate peq from the force


    Parameters
    ----------
    F : numpy array, dtype=float
        Driving force, 1D array

    Returns
    -------
    result : numpy array, dtype=float
        Equilibirum probability distribution (peq), 1D array

    """
    result = np.exp(self.Integrate(F))
    result /= np.sum(result * self.w_d_)
    return result


def calc_F(self, peq):
    """Calculate force from the peq


     Parameters
     ----------
     peq : numpy array, dtype=float
         Equilibirum probability distribution (peq), 1D array

     Returns
     -------
     F : numpy array, dtype=float
         Driving force, 1D array
     """
    return self.dmat_d_.dot(np.log(peq))
