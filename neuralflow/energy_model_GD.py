# -*- coding: utf-8 -*-

"""This is a part of neuralflow package/EnergyModel class.
This source file contains functions related to gradient descent optimization"""
import numpy as np
from tqdm import tqdm


def FeatureComplexity(self, peq=None):
    """Calculate feature complexity from a givein peq for the case of equilibrium inference


    Parameters
    ----------
    peq : numpy array, dtype=float
        The equilibrium probability density evaluated on SEM grid. The default is self.peq_

    Returns
    -------
    float
        Feature complexity (only valid for equilibrium or stationary inference).

    """

    if peq is None:
        peq = self.peq_
    return np.sum(4 * ((self.dmat_d_.dot(np.sqrt(peq)))**2) * self.w_d_)


def FeatureComplexityFderiv(self, peq=None):
    """Variational derivative of feature complexity wrt force F for the case of equilibrium inference


    Parameters
    ----------
    peq : numpy array, dtype=float
        The equilibrium probability density evaluated on SEM grid. The default is self.peq_

    Returns
    -------
    numpy array
        The variational derivative of loglikelihood w.r.t. feature complexity (only valid for equilibrium or stationary inference).

    """
    if peq is None:
        peq = self.peq_
    return self.Integrate(-4 * ((self.dmat_d_.dot(np.sqrt(peq)))**2) + 2 * self.dmat_d_.dot(self.dmat_d_.dot(peq)) + self.FeatureComplexity(peq=peq) * peq)


def _GD_optimization(self, data, save, inference, optimization):
    """Perfrom gradient-descent optimization

    Parameters
    ----------
    data : dictionary 
        Can include the following key-value pairs:
        dataTR : numpy array (N,2), dtype=np.ndarray.
            Training spike data packed as numpy array of the size (N,2), where each elements is a 1D array.
            N is the number of trials, and for each trial the first column contains inter spike intervals (ISIs) in seconds for all neurons,
            and the second column contains the corresponding neuronal IDs (trial termination, if recorded, is indicated with -1).
            data[i][0] - 1D array, a sequence of ISIs of all neurons for the trial i. The last entry can be time interval between the last spike
            and trial termination time.
            data[i][1] - 1D array, neuronal IDs of type int64 for the trial i. The last entry is -1 if the trial termination time is recorded.
            Example: neuron 0 spiked at times 0.12, 0.15, 0.25, and neuron 1 spiked at times 0.05, 0.2. Trial 0 started at t=0 and ended at t=0.28.
            In this case data[0][0]=np.array([0.05,0.07,0.03,0.05,0.05,0.03]), and data[0][1]=np.array([1,0,0,1,0,-1]).
        dataCV : validation data in the same format, optional
    save : dictionary
        Options for saving the results:
            path : str
                path for saving. The default is None, in which case nothing will be saved (the results will only be in RAM).
            stride : int
                Save intermediate files every stride number of iterations. The default is max_iteration (only save the final file).
            schedule : numpy array, dtype=int
                1D array with iteration numbers on which the fitted parameters (e.g. peq and/or D. p0) are saved to RAM/HDD.
                The default is np.arange(1-new_sim, max_iteration+1), which includes all of the iteration numbers, and saves fitting results on each iteration.
            sim_start : int
                Starting iteration number (0 if new simulation, otherwise int > 0). The default is 0.
            sim_id: int
                A unique simulation id needed to generate the filenames for the files with the results. The default is None.
        The default is None.
    inference : dictionary
        Metadata for training and validation data:
            metadataTR : dictionary
                Training metadata that can contain the following:
                    last_event_is_spike : bool
                        If true, trial termination time will be ignored (even if recorded). Otherwise, trial termination time will be used. The default is True.
                    absorption_event : bool
                        If true, absorption operator will be applied in the end of the loglikelihood chain. The default is False.
                The default is None.
            metadataCV : dictionary
                Validation metadata that can contain the following:
                    last_event_is_spike : bool
                        If true, trial termination time will be ignored (even if recorded). Otherwise, trial termination time will be used. The default is True.
                    absorption_event : bool
                        If true, absorption operator will be applied in the end of the loglikelihood chain. The default is False.
                The default is None.
    optimization : dictionary
        Optimization options:
            gamma : dictionary
                Dictionary that specifies the learning rates. Also specifies which parameters are to be optimized. Availible options are 'F', 'F0', and 'D'.
                Example: gamma={'F':0.01} - will optimize the driving force only, gamma={'F':0.01, 'D':0.001} will optimize both the driving force and
                noise magnitude using the provided learning rates.
            max_iteration : int
                Maximum number of iterations. The default is 100.
            loglik_tol : float
                Threshold for optimization termination due to the lack of likelihood improvement. If relative loglikelihood improvement is less than loglik_tol,
                stop the optimization. If set to zero, the optimization will not be stopped due to the lack of likelihood imporvement (in this case the
                optimization will terminate after reaching max_iteration). The default is 0.
            etaf : float
                Regularization strength for the potential function (set to zero for unregularized optimization). Only valid for stationary inference.
                The default is 0.
            gd_type : str
                Determines optimization strategy for several optimization parameters. Availible options are "coordinate_descent", in which case each of the
                gradients will be recalculated after each parameter update, or "simultaneous_update", in which case all of the parameters will be updated
                on each iteration by a single call to the _get_loglik_data function. The default is 'simultaneous_update'.


    Returns
    -------
    results: dictionary
        Dictionary with the fitted parameters. The possible entries are:
            'loglik': numpy array, float
                Training negative loglikelihoods saved on each iteration.
            'loglikCV':  numpy array, float
                Validation negative loglikelihoods saved on each iteration (only if dataCV is provided).
            'peqs' : numpy array (self.N,Niter)
                Fitted equilibirum probability distributions saved on the iterations specified by the schedule array (only if 'F' is in gamma).
            'p0s' : numpy array (N,Niter)
                Fitted initial probability distributions saved on the iterations specified by the schedule array (only if 'F0' is in gamma).
            'Ds': numpy array (Niter,)
                Fitted diffusion coefficients saved on the iterations specified by the schedule array  (only if 'D' is in gamma).
            'iter_num' : numpy array (Niter,)
                Iteration number on which peqs/p0s/Ds are saved.
    """

    # Unpack the parameters
    dataTR, dataCV = data['dataTR'], data['dataCV']
    if save is not None:
        path, stride, schedule, sim_start, sim_id = save['path'], save['stride'], save['schedule'], save['sim_start'], save['sim_id']
    if inference is not None:
        metadataTR, metadataCV = inference['metadataTR'], inference['metadataCV']
        if metadataCV is None:
            metadataCV = metadataTR
    gamma, max_iteration, loglik_tol, etaf, gd_type = optimization['gamma'], optimization['max_iteration'], optimization['loglik_tol'], optimization['etaf'], optimization['gd_type']

    # List of parameters to be optimized
    params_to_opt = list(gamma.keys())

    # initialize model parameters
    peq, D, fr = self.peq_, self.D_, self.fr_

    # initialize rho0, which is a scaled p0
    if self.p0_ is not None:
        p0 = self.p0_
        rho0 = p0 / np.sqrt(peq)
        rho0[peq < 10**-10] = 0
    else:
        rho0 = None  # Start from equilibrium distribution of the latent states

    # Determine this is a new or resumed simulation:
    new_sim = 1 if save is None or sim_start == 0 else 0

    # First index in results_save dict for model parameters and logliks
    iter_start, iter_start_ll = 0, 0

    # Index (in the schedule array) of the next iteration to be saved
    next_save = new_sim

    # Initialize results dictionary, save initial parameters
    results = {}
    results['logliks'] = np.zeros(max_iteration + new_sim)
    if dataCV is not None:
        results['logliksCV'] = np.zeros(max_iteration + new_sim)
    results['iter_num'] = np.zeros(schedule.size, dtype='int64')
    if new_sim:
        results['iter_num'][0] = 0
    if 'F' in params_to_opt:
        results['peqs'] = np.zeros((self.N, schedule.size))
        if new_sim:
            results['peqs'][..., 0] = peq
    if 'D' in params_to_opt:
        results['Ds'] = np.zeros(schedule.size)
        if new_sim:
            results['Ds'][0] = D
    if 'F0' in params_to_opt:
        results['p0s'] = np.zeros((self.N, schedule.size))
        if new_sim:
            results['p0s'][..., 0] = p0

    # Run GD
    oldLL = 0

    iter_list = range(new_sim + sim_start, new_sim + sim_start + max_iteration)
    if self.verbose:
        iter_list = tqdm(iter_list)

    for i, iIter in enumerate(iter_list):

        # compute gradients and loglikelihood
        if gd_type == 'simultaneous_update':
            gradients = self._get_loglik_data(dataTR, metadataTR, peq, rho0, D, fr, 'gradient', params_to_opt, None)
            # Extract loglikelihoods
            ll = gradients['loglik']
            if dataCV is not None:
                llCV = self.score(dataCV, metadataCV, peq, rho0, D, fr)

        for inum, param in enumerate(params_to_opt):
            if gd_type == 'coordinate_descent':
                gradients = self._get_loglik_data(dataTR, metadataTR, peq, rho0, D, fr, 'gradient', [param], None)
                if inum == 0:
                    ll = gradients['loglik']
                    if dataCV is not None:
                        llCV = self.score(dataCV, metadataCV, peq, rho0, D, fr)

            # Calculate the gradient descent update
            gd_update = gamma[param] * gradients[param]

            # Perform the updates:
            if param == 'F':
                # find current force, update it, update peq and p0:
                F = (self.dmat_d_.dot(np.log(peq)))
                F -= gd_update
                if etaf != 0:
                    F -= etaf * self.FeatureComplexityFderiv(peq=peq)
                peq = np.exp(self.Integrate(F))
                peq = np.maximum(peq, 10**(-10))
                peq /= np.sum(peq * self.w_d_)
                # Update rho0 as denominator has been changed
                if rho0 is not None:
                    rho0 = p0 / np.sqrt(peq)
                    rho0[peq < 10**-3] = 0
            elif param == 'D':
                D -= gd_update
                D = np.maximum(D, 0)
            elif param == 'F0':
                F0 = self.dmat_d_.dot(np.log(p0))
                F0 -= gd_update
                p0 = np.exp(self.Integrate(F0))
                p0 = np.maximum(p0, 10**(-10))
                p0 /= np.sum(p0 * self.w_d_)
                rho0 = p0 / np.sqrt(peq)
                rho0[peq < 10**-3] = 0

        # Update loglik (delayed  by 1 iteration)
        if i + new_sim > 0:
            results['logliks'][i + new_sim - 1] = ll
            if dataCV is not None:
                results['logliksCV'][i + new_sim - 1] = llCV

        # Save the results only on the iteration defined by the schedule array
        if iIter == schedule[next_save]:
            if 'F' in params_to_opt:
                results['peqs'][..., next_save] = peq
            if 'D' in params_to_opt:
                results['Ds'][..., next_save] = D
            if 'F0' in params_to_opt:
                results['p0s'][..., next_save] = p0
            results['iter_num'][next_save] = iIter
            next_save += 1

        # Save intermediate results to the harddrive
        if path is not None and (i + 1) % stride == 0:
            # Create and fill intermediate results_save dictionary
            results_save = {}
            for keys in results:
                if keys.startswith('log'):
                    results_save[keys] = results[keys][..., iter_start_ll:i + 1 + new_sim]
                else:
                    results_save[keys] = results[keys][..., iter_start:next_save]

            # Score the last iteration
            results_save['logliks'][-1] = self.score(dataTR, metadataTR, peq, rho0, D, fr)
            if dataCV is not None:
                results_save['logliksCV'][-1] = self.score(dataCV, metadataCV, peq, rho0, D, fr)

            # Save to file
            self.SaveResults('iterations', optimizer='GD', path=path,
                             iter_start=schedule[iter_start], iter_end=iIter, results=results_save, sim_id=sim_id)

            # Update saving indices
            iter_start = next_save
            iter_start_ll = i + 1 + new_sim

        # check for convergence
        if abs(ll - oldLL) / (1 + abs(oldLL)) < loglik_tol:
            if self.verbose:
                tqdm.write('Optimization converged in {} iterations'.format(iIter + 1))
            self.converged_ = True
            break

        oldLL = ll

    if (self.verbose & (i + 1 == (max_iteration - 1))):
        tqdm.write("Warning: convergence has not been achieved "
                   "in {} iterations".format(max_iteration + 1))

    # Adjust results if algorithm convereged earlier. Update parameters
    all_iter = slice(next_save)
    if 'F' in params_to_opt:
        results['peqs'] = results['peqs'][..., all_iter]
        self.peq_ = np.copy(results['peqs'][..., -1])
    if 'D' in params_to_opt:
        results['Ds'] = results['Ds'][all_iter]
        self.D_ = np.copy(results['Ds'][..., -1])
    if 'F0' in params_to_opt:
        results['p0'] = results['p0s'][..., all_iter]
        self.p0_ = np.copy(results['p0s'][..., -1])
    results['iter_num'] = results['iter_num'][all_iter]
    results['logliks'] = results['logliks'][0:i + 1 + new_sim]
    if dataCV is not None:
        results['logliksCV'] = results['logliksCV'][0:i + 1 + new_sim]

    # Calculate the last loglik
    results['logliks'][i + new_sim] = self.score(dataTR, metadataTR, peq, rho0, D, fr)
    if dataCV is not None:
        results['logliksCV'][i + new_sim] = self.score(dataCV, metadataCV, peq, rho0, D, fr)

    # Save the final chunk of data
    if path is not None and iter_start < next_save:
        results_save = {}
        for keys in results:
            if keys.startswith('log'):
                results_save[keys] = results[keys][..., iter_start_ll:i + 1 + new_sim]
            else:
                results_save[keys] = results[keys][..., iter_start:next_save]
        self.SaveResults('iterations', optimizer='GD', path=path,
                         iter_start=schedule[iter_start], iter_end=iIter, results=results_save, sim_id=sim_id)

    return results
