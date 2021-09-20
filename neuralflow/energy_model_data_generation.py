# -*- coding: utf-8 -*-
"""This is a part of neuralflow package/EnergyModel class.
This source file contains functions for synthetic data generation."""

import numpy as np
import math
from tqdm import tqdm
from .energy_model_settings import MINIMUM_PEQ


def generate_data(self, deltaT=0.00001, time_epoch=[(0, 1)], last_event_is_spike=False):
    """Generate spike data and latent trajectories.


    Parameters
    ----------
    deltaT : float
        Size of the time bin in seconds for the numerical integration of the Langevin equation. The default is 0.00001.
    time_epoch : list
        List of N tuples, where N is the number of trials. Each tuple consists of start time and stop time in seconds. For the case of absorbing boundary, stop
        time will be the maximum allowed time for the trial to last (the trial will terminate before this time due to absorption, or at this time in an
        arbitrary latent state). The default is [(0,1)].
        Example: We want to generate 100 trials that start at t=0 and end at t=2, in this case time_epoch=[(0,2)]*100
    last_event_is_spike : bool
        If true, trial termination time will not be recorded. Otherwise, trial termination time will be recorded. The default is False.

    Returns
    -------
    data : numpy array (N,2), dtype=np.ndarray.
        Spike data packed as numpy array of the size (N,2), where each elements is a 1D array.
        N is the number of trials, and for each trial the first column contains inter spike intervals (ISIs) in seconds for all neurons,
        and the second column contains the corresponding neuronal IDs (trial termination, if recorded, is indicated with -1).
        data[i][0] - 1D array, a sequence of ISIs of all neurons for the trial i. The last entry can be time interval between the last spike
        and trial termination time.
        data[i][1] - 1D array, neuronal IDs of type int64 for the trial i. The last entry is -1 if the trial termination time is recorded.
        Example: neuron 0 spiked at times 0.12, 0.15, 0.25, and neuron 1 spiked at times 0.05, 0.2. Trial 0 started at t=0 and ended at t=0.28.
        In this case data[0][0]=np.array([0.05,0.07,0.03,0.05,0.05,0.03]), and data[0][1]=np.array([1,0,0,1,0,-1]).
    time_bins : numpy array (N,), dtype=np.ndarray
        For each trial contains times at which latent trajectory was recorded. N is the number of trials,
        and for each trial time is represented as 1D array of floats.
    x : numpy array (N,), dtype = np.ndarray
        Latent trajectories for each trial, N is the number of trials. Each entry is 1D array of floats.
    metadata : dictionary
        A dictionary with two entries:
            last_event_is_spike : bool
                Equals to the input parameter with the same name
            absorption_event : list (N,)
                List of strings for each trial with the following entries: 'absorbed', if the trial terminated due to trajectory absorption, or
                'observation_ended' if the trial terminated due to time out in an arbitrary latent state.
    """

    # By default, the boundary mode is reflecting
    boundary_mode = self.boundary_mode if self.boundary_mode is not None else 'reflecting'
    return self._generate_data(self.peq_, self.p0_, self.D_, self.firing_model_, self.num_neuron,
                               boundary_mode, deltaT, time_epoch, last_event_is_spike)


def _generate_data(self, peq, p0, D, firing_rate_model, num_neuron, boundary_mode, deltaT, time_epoch, last_event_is_spike):
    """Generates synthetic spike data and latent trajectories from a given model defined by (peq,p0,D,firing_rate_model).


    Parameters
    ----------
    peq : numpy array, dtype=float
        The equilibrium probability density evaluated on SEM grid.
    p0 : numpy array, dtype=float
        The initial probaiblity distribution.
    D : float
        Noise intensity.
    firing_model : list
        For each neuron, this list contains the firing rate functions. Each entry is either a function that returns an array of firing rate values,
        or a dictionary that specifies a model from ``firing_rate_models.py`` file.
    num_neuron : int
        A number of neuronal responses.
    boundary_mode : ENUM("absorbing","reflecting")
        Specify boundary mode that will apply the corresponding boundary condition for the latent trajectories.
    deltaT : float
        Size of the time bin in seconds for the integration of the Langevin equation.
    time_epoch : list
        List of N tuples, where N is the number of trials. Each tuple consists of start time and stop time in seconds. For the case of absorbing boundary,
        stop time will be the maximum allowed time for the trial to last (the trial will terminate before this time due to absorption, or at this time in
        an arbitrary latent state). The default is [(0,1)].
    last_event_is_spike : bool
        If true, trial termination time will not be recorded. Otherwise, trial termination time will be recorded.
    Returns
    -------
    See generate_data function.

    """
    num_trial = len(time_epoch)

    if p0 is None:
        p0 = peq  # If p0 not provided, assume equilibirum distribution.

    # generate diffusion trajectories
    x, time_bins, metadata = self._generate_diffusion(peq, p0, D, boundary_mode, deltaT, time_epoch)

    # initialize data arrays
    rate = np.empty((num_neuron, num_trial), dtype=np.ndarray)
    spikes = np.empty((num_neuron, num_trial), dtype=np.ndarray)

    # generate firing rates and spikes
    for iTrial in range(num_trial):
        for iCell in range(num_neuron):
            # Firing rate f(x(t))
            rate[iCell, iTrial] = firing_rate_model[iCell](x[iTrial])
            rt = rate[iCell, iTrial]
            # Generate spikes from rate
            spikes[iCell, iTrial] = self._generate_inhom_poisson(time_bins[iTrial][0:rt.shape[0]], rate[iCell, iTrial])

    #Calculate actual time epoch with the actual end of trial times (not timeouts)
    time_epoch_actual = [(time_epoch[i][0],time_bins[i][-1]+deltaT) for i in range(num_trial)]   

    # transform spikes to ISIs
    data = self.transform_spikes_to_isi(spikes, time_epoch_actual, last_event_is_spike)

    # record metadata
    metadata['last_event_is_spike'] = last_event_is_spike

    return data, time_bins, x, metadata


def transform_spikes_to_isi(self, spikes, time_epoch, last_event_is_spike=False):
    """Convert spike times to data array, which is a suitable format for optimization.


    Parameters
    ----------
    spikes : numpy array (num_neuron,N), dtype=np.ndarray
        A sequence of spike times for each neuron on each trial. Each entry is 1D array of floats.
    time_epoch : list of tuples
         List of N tuples, where N is the number of trials. Each tuple consists of the trial's start time and end time in seconds.
         Note that the end time should be an actual end time, but not the timeout in the case of last_event_is_spike is True.
    last_event_is_spike : bool
        If true, trial termination time will not be recorded. Otherwise, trial termination time will be recorded.

    Returns
    -------
    data : numpy array (N,2),dtype=np.ndarray.
        Spike data packed as numpy array of the size (N,2), where each elements is a 1D array of floats.
        N is the number of trials, and for each trial the first column contains the interspike intervals (ISIs),
        and the second column contains the corresponding neuronal indices.
    """

    num_neuron, num_trial = spikes.shape

    # initialize data array
    data = np.empty((num_trial, 2), dtype=np.ndarray)

    # indices of neurons that spiked
    spike_ind = np.empty(num_neuron, dtype=np.ndarray)

    # transform spikes to interspike intervals format
    for iTrial in range(num_trial):
        for iCell in range(num_neuron):
            spike_ind[iCell] = iCell * np.ones(len(spikes[iCell, iTrial]), dtype=np.int)
        all_spikes = np.concatenate(spikes[:, iTrial], axis=0)
        all_spike_ind = np.concatenate(spike_ind[:], axis=0)
        # create data array
        data[iTrial, 0] = np.zeros(len(all_spikes) + (not last_event_is_spike))

        if all_spikes.shape[0] == 0:
            data[iTrial, 1] = np.zeros(0)
            # If no spikes emitted, set to trial beginning time
            last_spike_time = time_epoch[iTrial][0]
        else:
            # sort spike times and neuron index arrays
            ind_sort = np.argsort(all_spikes)
            all_spikes = all_spikes[ind_sort]
            all_spike_ind = all_spike_ind[ind_sort]
            data[iTrial, 0][1:len(all_spikes)] = all_spikes[1:] - all_spikes[:-1]
            data[iTrial, 0][0] = all_spikes[0] - time_epoch[iTrial][0]  # handle the first ISI
            last_spike_time = all_spikes[-1]

        if not last_event_is_spike:
            data[iTrial, 0][-1] = time_epoch[iTrial][1] - last_spike_time
        # assign indicies of neurons which fired, -1 to absorption event
        data[iTrial, 1] = all_spike_ind if last_event_is_spike else np.concatenate((all_spike_ind, [-1]))
    return data


def _generate_inhom_poisson(self, time, rate):
    """Generate spike sequence from a given rate of inhomogenious Poisson process lambda(t) and time t


    Parameters
    ----------
    time : numpy array, dtype=float
        1D array of all time points
    rate : numpy array, dtype=float
        1D array of the corresponding firing rates

    Returns
    -------
    spikes : np.array,  dtype=float
        1D array of spike times

    """
    # calculate cumulative rate
    deltaT = time[1:] - time[:-1]
    r = np.cumsum(rate[0:-1] * deltaT)
    r = np.insert(r, 0, 0)
    deltaR = r[1:] - r[:-1]

    # generate 1.5 as many spikes as expected on average for exponential distribution with rate 1
    numX = math.ceil(1.5 * r[-1])

    # generate exponential distributed spikes with the average rate 1
    notEnough = True
    x = np.empty(0)
    xend = 0.0
    while notEnough:
        x = np.append(x, xend + np.cumsum(np.random.exponential(1.0, numX)))
        # check that we generated enough spikes
        if (not len(x) == 0):
            xend = x[-1]
        notEnough = xend < r[-1]

    # trim extra spikes
    x = x[x <= r[-1]]

    if len(x) == 0:
        spikes = np.empty(0)
    else:
        # for each x find index of the last rate which is smaller than x
        indJ = [np.where(r <= x[iSpike])[0][-1] for iSpike in range(len(x))]

        # compute rescaled spike times
        spikes = time[indJ] + (x - r[indJ]) * deltaT[indJ] / deltaR[indJ]

    return spikes


def _generate_diffusion(self, peq, p0, D, boundary_mode, deltaT, time_epoch):
    """Sample latent trajectory by integration of Langevin equation


    Parameters
    ----------
    peq : numpy array, dtype=float
        The equilibrium probability density evaluated on SEM grid.
    p0 : numpy array, dtype=float
        The initial probaiblity distribution.
    D : float
        Noise intensity.
    boundary_mode : string
        Specify boundary mode that will apply the corresponding boundary condition for optimization.
        Possbile option are "reflecting", "absorbing".
    deltaT : float
        Size of time bin in seconds used for integration of Langevin equation. The default is 0.00001.
    time_epoch : list of tuples
         List of N tuples, where N is number of trials. Each tuple consists of start time and stop time in seconds.The default is [(0,1)].

    Returns
    -------
    x : numpy array, dtype=float
        Latent trajectory, 1D array of floats
    time_bins : numpy array, dtype=float
       Times at which latent trajectory was recorded, 1D array of floats
    metadata : dictionary
        A part of metadata dictionary with only one entry:
            absorption_event : list (N,)
                List of strings for each trial with the following entries: 'absorbed', if the trial terminated due to trajectory absorption, or
                'observation_ended' if the trial terminated due to the timeout in an arbitrary latent state.
    """

    metadata = {}
    metadata['absorption_event'] = []
    num_trial = len(time_epoch)

    # pre-allocate output
    x = np.empty(num_trial, dtype=np.ndarray)
    time_bins = np.empty(num_trial, dtype=np.ndarray)

    # sample initial condition from the equilibrium distribution
    x0 = self._sample_from_p(p0, num_trial)

    # Normalization of peq
    peq0 = np.maximum(peq, 0)
    peq0 += MINIMUM_PEQ
    peq0 /= self.w_d_.dot(peq0)
    # compute force profile from the potential
    force = (self.dmat_d_.dot(peq0)) / peq0

    # fix very high force values on the boundary (so that particle does not travel too far)
    force[(np.abs(force) > 0.05 / (D * deltaT)) & (self.x_d_ < 0)] = 0.05 / (D * deltaT)
    force[(np.abs(force) > 0.05 / (D * deltaT)) & (self.x_d_ > 0)] = -0.05 / (D * deltaT)

    len_xd = len(force)

    if self.verbose:
        iter_list = tqdm(range(num_trial))
    else:
        iter_list = range(num_trial)

    for i, iTrial in enumerate(iter_list):
        # generate time bins
        time_bins[iTrial] = np.arange(time_epoch[iTrial][0], time_epoch[iTrial][1], deltaT)
        num_bin = len(time_bins[iTrial]) - 1
        y = np.zeros(num_bin + 1)
        y[0] = x0[iTrial]

        # generate random numbers
        noise = (np.sqrt(deltaT * 2 * D) * np.random.randn(num_bin))

        # account for absorbing boundary trajectories ending early
        max_ind = num_bin + 1

        # do Euler integration
        for iBin in range(num_bin):
            # find force at the current position by linear interpolation
            ind = np.argmax(self.x_d_ - y[iBin] >= 0)
            if ind == 0:
                f = force[0]
            elif ind == (len_xd - 1):
                f = force[-1]
            else:
                theta = (y[iBin] - self.x_d_[ind - 1]) / (self.x_d_[ind] - self.x_d_[ind - 1])
                f = (1.0 - theta) * force[ind - 1] + theta * force[ind]

            y[iBin + 1] = y[iBin] + D * f * deltaT + noise[iBin]

            # Handle boundaries:
            if boundary_mode == "reflecting":
                y[iBin + 1] = min(max(y[iBin + 1], 2 * self.x_d_[0] - y[iBin + 1]), 2 * self.x_d_[-1] - y[iBin + 1])

                # Regenerate the values if noise magnitude was very high
                error_state = True
                while error_state:
                    if y[iBin + 1] < self.x_d_[0] or y[iBin + 1] > self.x_d_[-1]:
                        y[iBin + 1] = y[iBin] + D * f * deltaT + noise[iBin]
                        y[iBin + 1] = min(max(y[iBin + 1], 2 * self.x_d_[0] - y[iBin + 1]), 2 * self.x_d_[-1] - y[iBin + 1])
                    else:
                        error_state = False
            elif boundary_mode == "absorbing":
                if y[iBin + 1] < self.x_d_[0] or y[iBin + 1] > self.x_d_[-1]:
                    max_ind = iBin
                    break

        metadata['absorption_event'].append('absorbed') if max_ind < num_bin + 1 else metadata['absorption_event'].append('observation_ended')
        x[iTrial] = y[:max_ind]
        time_bins[iTrial] = time_bins[iTrial][:max_ind]

    return x, time_bins, metadata


def _sample_from_p(self, p, num_sample):
    """Generate samples from a given probability distribution. Needed for initialization of the latent trajectories.


    Parameters
    ----------
    p : numpy array, dtype=float
        The probability distribution, 1D array of floats
    num_sample : int
        Number of samples

    Returns
    -------
    x : numpy array, dtype=float
        1D array of size num_sample that consists of samples randomnly drawn from a given probaiblity distribution

    """
    x = np.zeros(num_sample)
    pcum = np.cumsum(p * self.w_d_)

    y = np.random.uniform(0, 1, num_sample)

    for iSample in range(num_sample):
        # find index of the element closest to y[iSample]
        ind = (np.abs(pcum - y[iSample])).argmin()

        x[iSample] = self.x_d_[ind]

    return x
