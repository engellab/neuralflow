# -*- coding: utf-8 -*-

""" Utility functions to compute PSTH (trial-average firing rates)
"""

import numpy as np
import scipy


def FiringRate(spikes, time_window, dt, tbegin, tend):
    """Calculates firing rate from spike data using rectangular time_window
    with step dt.


    Parameters
    ----------
    spikes : numpy array
        Spike times.
    time_window : float
        The size of rectangular moving window.
    dt : float
        The distance between mid points of the consequtive bins.
    tbegin : float
        Start time.
    tend : float
        End time.

    Returns
    -------
    time_bins : numpy array
        An array of time bins centers.
    rate : numpy array
        Estimated firing rate at each time bin.

    """
    time_bins = np.linspace(
        tbegin + time_window / 2,
        tend - time_window / 2,
        int((tend - tbegin - time_window) / dt)
    )
    if len(time_bins) == 0:
        time_bins = np.array([0.5 * (tbegin + tend)])
    spikes = spikes.reshape((-1, 1))
    time_bins = time_bins.reshape((1, -1))
    rate = np.sum(
        np.abs(spikes - time_bins) < time_window / 2, axis=0
    ) / time_window
    return time_bins, rate


def extract_psth(spike_data, RTs, time_window, dt, tbegin, tend):
    """Extract psth for plotting
    """

    num_neurons, num_trials = spike_data.shape

    rate = np.zeros((np.linspace(
        tbegin+time_window/2, tend-time_window/2,
        int((tend-tbegin-time_window)/dt)
    ).size,
        num_trials
    ))
    rates, rates_SEM = [], []
    hand_rt_median = np.median(RTs)
    for neuron in range(num_neurons):
        for trial in range(num_trials):
            tb, rate[:, trial] = FiringRate(
                spike_data[neuron, trial], time_window, dt, tbegin, tend
            )
            rate[np.squeeze(tb) > RTs[trial], trial] = np.nan

        rates.append(np.nanmean(rate, axis=1)[np.squeeze(tb) < hand_rt_median])
        rates_SEM.append(
            scipy.stats.sem(rate, axis=1, nan_policy='omit')[
                np.squeeze(tb) < hand_rt_median
            ]
        )
    return tb, rates, rates_SEM
