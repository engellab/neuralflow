# -*- coding: utf-8 -*-
""" Functions for visualization used in 2021_learning_nonstationary_dynamics
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_spikes(data, handle, colormap, spike_spacing=0.7, spike_height=0.4,
                spike_linewidth=1.5):
    """Visualize spike rasters


    Parameters
    ----------
    data : numpy array (N,2), dtype=np.ndarray.
        Spike data in ISIs format.
    handle : matplotlib.axes._subplots.AxesSubplot
        Handle of the matplotlib axes
    colormap : List
        For each trial contains color in RGB format.
    spike_spacing : float
        Vertical spacing between the spikes. The default is 0.7.
    spike_height : float
        Height of the spike ticks. The default is 0.4.
    spike_linewidth : float
        Width of the spike ticks. The default is 1.5.

    Returns
    -------
    None.

    """
    num_trials = data.shape[0]

    # Convert ISIs into spikes, discard trial end time, and convert ms into
    # seconds
    spikes = []
    for i in range(num_trials):
        if data[i][1][-1] == -1:
            spikes.append(1000 * np.cumsum(data[i][0][:-1], axis=0))
        else:
            spikes.append(1000 * np.cumsum(data[i][0], axis=0))

    # Plot spike rasters
    for i in range(num_trials):
        for sp in spikes[num_trials - i - 1]:
            handle.plot(
                [sp, sp],
                [spike_spacing * i - spike_height / 2,
                 spike_spacing * i + spike_height / 2],
                linewidth=spike_linewidth,
                color=colormap[num_trials - i - 1]
            )

    # Some adjustements
    handle.set_yticklabels([])
    handle.set_yticks([])
    handle.spines["top"].set_visible(False)
    handle.spines["right"].set_visible(False)
    handle.spines["left"].set_visible(False)


def plot_fitting_results(figure_handle1, figure_handle2, init_model, results,
                         gt_model, ll_gt, iterations, colors, num_model=0):
    """Visualise fitting results: plot negative loglikelihood vs. iteration
    number, and fitted potential functions on the selected iterations.
    This format was used for learning_nonstationary_dynamics (2.0.0 version)

    Parameters
    ----------
    figure_handle1,figure_handle2 : matplotlib handles
       Where to plot negative loglikelihood vs. iteration number and the fitted
       potentials, respectively.
    em_fit : EnergyModel
       Fitted EnergyModel object.
    em_gt : EnergyModel
        Ground-truth EnergyModel object which was used to generate the data.
    fit_options : dictionary
        Options used for fitting.
    iterations : numpy array
        List of iterations on which the potential function will be plotted.
        Same size as the colors list.
    colors : list
        List with RGB colors, where each entry is a list with three RGB values.
        These colors will be used to plot model potential on the selected
        iterations. Same size as iterations array.

    Returns
    -------
    """

    # Plot Relative loglikelihood
    ax = plt.subplot(figure_handle1)
    rel_lls = (ll_gt - results['logliks'][num_model]) / ll_gt
    iterations_all = np.arange(1, rel_lls.size + 1)
    ax.plot(iterations_all, rel_lls, linewidth=3, color=[0.35, 0.35, 0.35])
    ax.plot(iterations_all, np.zeros_like(iterations_all),
            '--', linewidth=3, color=[0.5, 0.5, 0.5])
    for i, iteration in enumerate(iterations):
        ax.plot(iterations_all[iteration], rel_lls[iteration],
                '.', markersize=60, color=colors[i])

    plt.ylabel(r'Relative $\log\mathscr{L}$', fontsize=15)
    plt.xlabel('Iteration number', fontsize=15)
    plt.xscale('log')
    ax.tick_params(axis='both', which='major', labelsize=15)

    # Plot the fitted potential functions
    ax = plt.subplot(figure_handle2)

    # Potential is negative log of peq
    ax.plot(gt_model.grid.x_d, -np.log(gt_model.peq[num_model]), linewidth=3,
            color='black', label='Ground-truth')
    ax.plot(
        init_model.grid.x_d, -np.log(results['peq'][0][num_model, :]),
        linewidth=3, color=[0.5, 0.5, 0.5], label='Initialization'
    )
    for i, iteration in enumerate(iterations):
        ax.plot(
            init_model.grid.x_d,
            -np.log(results['peq'][iteration][num_model, :]),
            linewidth=3, color=colors[i],
            label='Iteration {}'.format(iteration)
        )
    plt.legend(fontsize=15)
    plt.xlabel(r'Latent state, $x$', fontsize=15)
    plt.ylabel(r'Potential, $\Phi(x)$', fontsize=15)
    ax.tick_params(axis='both', which='major', labelsize=15)
