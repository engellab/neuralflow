#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Visualize optimization results in unittests
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os


def PlotResults(title, params_to_opt, nresults, results, em_gt, ll_gt,
                ll_gt_cv, save_fold=None):
    n_samples = results['logliks'].shape[0]

    title = (f'{title}, inference of {params_to_opt}')
    if type(params_to_opt) is str:
        params_to_opt = [params_to_opt]

    if 'C' in params_to_opt and 'Fr' in params_to_opt:
        params_to_opt.remove('C')

    # size of parameters
    param_mapping = {'F': 'peq', 'F0': 'p0', 'C': 'fr', 'Fr': 'fr', 'D': 'D'}
    param_size = {
        p: results[param_mapping[p]][0].shape[0] for p in params_to_opt
    }
    if n_samples > 1:
        title += (
            f', shared_params={[p for p in params_to_opt if param_size[p]==1]}'
        )
    num_plots = sum(param_size.values()) + n_samples
    num_cols = num_plots // 3 + (num_plots % 3 > 0)

    fig = plt.figure(figsize=(25, 5*num_cols))
    gs = gridspec.GridSpec(num_cols, 3, wspace=0.5, hspace=0.8)

    if nresults == 1:
        title += ' by line search'

    fig.suptitle(title)
    colors = plt.get_cmap('jet')(np.linspace(0.0, 1.0, nresults+1))

    subplot_num = -1
    for samp in range(n_samples):
        subplot_num += 1
        ax = plt.subplot(gs[subplot_num//3, subplot_num % 3])
        iters_to_plot = results['iter_num'].copy().astype('float')

        # Log scale does not disply zero, hack it
        if results['iter_num'][0] == 0:
            iters_to_plot[0] = 10**-1

        # Scale loglik
        ll0_tr = results['logliks'][samp][0]
        ll_tr = (
            (results['logliks'][samp][results['iter_num']] - ll0_tr) /
            np.abs(ll0_tr)
        )
        ax.semilogx(
            iters_to_plot, ll_tr, '--', linewidth=3, color=[0.1, 0.1, 0.1],
            label=f'Training negative loglik, datasample {samp}'
        )
        plt.plot(
            results['iter_num'],
            (ll_gt[samp] - ll0_tr) / np.abs(ll0_tr) * np.ones_like(
                results['iter_num']
            ),
            '-', linewidth=3, color=[0.1, 0.1, 0.1],
            label='Ground-truth Training'
        )
        if ll_gt_cv is not None:
            ll0_cv = results['logliksCV'][samp][0]
            ll_cv = (
                (results['logliksCV'][samp][results['iter_num']] - ll0_cv)
                / np.abs(ll0_cv)
            )
            ax.semilogx(
                iters_to_plot, ll_cv, '-.', linewidth=3, color='blue',
                label='Validation negative loglik, datasample {samp}'
            )
            plt.plot(
                results['iter_num'],
                (ll_gt_cv[samp]-ll0_cv)/np.abs(ll0_cv) * np.ones_like(
                    results['iter_num']
                ),
                '-', linewidth=3, color='blue',
                label='Ground-truth Validation'
            )

        iter_to_plot = (
            np.linspace(0, len(results['iter_num']) - 1, nresults+1)
            .round()
            .astype(int)
        )
        for i in range(nresults+1):
            iter_num = iter_to_plot[i]
            plt.plot(
                iters_to_plot[iter_num], ll_tr[iter_num], 'o',
                color=colors[i, :], markersize=15,
                label=f'Iteration {iter_num}'
            )
            if ll_gt_cv is not None:
                plt.plot(
                    iters_to_plot[iter_num], ll_cv[iter_num], 'o',
                    color=colors[i, :], markersize=15
                )
        plt.ylabel(r'Relative $\log\mathscr{L}$', fontsize=15)
        plt.xlabel('Iteration number', fontsize=15)
        plt.legend(ncol=2, prop={'size': 6})
        plt.gcf().suptitle(title)

        for i, param in enumerate(params_to_opt):
            if samp > param_size[param] - 1:
                continue
            subplot_num = subplot_num + 1
            ax = plt.subplot(gs[subplot_num//3, subplot_num % 3])

            if n_samples > 1:
                if param_size[param] > 1:
                    suffix = f', DS {samp}'
                else:
                    suffix = ', shared'
            else:
                suffix = ''
            if param == 'F':
                plt.plot(
                    em_gt.grid.x_d, -np.log(em_gt.peq[samp]), color='black',
                    linewidth=3, label='Ground-truth'
                )
                for i in range(nresults+1):
                    iter_num = iter_to_plot[i]
                    plt.plot(
                        em_gt.grid.x_d,
                        -np.log(results['peq'][iter_num][samp, :]),
                        color=colors[i, :], linewidth=3,
                        label=f'Iteration {results["iter_num"][iter_num]}'
                    )
                plt.ylabel(r'Potential, $\Phi(x)$' + suffix, fontsize=15)
                plt.xlabel(r'Latent state, $x$', fontsize=15)
            elif param == 'F0':
                plt.plot(
                    em_gt.grid.x_d, em_gt.p0[samp], color='black', linewidth=3,
                    label='Ground-truth'
                )
                for i in range(nresults+1):
                    iter_num = iter_to_plot[i]
                    plt.plot(
                        em_gt.grid.x_d, results['p0'][iter_num][samp, :],
                        color=colors[i, :], linewidth=3,
                        label=f'Iteration {results["iter_num"][iter_num]}'
                    )
                plt.ylabel(r'$p_0(x)$' + suffix, fontsize=15)
                plt.xlabel(r'Latent state, $x$', fontsize=15)
            elif param == 'D':
                all_iters = results['iter_num']
                plt.plot(
                    all_iters, np.ones_like(all_iters) * em_gt.D[samp],
                    color='black', linewidth=3, label='Ground-truth'
                )
                plt.plot(
                    all_iters,
                    [results['D'][i][samp] for i in range(len(results['D']))],
                    color=[0.5, 0.5, 0.5], linewidth=3
                )
                plt.ylabel(r'$D$' + suffix, fontsize=15)
                plt.xlabel(r'Epoch number', fontsize=15)
            elif param in ['Fr', 'C']:
                num_neurons = results['fr'][0].shape[2]
                for neuron in range(num_neurons):
                    plt.plot(
                        em_gt.grid.x_d, em_gt.fr[samp,
                                                 :, neuron], color='black',
                        linewidth=3,
                        label='Ground-truth' if neuron == 0 else None
                    )
                for i in range(nresults+1):
                    iter_num = iter_to_plot[i]
                    for neuron in range(num_neurons):
                        plt.plot(em_gt.grid.x_d,
                                 results['fr'][iter_num][samp, :, neuron],
                                 color=colors[i, :], linewidth=3
                                 )
                plt.ylabel(r'$fr(x)$' + suffix, fontsize=15)
                plt.xlabel(r'Latent state, $x$', fontsize=15)
            plt.legend(ncol=2, prop={'size': 6})

    if save_fold is not None:
        plt.savefig(os.path.join(save_fold, f'{title}.png'))
        plt.close(plt.gcf())
