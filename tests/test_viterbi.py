#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests and visualize Viterbi algorithm
"""

import logging
import unittest
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from neuralflow.model import model
from neuralflow.spike_data import SpikeData
from neuralflow.data_generation import SyntheticData
from neuralflow.grid import GLLgrid
from neuralflow.gradients import Grads
from neuralflow.viterbi import Viterbi

logger = logging.getLogger(__name__)


class TestViterbi(unittest.TestCase):
    """Perform optimization
    """
    @classmethod
    def setUpClass(cls):
        """
        """

        # Initialize pde solver and energy model class
        cls.boundary_mode = ['absorbing', 'reflecting']

        # Feel free to change
        cls.pde_solve_params = {'xbegin': -1, 'xend': 1, 'Np': 8, 'Ne': 16}

        cls.grid = GLLgrid(**cls.pde_solve_params)

        # Feel free to test different ground-truths
        cls.gt_param = {
            'peq_model': {'model': 'linear_pot', 'params': {'slope': -1}},
            'p0_model': {'model': 'single_well', 'params': {'miu': 100}},
            'D': 0.5,
            'fr_model': [
                *[{"model": "linear", "params": {"slope": 50, "bias": 60}}]*1,
                *[{"model": "linear", "params": {"slope": -50, "bias": 60}}]*1,
            ]
        }
        cls.gt_model = model.new_model(**cls.gt_param, grid=cls.grid)

        # Synthetic data generation
        num_trials = 3
        cls.data, cls.latent_trajectories, cls.time_bins = {}, {}, {}
        for bm in cls.boundary_mode:
            dg = SyntheticData(cls.gt_model, bm)
            logger.info(f'Generating {num_trials} trials of data using {bm} '
                        'boundary mode')
            data, cls.time_bins[bm], cls.latent_trajectories[bm] = (
                dg.generate_data(0, 3, num_trials, 0)
            )
            cls.data[bm] = SpikeData(data, 'ISIs')

    def test_Viterbi(self):
        """ Use Viterbi to predict most likely trajectory on each trial. Plot
        the predicted trajectories against the ground-truth trajectories.
        """

        # Optimize driving force
        for bm in self.boundary_mode:
            with self.subTest(
                    f'Testing Viterbi with {bm} boundary mode', bm=bm
            ):
                logger.info(f'Running Viterbi with {bm} boundary mode')

                grad = Grads(
                    self.pde_solve_params, bm,
                    num_neuron=self.gt_model.num_neuron,
                )
                viterbi = Viterbi(grad)
                trajectories_vit, _ = viterbi.run_viterbi(
                    self.data[bm], self.gt_model
                )

                # Plot the results
                norm = matplotlib.colors.Normalize(vmin=0, vmax=10)
                cmap = matplotlib.cm.tab10
                colors = {key: cmap(norm(key))[:-1] for key in range(10)}
                plt.figure()
                for t in range(min(len(self.data[bm].data), 3)):
                    plt.plot(
                        self.time_bins[bm][t], self.latent_trajectories[bm][t],
                        color=colors[t],
                        label=f'ground-truth trajectories, trial {t}',
                        alpha=0.5
                    )
                    plt.plot(
                        np.concatenate(
                            ([0], np.cumsum(self.data[bm].data[t][0]))),
                        trajectories_vit[t], '--', color=colors[t],
                        linewidth=2,
                        label=f'Viterbi trajectories, trial {t}',
                    )
                plt.title(f'Viterbi, boundary_mode = {bm}')
                plt.legend()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logging.getLogger().setLevel(logging.INFO)
    unittest.main()
