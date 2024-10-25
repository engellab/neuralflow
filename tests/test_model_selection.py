# -*- coding: utf-8 -*-
"""
A few tests for model selection using fc_tools.
Mostly tests that there are no errors in model selection.
"""

import logging
import unittest
from neuralflow.model import model
from neuralflow.data_generation import SyntheticData
from neuralflow.spike_data import SpikeData
from neuralflow.grid import GLLgrid
from neuralflow.feature_complexity.fc_base import FC_tools
from neuralflow.optimization import Optimization
from copy import deepcopy


class TestModelSelection(unittest.TestCase):
    """
    """
    @classmethod
    def setUpClass(cls):
        """Define the model and some unittest hyperparameters
        """

        cls.boundary_modes = ['absorbing', 'reflecting']
        cls.pde_solve_params = {'xbegin': -1, 'xend': 1, 'Np': 8, 'Ne': 16}

        cls.grid = GLLgrid(**cls.pde_solve_params)

        # Ground-truth model for data generation
        cls.em = model.new_model(
            peq_model={'model': 'linear_pot', 'params': {'slope': 5}},
            p0_model={'model': 'single_well', 'params': {'miu': 100}},
            D=0.3,
            fr_model=[
                {"model": "linear", "params": {"slope": 50, "bias": 60}},
                {"model": "linear", "params": {"slope": -50, "bias": 60}}
            ],
            grid=cls.grid,
        )
        cls.num_neurons = cls.em.fr.shape[2]

        # Generate data for absorbing and reflecting modes
        cls.data = {bm: [] for bm in cls.boundary_modes}
        for bm in cls.boundary_modes:
            dg = SyntheticData(cls.em, bm)
            data, _, _ = dg.generate_data(0, 1, 10, 0)
            cls.data[bm].append(SpikeData(data, 'ISIs'))
            data, _, _ = dg.generate_data(0, 1, 10, 0)
            cls.data[bm].append(SpikeData(data, 'ISIs'))

        cls.opt_params = {'max_epochs': 20, 'mini_batch_number': 2,
                          'learning_rate': {'alpha': 0.5}}

        # visualization
        cls.n_epochs_to_disp = 5

        # This determines which parameters are shared and which are not
        cls.params_size = {'peq': 1, 'D': 1, 'fr': 1, 'p0': 1}

        cls.model_param_mapping = {
            'F': 'peq_model', 'F0': 'p0_model', 'D': 'D', 'Fr': 'fr_model',
            'C': 'fr_model'
        }
        cls.params_to_opt = ['F', 'F0', 'D', 'Fr', 'C']

        cls.initial_guess = {
            'peq_model': {'model': 'uniform', 'params': {}},
            'p0_model': {'model': 'uniform', 'params': {}},
            'D': 1,
            'fr_model': [
                {"model": "linear", "params": {"slope": -5, "bias": 15}},
                {"model": "linear", "params": {"slope": 10, "bias": 100}},
            ],
            'grid': cls.grid
        }

    def test_FC_equilibrium(self):
        """ Optimize the model on two datasets and perform feature complexity
        analysis
        """
        # Set p0 to none for non-equilibeirum case. Setting p0 to None makes
        # the model equilibirum
        init_model_params = deepcopy(self.initial_guess)
        init_model_params['p0_model'] = None
        init_model = model.new_model(**init_model_params)
        assert not init_model.non_equilibrium, \
            'For some reason the model object is not equilibirum'

        # Also don't optimize F0 and D. We exclude p0 since the model is
        # equilibrium, so there is no p0. We exclude D since the formula
        # developed in Genkin & Engel 2020 paper does not use D.
        params_to_opt = ['F', 'C', 'Fr']

        for t_num, bm in enumerate(self.boundary_modes):
            with self.subTest(
                    f'Testing equilibirum model selection, boundary mode {bm}',
                    bm=bm
            ):

                optimization1 = Optimization(
                    self.data[bm][0],
                    init_model,
                    'ADAM',
                    {**self.opt_params,  **{'params_to_opt': params_to_opt}},
                    pde_solve_params=self.pde_solve_params,
                    boundary_mode=bm,
                )
                # run optimization
                optimization1.run_optimization()

                optimization2 = Optimization(
                    self.data[bm][0],
                    init_model,
                    'ADAM',
                    {**self.opt_params, **{'params_to_opt': params_to_opt}},
                    pde_solve_params=self.pde_solve_params,
                    boundary_mode=bm,
                )
                # run optimization
                optimization2.run_optimization()
                fc = FC_tools(non_equilibrium=False, model=init_model,
                              boundary_mode=bm, terminal_time=1)
                FCs1, min_inds_1, FCs2, min_inds_2, JS, FC_opt_ind = (
                    fc.FeatureConsistencyAnalysis(
                        optimization1.results, optimization2.results,
                        0.015, 3, 0
                    )
                )

    def test_FC_nonequilibrium(self):
        """ Optimize the model on two datasets and perform feature complexity
        analysis
        """
        init_model = model.new_model(**self.initial_guess)
        # In this case model should ne non-equilibrium, since p0_model is
        # specified
        assert init_model.non_equilibrium, \
            'For some reason the model object is not non-equilibirum'
        params_to_opt = self.params_to_opt

        for t_num, bm in enumerate(self.boundary_modes):
            with self.subTest(
                    'Testing nonequilibirum model selection, boundary mode'
                    f'{bm}',
                    bm=bm
            ):

                optimization1 = Optimization(
                    self.data[bm][0],
                    init_model,
                    'ADAM',
                    {**self.opt_params,  **{'params_to_opt': params_to_opt}},
                    pde_solve_params=self.pde_solve_params,
                    boundary_mode=bm,
                )
                # run optimization
                optimization1.run_optimization()

                optimization2 = Optimization(
                    self.data[bm][0],
                    init_model,
                    'ADAM',
                    {**self.opt_params, **{'params_to_opt': params_to_opt}},
                    pde_solve_params=self.pde_solve_params,
                    boundary_mode=bm,
                )
                # run optimization
                optimization2.run_optimization()
                fc = FC_tools(non_equilibrium=True, model=init_model,
                              boundary_mode=bm, terminal_time=1)
                FCs1, min_inds_1, FCs2, min_inds_2, JS, FC_opt_ind = (
                    fc.FeatureConsistencyAnalysis(
                        optimization1.results, optimization2.results,
                        0.015, 3, 0
                    )
                )


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logging.getLogger().setLevel(logging.DEBUG)
    unittest.main()
