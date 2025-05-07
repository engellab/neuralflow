#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unittests for the optimization. It tests various
optimization settings. It also visualizes the results

Testcases include:
    testForceInference : Adam optimization of the driving force F that defines
        potential Phi(x).
    testP0Inference : Adam optimization of p0(x) through F0.
    testDInference : Adam optimization of D.
    testFrInference : Adam optimization of Fr (and plots fr).
    testCInference : Adam optimization of C.
    testDLineSearch : Adam optimization of D by line search.
    testCLineSearch : Adam optimization of C by line search.
The following parameters can be adjusted in SetUpClass method:
    -pde_solve_params.
    -em_gt parameters for the ground-truth model.
    -num_trials for data generation.
    -with_validation = True/False - whether to calculate validation score.
    -max_epochs - number of iteraitons
    -optimizer
    -sgd_options - to do mini-batch descent.

"""


import logging
import unittest
from neuralflow.model import model
from visualization import PlotResults
from neuralflow.data_generation import SyntheticData
from neuralflow.spike_data import SpikeData
from neuralflow.grid import GLLgrid
from neuralflow.optimization import Optimization
from itertools import product
import pathlib
import numpy as np
from pkg_resources import working_set
GPU_support = any([pkg.key.startswith('cupy') for pkg in working_set])
logger = logging.getLogger(__name__)


class TestSharedOptimization(unittest.TestCase):
    """Perform optimization
    """
    @classmethod
    def setUpClass(cls):
        """
        """

        # Initialize pde solver and energy model class
        cls.boundary_mode = ['reflecting', 'absorbing']

        # Feel free to change
        cls.pde_solve_params = {'xbegin': -1, 'xend': 1, 'Np': 8, 'Ne': 16}

        cls.grid_cpu = GLLgrid(**cls.pde_solve_params)
        if GPU_support:
            cls.grid_gpu = GLLgrid(**cls.pde_solve_params, with_cuda=True)

        # Feel free to test different ground-truths
        gt_param_1 = {
            'peq_model': {
                "model": "manual",
                "params": {
                    "interp_x": [-1, -0.25, 1],
                    "interp_y": [2, 4, 0],
                    "bc_left": [[1, 0], [1, 0]],
                    "bc_right": [[1, 0], [1, 0]]
                }
            },
            'p0_model': {
                "model": "single_well", "params": {"miu": 200, "xmin": 0}
            },
            'D': 0.2,
            'fr_model': [
                {"model": "linear", "params": {"slope": 50, "bias": 60}},
                {"model": "peaks", "params": {
                    "center": [-0.3], "width": [0.7], "amp": [60]
                }},
                {"model": "peaks", "params": {
                    "center": [-0.6, 0.6], "width": [0.3, 0.5], "amp": [30, 80]
                }}
            ]
        }
        em_gt_1 = model.new_model(**gt_param_1, grid=cls.grid_cpu)

        # Feel free to test different ground-truths
        gt_param_2 = {
            'peq_model': {"model": "linear_pot", "params": {"slope": -3}},
            'p0_model': {"model": "single_well",
                         "params": {"miu": 200, "xmin": 0}
                         },
            'D': 0.2,
            'fr_model': [
                {"model": "linear", "params": {"slope": 50, "bias": 60}},
                {"model": "peaks", "params": {
                    "center": [-0.3], "width": [0.7], "amp": [60]
                }},
                {"model": "peaks", "params": {
                    "center": [-0.6, 0.6], "width": [0.3, 0.5], "amp": [30, 80]
                }}
            ]
        }
        em_gt_2 = model.new_model(**gt_param_2, grid=cls.grid_cpu)

        # Create a model from these two models
        cls.em_gt_cpu = model(
            np.concatenate((em_gt_1.peq, em_gt_2.peq), axis=0),
            np.concatenate((em_gt_1.p0, em_gt_2.p0), axis=0),
            np.concatenate((em_gt_1.D, em_gt_2.D), axis=0),
            np.concatenate((em_gt_1.fr, em_gt_2.fr), axis=0),
            params_size={'peq': 2, 'D': 2, 'fr': 2, 'p0': 2},
            grid=cls.grid_cpu
        )
        if GPU_support:
            cls.em_gt_gpu = model(
                np.concatenate((em_gt_1.peq, em_gt_2.peq), axis=0),
                np.concatenate((em_gt_1.p0, em_gt_2.p0), axis=0),
                np.concatenate((em_gt_1.D, em_gt_2.D), axis=0),
                np.concatenate((em_gt_1.fr, em_gt_2.fr), axis=0),
                params_size={'peq': 2, 'D': 2, 'fr': 2, 'p0': 2},
                grid=cls.grid_gpu,
                with_cuda=True
            )

        # Synthetic data generation
        with_cv = True
        num_training_trials = [50, 30]
        num_val_trials = [max(el // 10, 3) for el in num_training_trials]
        cls.dataTR = {bm: [] for bm in cls.boundary_mode}
        cls.dataCV = {bm: [] for bm in cls.boundary_mode}

        for samp in range(2):
            for bm in cls.boundary_mode:
                dg = SyntheticData(cls.em_gt_cpu, bm)
                logger.info(
                    f'Generating {num_training_trials[0]} trials of training '
                    f'data using {bm} boundary mode, datasample {samp}'
                )
                data, _, _ = dg.generate_data(
                    0, 1, num_training_trials[samp], samp)
                cls.dataTR[bm].append(
                    SpikeData(data, 'ISIs', with_cuda=GPU_support))
                if with_cv:
                    logger.info(
                        f'Generating {num_val_trials[0]} trials of validation '
                        f'data using {bm} boundary mode, datasample {samp}'
                        )
                    dataCV, _, _ = dg.generate_data(
                        0, 1, num_val_trials[samp], samp)
                    cls.dataCV[bm].append(
                        SpikeData(dataCV, 'ISIs', with_cuda=GPU_support)
                        )

        # optimization
        cls.optimizers = ['ADAM', 'GD']
        cls.opt_params = {'max_epochs': 20, 'mini_batch_number': 10}
        cls.opt_params_ls = {'max_epochs': 1, 'mini_batch_number': 1}

        cls.ls_options_simultaneous_inference = {
            'C_opt': {'epoch_schedule': [1, 10]},
            'D_opt': {'epoch_schedule': [1, 10]}
        }

        # visualization
        cls.n_epochs_to_disp = 5

        # This determines which parameters are shared and which are not
        cls.params_size = {'peq': 2, 'D': 2, 'fr': 2, 'p0': 2}

        cls.model_param_mapping = {
            'F': 'peq_model', 'F0': 'p0_model', 'D': 'D', 'Fr': 'fr_model',
            'C': 'fr_model'
            }
        cls.params_to_opt = ['F', 'F0', 'D', 'Fr', 'C']

        cls.initial_guess = {
            'peq_model': {'model': 'uniform', 'params': {}},
            'p0_model': {'model': 'uniform', 'params': {}},
            'D': 3,
            'fr_model': [
                {"model": "linear", "params": {"slope": -5, "bias": 15}},
                {"model": "linear", "params": {"slope": 10, "bias": 100}},
                {"model": "linear", "params": {"slope": 10, "bias": 100}}
                ],
        }
        cls.hyperparam = {
            'ADAM_alpha': {'simultaneous': 0.1},
            'GD_lr': {
                'F': 0.05, 'F0': 0.05, 'D': 0.005, 'Fr': 0.0005, 'C': 0.2
                }
        }

        # Create temporary directory for test results
        cls.res_fold = 'unit_test_shared_opt_results'
        pathlib.Path(cls.res_fold).mkdir(parents=True, exist_ok=True)

        cls.ll_gt = {bm: [None]*2 for bm in cls.boundary_mode}
        cls.ll_gt_cv = {bm: [None]*2 for bm in cls.boundary_mode}

        # Counter of the test number
        cls.test_num = 0

    def SharedSimultaneousInferenceTesting(self, with_cuda):
        """Inference of all of the parameters together
        """
        device = 'CPU' if not with_cuda else 'GPU'
        for t_num, (optimizer, bm) in enumerate(
                product(self.optimizers, self.boundary_mode)
        ):
            with self.subTest(
                    f'Testing {optimizer}, boundary mode {bm}, simultaneous '
                    'inference',
                    optimizer=optimizer, bm=bm
            ):
                self.test_num += 1
                logger.info(
                    f'Running test {self.test_num}, optimizer = {optimizer}, '
                    f'boundary mode {bm}, with_cuda {with_cuda}'
                )

                # learning rate is optimizer-dependent
                if optimizer == 'ADAM':
                    lr = {
                        'alpha': self.hyperparam['ADAM_alpha']['simultaneous']
                    }
                else:
                    lr = self.hyperparam['GD_lr'].copy()

                # Optimization params
                opt_params = {
                    **self.opt_params,
                    'params_to_opt': list(self.hyperparam['GD_lr'].keys()),
                    'learning_rate': lr
                }

                init_model = model.new_model(
                    **self.initial_guess,
                    grid=self.grid_gpu if with_cuda else self.grid_cpu,
                    params_size=self.params_size,
                    with_cuda=with_cuda
                )

                optimization = Optimization(
                    self.dataTR[bm],
                    init_model,
                    optimizer,
                    opt_params,
                    self.ls_options_simultaneous_inference,
                    pde_solve_params=self.pde_solve_params,
                    boundary_mode=bm,
                    dataCV=self.dataCV[bm],
                    device=device
                )

                # run optimization
                optimization.run_optimization()

                # Compute ground-truth loglik
                em_gt = self.em_gt_gpu if with_cuda else self.em_gt_cpu
                for samp in range(2):
                    if self.ll_gt[bm][samp] is None:
                        self.ll_gt[bm][samp] = (
                            optimization.optimizer.gradient.get_grad_data(
                                optimization.optimizer.get_dataTR(samp),
                                em_gt,
                                samp
                            )
                        )
                    if (
                            self.dataCV[bm] is not None and
                            self.ll_gt_cv[bm][samp] is None
                            ):
                        self.ll_gt_cv[bm][samp] = (
                            optimization.optimizer.gradient.get_grad_data(
                                optimization.optimizer.get_dataCV(
                                    samp), em_gt, samp
                            )
                        )

                # Visualize the results
                PlotResults(
                    f'Opt = {optimizer}, boundary mode = {bm}',
                    ['F', 'Fr', 'F0', 'D', 'C'],
                    self.n_epochs_to_disp,
                    optimization.results,
                    self.em_gt_cpu,
                    self.ll_gt[bm],
                    self.ll_gt_cv[bm],
                    self.res_fold
                )

                if self.res_fold is not None:
                    logger.info(f'Test result saved into {self.res_fold}')

    def test_SharedSimultaneousInferenceCPU(self):
        self.SharedSimultaneousInferenceTesting(False)

    @unittest.skipIf(not GPU_support, 'cupy not installed')
    def test_SharedSimultaneousInferenceGPU(self):
        self.SharedSimultaneousInferenceTesting(True)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logging.getLogger().setLevel(logging.DEBUG)
    unittest.main()
