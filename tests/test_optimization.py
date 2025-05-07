#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests optimization class. Note: it may take a lot of time (hours) to run
all of the tests.
"""

import logging
import unittest
from unittest.mock import patch
from neuralflow.model import model
from visualization import PlotResults
from neuralflow.data_generation import SyntheticData
from neuralflow.spike_data import SpikeData
from neuralflow.grid import GLLgrid
from neuralflow import firing_rate_models
from neuralflow.optimization import Optimization
from itertools import product
from copy import deepcopy
import pathlib
from pkg_resources import working_set
GPU_support = any([pkg.key.startswith('cupy') for pkg in working_set])
logger = logging.getLogger(__name__)


class TestOptmization(unittest.TestCase):
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

        cls.grid_cpu = GLLgrid(**cls.pde_solve_params)
        if GPU_support:
            cls.grid_gpu = GLLgrid(**cls.pde_solve_params, with_cuda=True)

        # Feel free to test different ground-truths
        cls.gt_param = {
            'peq_model': {'model': 'linear_pot', 'params': {'slope': -2}},
            'p0_model': {'model': 'single_well', 'params': {'miu': 100}},
            'D': 0.5,
            'fr_model': [
                {"model": "linear", "params": {"slope": 50, "bias": 60}},
                {"model": "linear", "params": {"slope": -50, "bias": 60}}
            ]
        }
        cls.em_gt_cpu = model.new_model(**cls.gt_param, grid=cls.grid_cpu)
        if GPU_support:
            cls.em_gt_gpu = model.new_model(
                **cls.gt_param,
                grid=cls.grid_gpu,
                with_cuda=True,
            )

        # Synthetic data generation
        with_cv = True
        num_training_trials = 50
        num_val_trials = max(num_training_trials // 10, 3)
        cls.dataTR = dict.fromkeys(cls.boundary_mode)
        cls.dataCV = dict.fromkeys(cls.boundary_mode)
        for bm in cls.boundary_mode:
            dg = SyntheticData(cls.em_gt_cpu, bm)
            logger.info(f'Generating {num_training_trials} trials of training '
                        f'data using {bm} boundary mode')
            data, _, _ = dg.generate_data(0, 1, num_training_trials, 0)
            cls.dataTR[bm] = SpikeData(data, 'ISIs', with_cuda=GPU_support)
            if with_cv:
                logger.info(f'Generating {num_val_trials} trials of validation'
                            f' data using {bm} boundary mode')
                dataCV, _, _ = dg.generate_data(0, 1, num_val_trials, 0)
                cls.dataCV[bm] = SpikeData(
                    dataCV, 'ISIs', with_cuda=GPU_support
                )

        # optimization
        cls.optimizers = ['ADAM', 'GD']
        cls.opt_params = {'max_epochs': 5, 'mini_batch_number': 10}
        cls.opt_params_ls = {'max_epochs': 1, 'mini_batch_number': 1}

        cls.ls_options = {
            'C_opt': {'epoch_schedule': [1]},
            'D_opt': {'epoch_schedule': [1]}
        }
        cls.ls_options_simultaneous_inference = {
            'C_opt': {'epoch_schedule': [1, 10]},
            'D_opt': {'epoch_schedule': [1, 10]}
        }

        # visualization
        cls.n_epochs_to_disp = 5

        cls.model_param_mapping = {
            'F': 'peq_model', 'F0': 'p0_model', 'D': 'D', 'Fr': 'fr_model',
            'C': 'fr_model'
            }
        cls.params_to_opt = ['F', 'F0', 'D', 'Fr', 'C']
        cls.initial_guess = {
            'F': {'model': 'uniform', 'params': {}},
            'F0': {'model': 'uniform', 'params': {}},
            'D': 3,
            'Fr': [{"model": "linear", "params": {"slope": -5, "bias": 15}},
                   {"model": "linear", "params": {"slope": 10, "bias": 100}}],
            'C': deepcopy(cls.gt_param['fr_model'])
        }
        cls.hyperparam = {
            'ADAM_alpha': {
                'F': 0.5, 'F0': 0.5, 'D': 0.1, 'Fr': 0.5, 'C': 0.5, 'all': 0.1
            },
            'GD_lr': {
                'F': 0.05, 'F0': 0.05, 'D': 0.005, 'Fr': 0.0005, 'C': 0.2
            }
        }

        # Create temporary directory for test results
        cls.res_fold = 'unit_test_opt_results'
        pathlib.Path(cls.res_fold).mkdir(parents=True, exist_ok=True)

        cls.ll_gt = dict.fromkeys(cls.boundary_mode)
        cls.ll_gt_cv = dict.fromkeys(cls.boundary_mode)

        # Counter of the test number
        cls.test_num = 0

    @staticmethod
    def Mock_adjust_firing_rate(self_var, param, device):
        """ To test FR gradient, mock the adjust_firing_rate function to
        correctly initialize firing rate: if we are optimizing Fr, then change
        shape but preserve C. If we are optimizing C, change C and preserve Fr
        """
        self_var.fr_av = [
            d.trial_average_fr() for d in self_var.optimizer.dataTR
        ]
        if device == 'CPU':
            model = self_var.optimizer.model
            grid = model.grid
        else:
            model = self_var.optimizer.model.cuda_var
            grid = self_var.optimizer.model.grid.cuda_var
        if param == 'Fr':
            # Make sure to preserve Ground-truth C, only change Fr
            model.fr[0, :, 0] = firing_rate_models.linear(grid.x_d, 0, 10)
            model.fr[0, :, 1] = firing_rate_models.linear(grid.x_d, 0, 110)
            model.C[0] = model.fr[0][0, ...].copy()
        elif param == 'C':
            # scale by a constant to have the correct shape but wrong C
            model.fr[0, :, 0] = model.fr[0, :, 0]*2
            model.fr[0, :, 1] = model.fr[0, :, 1]/2

    def SimultaneousInferenceTesting(self, with_cuda):
        """Inference of all of the parameters together - like in real data
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
                    f'boundary mode {bm} '
                )

                # learning rate is optimizer-dependent
                if optimizer == 'ADAM':
                    lr = {'alpha': self.hyperparam['ADAM_alpha']['all']}
                else:
                    lr = self.hyperparam['GD_lr'].copy()

                # Optimization params
                opt_params = {
                    **self.opt_params,
                    'params_to_opt': list(self.params_to_opt),
                    'learning_rate': lr
                }

                # initial guess
                model_param = deepcopy(self.gt_param)
                param_list = list(self.params_to_opt)
                if 'C' in param_list and 'Fr' in param_list:
                    param_list.remove('C')
                for param in param_list:
                    model_param[self.model_param_mapping[param]] = (
                        deepcopy(self.initial_guess[param])
                    )
                init_model = model.new_model(
                    **model_param,
                    grid=self.grid_gpu if with_cuda else self.grid_cpu,
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
                if self.ll_gt[bm] is None:
                    self.ll_gt[bm] = [
                        optimization.optimizer.gradient.get_grad_data(
                            optimization.optimizer.get_dataTR(0), em_gt, 0
                        )
                    ]
                    if with_cuda:
                        self.ll_gt[bm][0] = optimization.lib.asnumpy(
                            self.ll_gt[bm][0]
                        )
                if self.dataCV[bm] is not None and self.ll_gt_cv[bm] is None:
                    self.ll_gt_cv[bm] = [
                        optimization.optimizer.gradient.get_grad_data(
                            optimization.optimizer.get_dataCV(0), em_gt, 0
                        )
                    ]
                    if with_cuda:
                        self.ll_gt_cv[bm][0] = optimization.lib.asnumpy(
                            self.ll_gt_cv[bm][0]
                        )

                # Visualize the results
                PlotResults(
                    f'{device} Opt = {optimizer}, boundary mode = {bm}',
                    self.params_to_opt,
                    self.n_epochs_to_disp,
                    optimization.results,
                    self.em_gt_cpu,
                    self.ll_gt[bm],
                    self.ll_gt_cv[bm],
                    self.res_fold
                )
                if self.res_fold is not None:
                    logger.info(f'Test result saved into {self.res_fold}')

    def IndividualGradOptimizationTesting(self, with_cuda):
        """ Optimize 1 parameter at a time, providing the ground-truth for all
        other parameters.
        """
        device = 'CPU' if not with_cuda else 'GPU'
        # Optimize driving force
        for t_num, (optimizer, bm, param) in enumerate(
                product(
                    self.optimizers, self.boundary_mode, self.params_to_opt
                )
        ):
            with self.subTest(
                    f'Testing {optimizer}, boundary mode {bm}, param {param}',
                    optimizer=optimizer, bm=bm, param=param
            ):
                self.test_num += 1
                logger.info(
                    f'Running test {self.test_num}, optimizer = {optimizer}, '
                    f'boundary mode {bm}, param = {param}'
                )

                # learning rate is optimizer-dependent
                if optimizer == 'ADAM':
                    lr = {'alpha': self.hyperparam['ADAM_alpha'][param]}
                else:
                    lr = {param: self.hyperparam['GD_lr'][param]}

                # Optimization params
                opt_params = {
                    **self.opt_params,
                    'params_to_opt': [param],
                    'learning_rate': lr
                }

                # initial guess is the same as ground-truth except for param
                model_param = deepcopy(self.gt_param)
                model_param[self.model_param_mapping[param]] = (
                    self.initial_guess[param]
                )
                init_model = model.new_model(
                    **model_param,
                    grid=self.grid_gpu if with_cuda else self.grid_cpu,
                    with_cuda=with_cuda
                )

                with patch.object(
                    Optimization, "_adjust_firing_rate",
                    lambda x: TestOptmization.Mock_adjust_firing_rate(
                        x, param, device
                    )
                ):
                    optimization = Optimization(
                        self.dataTR[bm],
                        init_model,
                        optimizer,
                        opt_params,
                        pde_solve_params=self.pde_solve_params,
                        boundary_mode=bm,
                        dataCV=self.dataCV[bm],
                        device=device
                    )
                # run optimization
                optimization.run_optimization()

                # Compute ground-truth loglik
                em_gt = self.em_gt_gpu if with_cuda else self.em_gt_cpu
                if self.ll_gt[bm] is None:
                    self.ll_gt[bm] = [
                        optimization.optimizer.gradient.get_grad_data(
                            optimization.optimizer.get_dataTR(0), em_gt, 0
                        )
                    ]
                    if with_cuda:
                        self.ll_gt[bm][0] = optimization.lib.asnumpy(
                            self.ll_gt[bm][0]
                        )

                if self.dataCV[bm] is not None and self.ll_gt_cv[bm] is None:
                    self.ll_gt_cv[bm] = [
                        optimization.optimizer.gradient.get_grad_data(
                            optimization.optimizer.get_dataCV(0), em_gt, 0
                        )
                    ]
                    if with_cuda:
                        self.ll_gt_cv[bm][0] = optimization.lib.asnumpy(
                            self.ll_gt_cv[bm][0]
                        )

                # Visualize the results
                PlotResults(
                    f'{device} Opt = {optimizer}, boundary mode = {bm}',
                    param,
                    self.n_epochs_to_disp,
                    optimization.results,
                    self.em_gt_cpu,
                    self.ll_gt[bm],
                    self.ll_gt_cv[bm],
                    self.res_fold
                )

                if self.res_fold is not None:
                    logger.info(f'Test result saved into {self.res_fold}')

    def LineSearchTesting(self, with_cuda):
        """ Perform the line search of C and D
        """
        params = [p for p in self.params_to_opt if p in ['C', 'D']]
        for t_num, (optimizer, bm, param) in enumerate(
            product(
                self.optimizers, self.boundary_mode, params
            )
        ):

            device = 'CPU' if not with_cuda else 'GPU'
            self.test_num += 1
            logger.info(
                f'Running linesearch test {self.test_num}, optimizer = '
                f'{optimizer}, boundary mode {bm}, param = {param}'
            )

            # learning rate is optimizer-dependent
            if optimizer == 'ADAM':
                lr = {'alpha': self.hyperparam['ADAM_alpha'][param]}
            else:
                lr = {param: self.hyperparam['GD_lr'][param]}
            # Optimization params
            opt_params = {
                **self.opt_params_ls,
                'params_to_opt': [param],
                'learning_rate': lr
            }
            # initial guess is the same as ground-truth except for param
            model_param = deepcopy(self.gt_param)
            model_param[self.model_param_mapping[param]] = (
                self.initial_guess[param]
            )
            init_model = model.new_model(
                **model_param,
                grid=self.grid_gpu if with_cuda else self.grid_cpu,
                with_cuda=with_cuda
            )
            with patch.object(
                Optimization, "_adjust_firing_rate",
                lambda x: TestOptmization.Mock_adjust_firing_rate(
                    x, param, device
                )
            ):

                optimization = Optimization(
                    self.dataTR[bm],
                    init_model,
                    optimizer,
                    opt_params,
                    self.ls_options,
                    pde_solve_params=self.pde_solve_params,
                    boundary_mode=bm,
                    dataCV=self.dataCV[bm],
                    device=device
                )
            # run optimization
            optimization.run_optimization()

            # Compute ground-truth loglik
            em_gt = self.em_gt_gpu if with_cuda else self.em_gt_cpu
            if self.ll_gt[bm] is None:
                self.ll_gt[bm] = [
                    optimization.optimizer.gradient.get_grad_data(
                        optimization.optimizer.get_dataTR(0), em_gt, 0
                    )
                ]
                if with_cuda:
                    self.ll_gt[bm][0] = optimization.lib.asnumpy(
                        self.ll_gt[bm][0]
                    )
            if self.dataCV[bm] is not None and self.ll_gt_cv[bm] is None:
                self.ll_gt_cv[bm] = [
                    optimization.optimizer.gradient.get_grad_data(
                        optimization.optimizer.get_dataCV(0), em_gt, 0
                    )
                ]
                if with_cuda:
                    self.ll_gt_cv[bm][0] = optimization.lib.asnumpy(
                        self.ll_gt_cv[bm][0]
                    )
            PlotResults(
                f'{device} Line search opt = {optimizer}, boundary mode = '
                f'{bm}',
                param, 1, optimization.results, self.em_gt_cpu,
                self.ll_gt[bm], self.ll_gt_cv[bm], self.res_fold
            )
            if self.res_fold is not None:
                logger.info(f'Test result saved into {self.res_fold}')

    def test_SimultaneousInferenceCPU(self):
        self.SimultaneousInferenceTesting(False)

    def test_LineSearchCPU(self):
        self.LineSearchTesting(False)

    def test_IndividualGradOptimizationCPU(self):
        self.IndividualGradOptimizationTesting(False)

    @unittest.skipIf(not GPU_support, 'cupy not installed')
    def test_SimultaneousInferenceGPU(self):
        self.SimultaneousInferenceTesting(True)

    @unittest.skipIf(not GPU_support, 'cupy not installed')
    def test_LineSearchGPU(self):
        self.LineSearchTesting(True)

    @unittest.skipIf(not GPU_support, 'cupy not installed')
    def test_IndividualGradOptimizationGPU(self):
        self.IndividualGradOptimizationTesting(True)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logging.getLogger().setLevel(logging.INFO)
    unittest.main()
