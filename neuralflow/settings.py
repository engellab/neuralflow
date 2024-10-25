#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Settings and default parameters
"""
import numpy as np

MINIMUM_PEQ = 1.0e-10

opt_settings = {
    'max_epochs': 100,
    'mini_batch_number': 1,
    'params_to_opt': ['F', 'F0', 'D', 'Fr', 'C'],
    'beta1': 0.9,
    'beta2': 0.99,
    'epsilon': 10**-8,
    'etaf': 0,
}

line_search_setting = {
    'max_fun_eval': 3,
    'epoch_schedule': np.array([]),
    'nSearchPerEpoch': 1
}


implemented_bms = ['absorbing', 'reflecting']
implemented_optimizers = ['ADAM', 'GD']


# Line search hyperparams
_cmin = 1
_cmax = 150
_dmin = 0.04
_dmax = 10
