#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Settings and default parameters
"""
import numpy as np

# Important: there are allowed min and max values
# During optimization, the updated values will be
# clipped to these numbers. These values may be
# changed depending on the situation
MINIMUM_PEQ = 1.0e-5
MINIMUM_D = 0.01

# Default optimization settings
opt_settings = {
    'max_epochs': 100,
    'mini_batch_number': 1,
    'params_to_opt': ['F', 'F0', 'D', 'Fr', 'C'],
    'beta1': 0.9,
    'beta2': 0.99,
    'epsilon': 10**-8,
    'etaf': 0,
}

# Default LS settings
line_search_setting = {
    'max_fun_eval': 3,
    'epoch_schedule': np.array([]),
    'nSearchPerEpoch': 1
}

# For checking parameters and thoughing exceptions
implemented_bms = ['absorbing', 'reflecting']
implemented_optimizers = ['ADAM', 'GD']

# Line search hyperparams
_cmin = 1
_cmax = 150
_dmin = 0.01
_dmax = 10
