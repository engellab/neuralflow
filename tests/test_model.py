# -*- coding: utf-8 -*-
"""
Unit tests for the refactored model.py with condition_mapping support.
Tests cover:
- Basic model creation with condition_mapping
- Backward compatibility with params_size
- rho0 computation with unique (peq, p0) combinations
- Parameter retrieval methods
- Per-condition neuron counts
- Validation of condition_mapping
"""

import logging
import unittest
import numpy as np
from copy import deepcopy
from neuralflow.model import model, _convert_params_size_to_condition_mapping, _validate_condition_mapping
from neuralflow.grid import GLLgrid


class TestConditionMapping(unittest.TestCase):
    """Tests for the new condition_mapping functionality."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.grid = GLLgrid(xbegin=-1, xend=1, Np=8, Ne=16)
        cls.N = cls.grid.N
        
        # Create some test distributions
        cls.peq_uniform = np.ones(cls.N) / (cls.grid.xend - cls.grid.xbegin)
        cls.p0_uniform = np.ones(cls.N) / (cls.grid.xend - cls.grid.xbegin)
        
        # Create non-uniform peq for testing
        x = cls.grid.x_d
        cls.peq_gaussian = np.exp(-x**2 / 0.1)
        cls.peq_gaussian /= np.sum(cls.peq_gaussian * cls.grid.w_d)
        
        cls.p0_gaussian = np.exp(-(x - 0.3)**2 / 0.1)
        cls.p0_gaussian /= np.sum(cls.p0_gaussian * cls.grid.w_d)
        
        # Create firing rates
        cls.fr_linear = np.outer(np.linspace(0.5, 1.5, 5), np.ones(cls.N)).T  # shape (N, 5)
        cls.fr_constant = np.ones((cls.N, 3)) * 10

    def test_convert_params_size_to_condition_mapping(self):
        """Test conversion from old params_size to new condition_mapping."""
        params_size = {'peq': 2, 'p0': 1, 'D': 1, 'fr': 1}
        num_models = 2
        num_neuron = 3
        
        cm = _convert_params_size_to_condition_mapping(params_size, num_models, num_neuron)
        
        self.assertEqual(cm['num_conditions'], 2)
        self.assertEqual(cm['num_peq'], 2)
        self.assertEqual(cm['num_p0'], 1)
        self.assertEqual(cm['num_D'], 1)
        self.assertEqual(cm['num_fr'], 3)  # fr * num_neuron
        self.assertEqual(len(cm['peq_indices']), 2)
        self.assertEqual(len(cm['p0_indices']), 2)
        self.assertEqual(len(cm['D_indices']), 2)
        self.assertEqual(len(cm['fr_indices']), 2)
        
        # Check sharing logic
        self.assertEqual(cm['peq_indices'], [0, 1])  # Different peq for each condition
        self.assertEqual(cm['p0_indices'], [0, 0])   # Shared p0
        self.assertEqual(cm['D_indices'], [0, 0])    # Shared D

    def test_validate_condition_mapping_valid(self):
        """Test validation with valid condition_mapping."""
        cm = {
            'num_conditions': 2,
            'num_peq': 1,
            'num_p0': 1,
            'num_D': 1,
            'num_fr': 5,
            'peq_indices': [0, 0],
            'p0_indices': [0, 0],
            'D_indices': [0, 0],
            'fr_indices': [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]]
        }
        
        # Should not raise any error
        _validate_condition_mapping(cm, (1, self.N), (1, self.N), (1,), (5,))

    def test_validate_condition_mapping_missing_key(self):
        """Test validation with missing required key."""
        cm = {
            'num_conditions': 2,
            'num_peq': 1,
            # Missing other keys
        }
        
        with self.assertRaises(ValueError) as context:
            _validate_condition_mapping(cm, (1, self.N), (1, self.N), (1,), (5,))
        
        self.assertIn('missing required key', str(context.exception))

    def test_validate_condition_mapping_out_of_bounds(self):
        """Test validation with out of bounds indices."""
        cm = {
            'num_conditions': 2,
            'num_peq': 1,
            'num_p0': 1,
            'num_D': 1,
            'num_fr': 3,
            'peq_indices': [0, 1],  # Index 1 is out of bounds for num_peq=1
            'p0_indices': [0, 0],
            'D_indices': [0, 0],
            'fr_indices': [[0, 1, 2], [0, 1, 2]]
        }
        
        with self.assertRaises(ValueError) as context:
            _validate_condition_mapping(cm, (1, self.N), (1, self.N), (1,), (3,))
        
        self.assertIn('out of bounds', str(context.exception))

    def test_validate_condition_mapping_wrong_length(self):
        """Test validation with wrong indices length."""
        cm = {
            'num_conditions': 2,
            'num_peq': 1,
            'num_p0': 1,
            'num_D': 1,
            'num_fr': 3,
            'peq_indices': [0],  # Wrong length (should be 2)
            'p0_indices': [0, 0],
            'D_indices': [0, 0],
            'fr_indices': [[0, 1, 2], [0, 1, 2]]
        }
        
        with self.assertRaises(ValueError) as context:
            _validate_condition_mapping(cm, (1, self.N), (1, self.N), (1,), (3,))
        
        self.assertIn('must equal num_conditions', str(context.exception))


class TestModelCreation(unittest.TestCase):
    """Tests for model creation with condition_mapping."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.grid = GLLgrid(xbegin=-1, xend=1, Np=8, Ne=16)
        cls.N = cls.grid.N
        
        # Create normalized distributions
        x = cls.grid.x_d
        cls.peq1 = np.exp(-x**2 / 0.1)
        cls.peq1 /= np.sum(cls.peq1 * cls.grid.w_d)
        
        cls.peq2 = np.exp(-(x - 0.5)**2 / 0.1)
        cls.peq2 /= np.sum(cls.peq2 * cls.grid.w_d)
        
        cls.p01 = np.exp(-x**2 / 0.15)
        cls.p01 /= np.sum(cls.p01 * cls.grid.w_d)
        
        cls.p02 = np.exp(-(x + 0.3)**2 / 0.15)
        cls.p02 /= np.sum(cls.p02 * cls.grid.w_d)
        
        # Firing rates: shape (num_fr, N)
        cls.fr = np.random.rand(10, cls.N) + 0.1

    def test_model_creation_basic(self):
        """Test basic model creation with condition_mapping."""
        peq = np.vstack([self.peq1, self.peq2])  # (2, N)
        p0 = np.vstack([self.p01, self.p02])     # (2, N)
        D = np.array([0.1, 0.2])
        
        condition_mapping = {
            'num_conditions': 2,
            'num_peq': 2,
            'num_p0': 2,
            'num_D': 2,
            'num_fr': 10,
            'peq_indices': [0, 1],
            'p0_indices': [0, 1],
            'D_indices': [0, 1],
            'fr_indices': [
                [0, 1, 2, 3, 4],      # 5 neurons for condition 0
                [5, 6, 7, 8, 9]       # 5 neurons for condition 1
            ]
        }
        
        m = model(peq, p0, D, self.fr, condition_mapping, self.grid)
        
        self.assertEqual(m.num_models, 2)
        self.assertEqual(m.num_neuron, 10)
        self.assertEqual(m.get_num_neurons(0), 5)
        self.assertEqual(m.get_num_neurons(1), 5)
        self.assertTrue(m.non_equilibrium)

    def test_model_creation_different_neuron_counts(self):
        """Test model with different neuron counts per condition."""
        peq = self.peq1[np.newaxis, :]  # (1, N)
        p0 = np.vstack([self.p01, self.p02])  # (2, N)
        D = np.array([0.1, 0.2])
        
        condition_mapping = {
            'num_conditions': 2,
            'num_peq': 1,
            'num_p0': 2,
            'num_D': 2,
            'num_fr': 10,
            'peq_indices': [0, 0],     # Shared peq
            'p0_indices': [0, 1],
            'D_indices': [0, 1],
            'fr_indices': [
                [0, 1, 2, 3, 4],             # 5 neurons for condition 0
                [5, 6, 7, 8, 9, 10]          # Wrong! only 10 fr models
            ]
        }
        
        # This should fail due to fr index 10 being out of bounds
        with self.assertRaises(ValueError):
            model(peq, p0, D, self.fr, condition_mapping, self.grid)
        
        # Correct version
        condition_mapping['fr_indices'] = [
            [0, 1, 2, 3, 4],           # 5 neurons for condition 0
            [3, 4, 5, 6, 7, 8, 9]      # 7 neurons for condition 1 (with overlap)
        ]
        
        m = model(peq, p0, D, self.fr, condition_mapping, self.grid)
        
        self.assertEqual(m.get_num_neurons(0), 5)
        self.assertEqual(m.get_num_neurons(1), 7)
        self.assertEqual(m.num_neuron, 10)  # Maximum

    def test_model_creation_equilibrium(self):
        """Test equilibrium model creation (p0=None)."""
        peq = self.peq1[np.newaxis, :]
        D = np.array([0.1])
        
        condition_mapping = {
            'num_conditions': 1,
            'num_peq': 1,
            'num_p0': 1,
            'num_D': 1,
            'num_fr': 5,
            'peq_indices': [0],
            'p0_indices': [0],
            'D_indices': [0],
            'fr_indices': [[0, 1, 2, 3, 4]]
        }
        
        m = model(peq, None, D, self.fr[:5, :], condition_mapping, self.grid)
        
        self.assertFalse(m.non_equilibrium)
        np.testing.assert_array_almost_equal(m.p0[0, :], peq[0, :])


class TestRho0Computation(unittest.TestCase):
    """Tests for correct rho0 computation with unique (peq, p0) pairs."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.grid = GLLgrid(xbegin=-1, xend=1, Np=8, Ne=16)
        cls.N = cls.grid.N
        
        # Create normalized distributions
        x = cls.grid.x_d
        cls.peq1 = np.exp(-x**2 / 0.1)
        cls.peq1 /= np.sum(cls.peq1 * cls.grid.w_d)
        
        cls.peq2 = np.exp(-(x - 0.5)**2 / 0.1)
        cls.peq2 /= np.sum(cls.peq2 * cls.grid.w_d)
        
        cls.p01 = np.exp(-x**2 / 0.15)
        cls.p01 /= np.sum(cls.p01 * cls.grid.w_d)
        
        cls.p02 = np.exp(-(x + 0.3)**2 / 0.15)
        cls.p02 /= np.sum(cls.p02 * cls.grid.w_d)
        
        cls.fr = np.random.rand(10, cls.N) + 0.1

    def test_rho0_unique_pairs(self):
        """Test that rho0 is computed for each unique (peq, p0) pair."""
        # 4 conditions with peq_indices = [0, 1, 0, 1], p0_indices = [1, 0, 0, 1]
        # Unique pairs: (0,1), (1,0), (0,0), (1,1) = 4 unique rho0
        
        peq = np.vstack([self.peq1, self.peq2])  # 2 peq models
        p0 = np.vstack([self.p01, self.p02])      # 2 p0 models
        D = np.array([0.1, 0.1, 0.1, 0.1])
        
        condition_mapping = {
            'num_conditions': 4,
            'num_peq': 2,
            'num_p0': 2,
            'num_D': 4,
            'num_fr': 10,
            'peq_indices': [0, 1, 0, 1],
            'p0_indices': [1, 0, 0, 1],
            'D_indices': [0, 1, 2, 3],
            'fr_indices': [[i for i in range(5)] for _ in range(4)]
        }
        
        m = model(peq, p0, D, self.fr, condition_mapping, self.grid)
        
        # Should have 4 unique rho0 models
        self.assertEqual(m.condition_mapping['num_rho0'], 4)
        self.assertEqual(m.rho0.shape[0], 4)
        
        # Check that rho0_indices correctly maps each condition
        rho0_indices = m.condition_mapping['rho0_indices']
        self.assertEqual(len(rho0_indices), 4)
        
        # Each unique pair should have a unique rho0 index
        unique_pairs = set()
        for i in range(4):
            pair = (condition_mapping['peq_indices'][i], condition_mapping['p0_indices'][i])
            unique_pairs.add(pair)
        self.assertEqual(len(unique_pairs), 4)

    def test_rho0_shared_pairs(self):
        """Test rho0 when conditions share (peq, p0) pairs."""
        # 4 conditions with peq_indices = [0, 0, 1, 1], p0_indices = [0, 0, 1, 1]
        # Unique pairs: (0,0), (1,1) = 2 unique rho0
        
        peq = np.vstack([self.peq1, self.peq2])
        p0 = np.vstack([self.p01, self.p02])
        D = np.array([0.1, 0.1, 0.1, 0.1])
        
        condition_mapping = {
            'num_conditions': 4,
            'num_peq': 2,
            'num_p0': 2,
            'num_D': 4,
            'num_fr': 10,
            'peq_indices': [0, 0, 1, 1],
            'p0_indices': [0, 0, 1, 1],
            'D_indices': [0, 1, 2, 3],
            'fr_indices': [[i for i in range(5)] for _ in range(4)]
        }
        
        m = model(peq, p0, D, self.fr, condition_mapping, self.grid)
        
        # Should have only 2 unique rho0 models
        self.assertEqual(m.condition_mapping['num_rho0'], 2)
        self.assertEqual(m.rho0.shape[0], 2)
        
        # Conditions 0 and 1 should share the same rho0 index
        # Conditions 2 and 3 should share the same rho0 index
        rho0_indices = m.condition_mapping['rho0_indices']
        self.assertEqual(rho0_indices[0], rho0_indices[1])
        self.assertEqual(rho0_indices[2], rho0_indices[3])
        self.assertNotEqual(rho0_indices[0], rho0_indices[2])

    def test_rho0_correct_values(self):
        """Test that rho0 values are correctly computed as p0/sqrt(peq)."""
        peq = self.peq1[np.newaxis, :]
        p0 = self.p01[np.newaxis, :]
        D = np.array([0.1])
        
        condition_mapping = {
            'num_conditions': 1,
            'num_peq': 1,
            'num_p0': 1,
            'num_D': 1,
            'num_fr': 5,
            'peq_indices': [0],
            'p0_indices': [0],
            'D_indices': [0],
            'fr_indices': [[0, 1, 2, 3, 4]]
        }
        
        m = model(peq, p0, D, self.fr[:5, :], condition_mapping, self.grid)
        
        # Compute expected rho0
        expected_rho0 = p0[0, :] / np.maximum(np.sqrt(peq[0, :]), 1e-20)
        
        np.testing.assert_array_almost_equal(m.rho0[0, :], expected_rho0)


class TestParameterRetrieval(unittest.TestCase):
    """Tests for get_params and get_params_for_grad methods."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.grid = GLLgrid(xbegin=-1, xend=1, Np=8, Ne=16)
        cls.N = cls.grid.N
        
        x = cls.grid.x_d
        cls.peq1 = np.exp(-x**2 / 0.1)
        cls.peq1 /= np.sum(cls.peq1 * cls.grid.w_d)
        
        cls.peq2 = np.exp(-(x - 0.5)**2 / 0.1)
        cls.peq2 /= np.sum(cls.peq2 * cls.grid.w_d)
        
        cls.p01 = np.exp(-x**2 / 0.15)
        cls.p01 /= np.sum(cls.p01 * cls.grid.w_d)
        
        cls.p02 = np.exp(-(x + 0.3)**2 / 0.15)
        cls.p02 /= np.sum(cls.p02 * cls.grid.w_d)
        
        # Create firing rates with known values
        cls.fr = np.zeros((10, cls.N))
        for i in range(10):
            cls.fr[i, :] = (i + 1) * np.ones(cls.N)

    def test_get_params_basic(self):
        """Test basic parameter retrieval."""
        peq = np.vstack([self.peq1, self.peq2])
        p0 = np.vstack([self.p01, self.p02])
        D = np.array([0.1, 0.2])
        
        condition_mapping = {
            'num_conditions': 2,
            'num_peq': 2,
            'num_p0': 2,
            'num_D': 2,
            'num_fr': 10,
            'peq_indices': [0, 1],
            'p0_indices': [0, 1],
            'D_indices': [0, 1],
            'fr_indices': [
                [0, 1, 2, 3, 4],      # 5 neurons
                [5, 6, 7, 8, 9]       # 5 different neurons
            ]
        }
        
        m = model(peq, p0, D, self.fr, condition_mapping, self.grid)
        
        # Test condition 0
        peq_0, p0_0, D_0, fr_0 = m.get_params(0)
        # Note: peq gets clipped with MINIMUM_PEQ, so compare against stored peq
        np.testing.assert_array_almost_equal(peq_0, m.peq[0, :])
        np.testing.assert_array_almost_equal(p0_0, m.p0[0, :])
        self.assertEqual(D_0, 0.1)
        self.assertEqual(fr_0.shape, (self.N, 5))  # (grid.N, num_neurons)
        
        # Test condition 1
        peq_1, p0_1, D_1, fr_1 = m.get_params(1)
        np.testing.assert_array_almost_equal(peq_1, m.peq[1, :])
        np.testing.assert_array_almost_equal(p0_1, m.p0[1, :])
        self.assertEqual(D_1, 0.2)

    def test_get_params_shared(self):
        """Test parameter retrieval with shared parameters."""
        peq = self.peq1[np.newaxis, :]  # Shared peq
        p0 = self.p01[np.newaxis, :]    # Shared p0
        D = np.array([0.1])              # Shared D
        
        condition_mapping = {
            'num_conditions': 2,
            'num_peq': 1,
            'num_p0': 1,
            'num_D': 1,
            'num_fr': 10,
            'peq_indices': [0, 0],
            'p0_indices': [0, 0],
            'D_indices': [0, 0],
            'fr_indices': [
                [0, 1, 2, 3, 4],
                [5, 6, 7, 8, 9]
            ]
        }
        
        m = model(peq, p0, D, self.fr, condition_mapping, self.grid)
        
        # Both conditions should return same peq, p0, D
        peq_0, p0_0, D_0, fr_0 = m.get_params(0)
        peq_1, p0_1, D_1, fr_1 = m.get_params(1)
        
        np.testing.assert_array_almost_equal(peq_0, peq_1)
        np.testing.assert_array_almost_equal(p0_0, p0_1)
        self.assertEqual(D_0, D_1)

    def test_get_params_for_grad(self):
        """Test get_params_for_grad returns correct rho0."""
        peq = np.vstack([self.peq1, self.peq2])
        p0 = np.vstack([self.p01, self.p02])
        D = np.array([0.1, 0.2])
        
        condition_mapping = {
            'num_conditions': 2,
            'num_peq': 2,
            'num_p0': 2,
            'num_D': 2,
            'num_fr': 5,
            'peq_indices': [0, 1],
            'p0_indices': [0, 1],
            'D_indices': [0, 1],
            'fr_indices': [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]]
        }
        
        m = model(peq, p0, D, self.fr[:5, :], condition_mapping, self.grid)
        
        peq_0, rho0_0, D_0, fr_0 = m.get_params_for_grad(0)
        
        # rho0 should be p0 / sqrt(peq)
        expected_rho0 = self.p01 / np.maximum(np.sqrt(self.peq1), 1e-20)
        np.testing.assert_array_almost_equal(rho0_0, expected_rho0)

    def test_get_param_indices(self):
        """Test get_param_indices method."""
        peq = np.vstack([self.peq1, self.peq2])
        p0 = np.vstack([self.p01, self.p02])
        D = np.array([0.1, 0.2])
        
        condition_mapping = {
            'num_conditions': 2,
            'num_peq': 2,
            'num_p0': 2,
            'num_D': 2,
            'num_fr': 10,
            'peq_indices': [0, 1],
            'p0_indices': [0, 1],
            'D_indices': [0, 1],
            'fr_indices': [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
        }
        
        m = model(peq, p0, D, self.fr, condition_mapping, self.grid)
        
        indices_0 = m.get_param_indices(0)
        self.assertEqual(indices_0['peq_idx'], 0)
        self.assertEqual(indices_0['p0_idx'], 0)
        self.assertEqual(indices_0['D_idx'], 0)
        self.assertEqual(indices_0['fr_indices'], [0, 1, 2, 3, 4])
        
        indices_1 = m.get_param_indices(1)
        self.assertEqual(indices_1['peq_idx'], 1)
        self.assertEqual(indices_1['p0_idx'], 1)
        self.assertEqual(indices_1['D_idx'], 1)


class TestBackwardCompatibility(unittest.TestCase):
    """Tests for backward compatibility with old params_size format."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.grid = GLLgrid(xbegin=-1, xend=1, Np=8, Ne=16)
        cls.N = cls.grid.N
        
        x = cls.grid.x_d
        cls.peq = np.exp(-x**2 / 0.1)
        cls.peq /= np.sum(cls.peq * cls.grid.w_d)
        
        cls.p0 = np.exp(-x**2 / 0.15)
        cls.p0 /= np.sum(cls.p0 * cls.grid.w_d)
        
        cls.fr = np.random.rand(5, cls.N) + 0.1

    def test_params_size_backward_compat(self):
        """Test that old params_size format still works via new_model."""
        # Create model using old params_size format
        m = model.new_model(
            peq_model={'model': 'uniform', 'params': {}},
            p0_model={'model': 'uniform', 'params': {}},
            D=0.1,
            fr_model=[
                {"model": "linear", "params": {"slope": 50, "bias": 60}},
                {"model": "linear", "params": {"slope": -50, "bias": 60}}
            ],
            grid=self.grid,
            params_size={'peq': 1, 'p0': 1, 'D': 1, 'fr': 1}
        )
        
        # Check that params_size is available
        self.assertIn('peq', m.params_size)
        self.assertIn('p0', m.params_size)
        self.assertIn('D', m.params_size)
        self.assertIn('fr', m.params_size)
        
        # Check that condition_mapping was created
        self.assertIsNotNone(m.condition_mapping)
        self.assertEqual(m.condition_mapping['num_conditions'], 1)

    def test_num_models_attribute(self):
        """Test that num_models equals num_conditions."""
        condition_mapping = {
            'num_conditions': 3,
            'num_peq': 1,
            'num_p0': 1,
            'num_D': 1,
            'num_fr': 5,
            'peq_indices': [0, 0, 0],
            'p0_indices': [0, 0, 0],
            'D_indices': [0, 0, 0],
            'fr_indices': [[0, 1, 2, 3, 4] for _ in range(3)]
        }
        
        peq = self.peq[np.newaxis, :]
        p0 = self.p0[np.newaxis, :]
        D = np.array([0.1])
        
        m = model(peq, p0, D, self.fr, condition_mapping, self.grid)
        
        self.assertEqual(m.num_models, 3)


class TestNewModelFactory(unittest.TestCase):
    """Tests for the new_model factory method."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.grid = GLLgrid(xbegin=-1, xend=1, Np=8, Ne=16)

    def test_new_model_with_condition_mapping(self):
        """Test new_model with explicit condition_mapping."""
        m = model.new_model(
            peq_model={'model': 'uniform', 'params': {}},
            p0_model={'model': 'uniform', 'params': {}},
            D=0.1,
            fr_model=[
                {"model": "linear", "params": {"slope": 50, "bias": 60}},
                {"model": "linear", "params": {"slope": -50, "bias": 60}}
            ],
            grid=self.grid,
            condition_mapping={
                'num_conditions': 1,
                'num_peq': 1,
                'num_p0': 1,
                'num_D': 1,
                'num_fr': 2,
                'peq_indices': [0],
                'p0_indices': [0],
                'D_indices': [0],
                'fr_indices': [[0, 1]]
            }
        )
        
        self.assertEqual(m.num_models, 1)
        self.assertEqual(m.num_neuron, 2)

    def test_new_model_with_arrays(self):
        """Test new_model with arrays instead of model specs."""
        N = self.grid.N
        peq = np.ones(N) / 2
        p0 = np.ones(N) / 2
        D = 0.1
        fr = np.ones((N, 3)) * 10  # 3 neurons
        
        m = model.new_model(
            peq_model=peq,
            p0_model=p0,
            D=D,
            fr_model=fr,
            grid=self.grid
        )
        
        self.assertEqual(m.num_neuron, 3)
        self.assertEqual(m.num_models, 1)

    def test_new_model_equilibrium(self):
        """Test new_model creates equilibrium model when p0_model is None."""
        m = model.new_model(
            peq_model={'model': 'uniform', 'params': {}},
            p0_model=None,
            D=0.1,
            fr_model={"model": "linear", "params": {"slope": 0, "bias": 10}},
            grid=self.grid
        )
        
        self.assertFalse(m.non_equilibrium)


class TestNewModelMultipleConditions(unittest.TestCase):
    """Tests for new_model with multiple conditions and various sharing patterns."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.grid = GLLgrid(xbegin=-1, xend=1, Np=8, Ne=16)
        cls.N = cls.grid.N

    def test_new_model_shared_peq_p0_d_different_fr(self):
        """Test model with shared peq/p0/D but different (non-overlapping) fr per condition."""
        m = model.new_model(
            peq_model={'model': 'uniform', 'params': {}},
            p0_model={'model': 'uniform', 'params': {}},
            D=0.1,
            fr_model=[
                {"model": "linear", "params": {"slope": 50, "bias": 60}},
                {"model": "linear", "params": {"slope": -50, "bias": 60}},
                {"model": "linear", "params": {"slope": 30, "bias": 40}},
                {"model": "linear", "params": {"slope": -30, "bias": 40}},
            ],
            grid=self.grid,
            condition_mapping={
                'num_conditions': 2,
                'num_peq': 1,
                'num_p0': 1,
                'num_D': 1,
                'num_fr': 4,
                'peq_indices': [0, 0],
                'p0_indices': [0, 0],
                'D_indices': [0, 0],
                'fr_indices': [
                    [0, 1],      # condition 0: neurons 0, 1
                    [2, 3]       # condition 1: neurons 2, 3
                ]
            }
        )
        
        self.assertEqual(m.num_models, 2)
        self.assertEqual(m.num_neuron, 4)
        self.assertEqual(m.get_num_neurons(0), 2)
        self.assertEqual(m.get_num_neurons(1), 2)
        
        # Check that peq is shared
        peq_0, p0_0, D_0, fr_0 = m.get_params(0)
        peq_1, p1_1, D_1, fr_1 = m.get_params(1)
        np.testing.assert_array_almost_equal(peq_0, peq_1)
        np.testing.assert_array_almost_equal(p0_0, p1_1)
        self.assertEqual(D_0, D_1)
        
        # Check that fr are different (different slopes at the same x position)
        # At x=0 (middle of grid), linear function is just the bias
        # So compare at x=1 (last grid point) where slope matters
        self.assertNotEqual(fr_0[-1, 0], fr_1[-1, 0])

    def test_new_model_different_peq_shared_p0_d(self):
        """Test model with different peq but shared p0/D across conditions."""
        # Now using list of dicts for peq_model
        m = model.new_model(
            peq_model=[
                {'model': 'linear_pot', 'params': {'slope': 5}},
                {'model': 'linear_pot', 'params': {'slope': -5}}
            ],
            p0_model={'model': 'uniform', 'params': {}},
            D=0.1,
            fr_model=np.ones((self.grid.N, 1)) * 10,
            grid=self.grid,
            condition_mapping={
                'num_conditions': 2,
                'num_peq': 2,
                'num_p0': 1,
                'num_D': 1,
                'num_fr': 1,
                'peq_indices': [0, 1],
                'p0_indices': [0, 0],
                'D_indices': [0, 0],
                'fr_indices': [[0], [0]]
            }
        )
        
        self.assertEqual(m.condition_mapping['num_peq'], 2)
        self.assertEqual(m.condition_mapping['num_p0'], 1)
        
        # Check that peq is different
        peq_0, p0_0, D_0, fr_0 = m.get_params(0)
        peq_1, p1_1, D_1, fr_1 = m.get_params(1)
        
        # peq should be different (different slopes)
        self.assertFalse(np.allclose(peq_0, peq_1))
        # p0 and D should be the same
        np.testing.assert_array_almost_equal(p0_0, p1_1)
        self.assertEqual(D_0, D_1)

    def test_new_model_different_d_shared_others(self):
        """Test model with different D but shared peq/p0."""
        m = model.new_model(
            peq_model={'model': 'uniform', 'params': {}},
            p0_model={'model': 'uniform', 'params': {}},
            D=[0.1, 0.2],  # Different D values
            fr_model=[
                {"model": "linear", "params": {"slope": 50, "bias": 60}},
            ],
            grid=self.grid,
            condition_mapping={
                'num_conditions': 2,
                'num_peq': 1,
                'num_p0': 1,
                'num_D': 2,
                'num_fr': 1,
                'peq_indices': [0, 0],
                'p0_indices': [0, 0],
                'D_indices': [0, 1],
                'fr_indices': [[0], [0]]
            }
        )
        
        self.assertEqual(m.condition_mapping['num_D'], 2)
        
        peq_0, p0_0, D_0, fr_0 = m.get_params(0)
        peq_1, p1_1, D_1, fr_1 = m.get_params(1)
        
        self.assertEqual(D_0, 0.1)
        self.assertEqual(D_1, 0.2)
        np.testing.assert_array_almost_equal(peq_0, peq_1)

    def test_new_model_overlapping_fr(self):
        """Test model with overlapping fr models across conditions."""
        m = model.new_model(
            peq_model={'model': 'uniform', 'params': {}},
            p0_model={'model': 'uniform', 'params': {}},
            D=0.1,
            fr_model=[
                {"model": "linear", "params": {"slope": 50, "bias": 60}},   # fr 0
                {"model": "linear", "params": {"slope": -50, "bias": 60}},  # fr 1
                {"model": "linear", "params": {"slope": 30, "bias": 40}},   # fr 2
                {"model": "linear", "params": {"slope": -30, "bias": 40}},  # fr 3
                {"model": "linear", "params": {"slope": 10, "bias": 20}},   # fr 4
            ],
            grid=self.grid,
            condition_mapping={
                'num_conditions': 3,
                'num_peq': 1,
                'num_p0': 1,
                'num_D': 1,
                'num_fr': 5,
                'peq_indices': [0, 0, 0],
                'p0_indices': [0, 0, 0],
                'D_indices': [0, 0, 0],
                'fr_indices': [
                    [0, 1, 2],      # condition 0: neurons 0, 1, 2
                    [1, 2, 3],      # condition 1: neurons 1, 2, 3 (overlap: 1, 2)
                    [2, 3, 4]       # condition 2: neurons 2, 3, 4 (overlap with 1: 2, 3)
                ]
            }
        )
        
        self.assertEqual(m.num_models, 3)
        self.assertEqual(m.get_num_neurons(0), 3)
        self.assertEqual(m.get_num_neurons(1), 3)
        self.assertEqual(m.get_num_neurons(2), 3)
        
        # Get params for each condition
        peq_0, p0_0, D_0, fr_0 = m.get_params(0)
        peq_1, p1_1, D_1, fr_1 = m.get_params(1)
        peq_2, p2_2, D_2, fr_2 = m.get_params(2)
        
        # Check that the shared neurons have the same fr values
        # Neuron 1 (fr model 1) is shared between condition 0 and 1
        np.testing.assert_array_almost_equal(fr_0[:, 1], fr_1[:, 0])
        
        # Neuron 2 (fr model 2) is shared across all conditions
        np.testing.assert_array_almost_equal(fr_0[:, 2], fr_1[:, 1])
        np.testing.assert_array_almost_equal(fr_1[:, 1], fr_2[:, 0])
        
        # Neuron 3 (fr model 3) is shared between condition 1 and 2
        np.testing.assert_array_almost_equal(fr_1[:, 2], fr_2[:, 1])

    def test_new_model_all_different(self):
        """Test model where all parameters differ across conditions."""
        # Now using list of dicts for peq_model and p0_model
        m = model.new_model(
            peq_model=[
                {'model': 'linear_pot', 'params': {'slope': 5}},
                {'model': 'linear_pot', 'params': {'slope': -5}}
            ],
            p0_model=[
                {'model': 'single_well', 'params': {'miu': 100}},
                {'model': 'single_well', 'params': {'miu': 50}}
            ],
            D=[0.1, 0.2],
            fr_model=[
                {"model": "linear", "params": {"slope": 50, "bias": 60}},
                {"model": "linear", "params": {"slope": -50, "bias": 60}},
            ],
            grid=self.grid,
            condition_mapping={
                'num_conditions': 2,
                'num_peq': 2,
                'num_p0': 2,
                'num_D': 2,
                'num_fr': 2,
                'peq_indices': [0, 1],
                'p0_indices': [0, 1],
                'D_indices': [0, 1],
                'fr_indices': [[0], [1]]
            }
        )
        
        self.assertEqual(m.condition_mapping['num_peq'], 2)
        self.assertEqual(m.condition_mapping['num_p0'], 2)
        self.assertEqual(m.condition_mapping['num_D'], 2)
        self.assertEqual(m.condition_mapping['num_fr'], 2)
        
        # All parameters should be different
        peq_0, p0_0, D_0, fr_0 = m.get_params(0)
        peq_1, p1_1, D_1, fr_1 = m.get_params(1)
        
        self.assertFalse(np.allclose(peq_0, peq_1))
        self.assertFalse(np.allclose(p0_0, p1_1))
        self.assertNotEqual(D_0, D_1)
        self.assertFalse(np.allclose(fr_0, fr_1))

    def test_new_model_all_shared(self):
        """Test model where all parameters are shared across conditions."""
        m = model.new_model(
            peq_model={'model': 'uniform', 'params': {}},
            p0_model={'model': 'uniform', 'params': {}},
            D=0.1,
            fr_model=[
                {"model": "linear", "params": {"slope": 50, "bias": 60}},
                {"model": "linear", "params": {"slope": -50, "bias": 60}},
            ],
            grid=self.grid,
            condition_mapping={
                'num_conditions': 3,
                'num_peq': 1,
                'num_p0': 1,
                'num_D': 1,
                'num_fr': 2,
                'peq_indices': [0, 0, 0],
                'p0_indices': [0, 0, 0],
                'D_indices': [0, 0, 0],
                'fr_indices': [[0, 1], [0, 1], [0, 1]]
            }
        )
        
        self.assertEqual(m.num_models, 3)
        
        # All conditions should return identical parameters
        peq_0, p0_0, D_0, fr_0 = m.get_params(0)
        peq_1, p1_1, D_1, fr_1 = m.get_params(1)
        peq_2, p2_2, D_2, fr_2 = m.get_params(2)
        
        np.testing.assert_array_almost_equal(peq_0, peq_1)
        np.testing.assert_array_almost_equal(peq_1, peq_2)
        np.testing.assert_array_almost_equal(p0_0, p1_1)
        np.testing.assert_array_almost_equal(p1_1, p2_2)
        self.assertEqual(D_0, D_1)
        self.assertEqual(D_1, D_2)
        np.testing.assert_array_almost_equal(fr_0, fr_1)
        np.testing.assert_array_almost_equal(fr_1, fr_2)

    def test_new_model_mixed_sharing_complex(self):
        """Test complex model with mixed sharing patterns."""
        # 4 conditions:
        # - Conditions 0, 1 share peq
        # - Conditions 2, 3 share peq (different from 0,1)
        # - All share p0
        # - Conditions 0, 2 share D
        # - Conditions 1, 3 share D (different from 0,2)
        # - Overlapping fr models
        
        # Create different peq arrays
        from neuralflow.peq_models import linear_pot
        x = self.grid.x_d
        w = self.grid.w_d
        
        peq1 = linear_pot(x, w, slope=5)
        peq2 = linear_pot(x, w, slope=-5)
        peq = np.vstack([peq1, peq2])
        
        p0 = np.ones(self.grid.N) / 2  # uniform p0
        
        # Create fr arrays
        fr = np.zeros((3, self.grid.N))
        fr[0, :] = 50 * x + 60
        fr[1, :] = -50 * x + 60
        fr[2, :] = 30 * x + 40
        
        m = model.new_model(
            peq_model=peq,
            p0_model=p0,
            D=[0.1, 0.2],
            fr_model=fr.T,  # shape (N, num_fr)
            grid=self.grid,
            condition_mapping={
                'num_conditions': 4,
                'num_peq': 2,
                'num_p0': 1,
                'num_D': 2,
                'num_fr': 3,
                'peq_indices': [0, 0, 1, 1],
                'p0_indices': [0, 0, 0, 0],
                'D_indices': [0, 1, 0, 1],
                'fr_indices': [
                    [0, 1],      # condition 0: neurons 0, 1
                    [1, 2],      # condition 1: neurons 1, 2
                    [0, 2],      # condition 2: neurons 0, 2
                    [0, 1, 2]    # condition 3: neurons 0, 1, 2
                ]
            }
        )
        
        self.assertEqual(m.num_models, 4)
        self.assertEqual(m.get_num_neurons(0), 2)
        self.assertEqual(m.get_num_neurons(1), 2)
        self.assertEqual(m.get_num_neurons(2), 2)
        self.assertEqual(m.get_num_neurons(3), 3)
        
        # Check peq sharing (0,1 share; 2,3 share)
        peq_0, _, _, _ = m.get_params(0)
        peq_1, _, _, _ = m.get_params(1)
        peq_2, _, _, _ = m.get_params(2)
        peq_3, _, _, _ = m.get_params(3)
        
        np.testing.assert_array_almost_equal(peq_0, peq_1)
        np.testing.assert_array_almost_equal(peq_2, peq_3)
        self.assertFalse(np.allclose(peq_0, peq_2))
        
        # Check D sharing (0,2 share; 1,3 share)
        _, _, D_0, _ = m.get_params(0)
        _, _, D_1, _ = m.get_params(1)
        _, _, D_2, _ = m.get_params(2)
        _, _, D_3, _ = m.get_params(3)
        
        self.assertEqual(D_0, D_2)
        self.assertEqual(D_1, D_3)
        self.assertNotEqual(D_0, D_1)


class TestCArray(unittest.TestCase):
    """Tests for the C array (fr values at xbegin)."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.grid = GLLgrid(xbegin=-1, xend=1, Np=8, Ne=16)
        cls.N = cls.grid.N
        
        x = cls.grid.x_d
        cls.peq = np.exp(-x**2 / 0.1)
        cls.peq /= np.sum(cls.peq * cls.grid.w_d)
        
        cls.p0 = np.exp(-x**2 / 0.15)
        cls.p0 /= np.sum(cls.p0 * cls.grid.w_d)

    def test_c_array_shape(self):
        """Test that C array has correct shape (num_fr,)."""
        # Create fr with shape (num_fr, N)
        fr = np.random.rand(10, self.N) + 0.1
        
        condition_mapping = {
            'num_conditions': 2,
            'num_peq': 1,
            'num_p0': 1,
            'num_D': 1,
            'num_fr': 10,
            'peq_indices': [0, 0],
            'p0_indices': [0, 0],
            'D_indices': [0, 0],
            'fr_indices': [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
        }
        
        m = model(self.peq[np.newaxis, :], self.p0[np.newaxis, :], 
                  np.array([0.1]), fr, condition_mapping, self.grid)
        
        self.assertEqual(m.C.shape, (10,))

    def test_c_array_values(self):
        """Test that C array contains fr values at xbegin (index 0)."""
        fr = np.random.rand(5, self.N) + 0.1
        
        condition_mapping = {
            'num_conditions': 1,
            'num_peq': 1,
            'num_p0': 1,
            'num_D': 1,
            'num_fr': 5,
            'peq_indices': [0],
            'p0_indices': [0],
            'D_indices': [0],
            'fr_indices': [[0, 1, 2, 3, 4]]
        }
        
        m = model(self.peq[np.newaxis, :], self.p0[np.newaxis, :], 
                  np.array([0.1]), fr, condition_mapping, self.grid)
        
        # C should be fr[:, 0]
        np.testing.assert_array_almost_equal(m.C, fr[:, 0])


if __name__ == '__main__':
    logging.basicConfig(level=logging.WARNING)
    unittest.main()