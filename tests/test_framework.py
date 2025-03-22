#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test Framework
-------------
Unit testing framework for the Cosmic Consciousness Analysis codebase.
"""

import os
import sys
import unittest
import numpy as np

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core_framework.constants import DEFAULT_SEED, PHI, E, PI
from core_framework.data_handler import (
    load_cmb_data, 
    generate_simulated_cmb_data,
    generate_surrogate_data
)
from core_framework.statistics import (
    calculate_significance,
    bootstrap_confidence_interval
)

class TestDataHandler(unittest.TestCase):
    """Test the data handler module."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.seed = DEFAULT_SEED
        self.data_size = 1024
    
    def test_simulated_data_generation(self):
        """Test that simulated data is generated correctly."""
        data = generate_simulated_cmb_data(size=self.data_size, seed=self.seed)
        
        # Check data shape
        self.assertEqual(len(data), self.data_size)
        
        # Check data properties
        self.assertAlmostEqual(np.mean(data), 0.0, delta=0.1)
        self.assertAlmostEqual(np.std(data), 1.0, delta=0.1)
    
    def test_surrogate_data_generation(self):
        """Test that surrogate data is generated correctly."""
        original_data = generate_simulated_cmb_data(size=self.data_size, seed=self.seed)
        
        # Test phase randomization
        n_surrogates = 5
        surrogates = generate_surrogate_data(
            original_data, 
            method='phase_randomization', 
            n_surrogates=n_surrogates,
            seed=self.seed
        )
        
        # Check shape
        self.assertEqual(surrogates.shape, (n_surrogates, self.data_size))
        
        # Check that surrogates are different from original
        for i in range(n_surrogates):
            self.assertGreater(np.sum(np.abs(surrogates[i] - original_data)), 0)
        
        # Check that power spectra are similar
        original_psd = np.abs(np.fft.fft(original_data))**2
        for i in range(n_surrogates):
            surrogate_psd = np.abs(np.fft.fft(surrogates[i]))**2
            correlation = np.corrcoef(original_psd, surrogate_psd)[0, 1]
            self.assertGreater(correlation, 0.9)


class TestStatistics(unittest.TestCase):
    """Test the statistics module."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.seed = DEFAULT_SEED
        np.random.seed(self.seed)
        
        # Generate test data with a clear difference
        self.data1 = np.random.normal(0, 1, 100)
        self.data2 = np.random.normal(0.5, 1, 100)
    
    def test_significance_calculation(self):
        """Test significance calculation."""
        p_value = calculate_significance(self.data1, self.data2)
        
        # p-value should be small (significant difference)
        self.assertLess(p_value, 0.05)
    
    def test_bootstrap_confidence_interval(self):
        """Test bootstrap confidence interval calculation."""
        ci_lower, ci_upper = bootstrap_confidence_interval(
            self.data1, 
            statistic=np.mean,
            n_bootstrap=1000,
            confidence=0.95,
            seed=self.seed
        )
        
        # Check that confidence interval contains the true mean
        true_mean = np.mean(self.data1)
        self.assertLessEqual(ci_lower, true_mean)
        self.assertGreaterEqual(ci_upper, true_mean)


class TestConstants(unittest.TestCase):
    """Test the constants module."""
    
    def test_mathematical_constants(self):
        """Test that mathematical constants are defined correctly."""
        self.assertAlmostEqual(PHI, (1 + np.sqrt(5)) / 2, places=10)
        self.assertAlmostEqual(E, np.e, places=10)
        self.assertAlmostEqual(PI, np.pi, places=10)


if __name__ == '__main__':
    unittest.main()
