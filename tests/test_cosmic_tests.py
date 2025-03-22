#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test Cosmic Tests
----------------
Unit testing for the individual cosmic test classes.
"""

import os
import sys
import unittest
import numpy as np

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core_framework.constants import DEFAULT_SEED
from core_framework.data_handler import generate_simulated_cmb_data

from tests.coherence_tests.meta_coherence_test import MetaCoherenceTest
from tests.information_tests.transfer_entropy_test import TransferEntropyTest
from tests.scale_tests.scale_transition_test import ScaleTransitionTest
from tests.structural_tests.golden_ratio_test import GoldenRatioTest
from tests.structural_tests.fractal_analysis_test import FractalAnalysisTest


class TestCosmicTests(unittest.TestCase):
    """Test the individual cosmic test classes."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.seed = DEFAULT_SEED
        self.data_size = 2048
        
        # Generate test data with phi-related patterns
        self.data = generate_simulated_cmb_data(
            size=self.data_size, 
            seed=self.seed,
            phi_bias=0.2  # Stronger bias for testing
        )
        
        # Create temporary directory for test outputs
        self.test_dir = os.path.join(os.path.dirname(__file__), 'temp_test_output')
        if not os.path.exists(self.test_dir):
            os.makedirs(self.test_dir)
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove test output files
        for filename in os.listdir(self.test_dir):
            file_path = os.path.join(self.test_dir, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print("Error removing file {}: {}".format(file_path, e))
    
    def test_golden_ratio_test(self):
        """Test the GoldenRatioTest class."""
        test = GoldenRatioTest(seed=self.seed, data=self.data)
        
        # Run the test
        test.run()
        
        # Check that results are as expected
        self.assertIsNotNone(test.phi_optimality)
        self.assertIsNotNone(test.p_value)
        self.assertIsNotNone(test.confidence_interval)
        
        # Check that phi-optimality is positive (indicating golden ratio patterns)
        self.assertGreater(test.phi_optimality, 0)
        
        # Check that p-value is significant
        self.assertLess(test.p_value, 0.05)
        
        # Test report generation
        report = test.generate_report()
        self.assertIsInstance(report, str)
        self.assertGreater(len(report), 0)
        
        # Test visualization
        test.visualize(output_dir=self.test_dir)
        # Check that visualization files were created
        self.assertTrue(any(filename.startswith('golden_ratio_test') for filename in os.listdir(self.test_dir)))
    
    def test_fractal_analysis_test(self):
        """Test the FractalAnalysisTest class."""
        test = FractalAnalysisTest(seed=self.seed, data=self.data)
        
        # Run the test
        test.run()
        
        # Check that results are as expected
        self.assertIsNotNone(test.hurst_exponent)
        self.assertIsNotNone(test.p_value)
        
        # Check that Hurst exponent is in valid range
        self.assertGreaterEqual(test.hurst_exponent, 0)
        self.assertLessEqual(test.hurst_exponent, 1)
        
        # Test report generation
        report = test.generate_report()
        self.assertIsInstance(report, str)
        self.assertGreater(len(report), 0)
        
        # Test visualization
        test.visualize(output_dir=self.test_dir)
        # Check that visualization files were created
        self.assertTrue(any(filename.startswith('fractal_analysis_test') for filename in os.listdir(self.test_dir)))
    
    def test_meta_coherence_test(self):
        """Test the MetaCoherenceTest class."""
        test = MetaCoherenceTest(seed=self.seed, data=self.data)
        
        # Run the test
        test.run()
        
        # Check that results are as expected
        self.assertIsNotNone(test.meta_coherence)
        self.assertIsNotNone(test.p_value)
        
        # Test report generation
        report = test.generate_report()
        self.assertIsInstance(report, str)
        self.assertGreater(len(report), 0)
        
        # Test visualization
        test.visualize(output_dir=self.test_dir)
        # Check that visualization files were created
        self.assertTrue(any(filename.startswith('meta_coherence_test') for filename in os.listdir(self.test_dir)))
    
    def test_transfer_entropy_test(self):
        """Test the TransferEntropyTest class."""
        test = TransferEntropyTest(seed=self.seed, data=self.data)
        
        # Run the test
        test.run()
        
        # Check that results are as expected
        self.assertIsNotNone(test.transfer_entropy)
        self.assertIsNotNone(test.p_value)
        
        # Test report generation
        report = test.generate_report()
        self.assertIsInstance(report, str)
        self.assertGreater(len(report), 0)
        
        # Test visualization
        test.visualize(output_dir=self.test_dir)
        # Check that visualization files were created
        self.assertTrue(any(filename.startswith('transfer_entropy_test') for filename in os.listdir(self.test_dir)))
    
    def test_scale_transition_test(self):
        """Test the ScaleTransitionTest class."""
        test = ScaleTransitionTest(seed=self.seed, data=self.data)
        
        # Run the test
        test.run()
        
        # Check that results are as expected
        self.assertIsNotNone(test.scale_transitions)
        self.assertIsNotNone(test.p_value)
        
        # Test report generation
        report = test.generate_report()
        self.assertIsInstance(report, str)
        self.assertGreater(len(report), 0)
        
        # Test visualization
        test.visualize(output_dir=self.test_dir)
        # Check that visualization files were created
        self.assertTrue(any(filename.startswith('scale_transition_test') for filename in os.listdir(self.test_dir)))


if __name__ == '__main__':
    unittest.main()
