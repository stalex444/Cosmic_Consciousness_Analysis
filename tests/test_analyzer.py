#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test Analyzer
------------
Unit testing for the CosmicConsciousnessAnalyzer class.
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
from tests.structural_tests.golden_ratio_test import GoldenRatioTest
from analysis.analysis import CosmicConsciousnessAnalyzer


class TestAnalyzer(unittest.TestCase):
    """Test the CosmicConsciousnessAnalyzer class."""
    
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
        self.test_dir = os.path.join(os.path.dirname(__file__), 'temp_analyzer_output')
        if not os.path.exists(self.test_dir):
            os.makedirs(self.test_dir)
        
        # Create analyzer
        self.analyzer = CosmicConsciousnessAnalyzer(output_dir=self.test_dir)
    
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
    
    def test_add_test(self):
        """Test adding tests to the analyzer."""
        # Create test instances
        meta_coherence_test = MetaCoherenceTest(seed=self.seed, data=self.data)
        golden_ratio_test = GoldenRatioTest(seed=self.seed, data=self.data)
        
        # Add tests to analyzer
        self.analyzer.add_test(meta_coherence_test)
        self.analyzer.add_test(golden_ratio_test)
        
        # Check that tests were added
        self.assertEqual(len(self.analyzer.tests), 2)
        self.assertIn(meta_coherence_test, self.analyzer.tests)
        self.assertIn(golden_ratio_test, self.analyzer.tests)
    
    def test_run_all_tests(self):
        """Test running all tests."""
        # Add tests to analyzer
        self.analyzer.add_test(MetaCoherenceTest(seed=self.seed, data=self.data))
        self.analyzer.add_test(GoldenRatioTest(seed=self.seed, data=self.data))
        
        # Run all tests
        self.analyzer.run_all_tests()
        
        # Check that all tests were run
        for test in self.analyzer.tests:
            self.assertTrue(test.has_run)
            self.assertIsNotNone(test.p_value)
    
    def test_calculate_combined_significance(self):
        """Test calculating combined significance."""
        # Add tests to analyzer
        self.analyzer.add_test(MetaCoherenceTest(seed=self.seed, data=self.data))
        self.analyzer.add_test(GoldenRatioTest(seed=self.seed, data=self.data))
        
        # Run all tests
        self.analyzer.run_all_tests()
        
        # Calculate combined significance
        combined_p_value = self.analyzer.calculate_combined_significance()
        
        # Check that combined p-value is valid
        self.assertIsNotNone(combined_p_value)
        self.assertGreaterEqual(combined_p_value, 0)
        self.assertLessEqual(combined_p_value, 1)
    
    def test_generate_report(self):
        """Test generating a comprehensive report."""
        # Add tests to analyzer
        self.analyzer.add_test(MetaCoherenceTest(seed=self.seed, data=self.data))
        self.analyzer.add_test(GoldenRatioTest(seed=self.seed, data=self.data))
        
        # Run all tests
        self.analyzer.run_all_tests()
        
        # Generate report
        report = self.analyzer.generate_report()
        
        # Check that report is valid
        self.assertIsInstance(report, str)
        self.assertGreater(len(report), 0)
        
        # Check that report contains test names
        self.assertIn("Meta-Coherence Test", report)
        self.assertIn("Golden Ratio Test", report)
    
    def test_visualize_results(self):
        """Test visualizing results."""
        # Add tests to analyzer
        self.analyzer.add_test(MetaCoherenceTest(seed=self.seed, data=self.data))
        self.analyzer.add_test(GoldenRatioTest(seed=self.seed, data=self.data))
        
        # Run all tests
        self.analyzer.run_all_tests()
        
        # Visualize results
        self.analyzer.visualize_results()
        
        # Check that visualization files were created
        self.assertTrue(any(filename.endswith('.png') for filename in os.listdir(self.test_dir)))
    
    def test_save_results(self):
        """Test saving results."""
        # Add tests to analyzer
        self.analyzer.add_test(MetaCoherenceTest(seed=self.seed, data=self.data))
        self.analyzer.add_test(GoldenRatioTest(seed=self.seed, data=self.data))
        
        # Run all tests
        self.analyzer.run_all_tests()
        
        # Save results
        self.analyzer.save_results()
        
        # Check that results file was created
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, 'cosmic_analysis_results.json')))


if __name__ == '__main__':
    unittest.main()
