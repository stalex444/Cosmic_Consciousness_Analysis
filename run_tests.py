#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run Tests Script
--------------
Runs all unit tests for the Cosmic Consciousness Analysis codebase.
"""

import os
import sys
import unittest
import time

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

if __name__ == '__main__':
    print("=" * 80)
    print("COSMIC CONSCIOUSNESS ANALYSIS - TEST SUITE")
    print("=" * 80)
    
    start_time = time.time()
    
    # Discover and run all tests
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('tests', pattern='test_*.py')
    
    # Run tests
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)
    
    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print("Tests run: {}".format(result.testsRun))
    print("Errors: {}".format(len(result.errors)))
    print("Failures: {}".format(len(result.failures)))
    print("Skipped: {}".format(len(result.skipped)))
    print("Time elapsed: {:.2f} seconds".format(time.time() - start_time))
    
    # Exit with appropriate status code
    if result.wasSuccessful():
        print("\nAll tests passed successfully!")
        sys.exit(0)
    else:
        print("\nSome tests failed. See above for details.")
        sys.exit(1)
