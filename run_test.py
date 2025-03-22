#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test Runner Script
-----------------
Runs individual tests for debugging and development purposes.
"""

import os
import argparse
import numpy as np
import time

from core_framework.constants import DEFAULT_SEED
from core_framework.data_handler import load_cmb_data

from tests.coherence_tests.meta_coherence_test import MetaCoherenceTest
from tests.information_tests.transfer_entropy_test import TransferEntropyTest
from tests.scale_tests.scale_transition_test import ScaleTransitionTest
from tests.structural_tests.golden_ratio_test import GoldenRatioTest
from tests.structural_tests.fractal_analysis_test import FractalAnalysisTest


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run individual cosmic consciousness tests")
    
    parser.add_argument("--test", type=str, required=True,
                        choices=["meta-coherence", "transfer-entropy", "scale-transition", 
                                 "golden-ratio", "fractal-analysis"],
                        help="Test to run")
    parser.add_argument("--data-dir", type=str, default="results",
                        help="Directory for data storage and results")
    parser.add_argument("--simulated", action="store_true", default=True,
                        help="Use simulated data instead of real CMB data")
    parser.add_argument("--data-file", type=str, default=None,
                        help="Path to data file (if not using simulated data)")
    parser.add_argument("--data-size", type=int, default=4096,
                        help="Size of simulated data")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Random seed for reproducibility")
    parser.add_argument("--visualize", action="store_true", default=True,
                        help="Generate visualizations")
    parser.add_argument("--report", action="store_true", default=True,
                        help="Generate detailed report")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode with additional output")
    
    return parser.parse_args()


def main():
    """Main function to run a single test."""
    args = parse_arguments()
    
    # Create output directory if it doesn't exist
    test_output_dir = os.path.join(args.data_dir, args.test.replace("-", "_"))
    os.makedirs(test_output_dir, exist_ok=True)
    
    print("=" * 80)
    print("COSMIC CONSCIOUSNESS TEST RUNNER")
    print("=" * 80)
    print("Test: {}".format(args.test))
    print("Output directory: {}".format(test_output_dir))
    print("Using {} data".format("simulated" if args.simulated else "real"))
    if args.simulated:
        print("Data size: {}".format(args.data_size))
    else:
        print("Data file: {}".format(args.data_file if args.data_file else "None specified"))
    print("Random seed: {}".format(args.seed))
    print("Debug mode: {}".format("Enabled" if args.debug else "Disabled"))
    print("=" * 80)
    
    # Load data
    start_time = time.time()
    print("\nLoading data...")
    
    if args.simulated:
        data = load_cmb_data(simulated=True, seed=args.seed, size=args.data_size)
    else:
        data = load_cmb_data(simulated=False, filepath=args.data_file)
    
    print("Data loaded in {:.2f} seconds.".format(time.time() - start_time))
    
    # Initialize and run the selected test
    test = None
    
    if args.test == "meta-coherence":
        test = MetaCoherenceTest(seed=args.seed, data=data)
    elif args.test == "transfer-entropy":
        test = TransferEntropyTest(seed=args.seed, data=data)
    elif args.test == "scale-transition":
        test = ScaleTransitionTest(seed=args.seed, data=data)
    elif args.test == "golden-ratio":
        test = GoldenRatioTest(seed=args.seed, data=data)
    elif args.test == "fractal-analysis":
        test = FractalAnalysisTest(seed=args.seed, data=data)
    
    if test is None:
        print("Error: Unknown test '{}'".format(args.test))
        return
    
    # Run the test
    print("\nRunning {} test...".format(args.test))
    start_time = time.time()
    
    if args.debug:
        # Enable more verbose output
        test.debug = True
    
    test.run()
    print("Test completed in {:.2f} seconds.".format(time.time() - start_time))
    
    # Generate report
    if args.report:
        print("\nGenerating test report...")
        report = test.generate_report()
        report_path = os.path.join(test_output_dir, "{}_report.txt".format(args.test.replace("-", "_")))
        with open(report_path, "w") as f:
            f.write(report)
        print("Report saved to {}".format(report_path))
    
    # Generate visualizations
    if args.visualize:
        print("\nGenerating visualizations...")
        test.visualize(output_dir=test_output_dir)
        print("Visualizations saved to {}".format(test_output_dir))
    
    # Print summary of results
    print("\nTest Results Summary:")
    print("-" * 40)
    
    if hasattr(test, 'phi_optimality') and test.phi_optimality is not None:
        print("Phi-optimality: {:.4f}".format(test.phi_optimality))
    
    if hasattr(test, 'p_value') and test.p_value is not None:
        print("p-value: {:.4f}".format(test.p_value))
        print("Statistically significant: {}".format("Yes" if test.p_value < 0.05 else "No"))
    
    if hasattr(test, 'effect_size') and test.effect_size is not None:
        print("Effect size: {:.4f}".format(test.effect_size))
    
    print("-" * 40)
    
    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
