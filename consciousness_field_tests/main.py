#!/usr/bin/env python3
"""
Main entry point for Cosmic Consciousness Analysis.
"""

import os
import sys
import argparse
import time
from datetime import datetime

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import utilities
from consciousness_field_tests.utils.data_loader import get_data_loader
from consciousness_field_tests.utils.visualization import get_visualizer
from consciousness_field_tests.utils.statistics import get_statistical_analyzer

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Cosmic Consciousness Analysis')
    
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Directory containing the CMB data files')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Directory to save results')
    parser.add_argument('--test', type=str, default=None,
                        help='Specific test to run (default: run all tests)')
    parser.add_argument('--monte-carlo', type=int, default=1000,
                        help='Number of Monte Carlo simulations')
    parser.add_argument('--visualize', action='store_true',
                        help='Create visualizations')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed output')
    
    return parser.parse_args()

def main():
    """Main function."""
    # Parse arguments
    args = parse_arguments()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Print header
    print("=== COSMIC CONSCIOUSNESS ANALYSIS ===")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Monte Carlo simulations: {args.monte_carlo}")
    print()
    
    # Initialize utilities
    print("Initializing...")
    start_time = time.time()
    
    data_loader = get_data_loader(args.data_dir)
    visualizer = get_visualizer(args.output_dir)
    stats_analyzer = get_statistical_analyzer()
    
    print(f"Initialization completed in {time.time() - start_time:.2f} seconds.")
    print(f"Loaded data with {len(data_loader.data['ell'])} multipoles.")
    print()
    
    # If a specific test is requested, run only that test
    if args.test:
        # Import the requested test module
        try:
            if args.test.startswith('basic.'):
                test_name = args.test.split('.')[1]
                module_name = f"consciousness_field_tests.basic_tests.{test_name}"
            elif args.test.startswith('advanced.'):
                test_name = args.test.split('.')[1]
                module_name = f"consciousness_field_tests.advanced_tests.{test_name}"
            else:
                # Try to find the test in both basic and advanced tests
                try:
                    module_name = f"consciousness_field_tests.basic_tests.{args.test}"
                    __import__(module_name)
                except ImportError:
                    module_name = f"consciousness_field_tests.advanced_tests.{args.test}"
            
            module = __import__(module_name, fromlist=['run_test'])
            
            print(f"Running test: {args.test}")
            start_time = time.time()
            
            # Run the test
            results = module.run_test(
                data_loader=data_loader,
                visualizer=visualizer,
                stats_analyzer=stats_analyzer,
                monte_carlo_sims=args.monte_carlo,
                verbose=args.verbose
            )
            
            print(f"Test completed in {time.time() - start_time:.2f} seconds.")
            
            # Print results
            print("\n=== RESULTS ===")
            for key, value in results.items():
                if isinstance(value, (int, float)):
                    print(f"{key}: {value:.6f}")
                else:
                    print(f"{key}: {value}")
            
            # Create visualization if requested
            if args.visualize:
                visualizer.plot_test_results(results, args.test)
                print(f"Visualization saved to {os.path.join(args.output_dir, f'{args.test}_results.png')}")
            
        except ImportError:
            print(f"Error: Test '{args.test}' not found.")
            sys.exit(1)
        except AttributeError:
            print(f"Error: Test '{args.test}' does not have a run_test function.")
            sys.exit(1)
    else:
        # Run all tests by importing the run_all_tests module
        try:
            from consciousness_field_tests.run_all_tests import run_all_tests
            
            print("Running all tests...")
            start_time = time.time()
            
            # Run all tests
            results = run_all_tests(
                data_loader=data_loader,
                visualizer=visualizer,
                stats_analyzer=stats_analyzer,
                monte_carlo_sims=args.monte_carlo,
                verbose=args.verbose
            )
            
            print(f"All tests completed in {time.time() - start_time:.2f} seconds.")
            
            # Print summary results
            print("\n=== SUMMARY RESULTS ===")
            print(f"Combined p-value: {results['combined_p_value']:.6f}")
            print(f"Mean phi-optimality: {results['mean_phi_optimality']:.4f}")
            
            # Create visualization if requested
            if args.visualize:
                visualizer.plot_comprehensive_results(results['test_results'], results['sorted_tests'])
                print(f"Visualization saved to {os.path.join(args.output_dir, 'comprehensive_analysis_results.png')}")
            
        except ImportError:
            print("Error: run_all_tests module not found.")
            print("Please run individual tests using the --test option.")
            sys.exit(1)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
