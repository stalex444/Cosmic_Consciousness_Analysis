#!/usr/bin/env python3
"""
Cosmic Consciousness Analysis Runner

This script runs a comprehensive analysis of CMB data to search for evidence
of cosmic consciousness using the CosmicConsciousnessAnalyzer class.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys
from cosmic_consciousness_analyzer import CosmicConsciousnessAnalyzer

def parse_arguments():
    """Parse command line arguments."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run Cosmic Consciousness Analysis')
    parser.add_argument('--data-dir', type=str, default='data', help='Directory containing Planck data')
    parser.add_argument('--output-dir', type=str, default='analysis_results', help='Directory to save results')
    parser.add_argument('--monte-carlo', type=int, default=10000, help='Number of Monte Carlo simulations')
    parser.add_argument('--test', type=str, help='Run a specific test only')
    parser.add_argument('--save-csv', action='store_true', help='Save results to CSV file')
    parser.add_argument('--save-json', action='store_true', help='Save results to JSON file')
    parser.add_argument('--phi-optimality', action='store_true', help='Calculate phi-optimality for each test')
    parser.add_argument('--verbose', action='store_true', help='Print verbose output')
    args = parser.parse_args()
    
    return args

def main():
    """Main function to run the cosmic consciousness analysis."""
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check if data directory exists
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory '{args.data_dir}' not found.")
        sys.exit(1)
    
    if args.verbose:
        print(f"Using data directory: {os.path.abspath(args.data_dir)}")
        print(f"Output directory: {os.path.abspath(args.output_dir)}")
        print(f"Monte Carlo simulations: {args.monte_carlo}")
    
    # Initialize the analyzer
    try:
        analyzer = CosmicConsciousnessAnalyzer(data_dir=args.data_dir, 
                                              monte_carlo_sims=args.monte_carlo)
        if args.verbose:
            print("Analyzer initialized successfully.")
            print(f"Data loaded: {len(analyzer.data['ell'])} multipoles")
    except Exception as e:
        print(f"Error initializing analyzer: {e}")
        print("\nTroubleshooting suggestions:")
        print("1. Ensure the data directory contains the required Planck data files")
        print("2. Check file permissions")
        print("3. Verify that the data files are in the correct format")
        print("4. Make sure all dependencies are installed (numpy, scipy, matplotlib, astropy)")
        sys.exit(1)
    
    # Run the analysis
    if args.test:
        # Run a specific test
        test_method = getattr(analyzer, f"test_{args.test}", None)
        if test_method is None:
            print(f"Error: Test '{args.test}' not found.")
            print("Available tests:")
            for method in dir(analyzer):
                if method.startswith("test_") and callable(getattr(analyzer, method)):
                    print(f"  - {method[5:]}")
            sys.exit(1)
        
        print(f"Running {args.test} test...")
        result = test_method()
        
        # Save the result
        if args.save_csv:
            save_result_to_csv(result, f"{args.output_dir}/{args.test}_result.csv")
        if args.save_json:
            save_result_to_json(result, f"{args.output_dir}/{args.test}_result.json")
    else:
        # Run comprehensive analysis
        print("Running comprehensive analysis...")
        results = analyzer.run_comprehensive_analysis(output_dir=args.output_dir)
        
        # Save the results
        if args.save_csv:
            save_results_to_csv(results, f"{args.output_dir}/comprehensive_results.csv")
        if args.save_json:
            save_results_to_json(results, f"{args.output_dir}/comprehensive_results.json")
        
        # Calculate phi-optimality if requested
        if args.phi_optimality:
            print("\nCalculating phi-optimality for all tests...")
            calculate_phi_optimality(results)
    
    print("\nAnalysis complete! Results saved to the output directory.")

def run_specific_test(analyzer, test_name):
    """Run a specific test based on the test name."""
    
    print(f"Running {test_name} test...")
    
    if test_name == 'gr':
        results = analyzer.test_gr_significance()
        print(f"GR Signal: {results[4]:.2f}x excess, {results[2]:.2f}σ (p = {results[3]:.8f})")
    
    elif test_name == 'coherence':
        results = analyzer.test_coherence()
        print(f"Coherence: {1/results[4]:.2f}x stronger, {results[2]:.2f}σ (p = {results[3]:.8f})")
    
    elif test_name == 'gr_coherence':
        results = analyzer.test_gr_coherence()
        print(f"GR Coherence: {1/results[4]:.2f}x stronger, {results[2]:.2f}σ (p = {results[3]:.8f})")
    
    elif test_name == 'hierarchy':
        results = analyzer.test_hierarchical_organization()
        print(f"Hierarchical Organization: {results:.2f}x stronger than random")
    
    elif test_name == 'info':
        results = analyzer.test_information_integration()
        print(f"Information Integration: {results[2]:.2f}σ (p = {results[3]:.8f})")
    
    elif test_name == 'optimization':
        results = analyzer.test_optimization()
        print(f"Optimization: {results[4]:.2f}x, {results[2]:.2f}σ (p = {results[3]:.8f})")
    
    elif test_name == 'resonance':
        results = analyzer.test_resonance()
        print(f"Resonance: {results[4]:.2f}x, {results[2]:.2f}σ (p = {results[3]:.8f})")
    
    elif test_name == 'fractal':
        results = analyzer.test_fractal_structure()
        print(f"Fractal Structure: {results[2]:.2f}σ (p = {results[3]:.8f})")
    
    elif test_name == 'meta':
        results = analyzer.test_meta_coherence()
        print(f"Meta-Coherence: {results[4]:.2f}x, {results[2]:.2f}σ (p = {results[3]:.8f})")
    
    elif test_name == 'multiscale':
        results = analyzer.test_multiscale_patterns()
        print(f"Multi-Scale Patterns: {results[0]:.2f}x (95% CI: [{results[1]:.2f}, {results[2]:.2f}])")
    
    elif test_name == 'frequencies':
        results = analyzer.analyze_specific_frequencies()
        print(f"Peak Frequency Analysis: Mean Phi-Optimality = {results[4]:.3f}")
    
    else:
        print(f"Unknown test: {test_name}")
        print("Available tests: gr, coherence, gr_coherence, hierarchy, info, optimization, resonance, fractal, meta, multiscale, frequencies")

def calculate_phi_optimality(results):
    """
    Calculate phi-optimality for each test result.
    
    Phi-optimality measures how close a result is to the golden ratio (or its inverse).
    A value of 1 indicates perfect alignment with the golden ratio.
    A value of -1 indicates maximum deviation.
    
    Parameters:
    -----------
    results : dict
        Dictionary containing all test results
    """
    phi = (1 + np.sqrt(5)) / 2
    inv_phi = 1 / phi
    
    # Function to calculate phi-optimality
    def phi_opt(observed_ratio, target_ratio=inv_phi):
        # Ensure ratio is less than 1 (use inverse if needed)
        if observed_ratio > 1:
            observed_ratio = 1 / observed_ratio
            
        # Calculate phi-optimality (bounded between -1 and 1)
        optimality = max(-1, min(1, 1 - abs(observed_ratio - target_ratio) / target_ratio))
        return optimality
    
    # Calculate phi-optimality for each test
    optimalities = {}
    
    # GR test - ratio of actual/random power
    optimalities['gr'] = phi_opt(results['gr_test'][0] / results['gr_test'][1])
    
    # Coherence - ratio of actual/random variance (inverse)
    optimalities['coherence'] = phi_opt(results['coherence_test'][1] / results['coherence_test'][0])
    
    # GR coherence - ratio of actual/random variance (inverse)
    optimalities['gr_coherence'] = phi_opt(results['gr_coherence_test'][1] / results['gr_coherence_test'][0])
    
    # Hierarchical organization
    optimalities['hierarchy'] = phi_opt(results['hierarchy_test'])
    
    # Information integration - ratio of actual/random MI
    optimalities['info'] = phi_opt(results['info_test'][0] / results['info_test'][1])
    
    # Optimization - ratio of random/actual deviation
    optimalities['optimization'] = phi_opt(results['optimization_test'][4])
    
    # Resonance - ratio of actual/random resonance
    optimalities['resonance'] = phi_opt(results['resonance_test'][4])
    
    # Fractal - ratio of actual/random Hurst
    optimalities['fractal'] = phi_opt(results['fractal_test'][0] / results['fractal_test'][1])
    
    # Meta-coherence - ratio of actual/random meta-coherence
    optimalities['meta'] = phi_opt(results['meta_test'][4])
    
    # Multi-scale - ratio from bootstrap
    optimalities['multiscale'] = phi_opt(results['multiscale_test'][0])
    
    # Peak frequency analysis - already calculated as phi-optimality
    optimalities['frequencies'] = results['frequency_test'][4]
    
    # Calculate average phi-optimality
    avg_optimality = np.mean(list(optimalities.values()))
    
    # Print results
    print("\nPhi-Optimality Results:")
    print("------------------------")
    for test, opt in optimalities.items():
        print(f"{test.ljust(15)}: {opt:.4f}")
    print("------------------------")
    print(f"Average Phi-Optimality: {avg_optimality:.4f}")
    
    # Create visualization
    plt.figure(figsize=(12, 6))
    
    tests = list(optimalities.keys())
    values = list(optimalities.values())
    
    # Color bars by optimality
    colors = ['green' if v >= 0.8 else 'orange' if v >= 0.5 else 'red' for v in values]
    
    plt.bar(tests, values, color=colors)
    plt.axhline(y=avg_optimality, color='blue', linestyle='--', 
               label=f'Average: {avg_optimality:.4f}')
    
    plt.xlabel('Test')
    plt.ylabel('Phi-Optimality')
    plt.title('Phi-Optimality Across Tests')
    plt.ylim(-1, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.savefig(f"{args.output_dir}/phi_optimality.png")
    
    # Save results to file
    with open(f"{args.output_dir}/phi_optimality_results.txt", 'w') as f:
        f.write("=== PHI-OPTIMALITY RESULTS ===\n\n")
        for test, opt in optimalities.items():
            f.write(f"{test.ljust(15)}: {opt:.4f}\n")
        f.write("\n------------------------\n")
        f.write(f"Average Phi-Optimality: {avg_optimality:.4f}\n")
        f.write("\nA value of 1 indicates perfect alignment with the golden ratio.\n")
        f.write("A value of -1 indicates maximum deviation.\n")

def save_result_to_csv(result, filename):
    import csv
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Result"])
        writer.writerow([result])

def save_results_to_csv(results, filename):
    import csv
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Test", "Result"])
        for test, result in results.items():
            writer.writerow([test, result])

def save_result_to_json(result, filename):
    import json
    with open(filename, 'w') as jsonfile:
        json.dump({"result": result}, jsonfile)

def save_results_to_json(results, filename):
    import json
    with open(filename, 'w') as jsonfile:
        json.dump(results, jsonfile)

if __name__ == "__main__":
    main()
