#!/usr/bin/env python3
"""
Script to run all additional tests in the CosmicConsciousnessAnalyzer class.
This script runs the tests that weren't part of the original 10 core tests.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from cosmic_consciousness_analyzer import CosmicConsciousnessAnalyzer

def run_test(analyzer, test_name, test_method, results_dir):
    """Run a test and save the results."""
    print("\n" + "="*80)
    print(f"Running {test_name}...")
    print("="*80)
    
    try:
        # Create test-specific directory
        test_dir = os.path.join(results_dir, test_name.lower().replace(" ", "_"))
        os.makedirs(test_dir, exist_ok=True)
        
        # Run the test
        result = test_method()
        
        # Save results to file
        with open(os.path.join(test_dir, "results.txt"), "w") as f:
            f.write(f"{test_name} Results\n")
            f.write("="*len(test_name + " Results") + "\n\n")
            
            if isinstance(result, tuple):
                if len(result) >= 4 and all(isinstance(x, (int, float)) for x in result[:4]):
                    # Assume standard format: metric, random_metric, z_score, p_value
                    metric, random_metric, z_score, p_value = result[:4]
                    f.write(f"Metric Value: {metric:.6f}\n")
                    f.write(f"Random Expectation: {random_metric:.6f}\n")
                    f.write(f"Z-Score: {z_score:.6f}\n")
                    f.write(f"P-Value: {p_value:.6f}\n")
                    f.write(f"Significant: {p_value < 0.05}\n")
                    
                    # Calculate phi-optimality
                    if p_value < 1e-10:
                        phi_optimality = 1.0
                    elif p_value > 0.9:
                        phi_optimality = -1.0
                    else:
                        phi_optimality = 1.0 - 2.0 * p_value
                    f.write(f"Phi-Optimality: {phi_optimality:.6f}\n")
                else:
                    # Generic tuple result
                    for i, value in enumerate(result):
                        f.write(f"Result {i+1}: {value}\n")
            else:
                # Single value result
                f.write(f"Result: {result}\n")
        
        print(f"{test_name} completed successfully.")
        print(f"Results saved to {test_dir}/results.txt")
        return True
    except Exception as e:
        print(f"Error running {test_name}: {str(e)}")
        return False

def main():
    # Initialize the analyzer
    print("Initializing CosmicConsciousnessAnalyzer...")
    analyzer = CosmicConsciousnessAnalyzer(monte_carlo_sims=100)  # Reduced for faster testing
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("results", f"additional_tests_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    # Define the additional tests to run
    additional_tests = [
        ("Cross-Scale Correlations", analyzer.test_cross_scale_correlations),
        ("Pattern Persistence", analyzer.test_pattern_persistence),
        ("Predictive Power", analyzer.test_predictive_power),
        ("Optimization", analyzer.test_optimization),
        ("Golden Symmetries", analyzer.test_golden_symmetries),
        ("Phi Network", analyzer.test_phi_network),
        ("Spectral Gap", analyzer.test_spectral_gap),
        ("Recurrence Quantification", analyzer.test_recurrence_quantification),
        ("Scale-Frequency Coupling", analyzer.test_scale_frequency_coupling),
        ("Multi-Scale Coherence", analyzer.test_multi_scale_coherence),
        ("Coherence Phase", analyzer.test_coherence_phase),
        ("Extended Meta-Coherence", analyzer.test_extended_meta_coherence)
    ]
    
    # Run each test
    successful_tests = 0
    for test_name, test_method in additional_tests:
        if run_test(analyzer, test_name, test_method, results_dir):
            successful_tests += 1
    
    # Print summary
    print("\n" + "="*80)
    print(f"Testing complete: {successful_tests}/{len(additional_tests)} tests successful")
    print("="*80)
    print(f"Results saved to {results_dir}")

if __name__ == "__main__":
    main()
