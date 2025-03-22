#!/usr/bin/env python3
"""
Script to run a single test from the CosmicConsciousnessAnalyzer class.
"""

import os
import sys
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
    # Check for test name argument
    if len(sys.argv) < 2:
        print("Usage: python3 run_single_test.py <test_number>")
        print("Available tests:")
        print("1. Cross-Scale Correlations")
        print("2. Pattern Persistence")
        print("3. Predictive Power")
        print("4. Optimization")
        print("5. Golden Symmetries")
        print("6. Phi Network")
        print("7. Spectral Gap")
        print("8. Recurrence Quantification")
        print("9. Scale-Frequency Coupling")
        print("10. Multi-Scale Coherence")
        print("11. Coherence Phase")
        print("12. Extended Meta-Coherence")
        return
    
    try:
        test_number = int(sys.argv[1])
    except ValueError:
        print("Please provide a valid test number (1-12)")
        return
    
    # Define the additional tests
    additional_tests = [
        ("Cross-Scale Correlations", "test_cross_scale_correlations"),
        ("Pattern Persistence", "test_pattern_persistence"),
        ("Predictive Power", "test_predictive_power"),
        ("Optimization", "test_optimization"),
        ("Golden Symmetries", "test_golden_symmetries"),
        ("Phi Network", "test_phi_network"),
        ("Spectral Gap", "test_spectral_gap"),
        ("Recurrence Quantification", "test_recurrence_quantification"),
        ("Scale-Frequency Coupling", "test_scale_frequency_coupling"),
        ("Multi-Scale Coherence", "test_multi_scale_coherence"),
        ("Coherence Phase", "test_coherence_phase"),
        ("Extended Meta-Coherence", "test_extended_meta_coherence")
    ]
    
    if test_number < 1 or test_number > len(additional_tests):
        print(f"Please provide a valid test number (1-{len(additional_tests)})")
        return
    
    # Get the selected test
    test_name, test_method_name = additional_tests[test_number - 1]
    
    # Initialize the analyzer
    print("Initializing CosmicConsciousnessAnalyzer...")
    analyzer = CosmicConsciousnessAnalyzer(monte_carlo_sims=50)  # Reduced for faster testing
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("results", f"test_{test_number}_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    # Get the test method
    test_method = getattr(analyzer, test_method_name)
    
    # Run the test
    run_test(analyzer, test_name, test_method, results_dir)

if __name__ == "__main__":
    main()
