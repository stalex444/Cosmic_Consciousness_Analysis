#!/usr/bin/env python3
"""
Script to run a single test from the CosmicConsciousnessAnalyzer with timeout protection.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import signal
import traceback
from cosmic_consciousness_analyzer import CosmicConsciousnessAnalyzer

class TimeoutError(Exception):
    """Exception raised when a function times out."""
    pass

def timeout_handler(signum, frame):
    """Handler for timeout signal."""
    raise TimeoutError("Function timed out")

def run_with_timeout(func, timeout=300, *args, **kwargs):
    """Run a function with a timeout."""
    # Set the timeout handler
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    
    try:
        result = func(*args, **kwargs)
        signal.alarm(0)  # Disable the alarm
        return result
    except TimeoutError:
        print(f"Function timed out after {timeout} seconds")
        raise
    finally:
        signal.alarm(0)  # Ensure the alarm is disabled

def run_test(analyzer, test_name, test_method, results_dir, timeout=300):
    """Run a test and save the results."""
    print("\n" + "="*80)
    print(f"Running {test_name}...")
    print("="*80)
    
    try:
        # Create test-specific directory
        test_dir = os.path.join(results_dir, test_name.lower().replace(" ", "_"))
        os.makedirs(test_dir, exist_ok=True)
        
        # Run the test with timeout
        print(f"Starting test with {timeout} second timeout...")
        result = run_with_timeout(test_method, timeout)
        
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
                    
                    # Create visualization
                    plt.figure(figsize=(10, 6))
                    plt.bar(['Test Metric', 'Random Expectation'], 
                           [metric, random_metric],
                           color=['gold', 'gray'])
                    plt.ylabel('Metric Value')
                    plt.title(f'{test_name} Test Results')
                    plt.annotate(f'p-value: {p_value:.6f}', xy=(0.5, 0.9), 
                                xycoords='axes fraction', ha='center')
                    plt.annotate(f'Significant: {p_value < 0.05}', xy=(0.5, 0.85), 
                                xycoords='axes fraction', ha='center')
                    plt.tight_layout()
                    plt.savefig(os.path.join(test_dir, f'{test_name.lower().replace(" ", "_")}.png'))
                    plt.close()
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
    except TimeoutError:
        print(f"ERROR: {test_name} timed out after {timeout} seconds")
        with open(os.path.join(test_dir, "error.txt"), "w") as f:
            f.write(f"{test_name} Error\n")
            f.write("="*len(test_name + " Error") + "\n\n")
            f.write(f"Test timed out after {timeout} seconds\n")
        return False
    except Exception as e:
        print(f"ERROR: {test_name} failed with error: {str(e)}")
        traceback.print_exc()
        with open(os.path.join(test_dir, "error.txt"), "w") as f:
            f.write(f"{test_name} Error\n")
            f.write("="*len(test_name + " Error") + "\n\n")
            f.write(f"Error: {str(e)}\n\n")
            f.write("Traceback:\n")
            f.write(traceback.format_exc())
        return False

def main():
    # Check for test name argument
    if len(sys.argv) < 2:
        print("Usage: python3 run_test_with_timeout.py <test_number> [timeout_seconds]")
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
        print("13. Multi-Scale Patterns")
        return
    
    try:
        test_number = int(sys.argv[1])
        timeout = int(sys.argv[2]) if len(sys.argv) > 2 else 120  # Default timeout: 2 minutes
    except ValueError:
        print("Please provide a valid test number (1-13) and optional timeout in seconds")
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
        ("Extended Meta-Coherence", "test_extended_meta_coherence"),
        ("Multi-Scale Patterns", "test_multi_scale_patterns")
    ]
    
    if test_number < 1 or test_number > len(additional_tests):
        print(f"Please provide a valid test number (1-{len(additional_tests)})")
        return
    
    # Get the selected test
    test_name, test_method_name = additional_tests[test_number - 1]
    
    # Initialize the analyzer with reduced Monte Carlo simulations for faster testing
    print("Initializing CosmicConsciousnessAnalyzer...")
    print(f"Using timeout of {timeout} seconds")
    analyzer = CosmicConsciousnessAnalyzer(monte_carlo_sims=50)  # Reduced for faster testing
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("results", f"test_{test_number}_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    # Get the test method
    test_method = getattr(analyzer, test_method_name)
    
    # Run the test
    run_test(analyzer, test_name, test_method, results_dir, timeout)

if __name__ == "__main__":
    main()
