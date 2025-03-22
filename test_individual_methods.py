#!/usr/bin/env python3
"""
Test script to verify each method of the CosmicConsciousnessAnalyzer individually.
This helps identify any issues with specific methods before running the full analysis.
"""

import os
import sys
import traceback
import numpy as np
import matplotlib.pyplot as plt
from cosmic_consciousness_analyzer import CosmicConsciousnessAnalyzer

def test_method(analyzer, method_name, method_func):
    """Test a specific method and report the result."""
    print(f"\n{'='*50}")
    print(f"Testing {method_name}...")
    print(f"{'='*50}")
    
    try:
        result = method_func()
        print(f" {method_name} completed successfully")
        return True, result
    except Exception as e:
        print(f" {method_name} failed with error: {e}")
        print("\nTraceback:")
        traceback.print_exc()
        return False, None

def main():
    """Main function to test all methods individually."""
    
    # Check if data directory exists
    data_dir = "./data"
    if not os.path.exists(data_dir):
        if os.path.exists("./planck_data"):
            data_dir = "./planck_data"
        else:
            print("No data directory found. Please run create_sample_data.py first.")
            sys.exit(1)
    
    # Get absolute path to data directory
    data_dir = os.path.abspath(data_dir)
    
    # Create results directory
    results_dir = "./test_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize the analyzer with a small number of Monte Carlo iterations for faster testing
    try:
        analyzer = CosmicConsciousnessAnalyzer(data_dir=data_dir, monte_carlo_sims=100)
        print(f"Analyzer initialized successfully with data from {data_dir}")
    except Exception as e:
        print(f"Failed to initialize analyzer: {e}")
        sys.exit(1)
    
    # Change to results directory for saving output
    os.chdir(results_dir)
    
    # Define all test methods
    test_methods = [
        ("Golden Ratio Significance", analyzer.test_gr_significance),
        ("Coherence", analyzer.test_coherence),
        ("GR-Specific Coherence", analyzer.test_gr_coherence),
        ("Hierarchical Organization", analyzer.test_hierarchical_organization),
        ("Information Integration", analyzer.test_information_integration),
        ("Optimization", analyzer.test_optimization),
        ("Resonance", analyzer.test_resonance),
        ("Fractal Structure", analyzer.test_fractal_structure),
        ("Meta-Coherence", analyzer.test_meta_coherence),
        ("Multi-Scale Patterns", analyzer.test_multiscale_patterns),
        ("Peak Frequency Analysis", analyzer.analyze_specific_frequencies)
    ]
    
    # Test each method
    results = {}
    success_count = 0
    
    for name, func in test_methods:
        success, result = test_method(analyzer, name, func)
        results[name] = (success, result)
        if success:
            success_count += 1
    
    # Report overall results
    print(f"\n{'='*50}")
    print(f"Test Summary: {success_count}/{len(test_methods)} methods passed")
    print(f"{'='*50}")
    
    for name, (success, _) in results.items():
        status = " Passed" if success else " Failed"
        print(f"{name}: {status}")
    
    # If all tests pass, try running the comprehensive analysis
    if success_count == len(test_methods):
        print("\nAll individual tests passed. Attempting comprehensive analysis...")
        try:
            analyzer.run_comprehensive_analysis()
            print(" Comprehensive analysis completed successfully")
        except Exception as e:
            print(f" Comprehensive analysis failed with error: {e}")
            traceback.print_exc()
    
    print(f"\nTest results saved to {results_dir}")

if __name__ == "__main__":
    main()
