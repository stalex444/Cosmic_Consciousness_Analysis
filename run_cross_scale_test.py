#!/usr/bin/env python3
"""
Run the cross-scale correlations test from the Cosmic Consciousness Analyzer.
This script focuses on just running this specific test to debug and verify its functionality.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from cosmic_consciousness_analyzer import CosmicConsciousnessAnalyzer

def main():
    """Run the cross-scale correlations test and display results."""
    print("Initializing Cosmic Consciousness Analyzer...")
    
    # Use data directory
    data_dir = 'data'
    if not os.path.exists(data_dir):
        print(f"Error: Data directory '{data_dir}' not found.")
        if os.path.exists('planck_data'):
            print("Using 'planck_data' directory instead.")
            data_dir = 'planck_data'
        else:
            print("No valid data directory found.")
            sys.exit(1)
    
    print(f"Using data directory: {os.path.abspath(data_dir)}")
    
    # Initialize analyzer with fewer Monte Carlo simulations for faster testing
    try:
        analyzer = CosmicConsciousnessAnalyzer(data_dir=data_dir, monte_carlo_sims=1000)
        print("Analyzer initialized successfully.")
        print(f"Data loaded: {len(analyzer.data['ell'])} multipoles")
    except Exception as e:
        print(f"Error initializing analyzer: {e}")
        print("\nTroubleshooting suggestions:")
        print("1. Check that the data directory contains the required files")
        print("2. Verify that the EE spectrum file is correctly formatted")
        print("3. Ensure the covariance matrix file exists and is valid")
        print("4. Make sure all dependencies are installed (numpy, scipy, matplotlib, astropy)")
        sys.exit(1)
    
    # Run the cross-scale correlations test
    print("\nRunning cross-scale correlations test...")
    try:
        mean_phi_corr, mean_random_corr, z_score, p_value = analyzer.test_cross_scale_correlations()
        
        # Print results
        print("\n=== Cross-Scale Correlations Test Results ===")
        print(f"Mean phi-related correlation: {mean_phi_corr:.4f}")
        print(f"Mean random correlation: {mean_random_corr:.4f}")
        ratio = mean_phi_corr / mean_random_corr if mean_random_corr > 0 else 1.0
        print(f"Ratio: {ratio:.2f}x")
        print(f"Z-score: {z_score:.2f}σ")
        print(f"P-value: {p_value:.8f}")
        
        # Interpret results
        if p_value < 0.01:
            significance = "highly significant"
        elif p_value < 0.05:
            significance = "significant"
        elif p_value < 0.1:
            significance = "marginally significant"
        else:
            significance = "not significant"
        
        if ratio > 5:
            effect = "very strong"
        elif ratio > 2:
            effect = "strong"
        elif ratio > 1.5:
            effect = "moderate"
        elif ratio > 1.1:
            effect = "weak"
        else:
            effect = "negligible"
        
        print(f"\nInterpretation: The test shows a {effect} effect that is {significance}.")
        print(f"Scales separated by powers of the golden ratio show {ratio:.2f}x stronger correlation")
        print(f"than random scale relationships.")
        
        # Create visualization
        plt.figure(figsize=(10, 6))
        
        # Create bar chart comparing phi-related and random correlations
        plt.bar(['Phi-Related', 'Random'], [mean_phi_corr, mean_random_corr], color=['gold', 'gray'], alpha=0.7)
        plt.ylabel('Mean Correlation')
        plt.title(f'Cross-Scale Correlations: {ratio:.2f}x stronger for phi-related scales (p={p_value:.4f})')
        plt.grid(True, alpha=0.3)
        
        # Save and show the figure
        plt.tight_layout()
        plt.savefig('cross_scale_correlations.png')
        print("\nVisualization saved to 'cross_scale_correlations.png'")
        
        # Save results to file
        with open('cross_scale_results.txt', 'w') as f:
            f.write("=== CROSS-SCALE CORRELATIONS TEST RESULTS ===\n\n")
            f.write(f"Mean phi-related correlation: {mean_phi_corr:.4f}\n")
            f.write(f"Mean random correlation: {mean_random_corr:.4f}\n")
            f.write(f"Ratio: {ratio:.2f}x\n")
            f.write(f"Z-score: {z_score:.2f}σ\n")
            f.write(f"P-value: {p_value:.8f}\n\n")
            f.write(f"Interpretation: The test shows a {effect} effect that is {significance}.\n")
            f.write(f"Scales separated by powers of the golden ratio show {ratio:.2f}x stronger correlation ")
            f.write(f"than random scale relationships.\n")
        
        print("Results saved to 'cross_scale_results.txt'")
        
    except Exception as e:
        print(f"Error running cross-scale correlations test: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
