#!/usr/bin/env python3
"""
Run the pattern persistence test from the Cosmic Consciousness Analyzer.
This script focuses on just running this specific test to debug and verify its functionality.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from cosmic_consciousness_analyzer import CosmicConsciousnessAnalyzer

def main():
    """Run the pattern persistence test and display results."""
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
    
    # Run the pattern persistence test
    print("\nRunning pattern persistence test...")
    try:
        mean_gr_strength, mean_random, z_score, p_value, persistence_ratio = analyzer.test_pattern_persistence()
        
        # Print results
        print("\n=== Pattern Persistence Test Results ===")
        print(f"Mean GR pattern strength: {mean_gr_strength:.4f}")
        print(f"Mean random pattern strength: {mean_random:.4f}")
        ratio = mean_gr_strength / mean_random if mean_random > 0 else 1.0
        print(f"Strength ratio: {ratio:.2f}x")
        print(f"Z-score: {z_score:.2f}")
        print(f"P-value: {p_value:.8f}")
        print(f"Persistence ratio: {persistence_ratio:.2f}")
        
        # Interpret results
        if p_value < 0.01:
            significance = "highly significant"
        elif p_value < 0.05:
            significance = "significant"
        elif p_value < 0.1:
            significance = "marginally significant"
        else:
            significance = "not significant"
        
        if ratio > 2:
            effect = "strong"
        elif ratio > 1.5:
            effect = "moderate"
        elif ratio > 1.1:
            effect = "weak"
        else:
            effect = "negligible"
        
        persistence_desc = "highly persistent" if persistence_ratio > 2 else \
                          "moderately persistent" if persistence_ratio > 1.5 else \
                          "slightly persistent" if persistence_ratio > 1.1 else "not persistent"
        
        print(f"\nInterpretation: The test shows a {effect} effect that is {significance}.")
        print(f"Golden ratio patterns are {persistence_ratio:.2f}x more consistent across different")
        print(f"subsets of the data than random patterns, indicating they are {persistence_desc}.")
        
        # Create visualization
        plt.figure(figsize=(12, 6))
        
        # Plot 1: Strength comparison
        plt.subplot(1, 2, 1)
        plt.bar(['GR Patterns', 'Random'], [mean_gr_strength, mean_random], color=['gold', 'gray'], alpha=0.7)
        plt.ylabel('Pattern Strength')
        plt.title(f'Pattern Strength: {ratio:.2f}x stronger for GR patterns (p={p_value:.4f})')
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Persistence comparison
        plt.subplot(1, 2, 2)
        # Higher persistence ratio means lower variance in GR patterns compared to random
        plt.bar(['GR Patterns', 'Random'], [1, persistence_ratio], color=['gold', 'gray'], alpha=0.7)
        plt.ylabel('Relative Variance')
        plt.title(f'Pattern Persistence: {persistence_ratio:.2f}x more consistent')
        plt.grid(True, alpha=0.3)
        
        # Save and show the figure
        plt.tight_layout()
        plt.savefig('pattern_persistence.png')
        print("\nVisualization saved to 'pattern_persistence.png'")
        
        # Save results to file
        with open('pattern_persistence_results.txt', 'w') as f:
            f.write("=== PATTERN PERSISTENCE TEST RESULTS ===\n\n")
            f.write(f"Mean GR pattern strength: {mean_gr_strength:.4f}\n")
            f.write(f"Mean random pattern strength: {mean_random:.4f}\n")
            f.write(f"Strength ratio: {ratio:.2f}x\n")
            f.write(f"Z-score: {z_score:.2f}\n")
            f.write(f"P-value: {p_value:.8f}\n")
            f.write(f"Persistence ratio: {persistence_ratio:.2f}\n\n")
            f.write(f"Interpretation: The test shows a {effect} effect that is {significance}.\n")
            f.write(f"Golden ratio patterns are {persistence_ratio:.2f}x more consistent across different ")
            f.write(f"subsets of the data than random patterns, indicating they are {persistence_desc}.\n")
        
        print("Results saved to 'pattern_persistence_results.txt'")
        
    except Exception as e:
        print(f"Error running pattern persistence test: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
