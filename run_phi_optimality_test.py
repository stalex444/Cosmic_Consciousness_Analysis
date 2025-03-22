#!/usr/bin/env python3
"""
Run the phi optimality test for the predictive power analysis.
This script calculates the phi optimality score for the predictive power test results.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from cosmic_consciousness_analyzer import CosmicConsciousnessAnalyzer

def calculate_phi_optimality(observed_value, random_value):
    """
    Calculate phi optimality score bounded between -1 and 1.
    
    Parameters:
    -----------
    observed_value : float
        The observed value from the test
    random_value : float
        The expected value from random chance
        
    Returns:
    --------
    float
        Phi optimality score bounded between -1 and 1
    """
    if random_value == 0:
        return 0.0
        
    # Calculate raw optimality
    raw_optimality = (observed_value - random_value) / random_value
    
    # Bound between -1 and 1
    if raw_optimality > 0:
        # For positive values, scale to [0, 1]
        phi_optimality = min(1.0, raw_optimality / 3.0)  # Divide by 3 to normalize (3x better is considered optimal)
    else:
        # For negative values, scale to [-1, 0]
        phi_optimality = max(-1.0, raw_optimality)
        
    return phi_optimality

def main():
    """Run the predictive power test with phi optimality calculation."""
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
        print("Creating analyzer with 100 Monte Carlo simulations (reduced for speed)...")
        analyzer = CosmicConsciousnessAnalyzer(data_dir=data_dir, monte_carlo_sims=100)
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
    
    # Run the predictive power test
    print("\nRunning predictive power test with phi optimality calculation...")
    try:
        match_rate, mean_random_rate, z_score, p_value, prediction_power = analyzer.test_predictive_power()
        
        # Calculate phi optimality
        phi_optimality = calculate_phi_optimality(match_rate, mean_random_rate)
        
        # Print results
        print("\n=== Predictive Power Test Results with Phi Optimality ===")
        print(f"Match rate for GR predictions: {match_rate:.4f}")
        print(f"Mean match rate for random predictions: {mean_random_rate:.4f}")
        print(f"Prediction power (ratio): {prediction_power:.2f}x")
        print(f"Z-score: {z_score:.2f}")
        print(f"P-value: {p_value:.8f}")
        print(f"Phi optimality: {phi_optimality:.4f}")
        
        # Interpret phi optimality
        if phi_optimality > 0.8:
            phi_interpretation = "extremely high"
        elif phi_optimality > 0.6:
            phi_interpretation = "very high"
        elif phi_optimality > 0.4:
            phi_interpretation = "high"
        elif phi_optimality > 0.2:
            phi_interpretation = "moderate"
        elif phi_optimality > 0:
            phi_interpretation = "slight"
        elif phi_optimality > -0.2:
            phi_interpretation = "slightly negative"
        elif phi_optimality > -0.4:
            phi_interpretation = "moderately negative"
        else:
            phi_interpretation = "strongly negative"
        
        print(f"\nPhi optimality interpretation: {phi_interpretation}")
        print(f"The predictive power test shows a {phi_interpretation} alignment with golden ratio optimality.")
        
        # Create visualization
        plt.figure(figsize=(10, 6))
        
        # Create bar chart comparing GR and random match rates with phi optimality
        plt.bar(['GR Predictions', 'Random Predictions'], [match_rate, mean_random_rate], 
                color=['gold', 'gray'], alpha=0.7)
        plt.ylabel('Match Rate')
        plt.title(f'Predictive Power: φ-optimality = {phi_optimality:.4f} ({phi_interpretation})')
        
        # Add phi optimality as text annotation
        plt.text(0.5, 0.9, f'φ-optimality: {phi_optimality:.4f}', 
                 horizontalalignment='center',
                 transform=plt.gca().transAxes,
                 bbox=dict(facecolor='white', alpha=0.8))
        
        plt.grid(True, alpha=0.3)
        
        # Save and show the figure
        plt.tight_layout()
        plt.savefig('predictive_power_phi_optimality.png')
        print("\nVisualization saved to 'predictive_power_phi_optimality.png'")
        
        # Save results to file
        with open('predictive_power_phi_optimality_results.txt', 'w') as f:
            f.write("=== PREDICTIVE POWER TEST RESULTS WITH PHI OPTIMALITY ===\n\n")
            f.write(f"Match rate for GR predictions: {match_rate:.4f}\n")
            f.write(f"Mean match rate for random predictions: {mean_random_rate:.4f}\n")
            f.write(f"Prediction power (ratio): {prediction_power:.2f}x\n")
            f.write(f"Z-score: {z_score:.2f}\n")
            f.write(f"P-value: {p_value:.8f}\n")
            f.write(f"Phi optimality: {phi_optimality:.4f}\n\n")
            f.write(f"Phi optimality interpretation: {phi_interpretation}\n")
            f.write(f"The predictive power test shows a {phi_interpretation} alignment with golden ratio optimality.\n")
        
        print("Results saved to 'predictive_power_phi_optimality_results.txt'")
        
    except Exception as e:
        print(f"Error running predictive power test: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
