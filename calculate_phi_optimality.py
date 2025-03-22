#!/usr/bin/env python3
"""
Calculate phi optimality from existing predictive power test results.
This script reads the results from the predictive_power_results.txt file
and calculates the phi optimality score.
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt

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

def interpret_phi_optimality(phi_optimality):
    """Interpret phi optimality score with descriptive text."""
    if phi_optimality > 0.8:
        return "extremely high"
    elif phi_optimality > 0.6:
        return "very high"
    elif phi_optimality > 0.4:
        return "high"
    elif phi_optimality > 0.2:
        return "moderate"
    elif phi_optimality > 0:
        return "slight"
    elif phi_optimality > -0.2:
        return "slightly negative"
    elif phi_optimality > -0.4:
        return "moderately negative"
    else:
        return "strongly negative"

def main():
    """Calculate phi optimality from existing results."""
    print("Calculating phi optimality from existing predictive power test results...")
    
    # Check if results file exists
    results_file = 'predictive_power_results.txt'
    if not os.path.exists(results_file):
        print(f"Error: Results file '{results_file}' not found.")
        print("Please run run_predictive_power_test.py first to generate results.")
        return
    
    # Read results from file
    with open(results_file, 'r') as f:
        results_text = f.read()
    
    # Extract match rate and mean random rate
    match_rate_match = re.search(r'Match rate for GR predictions: ([\d\.]+)', results_text)
    random_rate_match = re.search(r'Mean match rate for random predictions: ([\d\.]+)', results_text)
    prediction_power_match = re.search(r'Prediction power \(ratio\): ([\d\.]+)x', results_text)
    
    if not match_rate_match or not random_rate_match or not prediction_power_match:
        print("Error: Could not extract results from file.")
        print("Please ensure the results file is properly formatted.")
        return
    
    match_rate = float(match_rate_match.group(1))
    mean_random_rate = float(random_rate_match.group(1))
    prediction_power = float(prediction_power_match.group(1))
    
    # Calculate phi optimality
    phi_optimality = calculate_phi_optimality(match_rate, mean_random_rate)
    phi_interpretation = interpret_phi_optimality(phi_optimality)
    
    # Print results
    print("\n=== Phi Optimality Results ===")
    print(f"Match rate for GR predictions: {match_rate:.4f}")
    print(f"Mean match rate for random predictions: {mean_random_rate:.4f}")
    print(f"Prediction power (ratio): {prediction_power:.2f}x")
    print(f"Phi optimality: {phi_optimality:.4f}")
    print(f"Interpretation: {phi_interpretation}")
    
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
        f.write(f"Phi optimality: {phi_optimality:.4f}\n\n")
        f.write(f"Phi optimality interpretation: {phi_interpretation}\n")
        f.write(f"The predictive power test shows a {phi_interpretation} alignment with golden ratio optimality.\n")
    
    print("Results saved to 'predictive_power_phi_optimality_results.txt'")

if __name__ == "__main__":
    main()
