#!/usr/bin/env python3
"""
Run the golden symmetries test from the CosmicConsciousnessAnalyzer.
This script tests for symmetries in the CMB data related to the golden ratio
and compares with other mathematical constants.
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from cosmic_consciousness_analyzer import CosmicConsciousnessAnalyzer
from calculate_phi_optimality import calculate_phi_optimality, interpret_phi_optimality

def main():
    """Run the golden symmetries test."""
    # Set data directory
    data_dir = os.path.join(os.getcwd(), 'planck_data')
    if not os.path.exists(data_dir):
        print(f"Error: Data directory '{data_dir}' not found.")
        sys.exit(1)
    
    print("=== GOLDEN SYMMETRIES TEST ===")
    print(f"Using data directory: {data_dir}")
    
    # Initialize analyzer with 1000 Monte Carlo simulations for faster testing
    print("Creating analyzer with 1000 Monte Carlo simulations...")
    start_time = time.time()
    analyzer = CosmicConsciousnessAnalyzer(data_dir=data_dir, monte_carlo_sims=1000)
    print(f"Analyzer initialized in {time.time() - start_time:.2f} seconds.")
    print(f"Data loaded: {len(analyzer.data['ell'])} multipoles")
    
    # Run the golden symmetries test
    print("\nRunning golden symmetries test...")
    start_time = time.time()
    
    mean_asymmetry, mean_alternative, z_score, p_value, symmetry_ratio = analyzer.test_golden_symmetries()
    
    print(f"Test completed in {time.time() - start_time:.2f} seconds.")
    
    # Calculate phi optimality
    phi_optimality = calculate_phi_optimality(symmetry_ratio, 1.0)
    phi_interpretation = interpret_phi_optimality(phi_optimality)
    
    # Print results
    print("\n=== RESULTS ===")
    print(f"Mean asymmetry for golden ratio: {mean_asymmetry:.4f}")
    print(f"Mean asymmetry for alternative constants: {mean_alternative:.4f}")
    print(f"Z-score: {z_score:.4f}")
    print(f"P-value: {p_value:.4f}")
    print(f"Symmetry ratio: {symmetry_ratio:.2f}x")
    print(f"Phi optimality: {phi_optimality:.4f}")
    print(f"Interpretation: {phi_interpretation}")
    
    # Create visualization
    print("\nCreating visualization...")
    
    # Create bar chart comparing golden ratio and alternative constants
    plt.figure(figsize=(10, 6))
    
    # Define constants and their names
    constants = ["Golden Ratio (φ)", "e", "π", "2"]
    
    # Run the test for each alternative constant individually
    phi_asymmetry = mean_asymmetry
    e_asymmetry = 0
    pi_asymmetry = 0
    two_asymmetry = 0
    
    # Use the alternative asymmetries already calculated
    alternative_constants = [np.e, np.pi, 2]
    alt_asymmetries = []
    
    for constant in alternative_constants:
        alt_fold = []
        for i in range(len(analyzer.data['ell'])):
            l_value = analyzer.data['ell'][i]
            power = analyzer.data['ee_power'][i]
            
            l_const = l_value * constant
            idx_const = np.abs(analyzer.data['ell'] - l_const).argmin()
            
            l_inv_const = l_value / constant
            idx_inv_const = np.abs(analyzer.data['ell'] - l_inv_const).argmin()
            
            if idx_const < len(analyzer.data['ell']) and idx_inv_const < len(analyzer.data['ell']):
                power_const = analyzer.data['ee_power'][idx_const]
                power_inv_const = analyzer.data['ee_power'][idx_inv_const]
                
                expected_power = np.sqrt(power_const * power_inv_const)
                symmetry_ratio = power / expected_power if expected_power != 0 else 1
                
                alt_fold.append(abs(1 - symmetry_ratio))
        
        alt_asymmetries.append(np.mean(alt_fold))
    
    if len(alternative_constants) == 3 and len(alt_asymmetries) == 3:
        e_asymmetry = alt_asymmetries[0]
        pi_asymmetry = alt_asymmetries[1]
        two_asymmetry = alt_asymmetries[2]
    
    # Combine asymmetries
    asymmetries = [phi_asymmetry, e_asymmetry, pi_asymmetry, two_asymmetry]
    
    # Plot bar chart
    plt.bar(constants, asymmetries, color=['gold', 'gray', 'gray', 'gray'], alpha=0.7)
    plt.ylabel('Mean Asymmetry (lower is better)')
    plt.title(f'Golden Ratio Symmetry Test: φ-optimality = {phi_optimality:.4f} ({phi_interpretation})')
    
    # Add phi optimality as text annotation
    plt.text(0.5, 0.9, f'φ-optimality: {phi_optimality:.4f}', 
             horizontalalignment='center',
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8))
    
    # Add p-value and symmetry ratio
    plt.text(0.5, 0.82, f'p-value: {p_value:.4f}', 
             horizontalalignment='center',
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.text(0.5, 0.74, f'Symmetry ratio: {symmetry_ratio:.2f}x', 
             horizontalalignment='center',
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.grid(True, alpha=0.3)
    
    # Save and show the figure
    plt.tight_layout()
    plt.savefig('golden_symmetries_test.png')
    print("Visualization saved to 'golden_symmetries_test.png'")
    
    # Save results to file
    with open('golden_symmetries_results.txt', 'w') as f:
        f.write("=== GOLDEN SYMMETRIES TEST RESULTS ===\n\n")
        f.write(f"Mean asymmetry for golden ratio: {mean_asymmetry:.4f}\n")
        f.write(f"Mean asymmetry for alternative constants: {mean_alternative:.4f}\n")
        f.write(f"Z-score: {z_score:.4f}\n")
        f.write(f"P-value: {p_value:.4f}\n")
        f.write(f"Symmetry ratio: {symmetry_ratio:.2f}x\n")
        f.write(f"Phi optimality: {phi_optimality:.4f}\n\n")
        f.write(f"Phi optimality interpretation: {phi_interpretation}\n")
        f.write(f"The golden symmetries test shows a {phi_interpretation} alignment with golden ratio optimality.\n")
    
    print("Results saved to 'golden_symmetries_results.txt'")

if __name__ == "__main__":
    main()
