#!/usr/bin/env python3
"""
Run the spectral gap test from the CosmicConsciousnessAnalyzer.
This script tests if the spectral gap in CMB data shows golden ratio optimization.
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
    """Run the spectral gap test."""
    # Set data directory
    data_dir = os.path.join(os.getcwd(), 'planck_data')
    if not os.path.exists(data_dir):
        print(f"Error: Data directory '{data_dir}' not found.")
        sys.exit(1)
    
    print("=== SPECTRAL GAP TEST ===")
    print(f"Using data directory: {data_dir}")
    
    # Initialize analyzer with 1000 Monte Carlo simulations for faster testing
    print("Creating analyzer with 1000 Monte Carlo simulations...")
    start_time = time.time()
    analyzer = CosmicConsciousnessAnalyzer(data_dir=data_dir, monte_carlo_sims=1000)
    print(f"Analyzer initialized in {time.time() - start_time:.2f} seconds.")
    print(f"Data loaded: {len(analyzer.data['ell'])} multipoles")
    
    # Run the spectral gap test
    print("\nRunning spectral gap test...")
    start_time = time.time()
    
    (spectral_gap, mean_random_gap, gap_z_score, gap_p_value, gap_ratio, 
     mean_phi_deviation, mean_random_dev, dev_z_score, dev_p_value, dev_ratio) = analyzer.test_spectral_gap()
    
    print(f"Test completed in {time.time() - start_time:.2f} seconds.")
    
    # Calculate phi optimality for both metrics
    gap_phi_optimality = calculate_phi_optimality(gap_ratio, 1.0)
    gap_interpretation = interpret_phi_optimality(gap_phi_optimality)
    
    dev_phi_optimality = calculate_phi_optimality(dev_ratio, 1.0)
    dev_interpretation = interpret_phi_optimality(dev_phi_optimality)
    
    # Print results
    print("\n=== RESULTS ===")
    print("1. Spectral Gap Analysis:")
    print(f"   Spectral gap: {spectral_gap:.4f}")
    print(f"   Random gap: {mean_random_gap:.4f}")
    print(f"   Z-score: {gap_z_score:.4f}")
    print(f"   P-value: {gap_p_value:.4f}")
    print(f"   Gap ratio: {gap_ratio:.2f}x")
    print(f"   Phi optimality: {gap_phi_optimality:.4f}")
    print(f"   Interpretation: {gap_interpretation}")
    
    print("\n2. Eigenvalue Phi Relationship Analysis:")
    print(f"   Mean phi deviation: {mean_phi_deviation:.4f}")
    print(f"   Random deviation: {mean_random_dev:.4f}")
    print(f"   Z-score: {dev_z_score:.4f}")
    print(f"   P-value: {dev_p_value:.4f}")
    print(f"   Deviation ratio: {dev_ratio:.2f}x")
    print(f"   Phi optimality: {dev_phi_optimality:.4f}")
    print(f"   Interpretation: {dev_interpretation}")
    
    # Create visualization
    print("\nCreating visualization...")
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # 1. Spectral Gap Visualization
    # Create similarity matrix for visualization
    power_spectrum = np.copy(analyzer.data['ee_power'])
    # Replace zeros and negative values with small positive values
    min_positive = np.min(power_spectrum[power_spectrum > 0]) / 10.0 if np.any(power_spectrum > 0) else 1e-10
    power_spectrum[power_spectrum <= 0] = min_positive
    
    n = len(power_spectrum)
    similarity_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            # Calculate similarity as inverse of normalized difference
            diff = abs(power_spectrum[i] - power_spectrum[j])
            mean_val = (power_spectrum[i] + power_spectrum[j]) / 2.0
            if mean_val > 0:
                similarity_matrix[i, j] = 1.0 / (1.0 + diff / mean_val)
            else:
                similarity_matrix[i, j] = 0.0
    
    # Ensure the matrix is symmetric and positive definite
    similarity_matrix = (similarity_matrix + similarity_matrix.T) / 2.0
    np.fill_diagonal(similarity_matrix, 1.0)
    
    im = ax1.imshow(similarity_matrix, cmap='viridis')
    ax1.set_title(f'Similarity Matrix with Spectral Gap: {spectral_gap:.4f}\n(φ-optimality: {gap_phi_optimality:.4f})')
    ax1.set_xlabel('Multipole Index')
    ax1.set_ylabel('Multipole Index')
    plt.colorbar(im, ax=ax1, label='Similarity')
    
    # 2. Eigenvalue Ratios Visualization
    # Calculate eigenvalues
    eigenvalues = np.linalg.eigvals(similarity_matrix)
    eigenvalues = np.sort(np.real(eigenvalues))[::-1]  # Sort in descending order, take real part
    
    # Calculate ratios
    eigenvalue_ratios = []
    for i in range(len(eigenvalues)-1):
        if eigenvalues[i+1] != 0:  # Avoid division by zero
            ratio = eigenvalues[i] / eigenvalues[i+1]
            eigenvalue_ratios.append(ratio)
    
    # Plot top 10 eigenvalues
    top_n = min(10, len(eigenvalues))
    ax2.bar(range(top_n), eigenvalues[:top_n], color='skyblue', alpha=0.7)
    ax2.set_title(f'Top {top_n} Eigenvalues\n(Phi Deviation: {mean_phi_deviation:.4f}, φ-optimality: {dev_phi_optimality:.4f})')
    ax2.set_xlabel('Eigenvalue Index')
    ax2.set_ylabel('Eigenvalue')
    
    # Add text annotations for phi relationships
    phi = analyzer.phi
    for i in range(min(5, len(eigenvalue_ratios))):
        ratio = eigenvalue_ratios[i]
        deviations = [abs(ratio - phi**power) for power in range(1, 4)]
        min_dev = min(deviations)
        power = deviations.index(min_dev) + 1
        
        ax2.text(i, eigenvalues[i], 
                f'λ{i}/λ{i+1}={ratio:.2f}\n(φ^{power}={phi**power:.2f})', 
                ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('spectral_gap_test.png')
    print("Visualization saved to 'spectral_gap_test.png'")
    
    # Save results to file
    with open('spectral_gap_results.txt', 'w') as f:
        f.write("=== SPECTRAL GAP TEST RESULTS ===\n\n")
        f.write("1. Spectral Gap Analysis:\n")
        f.write(f"   Spectral gap: {spectral_gap:.4f}\n")
        f.write(f"   Random gap: {mean_random_gap:.4f}\n")
        f.write(f"   Z-score: {gap_z_score:.4f}\n")
        f.write(f"   P-value: {gap_p_value:.4f}\n")
        f.write(f"   Gap ratio: {gap_ratio:.2f}x\n")
        f.write(f"   Phi optimality: {gap_phi_optimality:.4f}\n")
        f.write(f"   Interpretation: {gap_interpretation}\n\n")
        
        f.write("2. Eigenvalue Phi Relationship Analysis:\n")
        f.write(f"   Mean phi deviation: {mean_phi_deviation:.4f}\n")
        f.write(f"   Random deviation: {mean_random_dev:.4f}\n")
        f.write(f"   Z-score: {dev_z_score:.4f}\n")
        f.write(f"   P-value: {dev_p_value:.4f}\n")
        f.write(f"   Deviation ratio: {dev_ratio:.2f}x\n")
        f.write(f"   Phi optimality: {dev_phi_optimality:.4f}\n")
        f.write(f"   Interpretation: {dev_interpretation}\n\n")
        
        f.write("Summary:\n")
        f.write(f"The spectral gap test shows a {gap_interpretation} alignment with golden ratio optimality in the gap structure.\n")
        f.write(f"The eigenvalue ratio analysis shows a {dev_interpretation} alignment with golden ratio patterns in eigenvalue relationships.\n")
    
    print("Results saved to 'spectral_gap_results.txt'")

if __name__ == "__main__":
    main()
