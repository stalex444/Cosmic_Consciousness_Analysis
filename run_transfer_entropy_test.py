#!/usr/bin/env python3
"""
Run the transfer entropy test from the CosmicConsciousnessAnalyzer class.
This script tests if information transfer across scales follows golden ratio patterns.
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
    """Run the transfer entropy test."""
    # Set data directory
    data_dir = os.path.join(os.getcwd(), 'planck_data')
    if not os.path.exists(data_dir):
        print(f"Error: Data directory '{data_dir}' not found.")
        sys.exit(1)
    
    print("=== TRANSFER ENTROPY ANALYSIS ===")
    print(f"Using data directory: {data_dir}")
    
    # Create analyzer
    print("Creating analyzer with 100 Monte Carlo simulations...")
    start_time = time.time()
    analyzer = CosmicConsciousnessAnalyzer(data_dir=data_dir, monte_carlo_sims=100)
    print(f"Analyzer initialized in {time.time() - start_time:.2f} seconds.")
    print(f"Data loaded: {len(analyzer.data['ell'])} multipoles")
    print()
    
    # Run the test
    print("Running transfer entropy analysis...")
    start_time = time.time()
    result = analyzer.test_transfer_entropy()
    
    if result is None:
        print("Error: Not enough data for transfer entropy analysis.")
        sys.exit(1)
        
    phi_te, non_phi_te, te_ratio, z_score, p_value = result
    print(f"Test completed in {time.time() - start_time:.2f} seconds.")
    print()
    
    # Calculate phi optimality
    phi_optimality = calculate_phi_optimality(te_ratio, 1.0)
    interpretation = interpret_phi_optimality(phi_optimality)
    
    # Print results
    print("=== RESULTS ===")
    print(f"1. Phi-based scales transfer entropy: {phi_te:.6f}")
    print(f"2. Random scales transfer entropy: {non_phi_te:.6f}")
    print(f"3. Transfer entropy ratio: {te_ratio:.2f}x")
    print(f"4. Statistical significance:")
    print(f"   Z-score: {z_score:.4f}")
    print(f"   P-value: {p_value:.4f}")
    print(f"5. Phi optimality: {phi_optimality:.4f} ({interpretation})")
    print()
    
    # Create visualization
    print("Creating visualization...")
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Transfer entropy comparison
    plt.subplot(2, 1, 1)
    bars = plt.bar(['Phi-based Scales', 'Random Scales'], [phi_te, non_phi_te], 
                   color=['blue', 'gray'], alpha=0.7)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                 f'{height:.6f}', ha='center', va='bottom')
    
    plt.ylabel('Transfer Entropy (bits)')
    plt.title(f'Transfer Entropy Comparison (Ratio: {te_ratio:.2f}x, p={p_value:.4f})')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Distribution of random transfer entropies with phi-based value
    plt.subplot(2, 1, 2)
    
    # Generate a normal distribution based on mean and std of non_phi_entropies
    x = np.linspace(non_phi_te - 3*z_score, non_phi_te + 3*z_score, 1000)
    y = stats.norm.pdf(x, non_phi_te, non_phi_te/z_score if z_score != 0 else 0.001)
    
    plt.plot(x, y, 'k-', lw=2, label='Distribution of random scales')
    plt.axvline(phi_te, color='blue', linestyle='--', linewidth=2, 
                label=f'Phi-based scales: {phi_te:.6f}')
    plt.axvline(non_phi_te, color='gray', linestyle='-', linewidth=1.5,
                label=f'Mean random: {non_phi_te:.6f}')
    
    plt.xlabel('Transfer Entropy')
    plt.ylabel('Probability Density')
    plt.title(f'Distribution of Transfer Entropy (Z-score: {z_score:.2f})')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('transfer_entropy_results.png')
    print("Visualization saved to 'transfer_entropy_results.png'")
    
    # Save results to file
    with open('transfer_entropy_results.txt', 'w') as f:
        f.write("=== TRANSFER ENTROPY ANALYSIS RESULTS ===\n\n")
        
        f.write("1. Phi-based scales transfer entropy: {:.6f}\n".format(phi_te))
        f.write("2. Random scales transfer entropy: {:.6f}\n".format(non_phi_te))
        f.write("3. Transfer entropy ratio: {:.2f}x\n".format(te_ratio))
        f.write("4. Statistical significance:\n")
        f.write("   Z-score: {:.4f}\n".format(z_score))
        f.write("   P-value: {:.4f}\n\n".format(p_value))
        f.write("5. Phi optimality: {:.4f} ({})\n\n".format(phi_optimality, interpretation))
        
        f.write("Summary:\n")
        if p_value < 0.05 and phi_optimality > 0:
            f.write("The analysis shows significant information transfer across scales following golden ratio patterns in the CMB data.\n")
            if te_ratio > 1:
                f.write("The phi-based scales show higher transfer entropy than random scales, suggesting that the golden ratio pattern facilitates information transfer across scales.\n")
        else:
            f.write("The analysis does not show significant information transfer across scales following golden ratio patterns in the CMB data.\n")
        
        if phi_optimality > 0:
            f.write(f"Phi optimality: {phi_optimality:.4f} ({interpretation})\n")
    
    print("Results saved to 'transfer_entropy_results.txt'")

if __name__ == "__main__":
    main()
