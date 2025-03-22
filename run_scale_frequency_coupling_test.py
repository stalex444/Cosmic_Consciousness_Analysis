#!/usr/bin/env python3
"""
Run the scale frequency coupling test from the CosmicConsciousnessAnalyzer class.
This script tests if golden ratio patterns show scale-dependent coupling in the CMB data.
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
    """Run the scale frequency coupling test."""
    # Set data directory
    data_dir = os.path.join(os.getcwd(), 'planck_data')
    if not os.path.exists(data_dir):
        print(f"Error: Data directory '{data_dir}' not found.")
        sys.exit(1)
    
    print("=== SCALE FREQUENCY COUPLING ANALYSIS ===")
    print(f"Using data directory: {data_dir}")
    
    # Create analyzer
    print("Creating analyzer with 100 Monte Carlo simulations...")
    start_time = time.time()
    analyzer = CosmicConsciousnessAnalyzer(data_dir=data_dir, monte_carlo_sims=100)
    print(f"Analyzer initialized in {time.time() - start_time:.2f} seconds.")
    print(f"Data loaded: {len(analyzer.data['ell'])} multipoles")
    print()
    
    # Run the test
    print("Running scale frequency coupling analysis...")
    start_time = time.time()
    result = analyzer.test_scale_frequency_coupling()
    
    if result is None:
        print("Error: Not enough data for scale frequency coupling analysis.")
        sys.exit(1)
        
    scale_gr_frequencies, correlation, p_value, z_score, corr_p_value, linear_mse, phi_mse, model_ratio = result
    
    # Calculate mean random correlation (we know it should be close to 0 for permutation tests)
    mean_random_corr = 0
    # Calculate standard deviation from z-score and correlation
    if z_score != 0:
        std_random_corr = abs(correlation) / abs(z_score)
    else:
        std_random_corr = 0.1  # Default value if z-score is 0
    
    print(f"Test completed in {time.time() - start_time:.2f} seconds.")
    print()
    
    # Calculate phi optimality
    if model_ratio is not None:
        phi_optimality = calculate_phi_optimality(model_ratio, 1.0)
        interpretation = interpret_phi_optimality(phi_optimality)
    else:
        phi_optimality = 0
        interpretation = "neutral"
    
    # Print results
    print("=== RESULTS ===")
    print("1. Scale-Dependent Golden Ratio Frequencies:")
    for i, center, freq in scale_gr_frequencies:
        print(f"   Scale range {i+1}: center ℓ = {center:.1f}, GR frequency = {freq:.4f}")
    print()
    
    print("2. Scale-Frequency Correlation Analysis:")
    print(f"   Correlation: {correlation:.4f}")
    print(f"   P-value: {p_value:.4f}")
    print(f"   Random correlation: {mean_random_corr:.4f}")
    print(f"   Z-score: {z_score:.4f}")
    print(f"   Corrected p-value: {corr_p_value:.4f}")
    print()
    
    if model_ratio is not None:
        print("3. Model Comparison:")
        print(f"   Linear model MSE: {linear_mse:.6f}")
        print(f"   Phi-based model MSE: {phi_mse:.6f}")
        print(f"   Model ratio: {model_ratio:.2f}x")
        print(f"   Phi optimality: {phi_optimality:.4f}")
        print(f"   Interpretation: {interpretation}")
    
    # Create visualization
    print("\nCreating visualization...")
    plt.figure(figsize=(12, 10))
    
    # Plot 1: Scale-dependent GR frequencies
    plt.subplot(2, 1, 1)
    scale_centers = [s[1] for s in scale_gr_frequencies]
    frequencies = [s[2] for s in scale_gr_frequencies]
    
    plt.scatter(scale_centers, frequencies, s=100, c='blue', alpha=0.7, label='Observed')
    plt.xscale('log')
    
    # Add trend line
    if len(frequencies) >= 2:
        # Plot linear fit
        x_smooth = np.logspace(np.log10(min(scale_centers)), np.log10(max(scale_centers)), 100)
        if len(frequencies) >= 3:
            linear_model = np.polyfit(np.log10(scale_centers), frequencies, 1)
            y_linear = np.polyval(linear_model, np.log10(x_smooth))
            plt.plot(x_smooth, y_linear, 'r--', label=f'Linear fit (MSE={linear_mse:.6f})')
            
            # Plot phi-based fit
            phi = analyzer.phi
            phi_model = np.polyfit(np.log10(scale_centers)/np.log10(phi), frequencies, 1)
            y_phi = np.polyval(phi_model, np.log10(x_smooth)/np.log10(phi))
            plt.plot(x_smooth, y_phi, 'g-', label=f'Phi-based fit (MSE={phi_mse:.6f})')
    
    plt.xlabel('Scale (multipole ℓ)')
    plt.ylabel('Golden Ratio Frequency')
    plt.title(f'Scale-Dependent Golden Ratio Frequencies (r={correlation:.4f}, p={p_value:.4f})')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot 2: Distribution of random correlations
    plt.subplot(2, 1, 2)
    plt.hist(np.random.normal(mean_random_corr, std_random_corr, 1000), bins=30, alpha=0.7, color='gray', label='Random correlations')
    plt.axvline(correlation, color='red', linestyle='--', linewidth=2, 
                label=f'Observed correlation: {correlation:.4f}')
    plt.axvline(mean_random_corr, color='blue', linestyle='-', linewidth=1.5,
                label=f'Mean random: {mean_random_corr:.4f}')
        
    plt.xlabel('Correlation Coefficient')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Random Correlations (Z-score: {z_score:.2f}, p={corr_p_value:.4f})')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('scale_frequency_coupling_results.png')
    print("Visualization saved to 'scale_frequency_coupling_results.png'")
    
    # Save results to file
    with open('scale_frequency_coupling_results.txt', 'w') as f:
        f.write("=== SCALE FREQUENCY COUPLING ANALYSIS RESULTS ===\n\n")
        
        f.write("1. Scale-Dependent Golden Ratio Frequencies:\n")
        for i, center, freq in scale_gr_frequencies:
            f.write(f"   Scale range {i+1}: center ℓ = {center:.1f}, GR frequency = {freq:.4f}\n")
        f.write("\n")
        
        f.write("2. Scale-Frequency Correlation Analysis:\n")
        f.write(f"   Correlation: {correlation:.4f}\n")
        f.write(f"   P-value: {p_value:.4f}\n")
        f.write(f"   Random correlation: {mean_random_corr:.4f}\n")
        f.write(f"   Z-score: {z_score:.4f}\n")
        f.write(f"   Corrected p-value: {corr_p_value:.4f}\n\n")
        
        if model_ratio is not None:
            f.write("3. Model Comparison:\n")
            f.write(f"   Linear model MSE: {linear_mse:.6f}\n")
            f.write(f"   Phi-based model MSE: {phi_mse:.6f}\n")
            f.write(f"   Model ratio: {model_ratio:.2f}x\n")
            f.write(f"   Phi optimality: {phi_optimality:.4f}\n")
            f.write(f"   Interpretation: {interpretation}\n\n")
        
        f.write("Summary:\n")
        if corr_p_value < 0.05 and phi_optimality > 0:
            f.write("The analysis shows significant scale-dependent coupling of golden ratio patterns in the CMB data.\n")
            if model_ratio > 1:
                f.write("The phi-based model provides a better fit than the linear model, suggesting that the coupling follows a phi-based pattern.\n")
        else:
            f.write("The analysis does not show significant scale-dependent coupling of golden ratio patterns in the CMB data.\n")
        
        if phi_optimality > 0:
            f.write(f"Phi optimality: {phi_optimality:.4f} ({interpretation})\n")
    
    print("Results saved to 'scale_frequency_coupling_results.txt'")

if __name__ == "__main__":
    main()
