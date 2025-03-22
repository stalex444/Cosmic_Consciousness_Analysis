#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from cosmic_consciousness_analyzer import CosmicConsciousnessAnalyzer

def main():
    print("=== EXTENDED META-COHERENCE ANALYSIS ===")
    
    # Set up the data directory
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "planck_data")
    print(f"Using data directory: {data_dir}")
    
    # Create analyzer
    print("Creating analyzer with 1000 Monte Carlo simulations...")
    analyzer = CosmicConsciousnessAnalyzer(data_dir=data_dir, monte_carlo_sims=1000)
    print(f"Data loaded: {len(analyzer.data['ee_power'])} multipoles")
    
    # Run the test
    print("\nRunning extended meta-coherence analysis...")
    start_time = time.time()
    
    # Print available keys in analyzer.data
    print("Available keys in analyzer.data:", list(analyzer.data.keys()))
    
    results = analyzer.test_extended_meta_coherence()
    print(f"Test completed in {time.time() - start_time:.2f} seconds.")
    
    # Extract results
    meta_coherence, mean_shuffled_meta, meta_z, meta_p, meta_ratio = results['meta_coherence']
    skewness, mean_shuffled_skew, skew_z, skew_p, skew_ratio = results['skewness']
    kurtosis, mean_shuffled_kurt, kurt_z, kurt_p, kurt_ratio = results['kurtosis']
    entropy, mean_shuffled_entropy, entropy_z, entropy_p, entropy_ratio = results['entropy']
    power_law_exponent, mean_shuffled_exponent, exponent_z, exponent_p, exponent_ratio = results['power_law']
    
    # Display results
    print("\n=== RESULTS ===")
    print("1. Meta-Coherence:")
    print(f"   Actual: {meta_coherence:.6f}")
    print(f"   Random: {mean_shuffled_meta:.6f}")
    print(f"   Ratio: {meta_ratio:.2f}x")
    print(f"   Z-score: {meta_z:.4f}")
    print(f"   P-value: {meta_p:.4f}")
    
    print("\n2. Skewness:")
    print(f"   Actual: {skewness:.6f}")
    print(f"   Random: {mean_shuffled_skew:.6f}")
    print(f"   Ratio: {skew_ratio:.2f}x")
    print(f"   Z-score: {skew_z:.4f}")
    print(f"   P-value: {skew_p:.4f}")
    
    print("\n3. Kurtosis:")
    print(f"   Actual: {kurtosis:.6f}")
    print(f"   Random: {mean_shuffled_kurt:.6f}")
    print(f"   Ratio: {kurt_ratio:.2f}x")
    print(f"   Z-score: {kurt_z:.4f}")
    print(f"   P-value: {kurt_p:.4f}")
    
    print("\n4. Entropy:")
    print(f"   Actual: {entropy:.6f}")
    print(f"   Random: {mean_shuffled_entropy:.6f}")
    print(f"   Ratio: {entropy_ratio:.2f}x")
    print(f"   Z-score: {entropy_z:.4f}")
    print(f"   P-value: {entropy_p:.4f}")
    
    if power_law_exponent is not None:
        print("\n5. Power Law Exponent:")
        print(f"   Actual: {power_law_exponent:.6f}")
        print(f"   Random: {mean_shuffled_exponent:.6f}")
        print(f"   Ratio: {exponent_ratio:.2f}x")
        print(f"   Z-score: {exponent_z:.4f}")
        print(f"   P-value: {exponent_p:.4f}")
    else:
        print("\n5. Power Law Exponent: Insufficient data for analysis")
    
    # Calculate overall significance
    valid_p_values = [p for p in [meta_p, skew_p, kurt_p, entropy_p, exponent_p] if p is not None]
    combined_significance = np.mean(valid_p_values)
    print(f"\nOverall significance (mean p-value): {combined_significance:.4f}")
    
    # Create visualization
    print("\nCreating visualization...")
    create_visualization(analyzer, results)
    
    # Save results to file
    save_results_to_file(results, combined_significance)
    
    print("Results saved to 'extended_meta_coherence_results.txt'")
    print("Visualization saved to 'extended_meta_coherence_results.png'")

def create_visualization(analyzer, results):
    meta_coherence, mean_shuffled_meta, meta_z, meta_p, meta_ratio = results['meta_coherence']
    skewness, mean_shuffled_skew, skew_z, skew_p, skew_ratio = results['skewness']
    kurtosis, mean_shuffled_kurt, kurt_z, kurt_p, kurt_ratio = results['kurtosis']
    entropy, mean_shuffled_entropy, entropy_z, entropy_p, entropy_ratio = results['entropy']
    power_law_exponent, mean_shuffled_exponent, exponent_z, exponent_p, exponent_ratio = results['power_law']
    
    # Calculate local coherence for visualization
    window_size = 5
    step_size = 2
    local_coherence = []
    multipoles = []
    
    for i in range(0, len(analyzer.data['ee_power']) - window_size, step_size):
        window = analyzer.data['ee_power'][i:i+window_size]
        normalized = window / np.mean(window) if np.mean(window) != 0 else window
        local_coherence.append(np.var(normalized))
        multipoles.append(analyzer.data['ell'][i + window_size//2])
    
    # Create figure with subplots
    fig = plt.figure(figsize=(14, 12))
    gs = GridSpec(3, 2, figure=fig)
    
    # Plot 1: Local coherence across multipoles
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(multipoles, local_coherence, 'b-', label='Local Coherence')
    ax1.set_title('Local Coherence Across Multipoles')
    ax1.set_xlabel('Multipole (ℓ)')
    ax1.set_ylabel('Local Coherence')
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Histogram of local coherence
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.hist(local_coherence, bins=10, alpha=0.7, color='blue')
    ax2.axvline(np.mean(local_coherence), color='r', linestyle='--', label='Mean')
    ax2.set_title(f'Distribution of Local Coherence\nSkewness: {skewness:.3f}, Kurtosis: {kurtosis:.3f}')
    ax2.set_xlabel('Local Coherence')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Power spectrum of local coherence
    ax3 = fig.add_subplot(gs[1, 1])
    coherence_fft = np.abs(np.fft.fft(local_coherence))**2
    frequencies = np.fft.fftfreq(len(local_coherence))
    positive_freq_idx = frequencies > 0
    ax3.loglog(frequencies[positive_freq_idx], coherence_fft[positive_freq_idx], 'o-', markersize=3)
    
    if power_law_exponent is not None:
        # Add power law fit line
        x_range = np.logspace(np.log10(min(frequencies[positive_freq_idx])), 
                             np.log10(max(frequencies[positive_freq_idx])), 100)
        y_fit = 10**(power_law_exponent * np.log10(x_range) + np.log10(coherence_fft[positive_freq_idx][0]))
        ax3.loglog(x_range, y_fit, 'r--', 
                  label=f'Power Law Fit: α = {power_law_exponent:.2f}')
        ax3.legend()
    
    ax3.set_title('Power Spectrum of Local Coherence')
    ax3.set_xlabel('Frequency')
    ax3.set_ylabel('Power')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Comparison with random expectation
    ax4 = fig.add_subplot(gs[2, :])
    metrics = ['Meta-Coherence', 'Skewness', 'Kurtosis', 'Entropy']
    actual_values = [meta_ratio, skew_ratio, kurt_ratio, entropy_ratio]
    
    if power_law_exponent is not None and exponent_ratio is not None:
        metrics.append('Power Law')
        actual_values.append(exponent_ratio)
    
    colors = ['blue' if v > 1 else 'red' for v in actual_values]
    ax4.bar(metrics, actual_values, color=colors)
    ax4.axhline(y=1.0, color='k', linestyle='--', alpha=0.7, label='Random Expectation')
    
    # Add p-values as text
    for i, metric in enumerate(metrics):
        if metric == 'Meta-Coherence':
            p_val = meta_p
        elif metric == 'Skewness':
            p_val = skew_p
        elif metric == 'Kurtosis':
            p_val = kurt_p
        elif metric == 'Entropy':
            p_val = entropy_p
        else:  # Power Law
            p_val = exponent_p
            
        if p_val is not None:
            ax4.text(i, actual_values[i] + 0.1, f'p = {p_val:.4f}', 
                    ha='center', va='bottom', fontsize=9)
    
    ax4.set_title('Ratio of Actual vs. Random Expectation')
    ax4.set_ylabel('Ratio')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # Add overall title
    plt.suptitle('Extended Meta-Coherence Analysis of CMB Data', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save figure
    plt.savefig('extended_meta_coherence_results.png', dpi=300, bbox_inches='tight')

def save_results_to_file(results, combined_significance):
    meta_coherence, mean_shuffled_meta, meta_z, meta_p, meta_ratio = results['meta_coherence']
    skewness, mean_shuffled_skew, skew_z, skew_p, skew_ratio = results['skewness']
    kurtosis, mean_shuffled_kurt, kurt_z, kurt_p, kurt_ratio = results['kurtosis']
    entropy, mean_shuffled_entropy, entropy_z, entropy_p, entropy_ratio = results['entropy']
    power_law_exponent, mean_shuffled_exponent, exponent_z, exponent_p, exponent_ratio = results['power_law']
    
    with open('extended_meta_coherence_results.txt', 'w') as f:
        f.write("=== EXTENDED META-COHERENCE ANALYSIS RESULTS ===\n\n")
        
        f.write("1. Meta-Coherence:\n")
        f.write(f"   Actual: {meta_coherence:.6f}\n")
        f.write(f"   Random: {mean_shuffled_meta:.6f}\n")
        f.write(f"   Ratio: {meta_ratio:.2f}x\n")
        f.write(f"   Z-score: {meta_z:.4f}\n")
        f.write(f"   P-value: {meta_p:.4f}\n\n")
        
        f.write("2. Skewness:\n")
        f.write(f"   Actual: {skewness:.6f}\n")
        f.write(f"   Random: {mean_shuffled_skew:.6f}\n")
        f.write(f"   Ratio: {skew_ratio:.2f}x\n")
        f.write(f"   Z-score: {skew_z:.4f}\n")
        f.write(f"   P-value: {skew_p:.4f}\n\n")
        
        f.write("3. Kurtosis:\n")
        f.write(f"   Actual: {kurtosis:.6f}\n")
        f.write(f"   Random: {mean_shuffled_kurt:.6f}\n")
        f.write(f"   Ratio: {kurt_ratio:.2f}x\n")
        f.write(f"   Z-score: {kurt_z:.4f}\n")
        f.write(f"   P-value: {kurt_p:.4f}\n\n")
        
        f.write("4. Entropy:\n")
        f.write(f"   Actual: {entropy:.6f}\n")
        f.write(f"   Random: {mean_shuffled_entropy:.6f}\n")
        f.write(f"   Ratio: {entropy_ratio:.2f}x\n")
        f.write(f"   Z-score: {entropy_z:.4f}\n")
        f.write(f"   P-value: {entropy_p:.4f}\n\n")
        
        if power_law_exponent is not None:
            f.write("5. Power Law Exponent:\n")
            f.write(f"   Actual: {power_law_exponent:.6f}\n")
            f.write(f"   Random: {mean_shuffled_exponent:.6f}\n")
            f.write(f"   Ratio: {exponent_ratio:.2f}x\n")
            f.write(f"   Z-score: {exponent_z:.4f}\n")
            f.write(f"   P-value: {exponent_p:.4f}\n\n")
        else:
            f.write("5. Power Law Exponent: Insufficient data for analysis\n\n")
        
        f.write(f"Overall significance (mean p-value): {combined_significance:.4f}\n\n")
        
        # Add interpretation
        f.write("Summary:\n")
        if combined_significance < 0.05:
            f.write("The analysis shows significant evidence for non-random meta-coherence properties in the CMB data.\n")
            f.write("This suggests the presence of higher-order organizational principles that may be consistent with\n")
            f.write("patterns found in complex conscious systems.\n")
        elif combined_significance < 0.1:
            f.write("The analysis shows marginally significant evidence for non-random meta-coherence properties\n")
            f.write("in the CMB data. While not conclusive, these results suggest the possibility of higher-order\n")
            f.write("organizational principles that may warrant further investigation.\n")
        else:
            f.write("The analysis does not provide strong evidence for non-random meta-coherence properties\n")
            f.write("in the CMB data. The observed patterns are largely consistent with what would be expected\n")
            f.write("from random processes.\n")

if __name__ == "__main__":
    main()
