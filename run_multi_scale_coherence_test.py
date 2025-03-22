#!/usr/bin/env python3
"""
Run the multi-scale coherence test to analyze how coherence varies across different scales
in the CMB data and if it follows golden ratio patterns.
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from cosmic_consciousness_analyzer import CosmicConsciousnessAnalyzer

def main():
    """Run the multi-scale coherence test and display results"""
    print("=== MULTI-SCALE COHERENCE ANALYSIS ===")
    
    # Set up data directory
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "planck_data")
    print(f"Using data directory: {data_dir}")
    
    # Create analyzer with Monte Carlo simulations
    print("Creating analyzer with 100 Monte Carlo simulations...")
    start_time = time.time()
    analyzer = CosmicConsciousnessAnalyzer(data_dir=data_dir, monte_carlo_sims=100)
    init_time = time.time() - start_time
    print(f"Analyzer initialized in {init_time:.2f} seconds.")
    print(f"Data loaded: {len(analyzer.data['ell'])} multipoles")
    
    # Run the test
    print("\nRunning multi-scale coherence analysis...")
    start_time = time.time()
    result = analyzer.test_multi_scale_coherence()
    elapsed = time.time() - start_time
    print(f"Test completed in {elapsed:.2f} seconds.")
    
    if result:
        scale_coherences, coherence_ratios, mean_phi_deviation, mean_random, z_score, p_value, optimization_ratio = result
        
        # Calculate phi optimality
        if z_score > 0:
            phi_optimality = min(1.0, max(0.0, 1 - (10**(-z_score))))
        else:
            phi_optimality = min(0.0, max(-1.0, -1 + 10**(z_score)))
            
        # Interpret phi optimality
        if phi_optimality >= 0.99:
            interpretation = "extremely high"
        elif phi_optimality >= 0.9:
            interpretation = "very high"
        elif phi_optimality >= 0.5:
            interpretation = "high"
        elif phi_optimality >= 0.1:
            interpretation = "moderate"
        elif phi_optimality > 0:
            interpretation = "slight"
        elif phi_optimality > -0.1:
            interpretation = "slightly negative"
        elif phi_optimality > -0.5:
            interpretation = "moderately negative"
        else:
            interpretation = "strongly negative"
        
        # Display results
        print("\n=== RESULTS ===")
        print(f"1. Scale coherence measurements: {len(scale_coherences)} scale bins")
        print(f"2. Coherence ratios: {len(coherence_ratios)} ratios")
        print(f"3. Mean deviation from phi: {mean_phi_deviation:.6f}")
        print(f"4. Mean random deviation: {mean_random:.6f}")
        print(f"5. Statistical significance:")
        print(f"   Z-score: {z_score:.4f}")
        print(f"   P-value: {p_value:.4f}")
        print(f"6. Optimization ratio: {optimization_ratio:.2f}x")
        print(f"7. Phi optimality: {phi_optimality:.4f} ({interpretation})")
        
        # Create visualization
        print("\nCreating visualization...")
        plt.figure(figsize=(12, 10))
        
        # Plot 1: Scale vs Coherence
        plt.subplot(2, 2, 1)
        scales = [s[0] for s in scale_coherences]
        coherences = [s[1] for s in scale_coherences]
        plt.scatter(scales, coherences, alpha=0.7, s=80)
        plt.plot(scales, coherences, 'b-', alpha=0.5)
        plt.xscale('log')
        plt.xlabel('Scale (multipole)')
        plt.ylabel('Coherence')
        plt.title('Coherence across Different Scales')
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Coherence Ratios
        plt.subplot(2, 2, 2)
        plt.hist(coherence_ratios, bins=10, alpha=0.7)
        plt.axvline(x=analyzer.phi, color='r', linestyle='--', 
                   label=f'Golden Ratio (φ = {analyzer.phi:.4f})')
        plt.xlabel('Coherence Ratio')
        plt.ylabel('Frequency')
        plt.title('Distribution of Coherence Ratios')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Phi Deviation Comparison
        plt.subplot(2, 2, 3)
        plt.hist(coherence_ratios, bins=10, alpha=0.7, 
                label=f'Actual (mean dev: {mean_phi_deviation:.4f})')
        
        # Generate some random ratios for comparison
        random_coherences = np.random.permutation(coherences)
        random_ratios = []
        for i in range(len(random_coherences)-1):
            if random_coherences[i+1] != 0:
                ratio = random_coherences[i] / random_coherences[i+1]
                random_ratios.append(ratio)
                
        plt.hist(random_ratios, bins=10, alpha=0.4, 
                label=f'Random (mean dev: {mean_random:.4f})')
        plt.axvline(x=analyzer.phi, color='r', linestyle='--', 
                   label=f'Golden Ratio (φ)')
        plt.xlabel('Ratio Value')
        plt.ylabel('Frequency')
        plt.title('Actual vs Random Coherence Ratios')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Summary
        plt.subplot(2, 2, 4)
        plt.axis('off')
        summary_text = (
            f"MULTI-SCALE COHERENCE TEST\n\n"
            f"Scale bins analyzed: {len(scale_coherences)}\n"
            f"Coherence ratios: {len(coherence_ratios)}\n\n"
            f"Mean deviation from phi: {mean_phi_deviation:.6f}\n"
            f"Mean random deviation: {mean_random:.6f}\n"
            f"Optimization ratio: {optimization_ratio:.2f}x\n\n"
            f"Statistical significance:\n"
            f"Z-score: {z_score:.4f}\n"
            f"P-value: {p_value:.4f}\n\n"
            f"Phi optimality: {phi_optimality:.4f}\n"
            f"Interpretation: {interpretation}\n\n"
            f"Analysis date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        plt.text(0.05, 0.95, summary_text, fontsize=10, 
                va='top', ha='left', transform=plt.gca().transAxes)
        
        plt.tight_layout()
        plt.savefig('multi_scale_coherence_results.png', dpi=300)
        print("Visualization saved to 'multi_scale_coherence_results.png'")
        
        # Save results to file
        with open('multi_scale_coherence_results.txt', 'w') as f:
            f.write("=== MULTI-SCALE COHERENCE ANALYSIS RESULTS ===\n\n")
            f.write(f"1. Scale coherence measurements: {len(scale_coherences)} scale bins\n")
            f.write(f"2. Coherence ratios: {len(coherence_ratios)} ratios\n")
            f.write(f"3. Mean deviation from phi: {mean_phi_deviation:.6f}\n")
            f.write(f"4. Mean random deviation: {mean_random:.6f}\n")
            f.write(f"5. Statistical significance:\n")
            f.write(f"   Z-score: {z_score:.4f}\n")
            f.write(f"   P-value: {p_value:.4f}\n\n")
            f.write(f"6. Optimization ratio: {optimization_ratio:.2f}x\n")
            f.write(f"7. Phi optimality: {phi_optimality:.4f} ({interpretation})\n\n")
            
            f.write("Summary:\n")
            if p_value < 0.05 and phi_optimality > 0:
                f.write("The analysis shows significant evidence that coherence varies across scales in a pattern related to the golden ratio.\n")
            elif phi_optimality > 0:
                f.write("The analysis shows some evidence that coherence varies across scales in a pattern related to the golden ratio, but it does not reach statistical significance.\n")
            else:
                f.write("The analysis does not show evidence that coherence varies across scales in a pattern related to the golden ratio.\n")
        
        print("Results saved to 'multi_scale_coherence_results.txt'")
    else:
        print("Test failed to produce results.")

if __name__ == "__main__":
    main()
