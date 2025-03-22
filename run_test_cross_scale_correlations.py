#!/usr/bin/env python3
"""
Script to run the Cross-Scale Correlations Test from the CosmicConsciousnessAnalyzer.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from cosmic_consciousness_analyzer import CosmicConsciousnessAnalyzer

def main():
    """Run the Cross-Scale Correlations Test and save results."""
    print("="*80)
    print("Running Cross-Scale Correlations Test")
    print("="*80)
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("results", f"cross_scale_correlations_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize analyzer with reduced Monte Carlo simulations for faster testing
    print("Initializing CosmicConsciousnessAnalyzer...")
    analyzer = CosmicConsciousnessAnalyzer(monte_carlo_sims=100)
    
    try:
        # Run the test
        print("Running test_cross_scale_correlations...")
        result = analyzer.test_cross_scale_correlations()
        
        # Unpack results
        if isinstance(result, tuple) and len(result) >= 4:
            mean_phi_corr, mean_random_corr, z_score, p_value = result[:4]
            
            # Save results to file
            with open(os.path.join(results_dir, "results.txt"), "w") as f:
                f.write("Cross-Scale Correlations Test Results\n")
                f.write("="*35 + "\n\n")
                f.write(f"Mean Phi Correlation: {mean_phi_corr:.6f}\n")
                f.write(f"Mean Random Correlation: {mean_random_corr:.6f}\n")
                f.write(f"Z-Score: {z_score:.6f}\n")
                f.write(f"P-Value: {p_value:.6f}\n")
                f.write(f"Significant: {p_value < 0.05}\n")
                
                # Calculate phi-optimality
                if p_value < 1e-10:
                    phi_optimality = 1.0
                elif p_value > 0.9:
                    phi_optimality = -1.0
                else:
                    phi_optimality = 1.0 - 2.0 * p_value
                f.write(f"Phi-Optimality: {phi_optimality:.6f}\n")
            
            # Create visualization
            plt.figure(figsize=(10, 6))
            plt.bar(['Phi-Related Scales', 'Random Scales'], 
                   [mean_phi_corr, mean_random_corr],
                   color=['gold', 'gray'])
            plt.ylabel('Mean Correlation')
            plt.title('Cross-Scale Correlations Test Results')
            plt.annotate(f'p-value: {p_value:.6f}', xy=(0.5, 0.9), 
                        xycoords='axes fraction', ha='center')
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, 'cross_scale_correlations.png'))
            
            print(f"Results saved to {results_dir}")
            print(f"Mean Phi Correlation: {mean_phi_corr:.6f}")
            print(f"Mean Random Correlation: {mean_random_corr:.6f}")
            print(f"Z-Score: {z_score:.6f}")
            print(f"P-Value: {p_value:.6f}")
            print(f"Significant: {p_value < 0.05}")
        else:
            print("Unexpected result format")
            
    except Exception as e:
        print(f"Error running test: {str(e)}")

if __name__ == "__main__":
    main()
