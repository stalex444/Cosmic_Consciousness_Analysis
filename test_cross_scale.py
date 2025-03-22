#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple script to test the cross-scale correlations function.
"""

from cosmic_consciousness_analyzer import CosmicConsciousnessAnalyzer

# Initialize the analyzer with minimal Monte Carlo simulations
analyzer = CosmicConsciousnessAnalyzer(data_dir='data', monte_carlo_sims=100)

# Run the cross-scale correlations test
print("Running cross-scale correlations test...")
mean_phi_corr, mean_random_corr, z_score, p_value = analyzer.test_cross_scale_correlations()

# Print results
print("\nResults:")
print("Mean phi-related correlation: {:.4f}".format(mean_phi_corr))
print("Mean random correlation: {:.4f}".format(mean_random_corr))
ratio = mean_phi_corr / mean_random_corr if mean_random_corr > 0 else 1.0
print("Ratio: {:.2f}x".format(ratio))
print("Z-score: {:.2f} sigma".format(z_score))
print("P-value: {:.8f}".format(p_value))
