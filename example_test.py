#!/usr/bin/env python3
"""
Example script to demonstrate how to run a specific test
from the CosmicConsciousnessAnalyzer class.
"""

from cosmic_consciousness_analyzer import CosmicConsciousnessAnalyzer

# Initialize the analyzer
analyzer = CosmicConsciousnessAnalyzer(data_dir='./data')

# Run the golden ratio significance test
print("Running Golden Ratio Significance Test...")
gr_results = analyzer.test_gr_significance()

# Print the results
print(f"\nGolden Ratio Test Results:")
print(f"GR Power: {gr_results[0]:.3f} μK²")
print(f"Mean Random Power: {gr_results[1]:.3f} μK²")
print(f"Excess Factor: {gr_results[4]:.2f}x")
print(f"Z-score: {gr_results[2]:.2f}σ (p = {gr_results[3]:.8f})")

# Run the coherence test
print("\nRunning Coherence Test...")
coherence_results = analyzer.test_coherence()

# Print the results
print(f"\nCoherence Test Results:")
print(f"Actual variance: {coherence_results[0]:.4f}")
print(f"Mean shuffled variance: {coherence_results[1]:.4f}")
print(f"Variance ratio: {coherence_results[4]:.4f}")
print(f"Z-score: {coherence_results[2]:.2f}σ (p = {coherence_results[3]:.8f})")

print("\nNote: For a full analysis, run: python run_cosmic_analysis.py")
