#!/usr/bin/env python3
"""
Simple script to test the golden symmetries test.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from cosmic_consciousness_analyzer import CosmicConsciousnessAnalyzer

def main():
    """Run a simple test of the golden symmetries function."""
    try:
        # Set data directory
        data_dir = os.path.join(os.getcwd(), 'planck_data')
        
        print("=== SIMPLE GOLDEN SYMMETRIES TEST ===")
        print(f"Using data directory: {data_dir}")
        
        # Initialize analyzer
        print("Creating analyzer...")
        analyzer = CosmicConsciousnessAnalyzer(data_dir=data_dir, monte_carlo_sims=100)
        print(f"Data loaded: {len(analyzer.data['ell'])} multipoles")
        
        # Run the golden symmetries test
        print("\nRunning Golden Symmetries Test...")
        results = analyzer.test_golden_symmetries()
        
        print("\nTest Results:")
        print(f"Results: {results}")
        
        # Create a simple plot
        plt.figure(figsize=(10, 6))
        plt.plot(analyzer.data['ell'], analyzer.data['ee_power'], 'b-')
        plt.xlabel('Multipole ℓ')
        plt.ylabel('Power (μK²)')
        plt.title('CMB Power Spectrum')
        plt.grid(True, alpha=0.3)
        plt.savefig('simple_plot.png')
        print("Simple plot saved to 'simple_plot.png'")
        
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
