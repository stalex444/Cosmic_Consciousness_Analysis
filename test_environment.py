#!/usr/bin/env python3
"""
Test script to verify the environment is set up correctly.
"""

import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy
import astropy
import requests
import tqdm
import os

def test_imports():
    """Test that all required packages are properly installed."""
    packages = {
        "numpy": np,
        "matplotlib": matplotlib,
        "scipy": scipy,
        "astropy": astropy,
        "requests": requests,
        "tqdm": tqdm
    }
    
    # Try to import pywavelets
    try:
        import pywt
        packages["pywavelets"] = pywt
    except ImportError:
        print("PyWavelets not installed. Some functionality will be limited.")
    
    print("Python version:", sys.version)
    print("\nPackage versions:")
    for name, module in packages.items():
        print(f"{name}: {module.__version__}")
    
    print("\nAll required packages are installed!")
    
def test_golden_ratio_calculation():
    """Test basic golden ratio calculations."""
    phi = (1 + np.sqrt(5)) / 2
    print(f"\nGolden Ratio (φ): {phi}")
    print(f"Inverse Golden Ratio (1/φ): {1/phi}")
    
    # Test a simple golden ratio pattern
    print("\nTesting golden ratio pattern generation:")
    t = np.linspace(0, 10, 1000)
    f1 = 0.5
    f2 = f1 * phi
    f3 = f2 * phi
    
    signal = np.sin(2 * np.pi * f1 * t) + 0.5 * np.sin(2 * np.pi * f2 * t) + 0.25 * np.sin(2 * np.pi * f3 * t)
    
    # Plot a small segment of the signal
    plt.figure(figsize=(10, 4))
    plt.plot(t[:200], signal[:200])
    plt.title("Golden Ratio Frequency Pattern")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.grid(True)
    
    # Save the plot
    os.makedirs("planck_data", exist_ok=True)
    plt.savefig("planck_data/test_golden_ratio_pattern.png")
    print("Test plot saved to planck_data/test_golden_ratio_pattern.png")
    
    return True
    
def main():
    """Main function to run all tests."""
    print("Testing environment setup...\n")
    
    test_imports()
    test_golden_ratio_calculation()
    
    print("\nEnvironment test completed successfully!")

if __name__ == "__main__":
    main()
