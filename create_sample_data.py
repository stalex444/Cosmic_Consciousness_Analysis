#!/usr/bin/env python3
"""
Create sample CMB data for testing the Cosmic Consciousness Analyzer.
This script generates synthetic EE power spectrum data that mimics the
Planck CMB data, with some golden ratio patterns embedded.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def create_sample_data():
    """Create sample CMB data with golden ratio patterns."""
    
    # Create directories if they don't exist
    os.makedirs("data/power_spectra", exist_ok=True)
    
    # Calculate the golden ratio
    phi = (1 + np.sqrt(5)) / 2
    
    # Generate multipoles from ell=2 to ell=2500
    ell = np.arange(2, 2501)
    n_ell = len(ell)
    
    # Create a baseline power spectrum that follows a power law with oscillations
    # D_ell = ell(ell+1)C_ell/(2π) ~ ell^(-0.6) with oscillations
    baseline = 1e-5 * (ell/80.0)**(-0.6) * (1 + 0.1*np.sin(np.log(ell/30.0)*5))
    
    # Add acoustic peaks (simplified)
    peaks = np.zeros_like(baseline)
    peak_positions = [220, 540, 810, 1150, 1450, 1800, 2100]
    peak_heights = [25, 15, 8, 5, 3, 2, 1.5]
    peak_widths = [50, 70, 90, 100, 120, 140, 160]
    
    for pos, height, width in zip(peak_positions, peak_heights, peak_widths):
        peaks += height * 1e-5 * np.exp(-0.5 * ((ell - pos) / width)**2)
    
    # Add golden ratio patterns
    gr_pattern = np.zeros_like(baseline)
    
    # Generate golden ratio multipoles
    gr_multipoles = []
    current = 2  # Start with ell=2
    for _ in range(20):
        gr_multipoles.append(int(current))
        current *= phi
    
    # Enhance power at golden ratio multipoles
    for gr_ell in gr_multipoles:
        if gr_ell < 2500:
            idx = np.argmin(np.abs(ell - gr_ell))
            width = max(5, int(gr_ell/20))  # Width scales with multipole
            enhancement = 0.2 * 1e-5 * np.exp(-0.5 * ((ell - gr_ell) / width)**2)
            gr_pattern += enhancement
    
    # Combine all components
    power = baseline + peaks + gr_pattern
    
    # Add noise
    noise = np.random.normal(0, 0.05 * power, n_ell)
    power += noise
    
    # Ensure all values are positive
    power = np.maximum(power, 1e-10)
    
    # Calculate error bars (simplified)
    error = 0.1 * power + 1e-7
    
    # Create the EE power spectrum file
    ee_data = np.column_stack((ell, power, error))
    
    # Save to files
    np.savetxt("data/power_spectra/COM_PowerSpect_CMB-EE-binned_R3.02.txt", ee_data, 
               header="ell D_ell[µK²] error", comments='#')
    
    # Create a full (unbinned) version with more points
    ell_full = np.arange(2, 2501, 1)
    n_ell_full = len(ell_full)
    
    # Interpolate to get full spectrum
    from scipy.interpolate import interp1d
    power_interp = interp1d(ell, power, kind='cubic', bounds_error=False, fill_value='extrapolate')
    error_interp = interp1d(ell, error, kind='cubic', bounds_error=False, fill_value='extrapolate')
    
    power_full = power_interp(ell_full)
    error_full = error_interp(ell_full)
    
    # Add more fine-grained noise
    noise_full = np.random.normal(0, 0.02 * power_full, n_ell_full)
    power_full += noise_full
    
    # Ensure all values are positive
    power_full = np.maximum(power_full, 1e-10)
    
    # Create the full EE power spectrum file
    ee_data_full = np.column_stack((ell_full, power_full, error_full))
    
    np.savetxt("data/power_spectra/COM_PowerSpect_CMB-EE-full_R3.01.txt", ee_data_full,
               header="ell D_ell[µK²] error", comments='#')
    
    # Create a simple diagonal covariance matrix
    cov_matrix = np.diag(error**2)
    
    # Save as a .npy file since we don't have FITS writing capability
    np.save("data/cov_matrix.npy", cov_matrix)
    
    # Create a visualization of the data
    plt.figure(figsize=(12, 8))
    
    # Plot the power spectrum
    plt.subplot(2, 1, 1)
    plt.errorbar(ell, power, yerr=error, fmt='o', markersize=2, alpha=0.5)
    plt.plot(ell, baseline, 'r-', label='Baseline', alpha=0.5)
    plt.plot(ell, baseline + peaks, 'g-', label='With Peaks', alpha=0.5)
    
    # Mark golden ratio multipoles
    for gr_ell in gr_multipoles:
        if gr_ell < 2500:
            plt.axvline(x=gr_ell, color='gold', linestyle='--', alpha=0.3)
    
    plt.xlabel('Multipole ℓ')
    plt.ylabel('D_ℓ [µK²]')
    plt.title('Synthetic EE Power Spectrum with Golden Ratio Patterns')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    
    # Plot the golden ratio pattern contribution
    plt.subplot(2, 1, 2)
    plt.plot(ell, gr_pattern, 'gold', label='Golden Ratio Pattern')
    
    # Mark golden ratio multipoles
    for gr_ell in gr_multipoles:
        if gr_ell < 2500:
            plt.axvline(x=gr_ell, color='gold', linestyle='--', alpha=0.3)
    
    plt.xlabel('Multipole ℓ')
    plt.ylabel('D_ℓ [µK²]')
    plt.title('Golden Ratio Pattern Contribution')
    plt.xscale('log')
    
    plt.tight_layout()
    plt.savefig('data/synthetic_spectrum.png')
    
    print("Sample data created successfully in the 'data' directory.")
    print("Files created:")
    print("  - data/power_spectra/COM_PowerSpect_CMB-EE-binned_R3.02.txt")
    print("  - data/power_spectra/COM_PowerSpect_CMB-EE-full_R3.01.txt")
    print("  - data/cov_matrix.npy")
    print("  - data/synthetic_spectrum.png")

if __name__ == "__main__":
    create_sample_data()
