#!/usr/bin/env python3
"""
Main module for Cosmic Consciousness Analysis.
This module provides functions for analyzing cosmic data patterns.
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.cosmology import Planck15
import requests
import tqdm
import os
import json
from scipy import signal

# Constants
GOLDEN_RATIO = (1 + np.sqrt(5)) / 2  # φ ≈ 1.618033988749895
PLANCK_CONSTANT = 6.62607015e-34  # Planck constant in J⋅s
SPEED_OF_LIGHT = 299792458  # Speed of light in m/s
GRAVITATIONAL_CONSTANT = 6.67430e-11  # Gravitational constant in m³/(kg⋅s²)

class CosmicAnalyzer:
    """Class for analyzing cosmic data for consciousness patterns."""
    
    def __init__(self, data_dir="planck_data"):
        """Initialize the analyzer with a data directory."""
        self.data_dir = data_dir
        self.ensure_data_directory()
        
    def ensure_data_directory(self):
        """Ensure the data directory exists."""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            
    def download_planck_data(self, url, filename):
        """Download Planck mission data from a URL."""
        filepath = os.path.join(self.data_dir, filename)
        
        if os.path.exists(filepath):
            print(f"File {filename} already exists.")
            return filepath
            
        print(f"Downloading {filename} from {url}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024
        
        with open(filepath, 'wb') as file, tqdm.tqdm(
                desc=filename,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
            for data in response.iter_content(block_size):
                file.write(data)
                bar.update(len(data))
                
        print(f"Downloaded {filename} successfully.")
        return filepath
        
    def analyze_golden_ratio_patterns(self, data):
        """
        Analyze data for patterns related to the golden ratio.
        
        Parameters:
        -----------
        data : numpy.ndarray
            The data to analyze
            
        Returns:
        --------
        dict
            Results of the analysis
        """
        results = {}
        
        # Calculate power spectrum
        freqs, psd = signal.welch(data, fs=1.0, nperseg=1024)
        
        # Find peaks in the power spectrum
        peaks, _ = signal.find_peaks(psd)
        peak_freqs = freqs[peaks]
        
        # Calculate ratios between adjacent peak frequencies
        if len(peak_freqs) > 1:
            ratios = peak_freqs[1:] / peak_freqs[:-1]
            # Ensure ratios are less than 1 (invert if needed)
            ratios = np.array([1/r if r > 1 else r for r in ratios])
            
            # Calculate phi-optimality (how close ratios are to 1/φ)
            inverse_phi = 1 / GOLDEN_RATIO
            phi_optimality = 1 - np.abs(ratios - inverse_phi) / inverse_phi
            
            results['peak_frequencies'] = peak_freqs.tolist()
            results['frequency_ratios'] = ratios.tolist()
            results['phi_optimality'] = phi_optimality.tolist()
            results['average_phi_optimality'] = float(np.mean(phi_optimality))
            
        return results
        
    def visualize_results(self, results, title="Golden Ratio Analysis"):
        """
        Visualize the results of the analysis.
        
        Parameters:
        -----------
        results : dict
            Results from analyze_golden_ratio_patterns
        title : str
            Title for the plot
        """
        if 'phi_optimality' not in results:
            print("No results to visualize.")
            return
            
        plt.figure(figsize=(12, 8))
        
        # Plot phi-optimality for each ratio
        plt.subplot(2, 1, 1)
        plt.bar(range(len(results['phi_optimality'])), results['phi_optimality'])
        plt.axhline(y=results['average_phi_optimality'], color='r', linestyle='-', 
                   label=f"Avg: {results['average_phi_optimality']:.3f}")
        plt.xlabel('Ratio Index')
        plt.ylabel('Phi-Optimality')
        plt.title(f'Phi-Optimality of Frequency Ratios (1 = perfect match with 1/φ)')
        plt.legend()
        
        # Plot the ratios compared to 1/φ
        plt.subplot(2, 1, 2)
        ratios = results['frequency_ratios']
        plt.scatter(range(len(ratios)), ratios, label='Observed Ratios')
        plt.axhline(y=1/GOLDEN_RATIO, color='g', linestyle='--', 
                   label=f'1/φ = {1/GOLDEN_RATIO:.6f}')
        plt.xlabel('Ratio Index')
        plt.ylabel('Ratio Value')
        plt.title('Frequency Ratios Compared to 1/φ')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.data_dir, 'golden_ratio_analysis.png'))
        plt.show()
        
def main():
    """Main function to run the cosmic analysis."""
    analyzer = CosmicAnalyzer()
    print("Cosmic Consciousness Analysis initialized.")
    print(f"Golden Ratio (φ): {GOLDEN_RATIO}")
    print(f"Inverse Golden Ratio (1/φ): {1/GOLDEN_RATIO}")
    
    # Example: Generate synthetic data with golden ratio embedded patterns
    print("Generating synthetic data with embedded golden ratio patterns...")
    np.random.seed(42)
    t = np.linspace(0, 100, 10000)
    
    # Create a signal with frequencies in golden ratio relationships
    f1 = 0.1  # Base frequency
    f2 = f1 * GOLDEN_RATIO
    f3 = f2 * GOLDEN_RATIO
    
    signal = (np.sin(2 * np.pi * f1 * t) + 
              0.5 * np.sin(2 * np.pi * f2 * t) + 
              0.25 * np.sin(2 * np.pi * f3 * t))
    
    # Add some noise
    noisy_signal = signal + 0.1 * np.random.randn(len(t))
    
    # Analyze the signal
    results = analyzer.analyze_golden_ratio_patterns(noisy_signal)
    
    # Print results
    print("\nAnalysis Results:")
    print(f"Peak Frequencies: {results['peak_frequencies']}")
    print(f"Frequency Ratios: {results['frequency_ratios']}")
    print(f"Phi-Optimality: {results['phi_optimality']}")
    print(f"Average Phi-Optimality: {results['average_phi_optimality']:.4f}")
    
    # Visualize results
    analyzer.visualize_results(results)
    
    # Save results
    with open(os.path.join(analyzer.data_dir, 'synthetic_analysis_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    print("\nAnalysis complete. Results saved to planck_data/synthetic_analysis_results.json")

if __name__ == "__main__":
    main()
