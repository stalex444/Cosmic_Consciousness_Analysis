#!/usr/bin/env python3
"""
Module for analyzing consciousness patterns in cosmic data.
This module implements advanced analysis techniques to detect patterns
that may relate to consciousness in cosmic background radiation.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, stats
import os
import json
try:
    import pywt  # PyWavelets for wavelet analysis
except ImportError:
    print("PyWavelets not installed. Wavelet analysis will not be available.")
    pywt = None

# Constants
GOLDEN_RATIO = (1 + np.sqrt(5)) / 2  # φ ≈ 1.618033988749895
PI = np.pi  # π ≈ 3.141592653589793
E = np.e  # e ≈ 2.718281828459045
SQRT2 = np.sqrt(2)  # √2 ≈ 1.4142135623730951

class ConsciousnessPatternAnalyzer:
    """Class for analyzing consciousness patterns in cosmic data."""
    
    def __init__(self, output_dir="planck_data"):
        """Initialize the analyzer with an output directory."""
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
    def calculate_phi_optimality(self, observed_ratio, target_ratio):
        """
        Calculate how close an observed ratio is to a target ratio.
        
        Parameters:
        -----------
        observed_ratio : float
            The observed ratio
        target_ratio : float
            The target ratio to compare against
            
        Returns:
        --------
        float
            Phi-optimality score between -1 and 1, where:
            1 = perfect match
            0 = 100% deviation
            -1 = maximum deviation
        """
        # Calculate relative deviation
        deviation = abs(observed_ratio - target_ratio) / target_ratio
        
        # Convert to phi-optimality score bounded between -1 and 1
        phi_optimality = 1 - min(deviation, 2)
        
        return phi_optimality
        
    def analyze_multiple_constants(self, data, window_size=1024, step_size=512):
        """
        Analyze data for patterns related to multiple mathematical constants.
        
        Parameters:
        -----------
        data : numpy.ndarray
            The data to analyze
        window_size : int
            Size of the sliding window for analysis
        step_size : int
            Step size for the sliding window
            
        Returns:
        --------
        dict
            Results of the analysis for each constant
        """
        constants = {
            "phi": 1/GOLDEN_RATIO,  # Inverse golden ratio
            "pi": 1/PI,             # Inverse pi
            "e": 1/E,               # Inverse e
            "sqrt2": 1/SQRT2        # Inverse square root of 2
        }
        
        results = {const: [] for const in constants}
        window_results = []
        
        # Analyze data in windows
        for i in range(0, len(data) - window_size, step_size):
            window = data[i:i+window_size]
            
            # Calculate power spectrum
            freqs, psd = signal.welch(window, fs=1.0, nperseg=min(256, len(window)))
            
            # Find peaks in the power spectrum
            peaks, _ = signal.find_peaks(psd)
            peak_freqs = freqs[peaks]
            
            # Calculate ratios between adjacent peak frequencies
            if len(peak_freqs) > 1:
                ratios = peak_freqs[1:] / peak_freqs[:-1]
                # Ensure ratios are less than 1 (invert if needed)
                ratios = np.array([1/r if r > 1 else r for r in ratios])
                
                # Calculate optimality for each constant
                window_result = {
                    "start_idx": i,
                    "end_idx": i + window_size,
                    "peak_freqs": peak_freqs.tolist(),
                    "ratios": ratios.tolist(),
                    "optimality": {}
                }
                
                for const_name, const_value in constants.items():
                    optimality = [self.calculate_phi_optimality(r, const_value) for r in ratios]
                    avg_optimality = np.mean(optimality) if optimality else 0
                    
                    results[const_name].append(avg_optimality)
                    window_result["optimality"][const_name] = {
                        "values": optimality,
                        "average": avg_optimality
                    }
                
                window_results.append(window_result)
        
        # Calculate overall statistics
        overall_results = {}
        for const_name in constants:
            const_results = np.array(results[const_name])
            overall_results[const_name] = {
                "mean": float(np.mean(const_results)),
                "median": float(np.median(const_results)),
                "std": float(np.std(const_results)),
                "max": float(np.max(const_results)),
                "min": float(np.min(const_results))
            }
            
        return {
            "constants": constants,
            "window_results": window_results,
            "overall_results": overall_results
        }
        
    def analyze_temporal_evolution(self, data, window_size=1024, step_size=128):
        """
        Analyze how phi-optimality evolves over time.
        
        Parameters:
        -----------
        data : numpy.ndarray
            The data to analyze
        window_size : int
            Size of the sliding window for analysis
        step_size : int
            Step size for the sliding window
            
        Returns:
        --------
        dict
            Results of the temporal evolution analysis
        """
        time_points = []
        phi_optimality = []
        
        # Analyze data in windows
        for i in range(0, len(data) - window_size, step_size):
            window = data[i:i+window_size]
            time_points.append(i + window_size // 2)  # Center of the window
            
            # Calculate power spectrum
            freqs, psd = signal.welch(window, fs=1.0, nperseg=min(256, len(window)))
            
            # Find peaks in the power spectrum
            peaks, _ = signal.find_peaks(psd)
            peak_freqs = freqs[peaks]
            
            # Calculate ratios and phi-optimality
            if len(peak_freqs) > 1:
                ratios = peak_freqs[1:] / peak_freqs[:-1]
                # Ensure ratios are less than 1 (invert if needed)
                ratios = np.array([1/r if r > 1 else r for r in ratios])
                
                # Calculate phi-optimality
                inverse_phi = 1 / GOLDEN_RATIO
                optimality = [self.calculate_phi_optimality(r, inverse_phi) for r in ratios]
                avg_optimality = np.mean(optimality) if optimality else 0
            else:
                avg_optimality = 0
                
            phi_optimality.append(avg_optimality)
            
        return {
            "time_points": time_points,
            "phi_optimality": phi_optimality
        }
        
    def wavelet_analysis(self, data, wavelet='db4', level=5):
        """
        Perform wavelet analysis on the data.
        
        Parameters:
        -----------
        data : numpy.ndarray
            The data to analyze
        wavelet : str
            Wavelet to use for the analysis
        level : int
            Decomposition level
            
        Returns:
        --------
        dict
            Results of the wavelet analysis
        """
        if pywt is None:
            return {"error": "PyWavelets not installed"}
            
        # Perform wavelet decomposition
        coeffs = pywt.wavedec(data, wavelet, level=level)
        
        # Calculate energy in each level
        energies = [np.sum(np.square(c)) for c in coeffs]
        total_energy = sum(energies)
        
        # Normalize energies
        norm_energies = [e / total_energy for e in energies]
        
        # Calculate ratios between adjacent levels
        ratios = [norm_energies[i] / norm_energies[i+1] if norm_energies[i+1] > 0 else 0 
                 for i in range(len(norm_energies)-1)]
        
        # Calculate phi-optimality
        inverse_phi = 1 / GOLDEN_RATIO
        phi_optimality = [self.calculate_phi_optimality(r, inverse_phi) for r in ratios]
        
        return {
            "wavelet": wavelet,
            "level": level,
            "energies": energies,
            "normalized_energies": norm_energies,
            "energy_ratios": ratios,
            "phi_optimality": phi_optimality,
            "average_phi_optimality": float(np.mean(phi_optimality)) if phi_optimality else 0
        }
        
    def visualize_constant_comparison(self, results):
        """
        Visualize the comparison between different mathematical constants.
        
        Parameters:
        -----------
        results : dict
            Results from analyze_multiple_constants
        """
        constants = results["constants"]
        overall = results["overall_results"]
        
        # Bar chart of average optimality for each constant
        plt.figure(figsize=(12, 6))
        
        # Extract means for each constant
        const_names = list(constants.keys())
        means = [overall[const]["mean"] for const in const_names]
        
        # Create bar chart
        bars = plt.bar(const_names, means)
        
        # Add value labels on top of bars
        for bar, mean in zip(bars, means):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{mean:.3f}', ha='center', va='bottom')
        
        plt.xlabel('Mathematical Constant')
        plt.ylabel('Average Optimality')
        plt.title('Comparison of Pattern Alignment with Different Mathematical Constants')
        plt.ylim(0, max(means) * 1.2)  # Add some space for the labels
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'constant_comparison.png'))
        
        # Create a more detailed boxplot
        plt.figure(figsize=(12, 6))
        
        # Extract all data points for each constant
        data_to_plot = []
        for const in const_names:
            # Get all window results for this constant
            const_data = []
            for window in results["window_results"]:
                if const in window["optimality"]:
                    const_data.extend(window["optimality"][const]["values"])
            data_to_plot.append(const_data)
            
        # Create boxplot
        plt.boxplot(data_to_plot, labels=const_names)
        plt.xlabel('Mathematical Constant')
        plt.ylabel('Optimality')
        plt.title('Distribution of Optimality Scores for Different Mathematical Constants')
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'constant_distribution.png'))
        
    def visualize_temporal_evolution(self, results):
        """
        Visualize the temporal evolution of phi-optimality.
        
        Parameters:
        -----------
        results : dict
            Results from analyze_temporal_evolution
        """
        time_points = results["time_points"]
        phi_optimality = results["phi_optimality"]
        
        plt.figure(figsize=(12, 6))
        plt.plot(time_points, phi_optimality)
        
        # Add a trend line
        z = np.polyfit(time_points, phi_optimality, 1)
        p = np.poly1d(z)
        plt.plot(time_points, p(time_points), "r--", 
                label=f"Trend: y={z[0]:.2e}x+{z[1]:.2f}")
        
        plt.xlabel('Time Point')
        plt.ylabel('Phi-Optimality')
        plt.title('Temporal Evolution of Phi-Optimality')
        plt.legend()
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'temporal_evolution.png'))
        
def main():
    """Main function to demonstrate consciousness pattern analysis."""
    analyzer = ConsciousnessPatternAnalyzer()
    print("Consciousness Pattern Analyzer initialized.")
    
    # Generate synthetic data with embedded golden ratio patterns
    print("Generating synthetic data with embedded golden ratio patterns...")
    np.random.seed(42)
    t = np.linspace(0, 100, 10000)
    
    # Create a signal with frequencies in golden ratio relationships
    f1 = 0.1  # Base frequency
    f2 = f1 * GOLDEN_RATIO
    f3 = f2 * GOLDEN_RATIO
    
    signal_data = (np.sin(2 * np.pi * f1 * t) + 
                  0.5 * np.sin(2 * np.pi * f2 * t) + 
                  0.25 * np.sin(2 * np.pi * f3 * t))
    
    # Add some noise
    noisy_signal = signal_data + 0.1 * np.random.randn(len(t))
    
    # Analyze the signal with multiple constants
    print("Analyzing patterns related to multiple mathematical constants...")
    const_results = analyzer.analyze_multiple_constants(noisy_signal)
    
    # Save results
    with open(os.path.join(analyzer.output_dir, 'constant_analysis_results.json'), 'w') as f:
        # Convert numpy values to Python native types for JSON serialization
        json.dump({
            "constants": const_results["constants"],
            "overall_results": const_results["overall_results"]
        }, f, indent=4)
    
    # Visualize constant comparison
    analyzer.visualize_constant_comparison(const_results)
    
    # Analyze temporal evolution
    print("Analyzing temporal evolution of phi-optimality...")
    temporal_results = analyzer.analyze_temporal_evolution(noisy_signal)
    
    # Save results
    with open(os.path.join(analyzer.output_dir, 'temporal_evolution_results.json'), 'w') as f:
        json.dump({
            "time_points": temporal_results["time_points"],
            "phi_optimality": temporal_results["phi_optimality"]
        }, f, indent=4)
    
    # Visualize temporal evolution
    analyzer.visualize_temporal_evolution(temporal_results)
    
    # Perform wavelet analysis if PyWavelets is available
    if pywt is not None:
        print("Performing wavelet analysis...")
        wavelet_results = analyzer.wavelet_analysis(noisy_signal)
        
        # Save results
        with open(os.path.join(analyzer.output_dir, 'wavelet_analysis_results.json'), 'w') as f:
            json.dump(wavelet_results, f, indent=4)
    
    print("\nAnalysis complete. Results saved to the planck_data directory.")

if __name__ == "__main__":
    main()
