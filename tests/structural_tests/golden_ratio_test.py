#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Golden Ratio Test
---------------
Tests for golden ratio patterns in the structural properties of CMB data.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, stats
import os

from core_framework.base_test import BaseTest
from core_framework.constants import (
    CONSTANTS, COLORS, CONSTANT_NAMES, DEFAULT_MONTE_CARLO_ITERATIONS,
    DEFAULT_BOOTSTRAP_SAMPLES, PHI
)
from core_framework.data_handler import (
    generate_surrogate_data, segment_data, fibonacci_sequence
)
from core_framework.statistics import (
    bootstrap_confidence_interval, calculate_phi_optimality, find_best_constant
)
from core_framework.visualization import (
    create_multi_panel_figure, setup_figure, save_figure, plot_power_spectrum
)


class GoldenRatioTest(BaseTest):
    """
    Test for golden ratio patterns in the structural properties of CMB data.
    
    This test examines if the power spectrum of the CMB data shows enhanced
    power at frequencies related by the golden ratio, which would indicate
    a phi-based organizing principle.
    
    Attributes:
        n_surrogates (int): Number of surrogate datasets for significance testing
        bootstrap_samples (int): Number of bootstrap samples for confidence intervals
        frequency_bands (list): List of frequency bands to analyze
    """
    
    def __init__(self, name="Golden Ratio Test", seed=None, data=None):
        """
        Initialize the golden ratio test.
        
        Args:
            name (str, optional): Name of the test. Defaults to "Golden Ratio Test".
            seed (int, optional): Random seed for reproducibility. Defaults to None.
            data (ndarray, optional): CMB data to analyze. Defaults to None.
        """
        super(GoldenRatioTest, self).__init__(name=name, seed=seed, data=data)
        
        # Test parameters
        self.n_surrogates = 100
        self.bootstrap_samples = DEFAULT_BOOTSTRAP_SAMPLES
        
        # Generate frequency bands based on golden ratio
        self.frequency_bands = self._generate_frequency_bands()
    
    def _generate_frequency_bands(self, min_freq=0.001, max_freq=0.5, n_bands=12):
        """
        Generate frequency bands for analysis based on golden ratio.
        
        Args:
            min_freq (float, optional): Minimum frequency. Defaults to 0.001.
            max_freq (float, optional): Maximum frequency. Defaults to 0.5.
            n_bands (int, optional): Number of frequency bands. Defaults to 12.
            
        Returns:
            list: List of frequency bands (tuples of lower and upper bounds)
        """
        # Generate Fibonacci sequence
        fib_seq = fibonacci_sequence(n_bands + 5)
        
        # Filter and scale to desired range
        fib_seq = [f for f in fib_seq if f >= 1]  # Remove 0
        
        # Scale to desired frequency range
        min_fib = min(fib_seq)
        max_fib = max(fib_seq)
        
        freqs = [min_freq + (max_freq - min_freq) * (f - min_fib) / (max_fib - min_fib) for f in fib_seq]
        
        # Create frequency bands
        bands = []
        for i in range(len(freqs) - 1):
            bands.append((freqs[i], freqs[i+1]))
        
        # Limit to n_bands
        if len(bands) > n_bands:
            # Select evenly spaced bands
            indices = np.linspace(0, len(bands) - 1, n_bands).astype(int)
            bands = [bands[i] for i in indices]
        
        return bands
    
    def run_test(self):
        """
        Run the golden ratio test.
        
        Returns:
            dict: Test results
        """
        print("Running {}...".format(self.name))
        
        # Ensure data is loaded
        if self.data is None:
            self.load_data()
        
        # Initialize results
        self.results = {
            'frequency_bands': self.frequency_bands,
            'power_spectrum': None,
            'band_powers': None,
            'power_ratios': None,
            'phi_optimality': None,
            'surrogate_results': None,
            'p_value': None,
            'confidence_interval': None
        }
        
        # Calculate power spectrum
        frequencies, power_spectrum = self._calculate_power_spectrum()
        self.results['frequencies'] = frequencies
        self.results['power_spectrum'] = power_spectrum
        
        # Calculate band powers
        band_powers = self._calculate_band_powers(frequencies, power_spectrum)
        self.results['band_powers'] = band_powers
        
        # Calculate power ratios
        power_ratios = self._calculate_power_ratios(band_powers)
        self.results['power_ratios'] = power_ratios
        
        # Calculate phi optimality
        phi_optimality = calculate_phi_optimality(power_ratios, CONSTANTS)
        self.results['phi_optimality'] = phi_optimality
        
        # Find best constant
        best_constant, best_value = find_best_constant(phi_optimality)
        self.results['best_constant'] = best_constant
        self.results['best_value'] = best_value
        
        # Test significance
        p_value, surrogate_results = self._test_significance(power_ratios)
        self.results['p_value'] = p_value
        self.results['surrogate_results'] = surrogate_results
        
        # Calculate confidence interval
        confidence_interval = self._calculate_confidence_interval(power_ratios)
        self.results['confidence_interval'] = confidence_interval
        
        print("\n")
        return self.results
    
    def _calculate_power_spectrum(self):
        """
        Calculate the power spectrum of the data.
        
        Returns:
            tuple: Frequencies and power spectrum
        """
        # Calculate power spectrum using Welch's method
        fs = 1.0  # Sampling frequency (arbitrary)
        nperseg = min(1024, len(self.data) // 2)
        
        frequencies, power_spectrum = signal.welch(self.data, fs=fs, nperseg=nperseg)
        
        return frequencies, power_spectrum
    
    def _calculate_band_powers(self, frequencies, power_spectrum):
        """
        Calculate power in each frequency band.
        
        Args:
            frequencies (ndarray): Frequency values
            power_spectrum (ndarray): Power spectrum values
            
        Returns:
            dict: Power in each frequency band
        """
        band_powers = {}
        
        for i, (low_freq, high_freq) in enumerate(self.frequency_bands):
            # Find indices within frequency band
            indices = np.where((frequencies >= low_freq) & (frequencies < high_freq))[0]
            
            if len(indices) > 0:
                # Calculate mean power in band
                band_power = np.mean(power_spectrum[indices])
                band_powers[(low_freq, high_freq)] = band_power
        
        return band_powers
    
    def _calculate_power_ratios(self, band_powers):
        """
        Calculate ratios of powers between frequency bands.
        
        Args:
            band_powers (dict): Power in each frequency band
            
        Returns:
            dict: Power ratios between bands
        """
        power_ratios = {}
        bands = list(band_powers.keys())
        
        for i, band_i in enumerate(bands):
            for j, band_j in enumerate(bands):
                if j <= i:  # Only calculate upper triangle
                    continue
                
                # Calculate center frequencies
                freq_i = (band_i[0] + band_i[1]) / 2
                freq_j = (band_j[0] + band_j[1]) / 2
                
                # Calculate frequency ratio (always larger / smaller)
                if freq_j > freq_i:
                    freq_ratio = freq_j / freq_i
                else:
                    freq_ratio = freq_i / freq_j
                
                # Round ratio to 4 decimal places
                freq_ratio = round(freq_ratio, 4)
                
                # Calculate power ratio
                power_i = band_powers[band_i]
                power_j = band_powers[band_j]
                
                # Skip if powers are too small
                if power_i < 1e-10 or power_j < 1e-10:
                    continue
                
                # Calculate power ratio (geometric mean of both directions)
                power_ratio = np.sqrt((power_j / power_i) * (power_i / power_j))
                
                # Store or update ratio
                if freq_ratio in power_ratios:
                    power_ratios[freq_ratio] += power_ratio
                else:
                    power_ratios[freq_ratio] = power_ratio
        
        return power_ratios
    
    def _test_significance(self, power_ratios):
        """
        Test statistical significance of power ratios.
        
        Args:
            power_ratios (dict): Power ratios between frequency bands
            
        Returns:
            tuple: p-value and surrogate results
        """
        # Generate surrogate data
        surrogates = generate_surrogate_data(self.data, n_surrogates=self.n_surrogates, seed=self.seed)
        
        # Calculate power ratios for each surrogate
        surrogate_results = []
        
        for i in range(self.n_surrogates):
            # Create surrogate test
            surrogate_test = GoldenRatioTest(
                name="Surrogate Golden Ratio Test {}".format(i),
                seed=self.seed + i if self.seed is not None else None,
                data=surrogates[i]
            )
            
            # Set the same frequency bands
            surrogate_test.frequency_bands = self.frequency_bands
            
            # Calculate power spectrum
            surrogate_frequencies, surrogate_power_spectrum = surrogate_test._calculate_power_spectrum()
            
            # Calculate band powers
            surrogate_band_powers = surrogate_test._calculate_band_powers(surrogate_frequencies, surrogate_power_spectrum)
            
            # Calculate power ratios
            surrogate_power_ratios = surrogate_test._calculate_power_ratios(surrogate_band_powers)
            
            # Calculate phi optimality
            surrogate_phi_optimality = calculate_phi_optimality(surrogate_power_ratios, CONSTANTS)
            
            # Find best constant
            surrogate_best_constant, surrogate_best_value = find_best_constant(surrogate_phi_optimality)
            
            # Store results
            surrogate_results.append({
                'power_ratios': surrogate_power_ratios,
                'phi_optimality': surrogate_phi_optimality,
                'best_constant': surrogate_best_constant,
                'best_value': surrogate_best_value
            })
        
        # Calculate p-value based on phi optimality for golden ratio
        real_phi_values = [opt_dict['phi'] for opt_dict in self.results['phi_optimality'].values()]
        real_phi_optimality = np.mean(real_phi_values)
        
        surrogate_phi_values = []
        for s in surrogate_results:
            s_phi_values = [opt_dict['phi'] for opt_dict in s['phi_optimality'].values()]
            surrogate_phi_values.append(np.mean(s_phi_values))
        
        p_value = np.mean(np.array(surrogate_phi_values) >= real_phi_optimality)
        
        return p_value, surrogate_results
    
    def _calculate_confidence_interval(self, power_ratios):
        """
        Calculate bootstrap confidence interval for phi optimality.
        
        Args:
            power_ratios (dict): Power ratios between frequency bands
            
        Returns:
            tuple: Lower and upper bounds of confidence interval
        """
        # Generate bootstrap samples
        bootstrap_values = []
        
        for i in range(self.bootstrap_samples):
            # Resample data with replacement
            bootstrap_data = self._generate_bootstrap_sample()
            
            # Create bootstrap test
            bootstrap_test = GoldenRatioTest(
                name="Bootstrap Golden Ratio Test {}".format(i),
                seed=self.seed + i if self.seed is not None else None,
                data=bootstrap_data
            )
            
            # Set the same frequency bands
            bootstrap_test.frequency_bands = self.frequency_bands
            
            # Calculate power spectrum
            bootstrap_frequencies, bootstrap_power_spectrum = bootstrap_test._calculate_power_spectrum()
            
            # Calculate band powers
            bootstrap_band_powers = bootstrap_test._calculate_band_powers(bootstrap_frequencies, bootstrap_power_spectrum)
            
            # Calculate power ratios
            bootstrap_power_ratios = bootstrap_test._calculate_power_ratios(bootstrap_band_powers)
            
            # Calculate phi optimality
            bootstrap_phi_optimality = calculate_phi_optimality(bootstrap_power_ratios, CONSTANTS)
            
            # Store phi optimality for golden ratio
            bootstrap_values.append(np.mean([opt_dict['phi'] for opt_dict in bootstrap_phi_optimality.values()]))
        
        # Calculate confidence interval
        confidence_interval = bootstrap_confidence_interval(
            bootstrap_values,
            statistic_func=np.mean  # Use mean as the statistic function
        )
        
        return confidence_interval
    
    def _generate_bootstrap_sample(self):
        """
        Generate a bootstrap sample of the data.
        
        Returns:
            ndarray: Bootstrap sample
        """
        # Resample data with replacement
        indices = np.random.choice(len(self.data), size=len(self.data), replace=True)
        bootstrap_data = self.data[indices]
        
        return bootstrap_data
    
    def visualize(self, output_dir):
        """
        Visualize the results of the Golden Ratio Test.
        
        Args:
            output_dir (str): Directory to save visualizations
        """
        if not self.results:
            print("No results to visualize for Golden Ratio Test.")
            return
        
        # Create figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Plot power ratios
        ax = axes[0, 0]
        power_ratios = self.results['power_ratios']
        ratios = sorted(power_ratios.keys())
        values = [power_ratios[r] for r in ratios]
        
        ax.bar(range(len(ratios)), values, color='skyblue')
        ax.set_xticks(range(len(ratios)))
        ax.set_xticklabels(["{:.2f}".format(r) for r in ratios], rotation=45)
        ax.set_xlabel('Ratio')
        ax.set_ylabel('Power Ratio')
        ax.set_title('Power Ratios Between Frequency Bands')
        ax.grid(True, alpha=0.3)
        
        # 2. Plot phi optimality for each ratio
        ax = axes[0, 1]
        if isinstance(self.results['phi_optimality'], dict):
            # Extract phi values for each ratio
            phi_values = {}
            for ratio, opt_dict in self.results['phi_optimality'].items():
                if 'phi' in opt_dict:
                    phi_values[ratio] = opt_dict['phi']
            
            if phi_values:
                ratios = sorted(phi_values.keys())
                values = [phi_values[r] for r in ratios]
                
                ax.bar(range(len(ratios)), values, color='gold')
                ax.set_xticks(range(len(ratios)))
                ax.set_xticklabels(["{:.2f}".format(r) for r in ratios], rotation=45)
                ax.set_xlabel('Ratio')
                ax.set_ylabel('Phi Optimality')
                ax.set_title('Phi Optimality by Ratio')
                ax.grid(True, alpha=0.3)
                
                # Add horizontal line at phi bias
                ax.axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='Phi Bias')
                ax.legend()
        
        # 3. Plot surrogate distribution
        ax = axes[1, 0]
        if 'surrogate_results' in self.results:
            # Extract average phi optimality for each surrogate
            surrogate_values = []
            for s in self.results['surrogate_results']:
                if 'phi_optimality' in s:
                    if isinstance(s['phi_optimality'], dict):
                        # Calculate average across all ratios
                        phi_values = [opt_dict['phi'] for opt_dict in s['phi_optimality'].values() 
                                    if 'phi' in opt_dict]
                        if phi_values:
                            surrogate_values.append(np.mean(phi_values))
            
            if surrogate_values:
                # Calculate real phi optimality (average across all ratios)
                real_phi_values = [opt_dict['phi'] for opt_dict in self.results['phi_optimality'].values() 
                                if 'phi' in opt_dict]
                real_phi = np.mean(real_phi_values) if real_phi_values else 0
                
                # Plot histogram of surrogate values
                ax.hist(surrogate_values, bins=20, alpha=0.7, color='lightblue')
                ax.axvline(x=real_phi, color='red', linestyle='--', 
                          label='Observed Value (p={:.3f})'.format(self.results['p_value']))
                ax.set_xlabel('Phi Optimality')
                ax.set_ylabel('Frequency')
                ax.set_title('Surrogate Distribution')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        # 4. Plot confidence interval
        ax = axes[1, 1]
        if 'confidence_interval' in self.results:
            ci = self.results['confidence_interval']
            
            # Calculate real phi optimality (average across all ratios)
            real_phi_values = [opt_dict['phi'] for opt_dict in self.results['phi_optimality'].values() 
                            if 'phi' in opt_dict]
            real_phi = np.mean(real_phi_values) if real_phi_values else 0
            
            # Create a simple visualization of the confidence interval
            ax.errorbar(x=[0], y=[real_phi], yerr=[[real_phi-ci[0]], [ci[1]-real_phi]], 
                      fmt='o', color='blue', ecolor='lightblue', capsize=10, capthick=2, markersize=10)
            ax.set_xlim(-0.5, 0.5)
            ax.set_xticks([])
            ax.set_ylabel('Phi Optimality')
            ax.set_title('95% Confidence Interval')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(output_dir, 'golden_ratio_test.png')
        # Ensure the directory exists
        try:
            # Make sure we're not trying to create a directory for the file itself
            directory = os.path.dirname(output_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
        except OSError:
            if not os.path.isdir(os.path.dirname(output_path)):
                raise
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Golden Ratio Test visualization saved to: {}".format(output_path))
    
    def generate_report(self):
        """
        Generate a report of the test results.
        
        Returns:
            str: Report text
        """
        if not self.results:
            print("No results to report. Run the test first.")
            return ""
        
        report = []
        report.append("=" * 80)
        report.append("GOLDEN RATIO TEST REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Execution time
        if hasattr(self, 'execution_time') and self.execution_time is not None:
            report.append("Test completed in {:.2f} seconds.".format(self.execution_time))
            report.append("")
        
        # Power ratios
        report.append("TOP FREQUENCY RATIOS WITH ENHANCED POWER")
        report.append("-" * 50)
        
        # Sort ratios by power value
        sorted_ratios = sorted(self.results['power_ratios'].items(), key=lambda x: x[1], reverse=True)
        
        # Display top 10 ratios
        for i, (ratio, power_value) in enumerate(sorted_ratios[:10]):
            # Find closest constant
            closest_const = None
            min_diff = float('inf')
            
            for const_name, const_value in CONSTANTS.items():
                diff = abs(ratio - const_value)
                if diff < min_diff:
                    min_diff = diff
                    closest_const = const_name
            
            # Calculate percent difference
            percent_diff = 100 * min_diff / CONSTANTS[closest_const]
            
            report.append("{}. Ratio {:.4f} (Power = {:.6f})".format(i + 1, ratio, power_value))
            report.append("   Closest to {} = {:.4f} (diff: {:.2f}%)".format(
                closest_const, CONSTANTS[closest_const], percent_diff))
        
        report.append("")
        
        # Phi optimality
        report.append("PHI OPTIMALITY")
        report.append("-" * 50)
        for const, value in self.results['phi_optimality'].items():
            report.append("{}: {:.6f}".format(const, value))
        report.append("")
        report.append("Best constant: {} ({:.6f})".format(self.results['best_constant'], self.results['best_value']))
        report.append("")
        
        # Statistical significance
        report.append("STATISTICAL SIGNIFICANCE")
        report.append("-" * 50)
        report.append("p-value: {:.6f}".format(self.results['p_value']))
        report.append("Significance: {}".format("SIGNIFICANT (p < 0.05)" if self.results['p_value'] < 0.05 else "NOT SIGNIFICANT"))
        report.append("")
        
        # Confidence interval
        report.append("BOOTSTRAP CONFIDENCE INTERVAL (95%)")
        report.append("-" * 50)
        report.append("Lower bound: {:.6f}".format(self.results['confidence_interval'][0]))
        report.append("Upper bound: {:.6f}".format(self.results['confidence_interval'][1]))
        report.append("Excludes zero: {}".format("YES" if self.results['confidence_interval'][0] > 0 else "NO"))
        report.append("")
        
        # Interpretation
        report.append("INTERPRETATION")
        report.append("-" * 50)
        report.append("The golden ratio test examines if the power spectrum of the CMB data shows")
        report.append("enhanced power at frequencies related by the golden ratio, which would")
        report.append("indicate a phi-based organizing principle.")
        report.append("")
        
        if self.results['p_value'] < 0.05:
            report.append("The analysis reveals STATISTICALLY SIGNIFICANT golden ratio patterns")
            report.append("in the CMB data, suggesting that the power spectrum is organized in a")
            report.append("way that preferentially enhances power at frequencies related by phi.")
        else:
            report.append("The analysis does not detect statistically significant golden ratio")
            report.append("patterns in the CMB data at the p < 0.05 level.")
        
        report.append("")
        
        if self.results['best_constant'] == 'phi':
            report.append("Notably, the golden ratio (phi) shows the strongest optimization in the")
            report.append("frequency ratios with enhanced power, which aligns with the hypothesis")
            report.append("that phi-based organization may be a signature of consciousness-like")
            report.append("processes in cosmic structure.")
        else:
            report.append("The constant {} shows the strongest optimization in the frequency".format(self.results['best_constant']))
            report.append("ratios with enhanced power, which suggests that this mathematical")
            report.append("relationship may play a significant role in the organization of")
            report.append("cosmic structure.")
        
        report.append("")
        report.append("=" * 80)
        
        # Print report
        print("\n".join(report))
        
        # Store report in results
        self.results['report'] = "\n".join(report)
        
        return self.results['report']
    
    def visualize_results(self, save_path=None, show=False):
        """
        Create visualizations of the test results.
        
        Args:
            save_path (str, optional): Path to save visualizations. Defaults to None.
            show (bool, optional): Whether to display the visualizations. Defaults to False.
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        if not self.results:
            print("No results to visualize. Run the test first.")
            return None
        
        # Create figure with 2x2 grid
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # Plot power spectrum
        plot_power_spectrum(axes[0, 0], self.data, fs=1.0, method='welch', nperseg=1024)
        axes[0, 0].set_title("Power Spectrum of CMB Data")
        
        # Highlight frequency bands
        for low_freq, high_freq in self.frequency_bands:
            axes[0, 0].axvspan(low_freq, high_freq, alpha=0.2, color='red')
        
        # Plot power ratios
        sorted_ratios = sorted(self.results['power_ratios'].items(), key=lambda x: x[1], reverse=True)
        top_ratios = sorted_ratios[:10]
        
        ratios = [r[0] for r in top_ratios]
        power_values = [r[1] for r in top_ratios]
        
        # Create colors based on proximity to constants
        colors = []
        for ratio in ratios:
            closest_const = None
            min_diff = float('inf')
            
            for const_name, const_value in CONSTANTS.items():
                diff = abs(ratio - const_value)
                if diff < min_diff:
                    min_diff = diff
                    closest_const = const_name
            
            colors.append(COLORS.get(closest_const, 'gray'))
        
        axes[0, 1].bar(range(len(top_ratios)), power_values, color=colors)
        axes[0, 1].set_title("Top Frequency Ratios with Enhanced Power")
        axes[0, 1].set_xlabel("Rank")
        axes[0, 1].set_ylabel("Power Enhancement")
        
        # Add ratio labels
        axes[0, 1].set_xticks(range(len(top_ratios)))
        axes[0, 1].set_xticklabels(["{:.2f}".format(r) for r in ratios], rotation=45)
        
        # Plot phi optimality
        constants = list(self.results['phi_optimality'].keys())
        values = [self.results['phi_optimality'][c] for c in constants]
        colors = [COLORS.get(c, 'gray') for c in constants]
        
        axes[1, 0].bar(constants, values, color=colors)
        axes[1, 0].set_title("Optimization by Mathematical Constant")
        axes[1, 0].set_xlabel("Mathematical Constant")
        axes[1, 0].set_ylabel("Optimization Value")
        
        # Highlight best constant
        best_idx = constants.index(self.results['best_constant'])
        axes[1, 0].text(
            best_idx, values[best_idx] + 0.01,
            "Best",
            ha='center', va='bottom',
            fontweight='bold'
        )
        
        # Plot surrogate distribution
        surrogate_phi_values = [s['phi_optimality']['phi'] for s in self.results['surrogate_results']]
        
        axes[1, 1].hist(surrogate_phi_values, bins=20, alpha=0.7, color='gray')
        axes[1, 1].axvline(self.results['phi_optimality']['phi'], color='red', linestyle='--', linewidth=2)
        axes[1, 1].set_title("Phi Optimization vs Surrogate Distribution")
        axes[1, 1].set_xlabel("Phi Optimization Value")
        axes[1, 1].set_ylabel("Frequency")
        
        # Add p-value annotation
        axes[1, 1].text(
            0.95, 0.95,
            "p-value: {:.4f}".format(self.results['p_value']),
            transform=axes[1, 1].transAxes,
            ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
        )
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure if path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print("Figure saved to {}".format(save_path))
        elif not show:
            # Save to default location
            save_figure(fig, "golden_ratio_results.png")
        
        # Show figure if requested
        if show:
            plt.show()
        
        return fig


def main():
    """
    Run the golden ratio test.
    """
    # Create and run test
    test = GoldenRatioTest(seed=42)
    test.run()
    
    # Generate report and visualizations
    test.generate_report()
    test.visualize_results()


if __name__ == "__main__":
    main()
