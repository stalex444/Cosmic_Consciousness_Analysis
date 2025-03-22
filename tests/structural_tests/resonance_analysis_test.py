#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Resonance Analysis Test
--------------------
Tests for resonance patterns in the CMB power spectrum.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import pearsonr

from core_framework.base_test import BaseTest
from core_framework.statistics import calculate_significance, bootstrap_confidence_interval
from core_framework.data_handler import generate_surrogate_data
from core_framework.constants import DEFAULT_SEED, PHI


class ResonanceAnalysisTest(BaseTest):
    """Test for resonance patterns in the CMB power spectrum."""
    
    def __init__(self, data, seed=None):
        """
        Initialize the resonance analysis test.
        
        Args:
            data (ndarray): CMB data to analyze
            seed (int, optional): Random seed for reproducibility. Defaults to None.
        """
        super(ResonanceAnalysisTest, self).__init__(data=data, seed=seed)
        self.resonance_score = None
        self.p_value = None
        self.confidence_interval = None
        self.resonance_peaks = None
        self.significance_threshold = 0.05
    
    def run(self):
        """Run the resonance analysis test."""
        # Calculate power spectrum
        f, power_spectrum = signal.welch(self.data, fs=1.0, nperseg=min(len(self.data)//8, 256))
        
        # Find resonance peaks
        self.resonance_peaks = self._find_resonance_peaks(f, power_spectrum)
        
        # Calculate resonance score
        self.resonance_score = self._calculate_resonance_score(self.resonance_peaks)
        
        # Generate surrogate data for significance testing
        n_surrogates = 1000
        surrogate_data = generate_surrogate_data(
            self.data, 
            method='phase_randomization',
            n_surrogates=n_surrogates,
            seed=self.seed
        )
        
        # Calculate resonance scores for surrogate data
        surrogate_scores = np.zeros(n_surrogates)
        for i in range(n_surrogates):
            f_surr, power_surr = signal.welch(surrogate_data[i], fs=1.0, nperseg=min(len(surrogate_data[i])//8, 256))
            peaks_surr = self._find_resonance_peaks(f_surr, power_surr)
            surrogate_scores[i] = self._calculate_resonance_score(peaks_surr)
        
        # Calculate significance
        self.p_value = calculate_significance(self.resonance_score, surrogate_scores)
        
        # Calculate confidence interval
        self.confidence_interval = bootstrap_confidence_interval(
            self.data,
            statistic=lambda x: self._calculate_resonance_score_from_data(x),
            n_bootstrap=1000,
            confidence=0.95,
            seed=self.seed
        )
        
        # Set has_run flag
        self.has_run = True
    
    def _find_resonance_peaks(self, frequencies, power_spectrum, prominence=0.1):
        """
        Find resonance peaks in the power spectrum.
        
        Args:
            frequencies (ndarray): Frequency values
            power_spectrum (ndarray): Power spectrum values
            prominence (float, optional): Peak prominence threshold. Defaults to 0.1.
            
        Returns:
            dict: Dictionary with peak information
        """
        # Normalize power spectrum
        power_norm = power_spectrum / np.max(power_spectrum)
        
        # Find peaks
        peaks, properties = signal.find_peaks(power_norm, prominence=prominence)
        
        # Extract peak information
        peak_freqs = frequencies[peaks]
        peak_powers = power_norm[peaks]
        peak_prominences = properties['prominences']
        
        # Check for golden ratio relationships between peaks
        n_peaks = len(peaks)
        gr_relationships = []
        
        for i in range(n_peaks):
            for j in range(i+1, n_peaks):
                ratio_ij = peak_freqs[i] / peak_freqs[j]
                ratio_ji = peak_freqs[j] / peak_freqs[i]
                
                # Check if either ratio is close to phi
                if abs(ratio_ij - PHI) < 0.1:
                    gr_relationships.append((i, j, ratio_ij))
                elif abs(ratio_ji - PHI) < 0.1:
                    gr_relationships.append((j, i, ratio_ji))
        
        # Return peak information
        return {
            'frequencies': peak_freqs,
            'powers': peak_powers,
            'prominences': peak_prominences,
            'gr_relationships': gr_relationships
        }
    
    def _calculate_resonance_score(self, resonance_peaks):
        """
        Calculate resonance score from resonance peaks.
        
        Args:
            resonance_peaks (dict): Dictionary with peak information
            
        Returns:
            float: Resonance score
        """
        # If no peaks found, return 0
        if len(resonance_peaks['frequencies']) == 0:
            return 0.0
        
        # Calculate resonance score based on:
        # 1. Number of peaks (more peaks = higher score)
        # 2. Mean prominence of peaks (more prominent = higher score)
        # 3. Number of golden ratio relationships (more relationships = higher score)
        
        n_peaks = len(resonance_peaks['frequencies'])
        mean_prominence = np.mean(resonance_peaks['prominences'])
        n_gr_relationships = len(resonance_peaks['gr_relationships'])
        
        # Normalize each component
        max_peaks = 10  # Assume 10 peaks is the maximum expected
        norm_peaks = min(n_peaks / max_peaks, 1.0)
        
        norm_prominence = mean_prominence  # Already normalized
        
        max_gr_relationships = n_peaks * (n_peaks - 1) / 2  # Maximum possible relationships
        norm_gr_relationships = 0.0 if max_gr_relationships == 0 else n_gr_relationships / max_gr_relationships
        
        # Weighted sum of components
        resonance_score = 0.3 * norm_peaks + 0.3 * norm_prominence + 0.4 * norm_gr_relationships
        
        return resonance_score
    
    def _calculate_resonance_score_from_data(self, data):
        """
        Calculate resonance score directly from data.
        
        Args:
            data (ndarray): Data to analyze
            
        Returns:
            float: Resonance score
        """
        # Calculate power spectrum
        f, power_spectrum = signal.welch(data, fs=1.0, nperseg=min(len(data)//8, 256))
        
        # Find resonance peaks
        resonance_peaks = self._find_resonance_peaks(f, power_spectrum)
        
        # Calculate resonance score
        resonance_score = self._calculate_resonance_score(resonance_peaks)
        
        return resonance_score
    
    def generate_report(self):
        """
        Generate a report of the test results.
        
        Returns:
            str: Report text
        """
        if not self.has_run:
            return "Resonance analysis test has not been run yet."
        
        report = []
        report.append("=" * 50)
        report.append("RESONANCE ANALYSIS TEST RESULTS")
        report.append("=" * 50)
        report.append("")
        
        report.append("Resonance Score: {:.4f}".format(self.resonance_score))
        report.append("p-value: {:.4f}".format(self.p_value))
        report.append("95% Confidence Interval: [{:.4f}, {:.4f}]".format(
            self.confidence_interval[0], self.confidence_interval[1]
        ))
        
        report.append("")
        report.append("Peak Information:")
        report.append("- Number of peaks: {}".format(len(self.resonance_peaks['frequencies'])))
        report.append("- Mean peak prominence: {:.4f}".format(np.mean(self.resonance_peaks['prominences'])))
        report.append("- Number of golden ratio relationships: {}".format(len(self.resonance_peaks['gr_relationships'])))
        
        if self.resonance_peaks['gr_relationships']:
            report.append("")
            report.append("Golden Ratio Relationships:")
            for i, j, ratio in self.resonance_peaks['gr_relationships']:
                report.append("- f{} / f{} = {:.4f} (frequencies: {:.4f} Hz, {:.4f} Hz)".format(
                    i, j, ratio, 
                    self.resonance_peaks['frequencies'][i], 
                    self.resonance_peaks['frequencies'][j]
                ))
        
        report.append("")
        report.append("Interpretation:")
        if self.p_value < self.significance_threshold:
            report.append("- The CMB data shows statistically significant resonance patterns")
            report.append("  (p < {:.2f}).".format(self.significance_threshold))
        else:
            report.append("- The resonance patterns in the CMB data are not statistically")
            report.append("  significant (p >= {:.2f}).".format(self.significance_threshold))
        
        if self.resonance_score > 0.7:
            report.append("- The resonance score is high, indicating strong resonance patterns")
            report.append("  in the CMB power spectrum.")
        elif self.resonance_score > 0.4:
            report.append("- The resonance score is moderate, indicating some resonance patterns")
            report.append("  in the CMB power spectrum.")
        else:
            report.append("- The resonance score is low, indicating weak resonance patterns")
            report.append("  in the CMB power spectrum.")
        
        return "\n".join(report)
    
    def visualize(self, output_dir=None):
        """
        Create visualizations of the test results.
        
        Args:
            output_dir (str, optional): Directory to save visualizations. Defaults to None.
        """
        if not self.has_run:
            print("Cannot visualize results: Resonance analysis test has not been run yet.")
            return
        
        # Calculate power spectrum for visualization
        f, power_spectrum = signal.welch(self.data, fs=1.0, nperseg=min(len(self.data)//8, 256))
        power_norm = power_spectrum / np.max(power_spectrum)
        
        # Create figure with 2 subplots
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot power spectrum with peaks
        axes[0].plot(f, power_norm, 'b-', label='Power Spectrum')
        
        # Plot peaks
        peak_freqs = self.resonance_peaks['frequencies']
        peak_powers = self.resonance_peaks['powers']
        axes[0].plot(peak_freqs, peak_powers, 'ro', label='Resonance Peaks')
        
        axes[0].set_title('Power Spectrum with Resonance Peaks')
        axes[0].set_xlabel('Frequency (Hz)')
        axes[0].set_ylabel('Normalized Power')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot golden ratio relationships
        if self.resonance_peaks['gr_relationships']:
            # Create a relationship matrix
            n_peaks = len(peak_freqs)
            relationship_matrix = np.zeros((n_peaks, n_peaks))
            
            for i, j, ratio in self.resonance_peaks['gr_relationships']:
                relationship_matrix[i, j] = ratio
                relationship_matrix[j, i] = 1.0 / ratio
            
            # Plot as heatmap
            im = axes[1].imshow(relationship_matrix, cmap='viridis', origin='lower')
            axes[1].set_title('Golden Ratio Relationships Between Peaks')
            axes[1].set_xlabel('Peak Index')
            axes[1].set_ylabel('Peak Index')
            fig.colorbar(im, ax=axes[1], label='Frequency Ratio')
            
            # Add text annotations
            for i in range(n_peaks):
                for j in range(n_peaks):
                    if relationship_matrix[i, j] > 0:
                        axes[1].text(j, i, "{:.2f}".format(relationship_matrix[i, j]), 
                                    ha="center", va="center", color="white" if relationship_matrix[i, j] > 0.5 else "black")
        else:
            axes[1].text(0.5, 0.5, "No Golden Ratio Relationships Found", 
                        ha="center", va="center", fontsize=14, transform=axes[1].transAxes)
            axes[1].set_title('Golden Ratio Relationships Between Peaks')
        
        # Add overall title with p-value and resonance score
        significance_str = "Significant" if self.p_value < self.significance_threshold else "Not Significant"
        fig.suptitle('Resonance Analysis Results (p={:.4f}, score={:.4f}, {})'.format(
            self.p_value, self.resonance_score, significance_str), fontsize=16)
        
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save or show
        if output_dir is not None:
            import os
            output_path = os.path.join(output_dir, 'resonance_analysis_results.png')
            fig.savefig(output_path, dpi=300)
            plt.close(fig)
            print("Visualization saved to {}".format(output_path))
        else:
            plt.show()
