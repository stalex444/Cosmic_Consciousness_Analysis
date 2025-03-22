#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Hierarchical Organization Test
---------------------------
Tests for hierarchical patterns in the CMB data based on the golden ratio.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import pearsonr

from core_framework.base_test import BaseTest
from core_framework.statistics import calculate_significance, bootstrap_confidence_interval
from core_framework.data_handler import generate_surrogate_data, segment_data
from core_framework.constants import DEFAULT_SEED, PHI


class HierarchicalOrganizationTest(BaseTest):
    """Test for hierarchical organization patterns based on the golden ratio."""
    
    def __init__(self, data, seed=None):
        """
        Initialize the hierarchical organization test.
        
        Args:
            data (ndarray): CMB data to analyze
            seed (int, optional): Random seed for reproducibility. Defaults to None.
        """
        super(HierarchicalOrganizationTest, self).__init__(data=data, seed=seed)
        self.hierarchy_score = None
        self.p_value = None
        self.confidence_interval = None
        self.hierarchy_levels = None
        self.level_correlations = None
        self.significance_threshold = 0.05
    
    def run(self):
        """Run the hierarchical organization test."""
        # Calculate power spectrum
        power_spectrum = self._calculate_power_spectrum(self.data)
        
        # Generate hierarchical levels based on golden ratio
        self.hierarchy_levels = self._generate_hierarchy_levels(power_spectrum)
        
        # Calculate correlations between adjacent levels
        self.level_correlations = self._calculate_level_correlations(self.hierarchy_levels)
        
        # Calculate hierarchy score (mean of absolute correlations)
        self.hierarchy_score = np.mean(np.abs(self.level_correlations))
        
        # Generate surrogate data for significance testing
        n_surrogates = 1000
        surrogate_data = generate_surrogate_data(
            self.data, 
            method='phase_randomization',
            n_surrogates=n_surrogates,
            seed=self.seed
        )
        
        # Calculate hierarchy scores for surrogate data
        surrogate_scores = np.zeros(n_surrogates)
        for i in range(n_surrogates):
            surrogate_spectrum = self._calculate_power_spectrum(surrogate_data[i])
            surrogate_levels = self._generate_hierarchy_levels(surrogate_spectrum)
            surrogate_correlations = self._calculate_level_correlations(surrogate_levels)
            surrogate_scores[i] = np.mean(np.abs(surrogate_correlations))
        
        # Calculate significance
        self.p_value = calculate_significance(self.hierarchy_score, surrogate_scores)
        
        # Calculate confidence interval
        self.confidence_interval = bootstrap_confidence_interval(
            self.data,
            statistic=lambda x: self._calculate_hierarchy_score(x),
            n_bootstrap=1000,
            confidence=0.95,
            seed=self.seed
        )
        
        # Set has_run flag
        self.has_run = True
    
    def _calculate_power_spectrum(self, data):
        """
        Calculate power spectrum of the data.
        
        Args:
            data (ndarray): Data to analyze
            
        Returns:
            ndarray: Power spectrum
        """
        # Calculate power spectrum using Welch's method
        f, Pxx = signal.welch(data, fs=1.0, nperseg=min(len(data)//8, 256))
        
        return Pxx
    
    def _generate_hierarchy_levels(self, power_spectrum, n_levels=5):
        """
        Generate hierarchical levels based on golden ratio.
        
        Args:
            power_spectrum (ndarray): Power spectrum to analyze
            n_levels (int, optional): Number of hierarchy levels. Defaults to 5.
            
        Returns:
            list: List of power spectrum segments at different hierarchical levels
        """
        # Create hierarchical levels
        levels = []
        
        # Start with the full spectrum
        levels.append(power_spectrum)
        
        # Generate subsequent levels by dividing according to golden ratio
        current_spectrum = power_spectrum
        for i in range(1, n_levels):
            # Calculate split point according to golden ratio
            split_point = int(len(current_spectrum) / PHI)
            
            # Split the spectrum
            next_level = current_spectrum[:split_point]
            current_spectrum = current_spectrum[split_point:]
            
            # Add to levels
            levels.append(next_level)
        
        return levels
    
    def _calculate_level_correlations(self, hierarchy_levels):
        """
        Calculate correlations between adjacent hierarchy levels.
        
        Args:
            hierarchy_levels (list): List of power spectrum segments at different levels
            
        Returns:
            ndarray: Correlations between adjacent levels
        """
        n_levels = len(hierarchy_levels)
        correlations = np.zeros(n_levels - 1)
        
        for i in range(n_levels - 1):
            # Get adjacent levels
            level1 = hierarchy_levels[i]
            level2 = hierarchy_levels[i+1]
            
            # Ensure levels have the same length for correlation
            min_length = min(len(level1), len(level2))
            level1 = level1[:min_length]
            level2 = level2[:min_length]
            
            # Calculate correlation
            if min_length > 1:  # Need at least 2 points for correlation
                correlation, _ = pearsonr(level1, level2)
                correlations[i] = correlation
            else:
                correlations[i] = 0.0
        
        return correlations
    
    def _calculate_hierarchy_score(self, data):
        """
        Calculate hierarchy score for the given data.
        
        Args:
            data (ndarray): Data to analyze
            
        Returns:
            float: Hierarchy score
        """
        # Calculate power spectrum
        power_spectrum = self._calculate_power_spectrum(data)
        
        # Generate hierarchical levels
        hierarchy_levels = self._generate_hierarchy_levels(power_spectrum)
        
        # Calculate correlations between adjacent levels
        level_correlations = self._calculate_level_correlations(hierarchy_levels)
        
        # Calculate hierarchy score (mean of absolute correlations)
        hierarchy_score = np.mean(np.abs(level_correlations))
        
        return hierarchy_score
    
    def generate_report(self):
        """
        Generate a report of the test results.
        
        Returns:
            str: Report text
        """
        if not self.has_run:
            return "Hierarchical organization test has not been run yet."
        
        report = []
        report.append("=" * 50)
        report.append("HIERARCHICAL ORGANIZATION TEST RESULTS")
        report.append("=" * 50)
        report.append("")
        
        report.append("Hierarchy Score: {:.4f}".format(self.hierarchy_score))
        report.append("p-value: {:.4f}".format(self.p_value))
        report.append("95% Confidence Interval: [{:.4f}, {:.4f}]".format(
            self.confidence_interval[0], self.confidence_interval[1]
        ))
        
        report.append("")
        report.append("Level Correlations:")
        for i, corr in enumerate(self.level_correlations):
            report.append("- Level {}-{}: {:.4f}".format(i, i+1, corr))
        
        report.append("")
        report.append("Interpretation:")
        if self.p_value < self.significance_threshold:
            report.append("- The CMB data shows statistically significant hierarchical organization")
            report.append("  based on the golden ratio (p < {:.2f}).".format(self.significance_threshold))
        else:
            report.append("- The hierarchical organization in the CMB data is not statistically")
            report.append("  significant (p >= {:.2f}).".format(self.significance_threshold))
        
        if self.hierarchy_score > 0.7:
            report.append("- The hierarchy score is high, indicating strong hierarchical organization.")
        elif self.hierarchy_score > 0.4:
            report.append("- The hierarchy score is moderate, indicating some hierarchical organization.")
        else:
            report.append("- The hierarchy score is low, indicating weak hierarchical organization.")
        
        return "\n".join(report)
    
    def visualize(self, output_dir=None):
        """
        Create visualizations of the test results.
        
        Args:
            output_dir (str, optional): Directory to save visualizations. Defaults to None.
        """
        if not self.has_run:
            print("Cannot visualize results: Hierarchical organization test has not been run yet.")
            return
        
        # Create figure with 2 subplots
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot hierarchy levels
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.hierarchy_levels)))
        
        for i, level in enumerate(self.hierarchy_levels):
            axes[0].plot(level, color=colors[i], label='Level {}'.format(i))
        
        axes[0].set_title('Hierarchical Levels')
        axes[0].set_xlabel('Frequency Index')
        axes[0].set_ylabel('Power')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot level correlations
        x = np.arange(len(self.level_correlations))
        bars = axes[1].bar(x, self.level_correlations, color='skyblue')
        
        # Color bars based on correlation value
        for i, bar in enumerate(bars):
            if self.level_correlations[i] < 0:
                bar.set_color('#ff7f0e')  # Orange for negative correlations
        
        axes[1].axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        axes[1].set_title('Correlations Between Adjacent Levels')
        axes[1].set_xlabel('Level Pair')
        axes[1].set_ylabel('Correlation')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(['Level {}-{}'.format(i, i+1) for i in range(len(self.level_correlations))])
        axes[1].grid(True, alpha=0.3)
        
        # Add text with values
        for i, v in enumerate(self.level_correlations):
            axes[1].text(i, v + 0.02 * np.sign(v), "{:.2f}".format(v), ha='center')
        
        # Add overall title with p-value and hierarchy score
        significance_str = "Significant" if self.p_value < self.significance_threshold else "Not Significant"
        fig.suptitle('Hierarchical Organization Results (p={:.4f}, score={:.4f}, {})'.format(
            self.p_value, self.hierarchy_score, significance_str), fontsize=16)
        
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save or show
        if output_dir is not None:
            import os
            output_path = os.path.join(output_dir, 'hierarchical_organization_results.png')
            fig.savefig(output_path, dpi=300)
            plt.close(fig)
            print("Visualization saved to {}".format(output_path))
        else:
            plt.show()
