#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Golden Ratio Specific Coherence Test
----------------------------------
Tests coherence specifically in regions of the CMB spectrum related by the golden ratio.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from core_framework.base_test import BaseTest
from core_framework.statistics import calculate_significance, bootstrap_confidence_interval
from core_framework.data_handler import generate_surrogate_data, segment_data
from core_framework.constants import DEFAULT_SEED, PHI


class GRSpecificCoherenceTest(BaseTest):
    """Test for analyzing coherence specifically in golden ratio related regions."""
    
    def __init__(self, data, seed=None):
        """
        Initialize the GR-specific coherence test.
        
        Args:
            data (ndarray): CMB data to analyze
            seed (int, optional): Random seed for reproducibility. Defaults to None.
        """
        super(GRSpecificCoherenceTest, self).__init__(data=data, seed=seed)
        self.gr_coherence = None
        self.random_coherence = None
        self.p_value = None
        self.confidence_interval = None
        self.phi_optimality = None
        self.gr_pairs = None
        self.significance_threshold = 0.05
    
    def run(self):
        """Run the GR-specific coherence test."""
        # Identify golden ratio related pairs
        self.gr_pairs = self._identify_gr_pairs(self.data)
        
        # Calculate coherence for GR-related pairs
        self.gr_coherence = self._calculate_gr_coherence(self.data, self.gr_pairs)
        
        # Calculate coherence for random pairs (control)
        self.random_coherence = self._calculate_random_coherence(self.data, len(self.gr_pairs))
        
        # Calculate phi-optimality (normalized difference)
        self.phi_optimality = (self.gr_coherence - self.random_coherence) / (self.gr_coherence + self.random_coherence + 1e-10)
        
        # Generate surrogate data for significance testing
        n_surrogates = 1000
        surrogate_data = generate_surrogate_data(
            self.data, 
            method='phase_randomization',
            n_surrogates=n_surrogates,
            seed=self.seed
        )
        
        # Calculate phi-optimality for surrogate data
        surrogate_phi_optimality = np.zeros(n_surrogates)
        for i in range(n_surrogates):
            gr_coh = self._calculate_gr_coherence(surrogate_data[i], self.gr_pairs)
            rand_coh = self._calculate_random_coherence(surrogate_data[i], len(self.gr_pairs))
            surrogate_phi_optimality[i] = (gr_coh - rand_coh) / (gr_coh + rand_coh + 1e-10)
        
        # Calculate significance
        self.p_value = calculate_significance(self.phi_optimality, surrogate_phi_optimality)
        
        # Calculate confidence interval
        self.confidence_interval = bootstrap_confidence_interval(
            self.data,
            statistic=lambda x: self._calculate_phi_optimality(x),
            n_bootstrap=1000,
            confidence=0.95,
            seed=self.seed
        )
        
        # Set has_run flag
        self.has_run = True
    
    def _identify_gr_pairs(self, data, n_segments=16):
        """
        Identify pairs of segments related by the golden ratio.
        
        Args:
            data (ndarray): Data to analyze
            n_segments (int, optional): Number of segments. Defaults to 16.
            
        Returns:
            list: List of (i, j) pairs where i/j ≈ φ or j/i ≈ φ
        """
        # Segment the data
        segment_size = len(data) // n_segments
        segments = segment_data(data, scale=segment_size, overlap=0)
        
        # Find pairs related by golden ratio
        gr_pairs = []
        
        for i in range(1, n_segments):
            for j in range(i+1, n_segments):
                ratio_ij = float(i) / j
                ratio_ji = float(j) / i
                
                # Check if either ratio is close to phi
                if abs(ratio_ij - PHI) < 0.1 or abs(ratio_ji - PHI) < 0.1:
                    gr_pairs.append((i, j))
        
        return gr_pairs
    
    def _calculate_gr_coherence(self, data, gr_pairs):
        """
        Calculate coherence for golden ratio related pairs.
        
        Args:
            data (ndarray): Data to analyze
            gr_pairs (list): List of golden ratio related pairs
            
        Returns:
            float: Mean coherence for golden ratio related pairs
        """
        if not gr_pairs:
            return 0.0
        
        # Segment the data
        n_segments = max([max(i, j) for i, j in gr_pairs]) + 1
        segment_size = len(data) // n_segments
        segments = segment_data(data, scale=segment_size, overlap=0)
        
        # Calculate coherence for each GR pair
        coherence_values = []
        
        for i, j in gr_pairs:
            f, Cxy = signal.coherence(segments[i], segments[j], fs=1.0, nperseg=segment_size//4)
            coherence_values.append(np.mean(Cxy))
        
        # Return mean coherence
        return np.mean(coherence_values)
    
    def _calculate_random_coherence(self, data, n_pairs, n_segments=16):
        """
        Calculate coherence for random pairs (control).
        
        Args:
            data (ndarray): Data to analyze
            n_pairs (int): Number of random pairs to sample
            n_segments (int, optional): Number of segments. Defaults to 16.
            
        Returns:
            float: Mean coherence for random pairs
        """
        if n_pairs <= 0:
            return 0.0
        
        # Set random seed
        if self.seed is not None:
            np.random.seed(self.seed)
        
        # Segment the data
        segment_size = len(data) // n_segments
        segments = segment_data(data, scale=segment_size, overlap=0)
        
        # Generate random pairs (excluding those related by golden ratio)
        all_pairs = [(i, j) for i in range(n_segments) for j in range(i+1, n_segments)]
        gr_pairs_set = set(self.gr_pairs) if self.gr_pairs else set()
        non_gr_pairs = [pair for pair in all_pairs if pair not in gr_pairs_set]
        
        # Sample random pairs
        if len(non_gr_pairs) > n_pairs:
            random_pairs = np.random.choice(len(non_gr_pairs), size=n_pairs, replace=False)
            random_pairs = [non_gr_pairs[i] for i in random_pairs]
        else:
            random_pairs = non_gr_pairs
        
        # Calculate coherence for each random pair
        coherence_values = []
        
        for i, j in random_pairs:
            f, Cxy = signal.coherence(segments[i], segments[j], fs=1.0, nperseg=segment_size//4)
            coherence_values.append(np.mean(Cxy))
        
        # Return mean coherence
        return np.mean(coherence_values) if coherence_values else 0.0
    
    def _calculate_phi_optimality(self, data):
        """
        Calculate phi-optimality for the given data.
        
        Args:
            data (ndarray): Data to analyze
            
        Returns:
            float: Phi-optimality
        """
        # Identify golden ratio related pairs
        gr_pairs = self._identify_gr_pairs(data)
        
        # Calculate coherence for GR-related pairs
        gr_coherence = self._calculate_gr_coherence(data, gr_pairs)
        
        # Calculate coherence for random pairs (control)
        random_coherence = self._calculate_random_coherence(data, len(gr_pairs))
        
        # Calculate phi-optimality (normalized difference)
        phi_optimality = (gr_coherence - random_coherence) / (gr_coherence + random_coherence + 1e-10)
        
        return phi_optimality
    
    def generate_report(self):
        """
        Generate a report of the test results.
        
        Returns:
            str: Report text
        """
        if not self.has_run:
            return "GR-specific coherence test has not been run yet."
        
        report = []
        report.append("=" * 50)
        report.append("GOLDEN RATIO SPECIFIC COHERENCE TEST RESULTS")
        report.append("=" * 50)
        report.append("")
        
        report.append("Number of GR-related pairs: {}".format(len(self.gr_pairs)))
        report.append("Mean coherence for GR-related pairs: {:.4f}".format(self.gr_coherence))
        report.append("Mean coherence for random pairs: {:.4f}".format(self.random_coherence))
        report.append("Phi-optimality: {:.4f}".format(self.phi_optimality))
        report.append("p-value: {:.4f}".format(self.p_value))
        report.append("95% Confidence Interval: [{:.4f}, {:.4f}]".format(
            self.confidence_interval[0], self.confidence_interval[1]
        ))
        
        report.append("")
        report.append("Interpretation:")
        if self.p_value < self.significance_threshold:
            report.append("- The coherence in golden ratio related regions is statistically")
            report.append("  significant compared to random regions (p < {:.2f}).".format(self.significance_threshold))
        else:
            report.append("- The coherence in golden ratio related regions is not statistically")
            report.append("  significant compared to random regions (p >= {:.2f}).".format(self.significance_threshold))
        
        if self.phi_optimality > 0.2:
            report.append("- The phi-optimality is high, indicating strong preferential coherence")
            report.append("  in golden ratio related regions.")
        elif self.phi_optimality > 0.05:
            report.append("- The phi-optimality is moderate, indicating some preferential coherence")
            report.append("  in golden ratio related regions.")
        else:
            report.append("- The phi-optimality is low, indicating weak or no preferential coherence")
            report.append("  in golden ratio related regions.")
        
        return "\n".join(report)
    
    def visualize(self, output_dir=None):
        """
        Create visualizations of the test results.
        
        Args:
            output_dir (str, optional): Directory to save visualizations. Defaults to None.
        """
        if not self.has_run:
            print("Cannot visualize results: GR-specific coherence test has not been run yet.")
            return
        
        # Create figure with 2 subplots
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot coherence comparison
        labels = ['GR-Related\nPairs', 'Random\nPairs']
        values = [self.gr_coherence, self.random_coherence]
        
        axes[0].bar(labels, values, color=['#1f77b4', '#ff7f0e'])
        axes[0].set_title('Coherence Comparison')
        axes[0].set_ylabel('Mean Coherence')
        axes[0].grid(True, alpha=0.3)
        
        # Add text with values
        for i, v in enumerate(values):
            axes[0].text(i, v + 0.02, "{:.4f}".format(v), ha='center')
        
        # Plot GR pairs on a grid
        n_segments = max([max(i, j) for i, j in self.gr_pairs]) + 1 if self.gr_pairs else 16
        grid = np.zeros((n_segments, n_segments))
        
        for i, j in self.gr_pairs:
            grid[i, j] = 1
            grid[j, i] = 1
        
        im = axes[1].imshow(grid, cmap='Blues', origin='lower')
        axes[1].set_title('Golden Ratio Related Pairs')
        axes[1].set_xlabel('Segment Index')
        axes[1].set_ylabel('Segment Index')
        
        # Add overall title with p-value and phi-optimality
        significance_str = "Significant" if self.p_value < self.significance_threshold else "Not Significant"
        fig.suptitle('GR-Specific Coherence Results (p={:.4f}, φ-opt={:.4f}, {})'.format(
            self.p_value, self.phi_optimality, significance_str), fontsize=16)
        
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save or show
        if output_dir is not None:
            import os
            output_path = os.path.join(output_dir, 'gr_specific_coherence_results.png')
            fig.savefig(output_path, dpi=300)
            plt.close(fig)
            print("Visualization saved to {}".format(output_path))
        else:
            plt.show()
