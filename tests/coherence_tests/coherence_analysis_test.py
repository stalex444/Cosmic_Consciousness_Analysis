#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Coherence Analysis Test
---------------------
Tests whether the CMB spectrum exhibits more coherence than would be expected by random chance.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from core_framework.base_test import BaseTest
from core_framework.statistics import calculate_significance, bootstrap_confidence_interval
from core_framework.data_handler import generate_surrogate_data, segment_data
from core_framework.constants import DEFAULT_SEED


class CoherenceAnalysisTest(BaseTest):
    """Test for analyzing coherence patterns in CMB data."""
    
    def __init__(self, data, seed=None):
        """
        Initialize the coherence analysis test.
        
        Args:
            data (ndarray): CMB data to analyze
            seed (int, optional): Random seed for reproducibility. Defaults to None.
        """
        super(CoherenceAnalysisTest, self).__init__(data=data, seed=seed)
        self.coherence_score = None
        self.p_value = None
        self.confidence_interval = None
        self.coherence_matrix = None
        self.significance_threshold = 0.05
    
    def run(self):
        """Run the coherence analysis test."""
        # Calculate coherence matrix and score
        self.coherence_matrix = self._calculate_coherence_matrix(self.data)
        self.coherence_score = self._calculate_coherence_score(self.coherence_matrix)
        
        # Generate surrogate data for significance testing
        n_surrogates = 1000
        surrogate_data = generate_surrogate_data(
            self.data, 
            method='phase_randomization',
            n_surrogates=n_surrogates,
            seed=self.seed
        )
        
        # Calculate coherence scores for surrogate data
        surrogate_scores = np.zeros(n_surrogates)
        for i in range(n_surrogates):
            surrogate_matrix = self._calculate_coherence_matrix(surrogate_data[i])
            surrogate_scores[i] = self._calculate_coherence_score(surrogate_matrix)
        
        # Calculate significance
        self.p_value = calculate_significance(self.coherence_score, surrogate_scores)
        
        # Calculate confidence interval
        self.confidence_interval = bootstrap_confidence_interval(
            self.data,
            statistic=lambda x: self._calculate_coherence_score(self._calculate_coherence_matrix(x)),
            n_bootstrap=1000,
            confidence=0.95,
            seed=self.seed
        )
        
        # Set has_run flag
        self.has_run = True
    
    def _calculate_coherence_matrix(self, data, n_segments=8, segment_size=None, overlap=0.5):
        """
        Calculate coherence matrix for the given data.
        
        Args:
            data (ndarray): Data to analyze
            n_segments (int, optional): Number of segments. Defaults to 8.
            segment_size (int, optional): Size of each segment. Defaults to None (auto-calculated).
            overlap (float, optional): Overlap fraction. Defaults to 0.5.
            
        Returns:
            ndarray: Coherence matrix
        """
        # Determine segment size if not provided
        if segment_size is None:
            segment_size = len(data) // (n_segments * (1 - overlap) + overlap)
        
        # Segment the data
        segments = segment_data(data, scale=segment_size, overlap=overlap)
        n_segments = len(segments)
        
        # Calculate coherence matrix
        coherence_matrix = np.zeros((n_segments, n_segments))
        
        for i in range(n_segments):
            for j in range(i, n_segments):
                # Calculate magnitude squared coherence
                f, Cxy = signal.coherence(segments[i], segments[j], fs=1.0, nperseg=segment_size//4)
                
                # Use mean coherence as the measure
                coherence_matrix[i, j] = np.mean(Cxy)
                coherence_matrix[j, i] = coherence_matrix[i, j]  # Symmetry
        
        return coherence_matrix
    
    def _calculate_coherence_score(self, coherence_matrix):
        """
        Calculate coherence score from coherence matrix.
        
        Args:
            coherence_matrix (ndarray): Coherence matrix
            
        Returns:
            float: Coherence score
        """
        # Calculate mean coherence excluding diagonal
        n = coherence_matrix.shape[0]
        mask = np.ones((n, n), dtype=bool)
        np.fill_diagonal(mask, False)
        
        coherence_score = np.mean(coherence_matrix[mask])
        
        return coherence_score
    
    def generate_report(self):
        """
        Generate a report of the test results.
        
        Returns:
            str: Report text
        """
        if not self.has_run:
            return "Coherence analysis test has not been run yet."
        
        report = []
        report.append("=" * 50)
        report.append("COHERENCE ANALYSIS TEST RESULTS")
        report.append("=" * 50)
        report.append("")
        
        report.append("Coherence Score: {:.4f}".format(self.coherence_score))
        report.append("p-value: {:.4f}".format(self.p_value))
        report.append("95% Confidence Interval: [{:.4f}, {:.4f}]".format(
            self.confidence_interval[0], self.confidence_interval[1]
        ))
        
        report.append("")
        report.append("Interpretation:")
        if self.p_value < self.significance_threshold:
            report.append("- The CMB data shows statistically significant coherence patterns")
            report.append("  (p < {:.2f}).".format(self.significance_threshold))
        else:
            report.append("- The coherence patterns in the CMB data are not statistically")
            report.append("  significant (p >= {:.2f}).".format(self.significance_threshold))
        
        if self.coherence_score > 0.7:
            report.append("- The coherence score is high, indicating strong coherence across segments.")
        elif self.coherence_score > 0.4:
            report.append("- The coherence score is moderate, indicating some coherence across segments.")
        else:
            report.append("- The coherence score is low, indicating weak coherence across segments.")
        
        return "\n".join(report)
    
    def visualize(self, output_dir=None):
        """
        Create visualizations of the test results.
        
        Args:
            output_dir (str, optional): Directory to save visualizations. Defaults to None.
        """
        if not self.has_run:
            print("Cannot visualize results: Coherence analysis test has not been run yet.")
            return
        
        # Create figure with 2 subplots
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot coherence matrix as heatmap
        im = axes[0].imshow(self.coherence_matrix, cmap='viridis', origin='lower', vmin=0, vmax=1)
        axes[0].set_title('Coherence Matrix')
        axes[0].set_xlabel('Segment Index')
        axes[0].set_ylabel('Segment Index')
        fig.colorbar(im, ax=axes[0], label='Coherence')
        
        # Plot coherence distribution
        n = self.coherence_matrix.shape[0]
        mask = np.ones((n, n), dtype=bool)
        np.fill_diagonal(mask, False)
        coherence_values = self.coherence_matrix[mask]
        
        axes[1].hist(coherence_values, bins=20, alpha=0.7, color='blue')
        axes[1].axvline(self.coherence_score, color='red', linestyle='--', 
                        label='Mean: {:.4f}'.format(self.coherence_score))
        axes[1].set_title('Coherence Distribution')
        axes[1].set_xlabel('Coherence Value')
        axes[1].set_ylabel('Frequency')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Add overall title with p-value
        significance_str = "Significant" if self.p_value < self.significance_threshold else "Not Significant"
        fig.suptitle('Coherence Analysis Results (p={:.4f}, {})'.format(
            self.p_value, significance_str), fontsize=16)
        
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save or show
        if output_dir is not None:
            import os
            output_path = os.path.join(output_dir, 'coherence_analysis_results.png')
            fig.savefig(output_path, dpi=300)
            plt.close(fig)
            print("Visualization saved to {}".format(output_path))
        else:
            plt.show()
