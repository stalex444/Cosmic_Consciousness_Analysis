#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Information Integration Test
-------------------------
Measures mutual information between adjacent regions of the CMB power spectrum.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.feature_selection import mutual_info_regression

from core_framework.base_test import BaseTest
from core_framework.statistics import calculate_significance, bootstrap_confidence_interval
from core_framework.data_handler import generate_surrogate_data, segment_data
from core_framework.constants import DEFAULT_SEED, PHI


class InformationIntegrationTest(BaseTest):
    """Test for measuring information integration in CMB data."""
    
    def __init__(self, data, seed=None):
        """
        Initialize the information integration test.
        
        Args:
            data (ndarray): CMB data to analyze
            seed (int, optional): Random seed for reproducibility. Defaults to None.
        """
        super(InformationIntegrationTest, self).__init__(data=data, seed=seed)
        self.integration_score = None
        self.p_value = None
        self.confidence_interval = None
        self.mutual_info_matrix = None
        self.significance_threshold = 0.05
        self.n_segments = 10  # Number of segments to divide the data into
    
    def run(self):
        """Run the information integration test."""
        # Segment the data
        segments = segment_data(self.data, scale=len(self.data) // self.n_segments, overlap=0)
        
        # Calculate mutual information matrix
        self.mutual_info_matrix = self._calculate_mutual_info_matrix(segments)
        
        # Calculate integration score
        self.integration_score = self._calculate_integration_score(self.mutual_info_matrix)
        
        # Generate surrogate data for significance testing
        n_surrogates = 1000
        surrogate_data = generate_surrogate_data(
            self.data, 
            method='phase_randomization',
            n_surrogates=n_surrogates,
            seed=self.seed
        )
        
        # Calculate integration scores for surrogate data
        surrogate_scores = np.zeros(n_surrogates)
        for i in range(n_surrogates):
            surrogate_segments = segment_data(surrogate_data[i], scale=len(surrogate_data[i]) // self.n_segments, overlap=0)
            surrogate_mi_matrix = self._calculate_mutual_info_matrix(surrogate_segments)
            surrogate_scores[i] = self._calculate_integration_score(surrogate_mi_matrix)
        
        # Calculate significance
        self.p_value = calculate_significance(self.integration_score, surrogate_scores)
        
        # Calculate confidence interval
        self.confidence_interval = bootstrap_confidence_interval(
            self.data,
            statistic=lambda x: self._calculate_integration_score_from_data(x),
            n_bootstrap=1000,
            confidence=0.95,
            seed=self.seed
        )
        
        # Set has_run flag
        self.has_run = True
    
    def _calculate_mutual_info_matrix(self, segments):
        """
        Calculate mutual information matrix between segments.
        
        Args:
            segments (list): List of data segments
            
        Returns:
            ndarray: Mutual information matrix
        """
        n_segments = len(segments)
        mi_matrix = np.zeros((n_segments, n_segments))
        
        for i in range(n_segments):
            for j in range(i, n_segments):
                # Calculate mutual information
                # Reshape for sklearn's mutual_info_regression
                X = segments[i].reshape(-1, 1)
                y = segments[j]
                
                # Calculate mutual information
                mi = mutual_info_regression(X, y)[0]
                
                # Store in matrix (symmetric)
                mi_matrix[i, j] = mi
                mi_matrix[j, i] = mi
        
        return mi_matrix
    
    def _calculate_integration_score(self, mi_matrix):
        """
        Calculate integration score from mutual information matrix.
        
        Args:
            mi_matrix (ndarray): Mutual information matrix
            
        Returns:
            float: Integration score
        """
        # Calculate integration score as the sum of off-diagonal elements
        # normalized by the number of pairs
        n = mi_matrix.shape[0]
        mask = np.ones((n, n), dtype=bool)
        np.fill_diagonal(mask, False)
        
        # Sum off-diagonal elements and normalize
        integration_score = np.sum(mi_matrix[mask]) / (n * (n - 1))
        
        return integration_score
    
    def _calculate_integration_score_from_data(self, data):
        """
        Calculate integration score directly from data.
        
        Args:
            data (ndarray): Data to analyze
            
        Returns:
            float: Integration score
        """
        # Segment the data
        segments = segment_data(data, scale=len(data) // self.n_segments, overlap=0)
        
        # Calculate mutual information matrix
        mi_matrix = self._calculate_mutual_info_matrix(segments)
        
        # Calculate integration score
        integration_score = self._calculate_integration_score(mi_matrix)
        
        return integration_score
    
    def generate_report(self):
        """
        Generate a report of the test results.
        
        Returns:
            str: Report text
        """
        if not self.has_run:
            return "Information integration test has not been run yet."
        
        report = []
        report.append("=" * 50)
        report.append("INFORMATION INTEGRATION TEST RESULTS")
        report.append("=" * 50)
        report.append("")
        
        report.append("Integration Score: {:.4f}".format(self.integration_score))
        report.append("p-value: {:.4f}".format(self.p_value))
        report.append("95% Confidence Interval: [{:.4f}, {:.4f}]".format(
            self.confidence_interval[0], self.confidence_interval[1]
        ))
        
        report.append("")
        report.append("Interpretation:")
        if self.p_value < self.significance_threshold:
            report.append("- The CMB data shows statistically significant information integration")
            report.append("  (p < {:.2f}).".format(self.significance_threshold))
        else:
            report.append("- The information integration in the CMB data is not statistically")
            report.append("  significant (p >= {:.2f}).".format(self.significance_threshold))
        
        if self.integration_score > 0.5:
            report.append("- The integration score is high, indicating strong information sharing")
            report.append("  between different regions of the CMB spectrum.")
        elif self.integration_score > 0.2:
            report.append("- The integration score is moderate, indicating some information sharing")
            report.append("  between different regions of the CMB spectrum.")
        else:
            report.append("- The integration score is low, indicating weak information sharing")
            report.append("  between different regions of the CMB spectrum.")
        
        return "\n".join(report)
    
    def visualize(self, output_dir=None):
        """
        Create visualizations of the test results.
        
        Args:
            output_dir (str, optional): Directory to save visualizations. Defaults to None.
        """
        if not self.has_run:
            print("Cannot visualize results: Information integration test has not been run yet.")
            return
        
        # Create figure with 2 subplots
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot mutual information matrix as heatmap
        im = axes[0].imshow(self.mutual_info_matrix, cmap='viridis', origin='lower')
        axes[0].set_title('Mutual Information Matrix')
        axes[0].set_xlabel('Segment Index')
        axes[0].set_ylabel('Segment Index')
        fig.colorbar(im, ax=axes[0], label='Mutual Information')
        
        # Plot mutual information distribution
        n = self.mutual_info_matrix.shape[0]
        mask = np.ones((n, n), dtype=bool)
        np.fill_diagonal(mask, False)
        mi_values = self.mutual_info_matrix[mask]
        
        axes[1].hist(mi_values, bins=20, alpha=0.7, color='blue')
        axes[1].axvline(self.integration_score, color='red', linestyle='--', 
                        label='Mean: {:.4f}'.format(self.integration_score))
        axes[1].set_title('Mutual Information Distribution')
        axes[1].set_xlabel('Mutual Information')
        axes[1].set_ylabel('Frequency')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Add overall title with p-value
        significance_str = "Significant" if self.p_value < self.significance_threshold else "Not Significant"
        fig.suptitle('Information Integration Results (p={:.4f}, {})'.format(
            self.p_value, significance_str), fontsize=16)
        
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save or show
        if output_dir is not None:
            import os
            output_path = os.path.join(output_dir, 'information_integration_results.png')
            fig.savefig(output_path, dpi=300)
            plt.close(fig)
            print("Visualization saved to {}".format(output_path))
        else:
            plt.show()
