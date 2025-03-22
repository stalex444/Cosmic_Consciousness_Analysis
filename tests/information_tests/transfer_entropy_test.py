#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Transfer Entropy Test
-------------------
Tests for transfer entropy patterns in CMB data, examining information flow across scales.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from core_framework.base_test import BaseTest
from core_framework.constants import (
    CONSTANTS, COLORS, CONSTANT_NAMES, DEFAULT_MONTE_CARLO_ITERATIONS,
    DEFAULT_BOOTSTRAP_SAMPLES
)
from core_framework.data_handler import (
    generate_surrogate_data, segment_data, fibonacci_sequence
)
from core_framework.statistics import (
    calculate_transfer_entropy, bootstrap_confidence_interval, 
    calculate_phi_optimality, find_best_constant
)
from core_framework.visualization import (
    create_multi_panel_figure, setup_figure, save_figure
)


class TransferEntropyTest(BaseTest):
    """
    Test for transfer entropy patterns in CMB data.
    
    This test examines information flow across scales in the CMB data,
    which would indicate causal relationships and information processing.
    
    Attributes:
        scales (list): Scales to analyze
        n_surrogates (int): Number of surrogate datasets for significance testing
        bootstrap_samples (int): Number of bootstrap samples for confidence intervals
    """
    
    def __init__(self, name="Transfer Entropy Test", seed=None, data=None):
        """
        Initialize the transfer entropy test.
        
        Args:
            name (str, optional): Name of the test. Defaults to "Transfer Entropy Test".
            seed (int, optional): Random seed for reproducibility. Defaults to None.
            data (ndarray, optional): CMB data to analyze. Defaults to None.
        """
        super(TransferEntropyTest, self).__init__(name=name, seed=seed, data=data)
        
        # Generate scales for analysis
        self.scales = self._generate_scales()
        
        # Test parameters
        self.n_surrogates = 100
        self.bootstrap_samples = DEFAULT_BOOTSTRAP_SAMPLES
        self.delay_range = range(1, 6)  # Range of delays to test
    
    def _generate_scales(self, min_scale=8, max_scale=1024, n_scales=12):
        """
        Generate scales for analysis.
        
        Args:
            min_scale (int, optional): Minimum scale. Defaults to 8.
            max_scale (int, optional): Maximum scale. Defaults to 1024.
            n_scales (int, optional): Number of scales. Defaults to 12.
            
        Returns:
            list: List of scales
        """
        # Generate logarithmically spaced scales
        scales = np.logspace(np.log10(min_scale), np.log10(max_scale), n_scales).astype(int)
        
        # Remove duplicates and sort
        scales = sorted(list(set(scales)))
        
        return scales
    
    def run_test(self):
        """
        Run the transfer entropy test.
        
        Returns:
            dict: Test results
        """
        print("Running {}...".format(self.name))
        
        # Ensure data is loaded
        if self.data is None:
            self.load_data()
        
        # Initialize results
        self.results = {
            'scales': self.scales,
            'transfer_entropy_matrix': None,
            'optimal_delays': {},
            'scale_ratios': {},
            'phi_optimality': None,
            'surrogate_results': None,
            'p_value': None,
            'confidence_interval': None
        }
        
        # Calculate transfer entropy matrix
        te_matrix, optimal_delays = self._calculate_transfer_entropy_matrix()
        self.results['transfer_entropy_matrix'] = te_matrix
        self.results['optimal_delays'] = optimal_delays
        
        # Calculate scale ratios
        scale_ratios = self._calculate_scale_ratios()
        self.results['scale_ratios'] = scale_ratios
        
        # Calculate phi optimality
        phi_optimality = calculate_phi_optimality(scale_ratios, CONSTANTS)
        self.results['phi_optimality'] = phi_optimality
        
        # Find best constant
        best_constant, best_value = find_best_constant(phi_optimality)
        self.results['best_constant'] = best_constant
        self.results['best_value'] = best_value
        
        # Test significance
        p_value, surrogate_results = self._test_significance(scale_ratios)
        self.results['p_value'] = p_value
        self.results['surrogate_results'] = surrogate_results
        
        # Calculate confidence interval
        confidence_interval = self._calculate_confidence_interval(scale_ratios)
        self.results['confidence_interval'] = confidence_interval
        
        print("\n")
        return self.results
    
    def _calculate_transfer_entropy_matrix(self):
        """
        Calculate transfer entropy matrix across scales.
        
        Returns:
            tuple: Transfer entropy matrix and optimal delays
        """
        n_scales = len(self.scales)
        te_matrix = np.zeros((n_scales, n_scales))
        optimal_delays = {}
        
        for i, scale_i in enumerate(self.scales):
            segments_i = segment_data(self.data, scale_i, overlap=0.5)
            avg_segment_i = np.mean(segments_i, axis=0)
            
            for j, scale_j in enumerate(self.scales):
                if i == j:  # Skip self-comparison
                    continue
                
                segments_j = segment_data(self.data, scale_j, overlap=0.5)
                avg_segment_j = np.mean(segments_j, axis=0)
                
                # Find optimal delay
                max_te = 0
                optimal_delay = 1
                
                for delay in self.delay_range:
                    te = calculate_transfer_entropy(avg_segment_i, avg_segment_j, delay=delay)
                    
                    if te > max_te:
                        max_te = te
                        optimal_delay = delay
                
                # Store transfer entropy and optimal delay
                te_matrix[i, j] = max_te
                optimal_delays[(scale_i, scale_j)] = optimal_delay
        
        return te_matrix, optimal_delays
    
    def _calculate_scale_ratios(self):
        """
        Calculate ratios between scales with high transfer entropy.
        
        Returns:
            dict: Scale ratios and their transfer entropy values
        """
        te_matrix = self.results['transfer_entropy_matrix']
        
        # Find pairs with high transfer entropy
        high_te_pairs = []
        
        for i, scale_i in enumerate(self.scales):
            for j, scale_j in enumerate(self.scales):
                if i == j:  # Skip self-comparison
                    continue
                
                # Add pair if transfer entropy is above threshold
                # Use dynamic threshold based on matrix statistics
                threshold = np.mean(te_matrix) + np.std(te_matrix)
                
                if te_matrix[i, j] > threshold:
                    high_te_pairs.append((scale_i, scale_j, te_matrix[i, j]))
        
        # Calculate ratios
        scale_ratios = {}
        
        for scale_i, scale_j, te_value in high_te_pairs:
            # Calculate ratio (always larger / smaller)
            if scale_i > scale_j:
                ratio = float(scale_i) / scale_j
            else:
                ratio = float(scale_j) / scale_i
            
            # Round ratio to 4 decimal places
            ratio = round(ratio, 4)
            
            # Store or update ratio
            if ratio in scale_ratios:
                scale_ratios[ratio] += te_value
            else:
                scale_ratios[ratio] = te_value
        
        return scale_ratios
    
    def _test_significance(self, scale_ratios):
        """
        Test statistical significance of scale ratios.
        
        Args:
            scale_ratios (dict): Scale ratios and their transfer entropy values
            
        Returns:
            tuple: p-value and surrogate results
        """
        # Generate surrogate data
        surrogates = generate_surrogate_data(self.data, n_surrogates=self.n_surrogates, seed=self.seed)
        
        # Calculate scale ratios for each surrogate
        surrogate_results = []
        
        for i in range(self.n_surrogates):
            # Create surrogate test
            surrogate_test = TransferEntropyTest(
                name="Surrogate Transfer Entropy Test {}".format(i),
                seed=self.seed + i if self.seed is not None else None,
                data=surrogates[i]
            )
            
            # Set the same scales and parameters
            surrogate_test.scales = self.scales
            surrogate_test.delay_range = self.delay_range
            
            # Calculate transfer entropy matrix
            te_matrix, _ = surrogate_test._calculate_transfer_entropy_matrix()
            
            # Store matrix in surrogate test results
            surrogate_test.results = {
                'transfer_entropy_matrix': te_matrix
            }
            
            # Calculate scale ratios
            surrogate_ratios = surrogate_test._calculate_scale_ratios()
            
            # Calculate phi optimality
            surrogate_phi_optimality = calculate_phi_optimality(surrogate_ratios, CONSTANTS)
            
            # Find best constant
            surrogate_best_constant, surrogate_best_value = find_best_constant(surrogate_phi_optimality)
            
            # Store results
            surrogate_results.append({
                'scale_ratios': surrogate_ratios,
                'phi_optimality': surrogate_phi_optimality,
                'best_constant': surrogate_best_constant,
                'best_value': surrogate_best_value
            })
        
        # Calculate p-value based on phi optimality for golden ratio
        real_phi_optimality = self.results['phi_optimality']['phi']
        surrogate_phi_optimality = [s['phi_optimality']['phi'] for s in surrogate_results]
        
        p_value = np.mean(np.array(surrogate_phi_optimality) >= real_phi_optimality)
        
        return p_value, surrogate_results
    
    def _calculate_confidence_interval(self, scale_ratios):
        """
        Calculate bootstrap confidence interval for phi optimality.
        
        Args:
            scale_ratios (dict): Scale ratios and their transfer entropy values
            
        Returns:
            tuple: Lower and upper bounds of confidence interval
        """
        # Generate bootstrap samples
        bootstrap_values = []
        
        for i in range(self.bootstrap_samples):
            # Resample data with replacement
            bootstrap_data = self._generate_bootstrap_sample()
            
            # Create bootstrap test
            bootstrap_test = TransferEntropyTest(
                name="Bootstrap Transfer Entropy Test {}".format(i),
                seed=self.seed + i if self.seed is not None else None,
                data=bootstrap_data
            )
            
            # Set the same scales and parameters
            bootstrap_test.scales = self.scales
            bootstrap_test.delay_range = self.delay_range
            
            # Calculate transfer entropy matrix
            te_matrix, _ = bootstrap_test._calculate_transfer_entropy_matrix()
            
            # Store matrix in bootstrap test results
            bootstrap_test.results = {
                'transfer_entropy_matrix': te_matrix
            }
            
            # Calculate scale ratios
            bootstrap_ratios = bootstrap_test._calculate_scale_ratios()
            
            # Calculate phi optimality
            bootstrap_phi_optimality = calculate_phi_optimality(bootstrap_ratios, CONSTANTS)
            
            # Store phi optimality for golden ratio
            bootstrap_values.append(bootstrap_phi_optimality['phi'])
        
        # Calculate confidence interval
        confidence_interval = bootstrap_confidence_interval(bootstrap_values)
        
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
        report.append("TRANSFER ENTROPY TEST REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Execution time
        if hasattr(self, 'execution_time') and self.execution_time is not None:
            report.append("Test completed in {:.2f} seconds.".format(self.execution_time))
            report.append("")
        
        # Scale ratios
        report.append("TOP SCALE RATIOS WITH HIGH TRANSFER ENTROPY")
        report.append("-" * 50)
        
        # Sort ratios by transfer entropy value
        sorted_ratios = sorted(self.results['scale_ratios'].items(), key=lambda x: x[1], reverse=True)
        
        # Display top 10 ratios
        for i, (ratio, te_value) in enumerate(sorted_ratios[:10]):
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
            
            report.append("{}. Ratio {:.4f} (TE = {:.6f})".format(i + 1, ratio, te_value))
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
        report.append("The transfer entropy test examines information flow across scales in the")
        report.append("cosmic microwave background, which would indicate causal relationships and")
        report.append("information processing.")
        report.append("")
        
        if self.results['p_value'] < 0.05:
            report.append("The analysis reveals STATISTICALLY SIGNIFICANT transfer entropy patterns")
            report.append("in the CMB data, suggesting that information flows between different scales")
            report.append("in a non-random manner.")
        else:
            report.append("The analysis does not detect statistically significant transfer entropy")
            report.append("patterns in the CMB data at the p < 0.05 level.")
        
        report.append("")
        
        if self.results['best_constant'] == 'phi':
            report.append("Notably, the golden ratio (phi) shows the strongest optimization in the")
            report.append("scale ratios with high transfer entropy, which aligns with the hypothesis")
            report.append("that phi-based organization may be a signature of consciousness-like")
            report.append("processes in cosmic structure.")
        else:
            report.append("The constant {} shows the strongest optimization in the scale".format(self.results['best_constant']))
            report.append("ratios with high transfer entropy, which suggests that this mathematical")
            report.append("relationship may play a significant role in the information processing")
            report.append("of cosmic structure.")
        
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
        
        # Plot transfer entropy matrix
        im = axes[0, 0].imshow(self.results['transfer_entropy_matrix'], cmap='viridis', aspect='auto')
        axes[0, 0].set_title("Transfer Entropy Matrix")
        axes[0, 0].set_xlabel("Scale Index")
        axes[0, 0].set_ylabel("Scale Index")
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=axes[0, 0])
        cbar.set_label("Transfer Entropy")
        
        # Add scale labels
        n_scales = len(self.scales)
        step = max(1, n_scales // 10)  # Show at most 10 labels
        
        axes[0, 0].set_xticks(range(0, n_scales, step))
        axes[0, 0].set_yticks(range(0, n_scales, step))
        
        axes[0, 0].set_xticklabels([str(self.scales[i]) for i in range(0, n_scales, step)])
        axes[0, 0].set_yticklabels([str(self.scales[i]) for i in range(0, n_scales, step)])
        
        # Plot scale ratios
        sorted_ratios = sorted(self.results['scale_ratios'].items(), key=lambda x: x[1], reverse=True)
        top_ratios = sorted_ratios[:10]
        
        ratios = [r[0] for r in top_ratios]
        te_values = [r[1] for r in top_ratios]
        
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
        
        axes[0, 1].bar(range(len(top_ratios)), te_values, color=colors)
        axes[0, 1].set_title("Top Scale Ratios with High Transfer Entropy")
        axes[0, 1].set_xlabel("Rank")
        axes[0, 1].set_ylabel("Transfer Entropy")
        
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
            save_figure(fig, "transfer_entropy_results.png")
        
        # Show figure if requested
        if show:
            plt.show()
        
        return fig


def main():
    """
    Run the transfer entropy test.
    """
    # Create and run test
    test = TransferEntropyTest(seed=42)
    test.run()
    
    # Generate report and visualizations
    test.generate_report()
    test.visualize_results()


if __name__ == "__main__":
    main()
