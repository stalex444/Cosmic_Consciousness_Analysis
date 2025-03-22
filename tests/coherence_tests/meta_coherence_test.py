#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Meta-Coherence Test
------------------
Tests for meta-coherence patterns in CMB data, examining if coherence itself
shows coherent organization across scales.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, stats

from core_framework.base_test import BaseTest
from core_framework.constants import (
    CONSTANTS, COLORS, CONSTANT_NAMES, DEFAULT_MONTE_CARLO_ITERATIONS,
    DEFAULT_BOOTSTRAP_SAMPLES
)
from core_framework.data_handler import (
    generate_surrogate_data, segment_data, fibonacci_sequence
)
from core_framework.statistics import (
    bootstrap_confidence_interval, calculate_phi_optimality, find_best_constant
)
from core_framework.visualization import (
    create_multi_panel_figure, setup_figure, save_figure, plot_heatmap
)


class MetaCoherenceTest(BaseTest):
    """
    Test for meta-coherence patterns in CMB data.
    
    This test examines if coherence itself shows coherent organization across scales,
    which would indicate a higher-order organizing principle.
    
    Attributes:
        scales (list): Scales to analyze
        n_surrogates (int): Number of surrogate datasets for significance testing
        bootstrap_samples (int): Number of bootstrap samples for confidence intervals
    """
    
    def __init__(self, name="Meta-Coherence Test", seed=None, data=None):
        """
        Initialize the meta-coherence test.
        
        Args:
            name (str, optional): Name of the test. Defaults to "Meta-Coherence Test".
            seed (int, optional): Random seed for reproducibility. Defaults to None.
            data (ndarray, optional): CMB data to analyze. Defaults to None.
        """
        super(MetaCoherenceTest, self).__init__(name=name, seed=seed, data=data)
        
        # Generate scales for analysis
        self.scales = self._generate_scales()
        
        # Test parameters
        self.n_surrogates = 100
        self.bootstrap_samples = DEFAULT_BOOTSTRAP_SAMPLES
    
    def _generate_scales(self, min_scale=8, max_scale=1024, n_scales=15):
        """
        Generate scales for analysis.
        
        Args:
            min_scale (int, optional): Minimum scale. Defaults to 8.
            max_scale (int, optional): Maximum scale. Defaults to 1024.
            n_scales (int, optional): Number of scales. Defaults to 15.
            
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
        Run the meta-coherence test.
        
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
            'coherence_matrix': None,
            'meta_coherence': None,
            'surrogate_meta_coherence': None,
            'p_value': None,
            'confidence_interval': None,
            'phi_optimality': None
        }
        
        # Calculate coherence matrix
        coherence_matrix = self._calculate_coherence_matrix()
        self.results['coherence_matrix'] = coherence_matrix
        
        # Calculate meta-coherence
        meta_coherence = self._calculate_meta_coherence(coherence_matrix)
        self.results['meta_coherence'] = meta_coherence
        
        # Test significance
        p_value, surrogate_meta_coherence = self._test_significance(meta_coherence)
        self.results['p_value'] = p_value
        self.results['surrogate_meta_coherence'] = surrogate_meta_coherence
        
        # Calculate confidence interval
        confidence_interval = self._calculate_confidence_interval(meta_coherence)
        self.results['confidence_interval'] = confidence_interval
        
        # Calculate phi optimality
        phi_optimality = calculate_phi_optimality(meta_coherence, CONSTANTS)
        self.results['phi_optimality'] = phi_optimality
        
        # Find best constant
        best_constant, best_value = find_best_constant(phi_optimality)
        self.results['best_constant'] = best_constant
        self.results['best_value'] = best_value
        
        print("\n")
        return self.results
    
    def _calculate_coherence_matrix(self):
        """
        Calculate coherence matrix across scales.
        
        Returns:
            ndarray: Coherence matrix
        """
        n_scales = len(self.scales)
        coherence_matrix = np.zeros((n_scales, n_scales))
        
        for i, scale_i in enumerate(self.scales):
            segments_i = segment_data(self.data, scale_i, overlap=0.5)
            
            for j, scale_j in enumerate(self.scales):
                if j < i:  # Only calculate lower triangle (matrix is symmetric)
                    coherence_matrix[i, j] = coherence_matrix[j, i]
                    continue
                
                segments_j = segment_data(self.data, scale_j, overlap=0.5)
                
                # Calculate average coherence between segments at different scales
                coherence_values = []
                
                # Limit the number of comparisons for efficiency
                max_comparisons = 100
                n_segments_i = len(segments_i)
                n_segments_j = len(segments_j)
                
                # Determine step sizes for sampling
                step_i = max(1, n_segments_i // int(np.sqrt(max_comparisons)))
                step_j = max(1, n_segments_j // int(np.sqrt(max_comparisons)))
                
                for idx_i in range(0, n_segments_i, step_i):
                    for idx_j in range(0, n_segments_j, step_j):
                        # Skip if segments are the same
                        if i == j and idx_i == idx_j:
                            continue
                        
                        # Calculate coherence using Welch's method
                        segment_i = segments_i[idx_i]
                        segment_j = segments_j[idx_j]
                        
                        # Ensure segments are the same length for coherence calculation
                        min_length = min(len(segment_i), len(segment_j))
                        segment_i = segment_i[:min_length]
                        segment_j = segment_j[:min_length]
                        
                        # Calculate coherence
                        nperseg = min(256, min_length // 2)
                        if nperseg < 8:  # Skip if segments are too short
                            continue
                        
                        f, coh = signal.coherence(segment_i, segment_j, fs=1.0, nperseg=nperseg)
                        
                        # Use mean coherence as the measure
                        coherence_values.append(np.mean(coh))
                
                # Store average coherence
                coherence_matrix[i, j] = np.mean(coherence_values) if coherence_values else 0.0
        
        return coherence_matrix
    
    def _calculate_meta_coherence(self, coherence_matrix):
        """
        Calculate meta-coherence from the coherence matrix.
        
        Meta-coherence measures how coherently the coherence values are organized.
        
        Args:
            coherence_matrix (ndarray): Coherence matrix
            
        Returns:
            float: Meta-coherence value
        """
        # Calculate eigenvalues of coherence matrix
        eigenvalues = np.linalg.eigvalsh(coherence_matrix)
        
        # Sort eigenvalues in descending order
        eigenvalues = np.sort(eigenvalues)[::-1]
        
        # Calculate participation ratio
        participation_ratio = np.sum(eigenvalues)**2 / np.sum(eigenvalues**2)
        
        # Normalize by matrix size
        normalized_participation = participation_ratio / len(coherence_matrix)
        
        # Calculate meta-coherence
        meta_coherence = normalized_participation
        
        return meta_coherence
    
    def _test_significance(self, meta_coherence):
        """
        Test statistical significance of meta-coherence.
        
        Args:
            meta_coherence (float): Meta-coherence value
            
        Returns:
            tuple: p-value and list of surrogate meta-coherence values
        """
        # Generate surrogate data
        surrogates = generate_surrogate_data(self.data, n_surrogates=self.n_surrogates, seed=self.seed)
        
        # Calculate meta-coherence for each surrogate
        surrogate_meta_coherence = []
        
        for i in range(self.n_surrogates):
            # Create surrogate test
            surrogate_test = MetaCoherenceTest(
                name="Surrogate Meta-Coherence Test {}".format(i),
                seed=self.seed + i if self.seed is not None else None,
                data=surrogates[i]
            )
            
            # Set the same scales
            surrogate_test.scales = self.scales
            
            # Calculate coherence matrix
            coherence_matrix = surrogate_test._calculate_coherence_matrix()
            
            # Calculate meta-coherence
            surrogate_mc = surrogate_test._calculate_meta_coherence(coherence_matrix)
            surrogate_meta_coherence.append(surrogate_mc)
        
        # Calculate p-value
        p_value = np.mean(np.array(surrogate_meta_coherence) >= meta_coherence)
        
        return p_value, surrogate_meta_coherence
    
    def _calculate_confidence_interval(self, meta_coherence):
        """
        Calculate bootstrap confidence interval for meta-coherence.
        
        Args:
            meta_coherence (float): Meta-coherence value
            
        Returns:
            tuple: Lower and upper bounds of confidence interval
        """
        # Generate bootstrap samples
        bootstrap_values = []
        
        for i in range(self.bootstrap_samples):
            # Resample data with replacement
            bootstrap_data = self._generate_bootstrap_sample()
            
            # Create bootstrap test
            bootstrap_test = MetaCoherenceTest(
                name="Bootstrap Meta-Coherence Test {}".format(i),
                seed=self.seed + i if self.seed is not None else None,
                data=bootstrap_data
            )
            
            # Set the same scales
            bootstrap_test.scales = self.scales
            
            # Calculate coherence matrix
            coherence_matrix = bootstrap_test._calculate_coherence_matrix()
            
            # Calculate meta-coherence
            bootstrap_mc = bootstrap_test._calculate_meta_coherence(coherence_matrix)
            bootstrap_values.append(bootstrap_mc)
        
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
        report.append("META-COHERENCE TEST REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Execution time
        if hasattr(self, 'execution_time') and self.execution_time is not None:
            report.append("Test completed in {:.2f} seconds.".format(self.execution_time))
            report.append("")
        
        # Meta-coherence value
        report.append("META-COHERENCE VALUE: {:.6f}".format(self.results['meta_coherence']))
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
        
        # Phi optimality
        report.append("PHI OPTIMALITY")
        report.append("-" * 50)
        for const, value in self.results['phi_optimality'].items():
            report.append("{}: {:.6f}".format(const, value))
        report.append("")
        report.append("Best constant: {} ({:.6f})".format(self.results['best_constant'], self.results['best_value']))
        report.append("")
        
        # Interpretation
        report.append("INTERPRETATION")
        report.append("-" * 50)
        report.append("The meta-coherence test examines if coherence itself shows coherent")
        report.append("organization across scales, which would indicate a higher-order organizing")
        report.append("principle in the cosmic microwave background.")
        report.append("")
        
        if self.results['p_value'] < 0.05:
            report.append("The analysis reveals STATISTICALLY SIGNIFICANT meta-coherence in the CMB data,")
            report.append("suggesting that coherence patterns are not randomly distributed across scales")
            report.append("but show an organized structure.")
        else:
            report.append("The analysis does not detect statistically significant meta-coherence in")
            report.append("the CMB data at the p < 0.05 level.")
        
        report.append("")
        
        if self.results['best_constant'] == 'phi':
            report.append("Notably, the golden ratio (phi) shows the strongest optimization in the")
            report.append("meta-coherence structure, which aligns with the hypothesis that phi-based")
            report.append("organization may be a signature of consciousness-like processes in cosmic structure.")
        else:
            report.append("The constant {} shows the strongest optimization in the meta-coherence".format(self.results['best_constant']))
            report.append("structure, which suggests that this mathematical relationship may play a")
            report.append("significant role in the organization of cosmic structure.")
        
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
        
        # Plot coherence matrix
        plot_heatmap(
            axes[0, 0],
            self.results['coherence_matrix'],
            x_labels=[str(s) for s in self.scales],
            y_labels=[str(s) for s in self.scales],
            title="Coherence Matrix Across Scales",
            colorbar_label="Coherence"
        )
        axes[0, 0].set_xlabel("Scale")
        axes[0, 0].set_ylabel("Scale")
        
        # Plot meta-coherence vs surrogate distribution
        axes[0, 1].hist(self.results['surrogate_meta_coherence'], bins=20, alpha=0.7, color='gray')
        axes[0, 1].axvline(self.results['meta_coherence'], color='red', linestyle='--', linewidth=2)
        axes[0, 1].set_xlabel("Meta-Coherence Value")
        axes[0, 1].set_ylabel("Frequency")
        axes[0, 1].set_title("Meta-Coherence vs Surrogate Distribution")
        
        # Add p-value annotation
        axes[0, 1].text(
            0.95, 0.95,
            "p-value: {:.4f}".format(self.results['p_value']),
            transform=axes[0, 1].transAxes,
            ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
        )
        
        # Plot phi optimality
        constants = list(self.results['phi_optimality'].keys())
        values = [self.results['phi_optimality'][c] for c in constants]
        colors = [COLORS.get(c, 'gray') for c in constants]
        
        axes[1, 0].bar(constants, values, color=colors)
        axes[1, 0].set_xlabel("Mathematical Constant")
        axes[1, 0].set_ylabel("Optimization Value")
        axes[1, 0].set_title("Optimization by Mathematical Constant")
        
        # Highlight best constant
        best_idx = constants.index(self.results['best_constant'])
        axes[1, 0].text(
            best_idx, values[best_idx] + 0.01,
            "Best",
            ha='center', va='bottom',
            fontweight='bold'
        )
        
        # Plot eigenvalue spectrum
        eigenvalues = np.linalg.eigvalsh(self.results['coherence_matrix'])
        eigenvalues = np.sort(eigenvalues)[::-1]
        
        axes[1, 1].plot(range(1, len(eigenvalues) + 1), eigenvalues, 'o-')
        axes[1, 1].set_xlabel("Eigenvalue Index")
        axes[1, 1].set_ylabel("Eigenvalue")
        axes[1, 1].set_title("Eigenvalue Spectrum of Coherence Matrix")
        axes[1, 1].set_yscale('log')
        
        # Add participation ratio annotation
        participation_ratio = np.sum(eigenvalues)**2 / np.sum(eigenvalues**2)
        axes[1, 1].text(
            0.95, 0.95,
            "Participation Ratio: {:.2f}".format(participation_ratio),
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
            save_figure(fig, "meta_coherence_results.png")
        
        # Show figure if requested
        if show:
            plt.show()
        
        return fig


def main():
    """
    Run the meta-coherence test.
    """
    # Create and run test
    test = MetaCoherenceTest(seed=42)
    test.run()
    
    # Generate report and visualizations
    test.generate_report()
    test.visualize_results()


if __name__ == "__main__":
    main()
