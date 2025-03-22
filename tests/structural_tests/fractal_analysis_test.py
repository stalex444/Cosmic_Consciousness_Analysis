#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fractal Analysis Test
------------------
Tests for fractal patterns in the structural properties of CMB data.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, signal

from core_framework.base_test import BaseTest
from core_framework.constants import (
    CONSTANTS, COLORS, CONSTANT_NAMES, DEFAULT_MONTE_CARLO_ITERATIONS,
    DEFAULT_BOOTSTRAP_SAMPLES, PHI
)
from core_framework.data_handler import (
    generate_surrogate_data, segment_data, fibonacci_sequence
)
from core_framework.statistics import (
    bootstrap_confidence_interval, calculate_phi_optimality, find_best_constant,
    calculate_hurst_exponent
)
from core_framework.visualization import (
    create_multi_panel_figure, setup_figure, save_figure
)


class FractalAnalysisTest(BaseTest):
    """
    Test for fractal patterns in the structural properties of CMB data.
    
    This test examines if the CMB data exhibits fractal properties across scales,
    particularly focusing on the Hurst exponent and its relationship to the
    golden ratio.
    
    Attributes:
        n_surrogates (int): Number of surrogate datasets for significance testing
        bootstrap_samples (int): Number of bootstrap samples for confidence intervals
        scales (list): List of scales to analyze
    """
    
    def __init__(self, name="Fractal Analysis Test", seed=None, data=None):
        """
        Initialize the fractal analysis test.
        
        Args:
            name (str, optional): Name of the test. Defaults to "Fractal Analysis Test".
            seed (int, optional): Random seed for reproducibility. Defaults to None.
            data (ndarray, optional): CMB data to analyze. Defaults to None.
        """
        super(FractalAnalysisTest, self).__init__(name=name, seed=seed, data=data)
        
        # Test parameters
        self.n_surrogates = 100
        self.bootstrap_samples = DEFAULT_BOOTSTRAP_SAMPLES
        
        # Generate scales based on Fibonacci sequence
        self.scales = self._generate_scales()
    
    def _generate_scales(self, min_scale=8, max_scale=512, n_scales=10):
        """
        Generate scales for analysis based on Fibonacci sequence.
        
        Args:
            min_scale (int, optional): Minimum scale. Defaults to 8.
            max_scale (int, optional): Maximum scale. Defaults to 512.
            n_scales (int, optional): Number of scales. Defaults to 10.
            
        Returns:
            list: List of scales
        """
        # Generate Fibonacci sequence
        fib_seq = fibonacci_sequence(n_scales + 10)
        
        # Filter to desired range
        fib_seq = [f for f in fib_seq if f >= min_scale and f <= max_scale]
        
        # Limit to n_scales
        if len(fib_seq) > n_scales:
            # Select evenly spaced scales
            indices = np.linspace(0, len(fib_seq) - 1, n_scales).astype(int)
            scales = [fib_seq[i] for i in indices]
        else:
            scales = fib_seq
        
        return scales
    
    def run_test(self):
        """
        Run the fractal analysis test.
        
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
            'hurst_exponents': None,
            'scale_ratios': None,
            'phi_optimality': None,
            'surrogate_results': None,
            'p_value': None,
            'confidence_interval': None
        }
        
        # Calculate Hurst exponents at different scales
        hurst_exponents = self._calculate_multiscale_hurst()
        self.results['hurst_exponents'] = hurst_exponents
        
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
        p_value, surrogate_results = self._test_significance()
        self.results['p_value'] = p_value
        self.results['surrogate_results'] = surrogate_results
        
        # Calculate confidence interval
        confidence_interval = self._calculate_confidence_interval()
        self.results['confidence_interval'] = confidence_interval
        
        print("\n")
        return self.results
    
    def _calculate_multiscale_hurst(self):
        """
        Calculate Hurst exponents at multiple scales.
        
        Returns:
            dict: Hurst exponents at different scales
        """
        hurst_exponents = {}
        
        for scale in self.scales:
            # Segment data
            segments = segment_data(self.data, scale, overlap=0.5)
            
            # Calculate Hurst exponent for each segment
            segment_hurst = []
            for segment in segments:
                if len(segment) >= 8:  # Minimum length for reliable Hurst estimation
                    h = calculate_hurst_exponent(segment)
                    segment_hurst.append(h)
            
            # Store mean Hurst exponent
            if segment_hurst:
                hurst_exponents[scale] = np.mean(segment_hurst)
        
        return hurst_exponents
    
    def _calculate_scale_ratios(self):
        """
        Calculate ratios between scales where Hurst exponent changes significantly.
        
        Returns:
            dict: Scale ratios and their significance
        """
        hurst_exponents = self.results['hurst_exponents']
        scales = sorted(hurst_exponents.keys())
        
        # Calculate differences in Hurst exponent
        hurst_diffs = {}
        for i in range(len(scales) - 1):
            scale_i = scales[i]
            scale_j = scales[i + 1]
            
            hurst_i = hurst_exponents[scale_i]
            hurst_j = hurst_exponents[scale_j]
            
            # Calculate difference
            diff = abs(hurst_j - hurst_i)
            
            # Calculate ratio (always larger / smaller)
            ratio = float(scale_j) / float(scale_i)
            
            # Store ratio and difference
            hurst_diffs[ratio] = diff
        
        return hurst_diffs
    
    def _test_significance(self):
        """
        Test statistical significance of scale ratios.
        
        Returns:
            tuple: p-value and surrogate results
        """
        # Generate surrogate data
        surrogates = generate_surrogate_data(self.data, n_surrogates=self.n_surrogates, seed=self.seed)
        
        # Calculate scale ratios for each surrogate
        surrogate_results = []
        
        for i in range(self.n_surrogates):
            # Create surrogate test
            surrogate_test = FractalAnalysisTest(
                name="Surrogate Fractal Analysis Test {}".format(i),
                seed=self.seed + i if self.seed is not None else None,
                data=surrogates[i]
            )
            
            # Set the same scales
            surrogate_test.scales = self.scales
            
            # Calculate Hurst exponents
            surrogate_hurst = surrogate_test._calculate_multiscale_hurst()
            
            # Calculate scale ratios
            surrogate_scale_ratios = {}
            scales = sorted(surrogate_hurst.keys())
            
            for j in range(len(scales) - 1):
                scale_i = scales[j]
                scale_j = scales[j + 1]
                
                if scale_i in surrogate_hurst and scale_j in surrogate_hurst:
                    hurst_i = surrogate_hurst[scale_i]
                    hurst_j = surrogate_hurst[scale_j]
                    
                    # Calculate difference
                    diff = abs(hurst_j - hurst_i)
                    
                    # Calculate ratio
                    ratio = float(scale_j) / float(scale_i)
                    
                    # Store ratio and difference
                    surrogate_scale_ratios[ratio] = diff
            
            # Calculate phi optimality
            surrogate_phi_optimality = calculate_phi_optimality(surrogate_scale_ratios, CONSTANTS)
            
            # Find best constant
            surrogate_best_constant, surrogate_best_value = find_best_constant(surrogate_phi_optimality)
            
            # Store results
            surrogate_results.append({
                'hurst_exponents': surrogate_hurst,
                'scale_ratios': surrogate_scale_ratios,
                'phi_optimality': surrogate_phi_optimality,
                'best_constant': surrogate_best_constant,
                'best_value': surrogate_best_value
            })
        
        # Calculate p-value based on phi optimality for golden ratio
        real_phi_optimality = self.results['phi_optimality']['phi']
        surrogate_phi_optimality = [s['phi_optimality']['phi'] for s in surrogate_results]
        
        p_value = np.mean(np.array(surrogate_phi_optimality) >= real_phi_optimality)
        
        return p_value, surrogate_results
    
    def _calculate_confidence_interval(self):
        """
        Calculate bootstrap confidence interval for phi optimality.
        
        Returns:
            tuple: Lower and upper bounds of confidence interval
        """
        # Generate bootstrap samples
        bootstrap_values = []
        
        for i in range(self.bootstrap_samples):
            # Resample data with replacement
            indices = np.random.choice(len(self.data), size=len(self.data), replace=True)
            bootstrap_data = self.data[indices]
            
            # Create bootstrap test
            bootstrap_test = FractalAnalysisTest(
                name="Bootstrap Fractal Analysis Test {}".format(i),
                seed=self.seed + i if self.seed is not None else None,
                data=bootstrap_data
            )
            
            # Set the same scales
            bootstrap_test.scales = self.scales
            
            # Calculate Hurst exponents
            bootstrap_hurst = bootstrap_test._calculate_multiscale_hurst()
            
            # Calculate scale ratios
            bootstrap_scale_ratios = {}
            scales = sorted(bootstrap_hurst.keys())
            
            for j in range(len(scales) - 1):
                scale_i = scales[j]
                scale_j = scales[j + 1]
                
                if scale_i in bootstrap_hurst and scale_j in bootstrap_hurst:
                    hurst_i = bootstrap_hurst[scale_i]
                    hurst_j = bootstrap_hurst[scale_j]
                    
                    # Calculate difference
                    diff = abs(hurst_j - hurst_i)
                    
                    # Calculate ratio
                    ratio = float(scale_j) / float(scale_i)
                    
                    # Store ratio and difference
                    bootstrap_scale_ratios[ratio] = diff
            
            # Calculate phi optimality
            bootstrap_phi_optimality = calculate_phi_optimality(bootstrap_scale_ratios, CONSTANTS)
            
            # Store phi optimality for golden ratio
            bootstrap_values.append(bootstrap_phi_optimality['phi'])
        
        # Calculate confidence interval
        confidence_interval = bootstrap_confidence_interval(np.array(bootstrap_values), 
                                                           lambda x: np.mean(x))
        
        return confidence_interval
    
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
        report.append("FRACTAL ANALYSIS TEST REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Execution time
        if hasattr(self, 'execution_time') and self.execution_time is not None:
            report.append("Test completed in {:.2f} seconds.".format(self.execution_time))
            report.append("")
        
        # Hurst exponents
        report.append("HURST EXPONENTS AT DIFFERENT SCALES")
        report.append("-" * 50)
        
        # Sort scales
        sorted_scales = sorted(self.results['hurst_exponents'].keys())
        
        for scale in sorted_scales:
            hurst = self.results['hurst_exponents'][scale]
            report.append("Scale {}: H = {:.4f}".format(scale, hurst))
        
        report.append("")
        
        # Scale ratios
        report.append("SCALE RATIOS WITH SIGNIFICANT HURST CHANGES")
        report.append("-" * 50)
        
        # Sort ratios by difference
        sorted_ratios = sorted(self.results['scale_ratios'].items(), key=lambda x: x[1], reverse=True)
        
        for ratio, diff in sorted_ratios:
            # Find closest constant
            closest_const = None
            min_diff = float('inf')
            
            for const_name, const_value in CONSTANTS.items():
                diff_const = abs(ratio - const_value)
                if diff_const < min_diff:
                    min_diff = diff_const
                    closest_const = const_name
            
            # Calculate percent difference
            percent_diff = 100 * min_diff / CONSTANTS[closest_const]
            
            report.append("Ratio {:.4f} (Hurst diff = {:.6f})".format(ratio, diff))
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
        report.append("The fractal analysis test examines if the CMB data exhibits fractal properties")
        report.append("across scales, particularly focusing on the Hurst exponent and its relationship")
        report.append("to the golden ratio.")
        report.append("")
        
        # Mean Hurst exponent
        mean_hurst = np.mean(list(self.results['hurst_exponents'].values()))
        report.append("Mean Hurst exponent: {:.4f}".format(mean_hurst))
        
        if mean_hurst > 0.55:
            report.append("The CMB data shows persistent behavior (H > 0.5), indicating long-range")
            report.append("correlations and memory effects in the data.")
        elif mean_hurst < 0.45:
            report.append("The CMB data shows anti-persistent behavior (H < 0.5), indicating a")
            report.append("tendency to reverse direction more frequently than random processes.")
        else:
            report.append("The CMB data shows behavior close to a random walk (H ≈ 0.5), indicating")
            report.append("little to no long-range correlations.")
        
        report.append("")
        
        if self.results['p_value'] < 0.05:
            report.append("The analysis reveals STATISTICALLY SIGNIFICANT fractal patterns")
            report.append("in the CMB data, with scale transitions that preferentially occur at")
            report.append("ratios related to mathematical constants.")
        else:
            report.append("The analysis does not detect statistically significant fractal patterns")
            report.append("in the CMB data at the p < 0.05 level.")
        
        report.append("")
        
        if self.results['best_constant'] == 'phi':
            report.append("Notably, the golden ratio (phi) shows the strongest optimization in the")
            report.append("scale ratios where significant changes in fractal behavior occur, which")
            report.append("aligns with the hypothesis that phi-based organization may be a signature")
            report.append("of consciousness-like processes in cosmic structure.")
        else:
            report.append("The constant {} shows the strongest optimization in the scale".format(self.results['best_constant']))
            report.append("ratios where significant changes in fractal behavior occur, which suggests")
            report.append("that this mathematical relationship may play a significant role in the")
            report.append("organization of cosmic structure.")
        
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
        
        # Plot Hurst exponents vs scale
        scales = sorted(self.results['hurst_exponents'].keys())
        hurst_values = [self.results['hurst_exponents'][s] for s in scales]
        
        axes[0, 0].plot(scales, hurst_values, 'o-', color='blue', linewidth=2)
        axes[0, 0].set_title("Hurst Exponent vs Scale")
        axes[0, 0].set_xlabel("Scale")
        axes[0, 0].set_ylabel("Hurst Exponent")
        axes[0, 0].axhline(y=0.5, color='gray', linestyle='--', label='Random Walk (H=0.5)')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # Plot scale ratios
        sorted_ratios = sorted(self.results['scale_ratios'].items(), key=lambda x: x[1], reverse=True)
        top_ratios = sorted_ratios[:10] if len(sorted_ratios) > 10 else sorted_ratios
        
        ratios = [r[0] for r in top_ratios]
        diff_values = [r[1] for r in top_ratios]
        
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
        
        axes[0, 1].bar(range(len(top_ratios)), diff_values, color=colors)
        axes[0, 1].set_title("Top Scale Ratios with Significant Hurst Changes")
        axes[0, 1].set_xlabel("Rank")
        axes[0, 1].set_ylabel("Hurst Difference")
        
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
            save_figure(fig, "fractal_analysis_results.png")
        
        # Show figure if requested
        if show:
            plt.show()
        
        return fig


def main():
    """
    Run the fractal analysis test.
    """
    # Create and run test
    test = FractalAnalysisTest(seed=42)
    test.run()
    
    # Generate report and visualizations
    test.generate_report()
    test.visualize_results()


if __name__ == "__main__":
    main()
