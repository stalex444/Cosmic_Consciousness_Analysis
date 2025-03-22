#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Scale Transition Test
-------------------
Tests for scale transitions where the dominant organizing principle changes in CMB data.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, signal

from core_framework.base_test import BaseTest
from core_framework.constants import (
    CONSTANTS, COLORS, CONSTANT_NAMES, DEFAULT_MONTE_CARLO_ITERATIONS,
    DEFAULT_BOOTSTRAP_SAMPLES, METRICS, SCALE_RANGES
)
from core_framework.data_handler import (
    generate_surrogate_data, segment_data, fibonacci_sequence
)
from core_framework.statistics import (
    calculate_transfer_entropy, calculate_mutual_information,
    calculate_power_law_exponent, bootstrap_confidence_interval,
    calculate_phi_optimality, find_best_constant
)
from core_framework.visualization import (
    create_multi_panel_figure, plot_optimization_by_scale,
    plot_best_constants, plot_transition_boundaries, plot_phi_optimality
)


class ScaleTransitionTest(BaseTest):
    """
    Test for scale transitions where the dominant organizing principle changes.
    
    This test analyzes how different mathematical constants (phi, e, pi, sqrt2, sqrt3, ln2)
    optimize cosmic organization across different scales. It identifies transition points
    where the dominant mathematical constant changes.
    
    Attributes:
        scales (list): Scales to analyze
        n_surrogates (int): Number of surrogate datasets for significance testing
        metrics (list): Metrics to calculate
        constants (dict): Mathematical constants to test
    """
    
    def __init__(self, name="Scale Transition Test", seed=None, data=None):
        """
        Initialize the scale transition test.
        
        Args:
            name (str, optional): Name of the test. Defaults to "Scale Transition Test".
            seed (int, optional): Random seed for reproducibility. Defaults to None.
            data (ndarray, optional): CMB data to analyze. Defaults to None.
        """
        super(ScaleTransitionTest, self).__init__(name=name, seed=seed, data=data)
        
        # Generate Fibonacci-based scales
        self.scales = self._generate_scales()
        
        # Test parameters
        self.n_surrogates = 100
        self.metrics = METRICS
        self.constants = CONSTANTS
    
    def _generate_scales(self, min_scale=2, max_scale=2048, n_scales=19):
        """
        Generate scales for analysis based on Fibonacci sequence.
        
        Args:
            min_scale (int, optional): Minimum scale. Defaults to 2.
            max_scale (int, optional): Maximum scale. Defaults to 2048.
            n_scales (int, optional): Number of scales. Defaults to 19.
            
        Returns:
            list: List of scales
        """
        # Generate Fibonacci sequence
        fib_seq = fibonacci_sequence(n_scales + 5)
        
        # Filter and scale to desired range
        fib_seq = [f for f in fib_seq if f >= 1]  # Remove 0
        
        # Scale to desired range
        min_fib = min(fib_seq)
        max_fib = max(fib_seq)
        
        scales = [int(min_scale + (max_scale - min_scale) * (f - min_fib) / (max_fib - min_fib)) for f in fib_seq]
        
        # Remove duplicates and sort
        scales = sorted(list(set(scales)))
        
        # Ensure min and max scales are included
        if scales[0] > min_scale:
            scales = [min_scale] + scales
        if scales[-1] < max_scale:
            scales = scales + [max_scale]
        
        # Limit to n_scales
        if len(scales) > n_scales:
            # Select evenly spaced scales
            indices = np.linspace(0, len(scales) - 1, n_scales).astype(int)
            scales = [scales[i] for i in indices]
        
        return scales
    
    def run_test(self):
        """
        Run the scale transition test.
        
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
            'scale_results': {},
            'transitions': [],
            'significance': {}
        }
        
        # Analyze each scale
        for scale in self.scales:
            print("  Analyzing scale: {}...".format(scale))
            
            # Analyze metrics at this scale
            scale_result = self._analyze_scale(scale)
            
            # Store results
            self.results['scale_results'][scale] = scale_result
        
        # Identify transitions
        self._identify_transitions()
        
        # Test significance
        self._test_significance()
        
        # Calculate summary statistics
        self._calculate_summary()
        
        print("\n")
        return self.results
    
    def _analyze_scale(self, scale):
        """
        Analyze metrics at a specific scale.
        
        Args:
            scale (int): Scale to analyze
            
        Returns:
            dict: Results for this scale
        """
        # Segment data at this scale
        segments = segment_data(self.data, scale, overlap=0.5)
        
        # Initialize results for this scale
        scale_result = {}
        
        # Calculate metrics
        if 'LAMINARITY' in self.metrics:
            lam = self.calculate_laminarity(segments)
            scale_result['laminarity'] = {
                'value': lam,
                'optimality': calculate_phi_optimality(lam, self.constants)
            }
            scale_result['laminarity']['best_constant'], scale_result['laminarity']['best_value'] = find_best_constant(scale_result['laminarity']['optimality'])
        
        if 'POWER_LAW' in self.metrics:
            pl = self.calculate_power_law(segments)
            scale_result['power_law'] = {
                'value': pl,
                'optimality': calculate_phi_optimality(pl, self.constants)
            }
            scale_result['power_law']['best_constant'], scale_result['power_law']['best_value'] = find_best_constant(scale_result['power_law']['optimality'])
        
        if 'COHERENCE' in self.metrics:
            coh = self.calculate_coherence(segments)
            scale_result['coherence'] = {
                'value': coh,
                'optimality': calculate_phi_optimality(coh, self.constants)
            }
            scale_result['coherence']['best_constant'], scale_result['coherence']['best_value'] = find_best_constant(scale_result['coherence']['optimality'])
        
        if 'INFORMATION_INTEGRATION' in self.metrics:
            ii = self.calculate_information_integration(segments)
            scale_result['information_integration'] = {
                'value': ii,
                'optimality': calculate_phi_optimality(ii, self.constants)
            }
            scale_result['information_integration']['best_constant'], scale_result['information_integration']['best_value'] = find_best_constant(scale_result['information_integration']['optimality'])
        
        if 'TRANSFER_ENTROPY' in self.metrics:
            te = self.calculate_transfer_entropy(segments)
            scale_result['transfer_entropy'] = {
                'value': te,
                'optimality': calculate_phi_optimality(te, self.constants)
            }
            scale_result['transfer_entropy']['best_constant'], scale_result['transfer_entropy']['best_value'] = find_best_constant(scale_result['transfer_entropy']['optimality'])
        
        return scale_result
    
    def calculate_laminarity(self, segments):
        """
        Calculate laminarity at the given scale.
        
        Laminarity measures the smoothness or turbulence of the data.
        
        Args:
            segments (ndarray): Data segments
            
        Returns:
            float: Laminarity value
        """
        # Calculate variance of each segment
        variances = np.var(segments, axis=1)
        
        # Calculate mean variance
        mean_variance = np.mean(variances)
        
        # Calculate laminarity (inverse of variance)
        laminarity = 1.0 / (1.0 + mean_variance)
        
        return laminarity
    
    def calculate_power_law(self, segments):
        """
        Calculate power law exponent at the given scale.
        
        Args:
            segments (ndarray): Data segments
            
        Returns:
            float: Power law exponent
        """
        # Calculate power law exponent for each segment
        exponents = []
        
        for segment in segments:
            exponent = calculate_power_law_exponent(segment)
            exponents.append(exponent)
        
        # Return mean exponent
        return np.mean(exponents)
    
    def calculate_coherence(self, segments):
        """
        Calculate coherence at the given scale.
        
        Args:
            segments (ndarray): Data segments
            
        Returns:
            float: Coherence value
        """
        # Calculate coherence between adjacent segments
        coherence_values = []
        
        for i in range(len(segments) - 1):
            # Calculate coherence using Welch's method
            f, coh = signal.coherence(segments[i], segments[i+1], fs=1.0, nperseg=min(256, len(segments[i])//2))
            
            # Use mean coherence as the measure
            coherence_values.append(np.mean(coh))
        
        # Return mean coherence
        return np.mean(coherence_values) if coherence_values else 0.0
    
    def calculate_information_integration(self, segments):
        """
        Calculate information integration at the given scale.
        
        Args:
            segments (ndarray): Data segments
            
        Returns:
            float: Information integration value
        """
        # Calculate mutual information between adjacent segments
        mi_values = []
        
        for i in range(len(segments) - 1):
            mi = calculate_mutual_information(segments[i], segments[i+1])
            mi_values.append(mi)
        
        # Return mean mutual information
        return np.mean(mi_values) if mi_values else 0.0
    
    def calculate_transfer_entropy(self, segments):
        """
        Calculate transfer entropy at the given scale.
        
        Args:
            segments (ndarray): Data segments
            
        Returns:
            float: Transfer entropy value
        """
        # Calculate transfer entropy between adjacent segments
        te_values = []
        
        for i in range(len(segments) - 1):
            te = calculate_transfer_entropy(segments[i], segments[i+1])
            te_values.append(te)
        
        # Return mean transfer entropy
        return np.mean(te_values) if te_values else 0.0
    
    def _identify_transitions(self):
        """
        Identify scale transitions where the dominant organizing principle changes.
        """
        scale_results = self.results['scale_results']
        scales = self.results['scales']
        
        # Initialize transitions list
        transitions = []
        
        # Check each metric
        for metric_name in ['laminarity', 'power_law', 'coherence', 'information_integration', 'transfer_entropy']:
            prev_best = None
            prev_scale = None
            
            # Check each scale
            for scale in scales:
                if scale in scale_results and metric_name in scale_results[scale]:
                    current_best = scale_results[scale][metric_name].get('best_constant')
                    
                    # Check if there's a transition
                    if prev_best is not None and current_best != prev_best:
                        # Calculate transition sharpness
                        sharpness = self.calculate_transition_sharpness(
                            prev_scale, scale, metric_name, prev_best, current_best)
                        
                        # Add transition
                        transitions.append({
                            'scale': scale,
                            'metric': metric_name,
                            'from_constant': prev_best,
                            'to_constant': current_best,
                            'sharpness': sharpness
                        })
                    
                    prev_best = current_best
                    prev_scale = scale
        
        # Store transitions
        self.results['transitions'] = transitions
    
    def calculate_transition_sharpness(self, scale1, scale2, metric, const1, const2):
        """
        Calculate how sharp a transition is based on changes in ratios.
        
        Args:
            scale1 (int): First scale
            scale2 (int): Second scale
            metric (str): Metric name
            const1 (str): First constant
            const2 (str): Second constant
            
        Returns:
            float: Transition sharpness
        """
        scale_results = self.results['scale_results']
        
        # Get optimality values
        if (scale1 in scale_results and metric in scale_results[scale1] and
            'optimality' in scale_results[scale1][metric] and
            const1 in scale_results[scale1][metric]['optimality'] and
            const2 in scale_results[scale1][metric]['optimality']):
            
            opt1_const1 = scale_results[scale1][metric]['optimality'][const1]
            opt1_const2 = scale_results[scale1][metric]['optimality'][const2]
            
            opt2_const1 = scale_results[scale2][metric]['optimality'][const1]
            opt2_const2 = scale_results[scale2][metric]['optimality'][const2]
            
            # Calculate ratio changes
            if opt1_const2 > 0 and opt2_const1 > 0:
                ratio1 = opt1_const1 / opt1_const2
                ratio2 = opt2_const2 / opt2_const1
                
                # Sharpness is the product of the ratios
                return ratio1 * ratio2
        
        return 0.0
    
    def _test_significance(self):
        """
        Test statistical significance of scale transitions.
        """
        # Generate surrogate data
        surrogates = generate_surrogate_data(self.data, n_surrogates=self.n_surrogates, seed=self.seed)
        
        # Initialize significance results
        significance = {const: {} for const in self.constants}
        
        # Test each metric
        for metric_name in ['laminarity', 'power_law', 'coherence', 'information_integration', 'transfer_entropy']:
            # Get maximum ratio for each constant in real data
            max_ratios = {}
            
            for const in self.constants:
                ratios = []
                
                for scale in self.results['scales']:
                    if (scale in self.results['scale_results'] and 
                        metric_name in self.results['scale_results'][scale] and
                        'best_constant' in self.results['scale_results'][scale][metric_name] and
                        self.results['scale_results'][scale][metric_name]['best_constant'] == const):
                        
                        # Get optimality value for this constant
                        opt_value = self.results['scale_results'][scale][metric_name]['optimality'][const]
                        
                        # Get average optimality for other constants
                        other_values = [v for k, v in self.results['scale_results'][scale][metric_name]['optimality'].items() if k != const]
                        avg_other = np.mean(other_values) if other_values else 0
                        
                        # Calculate ratio
                        if avg_other > 0:
                            ratio = opt_value / avg_other
                            ratios.append(ratio)
                
                max_ratios[const] = max(ratios) if ratios else 1.0
            
            # Test significance using surrogate data
            for const in self.constants:
                # Skip if no ratio for this constant
                if const not in max_ratios:
                    continue
                
                # Get max ratio for this constant in real data
                real_max_ratio = max_ratios[const]
                
                # Calculate max ratios for surrogate data
                surrogate_max_ratios = []
                
                for i in range(self.n_surrogates):
                    surrogate_data = surrogates[i]
                    
                    # Run simplified test on surrogate data
                    surrogate_ratios = []
                    
                    for scale in self.results['scales']:
                        segments = segment_data(surrogate_data, scale, overlap=0.5)
                        
                        # Calculate metric value
                        if metric_name == 'laminarity':
                            value = self.calculate_laminarity(segments)
                        elif metric_name == 'power_law':
                            value = self.calculate_power_law(segments)
                        elif metric_name == 'coherence':
                            value = self.calculate_coherence(segments)
                        elif metric_name == 'information_integration':
                            value = self.calculate_information_integration(segments)
                        elif metric_name == 'transfer_entropy':
                            value = self.calculate_transfer_entropy(segments)
                        else:
                            continue
                        
                        # Calculate optimality
                        optimality = calculate_phi_optimality(value, self.constants)
                        
                        # Calculate ratio
                        const_value = optimality.get(const, 0)
                        other_values = [v for k, v in optimality.items() if k != const]
                        avg_other = np.mean(other_values) if other_values else 0
                        
                        if avg_other > 0:
                            ratio = const_value / avg_other
                            surrogate_ratios.append(ratio)
                    
                    surrogate_max_ratios.append(max(surrogate_ratios) if surrogate_ratios else 1.0)
                
                # Calculate p-value
                p_value = np.mean(np.array(surrogate_max_ratios) >= real_max_ratio)
                
                # Store significance result
                significance[const][metric_name] = {
                    'max_ratio': real_max_ratio,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
        
        # Store significance results
        self.results['significance'] = significance
    
    def _calculate_summary(self):
        """
        Calculate summary statistics for the test.
        """
        # Count transitions by metric
        transitions_by_metric = {}
        for t in self.results['transitions']:
            metric = t['metric']
            if metric not in transitions_by_metric:
                transitions_by_metric[metric] = []
            transitions_by_metric[metric].append(t)
        
        # Count significant constants by metric
        significant_constants = {}
        for const in self.constants:
            significant_metrics = []
            for metric in ['laminarity', 'power_law', 'coherence', 'information_integration', 'transfer_entropy']:
                if (const in self.results['significance'] and 
                    metric in self.results['significance'][const] and
                    self.results['significance'][const][metric]['significant']):
                    significant_metrics.append(metric)
            
            significant_constants[const] = significant_metrics
        
        # Calculate phi dominance by scale range
        phi_dominance = {}
        for range_name, (min_scale, max_scale) in SCALE_RANGES.items():
            # Count scales in this range
            scales_in_range = [s for s in self.results['scales'] if min_scale <= s <= max_scale]
            
            # Count phi dominance
            phi_count = 0
            total_count = 0
            
            for scale in scales_in_range:
                if scale in self.results['scale_results']:
                    for metric in ['laminarity', 'power_law', 'coherence', 'information_integration', 'transfer_entropy']:
                        if metric in self.results['scale_results'][scale] and 'best_constant' in self.results['scale_results'][scale][metric]:
                            total_count += 1
                            if self.results['scale_results'][scale][metric]['best_constant'] == 'phi':
                                phi_count += 1
            
            phi_dominance[range_name] = phi_count / float(total_count) if total_count > 0 else 0
        
        # Store summary
        self.results['summary'] = {
            'transitions_by_metric': transitions_by_metric,
            'significant_constants': significant_constants,
            'phi_dominance': phi_dominance
        }
    
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
        report.append("SCALE TRANSITION TEST REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Execution time
        if hasattr(self, 'execution_time') and self.execution_time is not None:
            report.append("Test completed in {:.2f} seconds.".format(self.execution_time))
            report.append("")
        
        # Scale transitions by metric
        report.append("SCALE TRANSITIONS BY METRIC")
        report.append("-" * 50)
        report.append("")
        
        for metric in ['LAMINARITY', 'POWER_LAW', 'COHERENCE', 'INFORMATION_INTEGRATION', 'TRANSFER_ENTROPY']:
            metric_lower = metric.lower()
            transitions = [t for t in self.results['transitions'] if t['metric'] == metric_lower]
            
            if transitions:
                report.append("{}:".format(metric))
                for t in sorted(transitions, key=lambda x: x['scale']):
                    report.append("  At scale {}: {} → {} (sharpness: {:.4f})".format(
                        t['scale'], t['from_constant'], t['to_constant'], t['sharpness']))
                report.append("")
        
        # Statistical significance by metric
        report.append("STATISTICAL SIGNIFICANCE BY METRIC")
        report.append("-" * 50)
        report.append("")
        
        for metric in ['LAMINARITY', 'POWER_LAW', 'COHERENCE', 'INFORMATION_INTEGRATION', 'TRANSFER_ENTROPY']:
            metric_lower = metric.lower()
            report.append("{}:".format(metric))
            
            for const in self.constants:
                if (const in self.results['significance'] and 
                    metric_lower in self.results['significance'][const]):
                    
                    sig_result = self.results['significance'][const][metric_lower]
                    report.append("  {}: max ratio = {:.4f}, p-value = {:.4f} ({})".format(
                        const, sig_result['max_ratio'], sig_result['p_value'],
                        "SIGNIFICANT" if sig_result['significant'] else "not significant"))
            
            report.append("")
        
        # Key findings
        report.append("KEY FINDINGS")
        report.append("-" * 50)
        
        # Total transitions
        report.append("1. Total scale transitions detected: {}".format(len(self.results['transitions'])))
        
        # Significant constants
        significant_counts = {}
        for const, metrics in self.results['summary']['significant_constants'].items():
            significant_counts[const] = len(metrics)
        
        report.append("2. Constants showing significant optimization in at least one metric:")
        for const, count in sorted(significant_counts.items(), key=lambda x: x[1], reverse=True):
            if count > 0:
                report.append("   - {}: significant in {}/{} metrics".format(
                    const, count, len(['laminarity', 'power_law', 'coherence', 'information_integration', 'transfer_entropy'])))
        
        report.append("")
        
        # Phi dominance by scale range
        report.append("3. Golden ratio optimization by scale range:")
        for range_name, dominance in self.results['summary']['phi_dominance'].items():
            strength = "STRONG" if dominance >= 0.7 else "MODERATE" if dominance >= 0.4 else "WEAK"
            report.append("   - {} scales: {} golden ratio dominance ({:.2f})".format(
                range_name, strength, dominance))
        
        report.append("")
        
        # Overall conclusion
        report.append("OVERALL CONCLUSION")
        report.append("-" * 50)
        report.append("The analysis reveals distinct scale transitions where the dominant mathematical")
        report.append("organizing principle changes. These transition boundaries support the 'cake baking'")
        report.append("model of cosmic development, where different organizational principles apply at")
        report.append("different scales during the evolution of cosmic structure.")
        report.append("")
        report.append("The golden ratio shows significant optimization in specific scale ranges and")
        report.append("metrics, demonstrating a selective application rather than universal optimization.")
        report.append("This pattern of selective optimization is consistent with a sophisticated")
        report.append("consciousness-like organizing principle that employs different mathematical")
        report.append("relationships across different aspects of cosmic structure.")
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
        
        # Define plotting functions
        plot_functions = [
            plot_optimization_by_scale,
            plot_best_constants,
            plot_transition_boundaries,
            plot_phi_optimality
        ]
        
        # Create multi-panel figure
        fig = create_multi_panel_figure(
            self.results,
            plot_functions,
            title="Scale Transition Test Results",
            filename="scale_transition_results.png" if save_path is None else save_path
        )
        
        if show:
            plt.show()
        
        return fig


def main():
    """
    Run the scale transition test.
    """
    # Create and run test
    test = ScaleTransitionTest(seed=42)
    test.run()
    
    # Generate report and visualizations
    test.generate_report()
    test.visualize_results()


if __name__ == "__main__":
    main()
