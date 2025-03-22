#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Analysis Module
-------------
Comprehensive analysis module to combine results across multiple tests.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import json
import datetime
import time
from joblib import Parallel, delayed

from core_framework.constants import (
    CONSTANTS, COLORS, CONSTANT_NAMES, DEFAULT_MONTE_CARLO_ITERATIONS,
    DEFAULT_BOOTSTRAP_SAMPLES
)
from core_framework.visualization import (
    create_multi_panel_figure, setup_figure, save_figure
)


class CosmicConsciousnessAnalyzer:
    """
    Comprehensive analyzer for cosmic consciousness tests.
    
    This class combines results from multiple tests to provide a comprehensive
    analysis of cosmic consciousness patterns in CMB data.
    
    Attributes:
        tests (list): List of test objects
        results (dict): Combined results from all tests
        name (str): Name of the analyzer
        output_dir (str): Directory for output files
    """
    
    def __init__(self, name="Cosmic Consciousness Analyzer", output_dir=None):
        """
        Initialize the cosmic consciousness analyzer.
        
        Args:
            name (str, optional): Name of the analyzer. Defaults to "Cosmic Consciousness Analyzer".
            output_dir (str, optional): Directory for output files. Defaults to None.
        """
        self.name = name
        self.tests = []
        self.results = {}
        
        # Set output directory
        if output_dir is None:
            self.output_dir = os.path.join(os.getcwd(), "results")
        else:
            self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def add_test(self, test):
        """
        Add a test to the analyzer.
        
        Args:
            test: Test object to add
        """
        self.tests.append(test)
    
    def run_all_tests(self, data=None):
        """
        Run all added tests.
        
        Args:
            data (ndarray, optional): Data to analyze. Defaults to None.
            
        Returns:
            dict: Combined results from all tests
        """
        print("Running all tests...")
        
        # Initialize results
        self.results = {
            'tests': {},
            'combined': {},
            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Run each test
        for test in self.tests:
            print("\nRunning test: {}".format(test.name))
            start_time = time.time()
            
            # Set data if provided
            if data is not None:
                test.data = data
            
            # Run test
            test_results = test.run()
            
            # Store results
            self.results['tests'][test.name] = test_results
            
            print("Test completed in {:.2f} seconds.".format(time.time() - start_time))
        
        # Combine results
        self._combine_results()
        
        print("\nAll tests completed.")
        return self.results
    
    def run_all_tests_parallel(self, data=None, n_jobs=-1):
        """
        Run all added tests in parallel using joblib.
        
        Args:
            data (ndarray, optional): Data to analyze. Defaults to None.
            n_jobs (int, optional): Number of jobs to run in parallel. -1 means using all processors.
                Defaults to -1.
            
        Returns:
            dict: Combined results from all tests
        """
        print("Running all tests in parallel...")
        
        # Initialize results
        self.results = {
            'tests': {},
            'combined': {},
            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Define function to run a single test
        def run_single_test(test, test_data):
            print("Starting test: {}".format(test.name))
            start_time = time.time()
            
            # Set data if provided
            if test_data is not None:
                test.data = test_data
            
            # Run test
            test_results = test.run()
            
            print("Test {} completed in {:.2f} seconds.".format(test.name, time.time() - start_time))
            return (test.name, test_results)
        
        # Run tests in parallel
        test_results = Parallel(n_jobs=n_jobs)(
            delayed(run_single_test)(test, data) for test in self.tests
        )
        
        # Store results
        for name, results in test_results:
            self.results['tests'][name] = results
        
        # Combine results
        self._combine_results()
        
        print("\nAll tests completed.")
        return self.results
    
    def _combine_results(self):
        """
        Combine results from all tests.
        """
        # Initialize combined results
        combined = {
            'phi_optimality': {},
            'p_values': {},
            'best_constants': {},
            'significance': {}
        }
        
        # Collect phi optimality and p-values from all tests
        for test_name, test_results in self.results['tests'].items():
            # Skip if test has no phi optimality
            if 'phi_optimality' not in test_results:
                continue
            
            # Store phi optimality
            combined['phi_optimality'][test_name] = test_results['phi_optimality']
            
            # Store p-value
            if 'p_value' in test_results:
                combined['p_values'][test_name] = test_results['p_value']
            
            # Store best constant
            if 'best_constant' in test_results and 'best_value' in test_results:
                combined['best_constants'][test_name] = {
                    'constant': test_results['best_constant'],
                    'value': test_results['best_value']
                }
        
        # Calculate combined significance using Fisher's method
        p_values = list(combined['p_values'].values())
        
        if p_values:
            # Calculate Fisher's combined probability test
            chi_square = -2 * np.sum(np.log(p_values))
            degrees_of_freedom = 2 * len(p_values)
            combined_p_value = 1.0 - stats.chi2.cdf(chi_square, degrees_of_freedom)
            
            combined['significance'] = {
                'fisher_chi_square': chi_square,
                'degrees_of_freedom': degrees_of_freedom,
                'combined_p_value': combined_p_value,
                'significant': combined_p_value < 0.05
            }
        
        # Calculate overall phi dominance
        phi_count = 0
        total_count = 0
        
        for test_name, best_constant in combined['best_constants'].items():
            total_count += 1
            if best_constant['constant'] == 'phi':
                phi_count += 1
        
        if total_count > 0:
            combined['phi_dominance'] = phi_count / float(total_count)
        else:
            combined['phi_dominance'] = 0.0
        
        # Store combined results
        self.results['combined'] = combined
    
    def generate_report(self):
        """
        Generate a comprehensive report of all test results.
        
        Returns:
            str: Report text
        """
        if not self.results:
            print("No results to report. Run tests first.")
            return ""
        
        report = []
        report.append("=" * 80)
        report.append("COSMIC CONSCIOUSNESS ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")
        report.append("Analysis completed on: {}".format(self.results['timestamp']))
        report.append("")
        
        # Combined significance
        report.append("COMBINED SIGNIFICANCE")
        report.append("-" * 50)
        
        if 'significance' in self.results['combined']:
            significance = self.results['combined']['significance']
            report.append("Fisher's method:")
            report.append("  Chi-square: {:.4f}".format(significance['fisher_chi_square']))
            report.append("  Degrees of freedom: {}".format(significance['degrees_of_freedom']))
            report.append("  Combined p-value: {:.6f}".format(significance['combined_p_value']))
            report.append("  Significant: {}".format(significance['significant']))
            report.append("")
        
        # Phi dominance
        if 'phi_dominance' in self.results['combined']:
            phi_dominance = self.results['combined']['phi_dominance']
            report.append("Phi dominance: {:.2f}%".format(phi_dominance * 100))
            report.append("")
        
        # Individual test results
        report.append("INDIVIDUAL TEST RESULTS")
        report.append("-" * 50)
        
        for test_name, test_results in self.results['tests'].items():
            report.append("\nTest: {}".format(test_name))
            
            # Phi optimality
            if 'phi_optimality' in test_results:
                # Handle the new nested dictionary structure
                if isinstance(test_results['phi_optimality'], dict):
                    # If it's a dictionary of dictionaries (from our fix)
                    if all(isinstance(v, dict) for v in test_results['phi_optimality'].values()):
                        # Calculate average phi optimality across all ratios
                        phi_values = []
                        for ratio_dict in test_results['phi_optimality'].values():
                            if 'phi' in ratio_dict:
                                phi_values.append(ratio_dict['phi'])
                        
                        if phi_values:
                            avg_phi_optimality = np.mean(phi_values)
                            report.append("  Avg Phi optimality: {:.6f}".format(avg_phi_optimality))
                    # If it's a simple dictionary with constants as keys
                    elif 'phi' in test_results['phi_optimality']:
                        report.append("  Phi optimality: {:.6f}".format(test_results['phi_optimality']['phi']))
                else:
                    # Original case - direct float value
                    report.append("  Phi optimality: {:.6f}".format(test_results['phi_optimality']))
            
            # P-value
            if 'p_value' in test_results:
                report.append("  P-value: {:.6f}".format(test_results['p_value']))
                report.append("  Significant: {}".format(test_results['p_value'] < 0.05))
            
            # Best constant
            if 'best_constant' in test_results and 'best_value' in test_results:
                # Handle the new nested dictionary structure
                if isinstance(test_results['best_value'], dict):
                    # If it's a dictionary with constants as keys
                    if 'phi' in test_results['best_value']:
                        report.append("  Best constant: {} ({:.6f})".format(
                            test_results['best_constant'], test_results['best_value']['phi']))
                else:
                    # Original case - direct float value
                    report.append("  Best constant: {} ({:.6f})".format(
                        test_results['best_constant'], test_results['best_value']))
            
            # Confidence interval
            if 'confidence_interval' in test_results:
                ci = test_results['confidence_interval']
                report.append("  95% Confidence interval: [{:.6f}, {:.6f}]".format(ci[0], ci[1]))
            
            # Additional metrics
            if 'metrics' in test_results:
                report.append("  Additional metrics:")
                for metric_name, metric_value in test_results['metrics'].items():
                    if isinstance(metric_value, float):
                        report.append("    {}: {:.6f}".format(metric_name, metric_value))
                    else:
                        report.append("    {}: {}".format(metric_name, metric_value))
        
        # Write report to file
        report_path = os.path.join(self.output_dir, "report.txt")
        with open(report_path, 'w') as f:
            f.write("\n".join(report))
        
        print("Report saved to {}".format(report_path))
        return "\n".join(report)
    
    def visualize_results(self):
        """
        Generate visualizations of the results.
        """
        if not self.results:
            print("No results to visualize. Run tests first.")
            return
        
        # Create visualizations directory
        vis_dir = os.path.join(self.output_dir, "visualizations")
        if not os.path.exists(vis_dir):
            os.makedirs(vis_dir)
        
        # Generate individual test visualizations
        for test_name, test_results in self.results['tests'].items():
            # Get test object
            test = next((t for t in self.tests if t.name == test_name), None)
            if test is None:
                continue
            
            # Generate visualization
            try:
                test.visualize(os.path.join(vis_dir, "{}.png".format(test_name.replace(" ", "_").lower())))
            except Exception as e:
                print("Error visualizing {}: {}".format(test_name, e))
        
        # Generate combined visualization
        self._visualize_combined_results(vis_dir)
    
    def _visualize_combined_results(self, output_dir):
        """
        Generate a combined visualization of all test results.
        
        Args:
            output_dir (str): Directory to save visualization
        """
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot phi optimality
        ax = axes[0, 0]
        phi_optimality = self.results['combined']['phi_optimality']
        test_names = list(phi_optimality.keys())
        
        # Extract values, handling the new dictionary structure
        values = []
        for test_name, phi_value in phi_optimality.items():
            if isinstance(phi_value, dict):
                # If it's a dictionary of dictionaries (from our fix)
                if all(isinstance(v, dict) for v in phi_value.values()):
                    # Calculate average phi optimality across all ratios
                    phi_values = [opt_dict['phi'] for opt_dict in phi_value.values() if 'phi' in opt_dict]
                    if phi_values:
                        values.append(np.mean(phi_values))
                    else:
                        values.append(0)  # Default if no phi values found
                # If it's a simple dictionary with 'phi' key
                elif 'phi' in phi_value:
                    values.append(phi_value['phi'])
                else:
                    values.append(0)  # Default if no phi value found
            else:
                # Original case - direct float value
                values.append(phi_value)
        
        ax.bar(range(len(test_names)), values, color='gold')
        ax.set_xticks(range(len(test_names)))
        ax.set_xticklabels(test_names, rotation=45, ha='right')
        ax.set_ylabel('Phi Optimality')
        ax.set_title('Phi Optimality by Test')
        ax.grid(True, alpha=0.3)
        
        # Plot p-values
        ax = axes[0, 1]
        p_values = self.results['combined']['p_values']
        test_names = list(p_values.keys())
        values = list(p_values.values())
        
        ax.bar(range(len(test_names)), values, color='blue')
        ax.set_xticks(range(len(test_names)))
        ax.set_xticklabels(test_names, rotation=45, ha='right')
        ax.set_ylabel('P-value')
        ax.set_title('P-values by Test')
        ax.axhline(0.05, color='r', linestyle='--', label='p=0.05')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot best constants
        ax = axes[1, 0]
        best_constants = self.results['combined']['best_constants']
        test_names = list(best_constants.keys())
        constants = [best_constants[name]['constant'] for name in test_names]
        
        # Count occurrences of each constant
        constant_counts = {}
        for constant in constants:
            if constant in constant_counts:
                constant_counts[constant] += 1
            else:
                constant_counts[constant] = 1
        
        # Plot pie chart
        constant_names = list(constant_counts.keys())
        constant_values = list(constant_counts.values())
        
        ax.pie(constant_values, labels=constant_names, autopct='%1.1f%%',
               colors=['gold' if c == 'phi' else 'gray' for c in constant_names])
        ax.set_title('Best Constants Distribution')
        
        # Plot combined significance
        ax = axes[1, 1]
        if 'significance' in self.results['combined']:
            significance = self.results['combined']['significance']
            
            # Create text box
            textstr = '\n'.join((
                'Fisher\'s Method:',
                'Chi-square: {:.4f}'.format(significance['fisher_chi_square']),
                'Degrees of freedom: {}'.format(significance['degrees_of_freedom']),
                'Combined p-value: {:.6f}'.format(significance['combined_p_value']),
                'Significant: {}'.format(significance['significant'])
            ))
            
            # Place text box
            ax.text(0.5, 0.5, textstr, transform=ax.transAxes,
                    fontsize=12, verticalalignment='center',
                    horizontalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            ax.set_title('Combined Significance')
        
        # Remove axis ticks
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(output_dir, "combined_results.png"), dpi=300)
        plt.close()
    
    def save_results(self):
        """
        Save results to a JSON file.
        """
        if not self.results:
            print("No results to save. Run tests first.")
            return
        
        # Create a deep copy of results and convert all non-serializable objects
        import copy
        
        def make_serializable(obj):
            """Recursively convert all non-serializable objects to serializable ones."""
            if isinstance(obj, dict):
                return {str(k): make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(item) for item in obj]
            elif isinstance(obj, tuple):
                return str(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int_, np.integer)):
                return int(obj)
            elif isinstance(obj, (np.float_, np.floating)):
                return float(obj)
            elif obj is None or isinstance(obj, (str, int, float, bool)):
                return obj
            else:
                return str(obj)
        
        # Create a serializable copy of the results
        results_copy = copy.deepcopy(self.results)
        
        # Remove data arrays which are not needed in the JSON
        for test_name in results_copy['tests']:
            if 'data' in results_copy['tests'][test_name]:
                del results_copy['tests'][test_name]['data']
            if 'surrogate_data' in results_copy['tests'][test_name]:
                del results_copy['tests'][test_name]['surrogate_data']
        
        # Convert to serializable format
        results_json = make_serializable(results_copy)
        
        # Save results
        results_path = os.path.join(self.output_dir, "results.json")
        with open(results_path, 'w') as f:
            json.dump(results_json, f, indent=2)
        
        print("Results saved to {}".format(results_path))
    
    def _prepare_results_for_json(self):
        """
        Prepare results for JSON serialization.
        
        Returns:
            dict: JSON-serializable results
        """
        # Create a deep copy of results
        results_json = {
            'tests': {},
            'combined': {},
            'timestamp': self.results['timestamp']
        }
        
        # Helper function to convert non-serializable objects to serializable ones
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, tuple):
                return str(obj)
            elif isinstance(obj, bool):
                return bool(obj)  # Ensure booleans are properly converted
            elif isinstance(obj, dict):
                return {str(k) if not isinstance(k, str) else k: convert_to_serializable(v) 
                        for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj
        
        # Process test results
        for test_name, test_results in self.results['tests'].items():
            results_json['tests'][test_name] = {}
            
            for key, value in test_results.items():
                # Skip data arrays
                if key in ['data', 'surrogate_data']:
                    continue
                
                # Convert to serializable
                results_json['tests'][test_name][key] = convert_to_serializable(value)
        
        # Process combined results
        for key, value in self.results['combined'].items():
            results_json['combined'][key] = convert_to_serializable(value)
        
        return results_json


def main():
    """
    Run a sample analysis.
    """
    from core_framework.data_handler import load_cmb_data
    from tests.coherence_tests.meta_coherence_test import MetaCoherenceTest
    from tests.structural_tests.golden_ratio_test import GoldenRatioTest
    
    # Load data
    data = load_cmb_data(simulated=True, seed=42)
    
    # Create analyzer
    analyzer = CosmicConsciousnessAnalyzer()
    
    # Add tests
    analyzer.add_test(MetaCoherenceTest(data=data))
    analyzer.add_test(GoldenRatioTest(data=data))
    
    # Run tests
    analyzer.run_all_tests_parallel(data=data, n_jobs=-1)
    
    # Generate report
    analyzer.generate_report()
    
    # Visualize results
    analyzer.visualize_results()
    
    # Save results
    analyzer.save_results()


if __name__ == "__main__":
    main()
