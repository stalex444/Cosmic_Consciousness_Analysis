#!/usr/bin/env python3
"""
Visualization utilities for Cosmic Consciousness Analysis.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from datetime import datetime

class Visualizer:
    """Class for creating visualizations of analysis results."""
    
    def __init__(self, output_dir="results"):
        """
        Initialize the visualizer.
        
        Parameters:
        -----------
        output_dir : str, optional
            Directory to save visualizations.
        """
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Set default plot style
        try:
            plt.style.use('seaborn-v0_8-whitegrid')
        except:
            try:
                plt.style.use('seaborn-whitegrid')  # For older versions of seaborn
            except:
                # Fallback to default style
                pass
        
        # Custom color map for significance
        self.significance_cmap = LinearSegmentedColormap.from_list(
            'significance', 
            [(0, 'blue'), (0.5, 'white'), (1, 'red')]
        )
    
    def plot_spectrum(self, ell, power, error=None, title="CMB Power Spectrum", 
                     highlight_points=None, highlight_color='red', 
                     filename="power_spectrum.png"):
        """
        Plot the CMB power spectrum.
        
        Parameters:
        -----------
        ell : array-like
            Multipole values.
        power : array-like
            Power spectrum values.
        error : array-like, optional
            Error bars for the power spectrum.
        title : str, optional
            Plot title.
        highlight_points : dict, optional
            Dictionary with keys 'ell' and 'power' for points to highlight.
        highlight_color : str, optional
            Color for highlighted points.
        filename : str, optional
            Output filename.
        """
        plt.figure(figsize=(12, 6))
        
        if error is not None:
            plt.errorbar(ell, power, yerr=error, fmt='o', markersize=2, 
                        alpha=0.3, elinewidth=0.5)
        else:
            plt.plot(ell, power, 'o', markersize=2, alpha=0.3)
        
        # Highlight specific points
        if highlight_points is not None:
            plt.scatter(highlight_points['ell'], highlight_points['power'], 
                       color=highlight_color, s=50, zorder=10)
        
        plt.xlabel('Multipole ℓ')
        plt.ylabel('EE Power (μK²)')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        
        # Save the figure
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_spectra_comparison(self, ell, power1, power2, title="Spectra Comparison", 
                               label1="Original", label2="Modified", 
                               filename="spectra_comparison.png"):
        """
        Plot a comparison of two power spectra.
        
        Parameters:
        -----------
        ell : array-like
            Multipole values.
        power1 : array-like
            First power spectrum.
        power2 : array-like
            Second power spectrum.
        title : str, optional
            Plot title.
        label1 : str, optional
            Label for the first spectrum.
        label2 : str, optional
            Label for the second spectrum.
        filename : str, optional
            Output filename.
        """
        plt.figure(figsize=(12, 6))
        
        plt.plot(ell, power1, 'o-', markersize=2, alpha=0.7, label=label1)
        plt.plot(ell, power2, 'o-', markersize=2, alpha=0.7, label=label2)
        
        plt.xlabel('Multipole ℓ')
        plt.ylabel('EE Power (μK²)')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save the figure
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_histogram(self, observed, random_distribution, title="Test Results", 
                      xlabel="Value", filename="histogram.png"):
        """
        Plot a histogram of random distribution with the observed value.
        
        Parameters:
        -----------
        observed : float
            Observed value.
        random_distribution : array-like
            Distribution of values under the null hypothesis.
        title : str, optional
            Plot title.
        xlabel : str, optional
            X-axis label.
        filename : str, optional
            Output filename.
        """
        plt.figure(figsize=(10, 6))
        
        # Plot histogram of random distribution
        plt.hist(random_distribution, bins=30, alpha=0.7, density=True, 
                color='skyblue', edgecolor='black', label='Random Distribution')
        
        # Plot observed value
        ylim = plt.ylim()
        plt.plot([observed, observed], [0, ylim[1]], 'r-', linewidth=2, 
                label=f'Observed Value: {observed:.4f}')
        
        # Calculate mean of random distribution
        random_mean = np.mean(random_distribution)
        plt.plot([random_mean, random_mean], [0, ylim[1]], 'g--', linewidth=2, 
                label=f'Random Mean: {random_mean:.4f}')
        
        plt.xlabel(xlabel)
        plt.ylabel('Probability Density')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save the figure
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_test_results(self, results, test_name="Test", filename=None):
        """
        Plot comprehensive results for a single test.
        
        Parameters:
        -----------
        results : dict
            Dictionary containing test results.
        test_name : str, optional
            Name of the test.
        filename : str, optional
            Output filename. If None, uses test_name.
        """
        if filename is None:
            filename = f"{test_name.lower().replace(' ', '_')}_results.png"
        
        # Extract results
        observed = results.get('observed', None)
        random_distribution = results.get('random', None)
        random_mean = results.get('random_mean', None)
        ratio = results.get('ratio', None)
        z_score = results.get('z_score', None)
        p_value = results.get('p_value', None)
        additional_metrics = results.get('additional_metrics', {})
        
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])
        
        # Plot histogram if random distribution is available
        if random_distribution is not None and len(random_distribution) > 0:
            ax1 = plt.subplot(gs[0, 0])
            
            # Plot histogram of random distribution
            ax1.hist(random_distribution, bins=30, alpha=0.7, density=True, 
                    color='skyblue', edgecolor='black', label='Random Distribution')
            
            # Plot observed value
            ylim = ax1.get_ylim()
            ax1.plot([observed, observed], [0, ylim[1]], 'r-', linewidth=2, 
                    label=f'Observed: {observed:.4f}')
            
            # Plot random mean
            if random_mean is not None:
                ax1.plot([random_mean, random_mean], [0, ylim[1]], 'g--', linewidth=2, 
                        label=f'Random Mean: {random_mean:.4f}')
            
            ax1.set_xlabel('Value')
            ax1.set_ylabel('Probability Density')
            ax1.set_title('Distribution of Random Values')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # Plot metrics
        ax2 = plt.subplot(gs[0, 1])
        
        metrics = []
        values = []
        
        if observed is not None:
            metrics.append('Observed')
            values.append(observed)
        
        if random_mean is not None:
            metrics.append('Random Mean')
            values.append(random_mean)
        
        if ratio is not None:
            metrics.append('Ratio')
            values.append(ratio)
        
        if z_score is not None:
            metrics.append('Z-Score')
            values.append(z_score)
        
        if p_value is not None:
            metrics.append('P-Value')
            values.append(p_value)
        
        # Add additional metrics
        for metric, value in additional_metrics.items():
            metrics.append(metric)
            values.append(value)
        
        # Plot bar chart of metrics
        bars = ax2.bar(metrics, values, color='skyblue', edgecolor='black')
        
        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.4f}',
                    ha='center', va='bottom', rotation=0)
        
        ax2.set_ylabel('Value')
        ax2.set_title('Test Metrics')
        ax2.grid(True, alpha=0.3)
        
        # Add text summary
        ax3 = plt.subplot(gs[1, :])
        ax3.axis('off')
        
        summary_text = f"Test: {test_name}\n\n"
        
        if observed is not None and random_mean is not None:
            summary_text += f"Observed Value: {observed:.6f}\n"
            summary_text += f"Random Mean: {random_mean:.6f}\n"
        
        if ratio is not None:
            summary_text += f"Ratio: {ratio:.2f}x\n"
        
        if z_score is not None:
            summary_text += f"Z-Score: {z_score:.2f}\n"
        
        if p_value is not None:
            summary_text += f"P-Value: {p_value:.6f}\n"
            
            # Add significance interpretation
            if p_value < 0.001:
                significance = "Highly Significant"
            elif p_value < 0.01:
                significance = "Very Significant"
            elif p_value < 0.05:
                significance = "Significant"
            elif p_value < 0.1:
                significance = "Marginally Significant"
            else:
                significance = "Not Significant"
                
            summary_text += f"Significance: {significance}\n\n"
            
            # Add phi-optimality if available
            if observed is not None and random_mean is not None:
                # Calculate phi-optimality
                ratio = observed / random_mean if random_mean != 0 else float('inf')
                
                if ratio > 1:
                    # Positive phi-optimality (better than random)
                    phi_optimality = 2 * (1 - 1/ratio) if ratio <= 2 else 1.0
                else:
                    # Negative phi-optimality (worse than random)
                    phi_optimality = -2 * (1 - ratio) if ratio >= 0.5 else -1.0
                    
                summary_text += f"\nPhi-Optimality: {phi_optimality:.4f}\n"
                
                # Add interpretation
                if phi_optimality >= 0.8:
                    interpretation = "Extremely High"
                elif phi_optimality >= 0.6:
                    interpretation = "Very High"
                elif phi_optimality >= 0.4:
                    interpretation = "High"
                elif phi_optimality >= 0.2:
                    interpretation = "Moderate"
                elif phi_optimality > 0:
                    interpretation = "Slight"
                elif phi_optimality == 0:
                    interpretation = "Neutral"
                elif phi_optimality > -0.2:
                    interpretation = "Slightly Negative"
                elif phi_optimality > -0.4:
                    interpretation = "Moderately Negative"
                elif phi_optimality > -0.6:
                    interpretation = "Negative"
                else:
                    interpretation = "Strongly Negative"
                    
                summary_text += f"Interpretation: {interpretation}\n"
        
        # Add timestamp
        summary_text += f"\nAnalysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        ax3.text(0.5, 0.5, summary_text, ha='center', va='center', fontsize=12)
        
        # Add title
        plt.suptitle(f"{test_name} Results", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        
        # Save the figure
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_comprehensive_results(self, test_results, sorted_tests, 
                                 filename="comprehensive_analysis_results.png"):
        """
        Plot comprehensive results for all tests.
        
        Parameters:
        -----------
        test_results : dict
            Dictionary mapping test names to test results.
        sorted_tests : list
            List of (test_name, phi_optimality) tuples, sorted by phi_optimality.
        filename : str, optional
            Output filename.
        """
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(2, 2)
        
        # Plot phi-optimality for all tests
        ax1 = plt.subplot(gs[0, 0])
        
        test_names = [test_name.split('.')[-1].replace('_', ' ').title() for test_name, _ in sorted_tests]
        phi_optimalities = [phi_opt for _, phi_opt in sorted_tests]
        
        # Create colormap based on phi-optimality
        colors = [self.significance_cmap((phi_opt + 1) / 2) for phi_opt in phi_optimalities]
        
        # Plot bar chart
        bars = ax1.barh(test_names, phi_optimalities, color=colors)
        
        # Add value labels
        for bar, value in zip(bars, phi_optimalities):
            width = bar.get_width()
            ax1.text(width, bar.get_y() + bar.get_height()/2.,
                    f'{value:.2f}',
                    ha='left', va='center')
        
        ax1.set_xlabel('Phi-Optimality')
        ax1.set_title('Tests Ranked by Phi-Optimality')
        ax1.grid(True, alpha=0.3)
        
        # Plot p-values for all tests
        ax2 = plt.subplot(gs[0, 1])
        
        p_values = []
        for test_name, _ in sorted_tests:
            p_value = test_results[test_name]['p_value']
            p_values.append(p_value)
        
        # Create colormap based on p-value (reversed)
        colors = [self.significance_cmap(1 - min(p, 0.1) * 10) for p in p_values]
        
        # Plot bar chart
        bars = ax2.barh(test_names, [-np.log10(p) for p in p_values], color=colors)
        
        # Add value labels
        for bar, p in zip(bars, p_values):
            width = bar.get_width()
            ax2.text(width, bar.get_y() + bar.get_height()/2.,
                    f'{p:.4f}',
                    ha='left', va='center')
        
        ax2.set_xlabel('-log10(p-value)')
        ax2.set_title('Tests Ranked by Statistical Significance')
        ax2.grid(True, alpha=0.3)
        
        # Plot combined results
        ax3 = plt.subplot(gs[1, :])
        
        # Calculate combined p-value using Fisher's method
        from scipy import stats
        
        valid_p = [p for p in p_values if not np.isnan(p) and p > 0]
        
        if valid_p:
            chi2 = -2 * np.sum(np.log(valid_p))
            df = 2 * len(valid_p)
            combined_p = 1 - stats.chi2.cdf(chi2, df)
            
            # Calculate mean phi-optimality
            mean_phi_optimality = np.mean(phi_optimalities)
            
            # Create text summary
            summary_text = "COMPREHENSIVE ANALYSIS SUMMARY\n\n"
            summary_text += f"Combined p-value (Fisher's method): {combined_p:.6f}\n"
            summary_text += f"Mean phi-optimality: {mean_phi_optimality:.4f}\n\n"
            
            # Determine overall significance
            if combined_p < 0.001:
                significance = "Highly Significant"
            elif combined_p < 0.01:
                significance = "Very Significant"
            elif combined_p < 0.05:
                significance = "Significant"
            elif combined_p < 0.1:
                significance = "Marginally Significant"
            else:
                significance = "Not Significant"
                
            summary_text += f"Overall Significance: {significance}\n\n"
            
            # Interpret phi-optimality
            if mean_phi_optimality >= 0.8:
                interpretation = "Extremely High"
            elif mean_phi_optimality >= 0.6:
                interpretation = "Very High"
            elif mean_phi_optimality >= 0.4:
                interpretation = "High"
            elif mean_phi_optimality >= 0.2:
                interpretation = "Moderate"
            elif mean_phi_optimality > 0:
                interpretation = "Slight"
            elif mean_phi_optimality == 0:
                interpretation = "Neutral"
            elif mean_phi_optimality > -0.2:
                interpretation = "Slightly Negative"
            elif mean_phi_optimality > -0.4:
                interpretation = "Moderately Negative"
            elif mean_phi_optimality > -0.6:
                interpretation = "Negative"
            else:
                interpretation = "Strongly Negative"
                
            summary_text += f"Phi-Optimality Interpretation: {interpretation}\n\n"
            
            # Add top 3 tests
            summary_text += "Top 3 Tests:\n"
            for i in range(min(3, len(sorted_tests))):
                test_name, phi_opt = sorted_tests[i]
                p_value = test_results[test_name]['p_value']
                formatted_name = test_name.split('.')[-1].replace('_', ' ').title()
                summary_text += f"{i+1}. {formatted_name}: φ-opt = {phi_opt:.2f}, p = {p_value:.4f}\n"
            
            # Add timestamp
            summary_text += f"\nAnalysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            
            ax3.axis('off')
            ax3.text(0.5, 0.5, summary_text, ha='center', va='center', fontsize=12)
        
        # Add title
        plt.suptitle("Comprehensive Analysis Results", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        
        # Save the figure
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()

# Convenience function to get a visualizer instance
def get_visualizer(output_dir="results"):
    """
    Get a configured visualizer instance.
    
    Parameters:
    -----------
    output_dir : str, optional
        Directory to save visualizations.
        
    Returns:
    --------
    Visualizer
        Configured visualizer instance.
    """
    return Visualizer(output_dir=output_dir)
