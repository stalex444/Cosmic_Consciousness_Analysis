#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualization Module
-------------------
Provides standardized plotting and visualization capabilities for cosmic tests.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from core_framework.constants import COLORS, CONSTANT_NAMES


def setup_figure(nrows=1, ncols=1, figsize=None, title=None, tight_layout=True):
    """
    Set up a figure for plotting.
    
    Args:
        nrows (int, optional): Number of rows. Defaults to 1.
        ncols (int, optional): Number of columns. Defaults to 1.
        figsize (tuple, optional): Figure size. Defaults to None.
        title (str, optional): Figure title. Defaults to None.
        tight_layout (bool, optional): Whether to use tight layout. Defaults to True.
        
    Returns:
        tuple: Figure and axes objects
    """
    if figsize is None:
        figsize = (6 * ncols, 5 * nrows)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    
    if title:
        fig.suptitle(title, fontsize=16)
    
    if tight_layout:
        fig.tight_layout(rect=[0, 0, 1, 0.95] if title else None)
    
    return fig, axes


def create_multi_panel_figure(results, plot_functions, title=None, filename=None, figsize=(12, 10)):
    """
    Create a multi-panel figure using the provided plot functions.
    
    Args:
        results (dict): Results dictionary
        plot_functions (list): List of plotting functions
        title (str, optional): Figure title. Defaults to None.
        filename (str, optional): Filename to save figure. Defaults to None.
        figsize (tuple, optional): Figure size. Defaults to (12, 10).
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    n_panels = len(plot_functions)
    
    if n_panels <= 2:
        nrows, ncols = 1, n_panels
    elif n_panels <= 4:
        nrows, ncols = 2, 2
    elif n_panels <= 6:
        nrows, ncols = 2, 3
    elif n_panels <= 9:
        nrows, ncols = 3, 3
    else:
        nrows = int(np.ceil(np.sqrt(n_panels)))
        ncols = int(np.ceil(n_panels / nrows))
    
    fig = plt.figure(figsize=figsize)
    
    if title:
        fig.suptitle(title, fontsize=16, y=0.98)
    
    # Create grid layout
    gs = gridspec.GridSpec(nrows, ncols, figure=fig)
    
    # Call each plotting function with its own subplot
    for i, plot_func in enumerate(plot_functions):
        if i < n_panels:
            ax = fig.add_subplot(gs[i // ncols, i % ncols])
            plot_func(ax, results)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95] if title else None)
    
    if filename:
        # Create figures directory if it doesn't exist
        figures_dir = os.path.join(os.getcwd(), "figures")
        if not os.path.exists(figures_dir):
            os.makedirs(figures_dir)
        
        filepath = os.path.join(figures_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print("Figure saved to {}".format(filepath))
    
    return fig


def plot_optimization_by_scale(ax, results, constants=None, log_scale=True):
    """
    Plot optimization by scale for each constant.
    
    Args:
        ax (matplotlib.axes.Axes): Axes to plot on
        results (dict): Results dictionary
        constants (list, optional): List of constants to plot. Defaults to None.
        log_scale (bool, optional): Whether to use log scale for x-axis. Defaults to True.
    """
    scales = results.get('scales', [])
    scale_results = results.get('scale_results', {})
    
    if not scales or not scale_results:
        ax.text(0.5, 0.5, "No scale results available", ha='center', va='center')
        ax.set_title("Optimization by Scale")
        return
    
    if constants is None:
        constants = list(COLORS.keys())
    
    for constant in constants:
        optimality = []
        for scale in scales:
            if scale in scale_results:
                # Average optimality across all metrics
                opt_values = []
                for metric in scale_results[scale]:
                    if isinstance(scale_results[scale][metric], dict) and 'optimality' in scale_results[scale][metric]:
                        if constant in scale_results[scale][metric]['optimality']:
                            opt_values.append(scale_results[scale][metric]['optimality'][constant])
                
                if opt_values:
                    optimality.append(np.mean(opt_values))
                else:
                    optimality.append(0)
            else:
                optimality.append(0)
        
        ax.plot(scales, optimality, 'o-', color=COLORS.get(constant, 'gray'), 
                label=CONSTANT_NAMES.get(constant, constant))
    
    ax.set_xlabel('Scale')
    ax.set_ylabel('Optimization')
    ax.set_title('Optimization by Scale')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    if log_scale:
        ax.set_xscale('log')


def plot_best_constants(ax, results, metrics=None):
    """
    Plot best constant by scale for each metric.
    
    Args:
        ax (matplotlib.axes.Axes): Axes to plot on
        results (dict): Results dictionary
        metrics (list, optional): List of metrics to plot. Defaults to None.
    """
    scales = results.get('scales', [])
    scale_results = results.get('scale_results', {})
    
    if not scales or not scale_results:
        ax.text(0.5, 0.5, "No scale results available", ha='center', va='center')
        ax.set_title("Best Constants by Scale")
        return
    
    if metrics is None:
        # Find all metrics in the results
        metrics = set()
        for scale in scale_results:
            metrics.update(scale_results[scale].keys())
        metrics = sorted(metrics)
    
    # Create a colormap for constants
    constants = list(COLORS.keys())
    constant_indices = {const: i for i, const in enumerate(constants)}
    
    # Plot best constant for each metric and scale
    for i, metric in enumerate(metrics):
        best_constants = []
        valid_scales = []
        
        for scale in scales:
            if scale in scale_results and metric in scale_results[scale]:
                if 'best_constant' in scale_results[scale][metric]:
                    best_const = scale_results[scale][metric]['best_constant']
                    best_constants.append(constant_indices.get(best_const, -1))
                    valid_scales.append(scale)
        
        if valid_scales:
            ax.scatter(valid_scales, [i] * len(valid_scales), c=[COLORS.get(constants[idx], 'gray') if idx >= 0 else 'gray' 
                                                               for idx in best_constants], 
                      s=100, label=metric)
    
    # Set y-ticks to metric names
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels(metrics)
    
    # Create a custom legend for constants
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS.get(const, 'gray'), 
                             markersize=10, label=CONSTANT_NAMES.get(const, const)) for const in constants]
    ax.legend(handles=legend_elements, loc='best')
    
    ax.set_xlabel('Scale')
    ax.set_title('Best Constants by Scale and Metric')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)


def plot_transition_boundaries(ax, results, metric=None, min_sharpness=0.1):
    """
    Plot transition boundaries where the dominant constant changes.
    
    Args:
        ax (matplotlib.axes.Axes): Axes to plot on
        results (dict): Results dictionary
        metric (str, optional): Specific metric to plot. Defaults to None (all metrics).
        min_sharpness (float, optional): Minimum sharpness to consider. Defaults to 0.1.
    """
    transitions = results.get('transitions', [])
    
    if not transitions:
        ax.text(0.5, 0.5, "No transitions detected", ha='center', va='center')
        ax.set_title("Transition Boundaries")
        return
    
    # Filter transitions by metric if specified
    if metric is not None:
        transitions = [t for t in transitions if t.get('metric') == metric]
    
    # Filter by sharpness
    transitions = [t for t in transitions if t.get('sharpness', 0) >= min_sharpness]
    
    if not transitions:
        ax.text(0.5, 0.5, "No significant transitions detected", ha='center', va='center')
        ax.set_title("Transition Boundaries")
        return
    
    # Sort transitions by scale
    transitions = sorted(transitions, key=lambda t: t.get('scale', 0))
    
    # Plot transitions
    for t in transitions:
        scale = t.get('scale', 0)
        from_const = t.get('from_constant', '')
        to_const = t.get('to_constant', '')
        sharpness = t.get('sharpness', 0)
        metric_name = t.get('metric', '')
        
        # Plot vertical line at transition scale
        ax.axvline(scale, color='gray', alpha=0.5, linestyle='--')
        
        # Add annotation
        if from_const in COLORS and to_const in COLORS:
            arrow_props = dict(arrowstyle='->', color='black')
            ax.annotate('{} → {}'.format(from_const, to_const), 
                       xy=(scale, 0.5), 
                       xytext=(scale, 0.5 + 0.1 * (transitions.index(t) % 5)), 
                       arrowprops=arrow_props,
                       rotation=90,
                       ha='center', va='bottom')
    
    # Set axis labels and title
    ax.set_xlabel('Scale')
    ax.set_ylabel('Transition Sharpness')
    ax.set_title('Scale Transition Boundaries' + (' for ' + metric if metric else ''))
    ax.set_xscale('log')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)


def plot_phi_optimality(ax, results, metrics=None, log_scale=True):
    """
    Plot phi optimality across scales.
    
    Args:
        ax (matplotlib.axes.Axes): Axes to plot on
        results (dict): Results dictionary
        metrics (list, optional): List of metrics to plot. Defaults to None.
        log_scale (bool, optional): Whether to use log scale for x-axis. Defaults to True.
    """
    scales = results.get('scales', [])
    scale_results = results.get('scale_results', {})
    
    if not scales or not scale_results:
        ax.text(0.5, 0.5, "No scale results available", ha='center', va='center')
        ax.set_title("Phi Optimality Across Scales")
        return
    
    if metrics is None:
        # Find all metrics in the results
        metrics = set()
        for scale in scale_results:
            metrics.update(scale_results[scale].keys())
        metrics = sorted(metrics)
    
    # Plot phi optimality for each metric
    for metric in metrics:
        phi_optimality = []
        valid_scales = []
        
        for scale in scales:
            if scale in scale_results and metric in scale_results[scale]:
                if 'optimality' in scale_results[scale][metric] and 'phi' in scale_results[scale][metric]['optimality']:
                    phi_optimality.append(scale_results[scale][metric]['optimality']['phi'])
                    valid_scales.append(scale)
        
        if valid_scales:
            ax.plot(valid_scales, phi_optimality, 'o-', label=metric)
    
    # Add reference line for random expectation
    ax.axhline(y=1/len(COLORS), color='gray', linestyle='--', label='Random')
    
    ax.set_xlabel('Scale')
    ax.set_ylabel('Phi Optimality')
    ax.set_title('Golden Ratio Optimality Across Scales')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    if log_scale:
        ax.set_xscale('log')


def plot_significance_by_constant(ax, results, metrics=None):
    """
    Plot statistical significance for each constant.
    
    Args:
        ax (matplotlib.axes.Axes): Axes to plot on
        results (dict): Results dictionary
        metrics (list, optional): List of metrics to plot. Defaults to None.
    """
    significance = results.get('significance', {})
    
    if not significance:
        ax.text(0.5, 0.5, "No significance results available", ha='center', va='center')
        ax.set_title("Statistical Significance by Constant")
        return
    
    if metrics is None:
        # Find all metrics in the results
        metrics = set()
        for const in significance:
            metrics.update(significance[const].keys())
        metrics = sorted(metrics)
    
    constants = list(COLORS.keys())
    x = np.arange(len(constants))
    width = 0.8 / len(metrics)
    
    for i, metric in enumerate(metrics):
        p_values = []
        
        for const in constants:
            if const in significance and metric in significance[const]:
                p_value = significance[const][metric].get('p_value', 1.0)
                # Convert p-value to -log10(p) for better visualization
                if p_value > 0:
                    p_values.append(-np.log10(p_value))
                else:
                    p_values.append(16)  # Cap at -log10(1e-16)
            else:
                p_values.append(0)
        
        ax.bar(x + i * width - 0.4, p_values, width, label=metric)
    
    # Add reference lines for significance levels
    ax.axhline(y=-np.log10(0.05), color='gray', linestyle='--', label='p=0.05')
    ax.axhline(y=-np.log10(0.01), color='gray', linestyle=':', label='p=0.01')
    ax.axhline(y=-np.log10(0.001), color='gray', linestyle='-.', label='p=0.001')
    
    ax.set_xticks(x)
    ax.set_xticklabels([CONSTANT_NAMES.get(const, const) for const in constants])
    ax.set_ylabel('-log10(p-value)')
    ax.set_title('Statistical Significance by Constant')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)


def plot_scale_dominance(ax, results, constant='phi', metrics=None):
    """
    Plot dominance of a specific constant across scales.
    
    Args:
        ax (matplotlib.axes.Axes): Axes to plot on
        results (dict): Results dictionary
        constant (str, optional): Constant to analyze. Defaults to 'phi'.
        metrics (list, optional): List of metrics to consider. Defaults to None.
    """
    scales = results.get('scales', [])
    scale_results = results.get('scale_results', {})
    
    if not scales or not scale_results:
        ax.text(0.5, 0.5, "No scale results available", ha='center', va='center')
        ax.set_title("{} Dominance Across Scales".format(CONSTANT_NAMES.get(constant, constant)))
        return
    
    if metrics is None:
        # Find all metrics in the results
        metrics = set()
        for scale in scale_results:
            metrics.update(scale_results[scale].keys())
        metrics = list(metrics)
    
    # Calculate dominance at each scale
    dominance = []
    
    for scale in scales:
        if scale in scale_results:
            count = 0
            total = 0
            
            for metric in metrics:
                if metric in scale_results[scale] and 'best_constant' in scale_results[scale][metric]:
                    if scale_results[scale][metric]['best_constant'] == constant:
                        count += 1
                    total += 1
            
            if total > 0:
                dominance.append((scale, count / float(total)))
    
    if not dominance:
        ax.text(0.5, 0.5, "No dominance data available", ha='center', va='center')
        ax.set_title("{} Dominance Across Scales".format(CONSTANT_NAMES.get(constant, constant)))
        return
    
    # Plot dominance
    scales_vals, dominance_vals = zip(*dominance)
    
    ax.bar(range(len(scales_vals)), dominance_vals, color=COLORS.get(constant, 'blue'))
    ax.set_xticks(range(len(scales_vals)))
    ax.set_xticklabels([str(s) for s in scales_vals], rotation=45)
    ax.set_xlabel('Scale')
    ax.set_ylabel('Dominance (fraction of metrics)')
    ax.set_title('{} Dominance Across Scales'.format(CONSTANT_NAMES.get(constant, constant)))
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='50% Threshold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)


def plot_metric_comparison(ax, results, metric1, metric2, color_by='scale'):
    """
    Create a scatter plot comparing two metrics.
    
    Args:
        ax (matplotlib.axes.Axes): Axes to plot on
        results (dict): Results dictionary
        metric1 (str): First metric to compare
        metric2 (str): Second metric to compare
        color_by (str, optional): How to color points. Defaults to 'scale'.
    """
    scales = results.get('scales', [])
    scale_results = results.get('scale_results', {})
    
    if not scales or not scale_results:
        ax.text(0.5, 0.5, "No scale results available", ha='center', va='center')
        ax.set_title("Metric Comparison: {} vs {}".format(metric1, metric2))
        return
    
    # Extract values for both metrics
    x_values = []
    y_values = []
    colors = []
    
    for scale in scales:
        if scale in scale_results:
            if metric1 in scale_results[scale] and metric2 in scale_results[scale]:
                if 'value' in scale_results[scale][metric1] and 'value' in scale_results[scale][metric2]:
                    x_values.append(scale_results[scale][metric1]['value'])
                    y_values.append(scale_results[scale][metric2]['value'])
                    
                    if color_by == 'scale':
                        colors.append(np.log10(scale) / np.log10(max(scales)))
                    elif color_by == 'best_constant':
                        best_const1 = scale_results[scale][metric1].get('best_constant', '')
                        best_const2 = scale_results[scale][metric2].get('best_constant', '')
                        
                        if best_const1 == best_const2:
                            colors.append(COLORS.get(best_const1, 'gray'))
                        else:
                            colors.append('gray')
    
    if not x_values or not y_values:
        ax.text(0.5, 0.5, "No comparison data available", ha='center', va='center')
        ax.set_title("Metric Comparison: {} vs {}".format(metric1, metric2))
        return
    
    # Create scatter plot
    if color_by == 'scale':
        scatter = ax.scatter(x_values, y_values, c=colors, cmap='viridis', s=50, alpha=0.7)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Log Scale')
    else:
        ax.scatter(x_values, y_values, c=colors, s=50, alpha=0.7)
    
    # Add correlation coefficient
    corr = np.corrcoef(x_values, y_values)[0, 1]
    ax.text(0.05, 0.95, 'r = {:.2f}'.format(corr), transform=ax.transAxes, 
           ha='left', va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    ax.set_xlabel(metric1)
    ax.set_ylabel(metric2)
    ax.set_title("Metric Comparison: {} vs {}".format(metric1, metric2))
    ax.grid(True, alpha=0.3)


def plot_power_spectrum(ax, data, fs=1.0, method='welch', nperseg=256, scale='log'):
    """
    Plot the power spectrum of a time series.
    
    Args:
        ax (matplotlib.axes.Axes): Axes to plot on
        data (ndarray): Input time series
        fs (float, optional): Sampling frequency. Defaults to 1.0.
        method (str, optional): Method for PSD estimation. Defaults to 'welch'.
        nperseg (int, optional): Length of each segment. Defaults to 256.
        scale (str, optional): Scale for y-axis. Defaults to 'log'.
    """
    from scipy import signal
    
    if method == 'welch':
        # Calculate PSD using Welch's method
        f, psd = signal.welch(data, fs=fs, nperseg=min(nperseg, len(data)//2))
    elif method == 'periodogram':
        # Calculate PSD using periodogram
        f, psd = signal.periodogram(data, fs=fs)
    else:
        raise ValueError("Unknown method: {}".format(method))
    
    # Plot power spectrum
    ax.plot(f, psd)
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Power Spectral Density')
    ax.set_title('Power Spectrum')
    ax.grid(True, alpha=0.3)
    
    if scale == 'log':
        ax.set_yscale('log')
    
    # Set x-axis to log scale if there are enough points
    if len(f) > 10 and f[1] > 0:
        ax.set_xscale('log')


def plot_heatmap(ax, data, x_labels=None, y_labels=None, title=None, cmap='viridis', 
                colorbar=True, colorbar_label=None):
    """
    Plot a heatmap.
    
    Args:
        ax (matplotlib.axes.Axes): Axes to plot on
        data (ndarray): 2D array of data
        x_labels (list, optional): Labels for x-axis. Defaults to None.
        y_labels (list, optional): Labels for y-axis. Defaults to None.
        title (str, optional): Title for the plot. Defaults to None.
        cmap (str, optional): Colormap. Defaults to 'viridis'.
        colorbar (bool, optional): Whether to add a colorbar. Defaults to True.
        colorbar_label (str, optional): Label for colorbar. Defaults to None.
    """
    im = ax.imshow(data, cmap=cmap, aspect='auto')
    
    if title:
        ax.set_title(title)
    
    if x_labels:
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=45, ha='right')
    
    if y_labels:
        ax.set_yticks(range(len(y_labels)))
        ax.set_yticklabels(y_labels)
    
    if colorbar:
        cbar = plt.colorbar(im, ax=ax)
        if colorbar_label:
            cbar.set_label(colorbar_label)


def save_figure(fig, filename, dpi=300, bbox_inches='tight'):
    """
    Save a figure to a file.
    
    Args:
        fig (matplotlib.figure.Figure): Figure to save
        filename (str): Filename
        dpi (int, optional): DPI. Defaults to 300.
        bbox_inches (str, optional): Bounding box. Defaults to 'tight'.
    """
    # Create figures directory if it doesn't exist
    figures_dir = os.path.join(os.getcwd(), "figures")
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)
    
    filepath = os.path.join(figures_dir, filename)
    fig.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches)
    print("Figure saved to {}".format(filepath))
    return filepath
