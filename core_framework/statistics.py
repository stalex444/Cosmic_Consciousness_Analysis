#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Statistics Module
----------------
Provides common statistical functions for significance testing and effect size calculation.
"""

import numpy as np
from scipy import stats
from core_framework.constants import DEFAULT_BOOTSTRAP_SAMPLES, DEFAULT_ALPHA


def calculate_significance(observed_value, null_distribution, alpha=DEFAULT_ALPHA):
    """
    Calculate the statistical significance of an observed value against a null distribution.
    
    Args:
        observed_value (float): The observed test statistic
        null_distribution (ndarray): Distribution of the test statistic under the null hypothesis
        alpha (float, optional): Significance level. Defaults to DEFAULT_ALPHA.
        
    Returns:
        dict: Dictionary containing p-value, significance, and confidence interval
    """
    # Calculate p-value
    p_value = np.mean(null_distribution >= observed_value)
    
    # Calculate confidence interval
    lower_ci = np.percentile(null_distribution, 100 * alpha / 2)
    upper_ci = np.percentile(null_distribution, 100 * (1 - alpha / 2))
    
    # Determine significance
    significant = p_value < alpha
    
    # Calculate effect size (Cohen's d)
    effect_size = (observed_value - np.mean(null_distribution)) / np.std(null_distribution)
    
    # Calculate ratio
    ratio = observed_value / np.mean(null_distribution) if np.mean(null_distribution) != 0 else float('inf')
    
    return {
        'p_value': p_value,
        'significant': significant,
        'confidence_interval': (lower_ci, upper_ci),
        'effect_size': effect_size,
        'ratio': ratio
    }


def test_significance(observed_value, null_distribution, alpha=DEFAULT_ALPHA):
    """
    Test the statistical significance of an observed value against a null distribution.
    
    Args:
        observed_value (float): The observed test statistic
        null_distribution (ndarray): Distribution of the test statistic under the null hypothesis
        alpha (float, optional): Significance level. Defaults to DEFAULT_ALPHA.
        
    Returns:
        dict: Dictionary containing p-value, significance, and confidence interval
    """
    return calculate_significance(observed_value, null_distribution, alpha)


def bootstrap_confidence_interval(data, statistic_func, n_bootstrap=DEFAULT_BOOTSTRAP_SAMPLES, alpha=DEFAULT_ALPHA):
    """
    Calculate bootstrap confidence intervals for a statistic.
    
    Args:
        data (ndarray or list): Input data
        statistic_func (callable): Function to calculate the statistic
        n_bootstrap (int, optional): Number of bootstrap samples. Defaults to DEFAULT_BOOTSTRAP_SAMPLES.
        alpha (float, optional): Significance level. Defaults to DEFAULT_ALPHA.
        
    Returns:
        tuple: Lower and upper confidence interval bounds
    """
    # Convert data to numpy array if it's a list
    data = np.array(data)
    
    bootstrap_stats = np.zeros(n_bootstrap)
    
    for i in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(len(data), size=len(data), replace=True)
        bootstrap_sample = data[indices]
        
        # Calculate statistic
        bootstrap_stats[i] = statistic_func(bootstrap_sample)
    
    # Calculate confidence interval
    lower_ci = np.percentile(bootstrap_stats, 100 * alpha / 2)
    upper_ci = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
    
    return (lower_ci, upper_ci)


def permutation_test(data1, data2, statistic_func, n_permutations=1000, alternative='two-sided'):
    """
    Perform a permutation test to compare two datasets.
    
    Args:
        data1 (ndarray): First dataset
        data2 (ndarray): Second dataset
        statistic_func (callable): Function to calculate the test statistic
        n_permutations (int, optional): Number of permutations. Defaults to 1000.
        alternative (str, optional): Alternative hypothesis. Defaults to 'two-sided'.
        
    Returns:
        dict: Dictionary containing p-value and observed statistic
    """
    # Calculate observed statistic
    observed_stat = statistic_func(data1, data2)
    
    # Combine data
    combined = np.concatenate([data1, data2])
    n1 = len(data1)
    n = len(combined)
    
    # Generate permutation distribution
    perm_stats = np.zeros(n_permutations)
    
    for i in range(n_permutations):
        # Shuffle data
        np.random.shuffle(combined)
        
        # Split into two groups
        perm_data1 = combined[:n1]
        perm_data2 = combined[n1:]
        
        # Calculate statistic
        perm_stats[i] = statistic_func(perm_data1, perm_data2)
    
    # Calculate p-value
    if alternative == 'two-sided':
        p_value = np.mean(np.abs(perm_stats) >= np.abs(observed_stat))
    elif alternative == 'greater':
        p_value = np.mean(perm_stats >= observed_stat)
    elif alternative == 'less':
        p_value = np.mean(perm_stats <= observed_stat)
    else:
        raise ValueError("Invalid alternative: {}".format(alternative))
    
    return {
        'p_value': p_value,
        'observed_statistic': observed_stat
    }


def calculate_transfer_entropy(x, y, bins=10, lag=1):
    """
    Calculate transfer entropy from x to y.
    
    Transfer entropy measures the directed information flow from x to y.
    
    Args:
        x (ndarray): Source time series
        y (ndarray): Target time series
        bins (int, optional): Number of bins for probability estimation. Defaults to 10.
        lag (int, optional): Time lag. Defaults to 1.
        
    Returns:
        float: Transfer entropy value
    """
    # Ensure x and y are the same length
    n = min(len(x), len(y))
    x = x[:n]
    y = y[:n]
    
    # Create lagged variables
    y_past = y[:-lag]
    y_future = y[lag:]
    x_past = x[:-lag]
    
    # Discretize data
    y_past_binned = np.digitize(y_past, np.linspace(min(y_past), max(y_past), bins))
    y_future_binned = np.digitize(y_future, np.linspace(min(y_future), max(y_future), bins))
    x_past_binned = np.digitize(x_past, np.linspace(min(x_past), max(x_past), bins))
    
    # Calculate probabilities
    p_y_future_y_past = np.zeros((bins, bins))
    p_y_future_y_past_x_past = np.zeros((bins, bins, bins))
    p_y_past = np.zeros(bins)
    p_y_past_x_past = np.zeros((bins, bins))
    
    for i in range(len(y_past_binned)):
        p_y_past[y_past_binned[i]-1] += 1
        p_y_future_y_past[y_future_binned[i]-1, y_past_binned[i]-1] += 1
        p_y_past_x_past[y_past_binned[i]-1, x_past_binned[i]-1] += 1
        p_y_future_y_past_x_past[y_future_binned[i]-1, y_past_binned[i]-1, x_past_binned[i]-1] += 1
    
    # Normalize
    p_y_past /= len(y_past_binned)
    p_y_future_y_past /= len(y_past_binned)
    p_y_past_x_past /= len(y_past_binned)
    p_y_future_y_past_x_past /= len(y_past_binned)
    
    # Calculate transfer entropy
    te = 0
    for i in range(bins):
        for j in range(bins):
            for k in range(bins):
                if p_y_future_y_past_x_past[i, j, k] > 0 and p_y_future_y_past[i, j] > 0 and p_y_past_x_past[j, k] > 0 and p_y_past[j] > 0:
                    te += p_y_future_y_past_x_past[i, j, k] * np.log2(p_y_future_y_past_x_past[i, j, k] * p_y_past[j] / (p_y_future_y_past[i, j] * p_y_past_x_past[j, k]))
    
    return te


def calculate_mutual_information(x, y, bins=10):
    """
    Calculate mutual information between x and y.
    
    Mutual information measures the shared information between two variables.
    
    Args:
        x (ndarray): First variable
        y (ndarray): Second variable
        bins (int, optional): Number of bins for probability estimation. Defaults to 10.
        
    Returns:
        float: Mutual information value
    """
    # Ensure x and y are the same length
    n = min(len(x), len(y))
    x = x[:n]
    y = y[:n]
    
    # Discretize data
    x_binned = np.digitize(x, np.linspace(min(x), max(x), bins))
    y_binned = np.digitize(y, np.linspace(min(y), max(y), bins))
    
    # Calculate probabilities
    p_xy = np.zeros((bins, bins))
    p_x = np.zeros(bins)
    p_y = np.zeros(bins)
    
    for i in range(n):
        p_x[x_binned[i]-1] += 1
        p_y[y_binned[i]-1] += 1
        p_xy[x_binned[i]-1, y_binned[i]-1] += 1
    
    # Normalize
    p_x /= n
    p_y /= n
    p_xy /= n
    
    # Calculate mutual information
    mi = 0
    for i in range(bins):
        for j in range(bins):
            if p_xy[i, j] > 0 and p_x[i] > 0 and p_y[j] > 0:
                mi += p_xy[i, j] * np.log2(p_xy[i, j] / (p_x[i] * p_y[j]))
    
    return mi


def calculate_coherence(x, y, fs=1.0, nperseg=256):
    """
    Calculate coherence between x and y.
    
    Coherence measures the linear correlation between two signals in the frequency domain.
    
    Args:
        x (ndarray): First time series
        y (ndarray): Second time series
        fs (float, optional): Sampling frequency. Defaults to 1.0.
        nperseg (int, optional): Length of each segment. Defaults to 256.
        
    Returns:
        tuple: Frequencies and coherence values
    """
    from scipy import signal
    
    # Ensure x and y are the same length
    n = min(len(x), len(y))
    x = x[:n]
    y = y[:n]
    
    # Calculate coherence
    f, Cxy = signal.coherence(x, y, fs=fs, nperseg=nperseg)
    
    return f, Cxy


def calculate_power_law_exponent(data, method='welch'):
    """
    Calculate the power law exponent of a time series.
    
    Args:
        data (ndarray): Input time series
        method (str, optional): Method for PSD estimation. Defaults to 'welch'.
        
    Returns:
        float: Power law exponent
    """
    from scipy import signal
    
    if method == 'welch':
        # Calculate PSD using Welch's method
        f, psd = signal.welch(data, fs=1.0, nperseg=min(256, len(data)//2))
    elif method == 'periodogram':
        # Calculate PSD using periodogram
        f, psd = signal.periodogram(data, fs=1.0)
    else:
        raise ValueError("Unknown method: {}".format(method))
    
    # Remove zero frequency
    mask = f > 0
    f = f[mask]
    psd = psd[mask]
    
    # Fit power law using log-log linear regression
    log_f = np.log10(f)
    log_psd = np.log10(psd)
    
    # Linear regression
    slope, _, _, _, _ = stats.linregress(log_f, log_psd)
    
    # Power law exponent is the negative of the slope
    exponent = -slope
    
    return exponent


def calculate_hurst_exponent(data, max_lag=100):
    """
    Calculate the Hurst exponent of a time series.
    
    The Hurst exponent measures the long-term memory of a time series.
    H < 0.5: anti-persistent series
    H = 0.5: random walk
    H > 0.5: persistent series
    
    Args:
        data (ndarray): Input time series
        max_lag (int, optional): Maximum lag. Defaults to 100.
        
    Returns:
        float: Hurst exponent
    """
    # Ensure data is long enough
    if len(data) < max_lag * 2:
        max_lag = len(data) // 2
    
    # Calculate range of lags
    lags = range(2, max_lag)
    
    # Calculate R/S for each lag
    rs = np.zeros(len(lags))
    
    for i, lag in enumerate(lags):
        # Split data into chunks
        n_chunks = len(data) // lag
        if n_chunks == 0:
            continue
        
        # Calculate R/S for each chunk and average
        rs_values = np.zeros(n_chunks)
        
        for j in range(n_chunks):
            chunk = data[j*lag:(j+1)*lag]
            
            # Calculate mean-adjusted series
            mean = np.mean(chunk)
            mean_adjusted = chunk - mean
            
            # Calculate cumulative deviation
            cumulative = np.cumsum(mean_adjusted)
            
            # Calculate range and standard deviation
            r = np.max(cumulative) - np.min(cumulative)
            s = np.std(chunk)
            
            # Avoid division by zero
            if s == 0:
                rs_values[j] = 0
            else:
                rs_values[j] = r / s
        
        rs[i] = np.mean(rs_values)
    
    # Linear regression on log-log scale
    log_lags = np.log10(lags)
    log_rs = np.log10(rs)
    
    # Remove any NaN or inf values
    mask = np.isfinite(log_rs)
    log_lags = log_lags[mask]
    log_rs = log_rs[mask]
    
    if len(log_lags) < 2:
        return 0.5  # Default to random walk
    
    # Linear regression
    slope, _, _, _, _ = stats.linregress(log_lags, log_rs)
    
    return slope


def fisher_combined_test(p_values):
    """
    Combine p-values using Fisher's method.
    
    Args:
        p_values (list): List of p-values
        
    Returns:
        float: Combined p-value
    """
    # Remove any p-values that are exactly 0 or 1
    p_values = [p for p in p_values if 0 < p < 1]
    
    if not p_values:
        return 1.0
    
    # Calculate test statistic
    chi2 = -2 * np.sum(np.log(p_values))
    
    # Degrees of freedom
    df = 2 * len(p_values)
    
    # Calculate combined p-value
    combined_p = 1 - stats.chi2.cdf(chi2, df)
    
    return combined_p


def calculate_phi_optimality(value, constants, method='ratio'):
    """
    Calculate optimality score for a value compared to mathematical constants.
    
    This function calculates how close a value is to various mathematical constants,
    particularly the golden ratio (phi).
    
    Args:
        value (float or dict): The value or dictionary of values to evaluate
        constants (dict): Dictionary of constants to compare against
        method (str, optional): Method for calculating optimality. Defaults to 'ratio'.
        
    Returns:
        dict: Dictionary with optimality scores for each constant
    """
    optimality = {}
    
    # Handle dictionary input
    if isinstance(value, dict):
        # Process each value in the dictionary
        result = {}
        for k, v in value.items():
            result[k] = calculate_phi_optimality(v, constants, method)
        return result
    
    # Handle single value input
    if method == 'ratio':
        # Calculate ratio-based optimality
        for name, constant in constants.items():
            # Calculate ratio
            ratio = value / constant
            
            # Calculate optimality (1 when equal, decreasing as ratio deviates from 1)
            optimality[name] = 1 / (1 + np.abs(np.log(ratio)))
    
    elif method == 'difference':
        # Calculate difference-based optimality
        for name, constant in constants.items():
            # Calculate normalized difference
            diff = np.abs(value - constant) / constant
            
            # Calculate optimality (1 when equal, decreasing as difference increases)
            optimality[name] = np.exp(-diff)
    
    else:
        raise ValueError("Unknown method: {}".format(method))
    
    return optimality


def find_best_constant(optimality_dict):
    """
    Find the constant with the highest optimality score.
    
    Args:
        optimality_dict (dict): Dictionary of optimality scores
        
    Returns:
        tuple: Name of best constant and its optimality score
    """
    if not optimality_dict:
        return None, 0
    
    best_constant = max(optimality_dict.items(), key=lambda x: x[1])
    return best_constant[0], best_constant[1]
