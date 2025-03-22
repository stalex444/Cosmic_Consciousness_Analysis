#!/usr/bin/env python3
"""
Statistical analysis utilities for Cosmic Consciousness Analysis.
"""

import numpy as np
from scipy import stats
import warnings

class StatisticalAnalyzer:
    """Class for statistical analysis of CMB data."""
    
    def __init__(self, monte_carlo_sims=10000):
        """
        Initialize the statistical analyzer.
        
        Parameters:
        -----------
        monte_carlo_sims : int, optional
            Number of Monte Carlo simulations for significance testing.
        """
        self.monte_carlo_sims = monte_carlo_sims
        self.phi = (1 + np.sqrt(5)) / 2  # Golden ratio
    
    def calculate_z_score(self, observed, random_distribution):
        """
        Calculate the z-score for an observed value against a random distribution.
        
        Parameters:
        -----------
        observed : float
            Observed value.
        random_distribution : array-like
            Distribution of values under the null hypothesis.
            
        Returns:
        --------
        tuple
            (z_score, p_value)
        """
        # Calculate mean and standard deviation of random distribution
        random_mean = np.mean(random_distribution)
        random_std = np.std(random_distribution)
        
        # Calculate z-score
        if random_std > 0:
            z_score = (observed - random_mean) / random_std
        else:
            z_score = 0.0
            warnings.warn("Standard deviation of random distribution is zero.")
        
        # Calculate p-value (two-tailed)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        return z_score, p_value
    
    def calculate_phi_optimality(self, observed, random_mean):
        """
        Calculate the phi-optimality metric.
        
        Parameters:
        -----------
        observed : float
            Observed value.
        random_mean : float
            Mean of the random distribution.
            
        Returns:
        --------
        float
            Phi-optimality value in range [-1, 1].
        """
        # Calculate ratio
        if random_mean != 0:
            ratio = observed / random_mean
        else:
            ratio = float('inf') if observed > 0 else 0.0
        
        # Calculate phi-optimality
        if ratio > 1:
            # Positive phi-optimality (better than random)
            phi_optimality = 2 * (1 - 1/ratio) if ratio <= 2 else 1.0
        else:
            # Negative phi-optimality (worse than random)
            phi_optimality = -2 * (1 - ratio) if ratio >= 0.5 else -1.0
        
        return phi_optimality
    
    def fishers_method(self, p_values):
        """
        Combine p-values using Fisher's method.
        
        Parameters:
        -----------
        p_values : array-like
            List of p-values to combine.
            
        Returns:
        --------
        float
            Combined p-value.
        """
        # Filter out NaN and zero p-values
        valid_p = [p for p in p_values if not np.isnan(p) and p > 0]
        
        if not valid_p:
            return np.nan
        
        # Calculate chi-squared statistic
        chi2 = -2 * np.sum(np.log(valid_p))
        
        # Calculate degrees of freedom
        df = 2 * len(valid_p)
        
        # Calculate combined p-value
        combined_p = 1 - stats.chi2.cdf(chi2, df)
        
        return combined_p
    
    def mutual_information(self, x, y, bins=10):
        """
        Calculate mutual information between two variables.
        
        Parameters:
        -----------
        x : array-like
            First variable.
        y : array-like
            Second variable.
        bins : int, optional
            Number of bins for histogram.
            
        Returns:
        --------
        float
            Mutual information value.
        """
        # Create joint histogram
        hist_xy, _, _ = np.histogram2d(x, y, bins=bins)
        
        # Calculate marginal histograms
        hist_x = np.sum(hist_xy, axis=1)
        hist_y = np.sum(hist_xy, axis=0)
        
        # Calculate entropies
        hist_xy = hist_xy / np.sum(hist_xy)
        hist_x = hist_x / np.sum(hist_x)
        hist_y = hist_y / np.sum(hist_y)
        
        # Remove zeros
        hist_xy = hist_xy[hist_xy > 0]
        hist_x = hist_x[hist_x > 0]
        hist_y = hist_y[hist_y > 0]
        
        # Calculate entropies
        h_xy = -np.sum(hist_xy * np.log2(hist_xy))
        h_x = -np.sum(hist_x * np.log2(hist_x))
        h_y = -np.sum(hist_y * np.log2(hist_y))
        
        # Calculate mutual information
        mi = h_x + h_y - h_xy
        
        return mi
    
    def transfer_entropy(self, source, target, delay=1, bins=10):
        """
        Calculate transfer entropy from source to target.
        
        Parameters:
        -----------
        source : array-like
            Source time series.
        target : array-like
            Target time series.
        delay : int, optional
            Time delay.
        bins : int, optional
            Number of bins for histogram.
            
        Returns:
        --------
        float
            Transfer entropy value.
        """
        # Ensure arrays are numpy arrays
        source = np.asarray(source)
        target = np.asarray(target)
        
        # Create delayed versions
        source_past = source[:-delay]
        target_past = target[:-delay]
        target_present = target[delay:]
        
        # Calculate entropies
        h_tp_given_tp_past = self._conditional_entropy(target_present, target_past, bins)
        h_tp_given_tp_past_s_past = self._conditional_entropy(target_present, np.column_stack((target_past, source_past)), bins)
        
        # Calculate transfer entropy
        te = h_tp_given_tp_past - h_tp_given_tp_past_s_past
        
        return te
    
    def _conditional_entropy(self, x, y, bins=10):
        """
        Calculate conditional entropy H(X|Y).
        
        Parameters:
        -----------
        x : array-like
            First variable.
        y : array-like
            Second variable.
        bins : int, optional
            Number of bins for histogram.
            
        Returns:
        --------
        float
            Conditional entropy value.
        """
        # Ensure y is 2D
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        # Calculate joint entropy H(X,Y)
        xy = np.column_stack((x, y))
        h_xy = self._entropy(xy, bins)
        
        # Calculate entropy H(Y)
        h_y = self._entropy(y, bins)
        
        # Calculate conditional entropy H(X|Y) = H(X,Y) - H(Y)
        h_x_given_y = h_xy - h_y
        
        return h_x_given_y
    
    def _entropy(self, x, bins=10):
        """
        Calculate Shannon entropy.
        
        Parameters:
        -----------
        x : array-like
            Input data.
        bins : int, optional
            Number of bins for histogram.
            
        Returns:
        --------
        float
            Entropy value.
        """
        # Ensure x is 2D
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        
        # Calculate histogram
        if x.shape[1] == 1:
            hist, _ = np.histogram(x, bins=bins)
        else:
            # For multidimensional data, use a different approach
            hist, _ = np.histogramdd(x, bins=bins)
        
        # Normalize histogram
        hist = hist / np.sum(hist)
        
        # Remove zeros
        hist = hist[hist > 0]
        
        # Calculate entropy
        entropy = -np.sum(hist * np.log2(hist))
        
        return entropy
    
    def power_law_exponent(self, x, y):
        """
        Calculate power law exponent for y = x^alpha.
        
        Parameters:
        -----------
        x : array-like
            Independent variable.
        y : array-like
            Dependent variable.
            
        Returns:
        --------
        tuple
            (alpha, r_squared)
        """
        # Take logarithms
        log_x = np.log(x)
        log_y = np.log(y)
        
        # Remove NaN and inf values
        valid = np.isfinite(log_x) & np.isfinite(log_y)
        log_x = log_x[valid]
        log_y = log_y[valid]
        
        if len(log_x) < 2:
            return np.nan, np.nan
        
        # Linear regression
        slope, intercept, r_value, _, _ = stats.linregress(log_x, log_y)
        
        # Calculate R-squared
        r_squared = r_value**2
        
        return slope, r_squared
    
    def hurst_exponent(self, time_series, max_lag=20):
        """
        Calculate Hurst exponent using R/S analysis.
        
        Parameters:
        -----------
        time_series : array-like
            Input time series.
        max_lag : int, optional
            Maximum lag for R/S analysis.
            
        Returns:
        --------
        tuple
            (hurst_exponent, r_squared)
        """
        # Ensure time_series is a numpy array
        time_series = np.asarray(time_series)
        
        # Calculate R/S for different lags
        lags = range(2, min(max_lag, len(time_series) // 2))
        rs_values = []
        
        for lag in lags:
            rs = self._rs_analysis(time_series, lag)
            rs_values.append(rs)
        
        # Convert to numpy arrays
        lags = np.array(lags)
        rs_values = np.array(rs_values)
        
        # Linear regression on log-log scale
        log_lags = np.log(lags)
        log_rs = np.log(rs_values)
        
        # Remove NaN and inf values
        valid = np.isfinite(log_lags) & np.isfinite(log_rs)
        log_lags = log_lags[valid]
        log_rs = log_rs[valid]
        
        if len(log_lags) < 2:
            return np.nan, np.nan
        
        # Linear regression
        slope, intercept, r_value, _, _ = stats.linregress(log_lags, log_rs)
        
        # Calculate R-squared
        r_squared = r_value**2
        
        return slope, r_squared
    
    def _rs_analysis(self, time_series, lag):
        """
        Calculate R/S statistic for a given lag.
        
        Parameters:
        -----------
        time_series : array-like
            Input time series.
        lag : int
            Lag value.
            
        Returns:
        --------
        float
            R/S statistic.
        """
        # Calculate returns
        returns = np.diff(time_series)
        
        # Split returns into segments
        n_segments = len(returns) // lag
        segments = np.array_split(returns[:n_segments * lag], n_segments)
        
        # Calculate R/S for each segment
        rs_values = []
        
        for segment in segments:
            # Calculate cumulative sum
            cum_sum = np.cumsum(segment - np.mean(segment))
            
            # Calculate range
            r = np.max(cum_sum) - np.min(cum_sum)
            
            # Calculate standard deviation
            s = np.std(segment)
            
            # Calculate R/S
            if s > 0:
                rs = r / s
            else:
                rs = 0
            
            rs_values.append(rs)
        
        # Return mean R/S
        return np.mean(rs_values)
    
    def detrended_fluctuation_analysis(self, time_series, min_scale=4, max_scale=None):
        """
        Perform detrended fluctuation analysis (DFA).
        
        Parameters:
        -----------
        time_series : array-like
            Input time series.
        min_scale : int, optional
            Minimum scale for DFA.
        max_scale : int, optional
            Maximum scale for DFA. If None, uses len(time_series) // 4.
            
        Returns:
        --------
        tuple
            (alpha, r_squared)
        """
        # Ensure time_series is a numpy array
        time_series = np.asarray(time_series)
        
        # Set maximum scale if not provided
        if max_scale is None:
            max_scale = len(time_series) // 4
        
        # Calculate scales (powers of 2)
        scales = np.logspace(np.log10(min_scale), np.log10(max_scale), num=10, dtype=int)
        scales = np.unique(scales)
        
        # Calculate fluctuation for each scale
        fluctuations = []
        
        for scale in scales:
            fluct = self._dfa_fluctuation(time_series, scale)
            fluctuations.append(fluct)
        
        # Convert to numpy arrays
        scales = np.array(scales)
        fluctuations = np.array(fluctuations)
        
        # Linear regression on log-log scale
        log_scales = np.log(scales)
        log_fluct = np.log(fluctuations)
        
        # Remove NaN and inf values
        valid = np.isfinite(log_scales) & np.isfinite(log_fluct)
        log_scales = log_scales[valid]
        log_fluct = log_fluct[valid]
        
        if len(log_scales) < 2:
            return np.nan, np.nan
        
        # Linear regression
        slope, intercept, r_value, _, _ = stats.linregress(log_scales, log_fluct)
        
        # Calculate R-squared
        r_squared = r_value**2
        
        return slope, r_squared
    
    def _dfa_fluctuation(self, time_series, scale):
        """
        Calculate DFA fluctuation for a given scale.
        
        Parameters:
        -----------
        time_series : array-like
            Input time series.
        scale : int
            Scale value.
            
        Returns:
        --------
        float
            DFA fluctuation.
        """
        # Calculate profile (cumulative sum)
        profile = np.cumsum(time_series - np.mean(time_series))
        
        # Calculate number of segments
        n_segments = len(profile) // scale
        
        if n_segments == 0:
            return np.nan
        
        # Truncate profile to multiple of scale
        profile = profile[:n_segments * scale]
        
        # Reshape profile into segments
        segments = profile.reshape((n_segments, scale))
        
        # Create time array
        time = np.arange(scale)
        
        # Calculate local trends and fluctuations
        fluctuations = []
        
        for segment in segments:
            # Fit polynomial
            coeffs = np.polyfit(time, segment, 1)
            trend = np.polyval(coeffs, time)
            
            # Calculate fluctuation (root mean square deviation)
            fluct = np.sqrt(np.mean((segment - trend)**2))
            fluctuations.append(fluct)
        
        # Return mean fluctuation
        return np.mean(fluctuations)

# Convenience function to get a statistical analyzer instance
def get_statistical_analyzer(monte_carlo_sims=10000):
    """
    Get a configured statistical analyzer instance.
    
    Parameters:
    -----------
    monte_carlo_sims : int, optional
        Number of Monte Carlo simulations for significance testing.
        
    Returns:
    --------
    StatisticalAnalyzer
        Configured statistical analyzer instance.
    """
    return StatisticalAnalyzer(monte_carlo_sims=monte_carlo_sims)
