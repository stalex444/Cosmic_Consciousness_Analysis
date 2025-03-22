#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data Handler Module
------------------
Centralizes data loading, preprocessing, and surrogate generation for CMB analysis.
"""

import os
import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from core_framework.constants import DEFAULT_SEED, DEFAULT_DATA_SIZE, PHI

# Import Planck data handler if available
try:
    from planck_data.planck_data_handler import load_planck_map, preprocess_for_analysis
    PLANCK_AVAILABLE = True
except ImportError:
    PLANCK_AVAILABLE = False


def load_cmb_data(simulated=True, seed=None, size=DEFAULT_DATA_SIZE, filepath=None, planck_map=None, field=0):
    """
    Load CMB data, either simulated or from file.
    
    Args:
        simulated (bool, optional): Whether to use simulated data. Defaults to True.
        seed (int, optional): Random seed for reproducibility. Defaults to None.
        size (int, optional): Size of simulated data. Defaults to DEFAULT_DATA_SIZE.
        filepath (str, optional): Path to real data file. Defaults to None.
        planck_map (ndarray, optional): Pre-loaded Planck HEALPix map. Defaults to None.
        field (int, optional): Field to use from Planck map (0=temperature, 1=Q, 2=U). Defaults to 0.
        
    Returns:
        ndarray: The loaded CMB data
    """
    if simulated:
        print("Generating simulated CMB data...")
        return generate_simulated_cmb_data(size=size, seed=seed)
    elif planck_map is not None:
        print("Using provided Planck map...")
        if PLANCK_AVAILABLE:
            return preprocess_for_analysis(planck_map, n_points=size, seed=seed)
        else:
            raise ImportError("Planck data handler not available. Please install required dependencies.")
    elif filepath is not None:
        if filepath.endswith('.fits') and PLANCK_AVAILABLE:
            print("Loading Planck FITS file: {}...".format(filepath))
            planck_map = load_planck_map(filepath, field=field)
            return preprocess_for_analysis(planck_map, n_points=size, seed=seed)
        else:
            print("Loading CMB data from {}...".format(filepath))
            if not os.path.exists(filepath):
                raise FileNotFoundError("CMB data file not found: {}".format(filepath))
            
            return np.load(filepath)
    else:
        filepath = os.path.join(os.getcwd(), "data", "planck_cmb_data.npy")
        
        print("Loading CMB data from {}...".format(filepath))
        if not os.path.exists(filepath):
            raise FileNotFoundError("CMB data file not found: {}".format(filepath))
        
        return np.load(filepath)


def generate_simulated_cmb_data(size=DEFAULT_DATA_SIZE, seed=None, phi_bias=0.1):
    """
    Generate simulated CMB data with embedded patterns.
    
    This function creates simulated CMB data with subtle patterns related to
    mathematical constants, particularly the golden ratio.
    
    Args:
        size (int, optional): Size of the data. Defaults to DEFAULT_DATA_SIZE.
        seed (int, optional): Random seed for reproducibility. Defaults to None.
        phi_bias (float, optional): Strength of golden ratio bias. Defaults to 0.1.
        
    Returns:
        ndarray: Simulated CMB data
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Base signal: Gaussian random noise (white noise)
    base_signal = np.random.normal(0, 1, size)
    
    # Add 1/f noise (pink noise) to simulate CMB power spectrum
    pink_noise = generate_pink_noise(size)
    
    # Add patterns related to the golden ratio
    phi_pattern = generate_phi_pattern(size, phi_bias)
    
    # Combine components
    cmb_data = 0.7 * base_signal + 0.2 * pink_noise + 0.1 * phi_pattern
    
    # Normalize
    cmb_data = (cmb_data - np.mean(cmb_data)) / np.std(cmb_data)
    
    return cmb_data


def generate_pink_noise(size=DEFAULT_DATA_SIZE):
    """
    Generate 1/f (pink) noise.
    
    Args:
        size (int, optional): Size of the noise array. Defaults to DEFAULT_DATA_SIZE.
        
    Returns:
        ndarray: Pink noise array
    """
    # Generate white noise
    white_noise = np.random.normal(0, 1, size)
    
    # Create frequency domain filter
    freq = np.fft.fftfreq(size)
    freq[0] = 1e-10  # Avoid division by zero
    
    # Apply 1/f filter
    fft_white = np.fft.fft(white_noise)
    fft_pink = fft_white / np.sqrt(np.abs(freq))
    pink_noise = np.real(np.fft.ifft(fft_pink))
    
    # Normalize
    pink_noise = (pink_noise - np.mean(pink_noise)) / np.std(pink_noise)
    
    return pink_noise


def generate_phi_pattern(size=DEFAULT_DATA_SIZE, strength=0.1):
    """
    Generate a pattern with golden ratio (phi) relationships.
    
    Args:
        size (int, optional): Size of the pattern array. Defaults to DEFAULT_DATA_SIZE.
        strength (float, optional): Strength of the pattern. Defaults to 0.1.
        
    Returns:
        ndarray: Pattern array with golden ratio relationships
    """
    # Initialize pattern
    pattern = np.zeros(size)
    
    # Create Fibonacci-like sequence for indices
    fib_indices = fibonacci_sequence(int(np.log(size) / np.log(PHI) * 2))
    fib_indices = [i for i in fib_indices if i < size]
    
    # Add peaks at Fibonacci indices
    for i in fib_indices:
        pattern[i] = 1.0
    
    # Smooth the pattern
    pattern = gaussian_filter1d(pattern, sigma=2)
    
    # Add some noise to make it subtle
    pattern = pattern + 0.2 * np.random.normal(0, 1, size)
    
    # Scale by strength
    pattern = strength * pattern
    
    # Normalize
    pattern = (pattern - np.mean(pattern)) / np.std(pattern)
    
    return pattern


def fibonacci_sequence(n):
    """
    Generate Fibonacci sequence up to n terms.
    
    Args:
        n (int): Number of terms
        
    Returns:
        list: Fibonacci sequence
    """
    sequence = [0, 1]
    for i in range(2, n):
        sequence.append(sequence[i-1] + sequence[i-2])
    return sequence


def generate_surrogate_data(data, method='phase_randomization', n_surrogates=1, seed=None):
    """
    Generate surrogate data for statistical testing.
    
    Args:
        data (ndarray): Original data
        method (str, optional): Surrogate generation method. Defaults to 'phase_randomization'.
        n_surrogates (int, optional): Number of surrogate datasets. Defaults to 1.
        seed (int, optional): Random seed. Defaults to None.
        
    Returns:
        ndarray: Surrogate data with shape (n_surrogates, len(data))
    """
    if seed is not None:
        np.random.seed(seed)
    
    if method == 'phase_randomization':
        return phase_randomization_surrogates(data, n_surrogates)
    elif method == 'bootstrap':
        return bootstrap_surrogates(data, n_surrogates)
    elif method == 'shuffle':
        return shuffle_surrogates(data, n_surrogates)
    else:
        raise ValueError("Unknown surrogate method: {}".format(method))


def phase_randomization_surrogates(data, n_surrogates=1):
    """
    Generate phase-randomized surrogate data.
    
    This preserves the power spectrum but randomizes the phases.
    
    Args:
        data (ndarray): Original data
        n_surrogates (int, optional): Number of surrogate datasets. Defaults to 1.
        
    Returns:
        ndarray: Surrogate data with shape (n_surrogates, len(data))
    """
    data_fft = np.fft.fft(data)
    magnitude = np.abs(data_fft)
    
    n = len(data)
    surrogates = np.zeros((n_surrogates, n))
    
    for i in range(n_surrogates):
        # Generate random phases
        random_phases = np.random.uniform(0, 2*np.pi, n)
        
        # Ensure conjugate symmetry for real output
        random_phases[0] = 0
        if n % 2 == 0:  # even length
            random_phases[n//2] = 0
            random_phases[1:n//2] = random_phases[1:n//2]
            random_phases[n//2+1:] = -random_phases[n//2-1:0:-1]
        else:  # odd length
            random_phases[1:(n+1)//2] = random_phases[1:(n+1)//2]
            random_phases[(n+1)//2:] = -random_phases[(n-1)//2:0:-1]
        
        # Create complex FFT with original magnitudes and random phases
        surrogate_fft = magnitude * np.exp(1j * random_phases)
        
        # Inverse FFT to get surrogate time series
        surrogate = np.real(np.fft.ifft(surrogate_fft))
        
        surrogates[i] = surrogate
    
    return surrogates


def bootstrap_surrogates(data, n_surrogates=1):
    """
    Generate bootstrap surrogate data.
    
    This resamples the data with replacement.
    
    Args:
        data (ndarray): Original data
        n_surrogates (int, optional): Number of surrogate datasets. Defaults to 1.
        
    Returns:
        ndarray: Surrogate data with shape (n_surrogates, len(data))
    """
    n = len(data)
    surrogates = np.zeros((n_surrogates, n))
    
    for i in range(n_surrogates):
        # Random indices with replacement
        indices = np.random.choice(n, size=n, replace=True)
        
        # Create surrogate by resampling
        surrogates[i] = data[indices]
    
    return surrogates


def shuffle_surrogates(data, n_surrogates=1):
    """
    Generate shuffled surrogate data.
    
    This randomly permutes the data.
    
    Args:
        data (ndarray): Original data
        n_surrogates (int, optional): Number of surrogate datasets. Defaults to 1.
        
    Returns:
        ndarray: Surrogate data with shape (n_surrogates, len(data))
    """
    n = len(data)
    surrogates = np.zeros((n_surrogates, n))
    
    for i in range(n_surrogates):
        # Create a copy of the data
        surrogate = data.copy()
        
        # Shuffle the copy
        np.random.shuffle(surrogate)
        
        surrogates[i] = surrogate
    
    return surrogates


def preprocess_data(data, normalize=True, detrend=False, filter_type=None, filter_params=None):
    """
    Preprocess CMB data.
    
    Args:
        data (ndarray): Raw data
        normalize (bool, optional): Whether to normalize data. Defaults to True.
        detrend (bool, optional): Whether to remove linear trend. Defaults to False.
        filter_type (str, optional): Type of filter to apply. Defaults to None.
        filter_params (dict, optional): Filter parameters. Defaults to None.
        
    Returns:
        ndarray: Preprocessed data
    """
    # Make a copy to avoid modifying the original
    processed_data = data.copy()
    
    # Remove NaN values
    if np.any(np.isnan(processed_data)):
        print("Warning: NaN values detected in data. Replacing with zeros.")
        processed_data = np.nan_to_num(processed_data)
    
    # Detrend if requested
    if detrend:
        processed_data = signal.detrend(processed_data)
    
    # Apply filter if requested
    if filter_type is not None:
        if filter_params is None:
            filter_params = {}
        
        if filter_type == 'lowpass':
            b, a = signal.butter(
                filter_params.get('order', 4),
                filter_params.get('cutoff', 0.1),
                btype='lowpass'
            )
            processed_data = signal.filtfilt(b, a, processed_data)
        
        elif filter_type == 'highpass':
            b, a = signal.butter(
                filter_params.get('order', 4),
                filter_params.get('cutoff', 0.1),
                btype='highpass'
            )
            processed_data = signal.filtfilt(b, a, processed_data)
        
        elif filter_type == 'bandpass':
            b, a = signal.butter(
                filter_params.get('order', 4),
                [filter_params.get('low_cutoff', 0.1), filter_params.get('high_cutoff', 0.4)],
                btype='bandpass'
            )
            processed_data = signal.filtfilt(b, a, processed_data)
    
    # Normalize if requested
    if normalize:
        processed_data = (processed_data - np.mean(processed_data)) / np.std(processed_data)
    
    return processed_data


def segment_data(data, scale, overlap=0.5):
    """
    Segment data into windows of specified scale.
    
    Args:
        data (ndarray): Input data
        scale (int): Window size
        overlap (float, optional): Overlap fraction. Defaults to 0.5.
        
    Returns:
        ndarray: Segmented data with shape (n_segments, scale)
    """
    n = len(data)
    step = int(scale * (1 - overlap))
    
    # Calculate number of segments
    n_segments = (n - scale) // step + 1
    
    # Create output array
    segments = np.zeros((n_segments, scale))
    
    # Extract segments
    for i in range(n_segments):
        start = i * step
        end = start + scale
        segments[i] = data[start:end]
    
    return segments
