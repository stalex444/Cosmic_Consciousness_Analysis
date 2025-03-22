#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Constants Module
---------------
Stores mathematical constants and configuration parameters used throughout the project.
"""

import numpy as np
import math

# Mathematical constants
PHI = (1 + np.sqrt(5)) / 2  # Golden ratio
E = np.e  # Euler's number
PI = np.pi  # Pi
SQRT2 = np.sqrt(2)  # Square root of 2
SQRT3 = np.sqrt(3)  # Square root of 3
LN2 = np.log(2)  # Natural logarithm of 2

# Dictionary of constants for easy iteration
CONSTANTS = {
    'phi': PHI,
    'e': E,
    'pi': PI,
    'sqrt2': SQRT2,
    'sqrt3': SQRT3,
    'ln2': LN2
}

# Constant names with proper formatting for display
CONSTANT_NAMES = {
    'phi': 'φ (Golden Ratio)',
    'e': 'e (Euler\'s Number)',
    'pi': 'π (Pi)',
    'sqrt2': '√2',
    'sqrt3': '√3',
    'ln2': 'ln(2)'
}

# Test configuration
DEFAULT_SEED = 42
DEFAULT_MONTE_CARLO_ITERATIONS = 1000
DEFAULT_BOOTSTRAP_SAMPLES = 10000
DEFAULT_ALPHA = 0.05  # Significance level
DEFAULT_DATA_SIZE = 4096

# Scale configuration
DEFAULT_MIN_SCALE = 2
DEFAULT_MAX_SCALE = 2048
DEFAULT_NUM_SCALES = 19  # Fibonacci-based scale progression

# Visualization configuration
COLORS = {
    'phi': '#E69F00',  # Orange
    'e': '#56B4E9',    # Blue
    'pi': '#009E73',   # Green
    'sqrt2': '#F0E442', # Yellow
    'sqrt3': '#0072B2', # Dark blue
    'ln2': '#D55E00'   # Red
}

# Metrics configuration
METRICS = [
    'LAMINARITY',
    'POWER_LAW',
    'COHERENCE',
    'INFORMATION_INTEGRATION',
    'TRANSFER_ENTROPY'
]

# Scale ranges
SCALE_RANGES = {
    'small': (2, 25),
    'medium': (37, 229),
    'large': (330, 2048)
}

# Significance thresholds
SIGNIFICANCE_THRESHOLDS = {
    'weak': 0.05,
    'moderate': 0.01,
    'strong': 0.001,
    'very_strong': 0.0001
}

# Phi optimality thresholds
PHI_OPTIMALITY_THRESHOLDS = {
    'weak': 0.3,
    'moderate': 0.5,
    'strong': 0.7,
    'very_strong': 0.9
}

# File paths
DEFAULT_RESULTS_DIR = 'results'
DEFAULT_FIGURES_DIR = 'figures'
DEFAULT_DATA_DIR = 'data'

# Report configuration
REPORT_SECTION_SEPARATOR = '=' * 80
REPORT_SUBSECTION_SEPARATOR = '-' * 50
