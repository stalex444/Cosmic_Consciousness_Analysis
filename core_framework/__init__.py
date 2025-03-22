"""
Core Framework for Cosmic Consciousness Analysis
-----------------------------------------------
This package provides the core framework for analyzing cosmic microwave background data
for evidence of conscious organization.
"""

from core_framework.base_test import BaseTest
from core_framework.constants import *
from core_framework.data_handler import load_cmb_data, generate_simulated_cmb_data, generate_surrogate_data
from core_framework.statistics import test_significance, bootstrap_confidence_interval, calculate_phi_optimality
from core_framework.visualization import create_multi_panel_figure, setup_figure, save_figure

__all__ = [
    'BaseTest',
    'load_cmb_data',
    'generate_simulated_cmb_data',
    'generate_surrogate_data',
    'test_significance',
    'bootstrap_confidence_interval',
    'calculate_phi_optimality',
    'create_multi_panel_figure',
    'setup_figure',
    'save_figure'
]
