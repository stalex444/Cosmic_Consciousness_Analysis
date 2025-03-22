#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Base Test Module
---------------
Provides the BaseTest class that serves as the foundation for all cosmic tests.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from abc import ABCMeta, abstractmethod


class BaseTest(object):
    """
    Abstract base class for all cosmic tests.
    
    This class defines the standard interface that all test implementations must follow,
    ensuring consistency across the test suite.
    
    Attributes:
        name (str): The name of the test
        data (ndarray): The CMB data being analyzed
        results (dict): Results of the test
        seed (int): Random seed for reproducibility
        start_time (float): Time when the test started
        execution_time (float): Time taken to execute the test
    """
    
    __metaclass__ = ABCMeta
    
    def __init__(self, name=None, seed=None, data=None):
        """
        Initialize the base test.
        
        Args:
            name (str, optional): Name of the test. Defaults to class name.
            seed (int, optional): Random seed for reproducibility. Defaults to None.
            data (ndarray, optional): CMB data to analyze. Defaults to None.
        """
        self.name = name or self.__class__.__name__
        self.seed = seed
        self.data = data
        self.results = {}
        self.start_time = None
        self.execution_time = None
        
        # Set random seed if provided
        if self.seed is not None:
            np.random.seed(self.seed)
    
    def load_data(self, data=None, simulated=True, size=4096):
        """
        Load CMB data for analysis.
        
        Args:
            data (ndarray, optional): Directly provide data. Defaults to None.
            simulated (bool, optional): Whether to use simulated data. Defaults to True.
            size (int, optional): Size of simulated data. Defaults to 4096.
            
        Returns:
            ndarray: The loaded CMB data
        """
        from core_framework.data_handler import load_cmb_data
        
        if data is not None:
            self.data = data
        else:
            self.data = load_cmb_data(simulated=simulated, seed=self.seed, size=size)
        
        return self.data
    
    def run(self):
        """
        Run the test and measure execution time.
        
        Returns:
            dict: Results of the test
        """
        print("Running {}...".format(self.name))
        self.start_time = time.time()
        
        # Ensure data is loaded
        if self.data is None:
            self.load_data()
        
        # Run the actual test implementation
        self.results = self.run_test()
        
        self.execution_time = time.time() - self.start_time
        print("{} completed in {:.2f} seconds.".format(self.name, self.execution_time))
        
        return self.results
    
    @abstractmethod
    def run_test(self):
        """
        Implement the test logic. Must be overridden by subclasses.
        
        Returns:
            dict: Results of the test
        """
        pass
    
    @abstractmethod
    def generate_report(self):
        """
        Generate a report of the test results. Must be overridden by subclasses.
        
        Returns:
            str: Report text
        """
        pass
    
    @abstractmethod
    def visualize_results(self, save_path=None, show=False):
        """
        Create visualizations of the test results. Must be overridden by subclasses.
        
        Args:
            save_path (str, optional): Path to save visualizations. Defaults to None.
            show (bool, optional): Whether to display the visualizations. Defaults to False.
        """
        pass
    
    def save_results(self, filename=None):
        """
        Save test results to a file.
        
        Args:
            filename (str, optional): Filename to save results to. Defaults to test name.
            
        Returns:
            str: Path to the saved file
        """
        if filename is None:
            filename = "{}_results.npz".format(self.name.lower())
        
        # Create results directory if it doesn't exist
        results_dir = os.path.join(os.getcwd(), "results")
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        filepath = os.path.join(results_dir, filename)
        np.savez(filepath, **self.results)
        
        print("Results saved to {}".format(filepath))
        return filepath
    
    def load_results(self, filename=None):
        """
        Load test results from a file.
        
        Args:
            filename (str, optional): Filename to load results from. Defaults to test name.
            
        Returns:
            dict: Loaded results
        """
        if filename is None:
            filename = "{}_results.npz".format(self.name.lower())
        
        filepath = os.path.join(os.getcwd(), "results", filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError("Results file not found: {}".format(filepath))
        
        loaded = np.load(filepath, allow_pickle=True)
        self.results = {key: loaded[key] for key in loaded.files}
        
        return self.results
    
    def test_significance(self, observed_value, null_distribution, alpha=0.05):
        """
        Test the statistical significance of an observed value against a null distribution.
        
        Args:
            observed_value (float): The observed test statistic
            null_distribution (ndarray): Distribution of the test statistic under the null hypothesis
            alpha (float, optional): Significance level. Defaults to 0.05.
            
        Returns:
            dict: Dictionary containing p-value, significance, and confidence interval
        """
        from core_framework.statistics import test_significance
        return test_significance(observed_value, null_distribution, alpha)
    
    def create_figure(self, nrows=1, ncols=1, figsize=None, title=None):
        """
        Create a figure for visualizations.
        
        Args:
            nrows (int, optional): Number of rows. Defaults to 1.
            ncols (int, optional): Number of columns. Defaults to 1.
            figsize (tuple, optional): Figure size. Defaults to None.
            title (str, optional): Figure title. Defaults to None.
            
        Returns:
            tuple: Figure and axes objects
        """
        if figsize is None:
            figsize = (6 * ncols, 5 * nrows)
        
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        
        if title:
            fig.suptitle(title, fontsize=16)
            
        return fig, axes
