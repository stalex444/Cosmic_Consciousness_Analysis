# Cosmic Consciousness Analysis - Developer Guide

This guide provides detailed information for developers who want to understand, modify, or extend the Cosmic Consciousness Analysis framework.

## Codebase Structure

The codebase is organized into three main layers:

1. **Core Framework Layer**: Contains the foundational components used by all tests
2. **Test Implementation Layer**: Contains the specific test implementations
3. **Analysis Integration Layer**: Integrates results from multiple tests

### Core Framework Layer

The core framework provides the foundation for all tests and includes:

- `base_test.py`: Abstract base class that defines the interface for all tests
- `constants.py`: Mathematical constants and configuration parameters
- `data_handler.py`: Functions for loading and preprocessing CMB data
- `statistics.py`: Common statistical functions for significance testing
- `visualization.py`: Utilities for creating standardized visualizations

### Test Implementation Layer

Tests are organized into categories based on their analytical focus:

- **Coherence Tests**: Analyze coherent patterns in the CMB data
  - `meta_coherence_test.py`: Analyzes coherence of local coherence measures
  
- **Information Tests**: Examine information flow and integration
  - `transfer_entropy_test.py`: Measures information flow between scales
  
- **Scale Tests**: Investigate transitions across different scales
  - `scale_transition_test.py`: Analyzes scale boundaries where organizational principles change
  
- **Structural Tests**: Analyze structural properties
  - `golden_ratio_test.py`: Examines golden ratio patterns
  - `fractal_analysis_test.py`: Analyzes fractal properties using Hurst exponent

### Analysis Integration Layer

- `analysis.py`: Contains the `CosmicConsciousnessAnalyzer` class that integrates results from multiple tests

## Creating a New Test

To create a new test, follow these steps:

1. **Choose the appropriate category** for your test (coherence, information, scale, or structural)
2. **Create a new Python file** in the corresponding directory
3. **Implement a test class** that inherits from `BaseTest`
4. **Implement the required methods**:
   - `__init__`: Initialize the test with data and parameters
   - `run`: Execute the test analysis
   - `generate_report`: Create a textual report of the results
   - `visualize`: Create visualizations of the results

### Example: Creating a New Structural Test

Here's an example of creating a new structural test that analyzes symmetry patterns in CMB data:

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Symmetry Test
------------
Analyzes symmetry patterns in CMB data.
"""

import numpy as np
import matplotlib.pyplot as plt

from core_framework.base_test import BaseTest
from core_framework.statistics import calculate_significance, bootstrap_confidence_interval
from core_framework.data_handler import generate_surrogate_data


class SymmetryTest(BaseTest):
    """Test for analyzing symmetry patterns in CMB data."""
    
    def __init__(self, data, seed=None):
        """
        Initialize the symmetry test.
        
        Args:
            data (ndarray): CMB data to analyze
            seed (int, optional): Random seed for reproducibility. Defaults to None.
        """
        super(SymmetryTest, self).__init__(data=data, seed=seed)
        self.symmetry_score = None
        self.p_value = None
        self.confidence_interval = None
    
    def run(self):
        """Run the symmetry test analysis."""
        # Calculate symmetry score
        self.symmetry_score = self._calculate_symmetry_score(self.data)
        
        # Generate surrogate data for significance testing
        n_surrogates = 1000
        surrogate_data = generate_surrogate_data(
            self.data, 
            method='phase_randomization',
            n_surrogates=n_surrogates,
            seed=self.seed
        )
        
        # Calculate symmetry scores for surrogate data
        surrogate_scores = np.zeros(n_surrogates)
        for i in range(n_surrogates):
            surrogate_scores[i] = self._calculate_symmetry_score(surrogate_data[i])
        
        # Calculate significance
        self.p_value = calculate_significance(self.symmetry_score, surrogate_scores)
        
        # Calculate confidence interval
        self.confidence_interval = bootstrap_confidence_interval(
            self.data,
            statistic=self._calculate_symmetry_score,
            n_bootstrap=1000,
            confidence=0.95,
            seed=self.seed
        )
        
        # Set has_run flag
        self.has_run = True
    
    def _calculate_symmetry_score(self, data):
        """
        Calculate symmetry score for the given data.
        
        Args:
            data (ndarray): Data to analyze
            
        Returns:
            float: Symmetry score
        """
        # Calculate autocorrelation
        autocorr = np.correlate(data, data, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # Normalize
        autocorr = autocorr / autocorr[0]
        
        # Calculate symmetry score as the correlation between the first and second
        # half of the autocorrelation function
        midpoint = len(autocorr) // 2
        first_half = autocorr[:midpoint]
        second_half = autocorr[midpoint:][::-1]  # Reverse the second half
        
        # Trim to same length
        min_length = min(len(first_half), len(second_half))
        first_half = first_half[:min_length]
        second_half = second_half[:min_length]
        
        # Calculate correlation
        symmetry_score = np.corrcoef(first_half, second_half)[0, 1]
        
        return symmetry_score
    
    def generate_report(self):
        """
        Generate a report of the test results.
        
        Returns:
            str: Report text
        """
        if not self.has_run:
            return "Symmetry test has not been run yet."
        
        report = []
        report.append("=" * 50)
        report.append("SYMMETRY TEST RESULTS")
        report.append("=" * 50)
        report.append("")
        
        report.append("Symmetry Score: {:.4f}".format(self.symmetry_score))
        report.append("p-value: {:.4f}".format(self.p_value))
        report.append("95% Confidence Interval: [{:.4f}, {:.4f}]".format(
            self.confidence_interval[0], self.confidence_interval[1]
        ))
        
        report.append("")
        report.append("Interpretation:")
        if self.p_value < 0.05:
            report.append("- The CMB data shows statistically significant symmetry patterns")
            report.append("  (p < 0.05).")
        else:
            report.append("- The symmetry patterns in the CMB data are not statistically")
            report.append("  significant (p >= 0.05).")
        
        if self.symmetry_score > 0.5:
            report.append("- The symmetry score is high, indicating strong symmetry patterns.")
        elif self.symmetry_score > 0.2:
            report.append("- The symmetry score is moderate, indicating some symmetry patterns.")
        else:
            report.append("- The symmetry score is low, indicating weak symmetry patterns.")
        
        return "\n".join(report)
    
    def visualize(self, output_dir=None):
        """
        Create visualizations of the test results.
        
        Args:
            output_dir (str, optional): Directory to save visualizations. Defaults to None.
        """
        if not self.has_run:
            print("Cannot visualize results: Symmetry test has not been run yet.")
            return
        
        # Create figure
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot autocorrelation
        autocorr = np.correlate(self.data, self.data, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / autocorr[0]
        
        axes[0].plot(autocorr)
        axes[0].set_title('Autocorrelation Function')
        axes[0].set_xlabel('Lag')
        axes[0].set_ylabel('Autocorrelation')
        axes[0].grid(True)
        
        # Plot symmetry comparison
        midpoint = len(autocorr) // 2
        first_half = autocorr[:midpoint]
        second_half = autocorr[midpoint:][::-1]  # Reverse the second half
        
        # Trim to same length
        min_length = min(len(first_half), len(second_half))
        first_half = first_half[:min_length]
        second_half = second_half[:min_length]
        
        axes[1].plot(first_half, label='First Half')
        axes[1].plot(second_half, label='Second Half (Reversed)')
        axes[1].set_title('Symmetry Comparison (Correlation: {:.4f})'.format(self.symmetry_score))
        axes[1].set_xlabel('Index')
        axes[1].set_ylabel('Autocorrelation')
        axes[1].legend()
        axes[1].grid(True)
        
        # Add overall title
        fig.suptitle('Symmetry Test Results', fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save or show
        if output_dir is not None:
            output_path = os.path.join(output_dir, 'symmetry_test_results.png')
            fig.savefig(output_path, dpi=300)
            plt.close(fig)
            print("Visualization saved to {}".format(output_path))
        else:
            plt.show()
```

## Running Tests

There are several ways to run tests:

1. **Run a single test** for debugging and development:
   ```bash
   python run_test.py --test golden-ratio
   ```

2. **Run all tests** with the main analysis script:
   ```bash
   python run_analysis.py --all
   ```

3. **Run unit tests** to verify the framework:
   ```bash
   python run_tests.py
   ```

## Adding a New Test to the Analyzer

To integrate your new test into the analyzer:

1. Import your test class in `run_analysis.py`:
   ```python
   from tests.structural_tests.symmetry_test import SymmetryTest
   ```

2. Add a command-line argument for your test:
   ```python
   parser.add_argument("--symmetry", action="store_true",
                      help="Run symmetry test")
   ```

3. Add your test to the tests_to_run list:
   ```python
   if run_all or args.symmetry:
       tests_to_run.append(SymmetryTest(seed=args.seed, data=data))
   ```

4. Add your test to the "all tests" list:
   ```python
   tests_to_run = [
       MetaCoherenceTest(seed=args.seed, data=data),
       TransferEntropyTest(seed=args.seed, data=data),
       ScaleTransitionTest(seed=args.seed, data=data),
       GoldenRatioTest(seed=args.seed, data=data),
       FractalAnalysisTest(seed=args.seed, data=data),
       SymmetryTest(seed=args.seed, data=data)
   ]
   ```

## Python 2.7 Compatibility

This codebase is designed to be compatible with Python 2.7. Key considerations:

1. **String Formatting**: Use `"{}".format()` instead of f-strings
2. **Division**: Use `float(a) / b` or `from __future__ import division` to ensure float division
3. **Unicode**: Use `u"string"` for Unicode strings
4. **Print Statements**: Use `print("text")` with parentheses
5. **Exception Handling**: Use `except Exception as e:` syntax

## Best Practices

1. **Documentation**: Always include detailed docstrings for classes and methods
2. **Type Hints**: Use docstring type annotations for better code understanding
3. **Error Handling**: Use try-except blocks to handle potential errors gracefully
4. **Testing**: Write unit tests for new functionality
5. **Reproducibility**: Always use random seeds for reproducible results
6. **Visualization**: Follow the established visualization style for consistency
