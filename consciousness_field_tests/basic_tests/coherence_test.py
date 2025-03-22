#!/usr/bin/env python3
"""
Coherence test for Cosmic Consciousness Analysis.

This test evaluates if the CMB spectrum shows more coherence than random chance.
"""

import numpy as np
from scipy import stats

def run_test(data_loader, visualizer, stats_analyzer, monte_carlo_sims=1000, verbose=False):
    """
    Run the coherence test.
    
    Parameters:
    -----------
    data_loader : DataLoader
        Data loader instance.
    visualizer : Visualizer
        Visualizer instance.
    stats_analyzer : StatisticalAnalyzer
        Statistical analyzer instance.
    monte_carlo_sims : int, optional
        Number of Monte Carlo simulations.
    verbose : bool, optional
        Whether to print detailed output.
        
    Returns:
    --------
    dict
        Dictionary containing test results.
    """
    # Get the data
    ell = data_loader.data['ell']
    power = data_loader.data['ee_power']
    
    # Normalize the power spectrum
    normalized_power = power / np.mean(power)
    
    # Calculate the variance of the normalized power spectrum
    actual_variance = np.var(normalized_power)
    
    if verbose:
        print("Running coherence test...")
        print(f"Actual variance: {actual_variance:.4f}")
    
    # Monte Carlo simulations
    shuffled_variances = []
    for i in range(monte_carlo_sims):
        # Shuffle the power spectrum
        shuffled_power = np.random.permutation(power)
        
        # Normalize the shuffled power spectrum
        normalized_shuffled = shuffled_power / np.mean(shuffled_power)
        
        # Calculate the variance of the normalized shuffled power spectrum
        shuffled_variance = np.var(normalized_shuffled)
        
        shuffled_variances.append(shuffled_variance)
    
    # Calculate statistics
    mean_shuffled_variance = np.mean(shuffled_variances)
    variance_ratio = actual_variance / mean_shuffled_variance if mean_shuffled_variance != 0 else 1.0
    
    # Calculate z-score and p-value
    z_score = stats_analyzer.calculate_z_score(actual_variance, shuffled_variances)
    p_value = stats_analyzer.calculate_p_value(actual_variance, shuffled_variances, alternative='greater')
    
    if verbose:
        print(f"Mean shuffled variance: {mean_shuffled_variance:.4f}")
        print(f"Variance ratio: {variance_ratio:.4f}")
        print(f"Z-score: {z_score:.2f}σ (p = {p_value:.8f})")
    
    # Create visualization if requested
    if visualizer:
        # Plot the power spectrum
        visualizer.plot_spectrum(
            ell, power, 
            title="CMB Power Spectrum", 
            filename="coherence_test_spectrum.png"
        )
        
        # Plot the test results
        visualizer.plot_test_results(
            {
                'observed': actual_variance,
                'random': shuffled_variances,
                'random_mean': mean_shuffled_variance,
                'ratio': variance_ratio,
                'z_score': z_score,
                'p_value': p_value,
                'additional_metrics': {
                    'Variance Ratio': variance_ratio
                }
            },
            "Coherence Test",
            "coherence_test_results.png"
        )
    
    # Return the results
    return {
        'observed': actual_variance,
        'random_mean': mean_shuffled_variance,
        'ratio': variance_ratio,
        'z_score': z_score,
        'p_value': p_value
    }

if __name__ == "__main__":
    # This script can be run directly for testing
    import os
    import sys
    
    # Add the project root to the Python path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    
    from consciousness_field_tests.utils.data_loader import get_data_loader
    from consciousness_field_tests.utils.visualization import get_visualizer
    from consciousness_field_tests.utils.statistics import get_statistical_analyzer
    
    print("=== COHERENCE TEST ===")
    
    # Initialize utilities
    data_loader = get_data_loader()
    visualizer = get_visualizer()
    stats_analyzer = get_statistical_analyzer()
    
    # Run the test
    results = run_test(
        data_loader=data_loader,
        visualizer=visualizer,
        stats_analyzer=stats_analyzer,
        monte_carlo_sims=1000,
        verbose=True
    )
    
    # Print results
    print("\n=== RESULTS ===")
    print(f"Actual variance: {results['observed']:.4f}")
    print(f"Mean shuffled variance: {results['random_mean']:.4f}")
    print(f"Variance ratio: {results['ratio']:.4f}")
    print(f"Z-score: {results['z_score']:.2f}")
    print(f"P-value: {results['p_value']:.8f}")
    
    # Calculate phi-optimality
    phi_optimality = stats_analyzer.calculate_phi_optimality(results['observed'], results['random_mean'])
    print(f"Phi-optimality: {phi_optimality:.4f}")
    
    print("\nTest complete!")
