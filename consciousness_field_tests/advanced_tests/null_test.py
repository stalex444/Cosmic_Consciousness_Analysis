#!/usr/bin/env python3
"""
Null test for Cosmic Consciousness Analysis.

This test serves as a control to validate the analysis methodology by testing
if the analysis can correctly identify the absence of conscious organization
in randomly generated data.
"""

import numpy as np
from scipy import stats

def run_test(data_loader, visualizer, stats_analyzer, monte_carlo_sims=1000, verbose=False):
    """
    Run the null test.
    
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
    
    if verbose:
        print("Running null test...")
        print(f"Generating random data with same statistical properties as the original data...")
    
    # Generate random data with the same mean and standard deviation
    mean_power = np.mean(power)
    std_power = np.std(power)
    random_power = np.random.normal(mean_power, std_power, size=len(power))
    
    # Get golden ratio multipoles
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio
    gr_multipoles = data_loader.get_golden_ratio_multipoles()
    
    # Calculate the mean power at golden ratio multipoles
    gr_indices = []
    for gr_ell in gr_multipoles:
        # Find the closest multipole
        idx = np.argmin(np.abs(ell - gr_ell))
        gr_indices.append(idx)
    
    # Calculate the mean power at golden ratio multipoles for random data
    random_gr_power = np.mean(random_power[gr_indices])
    
    # Calculate the mean power at non-golden ratio multipoles for random data
    non_gr_indices = list(set(range(len(ell))) - set(gr_indices))
    random_non_gr_power = np.mean(random_power[non_gr_indices])
    
    # Calculate the ratio of golden ratio to non-golden ratio power for random data
    random_ratio = random_gr_power / random_non_gr_power if random_non_gr_power != 0 else 1.0
    
    if verbose:
        print(f"Random GR power: {random_gr_power:.4f}")
        print(f"Random non-GR power: {random_non_gr_power:.4f}")
        print(f"Random ratio: {random_ratio:.4f}")
    
    # Monte Carlo simulations
    shuffled_ratios = []
    for i in range(monte_carlo_sims):
        # Generate new random data
        shuffled_power = np.random.normal(mean_power, std_power, size=len(power))
        
        # Calculate the mean power at golden ratio multipoles
        shuffled_gr_power = np.mean(shuffled_power[gr_indices])
        
        # Calculate the mean power at non-golden ratio multipoles
        shuffled_non_gr_power = np.mean(shuffled_power[non_gr_indices])
        
        # Calculate the ratio
        shuffled_ratio = shuffled_gr_power / shuffled_non_gr_power if shuffled_non_gr_power != 0 else 1.0
        
        shuffled_ratios.append(shuffled_ratio)
    
    # Calculate statistics
    mean_shuffled_ratio = np.mean(shuffled_ratios)
    ratio_of_ratios = random_ratio / mean_shuffled_ratio if mean_shuffled_ratio != 0 else 1.0
    
    # Calculate z-score and p-value
    z_score = stats_analyzer.calculate_z_score(random_ratio, shuffled_ratios)
    p_value = stats_analyzer.calculate_p_value(random_ratio, shuffled_ratios, alternative='two-sided')
    
    if verbose:
        print(f"Mean shuffled ratio: {mean_shuffled_ratio:.4f}")
        print(f"Ratio of ratios: {ratio_of_ratios:.4f}")
        print(f"Z-score: {z_score:.2f}σ (p = {p_value:.8f})")
    
    # Create visualization if requested
    if visualizer:
        # Plot the original and random power spectra
        visualizer.plot_spectra_comparison(
            ell, power, random_power,
            title="Original vs Random Power Spectrum",
            label1="Original",
            label2="Random",
            filename="null_test_spectra.png"
        )
        
        # Plot the test results
        visualizer.plot_test_results(
            {
                'observed': random_ratio,
                'random': shuffled_ratios,
                'random_mean': mean_shuffled_ratio,
                'ratio': ratio_of_ratios,
                'z_score': z_score,
                'p_value': p_value,
                'additional_metrics': {
                    'Random GR Power': random_gr_power,
                    'Random Non-GR Power': random_non_gr_power
                }
            },
            "Null Test",
            "null_test_results.png"
        )
    
    # Return the results
    return {
        'observed': random_ratio,
        'random_mean': mean_shuffled_ratio,
        'ratio': ratio_of_ratios,
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
    
    print("=== NULL TEST ===")
    
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
    print(f"Random ratio: {results['observed']:.4f}")
    print(f"Mean shuffled ratio: {results['random_mean']:.4f}")
    print(f"Ratio of ratios: {results['ratio']:.4f}")
    print(f"Z-score: {results['z_score']:.2f}")
    print(f"P-value: {results['p_value']:.8f}")
    
    # Calculate phi-optimality
    phi_optimality = stats_analyzer.calculate_phi_optimality(results['observed'], results['random_mean'])
    print(f"Phi-optimality: {phi_optimality:.4f}")
    
    # Interpret the results
    print("\nInterpretation:")
    if p_value > 0.05:
        print("The null test passed: random data does not show significant golden ratio patterns.")
        print("This validates the analysis methodology.")
    else:
        print("The null test failed: random data shows significant golden ratio patterns.")
        print("This suggests that the analysis methodology may need refinement.")
    
    print("\nTest complete!")
