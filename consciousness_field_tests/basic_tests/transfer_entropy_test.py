#!/usr/bin/env python3
"""
Transfer Entropy test for Cosmic Consciousness Analysis.

This test analyzes if information transfer across scales follows golden ratio patterns in the CMB data.
"""

import numpy as np
from scipy import stats

def run_test(data_loader, visualizer, stats_analyzer, monte_carlo_sims=1000, verbose=False):
    """
    Run the transfer entropy test.
    
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
    
    # Get golden ratio multipoles
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio
    gr_multipoles = data_loader.get_golden_ratio_multipoles()
    
    if verbose:
        print("Running transfer entropy test...")
        print(f"Golden ratio multipoles: {gr_multipoles}")
    
    # Define scales based on powers of the golden ratio
    phi_scales = []
    for i in range(len(gr_multipoles) - 1):
        scale_indices = np.where((ell >= gr_multipoles[i]) & (ell < gr_multipoles[i+1]))[0]
        if len(scale_indices) > 0:
            phi_scales.append(power[scale_indices])
    
    # Calculate transfer entropy between adjacent phi-related scales
    phi_transfer_entropy = 0
    for i in range(len(phi_scales) - 1):
        if len(phi_scales[i]) > 0 and len(phi_scales[i+1]) > 0:
            # Ensure equal length by truncating or padding
            min_len = min(len(phi_scales[i]), len(phi_scales[i+1]))
            source = phi_scales[i][:min_len]
            target = phi_scales[i+1][:min_len]
            
            # Calculate transfer entropy
            te = stats_analyzer.calculate_transfer_entropy(source, target)
            phi_transfer_entropy += te
    
    # Average the transfer entropy
    if len(phi_scales) > 1:
        phi_transfer_entropy /= (len(phi_scales) - 1)
    
    # Calculate transfer entropy between random scales
    random_scales = []
    random_indices = np.random.choice(len(ell), size=len(gr_multipoles), replace=False)
    random_indices.sort()
    
    for i in range(len(random_indices) - 1):
        scale_indices = np.where((ell >= ell[random_indices[i]]) & (ell < ell[random_indices[i+1]]))[0]
        if len(scale_indices) > 0:
            random_scales.append(power[scale_indices])
    
    # Calculate transfer entropy between adjacent random scales
    random_transfer_entropy = 0
    for i in range(len(random_scales) - 1):
        if len(random_scales[i]) > 0 and len(random_scales[i+1]) > 0:
            # Ensure equal length by truncating or padding
            min_len = min(len(random_scales[i]), len(random_scales[i+1]))
            source = random_scales[i][:min_len]
            target = random_scales[i+1][:min_len]
            
            # Calculate transfer entropy
            te = stats_analyzer.calculate_transfer_entropy(source, target)
            random_transfer_entropy += te
    
    # Average the transfer entropy
    if len(random_scales) > 1:
        random_transfer_entropy /= (len(random_scales) - 1)
    
    # Monte Carlo simulations
    shuffled_transfer_entropies = []
    for i in range(monte_carlo_sims):
        # Shuffle the power spectrum
        shuffled_power = np.random.permutation(power)
        
        # Define scales based on powers of the golden ratio
        shuffled_phi_scales = []
        for j in range(len(gr_multipoles) - 1):
            scale_indices = np.where((ell >= gr_multipoles[j]) & (ell < gr_multipoles[j+1]))[0]
            if len(scale_indices) > 0:
                shuffled_phi_scales.append(shuffled_power[scale_indices])
        
        # Calculate transfer entropy between adjacent phi-related scales
        shuffled_te = 0
        for j in range(len(shuffled_phi_scales) - 1):
            if len(shuffled_phi_scales[j]) > 0 and len(shuffled_phi_scales[j+1]) > 0:
                # Ensure equal length by truncating or padding
                min_len = min(len(shuffled_phi_scales[j]), len(shuffled_phi_scales[j+1]))
                source = shuffled_phi_scales[j][:min_len]
                target = shuffled_phi_scales[j+1][:min_len]
                
                # Calculate transfer entropy
                te = stats_analyzer.calculate_transfer_entropy(source, target)
                shuffled_te += te
        
        # Average the transfer entropy
        if len(shuffled_phi_scales) > 1:
            shuffled_te /= (len(shuffled_phi_scales) - 1)
        
        shuffled_transfer_entropies.append(shuffled_te)
    
    # Calculate statistics
    mean_shuffled_te = np.mean(shuffled_transfer_entropies)
    te_ratio = phi_transfer_entropy / mean_shuffled_te if mean_shuffled_te != 0 else 1.0
    
    # Calculate z-score and p-value
    z_score = stats_analyzer.calculate_z_score(phi_transfer_entropy, shuffled_transfer_entropies)
    p_value = stats_analyzer.calculate_p_value(phi_transfer_entropy, shuffled_transfer_entropies, alternative='greater')
    
    if verbose:
        print(f"Phi-based scales transfer entropy: {phi_transfer_entropy:.2f}")
        print(f"Random scales transfer entropy: {random_transfer_entropy:.2f}")
        print(f"Mean shuffled transfer entropy: {mean_shuffled_te:.2f}")
        print(f"Transfer entropy ratio: {te_ratio:.2f}x")
        print(f"Z-score: {z_score:.2f}σ (p = {p_value:.8f})")
    
    # Create visualization if requested
    if visualizer:
        # Plot the test results
        visualizer.plot_test_results(
            {
                'observed': phi_transfer_entropy,
                'random': shuffled_transfer_entropies,
                'random_mean': mean_shuffled_te,
                'ratio': te_ratio,
                'z_score': z_score,
                'p_value': p_value,
                'additional_metrics': {
                    'Random Scales TE': random_transfer_entropy,
                    'TE Ratio': te_ratio
                }
            },
            "Transfer Entropy Test",
            "transfer_entropy_test_results.png"
        )
    
    # Return the results
    return {
        'observed': phi_transfer_entropy,
        'random_mean': mean_shuffled_te,
        'ratio': te_ratio,
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
    
    print("=== TRANSFER ENTROPY TEST ===")
    
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
    print(f"Phi-based scales transfer entropy: {results['observed']:.4f}")
    print(f"Mean shuffled transfer entropy: {results['random_mean']:.4f}")
    print(f"Transfer entropy ratio: {results['ratio']:.2f}x")
    print(f"Z-score: {results['z_score']:.2f}")
    print(f"P-value: {results['p_value']:.8f}")
    
    # Calculate phi-optimality
    phi_optimality = stats_analyzer.calculate_phi_optimality(results['observed'], results['random_mean'])
    print(f"Phi-optimality: {phi_optimality:.4f}")
    
    print("\nTest complete!")
