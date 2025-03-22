#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Scale Transition Test Runner
----------------------------
This script demonstrates how to use the ScaleTransitionTest class
to analyze scale transitions in CMB data.
"""

from scale_transition_test import ScaleTransitionTest
import matplotlib.pyplot as plt
import numpy as np
import time

def run_test(seed=None, scales=None):
    """
    Run the scale transition test with optional parameters.
    
    Args:
        seed (int, optional): Random seed for reproducibility
        scales (list, optional): Custom scales to analyze
    
    Returns:
        dict: Test results
    """
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
    
    # Create the test instance
    test = ScaleTransitionTest()
    
    # Set custom scales if provided
    if scales is not None:
        test.scales = scales
    
    # Run the test
    start_time = time.time()
    results = test.run_test()
    execution_time = time.time() - start_time
    
    print("Test completed in {:.2f} seconds.".format(execution_time))
    
    # Generate report and visualizations
    test.generate_report()
    test.visualize_results()
    
    return results

def analyze_phi_dominance(results):
    """
    Perform additional analysis on phi dominance across scales.
    
    Args:
        results (dict): Test results from ScaleTransitionTest
    """
    scales = results['scales']
    scale_results = results['scale_results']
    
    # Count phi dominance at each scale
    phi_dominance = []
    for scale in scales:
        if scale in scale_results:
            phi_count = 0
            total = 0
            
            for metric in ['LAMINARITY', 'POWER_LAW', 'COHERENCE', 'INFORMATION_INTEGRATION', 'TRANSFER_ENTROPY']:
                if metric.lower() in scale_results[scale]:
                    if scale_results[scale][metric.lower()]['best_constant'] == 'phi':
                        phi_count += 1
                    total += 1
            
            if total > 0:
                phi_dominance.append((scale, phi_count / float(total)))
    
    # Plot additional phi dominance visualization
    plt.figure(figsize=(10, 6))
    scales_vals, dominance_vals = zip(*phi_dominance)
    
    # Create a more detailed visualization
    plt.bar(range(len(scales_vals)), dominance_vals, color='skyblue')
    plt.xticks(range(len(scales_vals)), [str(s) for s in scales_vals], rotation=45)
    plt.xlabel('Scale')
    plt.ylabel('Phi Dominance (fraction of metrics)')
    plt.title('Golden Ratio Dominance Across Scales')
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='50% Threshold')
    plt.tight_layout()
    plt.savefig('phi_dominance_by_scale.png', dpi=300)
    print("Additional phi dominance visualization saved as 'phi_dominance_by_scale.png'")

if __name__ == "__main__":
    # Run with default parameters
    print("Running Scale Transition Test with default parameters...")
    results = run_test(seed=42)  # Use seed 42 for reproducibility
    
    # Perform additional analysis
    analyze_phi_dominance(results)
    
    print("\nTest complete. Results saved to scale_transition_results.png and phi_dominance_by_scale.png")
