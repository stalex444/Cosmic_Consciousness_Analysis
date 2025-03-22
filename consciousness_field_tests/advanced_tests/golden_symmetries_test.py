#!/usr/bin/env python3
"""
Golden Symmetries Test Module.

This test analyzes symmetries in the CMB data related to the golden ratio
and compares with other mathematical constants.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
import time

def run_test(data_loader, visualizer, stats_analyzer, output_dir=None, verbose=True):
    """
    Run the golden symmetries test.
    
    Parameters:
    -----------
    data_loader : DataLoader
        Instance of DataLoader to access CMB data.
    visualizer : Visualizer
        Instance of Visualizer to create plots.
    stats_analyzer : StatisticalAnalyzer
        Instance of StatisticalAnalyzer for statistical calculations.
    output_dir : str, optional
        Directory to save output files. If None, files are saved in current directory.
    verbose : bool, optional
        Whether to print detailed output.
        
    Returns:
    --------
    dict
        Dictionary containing test results.
    """
    if verbose:
        print("=== GOLDEN SYMMETRIES TEST ===")
        start_time = time.time()
    
    # Get data
    ell = data_loader.data['ell']
    ee_power = data_loader.data['ee_power']
    
    # Calculate golden ratio symmetries
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio
    phi_fold = []
    
    for i in range(len(ell)):
        l_value = ell[i]
        power = ee_power[i]
        
        # Find closest ell values to l*phi and l/phi
        l_phi = l_value * phi
        idx_phi = np.abs(ell - l_phi).argmin()
        
        l_inv_phi = l_value / phi
        idx_inv_phi = np.abs(ell - l_inv_phi).argmin()
        
        if idx_phi < len(ell) and idx_inv_phi < len(ell):
            power_phi = ee_power[idx_phi]
            power_inv_phi = ee_power[idx_inv_phi]
            
            # Calculate expected power based on geometric mean
            expected_power = np.sqrt(power_phi * power_inv_phi)
            symmetry_ratio = power / expected_power if expected_power != 0 else 1
            
            phi_fold.append(abs(1 - symmetry_ratio))
    
    # Calculate mean asymmetry for golden ratio
    mean_asymmetry = np.mean(phi_fold)
    
    # Calculate alternative constants symmetries
    alternative_constants = [np.e, np.pi, 2]
    alt_asymmetries = []
    
    for constant in alternative_constants:
        alt_fold = []
        for i in range(len(ell)):
            l_value = ell[i]
            power = ee_power[i]
            
            l_const = l_value * constant
            idx_const = np.abs(ell - l_const).argmin()
            
            l_inv_const = l_value / constant
            idx_inv_const = np.abs(ell - l_inv_const).argmin()
            
            if idx_const < len(ell) and idx_inv_const < len(ell):
                power_const = ee_power[idx_const]
                power_inv_const = ee_power[idx_inv_const]
                
                expected_power = np.sqrt(power_const * power_inv_const)
                symmetry_ratio = power / expected_power if expected_power != 0 else 1
                
                alt_fold.append(abs(1 - symmetry_ratio))
        
        alt_asymmetries.append(np.mean(alt_fold))
    
    # Calculate mean asymmetry for alternative constants
    mean_alternative = np.mean(alt_asymmetries)
    
    # Calculate z-score and p-value
    z_score, p_value = stats_analyzer.calculate_z_score(mean_asymmetry, alt_asymmetries)
    
    # Calculate symmetry ratio (how much better is golden ratio compared to alternatives)
    symmetry_ratio = mean_alternative / mean_asymmetry if mean_asymmetry > 0 else 1.0
    
    # Calculate phi optimality
    phi_optimality = stats_analyzer.calculate_phi_optimality(symmetry_ratio, 1.0)
    
    # Interpret phi optimality
    if phi_optimality >= 0.75:
        phi_interpretation = "strong"
    elif phi_optimality >= 0.5:
        phi_interpretation = "moderate"
    elif phi_optimality >= 0.25:
        phi_interpretation = "weak"
    elif phi_optimality >= 0:
        phi_interpretation = "minimal"
    elif phi_optimality >= -0.25:
        phi_interpretation = "minimal negative"
    elif phi_optimality >= -0.5:
        phi_interpretation = "weak negative"
    elif phi_optimality >= -0.75:
        phi_interpretation = "moderate negative"
    else:
        phi_interpretation = "strong negative"
    
    # Create visualization
    if visualizer is not None:
        # Create bar chart comparing golden ratio and alternative constants
        plt.figure(figsize=(10, 6))
        
        # Define constants and their names
        constants = ["Golden Ratio (φ)", "e", "π", "2"]
        
        # Combine asymmetries
        asymmetries = [mean_asymmetry] + alt_asymmetries
        
        # Plot bar chart
        plt.bar(constants, asymmetries, color=['gold', 'gray', 'gray', 'gray'], alpha=0.7)
        plt.ylabel('Mean Asymmetry (lower is better)')
        plt.title(f'Golden Ratio Symmetry Test: φ-optimality = {phi_optimality:.4f} ({phi_interpretation})')
        
        # Add phi optimality as text annotation
        plt.text(0.5, 0.9, f'φ-optimality: {phi_optimality:.4f}', 
                 horizontalalignment='center',
                 transform=plt.gca().transAxes,
                 bbox=dict(facecolor='white', alpha=0.8))
        
        # Add p-value and symmetry ratio
        plt.text(0.5, 0.82, f'p-value: {p_value:.4f}', 
                 horizontalalignment='center',
                 transform=plt.gca().transAxes,
                 bbox=dict(facecolor='white', alpha=0.8))
        
        plt.text(0.5, 0.74, f'Symmetry ratio: {symmetry_ratio:.2f}x', 
                 horizontalalignment='center',
                 transform=plt.gca().transAxes,
                 bbox=dict(facecolor='white', alpha=0.8))
        
        plt.grid(True, alpha=0.3)
        
        # Save and show the figure
        plt.tight_layout()
        
        if output_dir:
            output_path = os.path.join(output_dir, 'golden_symmetries_test.png')
        else:
            output_path = 'golden_symmetries_test.png'
            
        plt.savefig(output_path)
        
        if verbose:
            print(f"Visualization saved to '{output_path}'")
    
    # Save results to file
    if output_dir:
        results_path = os.path.join(output_dir, 'golden_symmetries_results.txt')
    else:
        results_path = 'golden_symmetries_results.txt'
        
    with open(results_path, 'w') as f:
        f.write("=== GOLDEN SYMMETRIES TEST RESULTS ===\n\n")
        f.write(f"Mean asymmetry for golden ratio: {mean_asymmetry:.4f}\n")
        f.write(f"Mean asymmetry for alternative constants: {mean_alternative:.4f}\n")
        f.write(f"Z-score: {z_score:.4f}\n")
        f.write(f"P-value: {p_value:.4f}\n")
        f.write(f"Symmetry ratio: {symmetry_ratio:.2f}x\n")
        f.write(f"Phi optimality: {phi_optimality:.4f}\n\n")
        f.write(f"Phi optimality interpretation: {phi_interpretation}\n")
        f.write(f"The golden symmetries test shows a {phi_interpretation} alignment with golden ratio optimality.\n")
    
    if verbose:
        print(f"Results saved to '{results_path}'")
        print(f"Test completed in {time.time() - start_time:.2f} seconds.")
    
    # Prepare results dictionary
    results = {
        'test_name': 'Golden Symmetries Test',
        'mean_asymmetry': mean_asymmetry,
        'mean_alternative': mean_alternative,
        'z_score': z_score,
        'p_value': p_value,
        'symmetry_ratio': symmetry_ratio,
        'phi_optimality': phi_optimality,
        'phi_interpretation': phi_interpretation,
        'visualization_path': output_path if visualizer is not None else None,
        'results_path': results_path
    }
    
    return results

if __name__ == "__main__":
    # This allows the module to be run directly for testing
    from consciousness_field_tests.utils.data_loader import get_data_loader
    from consciousness_field_tests.utils.visualization import get_visualizer
    from consciousness_field_tests.utils.statistics import get_statistical_analyzer
    
    # Set data directory
    data_dir = os.path.join(os.getcwd(), 'planck_data')
    
    # Initialize utilities
    data_loader = get_data_loader(data_dir=data_dir)
    visualizer = get_visualizer()
    stats_analyzer = get_statistical_analyzer(monte_carlo_sims=1000)
    
    # Run test
    results = run_test(data_loader, visualizer, stats_analyzer, verbose=True)
    
    # Print key results
    print("\n=== KEY RESULTS ===")
    print(f"Phi optimality: {results['phi_optimality']:.4f} ({results['phi_interpretation']})")
    print(f"P-value: {results['p_value']:.4f}")
