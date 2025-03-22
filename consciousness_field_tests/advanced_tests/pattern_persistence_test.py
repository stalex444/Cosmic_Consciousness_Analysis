#!/usr/bin/env python3
"""
Pattern Persistence Test Module.

This test examines how consistent golden ratio patterns are across different subsets
of the CMB data, compared to random patterns.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def run_test(data_loader, visualizer, stats_analyzer, output_dir=None, verbose=True):
    """
    Run the pattern persistence test.
    
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
        print("=== PATTERN PERSISTENCE TEST ===")
        start_time = time.time()
    
    # Get data
    ell = data_loader.data['ell']
    ee_power = data_loader.data['ee_power']
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio
    
    # Define parameters for the test
    num_subsets = 10  # Number of data subsets to test
    subset_fraction = 0.8  # Fraction of data to include in each subset
    num_patterns = 20  # Number of patterns to test in each subset
    
    # Function to calculate pattern strength in a subset
    def calculate_pattern_strength(subset_ell, subset_power, pattern_ratio):
        pattern_strengths = []
        
        for i in range(len(subset_ell) - 1):
            for j in range(i + 1, len(subset_ell)):
                # Check if the ratio between multipoles is close to the pattern ratio
                ell_ratio = subset_ell[j] / subset_ell[i]
                if 0.9 * pattern_ratio <= ell_ratio <= 1.1 * pattern_ratio:
                    # Calculate correlation between powers
                    power_ratio = subset_power[j] / subset_power[i] if subset_power[i] != 0 else 0
                    # Measure how close the power ratio is to 1 (perfect correlation)
                    # or to pattern_ratio (perfect scaling relationship)
                    strength1 = 1 / (1 + abs(power_ratio - 1))
                    strength2 = 1 / (1 + abs(power_ratio - pattern_ratio))
                    # Take the maximum of these two measures
                    pattern_strengths.append(max(strength1, strength2))
        
        return np.mean(pattern_strengths) if pattern_strengths else 0
    
    # Test pattern persistence across different subsets
    gr_pattern_strengths = []
    random_pattern_strengths = []
    
    for i in range(num_subsets):
        # Create a random subset of the data
        subset_indices = np.random.choice(
            len(ell), 
            size=int(subset_fraction * len(ell)), 
            replace=False
        )
        subset_indices.sort()  # Keep indices in order
        
        subset_ell = ell[subset_indices]
        subset_power = ee_power[subset_indices]
        
        # Calculate golden ratio pattern strength
        gr_strength = calculate_pattern_strength(subset_ell, subset_power, phi)
        gr_pattern_strengths.append(gr_strength)
        
        # Calculate random pattern strengths for comparison
        subset_random_strengths = []
        for _ in range(num_patterns):
            # Use a random ratio between 1.1 and 2.5
            random_ratio = 1.1 + 1.4 * np.random.random()
            random_strength = calculate_pattern_strength(subset_ell, subset_power, random_ratio)
            subset_random_strengths.append(random_strength)
        
        # Average the random pattern strengths for this subset
        random_pattern_strengths.append(np.mean(subset_random_strengths))
    
    # Calculate mean pattern strengths
    mean_gr_strength = np.mean(gr_pattern_strengths)
    mean_random = np.mean(random_pattern_strengths)
    
    # Calculate statistical significance
    z_score, p_value = stats_analyzer.calculate_z_score(mean_gr_strength, random_pattern_strengths)
    
    # Calculate pattern persistence (inverse of coefficient of variation)
    # Lower variance = higher persistence
    gr_variance = np.var(gr_pattern_strengths)
    random_variance = np.var(random_pattern_strengths)
    
    # Persistence ratio: how much more consistent are GR patterns vs random
    # Higher ratio means GR patterns are more persistent
    persistence_ratio = random_variance / gr_variance if gr_variance > 0 else float('inf')
    
    # Calculate strength ratio
    strength_ratio = mean_gr_strength / mean_random if mean_random > 0 else float('inf')
    
    # Calculate phi optimality
    phi_optimality = stats_analyzer.calculate_phi_optimality(strength_ratio, 1.0)
    
    # Interpret results
    if p_value < 0.01:
        significance = "highly significant"
    elif p_value < 0.05:
        significance = "significant"
    elif p_value < 0.1:
        significance = "marginally significant"
    else:
        significance = "not significant"
    
    if strength_ratio > 2:
        effect = "strong"
    elif strength_ratio > 1.5:
        effect = "moderate"
    elif strength_ratio > 1.1:
        effect = "weak"
    else:
        effect = "negligible"
    
    persistence_desc = "highly persistent" if persistence_ratio > 2 else \
                      "moderately persistent" if persistence_ratio > 1.5 else \
                      "slightly persistent" if persistence_ratio > 1.1 else "not persistent"
    
    # Create visualization
    if visualizer is not None:
        plt.figure(figsize=(12, 6))
        
        # Plot 1: Strength comparison
        plt.subplot(1, 2, 1)
        plt.bar(['GR Patterns', 'Random'], [mean_gr_strength, mean_random], color=['gold', 'gray'], alpha=0.7)
        plt.ylabel('Pattern Strength')
        plt.title(f'Pattern Strength: {strength_ratio:.2f}x stronger for GR patterns (p={p_value:.4f})')
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Persistence comparison
        plt.subplot(1, 2, 2)
        # Higher persistence ratio means lower variance in GR patterns compared to random
        plt.bar(['GR Patterns', 'Random'], [1, persistence_ratio], color=['gold', 'gray'], alpha=0.7)
        plt.ylabel('Relative Variance')
        plt.title(f'Pattern Persistence: {persistence_ratio:.2f}x more consistent')
        plt.grid(True, alpha=0.3)
        
        # Save the figure
        plt.tight_layout()
        
        if output_dir:
            output_path = os.path.join(output_dir, 'pattern_persistence.png')
        else:
            output_path = 'pattern_persistence.png'
            
        plt.savefig(output_path, dpi=300)
        
        if verbose:
            print(f"Visualization saved to '{output_path}'")
    
    # Save results to file
    if output_dir:
        results_path = os.path.join(output_dir, 'pattern_persistence_results.txt')
    else:
        results_path = 'pattern_persistence_results.txt'
        
    with open(results_path, 'w') as f:
        f.write("=== PATTERN PERSISTENCE TEST RESULTS ===\n\n")
        f.write(f"Mean GR pattern strength: {mean_gr_strength:.4f}\n")
        f.write(f"Mean random pattern strength: {mean_random:.4f}\n")
        f.write(f"Strength ratio: {strength_ratio:.2f}x\n")
        f.write(f"Z-score: {z_score:.2f}\n")
        f.write(f"P-value: {p_value:.8f}\n")
        f.write(f"Persistence ratio: {persistence_ratio:.2f}\n\n")
        f.write(f"Interpretation: The test shows a {effect} effect that is {significance}.\n")
        f.write(f"Golden ratio patterns are {persistence_ratio:.2f}x more consistent across different ")
        f.write(f"subsets of the data than random patterns, indicating they are {persistence_desc}.\n")
    
    if verbose:
        print(f"Results saved to '{results_path}'")
        print(f"Test completed in {time.time() - start_time:.2f} seconds.")
    
    # Prepare results dictionary
    results = {
        'test_name': 'Pattern Persistence Test',
        'mean_gr_strength': mean_gr_strength,
        'mean_random': mean_random,
        'z_score': z_score,
        'p_value': p_value,
        'strength_ratio': strength_ratio,
        'persistence_ratio': persistence_ratio,
        'phi_optimality': phi_optimality,
        'interpretation': f"{effect} effect, {significance}, {persistence_desc}",
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
    stats_analyzer = get_statistical_analyzer()
    
    # Run test
    results = run_test(data_loader, visualizer, stats_analyzer, verbose=True)
    
    # Print key results
    print("\n=== KEY RESULTS ===")
    print(f"Pattern Strength Ratio: {results['strength_ratio']:.2f}x")
    print(f"Persistence Ratio: {results['persistence_ratio']:.2f}x")
    print(f"Phi optimality: {results['phi_optimality']:.4f}")
    print(f"P-value: {results['p_value']:.6f}")
