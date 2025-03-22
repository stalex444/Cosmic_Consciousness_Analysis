#!/usr/bin/env python3
"""
Multi-Scale Coherence Test Module.

This test analyzes how coherence varies across different scales in the CMB data
and if it follows golden ratio patterns.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def run_test(data_loader, visualizer, stats_analyzer, output_dir=None, verbose=True):
    """
    Run the multi-scale coherence test.
    
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
        print("=== MULTI-SCALE COHERENCE TEST ===")
        start_time = time.time()
    
    # Get data
    ell = data_loader.data['ell']
    ee_power = data_loader.data['ee_power']
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio
    
    # Define scale bins (logarithmically spaced)
    min_ell = min(ell)
    max_ell = max(ell)
    num_bins = 20
    
    scale_bins = np.logspace(np.log10(min_ell), np.log10(max_ell), num_bins+1)
    scale_coherences = []
    
    # Calculate coherence for each scale bin
    for i in range(num_bins):
        bin_start = scale_bins[i]
        bin_end = scale_bins[i+1]
        
        # Get multipoles in this bin
        bin_indices = np.where((ell >= bin_start) & (ell < bin_end))[0]
        
        if len(bin_indices) > 1:
            # Calculate coherence as the inverse of the coefficient of variation
            bin_powers = ee_power[bin_indices]
            mean_power = np.mean(bin_powers)
            std_power = np.std(bin_powers)
            
            # Coherence = mean/std (higher means more coherent)
            coherence = mean_power / std_power if std_power > 0 else float('inf')
            
            # Use the midpoint of the bin as the scale
            scale_midpoint = np.sqrt(bin_start * bin_end)
            
            scale_coherences.append((scale_midpoint, coherence))
    
    # Calculate ratios between adjacent coherence values
    coherence_ratios = []
    for i in range(len(scale_coherences)-1):
        if scale_coherences[i+1][1] != 0:
            ratio = scale_coherences[i][1] / scale_coherences[i+1][1]
            coherence_ratios.append(ratio)
    
    # Calculate mean deviation from phi
    phi_deviations = [abs(ratio - phi) for ratio in coherence_ratios]
    mean_phi_deviation = np.mean(phi_deviations)
    
    # Calculate mean random deviation (Monte Carlo)
    random_deviations = []
    for _ in range(1000):  # 1000 random trials
        # Shuffle the coherence values
        shuffled_coherences = [c[1] for c in scale_coherences]
        np.random.shuffle(shuffled_coherences)
        
        # Calculate ratios for shuffled values
        shuffled_ratios = []
        for i in range(len(shuffled_coherences)-1):
            if shuffled_coherences[i+1] != 0:
                ratio = shuffled_coherences[i] / shuffled_coherences[i+1]
                shuffled_ratios.append(ratio)
        
        # Calculate mean deviation from phi
        if shuffled_ratios:
            random_deviation = np.mean([abs(ratio - phi) for ratio in shuffled_ratios])
            random_deviations.append(random_deviation)
    
    # Calculate statistics
    mean_random = np.mean(random_deviations)
    z_score, p_value = stats_analyzer.calculate_z_score(mean_phi_deviation, random_deviations)
    
    # Calculate optimization ratio (how much better than random)
    # Note: Lower deviation is better, so we invert the ratio
    optimization_ratio = mean_random / mean_phi_deviation if mean_phi_deviation > 0 else float('inf')
    
    # Calculate phi optimality
    phi_optimality = stats_analyzer.calculate_phi_optimality(optimization_ratio, 1.0)
    
    # Interpret phi optimality
    if phi_optimality >= 0.75:
        interpretation = "extremely high"
    elif phi_optimality >= 0.5:
        interpretation = "very high"
    elif phi_optimality >= 0.25:
        interpretation = "high"
    elif phi_optimality >= 0:
        interpretation = "moderate"
    elif phi_optimality >= -0.25:
        interpretation = "slightly negative"
    elif phi_optimality >= -0.5:
        interpretation = "moderately negative"
    elif phi_optimality >= -0.75:
        interpretation = "strongly negative"
    else:
        interpretation = "extremely negative"
    
    # Create visualization
    if visualizer is not None:
        plt.figure(figsize=(12, 10))
        
        # Plot 1: Scale vs Coherence
        plt.subplot(2, 2, 1)
        scales = [s[0] for s in scale_coherences]
        coherences = [s[1] for s in scale_coherences]
        plt.scatter(scales, coherences, alpha=0.7, s=80)
        plt.plot(scales, coherences, 'b-', alpha=0.5)
        plt.xscale('log')
        plt.xlabel('Scale (multipole)')
        plt.ylabel('Coherence')
        plt.title('Coherence across Different Scales')
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Coherence Ratios
        plt.subplot(2, 2, 2)
        plt.hist(coherence_ratios, bins=10, alpha=0.7)
        plt.axvline(x=phi, color='r', linestyle='--', 
                   label=f'Golden Ratio (φ = {phi:.4f})')
        plt.xlabel('Coherence Ratio')
        plt.ylabel('Frequency')
        plt.title('Distribution of Coherence Ratios')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Phi Deviation Comparison
        plt.subplot(2, 2, 3)
        plt.hist(coherence_ratios, bins=10, alpha=0.7, 
                label=f'Actual (mean dev: {mean_phi_deviation:.4f})')
        
        # Generate some random ratios for comparison
        random_coherences = np.random.permutation(coherences)
        random_ratios = []
        for i in range(len(random_coherences)-1):
            if random_coherences[i+1] != 0:
                ratio = random_coherences[i] / random_coherences[i+1]
                random_ratios.append(ratio)
                
        plt.hist(random_ratios, bins=10, alpha=0.4, 
                label=f'Random (mean dev: {mean_random:.4f})')
        plt.axvline(x=phi, color='r', linestyle='--', 
                   label=f'Golden Ratio (φ)')
        plt.xlabel('Ratio Value')
        plt.ylabel('Frequency')
        plt.title('Actual vs Random Coherence Ratios')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Summary
        plt.subplot(2, 2, 4)
        plt.axis('off')
        summary_text = (
            f"MULTI-SCALE COHERENCE TEST\n\n"
            f"Scale bins analyzed: {len(scale_coherences)}\n"
            f"Coherence ratios: {len(coherence_ratios)}\n\n"
            f"Mean deviation from phi: {mean_phi_deviation:.6f}\n"
            f"Mean random deviation: {mean_random:.6f}\n"
            f"Optimization ratio: {optimization_ratio:.2f}x\n\n"
            f"Statistical significance:\n"
            f"Z-score: {z_score:.4f}\n"
            f"P-value: {p_value:.4f}\n\n"
            f"Phi optimality: {phi_optimality:.4f}\n"
            f"Interpretation: {interpretation}\n\n"
            f"Analysis date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        plt.text(0.05, 0.95, summary_text, fontsize=10, 
                va='top', ha='left', transform=plt.gca().transAxes)
        
        plt.tight_layout()
        
        if output_dir:
            output_path = os.path.join(output_dir, 'multi_scale_coherence_results.png')
        else:
            output_path = 'multi_scale_coherence_results.png'
            
        plt.savefig(output_path, dpi=300)
        
        if verbose:
            print(f"Visualization saved to '{output_path}'")
    
    # Save results to file
    if output_dir:
        results_path = os.path.join(output_dir, 'multi_scale_coherence_results.txt')
    else:
        results_path = 'multi_scale_coherence_results.txt'
        
    with open(results_path, 'w') as f:
        f.write("=== MULTI-SCALE COHERENCE ANALYSIS RESULTS ===\n\n")
        f.write(f"1. Scale coherence measurements: {len(scale_coherences)} scale bins\n")
        f.write(f"2. Coherence ratios: {len(coherence_ratios)} ratios\n")
        f.write(f"3. Mean deviation from phi: {mean_phi_deviation:.6f}\n")
        f.write(f"4. Mean random deviation: {mean_random:.6f}\n")
        f.write(f"5. Statistical significance:\n")
        f.write(f"   Z-score: {z_score:.4f}\n")
        f.write(f"   P-value: {p_value:.4f}\n\n")
        f.write(f"6. Optimization ratio: {optimization_ratio:.2f}x\n")
        f.write(f"7. Phi optimality: {phi_optimality:.4f} ({interpretation})\n\n")
        
        f.write("Summary:\n")
        if p_value < 0.05 and phi_optimality > 0:
            f.write("The analysis shows significant evidence that coherence varies across scales in a pattern related to the golden ratio.\n")
        elif phi_optimality > 0:
            f.write("The analysis shows some evidence that coherence varies across scales in a pattern related to the golden ratio, but it does not reach statistical significance.\n")
        else:
            f.write("The analysis does not show evidence that coherence varies across scales in a pattern related to the golden ratio.\n")
    
    if verbose:
        print(f"Results saved to '{results_path}'")
        print(f"Test completed in {time.time() - start_time:.2f} seconds.")
    
    # Prepare results dictionary
    results = {
        'test_name': 'Multi-Scale Coherence Test',
        'scale_coherences': scale_coherences,
        'coherence_ratios': coherence_ratios,
        'mean_phi_deviation': mean_phi_deviation,
        'mean_random': mean_random,
        'z_score': z_score,
        'p_value': p_value,
        'optimization_ratio': optimization_ratio,
        'phi_optimality': phi_optimality,
        'interpretation': interpretation,
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
    stats_analyzer = get_statistical_analyzer(monte_carlo_sims=100)
    
    # Run test
    results = run_test(data_loader, visualizer, stats_analyzer, verbose=True)
    
    # Print key results
    print("\n=== KEY RESULTS ===")
    print(f"Phi optimality: {results['phi_optimality']:.4f} ({results['interpretation']})")
    print(f"P-value: {results['p_value']:.6f}")
    print(f"Optimization ratio: {results['optimization_ratio']:.2f}x")
