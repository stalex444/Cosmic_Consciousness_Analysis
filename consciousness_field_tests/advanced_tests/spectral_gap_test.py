#!/usr/bin/env python3
"""
Spectral Gap Test Module.

This test examines if the spectral gap in CMB data shows golden ratio optimization.
The test analyzes both the spectral gap magnitude and the eigenvalue relationships.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def run_test(data_loader, visualizer, stats_analyzer, output_dir=None, verbose=True):
    """
    Run the spectral gap test.
    
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
        print("=== SPECTRAL GAP TEST ===")
        start_time = time.time()
    
    # Get data
    ell = data_loader.data['ell']
    ee_power = data_loader.data['ee_power']
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio
    
    # Create similarity matrix
    power_spectrum = np.copy(ee_power)
    
    # Replace zeros and negative values with small positive values
    min_positive = np.min(power_spectrum[power_spectrum > 0]) / 10.0 if np.any(power_spectrum > 0) else 1e-10
    power_spectrum[power_spectrum <= 0] = min_positive
    
    n = len(power_spectrum)
    similarity_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            # Calculate similarity as inverse of normalized difference
            diff = abs(power_spectrum[i] - power_spectrum[j])
            mean_val = (power_spectrum[i] + power_spectrum[j]) / 2.0
            if mean_val > 0:
                similarity_matrix[i, j] = 1.0 / (1.0 + diff / mean_val)
            else:
                similarity_matrix[i, j] = 0.0
    
    # Ensure the matrix is symmetric and positive definite
    similarity_matrix = (similarity_matrix + similarity_matrix.T) / 2.0
    np.fill_diagonal(similarity_matrix, 1.0)
    
    # Calculate eigenvalues
    eigenvalues = np.linalg.eigvals(similarity_matrix)
    eigenvalues = np.sort(np.real(eigenvalues))[::-1]  # Sort in descending order, take real part
    
    # Calculate spectral gap (difference between first and second eigenvalues)
    spectral_gap = eigenvalues[0] - eigenvalues[1] if len(eigenvalues) > 1 else 0
    
    # Calculate ratios between consecutive eigenvalues
    eigenvalue_ratios = []
    for i in range(len(eigenvalues)-1):
        if eigenvalues[i+1] != 0:  # Avoid division by zero
            ratio = eigenvalues[i] / eigenvalues[i+1]
            eigenvalue_ratios.append(ratio)
    
    # Calculate mean deviation from phi for eigenvalue ratios
    phi_deviations = []
    for ratio in eigenvalue_ratios:
        # Check deviation from phi and powers of phi
        deviations = [abs(ratio - phi**power) for power in range(1, 4)]
        phi_deviations.append(min(deviations))
    
    mean_phi_deviation = np.mean(phi_deviations) if phi_deviations else 1.0
    
    # Generate random matrices for comparison
    num_random_trials = 1000
    random_gaps = []
    random_deviations = []
    
    for _ in range(num_random_trials):
        # Create random similarity matrix
        random_matrix = np.random.random((n, n))
        random_matrix = (random_matrix + random_matrix.T) / 2.0  # Make symmetric
        np.fill_diagonal(random_matrix, 1.0)
        
        # Calculate eigenvalues
        random_eigenvalues = np.linalg.eigvals(random_matrix)
        random_eigenvalues = np.sort(np.real(random_eigenvalues))[::-1]
        
        # Calculate spectral gap
        random_gap = random_eigenvalues[0] - random_eigenvalues[1] if len(random_eigenvalues) > 1 else 0
        random_gaps.append(random_gap)
        
        # Calculate ratios and phi deviations
        random_ratios = []
        for i in range(len(random_eigenvalues)-1):
            if random_eigenvalues[i+1] != 0:
                ratio = random_eigenvalues[i] / random_eigenvalues[i+1]
                random_ratios.append(ratio)
        
        # Calculate mean deviation from phi
        random_phi_devs = []
        for ratio in random_ratios:
            deviations = [abs(ratio - phi**power) for power in range(1, 4)]
            random_phi_devs.append(min(deviations))
        
        random_deviations.append(np.mean(random_phi_devs) if random_phi_devs else 1.0)
    
    # Calculate statistics
    mean_random_gap = np.mean(random_gaps)
    gap_z_score, gap_p_value = stats_analyzer.calculate_z_score(spectral_gap, random_gaps)
    
    mean_random_dev = np.mean(random_deviations)
    dev_z_score, dev_p_value = stats_analyzer.calculate_z_score(mean_phi_deviation, random_deviations)
    
    # Calculate ratios
    gap_ratio = spectral_gap / mean_random_gap if mean_random_gap > 0 else float('inf')
    
    # For deviation, lower is better, so invert the ratio
    dev_ratio = mean_random_dev / mean_phi_deviation if mean_phi_deviation > 0 else float('inf')
    
    # Calculate phi optimality
    gap_phi_optimality = stats_analyzer.calculate_phi_optimality(gap_ratio, 1.0)
    dev_phi_optimality = stats_analyzer.calculate_phi_optimality(dev_ratio, 1.0)
    
    # Calculate combined phi optimality (average of the two)
    combined_phi_optimality = (gap_phi_optimality + dev_phi_optimality) / 2
    
    # Create visualization
    if visualizer is not None:
        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        # 1. Spectral Gap Visualization
        im = ax1.imshow(similarity_matrix, cmap='viridis')
        ax1.set_title(f'Similarity Matrix with Spectral Gap: {spectral_gap:.4f}\n(φ-optimality: {gap_phi_optimality:.4f})')
        ax1.set_xlabel('Multipole Index')
        ax1.set_ylabel('Multipole Index')
        plt.colorbar(im, ax=ax1, label='Similarity')
        
        # 2. Eigenvalue Ratios Visualization
        # Plot top 10 eigenvalues
        top_n = min(10, len(eigenvalues))
        ax2.bar(range(top_n), eigenvalues[:top_n], color='skyblue', alpha=0.7)
        ax2.set_title(f'Top {top_n} Eigenvalues\n(Phi Deviation: {mean_phi_deviation:.4f}, φ-optimality: {dev_phi_optimality:.4f})')
        ax2.set_xlabel('Eigenvalue Index')
        ax2.set_ylabel('Eigenvalue')
        
        # Add text annotations for phi relationships
        for i in range(min(5, len(eigenvalue_ratios))):
            ratio = eigenvalue_ratios[i]
            deviations = [abs(ratio - phi**power) for power in range(1, 4)]
            min_dev = min(deviations)
            power = deviations.index(min_dev) + 1
            
            ax2.text(i, eigenvalues[i], 
                    f'λ{i}/λ{i+1}={ratio:.2f}\n(φ^{power}={phi**power:.2f})', 
                    ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        if output_dir:
            output_path = os.path.join(output_dir, 'spectral_gap_test.png')
        else:
            output_path = 'spectral_gap_test.png'
            
        plt.savefig(output_path, dpi=300)
        
        if verbose:
            print(f"Visualization saved to '{output_path}'")
    
    # Save results to file
    if output_dir:
        results_path = os.path.join(output_dir, 'spectral_gap_results.txt')
    else:
        results_path = 'spectral_gap_results.txt'
        
    with open(results_path, 'w') as f:
        f.write("=== SPECTRAL GAP TEST RESULTS ===\n\n")
        f.write("1. Spectral Gap Analysis:\n")
        f.write(f"   Spectral gap: {spectral_gap:.4f}\n")
        f.write(f"   Random gap: {mean_random_gap:.4f}\n")
        f.write(f"   Z-score: {gap_z_score:.4f}\n")
        f.write(f"   P-value: {gap_p_value:.4f}\n")
        f.write(f"   Gap ratio: {gap_ratio:.2f}x\n")
        f.write(f"   Phi optimality: {gap_phi_optimality:.4f}\n")
        
        # Interpret gap phi optimality
        if gap_phi_optimality >= 0.75:
            gap_interpretation = "extremely high"
        elif gap_phi_optimality >= 0.5:
            gap_interpretation = "very high"
        elif gap_phi_optimality >= 0.25:
            gap_interpretation = "high"
        elif gap_phi_optimality >= 0:
            gap_interpretation = "moderate"
        elif gap_phi_optimality >= -0.25:
            gap_interpretation = "slightly negative"
        elif gap_phi_optimality >= -0.5:
            gap_interpretation = "moderately negative"
        elif gap_phi_optimality >= -0.75:
            gap_interpretation = "strongly negative"
        else:
            gap_interpretation = "extremely negative"
            
        f.write(f"   Interpretation: {gap_interpretation}\n\n")
        
        f.write("2. Eigenvalue Phi Relationship Analysis:\n")
        f.write(f"   Mean phi deviation: {mean_phi_deviation:.4f}\n")
        f.write(f"   Random deviation: {mean_random_dev:.4f}\n")
        f.write(f"   Z-score: {dev_z_score:.4f}\n")
        f.write(f"   P-value: {dev_p_value:.4f}\n")
        f.write(f"   Deviation ratio: {dev_ratio:.2f}x\n")
        f.write(f"   Phi optimality: {dev_phi_optimality:.4f}\n")
        
        # Interpret dev phi optimality
        if dev_phi_optimality >= 0.75:
            dev_interpretation = "extremely high"
        elif dev_phi_optimality >= 0.5:
            dev_interpretation = "very high"
        elif dev_phi_optimality >= 0.25:
            dev_interpretation = "high"
        elif dev_phi_optimality >= 0:
            dev_interpretation = "moderate"
        elif dev_phi_optimality >= -0.25:
            dev_interpretation = "slightly negative"
        elif dev_phi_optimality >= -0.5:
            dev_interpretation = "moderately negative"
        elif dev_phi_optimality >= -0.75:
            dev_interpretation = "strongly negative"
        else:
            dev_interpretation = "extremely negative"
            
        f.write(f"   Interpretation: {dev_interpretation}\n\n")
        
        f.write("Summary:\n")
        f.write(f"The spectral gap test shows a {gap_interpretation} alignment with golden ratio optimality in the gap structure.\n")
        f.write(f"The eigenvalue ratio analysis shows a {dev_interpretation} alignment with golden ratio patterns in eigenvalue relationships.\n")
    
    if verbose:
        print(f"Results saved to '{results_path}'")
        print(f"Test completed in {time.time() - start_time:.2f} seconds.")
    
    # Prepare results dictionary
    results = {
        'test_name': 'Spectral Gap Test',
        'spectral_gap': spectral_gap,
        'mean_random_gap': mean_random_gap,
        'gap_z_score': gap_z_score,
        'gap_p_value': gap_p_value,
        'gap_ratio': gap_ratio,
        'gap_phi_optimality': gap_phi_optimality,
        'gap_interpretation': gap_interpretation,
        'mean_phi_deviation': mean_phi_deviation,
        'mean_random_dev': mean_random_dev,
        'dev_z_score': dev_z_score,
        'dev_p_value': dev_p_value,
        'dev_ratio': dev_ratio,
        'dev_phi_optimality': dev_phi_optimality,
        'dev_interpretation': dev_interpretation,
        'combined_phi_optimality': combined_phi_optimality,
        'p_value': min(gap_p_value, dev_p_value),  # Use the more significant p-value
        'phi_optimality': combined_phi_optimality,  # Use the combined phi optimality
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
    print(f"Spectral Gap: {results['spectral_gap']:.4f} ({results['gap_phi_optimality']:.4f} φ-optimality)")
    print(f"Eigenvalue Phi Relationships: {results['mean_phi_deviation']:.4f} ({results['dev_phi_optimality']:.4f} φ-optimality)")
    print(f"Combined φ-optimality: {results['combined_phi_optimality']:.4f}")
    print(f"P-value: {results['p_value']:.6f}")
