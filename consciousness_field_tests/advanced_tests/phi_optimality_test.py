#!/usr/bin/env python3
"""
Phi Optimality Test Module.

This test calculates the phi optimality score for the predictive power of golden ratio patterns
in the CMB data.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import time

def run_test(data_loader, visualizer, stats_analyzer, output_dir=None, verbose=True):
    """
    Run the phi optimality test for predictive power analysis.
    
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
        print("=== PHI OPTIMALITY TEST ===")
        start_time = time.time()
    
    # Get data
    ell = data_loader.data['ell']
    ee_power = data_loader.data['ee_power']
    
    # Get golden ratio multipoles
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio
    gr_multipoles = []
    
    # Calculate golden ratio multipoles
    l_base = 2
    while l_base * phi**10 < max(ell):
        for i in range(11):
            l_gr = l_base * phi**i
            if l_gr <= max(ell):
                gr_multipoles.append(l_gr)
        l_base += 1
    
    gr_multipoles = sorted(list(set([int(round(l)) for l in gr_multipoles])))
    
    # Find closest actual multipoles
    gr_indices = [np.abs(ell - l_gr).argmin() for l_gr in gr_multipoles if l_gr <= max(ell)]
    gr_ell = ell[gr_indices]
    gr_power = ee_power[gr_indices]
    
    # Calculate predictive power
    # For each golden ratio multipole, predict the power at the next multipole
    # based on the pattern at the previous multipoles
    predictions = []
    actuals = []
    
    for i in range(2, len(gr_indices)):
        # Use the ratio of the previous two powers to predict the next one
        prev_ratio = gr_power[i-1] / gr_power[i-2] if gr_power[i-2] != 0 else 1
        predicted_power = gr_power[i-1] * prev_ratio
        actual_power = gr_power[i]
        
        predictions.append(predicted_power)
        actuals.append(actual_power)
    
    # Calculate match rate (within 20% is considered a match)
    matches = 0
    for pred, actual in zip(predictions, actuals):
        if abs(pred - actual) / actual < 0.2:  # Within 20%
            matches += 1
    
    match_rate = matches / len(predictions) if predictions else 0
    
    # Generate random predictions for comparison
    random_match_rates = []
    for _ in range(100):  # 100 random trials
        random_indices = np.random.choice(len(ell), size=len(gr_indices), replace=False)
        random_ell = ell[random_indices]
        random_power = ee_power[random_indices]
        
        random_predictions = []
        random_actuals = []
        
        for i in range(2, len(random_indices)):
            prev_ratio = random_power[i-1] / random_power[i-2] if random_power[i-2] != 0 else 1
            predicted_power = random_power[i-1] * prev_ratio
            actual_power = random_power[i]
            
            random_predictions.append(predicted_power)
            random_actuals.append(actual_power)
        
        random_matches = 0
        for pred, actual in zip(random_predictions, random_actuals):
            if abs(pred - actual) / actual < 0.2:  # Within 20%
                random_matches += 1
        
        random_match_rate = random_matches / len(random_predictions) if random_predictions else 0
        random_match_rates.append(random_match_rate)
    
    # Calculate statistics
    mean_random_rate = np.mean(random_match_rates)
    z_score, p_value = stats_analyzer.calculate_z_score(match_rate, random_match_rates)
    
    # Calculate prediction power ratio
    prediction_power = match_rate / mean_random_rate if mean_random_rate > 0 else 1.0
    
    # Calculate phi optimality
    phi_optimality = stats_analyzer.calculate_phi_optimality(match_rate, mean_random_rate)
    
    # Interpret phi optimality
    if phi_optimality >= 0.75:
        phi_interpretation = "extremely high"
    elif phi_optimality >= 0.5:
        phi_interpretation = "very high"
    elif phi_optimality >= 0.25:
        phi_interpretation = "high"
    elif phi_optimality >= 0:
        phi_interpretation = "moderate"
    elif phi_optimality >= -0.25:
        phi_interpretation = "slightly negative"
    elif phi_optimality >= -0.5:
        phi_interpretation = "moderately negative"
    elif phi_optimality >= -0.75:
        phi_interpretation = "strongly negative"
    else:
        phi_interpretation = "extremely negative"
    
    # Create visualization
    if visualizer is not None:
        plt.figure(figsize=(10, 6))
        
        # Create bar chart comparing GR and random match rates with phi optimality
        plt.bar(['GR Predictions', 'Random Predictions'], [match_rate, mean_random_rate], 
                color=['gold', 'gray'], alpha=0.7)
        plt.ylabel('Match Rate')
        plt.title(f'Predictive Power: φ-optimality = {phi_optimality:.4f} ({phi_interpretation})')
        
        # Add phi optimality as text annotation
        plt.text(0.5, 0.9, f'φ-optimality: {phi_optimality:.4f}', 
                 horizontalalignment='center',
                 transform=plt.gca().transAxes,
                 bbox=dict(facecolor='white', alpha=0.8))
        
        # Add p-value and prediction power
        plt.text(0.5, 0.82, f'p-value: {p_value:.6f}', 
                 horizontalalignment='center',
                 transform=plt.gca().transAxes,
                 bbox=dict(facecolor='white', alpha=0.8))
        
        plt.text(0.5, 0.74, f'Prediction power: {prediction_power:.2f}x', 
                 horizontalalignment='center',
                 transform=plt.gca().transAxes,
                 bbox=dict(facecolor='white', alpha=0.8))
        
        plt.grid(True, alpha=0.3)
        
        # Save the figure
        plt.tight_layout()
        
        if output_dir:
            output_path = os.path.join(output_dir, 'phi_optimality_test.png')
        else:
            output_path = 'phi_optimality_test.png'
            
        plt.savefig(output_path)
        
        if verbose:
            print(f"Visualization saved to '{output_path}'")
    
    # Save results to file
    if output_dir:
        results_path = os.path.join(output_dir, 'phi_optimality_results.txt')
    else:
        results_path = 'phi_optimality_results.txt'
        
    with open(results_path, 'w') as f:
        f.write("=== PHI OPTIMALITY TEST RESULTS ===\n\n")
        f.write(f"Match rate for GR predictions: {match_rate:.4f}\n")
        f.write(f"Mean match rate for random predictions: {mean_random_rate:.4f}\n")
        f.write(f"Prediction power (ratio): {prediction_power:.2f}x\n")
        f.write(f"Z-score: {z_score:.2f}\n")
        f.write(f"P-value: {p_value:.8f}\n")
        f.write(f"Phi optimality: {phi_optimality:.4f}\n\n")
        f.write(f"Phi optimality interpretation: {phi_interpretation}\n")
        f.write(f"The predictive power test shows a {phi_interpretation} alignment with golden ratio optimality.\n")
    
    if verbose:
        print(f"Results saved to '{results_path}'")
        print(f"Test completed in {time.time() - start_time:.2f} seconds.")
    
    # Prepare results dictionary
    results = {
        'test_name': 'Phi Optimality Test',
        'match_rate': match_rate,
        'mean_random_rate': mean_random_rate,
        'prediction_power': prediction_power,
        'z_score': z_score,
        'p_value': p_value,
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
    stats_analyzer = get_statistical_analyzer(monte_carlo_sims=100)
    
    # Run test
    results = run_test(data_loader, visualizer, stats_analyzer, verbose=True)
    
    # Print key results
    print("\n=== KEY RESULTS ===")
    print(f"Phi optimality: {results['phi_optimality']:.4f} ({results['phi_interpretation']})")
    print(f"P-value: {results['p_value']:.6f}")
    print(f"Prediction power: {results['prediction_power']:.2f}x")
