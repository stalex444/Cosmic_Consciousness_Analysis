#!/usr/bin/env python3
"""
Run the predictive power test with progress reporting.
This script runs the test_predictive_power method from the CosmicConsciousnessAnalyzer
and reports progress during the execution.
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from cosmic_consciousness_analyzer import CosmicConsciousnessAnalyzer
from calculate_phi_optimality import calculate_phi_optimality, interpret_phi_optimality

def main():
    """Run the predictive power test with progress reporting."""
    # Set data directory
    data_dir = os.path.join(os.getcwd(), 'planck_data')
    if not os.path.exists(data_dir):
        print(f"Error: Data directory '{data_dir}' not found.")
        sys.exit(1)
    
    print("=== PREDICTIVE POWER TEST WITH PROGRESS REPORTING ===")
    print(f"Using data directory: {data_dir}")
    
    # Initialize analyzer with 1000 Monte Carlo simulations for faster testing
    # but still scientifically valid
    print("Creating analyzer with 1000 Monte Carlo simulations...")
    start_time = time.time()
    analyzer = CosmicConsciousnessAnalyzer(data_dir=data_dir, monte_carlo_sims=1000)
    print(f"Analyzer initialized in {time.time() - start_time:.2f} seconds.")
    print(f"Data loaded: {len(analyzer.data['ell'])} multipoles")
    
    # Run the predictive power test with progress reporting
    print("\nRunning predictive power test...")
    print("Step 1: Finding peaks in CMB spectrum...")
    start_time = time.time()
    
    # We'll run the test directly but with our own progress reporting
    # First, extract the peak finding logic
    peaks = []
    peak_prominences = []
    
    # Calculate rolling average to smooth the data
    window_size = 15
    smoothed_power = np.convolve(analyzer.data['ee_power'], np.ones(window_size)/window_size, mode='same')
    
    # Find peaks with prominence
    for i in range(window_size, len(analyzer.data['ell'])-window_size):
        # Check if this is a local maximum in the smoothed data
        if (smoothed_power[i] > smoothed_power[i-1] and 
            smoothed_power[i] > smoothed_power[i+1]):
            
            # Calculate prominence
            left_min = np.min(smoothed_power[max(0, i-50):i])
            right_min = np.min(smoothed_power[i+1:min(len(smoothed_power), i+51)])
            higher_min = max(left_min, right_min)
            prominence = smoothed_power[i] - higher_min
            
            # Only consider peaks with sufficient prominence
            if prominence > 0.05 * np.max(smoothed_power):
                peaks.append(i)
                peak_prominences.append(prominence)
    
    print(f"Found {len(peaks)} initial peaks.")
    
    # Filter peaks if needed
    if len(peaks) > 20:
        prominence_threshold = sorted(peak_prominences, reverse=True)[19]
        filtered_peaks = [peaks[i] for i, p in enumerate(peak_prominences) if p >= prominence_threshold]
        peaks = filtered_peaks[:20]
        print(f"Filtered to top {len(peaks)} peaks by prominence.")
    
    # Fallback if no peaks found
    if not peaks:
        print("No peaks found with prominence method, falling back to simple method...")
        for i in range(1, len(analyzer.data['ell'])-1):
            if (analyzer.data['ee_power'][i] > analyzer.data['ee_power'][i-1] and 
                analyzer.data['ee_power'][i] > analyzer.data['ee_power'][i+1]):
                peaks.append(i)
        
        if len(peaks) > 20:
            peak_powers = [analyzer.data['ee_power'][i] for i in peaks]
            power_threshold = sorted(peak_powers, reverse=True)[19]
            filtered_peaks = [peaks[i] for i, p in enumerate(peak_powers) if p >= power_threshold]
            peaks = filtered_peaks[:20]
            print(f"Filtered to top {len(peaks)} peaks by power.")
    
    peak_ells = [analyzer.data['ell'][i] for i in peaks]
    print(f"Final peaks (first 5): {peak_ells[:5]}...")
    print(f"Peak finding completed in {time.time() - start_time:.2f} seconds.")
    
    # Generate predictions
    print("\nStep 2: Generating predictions based on GR relationships...")
    start_time = time.time()
    predictions = []
    
    # Add direct Fibonacci sequence multipoles
    fibonacci_multipoles = [2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597]
    for fib in fibonacci_multipoles:
        if fib > min(analyzer.data['ell']) and fib < max(analyzer.data['ell']):
            predictions.append(fib)
    
    # Add GR relationships from peaks
    for peak in peak_ells:
        # Add multiples of phi
        predictions.append(int(round(peak * analyzer.phi)))
        predictions.append(int(round(peak / analyzer.phi)))
        
        # Add phi^2 relationships
        predictions.append(int(round(peak * analyzer.phi * analyzer.phi)))
        predictions.append(int(round(peak / (analyzer.phi * analyzer.phi))))
    
    # Remove duplicates and ensure all predictions are within range
    predictions = list(set([p for p in predictions if p > min(analyzer.data['ell']) and p < max(analyzer.data['ell'])]))
    print(f"Generated {len(predictions)} unique predictions.")
    print(f"Prediction generation completed in {time.time() - start_time:.2f} seconds.")
    
    # Count matches
    print("\nStep 3: Counting matches between predictions and actual peaks...")
    start_time = time.time()
    tolerance = 20
    matches = 0
    for pred in predictions:
        for peak in peak_ells:
            if abs(pred - peak) <= tolerance:
                matches += 1
                break
    
    match_rate = matches / len(predictions) if len(predictions) > 0 else 0
    print(f"Found {matches} matches out of {len(predictions)} predictions.")
    print(f"Match rate: {match_rate:.4f}")
    print(f"Match counting completed in {time.time() - start_time:.2f} seconds.")
    
    # Run Monte Carlo simulations
    print("\nStep 4: Running Monte Carlo simulations for random predictions...")
    start_time = time.time()
    random_match_rates = []
    num_simulations = 1000  # Use 1000 for faster testing but still scientifically valid
    
    for i in range(num_simulations):
        if i % 100 == 0:
            print(f"Simulation {i}/{num_simulations}... ({i/num_simulations*100:.1f}%)")
            
        # Ensure we have the same number of predictions
        if len(predictions) > 0:
            random_predictions = np.random.choice(analyzer.data['ell'], size=len(predictions), replace=False)
            random_matches = 0
            for pred in random_predictions:
                for peak in peak_ells:
                    if abs(pred - peak) <= tolerance:
                        random_matches += 1
                        break
            random_match_rates.append(random_matches / len(random_predictions))
    
    # Calculate significance
    print("\nStep 5: Calculating significance...")
    if random_match_rates:
        mean_random_rate = np.mean(random_match_rates)
        std_random_rate = np.std(random_match_rates)
        
        z_score = (match_rate - mean_random_rate) / std_random_rate if std_random_rate > 0 else 0
        p_value = 1 - stats.norm.cdf(z_score)
        
        prediction_power = match_rate / mean_random_rate if mean_random_rate > 0 else 1.0
    else:
        mean_random_rate = 0
        z_score = 0
        p_value = 1.0
        prediction_power = 1.0
    
    print(f"Monte Carlo simulations completed in {time.time() - start_time:.2f} seconds.")
    
    # Calculate phi optimality
    print("\nStep 6: Calculating phi optimality...")
    phi_optimality = calculate_phi_optimality(match_rate, mean_random_rate)
    phi_interpretation = interpret_phi_optimality(phi_optimality)
    
    # Print results
    print("\n=== RESULTS ===")
    print(f"Match rate for GR predictions: {match_rate:.4f}")
    print(f"Mean match rate for random predictions: {mean_random_rate:.4f}")
    print(f"Standard deviation of random match rates: {std_random_rate:.4f}")
    print(f"Z-score: {z_score:.4f}")
    print(f"P-value: {p_value:.4f}")
    print(f"Prediction power (ratio): {prediction_power:.2f}x")
    print(f"Phi optimality: {phi_optimality:.4f}")
    print(f"Interpretation: {phi_interpretation}")

    # Save results to file
    with open('predictive_power_results.txt', 'w') as f:
        f.write("=== PREDICTIVE POWER TEST RESULTS ===\n\n")
        f.write(f"Number of peaks identified: {len(peak_ells)}\n")
        f.write(f"Number of predictions generated: {len(predictions)}\n")
        f.write(f"Match rate for GR predictions: {match_rate:.4f}\n")
        f.write(f"Mean match rate for random predictions: {mean_random_rate:.4f}\n")
        f.write(f"Standard deviation of random match rates: {std_random_rate:.4f}\n")
        f.write(f"Z-score: {z_score:.4f}\n")
        f.write(f"P-value: {p_value:.4f}\n")
        f.write(f"Prediction power (ratio): {prediction_power:.2f}x\n")
        f.write(f"Phi optimality: {phi_optimality:.4f}\n")
        f.write(f"Interpretation: {phi_interpretation}\n\n")
        f.write("=== PEAK MULTIPOLES ===\n")
        f.write(", ".join([str(ell) for ell in peak_ells]))
        f.write("\n\n=== PREDICTIONS ===\n")
        f.write(", ".join([str(pred) for pred in predictions]))
    
    print("\nResults saved to 'predictive_power_results.txt'")
    
    # Create visualization
    print("\nCreating visualization...")
    plt.figure(figsize=(12, 8))
    
    # Plot the power spectrum
    plt.subplot(2, 1, 1)
    plt.plot(analyzer.data['ell'], analyzer.data['ee_power'], 'b-', alpha=0.5, label='EE Power')
    plt.plot(analyzer.data['ell'], smoothed_power, 'r-', alpha=0.7, label='Smoothed Power')
    
    # Mark peaks
    peak_powers = [analyzer.data['ee_power'][i] for i in peaks]
    plt.scatter([analyzer.data['ell'][i] for i in peaks], peak_powers, color='green', s=100, marker='^', label='Detected Peaks')
    
    # Mark predictions
    for pred in predictions:
        plt.axvline(x=pred, color='gold', alpha=0.3, linestyle='--')
    
    # Mark matches
    for pred in predictions:
        for peak in peak_ells:
            if abs(pred - peak) <= tolerance:
                plt.axvline(x=pred, color='red', alpha=0.5, linestyle='-')
                break
    
    plt.xlabel('Multipole (ℓ)')
    plt.ylabel('Power')
    plt.title(f'CMB Power Spectrum with Peaks and Predictions\nMatch Rate: {match_rate:.4f}, Random: {mean_random_rate:.4f}, Power: {prediction_power:.2f}x')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot the phi optimality
    plt.subplot(2, 1, 2)
    plt.bar(['GR Predictions', 'Random Predictions'], [match_rate, mean_random_rate], 
            color=['gold', 'gray'], alpha=0.7)
    plt.ylabel('Match Rate')
    plt.title(f'Predictive Power: φ-optimality = {phi_optimality:.4f}')
    
    # Add phi optimality as text annotation
    plt.text(0.5, 0.9, f'φ-optimality: {phi_optimality:.4f}', 
             horizontalalignment='center',
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.grid(True, alpha=0.3)
    
    # Save and show the figure
    plt.tight_layout()
    plt.savefig('predictive_power_results.png')
    print("Visualization saved to 'predictive_power_results.png'")

if __name__ == "__main__":
    main()
