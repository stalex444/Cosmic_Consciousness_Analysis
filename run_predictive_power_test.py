#!/usr/bin/env python3
"""
Run the predictive power test from the Cosmic Consciousness Analyzer.
This script focuses on testing if golden ratio relationships can predict peak locations in the CMB spectrum.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from cosmic_consciousness_analyzer import CosmicConsciousnessAnalyzer

def main():
    """Run the predictive power test and display results."""
    print("Initializing Cosmic Consciousness Analyzer...")
    
    # Use data directory
    data_dir = 'data'
    if not os.path.exists(data_dir):
        print(f"Error: Data directory '{data_dir}' not found.")
        if os.path.exists('planck_data'):
            print("Using 'planck_data' directory instead.")
            data_dir = 'planck_data'
        else:
            print("No valid data directory found.")
            sys.exit(1)
    
    print(f"Using data directory: {os.path.abspath(data_dir)}")
    
    # Initialize analyzer with fewer Monte Carlo simulations for faster testing
    try:
        analyzer = CosmicConsciousnessAnalyzer(data_dir=data_dir, monte_carlo_sims=1000)
        print("Analyzer initialized successfully.")
        print(f"Data loaded: {len(analyzer.data['ell'])} multipoles")
    except Exception as e:
        print(f"Error initializing analyzer: {e}")
        print("\nTroubleshooting suggestions:")
        print("1. Check that the data directory contains the required files")
        print("2. Verify that the EE spectrum file is correctly formatted")
        print("3. Ensure the covariance matrix file exists and is valid")
        print("4. Make sure all dependencies are installed (numpy, scipy, matplotlib, astropy)")
        sys.exit(1)
    
    # Run the predictive power test
    print("\nRunning predictive power test...")
    try:
        match_rate, mean_random_rate, z_score, p_value, prediction_power = analyzer.test_predictive_power()
        
        # Print results
        print("\n=== Predictive Power Test Results ===")
        print(f"Match rate for GR predictions: {match_rate:.4f}")
        print(f"Mean match rate for random predictions: {mean_random_rate:.4f}")
        print(f"Prediction power (ratio): {prediction_power:.2f}x")
        print(f"Z-score: {z_score:.2f}")
        print(f"P-value: {p_value:.8f}")
        
        # Debug information
        print("\n=== Debug Information ===")
        # Find peaks using the same method as in the test
        peaks = []
        window_size = 15
        smoothed_power = np.convolve(analyzer.data['ee_power'], np.ones(window_size)/window_size, mode='same')
        
        for i in range(window_size, len(analyzer.data['ell'])-window_size):
            if (smoothed_power[i] > smoothed_power[i-1] and 
                smoothed_power[i] > smoothed_power[i+1]):
                
                left_min = np.min(smoothed_power[max(0, i-50):i])
                right_min = np.min(smoothed_power[i+1:min(len(smoothed_power), i+51)])
                higher_min = max(left_min, right_min)
                prominence = smoothed_power[i] - higher_min
                
                if prominence > 0.05 * np.max(smoothed_power):
                    peaks.append(i)
        
        # If we have too many peaks, keep only the most prominent ones
        if len(peaks) > 20:
            peak_prominences = []
            for i in peaks:
                left_min = np.min(smoothed_power[max(0, i-50):i])
                right_min = np.min(smoothed_power[i+1:min(len(smoothed_power), i+51)])
                higher_min = max(left_min, right_min)
                prominence = smoothed_power[i] - higher_min
                peak_prominences.append(prominence)
                
            prominence_threshold = sorted(peak_prominences, reverse=True)[19]
            filtered_peaks = [peaks[i] for i, p in enumerate(peak_prominences) if p >= prominence_threshold]
            peaks = filtered_peaks[:20]
        
        # Fallback if no peaks found
        if not peaks:
            print("No peaks found with prominence method, falling back to simple method")
            for i in range(1, len(analyzer.data['ell'])-1):
                if (analyzer.data['ee_power'][i] > analyzer.data['ee_power'][i-1] and 
                    analyzer.data['ee_power'][i] > analyzer.data['ee_power'][i+1]):
                    peaks.append(i)
            
            if len(peaks) > 20:
                peak_powers = [analyzer.data['ee_power'][i] for i in peaks]
                power_threshold = sorted(peak_powers, reverse=True)[19]
                filtered_peaks = [peaks[i] for i, p in enumerate(peak_powers) if p >= power_threshold]
                peaks = filtered_peaks[:20]
        
        peak_ells = [analyzer.data['ell'][i] for i in peaks]
        print(f"Number of peaks found: {len(peak_ells)}")
        print(f"Peak multipoles: {peak_ells[:10]}..." if len(peak_ells) > 10 else f"Peak multipoles: {peak_ells}")
        
        # Generate predictions based on GR relationships
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
        print(f"Number of GR predictions: {len(predictions)}")
        print(f"GR predictions: {predictions[:10]}..." if len(predictions) > 10 else f"GR predictions: {predictions}")
        
        # Count matches with increased tolerance
        tolerance = 20  # Increased for debugging - match the value in the analyzer
        matches = 0
        match_details = []
        for pred in predictions:
            for peak in peak_ells:
                if abs(pred - peak) <= tolerance:
                    matches += 1
                    match_details.append((pred, peak, abs(pred - peak)))
                    break
        
        print(f"Number of matches (tolerance={tolerance}): {matches}")
        print(f"Match details: {match_details[:5]}..." if len(match_details) > 5 else f"Match details: {match_details}")
        print(f"Match rate: {matches / len(predictions) if len(predictions) > 0 else 0:.4f}")
        
        # Interpret results
        if p_value < 0.01:
            significance = "highly significant"
        elif p_value < 0.05:
            significance = "significant"
        elif p_value < 0.1:
            significance = "marginally significant"
        else:
            significance = "not significant"
        
        if prediction_power > 2:
            effect = "strong"
        elif prediction_power > 1.5:
            effect = "moderate"
        elif prediction_power > 1.1:
            effect = "weak"
        else:
            effect = "negligible"
        
        print(f"\nInterpretation: The test shows a {effect} effect that is {significance}.")
        print(f"Golden ratio relationships predict peak locations {prediction_power:.2f}x better")
        print(f"than random predictions.")
        
        # Create visualization
        plt.figure(figsize=(10, 6))
        
        # Create bar chart comparing GR and random match rates
        plt.bar(['GR Predictions', 'Random Predictions'], [match_rate, mean_random_rate], 
                color=['gold', 'gray'], alpha=0.7)
        plt.ylabel('Match Rate')
        plt.title(f'Predictive Power: {prediction_power:.2f}x better for GR predictions (p={p_value:.4f})')
        plt.grid(True, alpha=0.3)
        
        # Save and show the figure
        plt.tight_layout()
        plt.savefig('predictive_power.png')
        print("\nVisualization saved to 'predictive_power.png'")
        
        # Save results to file
        with open('predictive_power_results.txt', 'w') as f:
            f.write("=== PREDICTIVE POWER TEST RESULTS ===\n\n")
            f.write(f"Match rate for GR predictions: {match_rate:.4f}\n")
            f.write(f"Mean match rate for random predictions: {mean_random_rate:.4f}\n")
            f.write(f"Prediction power (ratio): {prediction_power:.2f}x\n")
            f.write(f"Z-score: {z_score:.2f}\n")
            f.write(f"P-value: {p_value:.8f}\n\n")
            f.write(f"Interpretation: The test shows a {effect} effect that is {significance}.\n")
            f.write(f"Golden ratio relationships predict peak locations {prediction_power:.2f}x better ")
            f.write(f"than random predictions.\n")
        
        print("Results saved to 'predictive_power_results.txt'")
        
        # Additional analysis: Plot the spectrum with peaks and GR predictions
        plt.figure(figsize=(12, 6))
        
        # Plot the full spectrum
        plt.plot(analyzer.data['ell'], analyzer.data['ee_power'], 'b-', alpha=0.5, label='EE Power Spectrum')
        
        # Get the peaks directly from the analyzer
        peaks = []
        window_size = 15
        smoothed_power = np.convolve(analyzer.data['ee_power'], np.ones(window_size)/window_size, mode='same')
        
        for i in range(window_size, len(analyzer.data['ell'])-window_size):
            if (smoothed_power[i] > smoothed_power[i-1] and 
                smoothed_power[i] > smoothed_power[i+1]):
                
                left_min = np.min(smoothed_power[max(0, i-50):i])
                right_min = np.min(smoothed_power[i+1:min(len(smoothed_power), i+51)])
                higher_min = max(left_min, right_min)
                prominence = smoothed_power[i] - higher_min
                
                if prominence > 0.05 * np.max(smoothed_power):
                    peaks.append(i)
        
        # If we have too many peaks, keep only the most prominent ones
        if len(peaks) > 20:
            peak_prominences = []
            for i in peaks:
                left_min = np.min(smoothed_power[max(0, i-50):i])
                right_min = np.min(smoothed_power[i+1:min(len(smoothed_power), i+51)])
                higher_min = max(left_min, right_min)
                prominence = smoothed_power[i] - higher_min
                peak_prominences.append(prominence)
                
            prominence_threshold = sorted(peak_prominences, reverse=True)[19]
            filtered_peaks = [peaks[i] for i, p in enumerate(peak_prominences) if p >= prominence_threshold]
            peaks = filtered_peaks[:20]
        
        # Fallback if no peaks found
        if not peaks:
            for i in range(1, len(analyzer.data['ell'])-1):
                if (analyzer.data['ee_power'][i] > analyzer.data['ee_power'][i-1] and 
                    analyzer.data['ee_power'][i] > analyzer.data['ee_power'][i+1]):
                    peaks.append(i)
            
            if len(peaks) > 20:
                peak_powers = [analyzer.data['ee_power'][i] for i in peaks]
                power_threshold = sorted(peak_powers, reverse=True)[19]
                filtered_peaks = [peaks[i] for i, p in enumerate(peak_powers) if p >= power_threshold]
                peaks = filtered_peaks[:20]
        
        peak_ells = [analyzer.data['ell'][i] for i in peaks]
        peak_powers = [analyzer.data['ee_power'][i] for i in peaks]
        
        # Plot the peaks
        plt.scatter(peak_ells, peak_powers, color='red', s=80, label='Actual Peaks', zorder=3)
        
        # Generate and plot GR predictions
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
        
        # Get power values for predictions
        pred_powers = []
        for pred in predictions:
            idx = np.argmin(np.abs(analyzer.data['ell'] - pred))
            pred_powers.append(analyzer.data['ee_power'][idx])
        
        # Plot the predictions
        plt.scatter(predictions, pred_powers, color='gold', s=60, alpha=0.7, label='GR Predictions', zorder=2)
        
        # Highlight matches
        tolerance = 20
        for pred in predictions:
            for peak in peak_ells:
                if abs(pred - peak) <= tolerance:
                    idx = np.argmin(np.abs(analyzer.data['ell'] - pred))
                    plt.scatter([pred], [analyzer.data['ee_power'][idx]], color='green', s=100, 
                               marker='*', label='Match', zorder=4)
                    # Only add the label once
                    if 'Match' in plt.gca().get_legend_handles_labels()[1]:
                        plt.scatter([pred], [analyzer.data['ee_power'][idx]], color='green', s=100, 
                                  marker='*', zorder=4)
                    break
        
        plt.xlabel('Multipole ℓ')
        plt.ylabel('EE Power')
        plt.title('CMB Spectrum with Peaks and GR Predictions')
        plt.legend()
        plt.xscale('log')
        plt.grid(True, alpha=0.3)
        
        # Save the spectrum visualization
        plt.tight_layout()
        plt.savefig('spectrum_with_predictions.png')
        print("Spectrum visualization saved to 'spectrum_with_predictions.png'")
        
    except Exception as e:
        print(f"Error running predictive power test: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
