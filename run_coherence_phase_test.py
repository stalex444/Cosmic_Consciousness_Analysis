#!/usr/bin/env python3
"""
Run the coherence phase test from the CosmicConsciousnessAnalyzer class.
This script analyzes phase relationships in the CMB data for coherence patterns
and golden ratio relationships.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from cosmic_consciousness_analyzer import CosmicConsciousnessAnalyzer

def main():
    """Run the coherence phase test and display results"""
    print("=== COHERENCE PHASE ANALYSIS ===")
    
    # Set up data directory
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "planck_data")
    print(f"Using data directory: {data_dir}")
    
    # Create analyzer with Monte Carlo simulations
    print("Creating analyzer with 100 Monte Carlo simulations...")
    start_time = time.time()
    analyzer = CosmicConsciousnessAnalyzer(data_dir=data_dir, monte_carlo_sims=100)
    init_time = time.time() - start_time
    print(f"Analyzer initialized in {init_time:.2f} seconds.")
    print(f"Data loaded: {len(analyzer.data['ell'])} multipoles")
    
    # Run the test
    print("\nRunning coherence phase analysis...")
    start_time = time.time()
    result = analyzer.test_coherence_phase()
    print(f"Test completed in {time.time() - start_time:.2f} seconds.")
    
    if result is None:
        print("Not enough data for coherence phase analysis.")
        return
    
    # Unpack results
    kuramoto, mean_surrogate, z_score, p_value, coherence_ratio, phi_phase_ratio, mean_random_ratio, phi_z_score, phi_p_value, phase_optimization = result
    
    # Calculate phi optimality
    phi_optimality = calculate_phi_optimality(phase_optimization, 1.0)
    phi_interp = interpret_phi_optimality(phi_optimality)
    
    # Display results
    print("\n=== RESULTS ===")
    print(f"1. Kuramoto order parameter: {kuramoto:.6f}")
    print(f"2. Random surrogate Kuramoto: {mean_surrogate:.6f}")
    print(f"3. Phase coherence ratio: {coherence_ratio:.2f}x")
    print(f"4. Statistical significance:")
    print(f"   Z-score: {z_score:.4f}")
    print(f"   P-value: {p_value:.4f}")
    print(f"5. Golden angle phase relationships:")
    print(f"   Phi phase ratio: {phi_phase_ratio:.6f}")
    print(f"   Random phase ratio: {mean_random_ratio:.6f}")
    print(f"   Optimization ratio: {phase_optimization:.2f}x")
    print(f"   Z-score: {phi_z_score:.4f}")
    print(f"   P-value: {phi_p_value:.4f}")
    print(f"6. Phi optimality: {phi_optimality:.4f} ({phi_interp})")
    
    # Create visualization
    print("\nCreating visualization...")
    create_visualization(kuramoto, mean_surrogate, coherence_ratio, z_score, p_value,
                        phi_phase_ratio, mean_random_ratio, phi_z_score, phi_p_value, 
                        phase_optimization, phi_optimality)
    
    # Save results to file
    save_results(kuramoto, mean_surrogate, coherence_ratio, z_score, p_value,
                phi_phase_ratio, mean_random_ratio, phi_z_score, phi_p_value, 
                phase_optimization, phi_optimality, phi_interp)
    
    print("Visualization saved to 'coherence_phase_results.png'")
    print("Results saved to 'coherence_phase_results.txt'")

def calculate_phi_optimality(ratio, baseline=1.0):
    """Calculate phi optimality score bounded between -1 and 1"""
    if ratio > baseline:
        # Positive optimality (ratio > baseline)
        return 2 / (1 + np.exp(-0.5 * (ratio - baseline))) - 1
    else:
        # Negative optimality (ratio < baseline)
        return 1 - 2 / (1 + np.exp(0.5 * (baseline - ratio)))

def interpret_phi_optimality(score):
    """Interpret phi optimality score"""
    if score > 0.9:
        return "extremely high"
    elif score > 0.7:
        return "very high"
    elif score > 0.3:
        return "high"
    elif score > 0.1:
        return "moderate"
    elif score > -0.1:
        return "slight"
    elif score > -0.3:
        return "slightly negative"
    elif score > -0.7:
        return "moderately negative"
    elif score > -0.9:
        return "strongly negative"
    else:
        return "extremely negative"

def create_visualization(kuramoto, mean_surrogate, coherence_ratio, z_score, p_value,
                        phi_phase_ratio, mean_random_ratio, phi_z_score, phi_p_value, 
                        phase_optimization, phi_optimality):
    """Create visualization of coherence phase results"""
    plt.figure(figsize=(12, 10))
    
    # Create a 2x2 grid of subplots
    plt.subplot(2, 2, 1)
    plt.bar(['CMB Data', 'Random Surrogate'], [kuramoto, mean_surrogate], color=['blue', 'gray'])
    plt.title('Kuramoto Order Parameter')
    plt.ylabel('Coherence (higher = more coherent)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add ratio and significance
    plt.text(0.5, 0.9, f"Ratio: {coherence_ratio:.2f}x", transform=plt.gca().transAxes, ha='center')
    plt.text(0.5, 0.8, f"Z-score: {z_score:.2f}", transform=plt.gca().transAxes, ha='center')
    plt.text(0.5, 0.7, f"P-value: {p_value:.4f}", transform=plt.gca().transAxes, ha='center')
    
    # Golden angle phase relationships
    plt.subplot(2, 2, 2)
    plt.bar(['CMB Data', 'Random'], [phi_phase_ratio, mean_random_ratio], color=['gold', 'gray'])
    plt.title('Golden Angle Phase Relationships')
    plt.ylabel('Proportion of phase differences')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add ratio and significance
    plt.text(0.5, 0.9, f"Ratio: {phase_optimization:.2f}x", transform=plt.gca().transAxes, ha='center')
    plt.text(0.5, 0.8, f"Z-score: {phi_z_score:.2f}", transform=plt.gca().transAxes, ha='center')
    plt.text(0.5, 0.7, f"P-value: {phi_p_value:.4f}", transform=plt.gca().transAxes, ha='center')
    
    # Phi optimality gauge
    plt.subplot(2, 1, 2)
    create_phi_optimality_gauge(phi_optimality)
    
    plt.tight_layout()
    plt.savefig('coherence_phase_results.png', dpi=300, bbox_inches='tight')

def create_phi_optimality_gauge(phi_optimality):
    """Create a gauge visualization for phi optimality"""
    # Create a semicircle for the gauge
    theta = np.linspace(-np.pi, 0, 100)
    r = 1.0
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    # Plot the gauge background
    plt.plot(x, y, 'k-', linewidth=2)
    
    # Add colored regions
    colors = ['red', 'orange', 'yellow', 'yellowgreen', 'green']
    regions = np.linspace(-np.pi, 0, len(colors) + 1)
    
    for i in range(len(colors)):
        theta_region = np.linspace(regions[i], regions[i+1], 100)
        x_region = r * np.cos(theta_region)
        y_region = r * np.sin(theta_region)
        plt.fill_between(x_region, 0, y_region, color=colors[i], alpha=0.3)
    
    # Convert phi_optimality to angle
    angle = -np.pi * (1 - (phi_optimality + 1) / 2)
    
    # Plot the needle
    plt.plot([0, r * np.cos(angle)], [0, r * np.sin(angle)], 'k-', linewidth=3)
    plt.plot([0], [0], 'ko', markersize=10)
    
    # Add labels
    plt.text(-1.1, -0.1, 'Negative', fontsize=12)
    plt.text(-0.5, -0.1, 'Neutral', fontsize=12)
    plt.text(0.5, -0.1, 'Positive', fontsize=12)
    
    # Add the value
    plt.text(0, -0.5, f'φ-optimality: {phi_optimality:.4f}', fontsize=14, ha='center')
    
    # Remove axes
    plt.axis('equal')
    plt.axis('off')
    plt.title('Phi Optimality Gauge')

def save_results(kuramoto, mean_surrogate, coherence_ratio, z_score, p_value,
                phi_phase_ratio, mean_random_ratio, phi_z_score, phi_p_value, 
                phase_optimization, phi_optimality, phi_interp):
    """Save results to a text file"""
    with open('coherence_phase_results.txt', 'w') as f:
        f.write("=== COHERENCE PHASE ANALYSIS RESULTS ===\n\n")
        
        f.write(f"1. Kuramoto order parameter: {kuramoto:.6f}\n")
        f.write(f"2. Random surrogate Kuramoto: {mean_surrogate:.6f}\n")
        f.write(f"3. Phase coherence ratio: {coherence_ratio:.2f}x\n")
        f.write(f"4. Statistical significance:\n")
        f.write(f"   Z-score: {z_score:.4f}\n")
        f.write(f"   P-value: {p_value:.4f}\n\n")
        
        f.write(f"5. Golden angle phase relationships:\n")
        f.write(f"   Phi phase ratio: {phi_phase_ratio:.6f}\n")
        f.write(f"   Random phase ratio: {mean_random_ratio:.6f}\n")
        f.write(f"   Optimization ratio: {phase_optimization:.2f}x\n")
        f.write(f"   Z-score: {phi_z_score:.4f}\n")
        f.write(f"   P-value: {phi_p_value:.4f}\n\n")
        
        f.write(f"6. Phi optimality: {phi_optimality:.4f} ({phi_interp})\n\n")
        
        # Add summary interpretation
        f.write("Summary:\n")
        if phi_optimality > 0.3:
            f.write("The analysis shows strong evidence for coherent phase relationships in the CMB data, ")
            f.write("with significant golden ratio patterns in phase differences.\n")
        elif phi_optimality > 0:
            f.write("The analysis shows moderate evidence for coherent phase relationships in the CMB data, ")
            f.write("with some golden ratio patterns in phase differences.\n")
        else:
            f.write("The analysis does not show evidence for coherent phase relationships in the CMB data ")
            f.write("or golden ratio patterns in phase differences.\n")

if __name__ == "__main__":
    main()
