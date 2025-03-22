#!/usr/bin/env python3
"""
Run the recurrence quantification analysis test on CMB data.
This test detects deterministic structure in the CMB power spectrum.
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from cosmic_consciousness_analyzer import CosmicConsciousnessAnalyzer

def calculate_phi_optimality(value, random_value, p_value):
    """Calculate phi-optimality score between -1 and 1"""
    # Ratio-based component (how much better than random)
    if random_value > 0 and value > 0:
        ratio = value / random_value
        # Cap ratio component at 10x
        ratio_component = min(ratio, 10) / 10
    else:
        ratio_component = 0.5  # Neutral if we can't calculate ratio
    
    # Statistical significance component
    if p_value <= 0.5:
        # For significant results (p < 0.5), scale from 0 to 1
        sig_component = 1 - (p_value / 0.5)
    else:
        # For non-significant results (p > 0.5), scale from 0 to -1
        sig_component = -((p_value - 0.5) / 0.5)
    
    # Combine components with more weight to significance
    phi_optimality = 0.4 * ratio_component + 0.6 * sig_component
    
    return phi_optimality

def interpret_phi_optimality(value):
    """Provide text interpretation of phi-optimality value"""
    if value > 0.9:
        return "extremely high"
    elif value > 0.7:
        return "very high"
    elif value > 0.4:
        return "high"
    elif value > 0.2:
        return "moderate"
    elif value > 0.05:
        return "slight"
    elif value > -0.05:
        return "neutral"
    elif value > -0.2:
        return "slightly negative"
    elif value > -0.4:
        return "moderately negative"
    elif value > -0.7:
        return "strongly negative"
    else:
        return "extremely negative"

def main():
    """Run the recurrence quantification analysis test."""
    print("=== RECURRENCE QUANTIFICATION ANALYSIS ===")
    
    # Set data directory
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'planck_data')
    print(f"Using data directory: {data_dir}")
    
    # Create analyzer
    print("Creating analyzer with 100 Monte Carlo simulations...")
    start_time = time.time()
    analyzer = CosmicConsciousnessAnalyzer(data_dir=data_dir, monte_carlo_sims=100)
    print(f"Analyzer initialized in {time.time() - start_time:.2f} seconds.")
    print(f"Data loaded: {len(analyzer.data['ee_power'])} multipoles")
    print()
    
    # Run the test
    print("Running recurrence quantification analysis...")
    start_time = time.time()
    (RR, DET, LAM, mean_surr_DET, std_surr_DET, DET_z_score, DET_p_value, DET_ratio, 
     mean_surr_LAM, std_surr_LAM, LAM_z_score, LAM_p_value, LAM_ratio) = analyzer.test_recurrence_quantification()
    print(f"Test completed in {time.time() - start_time:.2f} seconds.")
    print()
    
    # Calculate phi-optimality scores
    det_phi_optimality = calculate_phi_optimality(DET, mean_surr_DET, DET_p_value)
    lam_phi_optimality = calculate_phi_optimality(LAM, mean_surr_LAM, LAM_p_value)
    
    # Print results
    print("=== RESULTS ===")
    print("1. Recurrence Rate:")
    print(f"   RR: {RR:.4f}")
    print()
    
    print("2. Determinism Analysis:")
    print(f"   Determinism (DET): {DET:.4f}")
    print(f"   Random DET: {mean_surr_DET:.4f}")
    print(f"   Z-score: {DET_z_score:.4f}")
    print(f"   P-value: {DET_p_value:.4f}")
    print(f"   DET ratio: {DET_ratio:.2f}x")
    print(f"   Phi optimality: {det_phi_optimality:.4f}")
    print(f"   Interpretation: {interpret_phi_optimality(det_phi_optimality)}")
    print()
    
    print("3. Laminarity Analysis:")
    print(f"   Laminarity (LAM): {LAM:.4f}")
    print(f"   Random LAM: {mean_surr_LAM:.4f}")
    print(f"   Z-score: {LAM_z_score:.4f}")
    print(f"   P-value: {LAM_p_value:.4f}")
    print(f"   LAM ratio: {LAM_ratio:.2f}x")
    print(f"   Phi optimality: {lam_phi_optimality:.4f}")
    print(f"   Interpretation: {interpret_phi_optimality(lam_phi_optimality)}")
    
    # Create visualization
    print("\nCreating visualization...")
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # 1. Recurrence Plot Visualization
    # Ensure we have valid data for embedding
    power_spectrum = np.copy(analyzer.data['ee_power'])
    min_positive = np.min(power_spectrum[power_spectrum > 0]) / 10.0 if np.any(power_spectrum > 0) else 1e-10
    power_spectrum[power_spectrum <= 0] = min_positive
    
    # Create time-delayed embedding
    embedding_dimension = 3
    delay = 1
    embedding = []
    for i in range(len(power_spectrum) - (embedding_dimension-1)*delay):
        point = [power_spectrum[i + j*delay] for j in range(embedding_dimension)]
        embedding.append(point)
    
    embedding = np.array(embedding)
    
    # Calculate distance matrix
    from scipy.spatial.distance import pdist, squareform
    distances = squareform(pdist(embedding, 'euclidean'))
    
    # Create recurrence matrix
    threshold = 0.1 * np.max(distances)
    recurrence_matrix = distances < threshold
    
    # Create a custom colormap (black and white)
    cmap = LinearSegmentedColormap.from_list('rp_cmap', ['white', 'black'])
    
    # Plot recurrence matrix
    im = ax1.imshow(recurrence_matrix, cmap=cmap, aspect='auto')
    ax1.set_title(f'Recurrence Plot (RR: {RR:.4f})')
    ax1.set_xlabel('Time Index')
    ax1.set_ylabel('Time Index')
    
    # 2. Diagonal and Vertical Line Distributions
    # Calculate diagonal line lengths
    diag_line_lengths = []
    for i in range(-(len(recurrence_matrix)-1), len(recurrence_matrix)):
        diagonal = np.diag(recurrence_matrix, k=i)
        current_length = 0
        for point in diagonal:
            if point:
                current_length += 1
            elif current_length >= 2:  # min_line_length
                diag_line_lengths.append(current_length)
                current_length = 0
            else:
                current_length = 0
        if current_length >= 2:
            diag_line_lengths.append(current_length)
    
    # Calculate vertical line lengths
    vert_line_lengths = []
    for i in range(len(recurrence_matrix)):
        vertical = recurrence_matrix[:, i]
        current_length = 0
        for point in vertical:
            if point:
                current_length += 1
            elif current_length >= 2:  # min_line_length
                vert_line_lengths.append(current_length)
                current_length = 0
            else:
                current_length = 0
        if current_length >= 2:
            vert_line_lengths.append(current_length)
    
    # Plot line length distributions
    if diag_line_lengths:
        ax2.hist(diag_line_lengths, bins=range(2, max(diag_line_lengths) + 2), 
                 alpha=0.7, label=f'Diagonal (DET: {DET:.4f})', color='blue')
    if vert_line_lengths:
        ax2.hist(vert_line_lengths, bins=range(2, max(vert_line_lengths) + 2), 
                 alpha=0.7, label=f'Vertical (LAM: {LAM:.4f})', color='red')
    
    ax2.set_title('Line Length Distributions')
    ax2.set_xlabel('Line Length')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    
    # Add overall title with phi-optimality
    avg_phi_optimality = (det_phi_optimality + lam_phi_optimality) / 2
    fig.suptitle(f'Recurrence Quantification Analysis (φ-optimality: {avg_phi_optimality:.4f})', 
                 fontsize=16)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Save visualization
    plt.savefig('recurrence_quantification_results.png', dpi=300, bbox_inches='tight')
    print("Visualization saved to 'recurrence_quantification_results.png'")
    
    # Save results to file
    with open('recurrence_quantification_results.txt', 'w') as f:
        f.write("=== RECURRENCE QUANTIFICATION ANALYSIS RESULTS ===\n\n")
        
        f.write("1. Recurrence Rate:\n")
        f.write(f"   RR: {RR:.4f}\n\n")
        
        f.write("2. Determinism Analysis:\n")
        f.write(f"   Determinism (DET): {DET:.4f}\n")
        f.write(f"   Random DET: {mean_surr_DET:.4f}\n")
        f.write(f"   Z-score: {DET_z_score:.4f}\n")
        f.write(f"   P-value: {DET_p_value:.4f}\n")
        f.write(f"   DET ratio: {DET_ratio:.2f}x\n")
        f.write(f"   Phi optimality: {det_phi_optimality:.4f}\n")
        f.write(f"   Interpretation: {interpret_phi_optimality(det_phi_optimality)}\n\n")
        
        f.write("3. Laminarity Analysis:\n")
        f.write(f"   Laminarity (LAM): {LAM:.4f}\n")
        f.write(f"   Random LAM: {mean_surr_LAM:.4f}\n")
        f.write(f"   Z-score: {LAM_z_score:.4f}\n")
        f.write(f"   P-value: {LAM_p_value:.4f}\n")
        f.write(f"   LAM ratio: {LAM_ratio:.2f}x\n")
        f.write(f"   Phi optimality: {lam_phi_optimality:.4f}\n")
        f.write(f"   Interpretation: {interpret_phi_optimality(lam_phi_optimality)}\n\n")
        
        f.write("Summary:\n")
        f.write(f"The recurrence quantification analysis shows a {interpret_phi_optimality(avg_phi_optimality)} ")
        f.write("alignment with deterministic structure in the CMB power spectrum.\n")
        f.write(f"Average phi-optimality: {avg_phi_optimality:.4f}\n")
    
    print("Results saved to 'recurrence_quantification_results.txt'")

if __name__ == "__main__":
    main()
