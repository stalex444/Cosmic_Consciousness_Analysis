#!/usr/bin/env python3
"""
Visualize the golden symmetries test results from the CosmicConsciousnessAnalyzer.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from cosmic_consciousness_analyzer import CosmicConsciousnessAnalyzer

def main():
    """Run the golden symmetries test and visualize the results."""
    # Set data directory
    data_dir = os.path.join(os.getcwd(), 'planck_data')
    
    print("=== GOLDEN SYMMETRIES VISUALIZATION ===")
    print(f"Using data directory: {data_dir}")
    
    # Initialize analyzer
    print("Creating analyzer...")
    analyzer = CosmicConsciousnessAnalyzer(data_dir=data_dir, monte_carlo_sims=1000)
    print(f"Data loaded: {len(analyzer.data['ell'])} multipoles")
    
    # Run the golden symmetries test
    print("\nRunning Golden Symmetries Test...")
    asymmetry, mean_alternative, z_score, p_value, symmetry_ratio = analyzer.test_golden_symmetries()
    
    # Create visualization
    print("\nCreating visualization...")
    
    # Calculate symmetry measures for each multipole
    phi_fold = []
    e_fold = []
    pi_fold = []
    two_fold = []
    
    multipoles = []
    
    for i in range(len(analyzer.data['ell'])):
        l_value = analyzer.data['ell'][i]
        power = analyzer.data['ee_power'][i]
        
        # Calculate phi-related multipoles
        l_phi = l_value * analyzer.phi
        idx_phi = np.abs(analyzer.data['ell'] - l_phi).argmin()
        
        l_inv_phi = l_value / analyzer.phi
        idx_inv_phi = np.abs(analyzer.data['ell'] - l_inv_phi).argmin()
        
        # Calculate e-related multipoles
        l_e = l_value * np.e
        idx_e = np.abs(analyzer.data['ell'] - l_e).argmin()
        
        l_inv_e = l_value / np.e
        idx_inv_e = np.abs(analyzer.data['ell'] - l_inv_e).argmin()
        
        # Calculate pi-related multipoles
        l_pi = l_value * np.pi
        idx_pi = np.abs(analyzer.data['ell'] - l_pi).argmin()
        
        l_inv_pi = l_value / np.pi
        idx_inv_pi = np.abs(analyzer.data['ell'] - l_inv_pi).argmin()
        
        # Calculate 2-related multipoles
        l_two = l_value * 2
        idx_two = np.abs(analyzer.data['ell'] - l_two).argmin()
        
        l_inv_two = l_value / 2
        idx_inv_two = np.abs(analyzer.data['ell'] - l_inv_two).argmin()
        
        # Calculate symmetry measures
        if idx_phi < len(analyzer.data['ell']) and idx_inv_phi < len(analyzer.data['ell']):
            power_phi = analyzer.data['ee_power'][idx_phi]
            power_inv_phi = analyzer.data['ee_power'][idx_inv_phi]
            
            # Use absolute values to handle negative powers
            abs_power = abs(power)
            abs_power_phi = abs(power_phi)
            abs_power_inv_phi = abs(power_inv_phi)
            
            # Perfect symmetry would give power = sqrt(power_phi * power_inv_phi)
            expected_power = np.sqrt(abs_power_phi * abs_power_inv_phi)
            symmetry_ratio_phi = abs_power / expected_power if expected_power != 0 else 1
            
            phi_fold.append(abs(1 - symmetry_ratio_phi))
            multipoles.append(l_value)
            
            # Calculate for e
            if idx_e < len(analyzer.data['ell']) and idx_inv_e < len(analyzer.data['ell']):
                power_e = analyzer.data['ee_power'][idx_e]
                power_inv_e = analyzer.data['ee_power'][idx_inv_e]
                
                abs_power_e = abs(power_e)
                abs_power_inv_e = abs(power_inv_e)
                
                expected_power_e = np.sqrt(abs_power_e * abs_power_inv_e)
                symmetry_ratio_e = abs_power / expected_power_e if expected_power_e != 0 else 1
                
                e_fold.append(abs(1 - symmetry_ratio_e))
            else:
                e_fold.append(np.nan)
            
            # Calculate for pi
            if idx_pi < len(analyzer.data['ell']) and idx_inv_pi < len(analyzer.data['ell']):
                power_pi = analyzer.data['ee_power'][idx_pi]
                power_inv_pi = analyzer.data['ee_power'][idx_inv_pi]
                
                abs_power_pi = abs(power_pi)
                abs_power_inv_pi = abs(power_inv_pi)
                
                expected_power_pi = np.sqrt(abs_power_pi * abs_power_inv_pi)
                symmetry_ratio_pi = abs_power / expected_power_pi if expected_power_pi != 0 else 1
                
                pi_fold.append(abs(1 - symmetry_ratio_pi))
            else:
                pi_fold.append(np.nan)
            
            # Calculate for 2
            if idx_two < len(analyzer.data['ell']) and idx_inv_two < len(analyzer.data['ell']):
                power_two = analyzer.data['ee_power'][idx_two]
                power_inv_two = analyzer.data['ee_power'][idx_inv_two]
                
                abs_power_two = abs(power_two)
                abs_power_inv_two = abs(power_inv_two)
                
                expected_power_two = np.sqrt(abs_power_two * abs_power_inv_two)
                symmetry_ratio_two = abs_power / expected_power_two if expected_power_two != 0 else 1
                
                two_fold.append(abs(1 - symmetry_ratio_two))
            else:
                two_fold.append(np.nan)
    
    # Create visualization
    plt.figure(figsize=(14, 8))
    
    # Plot the symmetry measures
    plt.subplot(2, 1, 1)
    plt.plot(multipoles, phi_fold, 'o-', color='gold', label='Golden Ratio (φ)', alpha=0.7)
    plt.plot(multipoles, e_fold, 'o-', color='blue', label='e', alpha=0.7)
    plt.plot(multipoles, pi_fold, 'o-', color='green', label='π', alpha=0.7)
    plt.plot(multipoles, two_fold, 'o-', color='red', label='2', alpha=0.7)
    
    plt.xlabel('Multipole ℓ')
    plt.ylabel('Asymmetry Measure (0 = perfect symmetry)')
    plt.title('Symmetry Measures for Different Constants')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot the mean asymmetry for each constant
    plt.subplot(2, 1, 2)
    constants = ['Golden Ratio (φ)', 'e', 'π', '2']
    mean_asymmetries = [
        np.nanmean(phi_fold),
        np.nanmean(e_fold),
        np.nanmean(pi_fold),
        np.nanmean(two_fold)
    ]
    
    colors = ['gold', 'blue', 'green', 'red']
    
    bars = plt.bar(constants, mean_asymmetries, color=colors, alpha=0.7)
    
    # Add a horizontal line for the mean asymmetry
    plt.axhline(y=np.mean(mean_asymmetries), color='black', linestyle='--', 
                label=f'Mean asymmetry: {np.mean(mean_asymmetries):.4f}')
    
    plt.ylabel('Mean Asymmetry')
    plt.title(f'Mean Asymmetry for Different Constants (Lower is Better)')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add text annotations with values
    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{mean_asymmetries[i]:.4f}',
                ha='center', va='bottom', fontsize=10)
    
    # Add p-value and z-score as text annotation
    plt.text(0.5, 0.9, f'Golden Ratio Symmetry Test: z = {z_score:.2f}σ (p = {p_value:.6f})',
             horizontalalignment='center',
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('golden_symmetries_visualization.png')
    plt.show()
    
    print(f"Visualization saved to 'golden_symmetries_visualization.png'")
    print(f"\nGolden Symmetries Test Results:")
    print(f"Phi asymmetry: {asymmetry:.4f}")
    print(f"Mean alternative asymmetry: {mean_alternative:.4f}")
    print(f"Symmetry ratio: {symmetry_ratio:.2f}x")
    print(f"Z-score: {z_score:.2f}σ (p = {p_value:.6f})")

if __name__ == "__main__":
    main()
