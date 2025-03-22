#!/usr/bin/env python3
"""
Phi-Optimality Demonstration

This script demonstrates the phi-optimality calculation and its properties.
Phi-optimality measures how close a ratio is to the golden ratio (or its inverse).
"""

import numpy as np
import matplotlib.pyplot as plt

def calculate_phi_optimality(observed_ratio, target_ratio=None):
    """
    Calculate phi-optimality for a given ratio.
    
    Parameters:
    -----------
    observed_ratio : float
        The observed ratio to evaluate
    target_ratio : float, optional
        The target ratio to compare against. Default is the inverse golden ratio.
    
    Returns:
    --------
    float
        Phi-optimality value between -1 and 1
    """
    # Calculate the golden ratio
    phi = (1 + np.sqrt(5)) / 2
    
    # Default to inverse golden ratio if no target provided
    if target_ratio is None:
        target_ratio = 1 / phi
    
    # Ensure ratio is less than 1 (use inverse if needed)
    if observed_ratio > 1:
        observed_ratio = 1 / observed_ratio
    
    # Calculate phi-optimality (bounded between -1 and 1)
    optimality = max(-1, min(1, 1 - abs(observed_ratio - target_ratio) / target_ratio))
    
    return optimality

def main():
    """Main function to demonstrate phi-optimality."""
    
    # Calculate the golden ratio
    phi = (1 + np.sqrt(5)) / 2
    inv_phi = 1 / phi
    
    print(f"Golden Ratio (φ): {phi:.8f}")
    print(f"Inverse Golden Ratio (1/φ): {inv_phi:.8f}")
    
    # Example ratios to test
    example_ratios = [
        (inv_phi, "Inverse Golden Ratio (1/φ)"),
        (phi, "Golden Ratio (φ)"),
        (0.5, "1/2"),
        (1/3, "1/3"),
        (2/3, "2/3"),
        (3/5, "3/5 (Fibonacci approximation)"),
        (5/8, "5/8 (Fibonacci approximation)"),
        (8/13, "8/13 (Fibonacci approximation)"),
        (0.7, "0.7"),
        (0.4, "0.4"),
        (0.1, "0.1"),
        (0.9, "0.9")
    ]
    
    # Calculate phi-optimality for each ratio
    print("\nPhi-Optimality Examples:")
    print("------------------------")
    for ratio, name in example_ratios:
        optimality = calculate_phi_optimality(ratio)
        print(f"{name.ljust(30)}: {ratio:.6f} -> {optimality:.6f}")
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Phi-optimality function
    plt.subplot(2, 1, 1)
    
    x = np.linspace(0.01, 1, 1000)
    y = [calculate_phi_optimality(r) for r in x]
    
    plt.plot(x, y, 'b-')
    plt.axvline(x=inv_phi, color='red', linestyle='--', 
               label=f'Inverse Golden Ratio: {inv_phi:.6f}')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.2)
    
    # Mark example ratios
    for ratio, name in example_ratios:
        actual_ratio = ratio if ratio <= 1 else 1/ratio
        optimality = calculate_phi_optimality(ratio)
        plt.plot(actual_ratio, optimality, 'go', markersize=8)
        plt.annotate(name.split()[0], 
                    xy=(actual_ratio, optimality),
                    xytext=(0, 10), textcoords='offset points',
                    ha='center', fontsize=8)
    
    plt.xlabel('Ratio')
    plt.ylabel('Phi-Optimality')
    plt.title('Phi-Optimality Function')
    plt.ylim(-1.1, 1.1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot 2: Bar chart of example ratios
    plt.subplot(2, 1, 2)
    
    ratios = [r for r, _ in example_ratios]
    names = [n.split('(')[0].strip() for _, n in example_ratios]
    optimalities = [calculate_phi_optimality(r) for r in ratios]
    
    # Color bars by optimality
    colors = ['green' if o >= 0.8 else 'orange' if o >= 0.5 else 
              'blue' if o >= 0 else 'red' for o in optimalities]
    
    plt.bar(names, optimalities, color=colors)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.2)
    
    plt.xlabel('Ratio')
    plt.ylabel('Phi-Optimality')
    plt.title('Phi-Optimality of Various Ratios')
    plt.ylim(-1.1, 1.1)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('phi_optimality_demo.png')
    plt.show()
    
    print("\nVisualization saved as 'phi_optimality_demo.png'")

if __name__ == "__main__":
    main()
