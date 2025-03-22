#!/usr/bin/env python3
"""
Run the phi network test from the CosmicConsciousnessAnalyzer.
This script tests if multipoles related by powers of phi form stronger networks than random.
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from cosmic_consciousness_analyzer import CosmicConsciousnessAnalyzer
from calculate_phi_optimality import calculate_phi_optimality, interpret_phi_optimality
import networkx as nx

def main():
    """Run the phi network test."""
    # Set data directory
    data_dir = os.path.join(os.getcwd(), 'planck_data')
    if not os.path.exists(data_dir):
        print(f"Error: Data directory '{data_dir}' not found.")
        sys.exit(1)
    
    print("=== PHI NETWORK TEST ===")
    print(f"Using data directory: {data_dir}")
    
    # Initialize analyzer with 1000 Monte Carlo simulations for faster testing
    print("Creating analyzer with 1000 Monte Carlo simulations...")
    start_time = time.time()
    analyzer = CosmicConsciousnessAnalyzer(data_dir=data_dir, monte_carlo_sims=1000)
    print(f"Analyzer initialized in {time.time() - start_time:.2f} seconds.")
    print(f"Data loaded: {len(analyzer.data['ell'])} multipoles")
    
    # Run the phi network test
    print("\nRunning phi network test...")
    start_time = time.time()
    
    network_density, coherence_strength, mean_random, z_score, p_value, network_ratio = analyzer.test_phi_network()
    
    print(f"Test completed in {time.time() - start_time:.2f} seconds.")
    
    # Calculate phi optimality
    phi_optimality = calculate_phi_optimality(network_ratio, 1.0)
    phi_interpretation = interpret_phi_optimality(phi_optimality)
    
    # Print results
    print("\n=== RESULTS ===")
    print(f"Network density: {network_density:.4f}")
    print(f"Coherence strength: {coherence_strength:.4f}")
    print(f"Random network coherence: {mean_random:.4f}")
    print(f"Z-score: {z_score:.4f}")
    print(f"P-value: {p_value:.4f}")
    print(f"Network ratio: {network_ratio:.2f}x")
    print(f"Phi optimality: {phi_optimality:.4f}")
    print(f"Interpretation: {phi_interpretation}")
    
    # Create visualization
    print("\nCreating visualization...")
    
    # Create a network visualization
    try:
        # Create a network graph
        G = nx.Graph()
        
        # Add nodes (multipoles)
        for i, l in enumerate(analyzer.data['ell']):
            G.add_node(i, multipole=l, power=analyzer.data['ee_power'][i])
        
        # Add edges (phi relationships)
        phi = analyzer.phi
        tolerance = 0.1
        phi_connections = []
        
        for i in range(len(analyzer.data['ell'])):
            for j in range(i+1, len(analyzer.data['ell'])):
                l1 = analyzer.data['ell'][i]
                l2 = analyzer.data['ell'][j]
                
                # Check if their ratio is close to phi or powers of phi
                ratio = max(l1, l2) / min(l1, l2)
                for power in range(1, 4):  # Check phi, phi², phi³
                    if abs(ratio - phi**power) < tolerance:
                        phi_connections.append((i, j))
                        G.add_edge(i, j, power=power)
                        break
        
        # Create the plot
        plt.figure(figsize=(12, 10))
        
        # Position nodes using a spring layout
        pos = nx.spring_layout(G, seed=42)
        
        # Get node sizes based on power
        node_sizes = [50 + 10 * analyzer.data['ee_power'][i] for i in range(len(analyzer.data['ell']))]
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                              node_color='skyblue', alpha=0.8)
        
        # Draw edges with different colors based on power of phi
        edge_colors = []
        for u, v, data in G.edges(data=True):
            if data.get('power') == 1:
                edge_colors.append('gold')
            elif data.get('power') == 2:
                edge_colors.append('orange')
            else:
                edge_colors.append('red')
        
        nx.draw_networkx_edges(G, pos, width=1.5, alpha=0.6, edge_color=edge_colors)
        
        # Add labels for a subset of nodes (to avoid clutter)
        labels = {}
        for i, l in enumerate(analyzer.data['ell']):
            if i % 5 == 0:  # Label every 5th node
                labels[i] = str(l)
        
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)
        
        # Add a title and legend
        plt.title(f'Phi Network in CMB Data (φ-optimality: {phi_optimality:.4f}, p-value: {p_value:.4f})')
        
        # Add a custom legend
        plt.plot([0], [0], color='gold', linestyle='-', linewidth=2, label='φ¹ relationship')
        plt.plot([0], [0], color='orange', linestyle='-', linewidth=2, label='φ² relationship')
        plt.plot([0], [0], color='red', linestyle='-', linewidth=2, label='φ³ relationship')
        
        plt.legend(loc='lower right')
        plt.axis('off')
        
        # Add phi optimality and p-value as text
        plt.text(0.02, 0.98, f"Network density: {network_density:.4f}", 
                transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.text(0.02, 0.94, f"Coherence strength: {coherence_strength:.4f}", 
                transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.text(0.02, 0.90, f"Random coherence: {mean_random:.4f}", 
                transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.text(0.02, 0.86, f"φ-optimality: {phi_optimality:.4f} ({phi_interpretation})", 
                transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.text(0.02, 0.82, f"p-value: {p_value:.4f}", 
                transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Save and show the figure
        plt.tight_layout()
        plt.savefig('phi_network_test.png')
        print("Visualization saved to 'phi_network_test.png'")
    except ImportError:
        print("Warning: NetworkX not installed. Skipping network visualization.")
        
        # Create a simple bar chart instead
        plt.figure(figsize=(10, 6))
        
        # Define data for comparison
        labels = ["Phi Network", "Random Network"]
        values = [coherence_strength, mean_random]
        
        # Plot bar chart
        plt.bar(labels, values, color=['gold', 'gray'], alpha=0.7)
        plt.ylabel('Coherence Strength')
        plt.title(f'Phi Network Test: φ-optimality = {phi_optimality:.4f} ({phi_interpretation})')
        
        # Add phi optimality as text annotation
        plt.text(0.5, 0.9, f'φ-optimality: {phi_optimality:.4f}', 
                horizontalalignment='center',
                transform=plt.gca().transAxes,
                bbox=dict(facecolor='white', alpha=0.8))
        
        # Add p-value and network ratio
        plt.text(0.5, 0.82, f'p-value: {p_value:.4f}', 
                horizontalalignment='center',
                transform=plt.gca().transAxes,
                bbox=dict(facecolor='white', alpha=0.8))
        
        plt.text(0.5, 0.74, f'Network ratio: {network_ratio:.2f}x', 
                horizontalalignment='center',
                transform=plt.gca().transAxes,
                bbox=dict(facecolor='white', alpha=0.8))
        
        plt.grid(True, alpha=0.3)
        
        # Save and show the figure
        plt.tight_layout()
        plt.savefig('phi_network_test.png')
        print("Simple visualization saved to 'phi_network_test.png'")
    
    # Save results to file
    with open('phi_network_results.txt', 'w') as f:
        f.write("=== PHI NETWORK TEST RESULTS ===\n\n")
        f.write(f"Network density: {network_density:.4f}\n")
        f.write(f"Coherence strength: {coherence_strength:.4f}\n")
        f.write(f"Random network coherence: {mean_random:.4f}\n")
        f.write(f"Z-score: {z_score:.4f}\n")
        f.write(f"P-value: {p_value:.4f}\n")
        f.write(f"Network ratio: {network_ratio:.2f}x\n")
        f.write(f"Phi optimality: {phi_optimality:.4f}\n\n")
        f.write(f"Phi optimality interpretation: {phi_interpretation}\n")
        f.write(f"The phi network test shows a {phi_interpretation} alignment with golden ratio optimality.\n")
    
    print("Results saved to 'phi_network_results.txt'")

if __name__ == "__main__":
    main()
