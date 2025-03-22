#!/usr/bin/env python3
"""
Phi Network Test Module.

This test examines if multipoles related by powers of phi form stronger 
networks than random relationships in the CMB data.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import networkx as nx

def run_test(data_loader, visualizer, stats_analyzer, output_dir=None, verbose=True):
    """
    Run the phi network test.
    
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
        print("=== PHI NETWORK TEST ===")
        start_time = time.time()
    
    # Get data
    ell = data_loader.data['ell']
    ee_power = data_loader.data['ee_power']
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio
    
    # Create a network graph
    G = nx.Graph()
    
    # Add nodes (multipoles)
    for i, l in enumerate(ell):
        G.add_node(i, multipole=l, power=ee_power[i])
    
    # Add edges (phi relationships)
    tolerance = 0.1
    phi_connections = []
    
    for i in range(len(ell)):
        for j in range(i+1, len(ell)):
            l1 = ell[i]
            l2 = ell[j]
            
            # Check if their ratio is close to phi or powers of phi
            ratio = max(l1, l2) / min(l1, l2)
            for power in range(1, 4):  # Check phi, phi², phi³
                if abs(ratio - phi**power) < tolerance:
                    phi_connections.append((i, j))
                    G.add_edge(i, j, power=power)
                    break
    
    # Calculate network metrics
    network_density = nx.density(G)
    
    # Calculate coherence strength as the average correlation between connected nodes
    coherence_strength = 0
    edge_count = 0
    
    for u, v in G.edges():
        # Calculate correlation between the powers at these multipoles
        power_u = ee_power[u]
        power_v = ee_power[v]
        
        # Use a simple correlation measure
        mean_power = (power_u + power_v) / 2
        power_diff = abs(power_u - power_v)
        
        # Coherence is higher when powers are similar
        edge_coherence = 1 / (1 + power_diff / mean_power) if mean_power > 0 else 0
        coherence_strength += edge_coherence
        edge_count += 1
    
    # Average coherence
    coherence_strength = coherence_strength / edge_count if edge_count > 0 else 0
    
    # Generate random networks for comparison
    random_coherences = []
    num_random_networks = 1000
    
    for _ in range(num_random_networks):
        # Create a random network with same number of nodes and edges
        random_G = nx.gnm_random_graph(len(ell), edge_count)
        
        # Calculate coherence for random network
        random_coherence = 0
        random_edge_count = 0
        
        for u, v in random_G.edges():
            # Calculate correlation between the powers at these multipoles
            power_u = ee_power[u]
            power_v = ee_power[v]
            
            # Use a simple correlation measure
            mean_power = (power_u + power_v) / 2
            power_diff = abs(power_u - power_v)
            
            # Coherence is higher when powers are similar
            edge_coherence = 1 / (1 + power_diff / mean_power) if mean_power > 0 else 0
            random_coherence += edge_coherence
            random_edge_count += 1
        
        # Average coherence
        random_coherence = random_coherence / random_edge_count if random_edge_count > 0 else 0
        random_coherences.append(random_coherence)
    
    # Calculate statistics
    mean_random = np.mean(random_coherences)
    z_score, p_value = stats_analyzer.calculate_z_score(coherence_strength, random_coherences)
    
    # Calculate network ratio
    network_ratio = coherence_strength / mean_random if mean_random > 0 else float('inf')
    
    # Calculate phi optimality
    phi_optimality = stats_analyzer.calculate_phi_optimality(network_ratio, 1.0)
    
    # Create visualization
    if visualizer is not None:
        try:
            # Create the plot
            plt.figure(figsize=(12, 10))
            
            # Position nodes using a spring layout
            pos = nx.spring_layout(G, seed=42)
            
            # Get node sizes based on power
            node_sizes = [50 + 10 * ee_power[i] for i in range(len(ell))]
            
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
            for i, l in enumerate(ell):
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
            
            plt.text(0.02, 0.86, f"φ-optimality: {phi_optimality:.4f}", 
                    transform=plt.gca().transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.text(0.02, 0.82, f"p-value: {p_value:.4f}", 
                    transform=plt.gca().transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
        except Exception as e:
            if verbose:
                print(f"Warning: Network visualization failed: {e}")
            
            # Create a simple bar chart instead
            plt.figure(figsize=(10, 6))
            
            # Define data for comparison
            labels = ["Phi Network", "Random Network"]
            values = [coherence_strength, mean_random]
            
            # Plot bar chart
            plt.bar(labels, values, color=['gold', 'gray'], alpha=0.7)
            plt.ylabel('Coherence Strength')
            plt.title(f'Phi Network Test: φ-optimality = {phi_optimality:.4f}')
            
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
        
        # Save the figure
        plt.tight_layout()
        
        if output_dir:
            output_path = os.path.join(output_dir, 'phi_network_test.png')
        else:
            output_path = 'phi_network_test.png'
            
        plt.savefig(output_path, dpi=300)
        
        if verbose:
            print(f"Visualization saved to '{output_path}'")
    
    # Save results to file
    if output_dir:
        results_path = os.path.join(output_dir, 'phi_network_results.txt')
    else:
        results_path = 'phi_network_results.txt'
        
    with open(results_path, 'w') as f:
        f.write("=== PHI NETWORK TEST RESULTS ===\n\n")
        f.write(f"Network density: {network_density:.4f}\n")
        f.write(f"Coherence strength: {coherence_strength:.4f}\n")
        f.write(f"Random network coherence: {mean_random:.4f}\n")
        f.write(f"Z-score: {z_score:.4f}\n")
        f.write(f"P-value: {p_value:.4f}\n")
        f.write(f"Network ratio: {network_ratio:.2f}x\n")
        f.write(f"Phi optimality: {phi_optimality:.4f}\n\n")
        
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
            
        f.write(f"Phi optimality interpretation: {interpretation}\n")
        f.write(f"The phi network test shows a {interpretation} alignment with golden ratio optimality.\n")
    
    if verbose:
        print(f"Results saved to '{results_path}'")
        print(f"Test completed in {time.time() - start_time:.2f} seconds.")
    
    # Prepare results dictionary
    results = {
        'test_name': 'Phi Network Test',
        'network_density': network_density,
        'coherence_strength': coherence_strength,
        'mean_random': mean_random,
        'z_score': z_score,
        'p_value': p_value,
        'network_ratio': network_ratio,
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
    stats_analyzer = get_statistical_analyzer()
    
    # Run test
    results = run_test(data_loader, visualizer, stats_analyzer, verbose=True)
    
    # Print key results
    print("\n=== KEY RESULTS ===")
    print(f"Network density: {results['network_density']:.4f}")
    print(f"Coherence strength: {results['coherence_strength']:.4f}")
    print(f"Network ratio: {results['network_ratio']:.2f}x")
    print(f"Phi optimality: {results['phi_optimality']:.4f} ({results['interpretation']})")
    print(f"P-value: {results['p_value']:.6f}")
