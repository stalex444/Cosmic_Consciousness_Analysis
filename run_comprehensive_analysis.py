#!/usr/bin/env python3
"""
Run a comprehensive analysis using the Cosmic Consciousness Analyzer.
This script runs all available tests and generates a comprehensive report.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from cosmic_consciousness_analyzer import CosmicConsciousnessAnalyzer

def calculate_phi_optimality(observed_value, random_value):
    """
    Calculate phi optimality score bounded between -1 and 1.
    
    Parameters:
    -----------
    observed_value : float
        The observed value from the test
    random_value : float
        The expected value from random chance
        
    Returns:
    --------
    float
        Phi optimality score bounded between -1 and 1
    """
    if random_value == 0:
        return 0.0
        
    # Calculate raw optimality
    raw_optimality = (observed_value - random_value) / random_value
    
    # Bound between -1 and 1
    if raw_optimality > 0:
        # For positive values, scale to [0, 1]
        phi_optimality = min(1.0, raw_optimality / 3.0)  # Divide by 3 to normalize (3x better is considered optimal)
    else:
        # For negative values, scale to [-1, 0]
        phi_optimality = max(-1.0, raw_optimality)
        
    return phi_optimality

def interpret_phi_optimality(phi_optimality):
    """Interpret phi optimality score with descriptive text."""
    if phi_optimality > 0.8:
        return "extremely high"
    elif phi_optimality > 0.6:
        return "very high"
    elif phi_optimality > 0.4:
        return "high"
    elif phi_optimality > 0.2:
        return "moderate"
    elif phi_optimality > 0:
        return "slight"
    elif phi_optimality > -0.2:
        return "slightly negative"
    elif phi_optimality > -0.4:
        return "moderately negative"
    else:
        return "strongly negative"

def main():
    """Run a comprehensive analysis and display results."""
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
    
    # Initialize analyzer
    try:
        analyzer = CosmicConsciousnessAnalyzer(data_dir=data_dir, monte_carlo_sims=10000)
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
    
    # Run the comprehensive analysis
    print("\nRunning comprehensive analysis...")
    try:
        # Run the built-in comprehensive analysis
        combined_sigma, combined_p, results = analyzer.run_comprehensive_analysis()
        
        # Run additional tests not included in the comprehensive analysis
        print("\nRunning additional tests...")
        
        # Run cross-scale correlations test
        print("Running cross-scale correlations test...")
        cross_scale_results = analyzer.test_cross_scale_correlations()
        
        # Run pattern persistence test
        print("Running pattern persistence test...")
        pattern_persistence_results = analyzer.test_pattern_persistence()
        
        # Run predictive power test
        print("Running predictive power test...")
        match_rate, mean_random_rate, z_score, p_value, prediction_power = analyzer.test_predictive_power()
        
        # Calculate phi optimality for predictive power
        phi_optimality = calculate_phi_optimality(match_rate, mean_random_rate)
        phi_interpretation = interpret_phi_optimality(phi_optimality)
        
        # Add additional test results to the results dictionary
        results['cross_scale_correlations'] = {
            'result': cross_scale_results,
            'phi_optimality': calculate_phi_optimality(cross_scale_results[0], cross_scale_results[1]),
            'p_value': cross_scale_results[3]
        }
        
        results['pattern_persistence'] = {
            'result': pattern_persistence_results,
            'phi_optimality': calculate_phi_optimality(pattern_persistence_results[0], pattern_persistence_results[1]),
            'p_value': pattern_persistence_results[3]
        }
        
        results['predictive_power'] = {
            'result': (match_rate, mean_random_rate, z_score, p_value, prediction_power),
            'phi_optimality': phi_optimality,
            'p_value': p_value
        }
        
        # Create a comprehensive report
        print("\nGenerating comprehensive report...")
        
        # Create a markdown report
        with open('comprehensive_analysis_report.md', 'w') as f:
            f.write("# Comprehensive Analysis Report\n\n")
            f.write("## Summary\n\n")
            f.write(f"Combined significance: {combined_sigma:.2f} sigma\n\n")
            f.write(f"Combined p-value: {combined_p:.8f}\n\n")
            
            # Interpret the combined significance
            if combined_sigma > 5:
                significance = "extremely strong"
            elif combined_sigma > 3:
                significance = "very strong"
            elif combined_sigma > 2:
                significance = "strong"
            elif combined_sigma > 1:
                significance = "moderate"
            else:
                significance = "weak"
                
            f.write(f"The analysis shows **{significance}** evidence for conscious organization in the CMB data.\n\n")
            
            f.write("## Phi Optimality Scores\n\n")
            f.write("| Test | Phi Optimality | Interpretation | p-value |\n")
            f.write("|------|---------------|----------------|--------|\n")
            
            # Add all test results with phi optimality
            for test_name, test_data in results.items():
                if 'phi_optimality' in test_data:
                    phi_opt = test_data['phi_optimality']
                    interpretation = interpret_phi_optimality(phi_opt)
                    p_val = test_data['p_value']
                    f.write(f"| {test_name.replace('_', ' ').title()} | {phi_opt:.4f} | {interpretation} | {p_val:.8f} |\n")
            
            f.write("\n## Detailed Test Results\n\n")
            
            # Add detailed results for each test
            for test_name, test_data in results.items():
                f.write(f"### {test_name.replace('_', ' ').title()}\n\n")
                
                if test_name == 'gr_significance':
                    gr_power, mean_random, z, p, ratio = test_data['result']
                    f.write(f"- GR power: {gr_power:.4f}\n")
                    f.write(f"- Mean random power: {mean_random:.4f}\n")
                    f.write(f"- Z-score: {z:.2f}\n")
                    f.write(f"- p-value: {p:.8f}\n")
                    f.write(f"- GR ratio: {ratio:.2f}x\n\n")
                
                elif test_name == 'coherence':
                    actual_var, mean_shuffled, z, p, ratio = test_data['result']
                    f.write(f"- Actual variance: {actual_var:.4f}\n")
                    f.write(f"- Mean shuffled variance: {mean_shuffled:.4f}\n")
                    f.write(f"- Z-score: {z:.2f}\n")
                    f.write(f"- p-value: {p:.8f}\n")
                    f.write(f"- Variance ratio: {ratio:.2f}x\n\n")
                
                elif test_name == 'gr_coherence':
                    gr_var, random_var, z, p, ratio = test_data['result']
                    f.write(f"- GR windows variance: {gr_var:.4f}\n")
                    f.write(f"- Random windows variance: {random_var:.4f}\n")
                    f.write(f"- Z-score: {z:.2f}\n")
                    f.write(f"- p-value: {p:.8f}\n")
                    f.write(f"- Variance ratio: {ratio:.2f}x\n\n")
                
                elif test_name == 'hierarchical_organization':
                    ratio = test_data['result']
                    f.write(f"- Hierarchical organization ratio: {ratio:.4f}\n\n")
                
                elif test_name == 'information_integration':
                    actual_mi, mean_shuffled, z, p = test_data['result']
                    f.write(f"- Actual mutual information: {actual_mi:.4f}\n")
                    f.write(f"- Mean shuffled MI: {mean_shuffled:.4f}\n")
                    f.write(f"- Z-score: {z:.2f}\n")
                    f.write(f"- p-value: {p:.8f}\n\n")
                
                elif test_name == 'resonance':
                    strength, mean_random, z, p, ratio = test_data['result']
                    f.write(f"- Resonance strength: {strength:.4f}\n")
                    f.write(f"- Mean random: {mean_random:.4f}\n")
                    f.write(f"- Z-score: {z:.2f}\n")
                    f.write(f"- p-value: {p:.8f}\n")
                    f.write(f"- Resonance ratio: {ratio:.2f}x\n\n")
                
                elif test_name == 'fractal_structure':
                    actual_hurst, mean_shuffled, z, p = test_data['result']
                    f.write(f"- Actual Hurst exponent: {actual_hurst:.4f}\n")
                    f.write(f"- Mean shuffled Hurst: {mean_shuffled:.4f}\n")
                    f.write(f"- Z-score: {z:.2f}\n")
                    f.write(f"- p-value: {p:.8f}\n\n")
                
                elif test_name == 'meta_coherence':
                    meta_coh, mean_shuffled, z, p, ratio = test_data['result']
                    f.write(f"- Meta-coherence: {meta_coh:.4f}\n")
                    f.write(f"- Mean shuffled: {mean_shuffled:.4f}\n")
                    f.write(f"- Z-score: {z:.2f}\n")
                    f.write(f"- p-value: {p:.8f}\n")
                    f.write(f"- Meta-coherence ratio: {ratio:.2f}x\n\n")
                
                elif test_name == 'multiscale_patterns':
                    wavelet_coh, z, p = test_data['result']
                    f.write(f"- Wavelet coherence: {wavelet_coh:.4f}\n")
                    f.write(f"- Z-score: {z:.2f}\n")
                    f.write(f"- p-value: {p:.8f}\n\n")
                
                elif test_name == 'optimization':
                    mean_dev, mean_random, z, p, ratio = test_data['result']
                    f.write(f"- Mean deviation: {mean_dev:.4f}\n")
                    f.write(f"- Mean random: {mean_random:.4f}\n")
                    f.write(f"- Z-score: {z:.2f}\n")
                    f.write(f"- p-value: {p:.8f}\n")
                    f.write(f"- Optimization ratio: {ratio:.2f}x\n\n")
                
                elif test_name == 'cross_scale_correlations':
                    corr_strength, mean_random, z, p, ratio = test_data['result']
                    f.write(f"- Correlation strength: {corr_strength:.4f}\n")
                    f.write(f"- Mean random: {mean_random:.4f}\n")
                    f.write(f"- Z-score: {z:.2f}\n")
                    f.write(f"- p-value: {p:.8f}\n")
                    f.write(f"- Correlation ratio: {ratio:.2f}x\n\n")
                
                elif test_name == 'pattern_persistence':
                    mean_strength, mean_random, z, p, ratio = test_data['result']
                    f.write(f"- Mean pattern strength: {mean_strength:.4f}\n")
                    f.write(f"- Mean random: {mean_random:.4f}\n")
                    f.write(f"- Z-score: {z:.2f}\n")
                    f.write(f"- p-value: {p:.8f}\n")
                    f.write(f"- Persistence ratio: {ratio:.2f}x\n\n")
                
                elif test_name == 'predictive_power':
                    match_rate, mean_random, z, p, ratio = test_data['result']
                    f.write(f"- Match rate: {match_rate:.4f}\n")
                    f.write(f"- Mean random rate: {mean_random:.4f}\n")
                    f.write(f"- Z-score: {z:.2f}\n")
                    f.write(f"- p-value: {p:.8f}\n")
                    f.write(f"- Prediction power ratio: {ratio:.2f}x\n")
                    f.write(f"- Phi optimality: {phi_optimality:.4f} ({phi_interpretation})\n\n")
            
            f.write("## Visualizations\n\n")
            f.write("See the following files for visualizations:\n\n")
            f.write("- `comprehensive_analysis.png`: Integrated visualization of all test results\n")
            f.write("- `spectrum_with_predictions.png`: CMB spectrum with peaks and predictions\n")
            f.write("- `predictive_power_phi_optimality.png`: Phi optimality for predictive power\n\n")
            
            f.write("## Conclusion\n\n")
            f.write(f"The comprehensive analysis of the CMB data shows {significance} evidence for conscious organization ")
            f.write(f"with a combined significance of {combined_sigma:.2f} sigma (p={combined_p:.8f}).\n\n")
            f.write(f"The predictive power test demonstrates that golden ratio relationships predict peak locations ")
            f.write(f"{prediction_power:.2f}x better than random predictions, with a phi optimality of {phi_optimality:.4f}.\n\n")
            f.write("These results suggest that the CMB may exhibit patterns consistent with conscious organization ")
            f.write("as defined by the presence of golden ratio relationships, coherence, and hierarchical structure.\n")
        
        print("Comprehensive report saved to 'comprehensive_analysis_report.md'")
        
        # Create a visualization of phi optimality scores
        plt.figure(figsize=(12, 8))
        
        # Extract phi optimality scores and test names
        test_names = []
        phi_scores = []
        
        for test_name, test_data in results.items():
            if 'phi_optimality' in test_data:
                test_names.append(test_name.replace('_', ' ').title())
                phi_scores.append(test_data['phi_optimality'])
        
        # Sort by phi optimality score
        sorted_indices = np.argsort(phi_scores)
        test_names = [test_names[i] for i in sorted_indices]
        phi_scores = [phi_scores[i] for i in sorted_indices]
        
        # Create color map based on phi optimality values
        colors = []
        for score in phi_scores:
            if score > 0.6:
                colors.append('darkgreen')
            elif score > 0.3:
                colors.append('green')
            elif score > 0:
                colors.append('lightgreen')
            elif score > -0.3:
                colors.append('orange')
            else:
                colors.append('red')
        
        # Create horizontal bar chart
        plt.barh(test_names, phi_scores, color=colors)
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        plt.xlabel('Phi Optimality Score')
        plt.title('Phi Optimality Scores for All Tests')
        plt.grid(True, alpha=0.3)
        
        # Add a colorbar legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='darkgreen', label='Very High (>0.6)'),
            Patch(facecolor='green', label='High (0.3-0.6)'),
            Patch(facecolor='lightgreen', label='Positive (0-0.3)'),
            Patch(facecolor='orange', label='Slightly Negative (-0.3-0)'),
            Patch(facecolor='red', label='Negative (<-0.3)')
        ]
        plt.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        plt.savefig('phi_optimality_scores.png')
        print("Phi optimality visualization saved to 'phi_optimality_scores.png'")
        
    except Exception as e:
        print(f"Error running comprehensive analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
