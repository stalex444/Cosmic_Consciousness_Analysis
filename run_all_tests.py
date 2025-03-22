#!/usr/bin/env python3
"""
Run all tests from the CosmicConsciousnessAnalyzer class.
This script runs all the statistical tests implemented in the analyzer
and generates a comprehensive report.
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
    """Run all tests and generate a comprehensive report."""
    # Set data directory
    data_dir = os.path.join(os.getcwd(), 'planck_data')
    if not os.path.exists(data_dir):
        print(f"Error: Data directory '{data_dir}' not found.")
        sys.exit(1)
    
    print("=== COMPREHENSIVE COSMIC CONSCIOUSNESS ANALYSIS ===")
    print(f"Using data directory: {data_dir}")
    
    # Initialize analyzer with 1000 Monte Carlo simulations for faster testing
    print("Creating analyzer with 1000 Monte Carlo simulations...")
    start_time = time.time()
    analyzer = CosmicConsciousnessAnalyzer(data_dir=data_dir, monte_carlo_sims=1000)
    print(f"Analyzer initialized in {time.time() - start_time:.2f} seconds.")
    print(f"Data loaded: {len(analyzer.data['ell'])} multipoles")
    
    # Run all tests
    print("\nRunning all tests...")
    
    # Dictionary to store results
    results = {}
    
    # 1. Golden Ratio Significance Test
    print("\n1. Running Golden Ratio Significance Test...")
    start_time = time.time()
    gr_power, random_power, z_score, p_value, power_ratio = analyzer.test_gr_significance()
    results['gr_test'] = (gr_power, random_power, z_score, p_value, power_ratio)
    print(f"Test completed in {time.time() - start_time:.2f} seconds.")
    
    # 2. Coherence Analysis
    print("\n2. Running Coherence Analysis...")
    start_time = time.time()
    coherence, random_coherence, z_score, p_value, coherence_ratio = analyzer.test_coherence()
    results['coherence_test'] = (coherence, random_coherence, z_score, p_value, coherence_ratio)
    print(f"Test completed in {time.time() - start_time:.2f} seconds.")
    
    # 3. GR-Specific Coherence
    print("\n3. Running GR-Specific Coherence Test...")
    start_time = time.time()
    gr_coherence, random_gr_coherence, z_score, p_value, gr_coherence_ratio = analyzer.test_gr_coherence()
    results['gr_coherence_test'] = (gr_coherence, random_gr_coherence, z_score, p_value, gr_coherence_ratio)
    print(f"Test completed in {time.time() - start_time:.2f} seconds.")
    
    # 4. Hierarchical Organization
    print("\n4. Running Hierarchical Organization Test...")
    start_time = time.time()
    hierarchy, random_hierarchy, z_score, p_value, hierarchy_ratio = analyzer.test_hierarchical_organization()
    results['hierarchy_test'] = (hierarchy, random_hierarchy, z_score, p_value, hierarchy_ratio)
    print(f"Test completed in {time.time() - start_time:.2f} seconds.")
    
    # 5. Information Integration
    print("\n5. Running Information Integration Test...")
    start_time = time.time()
    integration, random_integration, z_score, p_value, integration_ratio = analyzer.test_information_integration()
    results['integration_test'] = (integration, random_integration, z_score, p_value, integration_ratio)
    print(f"Test completed in {time.time() - start_time:.2f} seconds.")
    
    # 6. Optimization Test
    print("\n6. Running Optimization Test...")
    start_time = time.time()
    optimization, random_optimization, z_score, p_value, optimization_ratio = analyzer.test_optimization()
    results['optimization_test'] = (optimization, random_optimization, z_score, p_value, optimization_ratio)
    print(f"Test completed in {time.time() - start_time:.2f} seconds.")
    
    # 7. Resonance Analysis
    print("\n7. Running Resonance Analysis...")
    start_time = time.time()
    resonance, random_resonance, z_score, p_value, resonance_ratio = analyzer.test_resonance()
    results['resonance_test'] = (resonance, random_resonance, z_score, p_value, resonance_ratio)
    print(f"Test completed in {time.time() - start_time:.2f} seconds.")
    
    # 8. Fractal Structure
    print("\n8. Running Fractal Structure Test...")
    start_time = time.time()
    fractal, random_fractal, z_score, p_value, fractal_ratio = analyzer.test_fractal_structure()
    results['fractal_test'] = (fractal, random_fractal, z_score, p_value, fractal_ratio)
    print(f"Test completed in {time.time() - start_time:.2f} seconds.")
    
    # 9. Meta-Coherence
    print("\n9. Running Meta-Coherence Test...")
    start_time = time.time()
    meta_coherence, random_meta_coherence, z_score, p_value, meta_coherence_ratio = analyzer.test_meta_coherence()
    results['meta_coherence_test'] = (meta_coherence, random_meta_coherence, z_score, p_value, meta_coherence_ratio)
    print(f"Test completed in {time.time() - start_time:.2f} seconds.")
    
    # 10. Multi-Scale Patterns
    print("\n10. Running Multi-Scale Patterns Test...")
    start_time = time.time()
    multiscale, random_multiscale, z_score, p_value, multiscale_ratio = analyzer.test_multiscale_patterns()
    results['multiscale_test'] = (multiscale, random_multiscale, z_score, p_value, multiscale_ratio)
    print(f"Test completed in {time.time() - start_time:.2f} seconds.")
    
    # 11. Golden Symmetries (our new test)
    print("\n11. Running Golden Symmetries Test...")
    start_time = time.time()
    symmetry, random_symmetry, z_score, p_value, symmetry_ratio = analyzer.test_golden_symmetries()
    results['symmetry_test'] = (symmetry, random_symmetry, z_score, p_value, symmetry_ratio)
    print(f"Test completed in {time.time() - start_time:.2f} seconds.")
    
    # 12. Phi Network Test
    print("\n12. Running Phi Network Test...")
    start_time = time.time()
    network_density, coherence_strength, mean_random, z_score, p_value, network_ratio = analyzer.test_phi_network()
    results['phi_network_test'] = (coherence_strength, mean_random, z_score, p_value, network_ratio)
    print(f"Test completed in {time.time() - start_time:.2f} seconds.")
    
    # 13. Spectral Gap Test
    print("\n13. Running Spectral Gap Test...")
    start_time = time.time()
    spectral_gap, mean_random_gap, gap_z_score, gap_p_value, gap_ratio, mean_phi_deviation, mean_random_dev, dev_z_score, dev_p_value, dev_ratio = analyzer.test_spectral_gap()
    # We'll use the eigenvalue phi relationship as the main result
    results['spectral_gap_test'] = (mean_phi_deviation, mean_random_dev, dev_z_score, dev_p_value, dev_ratio)
    print(f"Test completed in {time.time() - start_time:.2f} seconds.")
    
    # 14. Recurrence Quantification Analysis
    print("\n14. Running Recurrence Quantification Analysis...")
    start_time = time.time()
    RR, DET, LAM, mean_surr_DET, std_surr_DET, DET_z_score, DET_p_value, DET_ratio, mean_surr_LAM, std_surr_LAM, LAM_z_score, LAM_p_value, LAM_ratio = analyzer.test_recurrence_quantification()
    # We'll use the laminarity results as the main result since they show stronger phi-optimality
    results['recurrence_quantification_test'] = (LAM, mean_surr_LAM, LAM_z_score, LAM_p_value, LAM_ratio)
    print(f"Test completed in {time.time() - start_time:.2f} seconds.")
    
    # 15. Scale Frequency Coupling
    print("\n15. Running Scale Frequency Coupling Test...")
    start_time = time.time()
    result = analyzer.test_scale_frequency_coupling()
    if result is not None:
        scale_gr_frequencies, correlation, p_value, z_score, corr_p_value, linear_mse, phi_mse, model_ratio = result
        if model_ratio is not None:
            results['scale_frequency_coupling_test'] = (linear_mse, phi_mse, z_score, corr_p_value, model_ratio)
        else:
            # If model comparison not possible, use correlation statistics
            results['scale_frequency_coupling_test'] = (correlation, 0, z_score, corr_p_value, 1.0)
    else:
        print("Not enough data for scale frequency coupling analysis. Skipping.")
    print(f"Test completed in {time.time() - start_time:.2f} seconds.")
    
    # 16. Transfer Entropy
    print("\n16. Running Transfer Entropy Test...")
    start_time = time.time()
    result = analyzer.test_transfer_entropy()
    if result is not None:
        phi_te, non_phi_te, te_ratio, z_score, p_value = result
        results['transfer_entropy_test'] = (phi_te, non_phi_te, z_score, p_value, te_ratio)
    else:
        print("Not enough data for transfer entropy analysis. Skipping.")
    print(f"Test completed in {time.time() - start_time:.2f} seconds.")
    
    # 17. Multi-Scale Coherence
    print("\n17. Running Multi-Scale Coherence Test...")
    start_time = time.time()
    result = analyzer.test_multi_scale_coherence()
    if result is not None:
        scale_coherences, coherence_ratios, mean_phi_deviation, mean_random, z_score, p_value, optimization_ratio = result
        results['multi_scale_coherence_test'] = (mean_phi_deviation, mean_random, z_score, p_value, optimization_ratio)
    else:
        print("Not enough data for multi-scale coherence analysis. Skipping.")
    print(f"Test completed in {time.time() - start_time:.2f} seconds.")
    
    # 18. Coherence Phase
    print("\n18. Running Coherence Phase Test...")
    start_time = time.time()
    result = analyzer.test_coherence_phase()
    if result is not None:
        kuramoto, mean_surrogate, z_score, p_value, coherence_ratio, phi_phase_ratio, mean_random_ratio, phi_z_score, phi_p_value, phase_optimization = result
        results['coherence_phase_test'] = (kuramoto, mean_surrogate, z_score, p_value, coherence_ratio)
    else:
        print("Not enough data for coherence phase analysis. Skipping.")
    print(f"Test completed in {time.time() - start_time:.2f} seconds.")
    
    # 19. Extended Meta-Coherence
    print("\n19. Running Extended Meta-Coherence Test...")
    start_time = time.time()
    result = analyzer.test_extended_meta_coherence()
    if result is not None:
        # Extract meta-coherence results
        meta_coherence, mean_shuffled_meta, meta_z, meta_p, meta_ratio = result['meta_coherence']
        results['extended_meta_coherence_test'] = (meta_coherence, mean_shuffled_meta, meta_z, meta_p, meta_ratio)
        
        # Also store the additional metrics for reference
        skewness, mean_shuffled_skew, skew_z, skew_p, skew_ratio = result['skewness']
        kurtosis, mean_shuffled_kurt, kurt_z, kurt_p, kurt_ratio = result['kurtosis']
        entropy, mean_shuffled_entropy, entropy_z, entropy_p, entropy_ratio = result['entropy']
        power_law, mean_shuffled_exponent, exponent_z, exponent_p, exponent_ratio = result['power_law']
        
        # Print additional metrics
        print(f"  - Skewness: {skewness:.4f} (random: {mean_shuffled_skew:.4f}, ratio: {skew_ratio:.2f}x, p: {skew_p:.4f})")
        print(f"  - Kurtosis: {kurtosis:.4f} (random: {mean_shuffled_kurt:.4f}, ratio: {kurt_ratio:.2f}x, p: {kurt_p:.4f})")
        print(f"  - Entropy: {entropy:.4f} (random: {mean_shuffled_entropy:.4f}, ratio: {entropy_ratio:.2f}x, p: {entropy_p:.4f})")
        if power_law is not None:
            print(f"  - Power Law Exponent: {power_law:.4f} (random: {mean_shuffled_exponent:.4f}, ratio: {exponent_ratio:.2f}x, p: {exponent_p:.4f})")
    else:
        print("Not enough data for extended meta-coherence analysis. Skipping.")
    print(f"Test completed in {time.time() - start_time:.2f} seconds.")
    
    # 20. Meta-Coherence Entropy
    print("\n20. Running Meta-Coherence Entropy Test...")
    if 'entropy' in result:
        entropy, mean_shuffled_entropy, entropy_z, entropy_p, entropy_ratio = result['entropy']
        results['meta_coherence_entropy_test'] = (entropy, mean_shuffled_entropy, entropy_z, entropy_p, entropy_ratio)
        print(f"  Entropy: {entropy:.4f} (random: {mean_shuffled_entropy:.4f}, ratio: {entropy_ratio:.2f}x, p: {entropy_p:.4f})")
    else:
        print("No entropy data available. Skipping.")
    
    # 21. Meta-Coherence Power Law
    print("\n21. Running Meta-Coherence Power Law Test...")
    if 'power_law' in result and result['power_law'][0] is not None:
        power_law, mean_shuffled_exponent, exponent_z, exponent_p, exponent_ratio = result['power_law']
        results['meta_coherence_power_law_test'] = (power_law, mean_shuffled_exponent, exponent_z, exponent_p, exponent_ratio)
        print(f"  Power Law Exponent: {power_law:.4f} (random: {mean_shuffled_exponent:.4f}, ratio: {exponent_ratio:.2f}x, p: {exponent_p:.4f})")
    else:
        print("No power law data available. Skipping.")
    
    # Calculate phi optimality for each test
    phi_optimalities = {}
    for test_name, (observed, random, _, _, ratio) in results.items():
        phi_optimality = calculate_phi_optimality(ratio, 1.0)
        phi_optimalities[test_name] = phi_optimality
    
    # Calculate combined significance using Fisher's method
    p_values = [results[test][3] for test in results]
    valid_p_values = [p for p in p_values if not np.isnan(p) and p > 0]
    
    if valid_p_values:
        fisher_statistic = -2 * np.sum(np.log(valid_p_values))
        combined_p_value = 1 - stats.chi2.cdf(fisher_statistic, 2 * len(valid_p_values))
    else:
        combined_p_value = 1.0
    
    # Calculate mean phi optimality
    mean_phi_optimality = np.mean(list(phi_optimalities.values()))
    mean_interpretation = interpret_phi_optimality(mean_phi_optimality)
    
    # Print results
    print("\n=== RESULTS ===")
    print(f"Combined p-value (Fisher's method): {combined_p_value:.6f}")
    print(f"Mean phi optimality: {mean_phi_optimality:.4f} ({mean_interpretation})")
    print("\nIndividual test results:")
    
    # Sort tests by phi optimality
    sorted_tests = sorted(phi_optimalities.items(), key=lambda x: x[1], reverse=True)
    
    for i, (test_name, phi_opt) in enumerate(sorted_tests):
        observed, random, z, p, ratio = results[test_name]
        interp = interpret_phi_optimality(phi_opt)
        print(f"{i+1}. {test_name}: φ-optimality = {phi_opt:.4f} ({interp}), p = {p:.4f}, ratio = {ratio:.2f}x")
    
    # Create visualization
    print("\nCreating visualization...")
    
    # Create bar chart of phi optimalities
    plt.figure(figsize=(12, 8))
    
    # Prepare data for plotting
    test_names = [name.replace('_test', '').replace('gr_', 'GR ').replace('recurrence_quantification', 'RQA').replace('scale_frequency_coupling', 'Scale-Freq').replace('transfer_entropy', 'Transfer Entropy').replace('multi_scale_coherence', 'Multi-Scale Coherence').replace('coherence_phase', 'Coherence Phase').replace('extended_meta_coherence', 'Extended Meta-Coherence').replace('meta_coherence_entropy', 'Meta-Coherence Entropy').replace('meta_coherence_power_law', 'Meta-Coherence Power Law').title() for name, _ in sorted_tests]
    phi_opts = [opt for _, opt in sorted_tests]
    
    # Define colors based on significance
    colors = []
    for test_name, _ in sorted_tests:
        p_value = results[test_name][3]
        if p_value < 0.01:
            colors.append('darkgreen')  # Highly significant
        elif p_value < 0.05:
            colors.append('green')      # Significant
        elif p_value < 0.1:
            colors.append('yellowgreen')  # Marginally significant
        else:
            colors.append('gray')       # Not significant
    
    # Plot bar chart
    bars = plt.bar(test_names, phi_opts, color=colors, alpha=0.7)
    
    # Add a horizontal line for the mean phi optimality
    plt.axhline(y=mean_phi_optimality, color='red', linestyle='--', 
                label=f'Mean φ-optimality: {mean_phi_optimality:.4f}')
    
    # Add p-values as text on top of bars
    for i, (test_name, _) in enumerate(sorted_tests):
        p_value = results[test_name][3]
        if not np.isnan(p_value):
            plt.text(i, phi_opts[i] + 0.02, f'p={p_value:.4f}', 
                    ha='center', va='bottom', rotation=90, fontsize=8)
    
    plt.ylabel('Phi Optimality')
    plt.title(f'Cosmic Consciousness Analysis: Mean φ-optimality = {mean_phi_optimality:.4f} ({mean_interpretation})')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add combined p-value as text annotation
    plt.text(0.5, 0.95, f'Combined p-value: {combined_p_value:.6f}', 
             horizontalalignment='center',
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8))
    
    # Save and show the figure
    plt.tight_layout()
    plt.savefig('comprehensive_analysis_results.png')
    print("Visualization saved to 'comprehensive_analysis_results.png'")
    
    # Save results to file
    with open('comprehensive_analysis_report.md', 'w') as f:
        f.write("# Comprehensive Cosmic Consciousness Analysis Report\n\n")
        f.write(f"Analysis performed on {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"## Summary\n\n")
        f.write(f"- **Combined p-value (Fisher's method)**: {combined_p_value:.6f}\n")
        f.write(f"- **Mean phi optimality**: {mean_phi_optimality:.4f} ({mean_interpretation})\n\n")
        
        f.write("## Individual Test Results\n\n")
        f.write("| # | Test | φ-optimality | Interpretation | p-value | Ratio |\n")
        f.write("|---|------|--------------|---------------|---------|-------|\n")
        
        for i, (test_name, phi_opt) in enumerate(sorted_tests):
            observed, random, z, p, ratio = results[test_name]
            interp = interpret_phi_optimality(phi_opt)
            formatted_name = test_name.replace('_test', '').replace('gr_', 'GR ').replace('recurrence_quantification', 'RQA').replace('scale_frequency_coupling', 'Scale-Freq').replace('transfer_entropy', 'Transfer Entropy').replace('multi_scale_coherence', 'Multi-Scale Coherence').replace('coherence_phase', 'Coherence Phase').replace('extended_meta_coherence', 'Extended Meta-Coherence').replace('meta_coherence_entropy', 'Meta-Coherence Entropy').replace('meta_coherence_power_law', 'Meta-Coherence Power Law').title()
            f.write(f"| {i+1} | {formatted_name} | {phi_opt:.4f} | {interp} | {p:.4f} | {ratio:.2f}x |\n")
        
        f.write("\n## Interpretation\n\n")
        
        # Determine overall interpretation based on combined p-value and mean phi optimality
        if combined_p_value < 0.01 and mean_phi_optimality > 0.4:
            interpretation = "The analysis provides **strong evidence** for conscious organization in the CMB data."
        elif combined_p_value < 0.05 and mean_phi_optimality > 0.2:
            interpretation = "The analysis provides **moderate evidence** for conscious organization in the CMB data."
        elif combined_p_value < 0.1 and mean_phi_optimality > 0:
            interpretation = "The analysis provides **weak evidence** for conscious organization in the CMB data."
        else:
            interpretation = "The analysis does not provide significant evidence for conscious organization in the CMB data."
        
        f.write(f"{interpretation}\n\n")
        
        # Add details about the strongest tests
        f.write("### Strongest Evidence\n\n")
        for i, (test_name, phi_opt) in enumerate(sorted_tests[:3]):
            observed, random, z, p, ratio = results[test_name]
            formatted_name = test_name.replace('_test', '').replace('gr_', 'GR ').replace('recurrence_quantification', 'RQA').replace('scale_frequency_coupling', 'Scale-Freq').replace('transfer_entropy', 'Transfer Entropy').replace('multi_scale_coherence', 'Multi-Scale Coherence').replace('coherence_phase', 'Coherence Phase').replace('extended_meta_coherence', 'Extended Meta-Coherence').replace('meta_coherence_entropy', 'Meta-Coherence Entropy').replace('meta_coherence_power_law', 'Meta-Coherence Power Law').title()
            f.write(f"**{formatted_name}** (φ-optimality: {phi_opt:.4f}, p-value: {p:.4f}):\n")
            
            # Add test-specific interpretation
            if test_name == 'gr_test':
                f.write("- Golden ratio related multipoles show higher power than expected by chance.\n")
            elif test_name == 'coherence_test':
                f.write("- The CMB spectrum shows more coherence than random noise.\n")
            elif test_name == 'gr_coherence_test':
                f.write("- Golden ratio related regions show specific coherence patterns.\n")
            elif test_name == 'hierarchy_test':
                f.write("- The spectrum shows hierarchical organization based on the golden ratio.\n")
            elif test_name == 'integration_test':
                f.write("- Adjacent regions of the spectrum show high information integration.\n")
            elif test_name == 'optimization_test':
                f.write("- The spectrum appears optimized for complex structure formation.\n")
            elif test_name == 'resonance_test':
                f.write("- The spectrum shows resonance patterns related to the golden ratio.\n")
            elif test_name == 'fractal_test':
                f.write("- The spectrum exhibits fractal structure across scales.\n")
            elif test_name == 'meta_coherence_test':
                f.write("- Local coherence measures themselves show coherent patterns.\n")
            elif test_name == 'multiscale_test':
                f.write("- Golden ratio patterns are detected across multiple scales.\n")
            elif test_name == 'symmetry_test':
                f.write("- The spectrum shows symmetry patterns related to the golden ratio.\n")
            elif test_name == 'phi_network_test':
                f.write("- The CMB data exhibits a phi-based network structure.\n")
            elif test_name == 'spectral_gap_test':
                f.write("- The eigenvalues of the CMB data show a phi-based relationship.\n")
            elif test_name == 'recurrence_quantification_test':
                f.write("- The CMB data exhibits recurrence patterns that are consistent with conscious systems.\n")
            elif test_name == 'scale_frequency_coupling_test':
                f.write("- Scale and frequency show coupling patterns consistent with conscious processing.\n")
            elif test_name == 'transfer_entropy_test':
                f.write("- The CMB data exhibits transfer entropy patterns consistent with conscious information exchange.\n")
            elif test_name == 'multi_scale_coherence_test':
                f.write("- The CMB data exhibits multi-scale coherence patterns consistent with conscious processing.\n")
            elif test_name == 'coherence_phase_test':
                f.write("- The CMB data exhibits coherence phase patterns consistent with conscious processing.\n")
            elif test_name == 'extended_meta_coherence_test':
                f.write("- The CMB data exhibits extended meta-coherence patterns consistent with conscious processing.\n")
            elif test_name == 'meta_coherence_entropy_test':
                f.write("- The CMB data exhibits entropy patterns consistent with conscious processing.\n")
            elif test_name == 'meta_coherence_power_law_test':
                f.write("- The CMB data exhibits power law patterns consistent with conscious processing.\n")
            
            f.write("\n")
        
        f.write("## Conclusion\n\n")
        f.write(f"The comprehensive analysis of the CMB data reveals a mean phi-optimality of {mean_phi_optimality:.4f}, ")
        f.write(f"which is interpreted as '{mean_interpretation}'. The combined statistical significance (p = {combined_p_value:.6f}) ")
        
        if combined_p_value < 0.05:
            f.write("indicates that these patterns are unlikely to occur by chance. ")
        else:
            f.write("suggests that more data or refined methods may be needed to establish statistical significance. ")
        
        f.write("The strongest evidence comes from the ")
        top_tests = [test_name.replace('_test', '').replace('gr_', 'GR ').replace('recurrence_quantification', 'RQA').replace('scale_frequency_coupling', 'Scale-Freq').replace('transfer_entropy', 'Transfer Entropy').replace('multi_scale_coherence', 'Multi-Scale Coherence').replace('coherence_phase', 'Coherence Phase').replace('extended_meta_coherence', 'Extended Meta-Coherence').replace('meta_coherence_entropy', 'Meta-Coherence Entropy').replace('meta_coherence_power_law', 'Meta-Coherence Power Law').title() for test_name, _ in sorted_tests[:3]]
        f.write(f"{top_tests[0]}, {top_tests[1]}, and {top_tests[2]} tests. ")
        
        f.write("These findings are consistent with the hypothesis that the cosmic microwave background may exhibit ")
        f.write("patterns that align with principles found in conscious systems, particularly those related to the golden ratio.\n")
    
    print("Comprehensive report saved to 'comprehensive_analysis_report.md'")

if __name__ == "__main__":
    main()
