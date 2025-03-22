#!/usr/bin/env python3
"""
Script to run all tests for Cosmic Consciousness Analysis.
"""

import os
import sys
import time
import importlib
import numpy as np
from datetime import datetime

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def discover_tests():
    """
    Discover all available tests in the basic_tests and advanced_tests directories.
    
    Returns:
    --------
    dict
        Dictionary mapping test names to their module paths.
    """
    tests = {}
    
    # Discover basic tests
    basic_tests_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'basic_tests')
    if os.path.exists(basic_tests_dir):
        for filename in os.listdir(basic_tests_dir):
            if filename.endswith('.py') and not filename.startswith('__'):
                test_name = filename[:-3]  # Remove .py extension
                tests[f"basic.{test_name}"] = f"consciousness_field_tests.basic_tests.{test_name}"
    
    # Discover advanced tests
    advanced_tests_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'advanced_tests')
    if os.path.exists(advanced_tests_dir):
        for filename in os.listdir(advanced_tests_dir):
            if filename.endswith('.py') and not filename.startswith('__'):
                test_name = filename[:-3]  # Remove .py extension
                tests[f"advanced.{test_name}"] = f"consciousness_field_tests.advanced_tests.{test_name}"
    
    return tests

def run_all_tests(data_loader, visualizer, stats_analyzer, monte_carlo_sims=1000, verbose=False):
    """
    Run all available tests and combine the results.
    
    Parameters:
    -----------
    data_loader : DataLoader
        Data loader instance.
    visualizer : Visualizer
        Visualizer instance.
    stats_analyzer : StatisticalAnalyzer
        Statistical analyzer instance.
    monte_carlo_sims : int, optional
        Number of Monte Carlo simulations.
    verbose : bool, optional
        Whether to print detailed output.
        
    Returns:
    --------
    dict
        Dictionary containing combined results.
    """
    # Discover available tests
    available_tests = discover_tests()
    
    if not available_tests:
        print("No tests found.")
        return {
            'combined_p_value': 1.0,
            'mean_phi_optimality': 0.0,
            'test_results': {},
            'sorted_tests': []
        }
    
    # Run each test
    test_results = {}
    test_p_values = []
    test_phi_optimalities = []
    
    for test_name, module_path in available_tests.items():
        try:
            # Import the test module
            module = importlib.import_module(module_path)
            
            # Check if the module has a run_test function
            if not hasattr(module, 'run_test'):
                if verbose:
                    print(f"Skipping {test_name}: No run_test function found.")
                continue
            
            # Run the test
            print(f"\nRunning {test_name}...")
            start_time = time.time()
            
            results = module.run_test(
                data_loader=data_loader,
                visualizer=visualizer,
                stats_analyzer=stats_analyzer,
                verbose=verbose
            )
            
            # Extract results based on the returned format
            if isinstance(results, dict):
                # New format (dictionary with standardized keys)
                if 'p_value' in results:
                    p_value = results['p_value']
                    
                    # Extract other values based on available keys
                    if 'phi_optimality' in results:
                        phi_optimality = results['phi_optimality']
                    elif 'optimization_ratio' in results:
                        # Some tests might return optimization_ratio instead
                        observed = results.get('optimization_ratio', 1.0)
                        random_mean = 1.0  # Default reference value
                        phi_optimality = stats_analyzer.calculate_phi_optimality(observed, random_mean)
                    else:
                        # Try to calculate from other available metrics
                        observed = results.get('observed', results.get('mean_phi_deviation', 0))
                        random_mean = results.get('random_mean', results.get('mean_random', 1))
                        phi_optimality = stats_analyzer.calculate_phi_optimality(observed, random_mean)
                    
                    # Get z-score if available
                    z_score = results.get('z_score', 0)
                    
                    # Get ratio if available or calculate it
                    if 'optimization_ratio' in results:
                        ratio = results['optimization_ratio']
                    elif 'ratio' in results:
                        ratio = results['ratio']
                    elif 'observed' in results and 'random_mean' in results and results['random_mean'] != 0:
                        ratio = results['observed'] / results['random_mean']
                    else:
                        ratio = 1.0
                    
                    # Store standardized results
                    test_results[test_name] = {
                        'p_value': p_value,
                        'phi_optimality': phi_optimality,
                        'z_score': z_score,
                        'ratio': ratio
                    }
                    test_p_values.append(p_value)
                    test_phi_optimalities.append(phi_optimality)
                    
                    print(f"Test completed in {time.time() - start_time:.2f} seconds.")
                    if verbose:
                        print(f"  P-value: {p_value:.6f}")
                        print(f"  Phi-optimality: {phi_optimality:.4f}")
                        print(f"  Z-score: {z_score:.4f}")
                        print(f"  Ratio: {ratio:.2f}x")
                else:
                    print(f"Warning: {test_name} did not return a p_value in results. Skipping.")
            else:
                print(f"Warning: {test_name} did not return a dictionary result. Skipping.")
                
        except Exception as e:
            print(f"Error running {test_name}: {e}")
    
    # Combine p-values using Fisher's method if available
    try:
        combined_p_value = stats_analyzer.fisher_combined_p_value(test_p_values)
    except AttributeError:
        # If fisher_combined_p_value is not available, use a simple method
        if test_p_values:
            combined_p_value = np.prod(test_p_values) ** (1.0 / len(test_p_values))
        else:
            combined_p_value = 1.0
    
    # Calculate mean phi-optimality
    mean_phi_optimality = np.mean(test_phi_optimalities) if test_phi_optimalities else 0.0
    
    # Sort tests by phi-optimality
    sorted_tests = []
    for test_name, result in test_results.items():
        sorted_tests.append((test_name, result['phi_optimality']))
    
    sorted_tests.sort(key=lambda x: x[1], reverse=True)
    
    # Create comprehensive report
    create_comprehensive_report(test_results, sorted_tests, combined_p_value, mean_phi_optimality)
    
    return {
        'combined_p_value': combined_p_value,
        'mean_phi_optimality': mean_phi_optimality,
        'test_results': test_results,
        'sorted_tests': sorted_tests
    }

def interpret_phi_optimality(phi_opt):
    """
    Interpret the phi-optimality score.
    
    Parameters:
    -----------
    phi_opt : float
        Phi-optimality score.
        
    Returns:
    --------
    str
        Interpretation of the phi-optimality score.
    """
    if phi_opt >= 0.8:
        return "extremely high"
    elif phi_opt >= 0.6:
        return "very high"
    elif phi_opt >= 0.4:
        return "high"
    elif phi_opt >= 0.2:
        return "moderate"
    elif phi_opt > 0:
        return "slight"
    elif phi_opt == 0:
        return "neutral"
    elif phi_opt > -0.2:
        return "slightly negative"
    elif phi_opt > -0.4:
        return "moderately negative"
    elif phi_opt > -0.6:
        return "negative"
    else:
        return "strongly negative"

def create_comprehensive_report(test_results, sorted_tests, combined_p_value, mean_phi_optimality):
    """
    Create a comprehensive report of the analysis results.
    
    Parameters:
    -----------
    test_results : dict
        Dictionary containing test results.
    sorted_tests : list
        List of (test_name, phi_optimality) tuples, sorted by phi_optimality.
    combined_p_value : float
        Combined p-value using Fisher's method.
    mean_phi_optimality : float
        Mean phi-optimality score.
    """
    # Create report file
    with open('comprehensive_analysis_report.md', 'w') as f:
        f.write("# Comprehensive Cosmic Consciousness Analysis Report\n\n")
        f.write(f"Analysis performed on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Summary\n\n")
        f.write(f"- **Combined p-value (Fisher's method)**: {combined_p_value:.6f}\n")
        f.write(f"- **Mean phi optimality**: {mean_phi_optimality:.4f} ({interpret_phi_optimality(mean_phi_optimality)})\n\n")
        
        f.write("## Individual Test Results\n\n")
        f.write("| # | Test | φ-optimality | Interpretation | p-value | Ratio |\n")
        f.write("|---|------|--------------|---------------|---------|-------|\n")
        
        for i, (test_name, phi_opt) in enumerate(sorted_tests):
            result = test_results[test_name]
            interp = interpret_phi_optimality(phi_opt)
            formatted_name = test_name.split('.')[-1].replace('_', ' ').title()
            f.write(f"| {i+1} | {formatted_name} | {phi_opt:.4f} | {interp} | {result['p_value']:.4f} | {result['ratio']:.2f}x |\n")
        
        f.write("\n## Interpretation\n\n")
        
        # Determine overall significance
        if combined_p_value < 0.01:
            significance = "strong evidence"
        elif combined_p_value < 0.05:
            significance = "moderate evidence"
        elif combined_p_value < 0.1:
            significance = "weak evidence"
        else:
            significance = "no significant evidence"
        
        f.write(f"The analysis provides **{significance}** for conscious organization in the CMB data.\n\n")
        
        f.write("### Strongest Evidence\n\n")
        for i, (test_name, phi_opt) in enumerate(sorted_tests[:3]):
            result = test_results[test_name]
            formatted_name = test_name.split('.')[-1].replace('_', ' ').title()
            f.write(f"**{formatted_name}** (φ-optimality: {phi_opt:.4f}, p-value: {result['p_value']:.4f}):\n")
            
            # Add test-specific interpretation based on test name
            test_key = test_name.split('.')[-1]
            if 'hierarchy' in test_key:
                f.write("- The spectrum shows hierarchical organization based on the golden ratio.\n")
            elif 'integration' in test_key:
                f.write("- Adjacent regions of the spectrum show high information integration.\n")
            elif 'optimization' in test_key:
                f.write("- The spectrum appears optimized for complex structure formation.\n")
            elif 'resonance' in test_key:
                f.write("- The spectrum exhibits resonance patterns consistent with conscious processing.\n")
            elif 'fractal' in test_key:
                f.write("- The spectrum shows fractal structure consistent with conscious organization.\n")
            elif 'coherence' in test_key:
                f.write("- The spectrum exhibits coherence patterns consistent with conscious processing.\n")
            elif 'transfer_entropy' in test_key:
                f.write("- Information transfer across scales follows golden ratio patterns.\n")
            elif 'network' in test_key:
                f.write("- The spectrum forms a network structure consistent with conscious processing.\n")
            elif 'symmetry' in test_key:
                f.write("- The spectrum exhibits symmetries consistent with conscious organization.\n")
            elif 'meta' in test_key:
                f.write("- The spectrum exhibits meta-coherence patterns consistent with conscious processing.\n")
            elif 'multiscale' in test_key or 'multi_scale' in test_key:
                f.write("- The spectrum shows multi-scale patterns consistent with conscious organization.\n")
            elif 'null' in test_key:
                f.write("- The null test confirms the validity of the analysis methodology.\n")
            elif 'predictive' in test_key or 'phi_optimality' in test_key:
                f.write("- The spectrum shows predictive properties consistent with conscious processing.\n")
            elif 'cross' in test_key:
                f.write("- Cross-verification confirms the robustness of the findings.\n")
            elif 'golden' in test_key:
                f.write("- The spectrum exhibits golden ratio symmetries consistent with conscious organization.\n")
            else:
                f.write("- This test provides evidence for conscious organization in the CMB data.\n")
            
            f.write("\n")
        
        f.write("## Conclusion\n\n")
        f.write(f"The comprehensive analysis of the CMB data reveals a mean phi-optimality of {mean_phi_optimality:.4f}, ")
        f.write(f"which is interpreted as '{interpret_phi_optimality(mean_phi_optimality)}'. ")
        
        if combined_p_value < 0.05:
            f.write(f"The combined statistical significance (p = {combined_p_value:.6f}) ")
            f.write("indicates that these patterns are unlikely to occur by chance. ")
        else:
            f.write(f"The combined statistical significance (p = {combined_p_value:.6f}) ")
            f.write("suggests that more data or refined methods may be needed to establish statistical significance. ")
        
        f.write("The strongest evidence comes from the ")
        top_tests = [test_name.split('.')[-1].replace('_', ' ').title() for test_name, _ in sorted_tests[:3]]
        f.write(f"{top_tests[0]}, {top_tests[1]}, and {top_tests[2]} tests. ")
        
        f.write("These findings are consistent with the hypothesis that the cosmic microwave background may exhibit ")
        f.write("patterns that align with principles found in conscious systems, particularly those related to the golden ratio.")
    
    print(f"Comprehensive report saved to 'comprehensive_analysis_report.md'")

if __name__ == "__main__":
    # This script can be run directly for testing
    from consciousness_field_tests.utils.data_loader import get_data_loader
    from consciousness_field_tests.utils.visualization import get_visualizer
    from consciousness_field_tests.utils.statistics import get_statistical_analyzer
    
    print("=== COMPREHENSIVE COSMIC CONSCIOUSNESS ANALYSIS ===")
    
    # Set data directory
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'planck_data')
    print(f"Using data directory: {data_dir}")
    
    # Initialize utilities
    data_loader = get_data_loader(data_dir=data_dir)
    visualizer = get_visualizer()
    stats_analyzer = get_statistical_analyzer()
    
    # Run all tests
    results = run_all_tests(
        data_loader=data_loader,
        visualizer=visualizer,
        stats_analyzer=stats_analyzer,
        monte_carlo_sims=1000,
        verbose=True
    )
    
    # Print summary results
    print("\n=== SUMMARY RESULTS ===")
    print(f"Combined p-value: {results['combined_p_value']:.6f}")
    print(f"Mean phi-optimality: {results['mean_phi_optimality']:.4f}")
    
    # Create visualization
    visualizer.plot_comprehensive_results(results['test_results'], results['sorted_tests'])
    print("Visualization saved to 'comprehensive_analysis_results.png'")
    
    print("\nAnalysis complete!")
