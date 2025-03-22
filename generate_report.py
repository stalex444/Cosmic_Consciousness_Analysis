#!/usr/bin/env python3
"""
Generate a comprehensive report from Cosmic Consciousness Analysis results.
This script creates a text report that can be easily copied and pasted into Claude or other text platforms.
"""

import os
import json
import numpy as np
import argparse
from datetime import datetime

def load_results(results_dir):
    """Load results from JSON files in the results directory."""
    results = {}
    
    # Try to load comprehensive results first
    comprehensive_file = os.path.join(results_dir, "comprehensive_results.json")
    if os.path.exists(comprehensive_file):
        with open(comprehensive_file, 'r') as f:
            results = json.load(f)
    else:
        # Load individual test results
        for file in os.listdir(results_dir):
            if file.endswith("_result.json"):
                test_name = file.replace("_result.json", "")
                with open(os.path.join(results_dir, file), 'r') as f:
                    results[test_name] = json.load(f)["result"]
    
    return results

def format_p_value(p_value):
    """Format p-value for reporting."""
    if p_value < 0.001:
        return "p < 0.001"
    elif p_value < 0.01:
        return f"p = {p_value:.4f}"
    else:
        return f"p = {p_value:.5f}"

def interpret_significance(p_value):
    """Interpret the significance of a p-value."""
    if p_value < 0.001:
        return "highly significant"
    elif p_value < 0.01:
        return "very significant"
    elif p_value < 0.05:
        return "significant"
    elif p_value < 0.1:
        return "marginally significant"
    else:
        return "not significant"

def interpret_effect_size(ratio):
    """Interpret the effect size based on ratio."""
    if ratio > 10:
        return "extremely strong"
    elif ratio > 5:
        return "very strong"
    elif ratio > 2:
        return "strong"
    elif ratio > 1.5:
        return "moderate"
    elif ratio > 1.1:
        return "weak"
    else:
        return "negligible"

def calculate_phi_optimality(value, target=0.618):
    """Calculate phi-optimality for a value compared to golden ratio."""
    # Ensure the value is between 0 and 1
    if value > 1:
        value = 1/value
    
    # Calculate phi-optimality (bounded between -1 and 1)
    phi_opt = 1 - min(2, abs(value - target) / target)
    return phi_opt

def generate_report(results, output_file=None):
    """Generate a comprehensive report from the results."""
    report = []
    
    # Header
    report.append("# Cosmic Consciousness Analysis Report")
    report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Summary of findings
    report.append("## Summary of Findings")
    
    if "combined_zscore" in results and "combined_pvalue" in results:
        combined_z = results["combined_zscore"]
        combined_p = results["combined_pvalue"]
        
        significance = interpret_significance(combined_p)
        
        report.append(f"The combined analysis of all tests yielded a z-score of {combined_z:.2f}σ ({format_p_value(combined_p)}), which is {significance}.")
        
        if combined_p < 0.05:
            report.append("\nThe analysis provides evidence for non-random organization in the Cosmic Microwave Background (CMB) data that is consistent with patterns found in conscious systems.")
        else:
            report.append("\nThe analysis does not provide strong evidence for non-random organization in the Cosmic Microwave Background (CMB) data that would be consistent with patterns found in conscious systems.")
    
    # Count significant tests
    significant_tests = 0
    total_tests = 0
    
    for key in results:
        if key.endswith("_pvalue") and not key.startswith("combined"):
            total_tests += 1
            if results[key] < 0.05:
                significant_tests += 1
    
    if total_tests > 0:
        report.append(f"\n{significant_tests} out of {total_tests} tests showed statistically significant results (p < 0.05).")
    
    # Detailed results
    report.append("\n## Detailed Results")
    
    # Golden Ratio Significance
    if "gr_test" in results and "gr_pvalue" in results:
        report.append("\n### 1. Golden Ratio Significance Test")
        gr_ratio = results["gr_test"]
        gr_p = results["gr_pvalue"]
        gr_z = results.get("gr_zscore", 0)
        
        effect = interpret_effect_size(gr_ratio)
        significance = interpret_significance(gr_p)
        
        report.append(f"- Result: {gr_ratio:.2f}x ({effect} effect)")
        report.append(f"- Statistical significance: {gr_z:.2f}σ ({format_p_value(gr_p)}, {significance})")
        report.append(f"- Interpretation: Power at golden ratio-related multipoles is {gr_ratio:.2f} times stronger than at random multipoles.")
    
    # Coherence
    if "coherence_test" in results and "coherence_pvalue" in results:
        report.append("\n### 2. Coherence Test")
        coh_ratio = results["coherence_test"]
        coh_p = results["coherence_pvalue"]
        coh_z = results.get("coherence_zscore", 0)
        
        effect = interpret_effect_size(coh_ratio)
        significance = interpret_significance(coh_p)
        
        report.append(f"- Result: {coh_ratio:.4f} ({effect} effect)")
        report.append(f"- Statistical significance: {coh_z:.2f}σ ({format_p_value(coh_p)}, {significance})")
        report.append(f"- Interpretation: The CMB spectrum shows {effect} coherence compared to random spectra.")
    
    # GR-Specific Coherence
    if "gr_coherence_test" in results and "gr_coherence_pvalue" in results:
        report.append("\n### 3. GR-Specific Coherence Test")
        gr_coh_ratio = results["gr_coherence_test"]
        gr_coh_p = results["gr_coherence_pvalue"]
        gr_coh_z = results.get("gr_coherence_zscore", 0)
        
        effect = interpret_effect_size(gr_coh_ratio)
        significance = interpret_significance(gr_coh_p)
        
        report.append(f"- Result: {gr_coh_ratio:.4f}x ({effect} effect)")
        report.append(f"- Statistical significance: {gr_coh_z:.2f}σ ({format_p_value(gr_coh_p)}, {significance})")
        report.append(f"- Interpretation: Windows centered on golden ratio multipoles show {gr_coh_ratio:.2f} times more coherence than random windows.")
    
    # Hierarchical Organization
    if "hierarchy_test" in results:
        report.append("\n### 4. Hierarchical Organization Test")
        hierarchy_ratio = results["hierarchy_test"]
        
        effect = interpret_effect_size(hierarchy_ratio)
        
        report.append(f"- Result: {hierarchy_ratio:.2f}x ({effect} effect)")
        report.append(f"- Interpretation: The CMB spectrum shows {effect} hierarchical organization with golden ratio-related multipoles.")
    
    # Information Integration
    if "information_test" in results and "information_pvalue" in results:
        report.append("\n### 5. Information Integration Test")
        info_ratio = results["information_test"]
        info_p = results["information_pvalue"]
        info_z = results.get("information_zscore", 0)
        
        effect = interpret_effect_size(info_ratio)
        significance = interpret_significance(info_p)
        
        report.append(f"- Result: {info_ratio:.2f}x ({effect} effect)")
        report.append(f"- Statistical significance: {info_z:.2f}σ ({format_p_value(info_p)}, {significance})")
        report.append(f"- Interpretation: The CMB spectrum shows {effect} information integration compared to random spectra.")
    
    # Optimization
    if "optimization_test" in results and "optimization_pvalue" in results:
        report.append("\n### 6. Optimization Test")
        opt_ratio = results["optimization_test"]
        opt_p = results["optimization_pvalue"]
        opt_z = results.get("optimization_zscore", 0)
        
        effect = interpret_effect_size(opt_ratio)
        significance = interpret_significance(opt_p)
        
        report.append(f"- Result: {opt_ratio:.2f}x ({effect} effect)")
        report.append(f"- Statistical significance: {opt_z:.2f}σ ({format_p_value(opt_p)}, {significance})")
        report.append(f"- Interpretation: The CMB spectrum shows {effect} optimization toward golden ratio patterns compared to random spectra.")
    
    # Resonance
    if "resonance_test" in results and "resonance_pvalue" in results:
        report.append("\n### 7. Resonance Test")
        res_ratio = results["resonance_test"]
        res_p = results["resonance_pvalue"]
        res_z = results.get("resonance_zscore", 0)
        
        effect = interpret_effect_size(res_ratio)
        significance = interpret_significance(res_p)
        
        report.append(f"- Result: {res_ratio:.2f}x ({effect} effect)")
        report.append(f"- Statistical significance: {res_z:.2f}σ ({format_p_value(res_p)}, {significance})")
        report.append(f"- Interpretation: The CMB spectrum shows {effect} resonance at golden ratio-related frequencies compared to random frequencies.")
    
    # Fractal Structure
    if "fractal_test" in results and "fractal_pvalue" in results:
        report.append("\n### 8. Fractal Structure Test")
        fractal_ratio = results["fractal_test"]
        fractal_p = results["fractal_pvalue"]
        fractal_z = results.get("fractal_zscore", 0)
        
        effect = interpret_effect_size(fractal_ratio)
        significance = interpret_significance(fractal_p)
        
        report.append(f"- Result: {fractal_ratio:.4f}x ({effect} effect)")
        report.append(f"- Statistical significance: {fractal_z:.2f}σ ({format_p_value(fractal_p)}, {significance})")
        report.append(f"- Interpretation: The CMB spectrum shows {effect} fractal structure compared to random spectra.")
    
    # Meta-Coherence
    if "meta_test" in results and "meta_pvalue" in results:
        report.append("\n### 9. Meta-Coherence Test")
        meta_ratio = results["meta_test"]
        meta_p = results["meta_pvalue"]
        meta_z = results.get("meta_zscore", 0)
        
        effect = interpret_effect_size(meta_ratio)
        significance = interpret_significance(meta_p)
        
        report.append(f"- Result: {meta_ratio:.2f}x ({effect} effect)")
        report.append(f"- Statistical significance: {meta_z:.2f}σ ({format_p_value(meta_p)}, {significance})")
        report.append(f"- Interpretation: The CMB spectrum shows {effect} meta-coherence compared to random spectra.")
    
    # Multi-Scale Patterns
    if "multiscale_test" in results and "multiscale_pvalue" in results:
        report.append("\n### 10. Multi-Scale Patterns Test")
        multiscale_ratio = results["multiscale_test"]
        multiscale_p = results["multiscale_pvalue"]
        multiscale_z = results.get("multiscale_zscore", 0)
        
        effect = interpret_effect_size(multiscale_ratio)
        significance = interpret_significance(multiscale_p)
        
        report.append(f"- Result: {multiscale_ratio:.2f}x ({effect} effect)")
        report.append(f"- Statistical significance: {multiscale_z:.2f}σ ({format_p_value(multiscale_p)}, {significance})")
        report.append(f"- Interpretation: The CMB spectrum shows {effect} multi-scale patterns compared to random spectra.")
    
    # Peak Frequency Analysis
    if "peak_frequency_test" in results:
        report.append("\n### 11. Peak Frequency Analysis")
        peak_phi = results["peak_frequency_test"]
        
        report.append(f"- Mean phi-optimality: {peak_phi:.4f}")
        report.append(f"- Interpretation: The average phi-optimality of peak frequency ratios is {peak_phi:.4f}, where 1.0 would indicate perfect alignment with the golden ratio.")
    
    # Cross-Scale Correlations
    if "cross_scale_test" in results:
        report.append("\n### 12. Cross-Scale Correlations Test")
        if isinstance(results["cross_scale_test"], (list, tuple)) and len(results["cross_scale_test"]) >= 4:
            cross_scale_ratio = results["cross_scale_test"][0] / results["cross_scale_test"][1]
            cross_scale_z = results["cross_scale_test"][2]
            cross_scale_p = results["cross_scale_test"][3]
            
            effect = interpret_effect_size(cross_scale_ratio)
            significance = interpret_significance(cross_scale_p)
            
            report.append(f"- Result: {cross_scale_ratio:.2f}x ({effect} effect)")
            report.append(f"- Statistical significance: {cross_scale_z:.2f}σ ({format_p_value(cross_scale_p)}, {significance})")
            report.append(f"- Interpretation: Scales separated by powers of the golden ratio show {effect} correlation compared to random scale relationships.")
        else:
            report.append(f"- Result: {results['cross_scale_test']}")
    
    # Phi-Optimality Analysis
    report.append("\n## Phi-Optimality Analysis")
    report.append("Phi-optimality measures how closely a value aligns with the golden ratio (φ ≈ 0.618), with 1.0 indicating perfect alignment.")
    
    phi_optimalities = {}
    
    # Calculate phi-optimality for each test result
    if "gr_test" in results:
        phi_optimalities["Golden Ratio Significance"] = calculate_phi_optimality(results["gr_test"])
    
    if "coherence_test" in results:
        phi_optimalities["Coherence"] = calculate_phi_optimality(results["coherence_test"])
    
    if "gr_coherence_test" in results:
        phi_optimalities["GR-Specific Coherence"] = calculate_phi_optimality(results["gr_coherence_test"])
    
    if "hierarchy_test" in results:
        phi_optimalities["Hierarchical Organization"] = calculate_phi_optimality(results["hierarchy_test"])
    
    if "information_test" in results:
        phi_optimalities["Information Integration"] = calculate_phi_optimality(results["information_test"])
    
    if "optimization_test" in results:
        phi_optimalities["Optimization"] = calculate_phi_optimality(results["optimization_test"])
    
    if "resonance_test" in results:
        phi_optimalities["Resonance"] = calculate_phi_optimality(results["resonance_test"])
    
    if "fractal_test" in results:
        phi_optimalities["Fractal Structure"] = calculate_phi_optimality(results["fractal_test"])
    
    if "meta_test" in results:
        phi_optimalities["Meta-Coherence"] = calculate_phi_optimality(results["meta_test"])
    
    if "multiscale_test" in results:
        phi_optimalities["Multi-Scale Patterns"] = calculate_phi_optimality(results["multiscale_test"])
    
    if "peak_frequency_test" in results:
        phi_optimalities["Peak Frequency Analysis"] = results["peak_frequency_test"]
    
    if "cross_scale_test" in results:
        if isinstance(results["cross_scale_test"], (list, tuple)) and len(results["cross_scale_test"]) >= 4:
            cross_scale_ratio = results["cross_scale_test"][0] / results["cross_scale_test"][1]
            phi_optimalities["Cross-Scale Correlations"] = calculate_phi_optimality(cross_scale_ratio)
    
    # Sort by phi-optimality
    sorted_optimalities = sorted(phi_optimalities.items(), key=lambda x: x[1], reverse=True)
    
    for test, opt in sorted_optimalities:
        report.append(f"- {test}: {opt:.4f}")
    
    if phi_optimalities:
        mean_opt = np.mean(list(phi_optimalities.values()))
        report.append(f"\nMean phi-optimality across all tests: {mean_opt:.4f}")
    
    # Conclusion
    report.append("\n## Conclusion")
    
    if "combined_pvalue" in results:
        combined_p = results["combined_pvalue"]
        if combined_p < 0.01:
            report.append("The analysis provides strong evidence for non-random organization in the Cosmic Microwave Background (CMB) data that is consistent with patterns found in conscious systems. The golden ratio-related patterns, coherence, and resonance tests in particular show significant results that suggest a deeper organizing principle in the cosmic structure.")
        elif combined_p < 0.05:
            report.append("The analysis provides moderate evidence for non-random organization in the Cosmic Microwave Background (CMB) data that shows some similarities to patterns found in conscious systems. While not all tests show significant results, there are enough patterns to suggest that further investigation is warranted.")
        else:
            report.append("The analysis does not provide strong statistical evidence for non-random organization in the Cosmic Microwave Background (CMB) data that would be consistent with patterns found in conscious systems. However, some individual tests show interesting patterns that could be explored further with more refined methods or larger datasets.")
    
    # Join the report
    full_report = "\n".join(report)
    
    # Save to file if requested
    if output_file:
        with open(output_file, 'w') as f:
            f.write(full_report)
        print(f"Report saved to {output_file}")
    
    return full_report

def main():
    """Main function to parse arguments and generate report."""
    parser = argparse.ArgumentParser(description='Generate a report from Cosmic Consciousness Analysis results')
    parser.add_argument('--results-dir', type=str, default='analysis_results', help='Directory containing analysis results')
    parser.add_argument('--output-file', type=str, default='cosmic_consciousness_report.md', help='Output file for the report')
    
    args = parser.parse_args()
    
    # Check if results directory exists
    if not os.path.exists(args.results_dir):
        print(f"Error: Results directory '{args.results_dir}' not found.")
        return
    
    # Load results
    results = load_results(args.results_dir)
    
    # Generate report
    generate_report(results, args.output_file)
    
    print(f"Report generated successfully. You can now copy the contents of '{args.output_file}' and paste into Claude.")

if __name__ == "__main__":
    main()
