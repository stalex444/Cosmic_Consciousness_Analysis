#!/usr/bin/env python3
"""
Script to analyze and summarize the results of cosmic consciousness analysis.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import argparse

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Analyze and summarize results")
    
    parser.add_argument("--data-dir", type=str, default="planck_data",
                        help="Directory containing results")
    parser.add_argument("--output", type=str, default="analysis_summary.pdf",
                        help="Output file for the summary report")
    
    return parser.parse_args()

def load_results(data_dir):
    """Load all result files from the data directory."""
    results = {}
    
    # Try to load golden ratio analysis results
    synthetic_results_path = os.path.join(data_dir, "synthetic_analysis_results.json")
    if os.path.exists(synthetic_results_path):
        with open(synthetic_results_path, 'r') as f:
            results["golden_ratio"] = json.load(f)
    
    # Try to load constant analysis results
    constant_results_path = os.path.join(data_dir, "constant_analysis_results.json")
    if os.path.exists(constant_results_path):
        with open(constant_results_path, 'r') as f:
            results["constants"] = json.load(f)
    
    # Try to load temporal evolution results
    temporal_results_path = os.path.join(data_dir, "temporal_evolution_results.json")
    if os.path.exists(temporal_results_path):
        with open(temporal_results_path, 'r') as f:
            results["temporal"] = json.load(f)
    
    # Try to load wavelet analysis results
    wavelet_results_path = os.path.join(data_dir, "wavelet_analysis_results.json")
    if os.path.exists(wavelet_results_path):
        with open(wavelet_results_path, 'r') as f:
            results["wavelet"] = json.load(f)
    
    return results

def create_summary_report(results, output_file):
    """Create a comprehensive summary report of all analyses."""
    plt.figure(figsize=(12, 16))
    gs = GridSpec(4, 2)
    
    # Title
    plt.suptitle("Cosmic Consciousness Analysis Summary", fontsize=16, y=0.98)
    
    # Golden Ratio Analysis
    if "golden_ratio" in results:
        ax1 = plt.subplot(gs[0, 0])
        gr_results = results["golden_ratio"]
        
        if "phi_optimality" in gr_results:
            ax1.bar(range(len(gr_results["phi_optimality"])), gr_results["phi_optimality"])
            ax1.axhline(y=gr_results["average_phi_optimality"], color='r', linestyle='-',
                       label=f"Avg: {gr_results['average_phi_optimality']:.3f}")
            ax1.set_xlabel('Ratio Index')
            ax1.set_ylabel('Phi-Optimality')
            ax1.set_title('Golden Ratio Analysis')
            ax1.legend()
    
    # Constants Comparison
    if "constants" in results:
        ax2 = plt.subplot(gs[0, 1])
        const_results = results["constants"]
        
        if "overall_results" in const_results:
            const_names = list(const_results["overall_results"].keys())
            means = [const_results["overall_results"][const]["mean"] for const in const_names]
            
            bars = ax2.bar(const_names, means)
            for bar, mean in zip(bars, means):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{mean:.3f}', ha='center', va='bottom')
            
            ax2.set_xlabel('Mathematical Constant')
            ax2.set_ylabel('Average Optimality')
            ax2.set_title('Constants Comparison')
    
    # Temporal Evolution
    if "temporal" in results:
        ax3 = plt.subplot(gs[1, :])
        temporal_results = results["temporal"]
        
        if "time_points" in temporal_results and "phi_optimality" in temporal_results:
            time_points = temporal_results["time_points"]
            phi_optimality = temporal_results["phi_optimality"]
            
            ax3.plot(time_points, phi_optimality)
            
            # Add a trend line
            if len(time_points) > 1:
                z = np.polyfit(time_points, phi_optimality, 1)
                p = np.poly1d(z)
                ax3.plot(time_points, p(time_points), "r--", 
                        label=f"Trend: y={z[0]:.2e}x+{z[1]:.2f}")
            
            ax3.set_xlabel('Time Point')
            ax3.set_ylabel('Phi-Optimality')
            ax3.set_title('Temporal Evolution of Phi-Optimality')
            ax3.legend()
    
    # Wavelet Analysis
    if "wavelet" in results:
        ax4 = plt.subplot(gs[2, :])
        wavelet_results = results["wavelet"]
        
        if "phi_optimality" in wavelet_results:
            ax4.bar(range(len(wavelet_results["phi_optimality"])), wavelet_results["phi_optimality"])
            ax4.axhline(y=wavelet_results["average_phi_optimality"], color='r', linestyle='-',
                       label=f"Avg: {wavelet_results['average_phi_optimality']:.3f}")
            ax4.set_xlabel('Decomposition Level')
            ax4.set_ylabel('Phi-Optimality')
            ax4.set_title(f'Wavelet Analysis ({wavelet_results["wavelet"]})')
            ax4.legend()
    
    # Summary Text
    ax5 = plt.subplot(gs[3, :])
    ax5.axis('off')
    
    summary_text = "Analysis Summary:\n\n"
    
    if "golden_ratio" in results and "average_phi_optimality" in results["golden_ratio"]:
        summary_text += f"- Golden Ratio Analysis: Average Phi-Optimality = {results['golden_ratio']['average_phi_optimality']:.4f}\n"
    
    if "constants" in results and "overall_results" in results["constants"]:
        summary_text += "- Constants Comparison:\n"
        for const, result in results["constants"]["overall_results"].items():
            summary_text += f"  * {const}: mean={result['mean']:.4f}, median={result['median']:.4f}\n"
    
    if "wavelet" in results and "average_phi_optimality" in results["wavelet"]:
        summary_text += f"- Wavelet Analysis: Average Phi-Optimality = {results['wavelet']['average_phi_optimality']:.4f}\n"
    
    # Add interpretation
    summary_text += "\nInterpretation:\n"
    
    # Check if golden ratio has highest optimality among constants
    if "constants" in results and "overall_results" in results["constants"]:
        const_means = {const: results["constants"]["overall_results"][const]["mean"] 
                      for const in results["constants"]["overall_results"]}
        max_const = max(const_means, key=const_means.get)
        
        if max_const == "phi":
            summary_text += "- The golden ratio (phi) shows the strongest alignment with the observed patterns,\n  suggesting a potential connection to consciousness-related phenomena.\n"
        else:
            summary_text += f"- The constant {max_const} shows the strongest alignment with the observed patterns,\n  which may indicate different underlying mechanisms than expected.\n"
    
    # Check for temporal trends
    if "temporal" in results and "time_points" in results["temporal"] and "phi_optimality" in results["temporal"]:
        time_points = results["temporal"]["time_points"]
        phi_optimality = results["temporal"]["phi_optimality"]
        
        if len(time_points) > 1:
            z = np.polyfit(time_points, phi_optimality, 1)
            if z[0] > 0.001:
                summary_text += "- There is an increasing trend in phi-optimality over time,\n  suggesting an evolving or strengthening pattern.\n"
            elif z[0] < -0.001:
                summary_text += "- There is a decreasing trend in phi-optimality over time,\n  suggesting a weakening or dissolving pattern.\n"
            else:
                summary_text += "- Phi-optimality remains relatively stable over time,\n  suggesting a consistent underlying pattern.\n"
    
    ax5.text(0, 1, summary_text, va='top', fontsize=10, linespacing=1.5)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_file)
    print(f"Summary report saved to {output_file}")

def main():
    """Main function to analyze results."""
    args = parse_arguments()
    
    print(f"Loading results from {args.data_dir}...")
    results = load_results(args.data_dir)
    
    if not results:
        print("No result files found. Please run the analysis first.")
        return
    
    print("Creating summary report...")
    create_summary_report(results, os.path.join(args.data_dir, args.output))
    
    print("Analysis complete!")

if __name__ == "__main__":
    main()
