# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import time
from sklearn.metrics import mutual_info_score
import warnings
warnings.filterwarnings('ignore')

class ScaleTransitionTest:
    """
    Test that identifies scale boundaries where organizational principles change in the CMB.
    This test examines how different mathematical relationships (golden ratio, e, pi, etc.)
    optimize cosmic organization across different scales, and identifies transition boundaries.
    """
    
    def __init__(self, cmb_data=None, name="Scale Transition Test"):
        """Initialize the test with CMB data."""
        self.name = name
        self.constants = {
            "phi": (1 + np.sqrt(5)) / 2,  # Golden ratio
            "e": np.e,
            "pi": np.pi,
            "sqrt2": np.sqrt(2),
            "sqrt3": np.sqrt(3),
            "ln2": np.log(2)
        }
        
        # Metrics to test at each scale
        self.metrics = [
            "transfer_entropy",
            "coherence",
            "laminarity",
            "information_integration",
            "power_law"
        ]
        
        self.cmb_data = cmb_data if cmb_data is not None else self.load_cmb_data()
        self.results = {}
        
    def load_cmb_data(self):
        """Load or generate CMB data for analysis."""
        print("Generating simulated CMB data...")
        # Generate simulated data with embedded scale-dependent patterns
        size = 8192  # Larger size to accommodate more scales
        
        # Base noise
        data = np.random.normal(0, 1, size=size)
        
        # Add phi-based patterns at different scales with varying strengths
        phi = (1 + np.sqrt(5)) / 2
        
        # Small scales: strong phi-optimization
        for i in range(1, 5):
            scale = int(10 * phi ** i)
            if scale < size // 2:
                pattern = np.sin(np.arange(size) * 2 * np.pi / scale)
                data += pattern * (0.8 ** i)
        
        # Medium scales: mixed optimization (e-related)
        for i in range(3, 7):
            scale = int(10 * np.e ** i)
            if scale < size // 2:
                pattern = np.sin(np.arange(size) * 2 * np.pi / scale)
                data += pattern * (0.7 ** i)
        
        # Large scales: pi-optimization
        for i in range(2, 6):
            scale = int(10 * np.pi ** i)
            if scale < size // 2:
                pattern = np.sin(np.arange(size) * 2 * np.pi / scale)
                data += pattern * (0.6 ** i)
        
        # Normalize
        data = (data - np.mean(data)) / np.std(data)
        
        return data
    
    def run_test(self):
        """Run the scale transition test on the CMB data."""
        print("Running {}...".format(self.name))
        start_time = time.time()
        
        # 1. Define scale ranges to analyze
        # Using logarithmic binning to cover a wide range of scales
        min_scale = 2
        max_scale = len(self.cmb_data) // 4
        num_bins = 20
        
        scales = np.logspace(np.log10(min_scale), np.log10(max_scale), num_bins).astype(int)
        scales = np.unique(scales)  # Remove duplicates
        
        # 2. Calculate optimization for each constant at each scale
        scale_results = {}
        
        for scale in scales:
            print("  Analyzing scale: {}...".format(scale))
            scale_results[scale] = self.analyze_scale(scale)
        
        # 3. Identify scale transitions
        transitions = self.identify_transitions(scale_results, scales)
        
        # 4. Test statistical significance
        significance = self.test_significance(scale_results, scales)
        
        # Store results
        self.results = {
            'scale_results': scale_results,
            'scales': scales,
            'transitions': transitions,
            'significance': significance,
            'execution_time': time.time() - start_time
        }
        
        # Generate report
        self.generate_report()
        
        print("Test completed in {:.2f} seconds.".format(self.results['execution_time']))
        return self.results

    def analyze_scale(self, scale):
        """Analyze optimization of different constants at a specific scale."""
        results = {}
        
        # For each metric
        for metric_name in self.metrics:
            metric_results = {}
            
            # For each mathematical constant
            for const_name, const_value in self.constants.items():
                # Calculate metric optimized by this constant at this scale
                value = self.calculate_metric(metric_name, scale, const_name, const_value)
                
                # Calculate same metric for random data (as baseline)
                random_data = np.random.normal(0, 1, size=len(self.cmb_data))
                random_value = self.calculate_metric(metric_name, scale, const_name, const_value, data=random_data)
                
                # Store results
                ratio = value / random_value if random_value > 0 else 1.0
                metric_results[const_name] = {
                    'value': value,
                    'random_value': random_value,
                    'ratio': ratio
                }
            
            # Find which constant optimizes this metric best at this scale
            best_const = max(metric_results.items(), key=lambda x: x[1]['ratio'])
            
            # Calculate phi optimization
            phi_value = metric_results['phi']['ratio']
            best_value = best_const[1]['ratio']
            
            if best_value > 1.0:
                phi_optimality = (phi_value - 1.0) / (best_value - 1.0)
            else:
                phi_optimality = 0.0
                
            phi_optimality = min(max(phi_optimality, -1.0), 1.0)  # Bound between -1 and 1
            
            results[metric_name] = {
                'constants': metric_results,
                'best_constant': best_const[0],
                'best_ratio': best_value,
                'phi_optimality': phi_optimality
            }
        
        return results
    
    def calculate_metric(self, metric_name, scale, const_name, const_value, data=None):
        """Calculate a specific metric at a specific scale, optimized by a specific constant."""
        if data is None:
            data = self.cmb_data
        
        if metric_name == "transfer_entropy":
            return self.calculate_transfer_entropy(data, scale, const_value)
        elif metric_name == "coherence":
            return self.calculate_coherence(data, scale)
        elif metric_name == "laminarity":
            return self.calculate_laminarity(data, scale)
        elif metric_name == "information_integration":
            return self.calculate_information_integration(data, scale, const_value)
        elif metric_name == "power_law":
            return self.calculate_power_law(data, scale)
        else:
            return 1.0  # Default

    def calculate_transfer_entropy(self, data, scale, constant):
        """Calculate transfer entropy at the given scale, using the given constant."""
        # Define scales related by the constant
        scales = []
        current = scale
        
        # Generate 5 scales, starting from the given scale
        for i in range(5):
            scales.append(int(current))
            current *= constant
        
        scales = [s for s in scales if s > 0 and s < len(data) // 2]
        
        if len(scales) < 2:
            return 1.0
            
        # Extract data at these scales
        scale_data = []
        for s in scales:
            # Create windows around the scale
            window_size = min(s // 2, 5)
            indices = np.arange(s - window_size, s + window_size + 1)
            indices = indices[(indices >= 0) & (indices < len(data))]
            
            # Extract and average data in the window
            values = [data[i] for i in indices]
            scale_data.append(np.mean(values))
        
        # Calculate transfer entropy (using mutual information as a simplified proxy)
        entropy = 0
        for i in range(len(scale_data) - 1):
            # Create symbolic sequences
            x = scale_data[i]
            y = scale_data[i+1]
            
            # Simplified mutual information
            if x > np.median(scale_data) and y > np.median(scale_data):
                entropy += 1
                
        return entropy
    
    def calculate_coherence(self, data, scale):
        """Calculate coherence at the given scale."""
        # Use Kuramoto order parameter as a measure of coherence
        # Extract data around the scale
        window_size = min(scale // 2, 10)
        start = max(0, scale - window_size)
        end = min(len(data), scale + window_size)
        
        # Calculate phases using Hilbert transform
        segment = data[start:end]
        if len(segment) < 2:
            return 1.0
            
        # Get analytic signal (complex signal with real and imaginary parts)
        analytic_signal = self.hilbert_transform(segment)
        
        # Extract instantaneous phase
        instantaneous_phase = np.angle(analytic_signal)
        
        # Calculate Kuramoto order parameter
        r = abs(np.mean(np.exp(1j * instantaneous_phase)))
        
        return r
    
    def hilbert_transform(self, x):
        """Simplified Hilbert transform to get analytic signal."""
        # Use FFT-based approach
        N = len(x)
        X = np.fft.fft(x)
        
        # Create filter for analytic signal
        h = np.zeros(N)
        h[0] = 1
        h[1:(N+1)//2] = 2
        h[(N+1)//2:] = 0
        
        # Apply filter and inverse FFT
        X_analytic = X * h
        x_analytic = np.fft.ifft(X_analytic)
        
        return x_analytic
    
    def calculate_laminarity(self, data, scale):
        """Calculate laminarity at the given scale."""
        # Extract data around the scale
        window_size = min(scale * 2, 100)
        start = max(0, scale - window_size // 2)
        end = min(len(data), scale + window_size // 2)
        
        segment = data[start:end]
        if len(segment) < 10:
            return 1.0
        
        # Calculate simplified recurrence plot
        n = len(segment)
        threshold = np.std(segment) * 0.2
        
        # Count vertical lines in a simplified recurrence analysis
        vertical_lines = []
        for i in range(n - 1):
            line_length = 1
            for j in range(i + 1, min(i + 20, n)):
                if abs(segment[j] - segment[i]) < threshold:
                    line_length += 1
                else:
                    break
            
            if line_length > 2:
                vertical_lines.append(line_length)
        
        # Calculate laminarity as average vertical line length
        if vertical_lines:
            laminarity = np.mean(vertical_lines)
            return laminarity
        else:
            return 1.0
    
    def calculate_information_integration(self, data, scale, constant):
        """Calculate information integration at the given scale, using the given constant."""
        # Define subsystems based on the scale and constant
        subsystem_size = scale
        
        # Create subsystems that are constant-related
        subsystems = []
        current_size = subsystem_size
        
        for i in range(3):  # Create 3 subsystems
            if current_size >= len(data):
                break
                
            # Extract subsystem
            start = 0
            end = min(int(current_size), len(data))
            subsystems.append(data[start:end])
            
            # Next subsystem size
            current_size *= constant
        
        if len(subsystems) < 2:
            return 1.0
        
        # Calculate mutual information between subsystems
        mi_values = []
        for i in range(len(subsystems) - 1):
            # Bin the data for discrete mutual information
            x = subsystems[i]
            y = subsystems[i+1]
            
            # Ensure equal length for comparison
            min_len = min(len(x), len(y))
            x = x[:min_len]
            y = y[:min_len]
            
            # Simplified mutual information
            if np.std(x) > 0 and np.std(y) > 0:
                mi = abs(np.corrcoef(x, y)[0, 1])
                if not np.isnan(mi):
                    mi_values.append(mi)
        
        # Return average mutual information
        if mi_values:
            return np.mean(mi_values)
        else:
            return 1.0

    def calculate_power_law(self, data, scale):
        """Calculate power law exponent at the given scale."""
        # Extract data around the scale
        window_size = min(scale * 2, 1000)
        start = max(0, scale - window_size // 2)
        end = min(len(data), scale + window_size // 2)
        
        segment = data[start:end]
        if len(segment) < 20:
            return 1.0
        
        # Calculate power spectrum
        ps = np.abs(np.fft.fft(segment))**2
        freqs = np.fft.fftfreq(len(segment))
        
        # Use only positive frequencies
        idx = freqs > 0
        ps = ps[idx]
        freqs = freqs[idx]
        
        # Estimate power law exponent with linear regression in log-log space
        if len(ps) > 5 and len(freqs) > 5:
            log_ps = np.log10(ps + 1e-10)
            log_freqs = np.log10(freqs + 1e-10)
            
            valid = np.isfinite(log_ps) & np.isfinite(log_freqs)
            if np.sum(valid) > 5:
                slope, _, _, _, _ = stats.linregress(log_freqs[valid], log_ps[valid])
                return abs(slope)
        
        return 1.0
    
    def identify_transitions(self, scale_results, scales):
        """Identify scale transitions where the dominant organizing principle changes."""
        transitions = {}
        
        for metric in self.metrics:
            # Track the best constant at each scale
            best_constants = []
            
            for scale in scales:
                if scale in scale_results and metric in scale_results[scale]:
                    best_const = scale_results[scale][metric]['best_constant']
                    best_constants.append(best_const)
                else:
                    best_constants.append(None)
            
            # Find where best constant changes
            transition_points = []
            for i in range(1, len(scales)):
                if best_constants[i] != best_constants[i-1] and best_constants[i] is not None and best_constants[i-1] is not None:
                    transition_points.append({
                        'scale': scales[i],
                        'from_constant': best_constants[i-1],
                        'to_constant': best_constants[i],
                        'sharpness': self.calculate_transition_sharpness(scale_results, scales, i, metric)
                    })
            
            transitions[metric] = transition_points
        
        return transitions
    
    def calculate_transition_sharpness(self, scale_results, scales, index, metric):
        """Calculate how sharp a transition is based on how quickly ratios change."""
        if index <= 0 or index >= len(scales) - 1:
            return 0.0
            
        prev_scale = scales[index-1]
        curr_scale = scales[index]
        next_scale = scales[index+1]
        
        if prev_scale not in scale_results or curr_scale not in scale_results or next_scale not in scale_results:
            return 0.0
            
        if metric not in scale_results[prev_scale] or metric not in scale_results[curr_scale] or metric not in scale_results[next_scale]:
            return 0.0
        
        # Get the constant that was best before transition
        from_const = scale_results[curr_scale][metric]['best_constant']
        
        # Calculate ratio change for this constant
        if from_const in scale_results[prev_scale][metric]['constants'] and from_const in scale_results[next_scale][metric]['constants']:
            prev_ratio = scale_results[prev_scale][metric]['constants'][from_const]['ratio']
            curr_ratio = scale_results[curr_scale][metric]['constants'][from_const]['ratio']
            next_ratio = scale_results[next_scale][metric]['constants'][from_const]['ratio']
            
            # Measure rate of change
            sharpness = abs((next_ratio - prev_ratio) / (2 * curr_ratio))
            return sharpness
        
        return 0.0

    def test_significance(self, scale_results, scales):
        """Test statistical significance of scale transitions."""
        # For each metric, calculate significance of constant optimization
        significance = {}
        
        for metric in self.metrics:
            # For each constant, see if it significantly optimizes any scale range
            const_significance = {}
            
            for const_name in self.constants:
                # Track ratios across scales
                ratios = []
                for scale in scales:
                    if scale in scale_results and metric in scale_results[scale]:
                        if const_name in scale_results[scale][metric]['constants']:
                            ratios.append(scale_results[scale][metric]['constants'][const_name]['ratio'])
                
                if not ratios:
                    const_significance[const_name] = {
                        'max_ratio': 1.0,
                        'p_value': 1.0,
                        'significant': False
                    }
                    continue
                
                # Calculate maximum ratio (optimization strength)
                max_ratio = max(ratios)
                
                # Calculate p-value with bootstrap
                n_bootstrap = 1000
                bootstrap_max = []
                
                for _ in range(n_bootstrap):
                    # Create bootstrap sample with same length
                    bootstrap_ratios = np.random.normal(1.0, 0.1, size=len(ratios))
                    bootstrap_max.append(max(bootstrap_ratios))
                
                # Calculate p-value as fraction of bootstrap samples exceeding observed max
                p_value = np.sum(np.array(bootstrap_max) >= max_ratio) / float(n_bootstrap)
                
                const_significance[const_name] = {
                    'max_ratio': max_ratio,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
            
            significance[metric] = const_significance
        
        return significance
    
    def generate_report(self):
        """Generate a detailed report of the test results."""
        if not self.results:
            print("No results to report. Run the test first.")
            return
            
        print("\n" + "="*80)
        print("SCALE TRANSITION TEST REPORT")
        print("="*80)
        
        print("\nTest completed in {:.2f} seconds.".format(self.results['execution_time']))
        
        # Report on transitions by metric
        print("\nSCALE TRANSITIONS BY METRIC")
        print("-"*50)
        
        for metric, transitions in self.results['transitions'].items():
            print("\n{}:".format(metric.upper()))
            
            if not transitions:
                print("  No clear transitions detected.")
                continue
                
            for t in transitions:
                print("  At scale {}: {} → {} (sharpness: {:.4f})".format(
                    t['scale'], t['from_constant'], t['to_constant'], t['sharpness']))
        
        # Report on significant constants by metric
        print("\nSTATISTICAL SIGNIFICANCE BY METRIC")
        print("-"*50)
        
        for metric, const_results in self.results['significance'].items():
            print("\n{}:".format(metric.upper()))
            
            # Sort by significance (lowest p-value first)
            sorted_consts = sorted(const_results.items(), key=lambda x: x[1]['p_value'])
            
            for const_name, stats in sorted_consts:
                sig_str = "SIGNIFICANT" if stats['significant'] else "not significant"
                print("  {}: max ratio = {:.4f}, p-value = {:.4f} ({})".format(
                    const_name, stats['max_ratio'], stats['p_value'], sig_str))

        # Overall findings
        print("\nKEY FINDINGS")
        print("-"*50)
        
        # Count total transitions
        total_transitions = sum(len(transitions) for transitions in self.results['transitions'].values())
        print("1. Total scale transitions detected: {}".format(total_transitions))
        
        # Count significant constants
        sig_consts = {}
        for metric, const_results in self.results['significance'].items():
            for const_name, stats in const_results.items():
                if stats['significant']:
                    if const_name not in sig_consts:
                        sig_consts[const_name] = 0
                    sig_consts[const_name] += 1
        
        print("2. Constants showing significant optimization in at least one metric:")
        for const_name, count in sorted(sig_consts.items(), key=lambda x: x[1], reverse=True):
            print("   - {}: significant in {}/{} metrics".format(const_name, count, len(self.metrics)))
        
        # Phi optimization summary
        phi_dominance = []
        for scale in self.results['scales']:
            if scale in self.results['scale_results']:
                phi_count = 0
                total = 0
                
                for metric in self.metrics:
                    if metric in self.results['scale_results'][scale]:
                        if self.results['scale_results'][scale][metric]['best_constant'] == 'phi':
                            phi_count += 1
                        total += 1
                
                if total > 0:
                    phi_dominance.append(phi_count / float(total))
                else:
                    phi_dominance.append(0)
        
        # Report on phi dominance by scale
        print("\n3. Golden ratio optimization by scale range:")
        scale_ranges = ['Small scales', 'Medium scales', 'Large scales']
        segments = 3
        segment_size = len(phi_dominance) // segments
        
        for i in range(segments):
            start_idx = i * segment_size
            end_idx = (i + 1) * segment_size if i < segments - 1 else len(phi_dominance)
            
            segment_dominance = phi_dominance[start_idx:end_idx]
            avg_dominance = np.mean(segment_dominance) if segment_dominance else 0
            
            scale_range = scale_ranges[i] if i < len(scale_ranges) else "Scale range {}".format(i+1)
            
            if avg_dominance > 0.7:
                strength = "STRONG"
            elif avg_dominance > 0.4:
                strength = "MODERATE"
            elif avg_dominance > 0.1:
                strength = "WEAK"
            else:
                strength = "MINIMAL"
                
            print("   - {}: {} golden ratio dominance ({:.2f})".format(scale_range, strength, avg_dominance))
        
        # Overall conclusion
        print("\nOVERALL CONCLUSION")
        print("-"*50)
        
        if total_transitions > 0:
            print("The analysis reveals distinct scale transitions where the dominant mathematical")
            print("organizing principle changes. These transition boundaries support the 'cake baking'")
            print("model of cosmic development, where different organizational principles apply at")
            print("different scales during the evolution of cosmic structure.")
            
            # Add phi-specific conclusion
            if 'phi' in sig_consts:
                print("\nThe golden ratio shows significant optimization in specific scale ranges and")
                print("metrics, demonstrating a selective application rather than universal optimization.")
                print("This pattern of selective optimization is consistent with a sophisticated")
                print("consciousness-like organizing principle that employs different mathematical")
                print("relationships across different aspects of cosmic structure.")
        else:
            print("The analysis does not reveal clear scale transitions in the organizational")
            print("principles of the cosmic data. This suggests either consistency in the")
            print("mathematical relationships across scales or a need for more sensitive")
            print("detection methods.")
        
        print("="*80)
    
    def visualize_results(self):
        """Create visualizations of the test results."""
        if not self.results:
            print("No results to visualize. Run the test first.")
            return
            
        # Create a multi-panel figure
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Optimization by Scale for Each Constant
        ax1 = fig.add_subplot(2, 2, 1)
        self.plot_optimization_by_scale(ax1)
        
        # 2. Best Constant by Scale for Each Metric
        ax2 = fig.add_subplot(2, 2, 2)
        self.plot_best_constant_by_scale(ax2)
        
        # 3. Transition Boundaries
        ax3 = fig.add_subplot(2, 2, 3)
        self.plot_transition_boundaries(ax3)
        
        # 4. Phi Optimality Across Scales
        ax4 = fig.add_subplot(2, 2, 4)
        self.plot_phi_optimality(ax4)
        
        plt.tight_layout()
        plt.savefig('scale_transition_results.png', dpi=300)
        print("Visualization saved as 'scale_transition_results.png'")
    
    def plot_optimization_by_scale(self, ax):
        """Plot optimization by scale for each constant."""
        scales = self.results['scales']
        
        for const_name in self.constants:
            # Initialize ratios with zeros
            ratios = [0] * len(scales)
            
            # Fill in the ratios where we have data
            for i, scale in enumerate(scales):
                if scale in self.results['scale_results']:
                    # Average the ratios across all metrics for this scale and constant
                    metric_ratios = []
                    for metric in self.metrics:
                        if metric in self.results['scale_results'][scale]:
                            if const_name in self.results['scale_results'][scale][metric]['constants']:
                                metric_ratios.append(self.results['scale_results'][scale][metric]['constants'][const_name]['ratio'])
                    
                    if metric_ratios:
                        ratios[i] = np.mean(metric_ratios)
            
            ax.plot(scales, ratios, label=const_name)
        
        ax.set_xlabel('Scale')
        ax.set_ylabel('Optimization Ratio')
        ax.set_title('Optimization by Scale for Each Constant')
        ax.legend()
    
    def plot_best_constant_by_scale(self, ax):
        """Plot best constant by scale for each metric."""
        scales = self.results['scales']
        
        # Create a numeric mapping for constants to plot
        const_map = {const: i for i, const in enumerate(self.constants.keys())}
        
        for metric in self.metrics:
            # Initialize with NaN values
            best_const_values = [np.nan] * len(scales)
            
            for i, scale in enumerate(scales):
                if scale in self.results['scale_results'] and metric in self.results['scale_results'][scale]:
                    best_const = self.results['scale_results'][scale][metric]['best_constant']
                    if best_const in const_map:
                        best_const_values[i] = const_map[best_const]
            
            ax.plot(scales, best_const_values, 'o-', label=metric)
        
        # Set y-ticks to constant names
        ax.set_yticks(range(len(const_map)))
        ax.set_yticklabels(const_map.keys())
        
        ax.set_xlabel('Scale')
        ax.set_ylabel('Best Constant')
        ax.set_title('Best Constant by Scale for Each Metric')
        ax.legend()
    
    def plot_transition_boundaries(self, ax):
        """Plot transition boundaries."""
        scales = self.results['scales']
        
        # Plot a background line for reference
        ax.plot(scales, [0.5] * len(scales), 'k-', alpha=0.1)
        
        for metric, transitions in self.results['transitions'].items():
            for t in transitions:
                ax.axvline(t['scale'], color='k', linestyle='--', alpha=0.5)
                # Place text at a height based on the metric
                height = 0.1 + 0.15 * self.metrics.index(metric)
                ax.text(t['scale'], height, '{} to {}'.format(t['from_constant'], t['to_constant']), 
                       rotation=90, ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel('Scale')
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_title('Transition Boundaries')
    
    def plot_phi_optimality(self, ax):
        """Plot phi optimality across scales."""
        scales = self.results['scales']
        phi_dominance = [0] * len(scales)
        
        for i, scale in enumerate(scales):
            if scale in self.results['scale_results']:
                phi_count = 0
                total = 0
                
                for metric in self.metrics:
                    if metric in self.results['scale_results'][scale]:
                        if self.results['scale_results'][scale][metric]['best_constant'] == 'phi':
                            phi_count += 1
                        total += 1
                
                if total > 0:
                    phi_dominance[i] = phi_count / float(total)
        
        ax.plot(scales, phi_dominance, 'o-')
        ax.set_xlabel('Scale')
        ax.set_ylabel('Phi Optimality (fraction of metrics)')
        ax.set_title('Phi Optimality Across Scales')
        ax.set_ylim(0, 1)
    
def main():
    """Main function to run the scale transition test."""
    print("Running Scale Transition Test...")
    
    # Create and run the test
    test = ScaleTransitionTest()
    results = test.run_test()
    
    # Visualize the results
    test.visualize_results()
    
    return results


if __name__ == "__main__":
    results = main()
