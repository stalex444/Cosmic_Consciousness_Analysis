import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd
import time
import math
from scipy import stats
from scipy.signal import coherence as scipy_coherence
import warnings
warnings.filterwarnings('ignore')

class CoherenceMeasureAnalysis:
    """
    This script specifically investigates why coherence measures show poor consistency
    across datasets while other metrics like laminarity show excellent consistency.
    """
    
    def __init__(self, datasets=None):
        """Initialize with multiple CMB datasets."""
        print("Loading datasets from previous analysis...")
        self.phi = (1 + math.sqrt(5)) / 2  # Golden ratio
        self.datasets = datasets if datasets is not None else self.load_datasets()
        self.results = {}
        
    def load_datasets(self):
        """Load or generate the same datasets used in cross-verification."""
        print("Loading datasets from previous analysis...")
        datasets = {}
        
        # Generate simulated datasets similar to cross-verification test
        size = 2000
        
        # Generate base data with similar properties but different noise
        base_data = np.random.normal(0, 1, size=size)
        
        # Add some phi-based patterns to the base data
        for i in range(1, 6):
            scale = int(10 * self.phi ** i)
            if scale < size / 2:
                pattern = np.sin(np.arange(size) * 2 * np.pi / scale)
                base_data += pattern * (0.5 ** i)
        
        # Create dataset 1: Planck-like
        planck_like = base_data.copy()
        for i in range(6, 10):
            scale = int(10 * self.phi ** i * 0.2)
            if scale > 1 and scale < size / 2:
                pattern = np.sin(np.arange(size) * 2 * np.pi / scale)
                planck_like += pattern * (0.2 ** i)
        planck_like += np.random.normal(0, 0.2, size=size)
        
        # Create dataset 2: WMAP-like
        wmap_like = base_data.copy()
        for i in range(6, 9):
            scale = int(10 * self.phi ** i * 0.3)
            if scale > 1 and scale < size / 2:
                pattern = np.sin(np.arange(size) * 2 * np.pi / scale)
                wmap_like += pattern * (0.3 ** i)
        wmap_like += np.random.normal(0, 0.4, size=size)
        
        # Create dataset 3: Ground-based simulation
        ground_based = base_data.copy()
        ground_based *= 1.05
        ground_based += np.random.normal(0, 0.5, size=size)
        low_freq_atm = np.sin(np.arange(size) * 2 * np.pi / 200) * 0.3
        ground_based += low_freq_atm
        
        # Normalize all datasets
        planck_like = (planck_like - np.mean(planck_like)) / np.std(planck_like)
        wmap_like = (wmap_like - np.mean(wmap_like)) / np.std(wmap_like)
        ground_based = (ground_based - np.mean(ground_based)) / np.std(ground_based)
        
        # Add to datasets dictionary
        datasets['PLANCK_SIM'] = planck_like
        datasets['WMAP_SIM'] = wmap_like
        datasets['GROUND_SIM'] = ground_based
        
        # Create a random dataset for comparison
        datasets['RANDOM'] = np.random.normal(0, 1, size=len(next(iter(datasets.values()))))
        
        return datasets

    def run_analysis(self):
        """Run the full coherence measure analysis."""
        print("Starting coherence measure analysis...")
        start_time = time.time()
        
        # Load datasets if not already loaded
        if not self.datasets:
            self.load_datasets()
            print("Loaded {} datasets for analysis.".format(len(self.datasets)))
        
        # Dictionary to store results
        self.results = {}
        
        # 1. Compare different coherence calculation methods
        print("Comparing coherence calculation methods...")
        coherence_methods = {
            'kuramoto': self.calculate_kuramoto_order,
            'phase_coherence': self.calculate_phase_coherence,
            'magnitude_coherence': self.calculate_magnitude_coherence,
            'wavelet_coherence': self.calculate_wavelet_coherence,
            'spectral_coherence': self.calculate_spectral_coherence
        }
        
        method_results = {}
        method_consistency = {}
        
        for method_name, method_func in coherence_methods.items():
            method_results[method_name] = {}
            
            for dataset_name, dataset in self.datasets.items():
                coherence = method_func(dataset)
                method_results[method_name][dataset_name] = coherence
            
            # Calculate consistency across datasets
            values = list(method_results[method_name].values())
            cv = np.std(values) / np.mean(values) if np.mean(values) > 0 else 1.0
            consistency = 1 - min(cv, 1.0)  # Bounded between 0 and 1
            method_consistency[method_name] = consistency
        
        self.results['method_results'] = method_results
        self.results['method_consistency'] = method_consistency
        
        # 2. Analyze scale dependency
        print("Analyzing scale dependency...")
        self.results['scale_dependency'] = self.analyze_scale_dependency()
        
        # 3. Compare phi vs non-phi coherence
        print("Comparing phi vs non-phi coherence...")
        self.results['phi_coherence'] = self.analyze_phi_vs_nonphi_coherence()
        
        # 4. Analyze noise sensitivity
        print("Analyzing noise sensitivity...")
        self.results['noise_sensitivity'] = self.analyze_noise_sensitivity()
        
        # Record execution time
        execution_time = time.time() - start_time
        self.results['execution_time'] = execution_time
        
        print("Analysis completed in {:.2f} seconds.".format(execution_time))
        
        # Generate report
        self.generate_report()
        
        return self.results
    
    def calculate_kuramoto_order(self, data):
        """Calculate Kuramoto order parameter (original method used)."""
        n = len(data)
        # Convert data to phases
        phases = np.angle(np.fft.fft(data))
        
        # Calculate order parameter
        r = abs(np.sum(np.exp(1j * phases))) / n
        return r
    
    def calculate_phase_coherence(self, data):
        """Calculate phase-based coherence between frequency bands."""
        # Extract phase information at different frequency bands
        fft_result = np.fft.fft(data)
        phase = np.angle(fft_result)
        
        # Define frequency bands (as indices in the FFT result)
        bands = [
            (1, 10),      # Very low frequencies
            (10, 30),     # Low frequencies
            (30, 100),    # Mid frequencies
            (100, 300)    # Higher frequencies
        ]
        
        # Calculate phase coherence between adjacent bands
        coherence_values = []
        for i in range(len(bands)-1):
            band1 = phase[bands[i][0]:bands[i][1]]
            band2 = phase[bands[i+1][0]:bands[i+1][1]]
            
            # Resample to match lengths if needed
            min_len = min(len(band1), len(band2))
            band1 = band1[:min_len]
            band2 = band2[:min_len]
            
            # Calculate circular correlation
            c = abs(np.mean(np.exp(1j * (band1 - band2))))
            coherence_values.append(c)
        
        # Return average coherence
        return np.mean(coherence_values) if coherence_values else 0
    
    def calculate_magnitude_coherence(self, data):
        """Calculate magnitude-based coherence between frequency bands."""
        # Extract magnitude information at different frequency bands
        fft_result = np.fft.fft(data)
        magnitude = np.abs(fft_result)
        
        # Define frequency bands (as indices in the FFT result)
        bands = [
            (1, 10),      # Very low frequencies
            (10, 30),     # Low frequencies
            (30, 100),    # Mid frequencies
            (100, 300)    # Higher frequencies
        ]
        
        # Calculate magnitude coherence between adjacent bands
        coherence_values = []
        for i in range(len(bands)-1):
            band1 = magnitude[bands[i][0]:bands[i][1]]
            band2 = magnitude[bands[i+1][0]:bands[i+1][1]]
            
            # Resample to match lengths if needed
            min_len = min(len(band1), len(band2))
            band1 = band1[:min_len]
            band2 = band2[:min_len]
            
            # Calculate correlation
            if np.std(band1) > 0 and np.std(band2) > 0:
                c = abs(np.corrcoef(band1, band2)[0,1])
                if not np.isnan(c):
                    coherence_values.append(c)
        
        # Return average coherence
        return np.mean(coherence_values) if coherence_values else 0
    
    def calculate_wavelet_coherence(self, data):
        """Calculate wavelet-based coherence across scales."""
        # Simple wavelet decomposition
        scales = [2, 4, 8, 16, 32, 64]
        wavelet_coeffs = []
        
        for scale in scales:
            # Simple approximation of wavelet transform using convolution with a Gaussian
            sigma = scale / 2
            t = np.arange(-3*sigma, 3*sigma+1)
            gaussian = np.exp(-t**2/(2*sigma**2))
            gaussian /= np.sum(gaussian)
            
            # Convolve with data and downsample
            conv = np.convolve(data, gaussian, mode='same')
            wavelet_coeffs.append(conv[::scale][:100])  # Take first 100 coefficients after downsampling
        
        # Calculate coherence between adjacent scales
        coherence_values = []
        for i in range(len(scales)-1):
            coeff1 = wavelet_coeffs[i]
            coeff2 = wavelet_coeffs[i+1]
            
            # Match lengths
            min_len = min(len(coeff1), len(coeff2))
            coeff1 = coeff1[:min_len]
            coeff2 = coeff2[:min_len]
            
            # Calculate correlation
            if np.std(coeff1) > 0 and np.std(coeff2) > 0:
                c = abs(np.corrcoef(coeff1, coeff2)[0,1])
                if not np.isnan(c):
                    coherence_values.append(c)
        
        # Return average coherence
        return np.mean(coherence_values) if coherence_values else 0
    
    def calculate_spectral_coherence(self, data):
        """Calculate spectral coherence using scipy's implementation."""
        # Split the data into two halves
        n = len(data)
        data1 = data[:n//2]
        data2 = data[n//2:]
        
        # Calculate magnitude squared coherence
        f, coh = scipy_coherence(data1, data2, fs=1.0, nperseg=min(256, n//4))
        
        # Return average coherence
        return np.mean(coh)

    def analyze_scale_dependency(self):
        """Analyze how coherence varies across different scales."""
        # Define scales to analyze
        scales = [10, 20, 50, 100, 200, 500]
        
        # For each dataset, calculate coherence at different scales
        scale_results = {}
        for dataset_name, dataset in self.datasets.items():
            if dataset_name == 'RANDOM':
                continue
                
            scale_results[dataset_name] = []
            for scale in scales:
                # Divide dataset into segments of specified scale
                segments = []
                for i in range(0, len(dataset) - scale, scale):
                    segments.append(dataset[i:i+scale])
                
                if len(segments) < 2:
                    scale_results[dataset_name].append(0)
                    continue
                
                # Calculate coherence between adjacent segments
                coherence_values = []
                for i in range(len(segments)-1):
                    seg1 = segments[i]
                    seg2 = segments[i+1]
                    
                    # Calculate simple correlation as coherence measure
                    if np.std(seg1) > 0 and np.std(seg2) > 0:
                        c = abs(np.corrcoef(seg1, seg2)[0,1])
                        if not np.isnan(c):
                            coherence_values.append(c)
                
                # Average coherence at this scale
                scale_results[dataset_name].append(np.mean(coherence_values) if coherence_values else 0)
        
        # Calculate consistency across datasets at each scale
        scale_consistency = []
        for i in range(len(scales)):
            values = [results[i] for results in scale_results.values()]
            cv = np.std(values) / np.mean(values) if np.mean(values) > 0 else 1.0
            scale_consistency.append(1 - min(cv, 1.0))
        
        return {
            'scales': scales,
            'dataset_results': scale_results,
            'consistency': scale_consistency
        }
    
    def analyze_phi_vs_nonphi_coherence(self):
        """Compare coherence at phi-related scales vs. non-phi-related scales."""
        # Define phi-related and non-phi-related scales
        phi_scales = [int(10 * self.phi**i) for i in range(1, 6) if 10 * self.phi**i < 500]
        
        # Create non-phi scales that aren't too close to phi scales
        non_phi_scales = []
        candidate_scales = range(10, 500, 10)
        for scale in candidate_scales:
            # Check if scale is far enough from any phi scale
            if all(abs(scale - phi_scale) > 5 for phi_scale in phi_scales):
                non_phi_scales.append(scale)
                if len(non_phi_scales) >= len(phi_scales):
                    break
        
        results = {}
        for dataset_name, dataset in self.datasets.items():
            if dataset_name == 'RANDOM':
                continue
                
            # Calculate coherence at phi-related scales
            phi_coherence = []
            for scale in phi_scales:
                # Divide dataset into segments
                segments = []
                for i in range(0, len(dataset) - scale, scale):
                    segments.append(dataset[i:i+scale])
                
                if len(segments) < 2:
                    continue
                
                # Calculate coherence between adjacent segments
                coherence_values = []
                for i in range(len(segments)-1):
                    seg1 = segments[i]
                    seg2 = segments[i+1]
                    
                    if np.std(seg1) > 0 and np.std(seg2) > 0:
                        c = abs(np.corrcoef(seg1, seg2)[0,1])
                        if not np.isnan(c):
                            coherence_values.append(c)
                
                if coherence_values:
                    phi_coherence.append(np.mean(coherence_values))
            
            # Calculate coherence at non-phi-related scales
            non_phi_coherence = []
            for scale in non_phi_scales:
                # Divide dataset into segments
                segments = []
                for i in range(0, len(dataset) - scale, scale):
                    segments.append(dataset[i:i+scale])
                
                if len(segments) < 2:
                    continue
                
                # Calculate coherence between adjacent segments
                coherence_values = []
                for i in range(len(segments)-1):
                    seg1 = segments[i]
                    seg2 = segments[i+1]
                    
                    if np.std(seg1) > 0 and np.std(seg2) > 0:
                        c = abs(np.corrcoef(seg1, seg2)[0,1])
                        if not np.isnan(c):
                            coherence_values.append(c)
                
                if coherence_values:
                    non_phi_coherence.append(np.mean(coherence_values))
            
            # Calculate average coherence and ratio
            avg_phi = np.mean(phi_coherence) if phi_coherence else 0
            avg_non_phi = np.mean(non_phi_coherence) if non_phi_coherence else 0
            ratio = avg_phi / avg_non_phi if avg_non_phi > 0 else 1.0
            
            results[dataset_name] = {
                'phi_coherence': avg_phi,
                'non_phi_coherence': avg_non_phi,
                'ratio': ratio
            }
        
        # Calculate consistency of the phi vs non-phi ratio
        ratios = [results[dataset]['ratio'] for dataset in results]
        cv = np.std(ratios) / np.mean(ratios) if np.mean(ratios) > 0 else 1.0
        consistency = 1 - min(cv, 1.0)
        
        return {
            'phi_scales': phi_scales,
            'non_phi_scales': non_phi_scales,
            'dataset_results': results,
            'ratio_consistency': consistency
        }

    def analyze_noise_sensitivity(self):
        """Analyze how sensitive coherence measures are to noise levels."""
        # Start with the first dataset
        base_dataset = list(self.datasets.values())[0]
        
        # Create versions with different noise levels
        noise_levels = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
        noisy_datasets = {}
        
        for noise in noise_levels:
            # Add noise while preserving the original signal
            noisy_data = base_dataset + np.random.normal(0, noise, size=len(base_dataset))
            # Normalize
            noisy_data = (noisy_data - np.mean(noisy_data)) / np.std(noisy_data)
            noisy_datasets["Noise_{}".format(noise)] = noisy_data
        
        # Calculate different coherence measures for each noise level
        coherence_methods = {
            'kuramoto': self.calculate_kuramoto_order,
            'phase_coherence': self.calculate_phase_coherence,
            'magnitude_coherence': self.calculate_magnitude_coherence,
            'wavelet_coherence': self.calculate_wavelet_coherence,
            'spectral_coherence': self.calculate_spectral_coherence
        }
        
        noise_results = {}
        for method_name, method_func in coherence_methods.items():
            noise_results[method_name] = []
            # Calculate coherence for original dataset
            original_coherence = method_func(base_dataset)
            
            for noise in noise_levels:
                noisy_data = noisy_datasets["Noise_{}".format(noise)]
                # Calculate coherence for noisy dataset
                noisy_coherence = method_func(noisy_data)
                # Calculate how much coherence changed (ratio to original)
                ratio = noisy_coherence / original_coherence if original_coherence > 0 else 0
                noise_results[method_name].append(ratio)
        
        return {
            'noise_levels': noise_levels,
            'method_results': noise_results
        }

    def generate_report(self):
        """Generate a comprehensive report of findings."""
        if not self.results:
            print("No results to report. Run the analysis first.")
            return
            
        print("\n" + "="*80)
        print("COHERENCE MEASURE ANALYSIS REPORT")
        print("="*80)
        
        print("\nAnalysis completed in {:.2f} seconds.".format(self.results['execution_time']))
        
        # 1. Report on different coherence methods
        print("\n1. COMPARISON OF COHERENCE CALCULATION METHODS")
        print("-"*50)
        
        print("\nConsistency across datasets:")
        # Sort methods by consistency
        sorted_methods = sorted(self.results['method_consistency'].items(), 
                               key=lambda x: x[1], reverse=True)
        
        for method, consistency in sorted_methods:
            interpretation = ""
            if consistency > 0.9:
                interpretation = "Excellent consistency"
            elif consistency > 0.7:
                interpretation = "Good consistency"
            elif consistency > 0.5:
                interpretation = "Moderate consistency"
            else:
                interpretation = "Poor consistency"
                
            print("  {}: {:.4f} - {}".format(method, consistency, interpretation))
        
        # 2. Report on scale dependency
        print("\n2. SCALE DEPENDENCY OF COHERENCE")
        print("-"*50)
        
        scale_data = self.results['scale_dependency']
        scales = scale_data['scales']
        consistency = scale_data['consistency']
        
        print("\nConsistency across datasets at different scales:")
        for i, scale in enumerate(scales):
            interpretation = ""
            if consistency[i] > 0.9:
                interpretation = "Excellent"
            elif consistency[i] > 0.7:
                interpretation = "Good"
            elif consistency[i] > 0.5:
                interpretation = "Moderate"
            else:
                interpretation = "Poor"
                
            print("  Scale {}: {:.4f} - {}".format(scale, consistency[i], interpretation))
        
        # Find most and least consistent scales
        most_consistent_idx = np.argmax(consistency)
        least_consistent_idx = np.argmin(consistency)
        
        print("\nMost consistent scale: {} (consistency: {:.4f})".format(scales[most_consistent_idx], consistency[most_consistent_idx]))
        print("Least consistent scale: {} (consistency: {:.4f})".format(scales[least_consistent_idx], consistency[least_consistent_idx]))
        
        # 3. Report on phi vs non-phi coherence
        print("\n3. PHI VS NON-PHI COHERENCE COMPARISON")
        print("-"*50)
        
        phi_data = self.results['phi_coherence']
        print("\nPhi-related scales: {}".format(phi_data['phi_scales']))
        print("Non-phi scales: {}".format(phi_data['non_phi_scales']))
        
        print("\nCoherence ratios (phi / non-phi) by dataset:")
        for dataset, result in phi_data['dataset_results'].items():
            interpretation = ""
            if result['ratio'] > 1.5:
                interpretation = "Strongly phi-favoring"
            elif result['ratio'] > 1.1:
                interpretation = "Moderately phi-favoring"
            elif result['ratio'] > 0.9:
                interpretation = "Neutral"
            else:
                interpretation = "Non-phi-favoring"
                
            print("  {}: {:.4f} - {}".format(dataset, result['ratio'], interpretation))
        
        print("\nConsistency of phi/non-phi ratio: {:.4f}".format(phi_data['ratio_consistency']))

        # 4. Report on noise sensitivity
        print("\n4. NOISE SENSITIVITY ANALYSIS")
        print("-"*50)
        
        noise_data = self.results['noise_sensitivity']
        noise_levels = noise_data['noise_levels']
        
        print("\nCoherence retention at highest noise level (5.0) by method:")
        for method, ratios in noise_data['method_results'].items():
            highest_noise_ratio = ratios[-1]
            
            interpretation = ""
            if highest_noise_ratio > 0.9:
                interpretation = "Extremely robust"
            elif highest_noise_ratio > 0.7:
                interpretation = "Very robust"
            elif highest_noise_ratio > 0.5:
                interpretation = "Moderately robust"
            elif highest_noise_ratio > 0.3:
                interpretation = "Somewhat sensitive"
            else:
                interpretation = "Highly sensitive"
                
            print("  {}: {:.4f} - {}".format(method, highest_noise_ratio, interpretation))
        
        # Find most and least robust methods
        method_names = list(noise_data['method_results'].keys())
        highest_noise_ratios = [results[-1] for results in noise_data['method_results'].values()]
        
        most_robust_idx = np.argmax(highest_noise_ratios)
        least_robust_idx = np.argmin(highest_noise_ratios)
        
        print("\nMost robust method: {} (retained {:.4f} of coherence)".format(method_names[most_robust_idx], highest_noise_ratios[most_robust_idx]))
        print("Least robust method: {} (retained {:.4f} of coherence)".format(method_names[least_robust_idx], highest_noise_ratios[least_robust_idx]))
        
        # 5. Overall conclusion
        print("\nOVERALL CONCLUSIONS")
        print("-"*50)
        
        # Find best coherence method
        best_method, best_consistency = sorted_methods[0]
        
        print("1. The '{}' method shows the best consistency ({:.4f}) across datasets.".format(best_method, best_consistency))
        
        # Scale findings
        if np.max(consistency) > 0.7:
            print("2. Coherence is most consistent at scale {} and least consistent at scale {}.".format(scales[most_consistent_idx], scales[least_consistent_idx]))
        else:
            print("2. Coherence shows poor consistency across all scales, suggesting fundamental scale-dependency in coherence measures.")
        
        # Phi vs non-phi findings
        if phi_data['ratio_consistency'] > 0.7:
            avg_ratio = np.mean([result['ratio'] for result in phi_data['dataset_results'].values()])
            if avg_ratio > 1.1:
                print("3. Phi-related scales consistently show higher coherence than non-phi scales (avg ratio: {:.2f}).".format(avg_ratio))
            elif avg_ratio < 0.9:
                print("3. Non-phi scales consistently show higher coherence than phi-related scales (avg ratio: {:.2f}).".format(avg_ratio))
            else:
                print("3. Phi-related and non-phi scales show similar coherence values across datasets.")
        else:
            print("3. The relationship between phi-related and non-phi coherence varies significantly across datasets.")
        
        # Noise sensitivity
        avg_noise_sensitivity = np.mean(highest_noise_ratios)
        if avg_noise_sensitivity < 0.5:
            print("4. Coherence measures are generally highly sensitive to noise, which may explain poor cross-dataset consistency.")
        else:
            print("4. Coherence measures show moderate noise robustness (avg: {:.2f}), but some methods are much more robust than others.".format(avg_noise_sensitivity))

        # Final insight on coherence anomaly
        print("\nEXPLANATION FOR COHERENCE ANOMALY")
        print("-"*50)
        
        # Construct explanation based on findings
        if np.max(self.results['method_consistency'].values()) > 0.7:
            print("The poor cross-dataset consistency previously observed for coherence measures appears to be method-specific. While the original Kuramoto method showed poor consistency, the {} method shows much better consistency ({:.4f}).".format(best_method, best_consistency))
        elif np.max(consistency) > 0.7:
            print("The poor cross-dataset consistency previously observed for coherence measures appears to be scale-specific. Consistency is good at scale {} ({:.4f}) but poor at other scales.".format(scales[most_consistent_idx], consistency[most_consistent_idx]))
        elif avg_noise_sensitivity < 0.5:
            print("The poor cross-dataset consistency previously observed for coherence measures likely results from high noise sensitivity. Different noise profiles across datasets significantly affect coherence measurements.")
        else:
            print("The poor cross-dataset consistency previously observed for coherence measures appears to result from a combination of factors: method sensitivity, scale dependency, and noise sensitivity. Coherence measures intrinsically capture different aspects of the data than metrics like laminarity.")
        
        print("="*80)
    
    def visualize_results(self):
        """Create visualizations of the analysis results."""
        if not self.results:
            print("No results to visualize. Run the analysis first.")
            return
            
        # Create a multi-panel figure
        fig = plt.figure(figsize=(18, 16))
        
        # 1. Method Consistency
        ax1 = fig.add_subplot(2, 2, 1)
        self.plot_method_consistency(ax1)
        
        # 2. Scale Dependency
        ax2 = fig.add_subplot(2, 2, 2)
        self.plot_scale_dependency(ax2)
        
        # 3. Phi vs Non-Phi Coherence
        ax3 = fig.add_subplot(2, 2, 3)
        self.plot_phi_nonphi_comparison(ax3)
        
        # 4. Noise Sensitivity
        ax4 = fig.add_subplot(2, 2, 4)
        self.plot_noise_sensitivity(ax4)
        
        plt.tight_layout()
        # Save figure instead of showing it
        plt.savefig('coherence_analysis_results.png', dpi=300)
        print("\nVisualization saved as 'coherence_analysis_results.png'")
    
    def plot_method_consistency(self, ax):
        """Plot consistency of different coherence methods."""
        methods = list(self.results['method_consistency'].keys())
        consistency_values = list(self.results['method_consistency'].values())
        
        # Sort by consistency value
        sorted_indices = np.argsort(consistency_values)[::-1]  # Descending
        sorted_methods = [methods[i] for i in sorted_indices]
        sorted_values = [consistency_values[i] for i in sorted_indices]
        
        # Create color gradient based on consistency value
        colors = plt.cm.viridis(np.array(sorted_values))
        
        ax.bar(sorted_methods, sorted_values, color=colors)
        ax.set_ylim(0, 1)
        ax.set_ylabel('Consistency Score')
        ax.set_title('Coherence Method Consistency Across Datasets')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add consistency thresholds
        ax.axhline(y=0.9, color='g', linestyle='--', alpha=0.7, label='Excellent')
        ax.axhline(y=0.7, color='y', linestyle='--', alpha=0.7, label='Good')
        ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='Moderate')
        ax.legend()
        
        # Rotate x-axis labels for better readability
        ax.set_xticklabels(sorted_methods, rotation=45, ha='right')
    
    def plot_scale_dependency(self, ax):
        """Plot consistency across datasets at different scales."""
        scale_data = self.results['scale_dependency']
        scales = scale_data['scales']
        consistency = scale_data['consistency']
        
        # Create color gradient based on consistency value
        colors = plt.cm.viridis(np.array(consistency))
        
        ax.bar(scales, consistency, color=colors)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Scale')
        ax.set_ylabel('Consistency Score')
        ax.set_title('Coherence Consistency Across Scales')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add consistency thresholds
        ax.axhline(y=0.9, color='g', linestyle='--', alpha=0.7, label='Excellent')
        ax.axhline(y=0.7, color='y', linestyle='--', alpha=0.7, label='Good')
        ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='Moderate')
        ax.legend()
    
    def plot_phi_nonphi_comparison(self, ax):
        """Plot comparison of phi and non-phi coherence across datasets."""
        phi_data = self.results['phi_coherence']
        
        # Extract datasets and ratios
        datasets = list(phi_data['dataset_results'].keys())
        ratios = [phi_data['dataset_results'][dataset]['ratio'] for dataset in datasets]
        
        # Create bars for phi and non-phi coherence values
        phi_values = [phi_data['dataset_results'][dataset]['phi_coherence'] for dataset in datasets]
        non_phi_values = [phi_data['dataset_results'][dataset]['non_phi_coherence'] for dataset in datasets]
        
        x = np.arange(len(datasets))
        width = 0.35
        
        ax.bar(x - width/2, phi_values, width, label='Phi-related scales', color='blue')
        ax.bar(x + width/2, non_phi_values, width, label='Non-phi scales', color='red')
        
        ax.set_xlabel('Dataset')
        ax.set_ylabel('Coherence Value')
        ax.set_title('Phi vs Non-Phi Coherence by Dataset')
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, rotation=45, ha='right')
        ax.legend()
        
        # Add ratio values as text
        for i, ratio in enumerate(ratios):
            ax.text(i, max(phi_values[i], non_phi_values[i]) + 0.02, 
                   "Ratio: {:.2f}".format(ratio), ha='center')
        
        # Add horizontal line at equal coherence (ratio=1)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        # Add note about consistency
        ax.text(0.5, 0.95, "Ratio Consistency: {:.4f}".format(phi_data['ratio_consistency']), 
               transform=ax.transAxes, ha='center', fontsize=10)
    
    def plot_noise_sensitivity(self, ax):
        """Plot sensitivity of different coherence methods to noise."""
        noise_data = self.results['noise_sensitivity']
        noise_levels = noise_data['noise_levels']
        
        # Plot line for each method
        for method, values in noise_data['method_results'].items():
            ax.plot(noise_levels, values, marker='o', label=method)
        
        ax.set_xlabel('Noise Level')
        ax.set_ylabel('Coherence Retention Ratio')
        ax.set_title('Noise Sensitivity of Coherence Methods')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        
        # Add horizontal line at full retention (1.0)
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)


def main():
    """Main function to run the coherence measure analysis."""
    print("Running Coherence Measure Analysis...")
    
    # Create and run the analysis
    analysis = CoherenceMeasureAnalysis()
    results = analysis.run_analysis()
    
    # Visualize the results
    analysis.visualize_results()
    
    return results


if __name__ == "__main__":
    results = main()
