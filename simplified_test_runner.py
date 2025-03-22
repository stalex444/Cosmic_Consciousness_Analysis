#!/usr/bin/env python3
"""
Simplified test runner for the additional tests in the Cosmic Consciousness Analysis framework.
This script runs the tests with minimal data to avoid hanging issues.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import traceback
from scipy import stats

class SimplifiedTestRunner:
    """Simplified test runner for Cosmic Consciousness Analysis tests."""
    
    def __init__(self, data_dir="planck_data", monte_carlo_sims=20):
        """
        Initialize the test runner.
        
        Parameters:
        -----------
        data_dir : str
            Directory containing Planck CMB data files
        monte_carlo_sims : int
            Number of Monte Carlo simulations for significance testing
        """
        self.data_dir = data_dir
        self.phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        self.monte_carlo_sims = monte_carlo_sims
        
        # Load data
        self.load_data()
    
    def load_data(self):
        """Load the EE spectrum data."""
        try:
            # Import EE power spectrum - try binned version first
            ee_file = os.path.join(self.data_dir, "power_spectra", "COM_PowerSpect_CMB-EE-binned_R3.02.txt")
            
            # Print the absolute path for debugging
            print(f"Looking for EE spectrum file: {os.path.abspath(ee_file)}")
            
            # Load the data
            ee_data = np.loadtxt(ee_file)
            
            # Store the data
            self.ell = ee_data[:, 0]
            self.ee_power = ee_data[:, 1]
            self.ee_error = ee_data[:, 2] if ee_data.shape[1] > 2 else np.sqrt(ee_data[:, 1])
            
            print(f"Loaded EE spectrum with {len(self.ell)} multipoles")
            
            # Calculate mean and standard deviation for normalization
            self.mean_power = np.mean(self.ee_power)
            self.std_power = np.std(self.ee_power)
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            traceback.print_exc()
            sys.exit(1)
    
    def run_cross_scale_correlations_test(self):
        """Run the Cross-Scale Correlations Test."""
        print("Running Cross-Scale Correlations Test...")
        
        # Define scales separated by powers of phi
        base_scales = [10, 20, 50, 100]
        phi_scales = []
        
        for base in base_scales:
            scale_family = [base]
            for i in range(1, 4):  # Add 3 phi-related scales
                scale_family.append(int(round(base * phi**i)))
            phi_scales.append(scale_family)
        
        # Calculate correlations between phi-related scales
        phi_correlations = []
        for family in phi_scales:
            for i in range(len(family)-1):
                scale1 = family[i]
                scale2 = family[i+1]
                
                # Find indices in the data
                idx1 = np.abs(self.ell - scale1).argmin()
                idx2 = np.abs(self.ell - scale2).argmin()
                
                # Calculate correlation (inverse of absolute difference in normalized power)
                power1 = self.ee_power[idx1]
                power2 = self.ee_power[idx2]
                
                # Normalize by the mean power
                norm_power1 = power1 / self.mean_power
                norm_power2 = power2 / self.mean_power
                
                # Calculate correlation (1 / absolute difference)
                correlation = 1.0 / (abs(norm_power1 - norm_power2) + 0.01)
                phi_correlations.append(correlation)
        
        # Calculate mean correlation for phi-related scales
        mean_phi_corr = np.mean(phi_correlations)
        
        # Compare with random scale relationships
        random_correlations = []
        for _ in range(self.monte_carlo_sims):
            # Select random scales
            random_scales = np.random.choice(self.ell, len(phi_correlations)*2)
            
            # Calculate correlations
            for i in range(0, len(random_scales), 2):
                if i+1 < len(random_scales):
                    scale1 = random_scales[i]
                    scale2 = random_scales[i+1]
                    
                    # Find indices in the data
                    idx1 = np.abs(self.ell - scale1).argmin()
                    idx2 = np.abs(self.ell - scale2).argmin()
                    
                    # Calculate correlation
                    power1 = self.ee_power[idx1]
                    power2 = self.ee_power[idx2]
                    
                    # Normalize by the mean power
                    norm_power1 = power1 / self.mean_power
                    norm_power2 = power2 / self.mean_power
                    
                    # Calculate correlation
                    correlation = 1.0 / (abs(norm_power1 - norm_power2) + 0.01)
                    random_correlations.append(correlation)
        
        # Calculate mean correlation for random scales
        mean_random_corr = np.mean(random_correlations)
        
        # Calculate statistical significance
        random_std = np.std(random_correlations)
        z_score = (mean_phi_corr - mean_random_corr) / random_std
        p_value = 1 - stats.norm.cdf(z_score)
        
        return mean_phi_corr, mean_random_corr, z_score, p_value
    
    def run_optimization_test(self):
        """Run the Optimization Test."""
        print("Running Optimization Test...")
        
        # Define a measure of how well-suited the spectrum is for complex structure formation
        # Use scales relevant for galaxy formation (simplified)
        galaxy_scales = [200, 500, 800]  # Multipoles related to galaxy formation scales
        
        # Find closest multipoles in our data
        galaxy_indices = [np.abs(self.ell - scale).argmin() for scale in galaxy_scales]
        galaxy_powers = [self.ee_power[i] for i in galaxy_indices]
        actual_scales = [self.ell[i] for i in galaxy_indices]
        
        # Calculate the power ratios
        power_ratios = [galaxy_powers[i]/galaxy_powers[i+1] for i in range(len(galaxy_powers)-1)]
        
        # Calculate how close these ratios are to the golden ratio
        gr_deviations = [abs(ratio - self.phi) for ratio in power_ratios]
        mean_deviation = np.mean(gr_deviations)
        
        # Monte Carlo: Compare with random expectation
        print("Running Monte Carlo simulations...")
        random_deviations = []
        
        for _ in range(self.monte_carlo_sims):
            random_scales = np.random.choice(self.ell, len(galaxy_scales), replace=False)
            
            # Find indices for random scales
            random_indices = [np.abs(self.ell - scale).argmin() for scale in random_scales]
            random_powers = [self.ee_power[i] for i in random_indices]
            
            # Calculate random power ratios
            random_ratios = [random_powers[i]/random_powers[i+1] for i in range(len(random_powers)-1)]
            
            # Calculate deviations from golden ratio
            random_gr_deviations = [abs(ratio - self.phi) for ratio in random_ratios]
            random_deviations.append(np.mean(random_gr_deviations))
        
        # Calculate significance
        mean_random = np.mean(random_deviations)
        std_random = np.std(random_deviations)
        
        z_score = (mean_random - mean_deviation) / std_random  # Note: reversed because smaller deviation is better
        p_value = stats.norm.cdf(z_score)
        
        optimization_ratio = mean_random / mean_deviation
        
        return mean_deviation, mean_random, z_score, p_value, optimization_ratio
    
    def run_golden_symmetries_test(self):
        """Run the Golden Symmetries Test."""
        print("Running Golden Symmetries Test...")
        
        # Define symmetry measure: look for points where the spectrum is symmetric around a golden ratio point
        symmetry_scores = []
        
        # Consider each point as a potential center of symmetry
        for i in range(1, min(50, len(self.ell) - 1)):  # Limit to 50 points to avoid hanging
            center_ell = self.ell[i]
            
            # Look for pairs of points at golden ratio distances from the center
            for j in range(1, 3):  # Check a few different distances
                distance = j * 10  # Base distance in multipole units
                
                # Calculate golden ratio related distances
                left_distance = distance / self.phi
                right_distance = distance * self.phi
                
                # Find closest points at these distances
                left_ell = center_ell - left_distance
                right_ell = center_ell + right_distance
                
                # Find indices of closest points
                left_idx = np.abs(self.ell - left_ell).argmin()
                right_idx = np.abs(self.ell - right_ell).argmin()
                
                # Calculate symmetry score (inverse of power difference)
                left_power = self.ee_power[left_idx]
                right_power = self.ee_power[right_idx]
                
                # Normalize by mean power
                norm_left = left_power / self.mean_power
                norm_right = right_power / self.mean_power
                
                # Calculate symmetry score (higher means more symmetric)
                symmetry = 1.0 / (abs(norm_left - norm_right) + 0.01)
                symmetry_scores.append(symmetry)
        
        # Calculate mean symmetry score
        mean_symmetry = np.mean(symmetry_scores)
        
        # Monte Carlo: Compare with random expectation
        print("Running Monte Carlo simulations...")
        random_symmetries = []
        
        for _ in range(self.monte_carlo_sims):
            random_scores = []
            
            # Generate random symmetry tests
            for _ in range(len(symmetry_scores)):
                # Pick random indices
                idx1 = np.random.randint(0, len(self.ell))
                idx2 = np.random.randint(0, len(self.ell))
                
                # Calculate symmetry score
                power1 = self.ee_power[idx1]
                power2 = self.ee_power[idx2]
                
                # Normalize by mean power
                norm1 = power1 / self.mean_power
                norm2 = power2 / self.mean_power
                
                # Calculate symmetry score
                symmetry = 1.0 / (abs(norm1 - norm2) + 0.01)
                random_scores.append(symmetry)
            
            random_symmetries.append(np.mean(random_scores))
        
        # Calculate significance
        mean_random = np.mean(random_symmetries)
        std_random = np.std(random_symmetries)
        
        z_score = (mean_symmetry - mean_random) / std_random
        p_value = 1 - stats.norm.cdf(z_score)
        
        return mean_symmetry, mean_random, z_score, p_value
    
    def run_phi_network_test(self):
        """Run the Phi Network Test."""
        print("Running Phi Network Test...")
        
        # Create a network where nodes are multipoles and edges connect multipoles related by the golden ratio
        # We'll use a simplified approach for this test
        
        # Find all pairs of multipoles related by the golden ratio
        phi_pairs = []
        
        # Limit the number of multipoles to avoid hanging
        max_ell = min(100, len(self.ell))
        
        for i in range(max_ell):
            for j in range(i+1, max_ell):
                ratio = self.ell[j] / self.ell[i]
                if abs(ratio - self.phi) < 0.1:  # Allow some tolerance
                    phi_pairs.append((i, j))
        
        # Calculate the clustering coefficient of the phi network
        # Clustering coefficient measures how interconnected the network is
        clustering = len(phi_pairs) / (max_ell * (max_ell - 1) / 2)
        
        # Monte Carlo: Compare with random networks
        print("Running Monte Carlo simulations...")
        random_clusterings = []
        
        for _ in range(self.monte_carlo_sims):
            # Generate random pairs
            random_pairs = []
            for _ in range(len(phi_pairs)):
                i = np.random.randint(0, max_ell)
                j = np.random.randint(0, max_ell)
                if i != j:
                    random_pairs.append((min(i, j), max(i, j)))
            
            # Calculate clustering coefficient
            random_clustering = len(set(random_pairs)) / (max_ell * (max_ell - 1) / 2)
            random_clusterings.append(random_clustering)
        
        # Calculate significance
        mean_random = np.mean(random_clusterings)
        std_random = np.std(random_clusterings)
        
        z_score = (clustering - mean_random) / std_random
        p_value = 1 - stats.norm.cdf(z_score)
        
        return clustering, mean_random, z_score, p_value
    
    def run_spectral_gap_test(self):
        """Run the Spectral Gap Test."""
        print("Running Spectral Gap Test...")
        
        # Analyze the distribution of gaps in the power spectrum
        # Calculate the differences between adjacent power values
        power_diffs = np.diff(self.ee_power)
        
        # Normalize by the mean power
        norm_diffs = power_diffs / self.mean_power
        
        # Calculate the mean absolute difference
        mean_diff = np.mean(np.abs(norm_diffs))
        
        # Monte Carlo: Compare with random expectation
        print("Running Monte Carlo simulations...")
        random_diffs = []
        
        for _ in range(self.monte_carlo_sims):
            # Generate random power spectrum by shuffling the actual power values
            random_power = np.random.permutation(self.ee_power)
            
            # Calculate differences
            random_power_diffs = np.diff(random_power)
            
            # Normalize
            random_norm_diffs = random_power_diffs / self.mean_power
            
            # Calculate mean absolute difference
            random_mean_diff = np.mean(np.abs(random_norm_diffs))
            random_diffs.append(random_mean_diff)
        
        # Calculate significance
        mean_random = np.mean(random_diffs)
        std_random = np.std(random_diffs)
        
        z_score = (mean_diff - mean_random) / std_random
        p_value = 1 - stats.norm.cdf(z_score)
        
        return mean_diff, mean_random, z_score, p_value
    
    def run_multi_scale_coherence_test(self):
        """Run the Multi-Scale Coherence Test."""
        print("Running Multi-Scale Coherence Test...")
        
        # Analyze coherence across multiple scales
        # We'll use a simplified approach for this test
        
        # Define scales to analyze
        scales = [10, 20, 50, 100, 200, 500]
        
        # Find closest multipoles in our data
        scale_indices = [np.abs(self.ell - scale).argmin() for scale in scales]
        scale_powers = [self.ee_power[i] for i in scale_indices]
        
        # Calculate coherence as the correlation between powers at different scales
        coherence_scores = []
        
        for i in range(len(scales)):
            for j in range(i+1, len(scales)):
                power_i = scale_powers[i]
                power_j = scale_powers[j]
                
                # Normalize by mean power
                norm_power_i = power_i / self.mean_power
                norm_power_j = power_j / self.mean_power
                
                # Calculate coherence (inverse of absolute difference)
                coherence = 1.0 / (abs(norm_power_i - norm_power_j) + 0.01)
                coherence_scores.append(coherence)
        
        # Calculate mean coherence
        mean_coherence = np.mean(coherence_scores)
        
        # Monte Carlo: Compare with random expectation
        print("Running Monte Carlo simulations...")
        random_coherences = []
        
        for _ in range(self.monte_carlo_sims):
            random_scores = []
            
            # Generate random coherence tests
            for _ in range(len(coherence_scores)):
                # Pick random indices
                idx1 = np.random.randint(0, len(self.ell))
                idx2 = np.random.randint(0, len(self.ell))
                
                # Calculate coherence
                power1 = self.ee_power[idx1]
                power2 = self.ee_power[idx2]
                
                # Normalize by mean power
                norm1 = power1 / self.mean_power
                norm2 = power2 / self.mean_power
                
                # Calculate coherence
                coherence = 1.0 / (abs(norm1 - norm2) + 0.01)
                random_scores.append(coherence)
            
            random_coherences.append(np.mean(random_scores))
        
        # Calculate significance
        mean_random = np.mean(random_coherences)
        std_random = np.std(random_coherences)
        
        z_score = (mean_coherence - mean_random) / std_random
        p_value = 1 - stats.norm.cdf(z_score)
        
        return mean_coherence, mean_random, z_score, p_value

def save_results(test_name, result, results_dir):
    """Save test results to file and create visualization."""
    if result is None:
        print(f"No results to save for {test_name}")
        return
    
    test_dir = os.path.join(results_dir, test_name.lower().replace(" ", "_"))
    os.makedirs(test_dir, exist_ok=True)
    
    # Save results to file
    with open(os.path.join(test_dir, "results.txt"), "w") as f:
        f.write(f"{test_name} Results\n")
        f.write("="*len(test_name + " Results") + "\n\n")
        
        if isinstance(result, tuple):
            if len(result) >= 4 and all(isinstance(x, (int, float)) for x in result[:4]):
                # Assume standard format: metric, random_metric, z_score, p_value
                metric, random_metric, z_score, p_value = result[:4]
                f.write(f"Metric Value: {metric:.6f}\n")
                f.write(f"Random Expectation: {random_metric:.6f}\n")
                f.write(f"Z-Score: {z_score:.6f}\n")
                f.write(f"P-Value: {p_value:.6f}\n")
                f.write(f"Significant: {p_value < 0.05}\n")
                
                # Calculate phi-optimality
                if p_value < 1e-10:
                    phi_optimality = 1.0
                elif p_value > 0.9:
                    phi_optimality = -1.0
                else:
                    phi_optimality = 1.0 - 2.0 * p_value
                f.write(f"Phi-Optimality: {phi_optimality:.6f}\n")
                
                # Create visualization
                plt.figure(figsize=(10, 6))
                plt.bar(['Test Metric', 'Random Expectation'], 
                       [metric, random_metric],
                       color=['gold', 'gray'])
                plt.ylabel('Metric Value')
                plt.title(f'{test_name} Test Results')
                plt.annotate(f'p-value: {p_value:.6f}', xy=(0.5, 0.9), 
                            xycoords='axes fraction', ha='center')
                plt.annotate(f'Significant: {p_value < 0.05}', xy=(0.5, 0.85), 
                            xycoords='axes fraction', ha='center')
                plt.tight_layout()
                plt.savefig(os.path.join(test_dir, f'{test_name.lower().replace(" ", "_")}.png'))
                plt.close()
            else:
                # Generic tuple result
                for i, value in enumerate(result):
                    f.write(f"Result {i+1}: {value}\n")
        else:
            # Single value result
            f.write(f"Result: {result}\n")
    
    print(f"Results saved to {test_dir}/results.txt")

def main():
    # Check for test number argument
    if len(sys.argv) < 2:
        print("Usage: python3 simplified_test_runner.py <test_number>")
        print("Available tests:")
        print("1. Cross-Scale Correlations")
        print("2. Optimization")
        print("3. Golden Symmetries")
        print("4. Phi Network")
        print("5. Spectral Gap")
        print("6. Multi-Scale Coherence")
        return
    
    try:
        test_number = int(sys.argv[1])
    except ValueError:
        print("Please provide a valid test number (1-6)")
        return
    
    # Define the tests to run
    tests = [
        "Cross-Scale Correlations",
        "Optimization",
        "Golden Symmetries",
        "Phi Network",
        "Spectral Gap",
        "Multi-Scale Coherence"
    ]
    
    if test_number < 1 or test_number > len(tests):
        print(f"Please provide a valid test number (1-{len(tests)})")
        return
    
    # Get the selected test
    test_name = tests[test_number - 1]
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("results", f"test_{test_number}_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize the test runner
    print("Initializing SimplifiedTestRunner...")
    runner = SimplifiedTestRunner(monte_carlo_sims=20)  # Reduced for faster testing
    
    # Run the test
    try:
        print(f"Running {test_name} test...")
        
        # Call the appropriate test method
        if test_number == 1:
            result = runner.run_cross_scale_correlations_test()
        elif test_number == 2:
            result = runner.run_optimization_test()
        elif test_number == 3:
            result = runner.run_golden_symmetries_test()
        elif test_number == 4:
            result = runner.run_phi_network_test()
        elif test_number == 5:
            result = runner.run_spectral_gap_test()
        elif test_number == 6:
            result = runner.run_multi_scale_coherence_test()
        
        # Save results
        save_results(test_name, result, results_dir)
        
        print(f"{test_name} test completed successfully")
        if isinstance(result, tuple) and len(result) >= 4:
            print(f"P-Value: {result[3]:.6f}")
            print(f"Significant: {result[3] < 0.05}")
    except Exception as e:
        print(f"Error running {test_name} test: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
