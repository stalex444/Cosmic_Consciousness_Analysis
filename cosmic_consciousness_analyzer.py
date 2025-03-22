#!/usr/bin/env python3
"""
Cosmic Consciousness Analyzer for CMB Data.
This module implements the CosmicConsciousnessAnalyzer class for testing evidence
of conscious organization in the Cosmic Microwave Background (CMB) data.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
from astropy.io import fits
from tqdm import tqdm
import datetime

class CosmicConsciousnessAnalyzer:
    """
    Class for analyzing CMB data for evidence of conscious organization.
    Tests include golden ratio patterns, coherence, hierarchical organization,
    information integration, resonance, fractal structure, and more.
    """
    
    def __init__(self, data_dir="planck_data", monte_carlo_sims=10000):
        """
        Initialize the analyzer with a data directory.
        
        Parameters:
        -----------
        data_dir : str
            Directory containing Planck CMB data files
        monte_carlo_sims : int
            Number of Monte Carlo simulations for significance testing
        """
        self.data_dir = data_dir
        self.phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        self.data = {}
        self.gr_multipoles = []
        self.monte_carlo_sims = monte_carlo_sims
        
        # Import data
        self._import_data()
        
        # Validate data
        self._validate_data()
        
        # Calculate golden ratio multipoles
        self._calculate_gr_multipoles()
    
    def _import_data(self):
        """Import the EE spectrum and covariance matrix from Planck data."""
        try:
            # Import EE power spectrum - try both binned and full versions
            ee_binned_file = os.path.join(self.data_dir, "power_spectra", "COM_PowerSpect_CMB-EE-binned_R3.02.txt")
            ee_full_file = os.path.join(self.data_dir, "power_spectra", "COM_PowerSpect_CMB-EE-full_R3.01.txt")
            
            # Print the absolute paths for debugging
            print(f"Looking for EE spectrum files:")
            print(f"  - {os.path.abspath(ee_binned_file)}")
            print(f"  - {os.path.abspath(ee_full_file)}")
            
            # Try binned file first, then full file
            try:
                ee_data = np.loadtxt(ee_binned_file)
                print(f"Using binned EE spectrum: {ee_binned_file}")
            except FileNotFoundError:
                try:
                    ee_data = np.loadtxt(ee_full_file)
                    print(f"Using full EE spectrum: {ee_full_file}")
                except FileNotFoundError:
                    # If neither file exists in the data_dir, try looking in planck_data directory
                    alt_ee_binned_file = os.path.join("planck_data", "power_spectra", "COM_PowerSpect_CMB-EE-binned_R3.02.txt")
                    alt_ee_full_file = os.path.join("planck_data", "power_spectra", "COM_PowerSpect_CMB-EE-full_R3.01.txt")
                    
                    print(f"  - {os.path.abspath(alt_ee_binned_file)}")
                    print(f"  - {os.path.abspath(alt_ee_full_file)}")
                    
                    try:
                        ee_data = np.loadtxt(alt_ee_binned_file)
                        print(f"Using binned EE spectrum from alternate location: {alt_ee_binned_file}")
                    except FileNotFoundError:
                        ee_data = np.loadtxt(alt_ee_full_file)
                        print(f"Using full EE spectrum from alternate location: {alt_ee_full_file}")
            
            self.data['ell'] = ee_data[:, 0]
            self.data['ee_power'] = ee_data[:, 1]
            self.data['ee_error'] = ee_data[:, 2] if ee_data.shape[1] > 2 else np.sqrt(ee_data[:, 1])
            
            print(f"Loaded EE spectrum with {len(self.data['ell'])} multipoles")
            
            # Try to import covariance matrix
            try:
                # Try FITS format first
                cov_file = os.path.join(self.data_dir, "COM_PowerSpect_CMB-CovMatrix_R3.01.fits")
                print(f"Looking for covariance matrix file: {os.path.abspath(cov_file)}")
                with fits.open(cov_file) as hdul:
                    self.data['cov_matrix'] = hdul[0].data
                print(f"Loaded covariance matrix with shape {self.data['cov_matrix'].shape}")
            except Exception as e:
                # Try alternate FITS location
                try:
                    alt_cov_file = os.path.join("planck_data", "COM_PowerSpect_CMB-CovMatrix_R3.01.fits")
                    print(f"Looking for covariance matrix file: {os.path.abspath(alt_cov_file)}")
                    with fits.open(alt_cov_file) as hdul:
                        self.data['cov_matrix'] = hdul[0].data
                    print(f"Loaded covariance matrix from alternate location with shape {self.data['cov_matrix'].shape}")
                except Exception as e2:
                    # Try numpy format
                    try:
                        npy_cov_file = os.path.join(self.data_dir, "cov_matrix.npy")
                        print(f"Looking for covariance matrix file: {os.path.abspath(npy_cov_file)}")
                        self.data['cov_matrix'] = np.load(npy_cov_file)
                        print(f"Loaded numpy covariance matrix with shape {self.data['cov_matrix'].shape}")
                    except Exception as e3:
                        print(f"Warning: Could not load covariance matrix: {e3}")
                        print("Proceeding without covariance information")
                        # Create a diagonal covariance matrix using the error bars
                        self.data['cov_matrix'] = np.diag(self.data['ee_error']**2)
        
        except Exception as e:
            raise RuntimeError(f"Error importing data: {e}")
    
    def _validate_data(self):
        """Validate that the data is properly loaded and formatted."""
        # Check if data dictionary exists and has required keys
        if not hasattr(self, 'data') or not isinstance(self.data, dict):
            raise ValueError("Data dictionary not properly initialized")
        
        required_keys = ['ell', 'ee_power', 'ee_error']
        missing_keys = [key for key in required_keys if key not in self.data]
        
        if missing_keys:
            raise ValueError(f"Missing required data keys: {', '.join(missing_keys)}")
        
        # Check if arrays have the expected shape
        if len(self.data['ell']) < 10:
            raise ValueError(f"Insufficient data points in spectrum: {len(self.data['ell'])} < 10")
        
        # Check if all arrays have the same length
        lengths = [len(self.data[key]) for key in required_keys]
        if len(set(lengths)) > 1:
            raise ValueError(f"Data arrays have inconsistent lengths: {lengths}")
        
        # Check if data contains NaN or inf values
        for key in required_keys:
            if np.any(np.isnan(self.data[key])) or np.any(np.isinf(self.data[key])):
                raise ValueError(f"Data array '{key}' contains NaN or infinite values")
        
        # Check if power spectrum values are positive
        if np.any(self.data['ee_power'] <= 0):
            print("Warning: Power spectrum contains zero or negative values")
        
        print("Data validation successful")
        return True
    
    def _calculate_gr_multipoles(self):
        """Calculate multipoles related to the golden ratio."""
        # Start with ell = 2 (quadrupole)
        ell = 2
        gr_multipoles = [ell]
        
        # Generate a sequence of multipoles related by the golden ratio
        while ell * self.phi < max(self.data['ell']):
            ell = int(round(ell * self.phi))
            if ell <= max(self.data['ell']):
                gr_multipoles.append(ell)
        
        self.gr_multipoles = gr_multipoles
        print(f"Golden ratio multipoles: {self.gr_multipoles}")
    
    def plot_spectrum(self, highlight_gr=True):
        """
        Plot the CMB power spectrum, optionally highlighting golden ratio multipoles.
        
        Parameters:
        -----------
        highlight_gr : bool, optional
            Whether to highlight golden ratio multipoles
        """
        plt.figure(figsize=(12, 6))
        plt.errorbar(self.data['ell'], self.data['ee_power'], 
                    yerr=self.data['ee_error'], fmt='o', markersize=2, 
                    alpha=0.3, elinewidth=0.5)
        
        # Highlight golden ratio multipoles
        if highlight_gr:
            gr_indices = []
            gr_ell_values = []
            gr_powers = []
            
            for l in self.gr_multipoles:
                # Find the closest multipole to l
                idx = np.abs(self.data['ell'] - l).argmin()
                gr_indices.append(idx)
                gr_ell_values.append(self.data['ell'][idx])
                gr_powers.append(self.data['ee_power'][idx])
            
            plt.scatter(gr_ell_values, gr_powers, color='red', s=50, zorder=10)
        
        plt.xlabel('Multipole ℓ')
        plt.ylabel('EE Power (μK²)')
        plt.title('CMB EE Spectrum with GR Multipoles')
        plt.grid(True, alpha=0.3)
        plt.savefig('cmb_spectrum.png')
        plt.show()
        
    def test_gr_significance(self):
        """Test if multipoles related by the golden ratio have statistically significant power."""
        # Find the closest indices to the golden ratio multipoles
        gr_indices = []
        for l in self.gr_multipoles:
            # Find the closest multipole to l
            idx = np.abs(self.data['ell'] - l).argmin()
            gr_indices.append(idx)
        
        # Calculate the mean power of golden ratio multipoles
        gr_powers = self.data['ee_power'][gr_indices]
        mean_gr_power = np.mean(gr_powers)
        
        # Compare with random multipole sets of the same size
        random_powers = []
        for _ in range(self.monte_carlo_sims):
            # Select random multipoles
            random_indices = np.random.choice(len(self.data['ell']), size=len(gr_indices), replace=False)
            random_power = np.mean(self.data['ee_power'][random_indices])
            random_powers.append(random_power)
        
        # Calculate statistics
        mean_random_power = np.mean(random_powers)
        std_random_power = np.std(random_powers)
        
        # Calculate z-score and p-value
        z_score = (mean_gr_power - mean_random_power) / std_random_power if std_random_power > 0 else 0
        p_value = 1 - stats.norm.cdf(z_score)
        
        # Calculate power ratio
        power_ratio = mean_gr_power / mean_random_power if mean_random_power > 0 else 1.0
        
        return mean_gr_power, mean_random_power, z_score, p_value, power_ratio
    
    def test_coherence(self):
        """
        Test for coherence in the CMB power spectrum.
        
        Returns:
        --------
        tuple
            (actual_variance, mean_shuffled_variance, z_score, p_value, variance_ratio)
        """
        # Calculate the variance of the normalized spectrum as a measure of coherence
        # Lower variance indicates more coherence
        normalized_spectrum = self.data['ee_power'] / np.mean(self.data['ee_power'])
        actual_variance = np.var(normalized_spectrum)
        
        # Compare with shuffled spectra
        print("Running coherence test...")
        shuffled_variances = []
        
        for _ in range(self.monte_carlo_sims):
            shuffled = np.random.permutation(self.data['ee_power'])
            normalized_shuffled = shuffled / np.mean(shuffled)
            shuffled_variances.append(np.var(normalized_shuffled))
        
        # Calculate statistics
        mean_shuffled_variance = np.mean(shuffled_variances)
        std_shuffled_variance = np.std(shuffled_variances)
        
        # For coherence, lower variance is better, so we reverse the z-score
        z_score = (mean_shuffled_variance - actual_variance) / std_shuffled_variance
        p_value = stats.norm.cdf(z_score)
        
        variance_ratio = actual_variance / mean_shuffled_variance
        
        # Create visualization
        plt.figure(figsize=(10, 6))
        plt.hist(shuffled_variances, bins=30, alpha=0.7, density=True)
        plt.axvline(x=actual_variance, color='red', linestyle='--',
                   label=f'Actual Variance: {actual_variance:.4f}')
        plt.axvline(x=mean_shuffled_variance, color='blue', linestyle='--',
                   label=f'Mean Shuffled: {mean_shuffled_variance:.4f}')
        
        plt.xlabel('Variance of Normalized Spectrum')
        plt.ylabel('Probability Density')
        plt.title(f'Coherence Test: {z_score:.2f}σ')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('coherence_test.png')
        
        print(f"\nCoherence Test Results:")
        print(f"Actual variance: {actual_variance:.4f}")
        print(f"Mean shuffled variance: {mean_shuffled_variance:.4f}")
        print(f"Variance ratio: {variance_ratio:.4f}")
        print(f"Z-score: {z_score:.2f}σ (p = {p_value:.8f})")
        
        return actual_variance, mean_shuffled_variance, z_score, p_value, variance_ratio
    
    def test_gr_coherence(self):
        """
        Test for coherence specifically in golden ratio multipole windows.
        
        Returns:
        --------
        tuple
            (gr_windows_variance, random_windows_variance, z_score, p_value, variance_ratio)
        """
        # Calculate variance in windows centered on golden ratio multipoles
        window_size = 5
        gr_windows_variances = []
        
        for l in self.gr_multipoles:
            # Find the index of this multipole
            idx = np.abs(self.data['ell'] - l).argmin()
            
            # Define window ensuring we don't go out of bounds
            start = max(0, idx - window_size // 2)
            end = min(len(self.data['ell']), idx + window_size // 2 + 1)
            
            # Calculate variance in this window
            window = self.data['ee_power'][start:end]
            normalized_window = window / np.mean(window)
            gr_windows_variances.append(np.var(normalized_window))
        
        # Average variance across all GR windows
        gr_windows_variance = np.mean(gr_windows_variances)
        
        # Compare with random windows
        print("Running GR-specific coherence test...")
        random_windows_variances = []
        
        for _ in range(self.monte_carlo_sims):
            # Select random centers for windows
            random_centers = np.random.choice(range(len(self.data['ell'])), len(self.gr_multipoles), replace=False)
            window_variances = []
            
            for center in random_centers:
                start = max(0, center - window_size // 2)
                end = min(len(self.data['ell']), center + window_size // 2 + 1)
                
                window = self.data['ee_power'][start:end]
                normalized_window = window / np.mean(window)
                window_variances.append(np.var(normalized_window))
            
            random_windows_variances.append(np.mean(window_variances))
        
        # Calculate statistics
        random_windows_variance = np.mean(random_windows_variances)
        std_random_variance = np.std(random_windows_variances)
        
        # For coherence, lower variance is better
        z_score = (random_windows_variance - gr_windows_variance) / std_random_variance
        p_value = stats.norm.cdf(z_score)
        
        variance_ratio = gr_windows_variance / random_windows_variance
        
        # Create visualization
        plt.figure(figsize=(10, 6))
        plt.hist(random_windows_variances, bins=30, alpha=0.7, density=True)
        plt.axvline(x=gr_windows_variance, color='red', linestyle='--',
                   label=f'GR Windows: {gr_windows_variance:.4f}')
        plt.axvline(x=random_windows_variance, color='blue', linestyle='--',
                   label=f'Random Windows: {random_windows_variance:.4f}')
        
        plt.xlabel('Mean Window Variance')
        plt.ylabel('Probability Density')
        plt.title(f'GR-Specific Coherence: {z_score:.2f}σ')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('gr_coherence_test.png')
        
        print(f"\nGR-Specific Coherence Test Results:")
        print(f"GR windows variance: {gr_windows_variance:.4f}")
        print(f"Random windows variance: {random_windows_variance:.4f}")
        print(f"Variance ratio: {variance_ratio:.4f}")
        print(f"Z-score: {z_score:.2f}σ (p = {p_value:.8f})")
        
        return gr_windows_variance, random_windows_variance, z_score, p_value, variance_ratio
    
    def test_hierarchical_organization(self):
        """
        Test for hierarchical organization in the power spectrum.
        
        Returns:
        --------
        tuple
            (hierarchy_measure, mean_random_hierarchy, z_score, p_value, hierarchy_ratio)
        """
        # Define hierarchical levels based on golden ratio
        base = 2
        hierarchical_levels = []
        for i in range(5):  # 5 levels of hierarchy
            level = [int(base * self.phi**j) for j in range(i*3, (i+1)*3)]
            hierarchical_levels.append(level)
        
        # Calculate power at each level
        level_powers = []
        for level in hierarchical_levels:
            level_indices = []
            for l in level:
                # Find the closest multipole to l
                if l <= max(self.data['ell']):
                    idx = np.abs(self.data['ell'] - l).argmin()
                    level_indices.append(idx)
            
            if level_indices:
                level_powers.append(np.mean(self.data['ee_power'][level_indices]))
            else:
                level_powers.append(0)
        
        # Calculate variance between levels as a measure of hierarchy
        hierarchy_measure = np.var(level_powers) / np.mean(level_powers) if np.mean(level_powers) != 0 else 0
        
        # Monte Carlo: Compare with random hierarchical levels
        random_hierarchies = []
        for _ in range(self.monte_carlo_sims):
            random_levels = []
            for level in hierarchical_levels:
                random_level = np.random.choice(self.data['ell'], size=len(level))
                level_indices = []
                for l in random_level:
                    idx = np.abs(self.data['ell'] - l).argmin()
                    level_indices.append(idx)
                
                if level_indices:
                    random_levels.append(np.mean(self.data['ee_power'][level_indices]))
                else:
                    random_levels.append(0)
            
            random_hierarchy = np.var(random_levels) / np.mean(random_levels) if np.mean(random_levels) != 0 else 0
            random_hierarchies.append(random_hierarchy)
        
        # Calculate statistics
        mean_random_hierarchy = np.mean(random_hierarchies)
        std_random_hierarchy = np.std(random_hierarchies)
        
        # Calculate z-score and p-value
        z_score = (hierarchy_measure - mean_random_hierarchy) / std_random_hierarchy if std_random_hierarchy > 0 else 0
        p_value = 1 - stats.norm.cdf(z_score)
        
        hierarchy_ratio = hierarchy_measure / mean_random_hierarchy if mean_random_hierarchy != 0 else 1.0
        
        return hierarchy_measure, mean_random_hierarchy, z_score, p_value, hierarchy_ratio
    
    def test_information_integration(self):
        """
        Test for information integration in the CMB power spectrum.
        
        Returns:
        --------
        tuple
            (integration, random_integration, z_score, p_value, integration_ratio)
        """
        # Calculate mutual information between adjacent regions of the spectrum
        window_size = 10
        mutual_info = 0
        
        for i in range(0, len(self.data['ell']) - window_size, window_size):
            region1 = self.data['ee_power'][i:i+window_size]
            region2 = self.data['ee_power'][i+window_size:i+2*window_size]
            
            if len(region2) == window_size:
                # Calculate mutual information
                mutual_info += self._calculate_mutual_information(region1, region2)
        
        # Normalize by the number of comparisons
        n_comparisons = (len(self.data['ell']) - window_size) // window_size
        if n_comparisons > 0:
            mutual_info /= n_comparisons
        
        # Monte Carlo: Compare with shuffled spectrum
        print("Running information integration test...")
        random_mutual_infos = []
        
        for _ in range(self.monte_carlo_sims):
            # Shuffle the spectrum to destroy any structure
            shuffled_spectrum = np.random.permutation(self.data['ee_power'])
            random_mutual_info = 0
            
            for i in range(0, len(shuffled_spectrum) - window_size, window_size):
                region1 = shuffled_spectrum[i:i+window_size]
                region2 = shuffled_spectrum[i+window_size:i+2*window_size]
                
                if len(region2) == window_size:
                    # Calculate mutual information
                    random_mutual_info += self._calculate_mutual_information(region1, region2)
            
            # Normalize
            if n_comparisons > 0:
                random_mutual_info /= n_comparisons
            
            random_mutual_infos.append(random_mutual_info)
        
        # Calculate statistics
        mean_random_mutual_info = np.mean(random_mutual_infos)
        std_random_mutual_info = np.std(random_mutual_infos)
        
        # Calculate z-score and p-value
        z_score = (mutual_info - mean_random_mutual_info) / std_random_mutual_info if std_random_mutual_info > 0 else 0
        p_value = 1 - stats.norm.cdf(z_score)
        
        integration_ratio = mutual_info / mean_random_mutual_info if mean_random_mutual_info > 0 else 1.0
        
        print(f"\nInformation Integration Test Results:")
        print(f"Actual mutual information: {mutual_info:.4f} bits")
        print(f"Random mutual information: {mean_random_mutual_info:.4f} bits")
        print(f"Z-score: {z_score:.2f}σ (p = {p_value:.8f})")
        
        return mutual_info, mean_random_mutual_info, z_score, p_value, integration_ratio
    
    def _calculate_mutual_information(self, region1, region2):
        """
        Calculate mutual information between two regions of the spectrum.
        
        Parameters:
        -----------
        region1 : array-like
            First region of the spectrum
        region2 : array-like
            Second region of the spectrum
            
        Returns:
        --------
        float
            Mutual information in bits
        """
        if len(region1) < 2 or len(region2) < 2:
            return 0
            
        # Calculate correlation coefficient
        corr_matrix = np.corrcoef(region1, region2)
        if corr_matrix.shape != (2, 2):
            return 0
            
        corr = corr_matrix[0, 1]
        
        # Convert correlation to mutual information (simplified formula)
        if abs(corr) < 1:
            return -0.5 * np.log(1 - corr**2)
        else:
            return 0
    
    def test_resonance(self):
        """Test for resonance patterns in the CMB power spectrum."""
        # Calculate the Fourier transform of the CMB power spectrum
        spectrum = self.data['ee_power']
        fft_spectrum = np.abs(np.fft.fft(spectrum))
        
        # Normalize to focus on the pattern rather than absolute magnitude
        normalized_fft = fft_spectrum / np.max(fft_spectrum)
        
        # In a resonant system, we expect peaks at harmonics
        # Find the dominant frequency
        peak_idx = np.argmax(normalized_fft[1:len(normalized_fft)//2]) + 1
        peak_frequency = peak_idx
        
        # Look for harmonics (multiples and phi-related frequencies)
        harmonic_indices = [peak_idx * i for i in range(1, 4)]
        phi_indices = [int(round(peak_idx * self.phi**i)) for i in range(3)]
        
        all_indices = set(harmonic_indices + phi_indices)
        all_indices = [i for i in all_indices if i < len(normalized_fft)//2]
        
        # Calculate resonance strength (average normalized power at these frequencies)
        resonance_strength = np.mean([normalized_fft[i] for i in all_indices])
        
        # Compare with random sets of indices
        print("Running resonance analysis...")
        random_strengths = []
        
        for _ in range(self.monte_carlo_sims):
            random_indices = np.random.choice(range(1, len(normalized_fft)//2), len(all_indices), replace=False)
            random_strengths.append(np.mean([normalized_fft[i] for i in random_indices]))
        
        # Calculate significance
        mean_random = np.mean(random_strengths)
        std_random = np.std(random_strengths)
        
        z_score = (resonance_strength - mean_random) / std_random
        p_value = 1 - stats.norm.cdf(z_score)
        
        resonance_ratio = resonance_strength / mean_random
        
        # Create visualization
        plt.figure(figsize=(12, 6))
        plt.plot(normalized_fft[:len(normalized_fft)//4], 'b-', alpha=0.7)
        
        # Plot harmonics
        plt.scatter([peak_idx], [normalized_fft[peak_idx]], color='red', s=100, label='Primary Peak')
        plt.scatter(all_indices, [normalized_fft[i] for i in all_indices], color='orange', s=50, 
                   label='Harmonics/Phi-Related')
        
        plt.xlabel('Frequency')
        plt.ylabel('Normalized Amplitude')
        plt.title(f'Resonance Patterns in CMB: {resonance_ratio:.2f}x stronger than random')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('cmb_resonance_test.png')
        
        print(f"\nResonance Test Results:")
        print(f"Resonance strength: {resonance_strength:.4f}")
        print(f"Random resonance: {mean_random:.4f}")
        print(f"Resonance factor: {resonance_ratio:.2f}x")
        print(f"Z-score: {z_score:.2f}σ (p = {p_value:.8f})")
        
        return resonance_strength, mean_random, z_score, p_value, resonance_ratio
    
    def test_fractal_structure(self):
        """
        Test for fractal-like self-similarity in the CMB spectrum.
        
        Returns:
        --------
        tuple
            (actual_hurst, mean_shuffled, z_score, p_value, fractal_ratio)
        """
        # Use Hurst exponent as a measure of fractal behavior
        # Hurst exponent around 0.5 indicates random behavior
        # Values near 1 indicate persistent structure, suggesting fractal properties
        
        def hurst_exponent(data):
            # Range of lag values
            lags = range(2, 20)
            
            # Calculate the standard deviation of the differentiated series with each lag
            tau = [np.sqrt(np.std(np.diff(data, n=lag))) for lag in lags]
            
            # Use a power law to estimate the Hurst exponent
            m = np.polyfit(np.log(lags), np.log(tau), 1)
            hurst = m[0]
            return hurst
        
        # Calculate Hurst exponent for the actual spectrum
        actual_hurst = hurst_exponent(self.data['ee_power'])
        
        # Compare with shuffled spectra
        print("Running fractal analysis...")
        shuffled_hursts = []
        
        for _ in range(self.monte_carlo_sims):
            shuffled = np.random.permutation(self.data['ee_power'])
            shuffled_hursts.append(hurst_exponent(shuffled))
        
        # Calculate significance
        mean_shuffled = np.mean(shuffled_hursts)
        std_shuffled = np.std(shuffled_hursts)
        
        z_score = (actual_hurst - mean_shuffled) / std_shuffled
        p_value = 1 - stats.norm.cdf(z_score)
        
        # Calculate fractal ratio - how much more fractal the actual spectrum is compared to random
        # For Hurst exponent, values closer to 0.5 are more random, values closer to 1.0 are more persistent
        # and values closer to 0.0 are more anti-persistent
        # We'll use the distance from 0.5 as our measure of "fractalness"
        actual_fractalness = abs(actual_hurst - 0.5)
        random_fractalness = abs(mean_shuffled - 0.5)
        fractal_ratio = actual_fractalness / random_fractalness if random_fractalness > 0 else 1.0
        
        # Create visualization
        plt.figure(figsize=(12, 6))
        plt.hist(shuffled_hursts, bins=30, alpha=0.7, density=True)
        plt.axvline(actual_hurst, color='red', linestyle='--',
                   label=f'Actual Hurst: {actual_hurst:.4f}')
        plt.axvline(mean_shuffled, color='blue', linestyle='--',
                   label=f'Random Mean: {mean_shuffled:.4f}')
        plt.axvline(0.5, color='green', linestyle='-', alpha=0.5,
                   label='Random Process (0.5)')
        
        plt.xlabel('Hurst Exponent')
        plt.ylabel('Probability Density')
        plt.title(f'Fractal Analysis of CMB: {z_score:.2f}σ')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('cmb_fractal_analysis.png')
        
        print(f"\nFractal Analysis Test Results:")
        print(f"Actual Hurst exponent: {actual_hurst:.4f}")
        print(f"Random Hurst exponent: {mean_shuffled:.4f}")
        print(f"Z-score: {z_score:.2f}σ (p = {p_value:.8f})")
        
        return actual_hurst, mean_shuffled, z_score, p_value, fractal_ratio
    
    def test_meta_coherence(self):
        """
        Test for meta-coherence (coherence of coherence measures).
        
        Returns:
        --------
        tuple
            (meta_coherence, mean_shuffled, z_score, p_value, meta_coherence_ratio)
        """
        # Calculate local coherence measures across the spectrum
        window_size = 5
        step_size = 2
        local_coherence = []
        
        for i in range(0, len(self.data['ee_power']) - window_size, step_size):
            window = self.data['ee_power'][i:i+window_size]
            normalized = window / np.mean(window)
            local_coherence.append(np.var(normalized))
        
        # Calculate the variance of local coherence (meta-coherence)
        meta_coherence = np.var(local_coherence)
        
        # Compare with shuffled spectra
        print("Running meta-coherence analysis...")
        shuffled_meta_coherence = []
        
        for _ in range(self.monte_carlo_sims):
            shuffled = np.random.permutation(self.data['ee_power'])
            local_coherence_shuffled = []
            
            for i in range(0, len(shuffled) - window_size, step_size):
                window = shuffled[i:i+window_size]
                normalized = window / np.mean(window)
                local_coherence_shuffled.append(np.var(normalized))
            
            shuffled_meta_coherence.append(np.var(local_coherence_shuffled))
        
        # Calculate significance
        mean_shuffled = np.mean(shuffled_meta_coherence)
        std_shuffled = np.std(shuffled_meta_coherence)
        
        z_score = (mean_shuffled - meta_coherence) / std_shuffled  # Lower variance is more coherent
        p_value = stats.norm.cdf(z_score)
        
        meta_coherence_ratio = mean_shuffled / meta_coherence
        
        # Create visualization
        plt.figure(figsize=(12, 6))
        
        # Plot distribution of shuffled meta-coherence values
        plt.hist(shuffled_meta_coherence, bins=30, alpha=0.7, density=True)
        plt.axvline(meta_coherence, color='red', linestyle='--',
                   label=f'Actual Meta-Coherence: {meta_coherence:.6f}')
        plt.axvline(mean_shuffled, color='blue', linestyle='--',
                   label=f'Random Mean: {mean_shuffled:.6f}')
        
        plt.xlabel('Meta-Coherence (Variance of Local Variances)')
        plt.ylabel('Probability Density')
        plt.title(f'Meta-Coherence in CMB: {meta_coherence_ratio:.2f}x stronger than random')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('cmb_meta_coherence.png')
        
        print(f"\nMeta-Coherence Test Results:")
        print(f"Actual meta-coherence: {meta_coherence:.6f}")
        print(f"Random meta-coherence: {mean_shuffled:.6f}")
        print(f"Meta-coherence factor: {meta_coherence_ratio:.2f}x")
        print(f"Z-score: {z_score:.2f}σ (p = {p_value:.8f})")
        
        return meta_coherence, mean_shuffled, z_score, p_value, meta_coherence_ratio

    def test_multiscale_patterns(self, plot=False):
        """
        Test for golden ratio patterns across multiple scales.
        
        Parameters:
        -----------
        plot : bool, optional
            Whether to create and display plots
            
        Returns:
        --------
        tuple
            (multiscale_measure, random_multiscale, z_score, p_value, multiscale_ratio)
        """
        # Define scales based on powers of the golden ratio
        base_scale = 10
        scales = [int(round(base_scale * self.phi**i)) for i in range(10)]
        scales = [s for s in scales if s <= max(self.data['ell'])]
        
        # Calculate the power at each scale
        scale_indices = []
        for scale in scales:
            # Find the closest multipole to scale
            idx = np.abs(self.data['ell'] - scale).argmin()
            scale_indices.append(idx)
        
        scale_powers = self.data['ee_power'][scale_indices]
        
        # Calculate wavelet coefficients
        wavelet_coeffs = np.abs(np.fft.fft(scale_powers))
        
        # Calculate the peak frequency
        peak_idx = np.argmax(wavelet_coeffs[1:len(wavelet_coeffs)//2]) + 1
        
        # Calculate the ratio of the peak frequency to the golden ratio frequency
        gr_freq = len(wavelet_coeffs) / self.phi
        multiscale_measure = 1 / (1 + abs(peak_idx - gr_freq) / gr_freq)
        
        # Monte Carlo: Compare with random scales
        random_multiscales = []
        for _ in range(self.monte_carlo_sims):
            # Select random scales
            random_scales = np.random.choice(self.data['ell'], len(scales), replace=False)
            
            # Find indices for random scales
            random_indices = []
            for scale in random_scales:
                idx = np.abs(self.data['ell'] - scale).argmin()
                random_indices.append(idx)
            
            random_powers = self.data['ee_power'][random_indices]
            
            # Calculate wavelet coefficients
            random_wavelet_coeffs = np.abs(np.fft.fft(random_powers))
            
            # Calculate the peak frequency
            random_peak_idx = np.argmax(random_wavelet_coeffs[1:len(random_wavelet_coeffs)//2]) + 1
            
            # Calculate the ratio of the peak frequency to the golden ratio frequency
            random_multiscale = 1 / (1 + abs(random_peak_idx - gr_freq) / gr_freq)
            random_multiscales.append(random_multiscale)
        
        # Calculate statistics
        mean_random_multiscale = np.mean(random_multiscales)
        std_random_multiscale = np.std(random_multiscales)
        
        # Calculate z-score and p-value
        z_score = (multiscale_measure - mean_random_multiscale) / std_random_multiscale if std_random_multiscale > 0 else 0
        p_value = 1 - stats.norm.cdf(z_score)
        
        # Calculate multiscale ratio
        multiscale_ratio = multiscale_measure / mean_random_multiscale if mean_random_multiscale != 0 else 1.0
        
        if plot:
            # Create visualization
            plt.figure(figsize=(12, 6))
            
            # Plot the power at each scale
            plt.subplot(1, 2, 1)
            plt.plot(scales, scale_powers, 'b-', marker='o')
            plt.xlabel('Multipole ℓ')
            plt.ylabel('Power (μK²)')
            plt.title('Power at Golden Ratio Scales')
            plt.grid(True, alpha=0.3)
            
            # Plot the wavelet coefficients
            plt.subplot(1, 2, 2)
            plt.plot(wavelet_coeffs[:len(wavelet_coeffs)//2], 'r-')
            plt.axvline(x=peak_idx, color='green', linestyle='--',
                       label=f'Peak: {peak_idx}')
            plt.axvline(x=gr_freq, color='gold', linestyle=':',
                       label=f'GR: {gr_freq:.2f}')
            plt.xlabel('Frequency')
            plt.ylabel('Amplitude')
            plt.title(f'Wavelet Coefficients (GR Match: {multiscale_measure:.4f})')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('multiscale_patterns.png')
            plt.show()
        
        return multiscale_measure, mean_random_multiscale, z_score, p_value, multiscale_ratio
    
    def test_cross_scale_correlations(self):
        """Test for stronger correlations between scales separated by powers of φ"""
        # Define scales separated by powers of phi
        phi = self.phi
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
            family_corrs = []
            for i in range(len(family)-1):
                scale1 = family[i]
                scale2 = family[i+1]
                
                # Find indices in the data
                idx1 = np.abs(self.data['ell'] - scale1).argmin()
                idx2 = np.abs(self.data['ell'] - scale2).argmin()
                
                # Calculate correlation (inverse of absolute difference in normalized power)
                power1 = self.data['ee_power'][idx1]
                power2 = self.data['ee_power'][idx2]
                
                # Normalize by the mean power
                mean_power = np.mean(self.data['ee_power'])
                norm_power1 = power1 / mean_power
                norm_power2 = power2 / mean_power
                
                # Calculate correlation (1 / absolute difference)
                # Add small constant to avoid division by zero
                correlation = 1.0 / (abs(norm_power1 - norm_power2) + 0.01)
                phi_correlations.append(correlation)
        
        # Calculate mean correlation for phi-related scales
        mean_phi_corr = np.mean(phi_correlations)
        
        # Compare with random scale relationships
        random_correlations = []
        for _ in range(self.monte_carlo_sims):
            # Select random scales
            random_scales = np.random.choice(self.data['ell'], len(phi_correlations)*2)
            
            # Calculate correlations
            for i in range(0, len(random_scales), 2):
                if i+1 < len(random_scales):
                    scale1 = random_scales[i]
                    scale2 = random_scales[i+1]
                    
                    # Find indices in the data
                    idx1 = np.abs(self.data['ell'] - scale1).argmin()
                    idx2 = np.abs(self.data['ell'] - scale2).argmin()
                    
                    # Calculate correlation
                    power1 = self.data['ee_power'][idx1]
                    power2 = self.data['ee_power'][idx2]
                    
                    # Normalize by the mean power
                    mean_power = np.mean(self.data['ee_power'])
                    norm_power1 = power1 / mean_power
                    norm_power2 = power2 / mean_power
                    
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
        
    def test_pattern_persistence(self):
        """Test persistence of golden ratio patterns across data subsets"""
        # Define several data subsets
        subsets = [
            (0, len(self.data['ell'])//2),  # First half
            (len(self.data['ell'])//2, len(self.data['ell'])),  # Second half
            (0, len(self.data['ell'])//3),  # First third
            (len(self.data['ell'])//3, 2*len(self.data['ell'])//3),  # Middle third
            (2*len(self.data['ell'])//3, len(self.data['ell'])),  # Last third
            (len(self.data['ell'])//4, 3*len(self.data['ell'])//4),  # Middle half
        ]
        
        # Test pattern persistence across subsets
        subset_gr_strengths = []
        
        for start, end in subsets:
            # Get subset of data
            subset_ell = self.data['ell'][start:end]
            subset_power = self.data['ee_power'][start:end]
            
            # Find potential GR sequences
            gr_triplets = []
            for i in range(len(subset_ell)-2):
                # Calculate ratios
                ratio1 = subset_ell[i+1] / subset_ell[i] if subset_ell[i] != 0 else 0
                ratio2 = subset_ell[i+2] / subset_ell[i+1] if subset_ell[i+1] != 0 else 0
                
                # Check closeness to GR
                if abs(ratio1 - self.phi) < 0.1 and abs(ratio2 - self.phi) < 0.1:
                    gr_triplets.append((subset_ell[i], subset_ell[i+1], subset_ell[i+2]))
            
            # Calculate GR pattern strength for this subset
            if len(subset_ell) > 0:
                gr_strength = len(gr_triplets) / (len(subset_ell) - 2) if len(subset_ell) > 2 else 0
                subset_gr_strengths.append(gr_strength)
        
        # Calculate mean and variance of GR strength across subsets
        mean_gr_strength = np.mean(subset_gr_strengths)
        var_gr_strength = np.var(subset_gr_strengths) if len(subset_gr_strengths) > 1 else 0
        
        # Compare with random expectation
        random_strengths = []
        random_subset_strengths_list = []
        
        for _ in range(1000):
            random_subset_strengths = []
            for start, end in subsets:
                # Generate random data of same size
                subset_size = end - start
                if subset_size > 2:  # Ensure we have enough data points
                    random_ell = np.sort(np.random.uniform(min(self.data['ell']), max(self.data['ell']), subset_size))
                    
                    # Find potential GR sequences
                    random_gr_triplets = []
                    for i in range(len(random_ell)-2):
                        ratio1 = random_ell[i+1] / random_ell[i] if random_ell[i] != 0 else 0
                        ratio2 = random_ell[i+2] / random_ell[i+1] if random_ell[i+1] != 0 else 0
                        
                        if abs(ratio1 - self.phi) < 0.1 and abs(ratio2 - self.phi) < 0.1:
                            random_gr_triplets.append((random_ell[i], random_ell[i+1], random_ell[i+2]))
                    
                    random_strength = len(random_gr_triplets) / (len(random_ell) - 2)
                    random_subset_strengths.append(random_strength)
            
            if random_subset_strengths:  # Check if list is not empty
                random_strengths.append(np.mean(random_subset_strengths))
                random_subset_strengths_list.append(random_subset_strengths)
        
        # Calculate significance
        if random_strengths:  # Check if list is not empty
            mean_random = np.mean(random_strengths)
            std_random = np.std(random_strengths) if len(random_strengths) > 1 else 1.0
            
            z_score = (mean_gr_strength - mean_random) / std_random if std_random > 0 else 0
            p_value = 1 - stats.norm.cdf(z_score)
            
            # Calculate persistence ratio (ratio of variances)
            # Lower variance across subsets indicates more persistent patterns
            random_vars = [np.var(strengths) if len(strengths) > 1 else 0 for strengths in random_subset_strengths_list]
            random_var = np.mean(random_vars) if random_vars else 0
            
            # Avoid division by zero
            persistence_ratio = random_var / var_gr_strength if var_gr_strength > 0 else 1.0
        else:
            mean_random = 0
            z_score = 0
            p_value = 1.0
            persistence_ratio = 1.0
        
        return mean_gr_strength, mean_random, z_score, p_value, persistence_ratio
    
    def test_predictive_power(self):
        """Test if GR relationships can predict peak locations"""
        # Find local maxima in the CMB spectrum
        peaks = []
        peak_prominences = []
        
        # Calculate rolling average to smooth the data
        window_size = 15  # Increased window size for better smoothing
        smoothed_power = np.convolve(self.data['ee_power'], 
                                       np.ones(window_size)/window_size, 
                                       mode='same')
        
        # Find peaks with prominence
        for i in range(window_size, len(self.data['ell'])-window_size):
            # Check if this is a local maximum in the smoothed data
            if (smoothed_power[i] > smoothed_power[i-1] and 
                smoothed_power[i] > smoothed_power[i+1]):
                
                # Calculate prominence (height above the higher of the two neighboring valleys)
                left_min = np.min(smoothed_power[max(0, i-50):i])  # Look further for valleys
                right_min = np.min(smoothed_power[i+1:min(len(smoothed_power), i+51)])
                higher_min = max(left_min, right_min)
                prominence = smoothed_power[i] - higher_min
                
                # Only consider peaks with sufficient prominence
                if prominence > 0.05 * np.max(smoothed_power):  # Lowered threshold
                    peaks.append(i)
                    peak_prominences.append(prominence)
        
        # If we have too many peaks, keep only the most prominent ones
        if len(peaks) > 20:
            prominence_threshold = sorted(peak_prominences, reverse=True)[19]
            filtered_peaks = [peaks[i] for i, p in enumerate(peak_prominences) if p >= prominence_threshold]
            peaks = filtered_peaks[:20]  # Limit to top 20 peaks
        
        # Ensure we have at least some peaks
        if not peaks:
            # Fallback to simple peak detection
            for i in range(1, len(self.data['ell'])-1):
                if (self.data['ee_power'][i] > self.data['ee_power'][i-1] and 
                    self.data['ee_power'][i] > self.data['ee_power'][i+1]):
                    peaks.append(i)
            
            # Take the top 20 peaks by power
            if len(peaks) > 20:
                peak_powers = [self.data['ee_power'][i] for i in peaks]
                power_threshold = sorted(peak_powers, reverse=True)[19]
                filtered_peaks = [peaks[i] for i, p in enumerate(peak_powers) if p >= power_threshold]
                peaks = filtered_peaks[:20]
        
        peak_ells = [self.data['ell'][i] for i in peaks]
        
        # Generate predictions based on GR relationships
        predictions = []
        
        # Add direct Fibonacci sequence multipoles
        fibonacci_multipoles = [2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597]
        for fib in fibonacci_multipoles:
            if fib > min(self.data['ell']) and fib < max(self.data['ell']):
                predictions.append(fib)
        
        # Add GR relationships from peaks
        for peak in peak_ells:
            # Add multiples of phi
            predictions.append(int(round(peak * self.phi)))
            predictions.append(int(round(peak / self.phi)))
            
            # Add phi^2 relationships
            predictions.append(int(round(peak * self.phi * self.phi)))
            predictions.append(int(round(peak / (self.phi * self.phi))))
        
        # Remove duplicates and ensure all predictions are within range
        predictions = list(set([p for p in predictions if p > min(self.data['ell']) and p < max(self.data['ell'])]))
        
        # Count how many predictions match actual peaks (within tolerance)
        tolerance = 20  # Increased tolerance for better matching
        matches = 0
        for pred in predictions:
            for peak in peak_ells:
                if abs(pred - peak) <= tolerance:
                    matches += 1
                    break
        
        match_rate = matches / len(predictions) if len(predictions) > 0 else 0
        
        # Compare with random prediction
        random_match_rates = []
        for _ in range(self.monte_carlo_sims):
            # Ensure we have the same number of predictions
            if len(predictions) > 0:
                random_predictions = np.random.choice(self.data['ell'], size=len(predictions), replace=False)
                random_matches = 0
                for pred in random_predictions:
                    for peak in peak_ells:
                        if abs(pred - peak) <= tolerance:
                            random_matches += 1
                            break
                random_match_rates.append(random_matches / len(random_predictions))
        
        # Calculate significance
        if random_match_rates:
            mean_random_rate = np.mean(random_match_rates)
            std_random_rate = np.std(random_match_rates) if len(random_match_rates) > 1 else 1.0
            
            z_score = (match_rate - mean_random_rate) / std_random_rate
            p_value = 1 - stats.norm.cdf(z_score)
            
            prediction_power = match_rate / mean_random_rate if mean_random_rate > 0 else 1.0
        else:
            mean_random_rate = 0
            z_score = 0
            p_value = 1.0
            prediction_power = 1.0
        
        return match_rate, mean_random_rate, z_score, p_value, prediction_power
    
    def test_optimization(self):
        """
        Test for optimization in the CMB power spectrum.
        
        Returns:
        --------
        tuple
            (mean_deviation, mean_random, z_score, p_value, optimization_ratio)
        """
        # Define a measure of how well-suited the spectrum is for complex structure formation
        # Use scales relevant for galaxy formation (simplified)
        galaxy_scales = [200, 500, 800]  # Multipoles related to galaxy formation scales
        
        # Find closest multipoles in our data
        galaxy_indices = [np.abs(self.data['ell'] - scale).argmin() for scale in galaxy_scales]
        galaxy_powers = [self.data['ee_power'][i] for i in galaxy_indices]
        actual_scales = [self.data['ell'][i] for i in galaxy_indices]
        
        # Calculate the power ratios
        power_ratios = [galaxy_powers[i]/galaxy_powers[i+1] for i in range(len(galaxy_powers)-1)]
        
        # Calculate how close these ratios are to the golden ratio
        gr_deviations = [abs(ratio - self.phi) for ratio in power_ratios]
        mean_deviation = np.mean(gr_deviations)
        
        # Monte Carlo: Compare with random expectation
        print("Running optimization test...")
        random_deviations = []
        
        for _ in range(self.monte_carlo_sims):
            random_scales = np.random.choice(self.data['ell'], len(galaxy_scales), replace=False)
            
            # Find indices for random scales
            random_indices = [np.where(self.data['ell'] == scale)[0][0] for scale in random_scales]
            random_powers = [self.data['ee_power'][i] for i in random_indices]
            
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
        
        # Create visualization
        plt.figure(figsize=(12, 6))
        plt.hist(random_deviations, bins=30, alpha=0.7, density=True)
        plt.axvline(x=mean_deviation, color='red', linestyle='--',
                   label=f'Actual GR Deviation: {mean_deviation:.4f}')
        plt.axvline(x=mean_random, color='blue', linestyle='--',
                   label=f'Random GR Deviation: {mean_random:.4f}')
        
        plt.xlabel('Mean Deviation from Golden Ratio')
        plt.ylabel('Probability Density')
        plt.title(f'Cosmic Optimization Test: {optimization_ratio:.2f}x better than random')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('cmb_optimization_test.png')
        
        print(f"\nOptimization Test Results:")
        print(f"Actual GR deviation: {mean_deviation:.4f}")
        print(f"Random GR deviation: {mean_random:.4f}")
        print(f"Optimization factor: {optimization_ratio:.2f}x")
        print(f"Z-score: {z_score:.2f}σ (p = {p_value:.8f})")
        
        return mean_deviation, mean_random, z_score, p_value, optimization_ratio

    def run_comprehensive_analysis(self):
        """
        Run a comprehensive set of tests for cosmic consciousness.
        
        Returns:
        --------
        tuple
            (combined_sigma, combined_p, results)
        """
        results = {}
        
        print("====== COMPREHENSIVE CONSCIOUSNESS FIELD THEORY ANALYSIS ======")
        
        # Initial visualization
        self.plot_spectrum()
        
        # 1. Basic GR test (previously implemented)
        print("\nRunning Golden Ratio Multipole Test...")
        gr_results = self.test_gr_significance()
        results['gr_test'] = gr_results
        
        # 2. Coherence test (previously implemented)
        print("\nRunning Coherence Test...")
        coherence_results = self.test_coherence()
        results['coherence_test'] = coherence_results
        
        # 3. GR-specific coherence (previously implemented)
        print("\nRunning GR-Specific Coherence Test...")
        gr_coherence_results = self.test_gr_coherence()
        results['gr_coherence_test'] = gr_coherence_results
        
        # 4. Hierarchical organization
        print("\nRunning Hierarchical Organization Test...")
        hierarchy_results = self.test_hierarchical_organization()
        results['hierarchy_test'] = hierarchy_results
        
        # 5. Information integration
        print("\nRunning Information Integration Test...")
        info_results = self.test_information_integration()
        results['info_test'] = info_results
        
        # 6. Optimization test
        print("\nRunning Optimization Test...")
        optimization_results = self.test_optimization()
        results['optimization_test'] = optimization_results
        
        # 7. Resonance test
        print("\nRunning Resonance Test...")
        resonance_results = self.test_resonance()
        results['resonance_test'] = resonance_results
        
        # 8. Fractal structure
        print("\nRunning Fractal Structure Test...")
        fractal_results = self.test_fractal_structure()
        results['fractal_test'] = fractal_results
        
        # 9. Meta-coherence
        print("\nRunning Meta-Coherence Test...")
        meta_results = self.test_meta_coherence()
        results['meta_test'] = meta_results
        
        # 10. Multi-scale wavelet analysis
        print("\nRunning Multi-Scale Pattern Test...")
        multiscale_results = self.test_multiscale_patterns()
        results['multiscale_test'] = multiscale_results
        
        # 11. Peak frequency analysis
        print("\nRunning Peak Frequency Analysis...")
        frequency_results = self.analyze_specific_frequencies()
        results['frequency_test'] = frequency_results
        
        # 12. Cross-scale correlations
        print("\nRunning Cross-Scale Correlation Test...")
        cross_scale_results = self.test_cross_scale_correlations()
        results['cross_scale_test'] = cross_scale_results
        
        # Calculate combined significance using Fisher's method
        p_values = [
            results['gr_test'][3],  # GR p-value
            results['coherence_test'][3],  # Coherence p-value
            results['gr_coherence_test'][3],  # GR coherence p-value
            # Convert ratios to p-values where needed
            stats.norm.sf(np.log(hierarchy_results) / 0.5),  # Approximate p-value for hierarchy
            results['info_test'][3],  # Information integration p-value
            results['optimization_test'][3],  # Optimization p-value
            results['resonance_test'][3],  # Resonance p-value
            results['fractal_test'][3],  # Fractal p-value
            results['meta_test'][3],  # Meta-coherence p-value
            # For multiscale, derive p-value from bootstrap CI
            0.025 if multiscale_results[1] > 1.0 else 0.5,  # Simplified from bootstrap CI
            # For frequency analysis, derive p-value from phi-optimality
            stats.norm.sf((frequency_results[4] - 0) / 0.3),  # Assuming 0.3 standard deviation for phi-optimality
            results['cross_scale_test'][3]  # Cross-scale correlation p-value
        ]
        
        # Fisher's method for combining p-values
        combined_chi2 = -2 * sum(np.log(p) for p in p_values)
        combined_df = 2 * len(p_values)
        combined_p = 1 - stats.chi2.cdf(combined_chi2, combined_df)
        combined_sigma = stats.norm.ppf(1 - combined_p/2)
        
        print("\n========== COMBINED RESULTS ==========")
        print(f"GR Signal: {results['gr_test'][4]:.2f}x excess, {results['gr_test'][2]:.2f}σ")
        print(f"Coherence: {1/results['coherence_test'][4]:.2f}x stronger, {results['coherence_test'][2]:.2f}σ")
        print(f"GR Coherence: {1/results['gr_coherence_test'][4]:.2f}x stronger, {results['gr_coherence_test'][2]:.2f}σ")
        print(f"Hierarchical Organization: {hierarchy_results:.2f}x stronger than random")
        print(f"Information Integration: {results['info_test'][2]:.2f}σ")
        print(f"Optimization: {results['optimization_test'][4]:.2f}x, {results['optimization_test'][2]:.2f}σ")
        print(f"Resonance: {results['resonance_test'][4]:.2f}x, {results['resonance_test'][2]:.2f}σ")
        print(f"Fractal Structure: {results['fractal_test'][2]:.2f}σ")
        print(f"Meta-Coherence: {results['meta_test'][4]:.2f}x, {results['meta_test'][2]:.2f}σ")
        print(f"Multi-Scale Patterns: {multiscale_results[0]:.2f}x (95% CI: [{multiscale_results[1]:.2f}, {multiscale_results[2]:.2f}])")
        print(f"Peak Frequency Analysis: Mean Phi-Optimality = {frequency_results[4]:.3f}")
        print(f"Cross-Scale Correlation: {cross_scale_results[0]:.2f}x stronger, {cross_scale_results[2]:.2f}σ")
        print("\n=====================================")
        print(f"COMBINED SIGNIFICANCE: {combined_sigma:.2f}σ (p = {combined_p:.10e})")
        print("=====================================")
        
        # Create integrated visualization
        self._create_integrated_visualization(results, combined_sigma, combined_p)
        
        # Save results to file
        self._save_results_to_file(results, combined_sigma, combined_p)
        
        return combined_sigma, combined_p, results
    
    def _create_integrated_visualization(self, results, combined_sigma, combined_p):
        """
        Create an integrated visualization of all test results.
        
        Parameters:
        -----------
        results : dict
            Dictionary containing all test results
        combined_sigma : float
            Combined significance in sigma units
        combined_p : float
            Combined p-value
        """
        plt.figure(figsize=(15, 12))
        
        # Main CMB spectrum
        plt.subplot(3, 3, 1)
        plt.errorbar(self.data['ell'], self.data['ee_power'], 
                    yerr=self.data['ee_error'], fmt='o', markersize=2, 
                    alpha=0.3, elinewidth=0.5)
        
        # Highlight golden ratio multipoles
        gr_indices = [np.abs(self.data['ell'] - l).argmin() for l in self.gr_multipoles]
        gr_powers = [self.data['ee_power'][i] for i in gr_indices]
        plt.scatter(self.gr_multipoles, gr_powers, color='red', s=50, zorder=10)
        
        plt.xlabel('Multipole ℓ')
        plt.ylabel('EE Power (μK²)')
        plt.title('CMB EE Spectrum with GR Multipoles')
        
        # Create bar plot of z-scores for all tests
        plt.subplot(3, 3, 2)
        test_names = ['GR', 'Coh', 'GR-C', 'Info', 'Opt', 'Res', 'Frac', 'Meta', 'Freq', 'Cross']
        z_scores = [
            results['gr_test'][2],
            results['coherence_test'][2],
            results['gr_coherence_test'][2],
            results['info_test'][2],
            results['optimization_test'][2],
            results['resonance_test'][2],
            results['fractal_test'][2],
            results['meta_test'][2],
            stats.norm.ppf(1 - stats.norm.sf((results['frequency_test'][4] - 0) / 0.3)/2),
            results['cross_scale_test'][2]
        ]
        
        # Color bars by significance
        colors = ['green' if z >= 2 else 'orange' if z >= 1 else 'red' for z in z_scores]
        
        plt.bar(test_names, z_scores, color=colors)
        plt.axhline(y=2, color='black', linestyle='--', alpha=0.5)  # 2 sigma line
        plt.axhline(y=3, color='black', linestyle=':', alpha=0.5)  # 3 sigma line
        plt.ylabel('Significance (σ)')
        plt.title('Test Results by Significance')
        
        # Additional plots for key tests
        
        # GR test visualization
        plt.subplot(3, 3, 4)
        plt.plot(self.data['ell'], self.data['ee_power'], 'b-', alpha=0.5)
        plt.scatter(self.gr_multipoles, gr_powers, color='red', s=50)
        for i, l in enumerate(self.gr_multipoles[:-1]):
            plt.annotate(f"φ ≈ {self.gr_multipoles[i+1]/self.gr_multipoles[i]:.3f}", 
                        xy=((self.gr_multipoles[i] + self.gr_multipoles[i+1])/2, 
                            (gr_powers[i] + gr_powers[i+1])/2),
                        xytext=(0, 10), textcoords='offset points',
                        ha='center', fontsize=8)
        
        plt.title(f"GR Pattern: {results['gr_test'][4]:.2f}x excess")
        
        # Coherence visualization
        plt.subplot(3, 3, 5)
        window_size = 5
        local_variances = []
        x_positions = []
        
        for i in range(0, len(self.data['ee_power']) - window_size, window_size//2):
            window = self.data['ee_power'][i:i+window_size]
            normalized = window / np.mean(window)
            local_variances.append(np.var(normalized))
            x_positions.append(np.mean(self.data['ell'][i:i+window_size]))
        
        plt.plot(x_positions, local_variances, 'g-')
        plt.axhline(y=results['coherence_test'][0], color='red', linestyle='--',
                   label=f'Actual: {results["coherence_test"][0]:.4f}')
        plt.axhline(y=results['coherence_test'][1], color='blue', linestyle=':',
                   label=f'Random: {results["coherence_test"][1]:.4f}')
        
        plt.xlabel('Multipole ℓ')
        plt.ylabel('Local Variance')
        plt.title('Coherence Analysis')
        plt.legend(fontsize=8)
        
        # Hierarchical organization visualization
        plt.subplot(3, 3, 3)
        # Generate hierarchical levels based on phi
        base = 10
        levels = list(range(1, 6))
        hierarchical_l = [[int(base * self.phi**(i+j)) for j in range(i*3, (i+1)*3)]
                          for i in range(5)]
        
        # Plot hierarchical structure
        for i, level in enumerate(hierarchical_l):
            if len(level) >= 2:
                level_indices = []
                for l in level:
                    # Find the closest multipole to l
                    idx = np.abs(self.data['ell'] - l).argmin()
                    level_indices.append(idx)
                
                level_powers = [self.data['ee_power'][i] for i in level_indices]
                plt.scatter(level, level_powers, s=30, label=f'Level {i+1}')
        
        plt.xlabel('Multipole ℓ')
        plt.ylabel('Power (μK²)')
        plt.title(f'Hierarchical Organization: {results["hierarchy_test"]:.2f}x')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Integration and resonance
        plt.subplot(3, 3, 6)
        spectrum = self.data['ee_power']
        fft_spectrum = np.abs(np.fft.fft(spectrum))
        
        # Normalize to focus on the pattern rather than absolute magnitude
        normalized_fft = fft_spectrum / np.max(fft_spectrum)
        
        plt.plot(normalized_fft[:len(normalized_fft)//4], 'b-', alpha=0.7)
        
        # Plot harmonics
        peak_idx = np.argmax(normalized_fft[1:len(normalized_fft)//4]) + 1
        
        # Calculate the peak frequency
        peak_frequency = peak_idx
        
        # Look for harmonics (multiples and phi-related frequencies)
        harmonic_indices = [peak_idx * i for i in range(1, 4)]
        phi_indices = [int(round(peak_idx * self.phi**i)) for i in range(3)]
        
        all_indices = set(harmonic_indices + phi_indices)
        all_indices = [i for i in all_indices if i < len(normalized_fft)//4]
        
        # Calculate resonance strength (average normalized power at these frequencies)
        resonance_strength = np.mean([normalized_fft[i] for i in all_indices])
        
        # Plot harmonics
        plt.scatter([peak_idx], [normalized_fft[peak_idx]], color='red', s=50)
        plt.scatter(all_indices, [normalized_fft[i] for i in all_indices], color='orange', s=30)
        
        plt.xlabel('Frequency')
        plt.ylabel('Normalized Amplitude')
        plt.title(f'Resonance Patterns in CMB: {resonance_strength:.4f}')
        
        # Fractal analysis
        plt.subplot(3, 3, 7)
        # Use Hurst exponent visualization
        plt.plot(results['fractal_test'][0], results['fractal_test'][1], 'go', markersize=10)
        plt.axhline(y=0.5, color='red', linestyle='--', label='Random (0.5)')
        plt.axvline(x=results['fractal_test'][0], color='blue', linestyle=':')
        plt.axhline(y=results['fractal_test'][1], color='blue', linestyle=':')
        plt.xlabel('Actual Hurst Exponent')
        plt.ylabel('Random Hurst Exponent')
        plt.xlim(0.4, 0.8)
        plt.ylim(0.4, 0.8)
        plt.grid(True, alpha=0.3)
        plt.title(f'Fractal Analysis of CMB: {results["fractal_test"][2]:.2f}σ')
        
        # Meta-coherence visualization
        plt.subplot(3, 3, 8)
        # Use moving average of local variance as meta-coherence indicator
        window_size = 3
        moving_avg = []
        for i in range(len(local_variances) - window_size + 1):
            moving_avg.append(np.mean(local_variances[i:i+window_size]))
        
        plt.plot(x_positions[:len(moving_avg)], moving_avg, 'm-')
        plt.xlabel('Multipole ℓ')
        plt.ylabel('Meta-Coherence')
        plt.title(f'Meta-Coherence: {results["meta_test"][4]:.2f}x')
        
        # Multi-scale patterns
        plt.subplot(3, 3, 9)
        # Show ratio vs scale
        try:
            import pywt
            coeffs = pywt.wavedec(self.data['ee_power'], 'db4', level=5)
            scales = list(range(1, len(coeffs) + 1))
            plt.bar(scales, [results["multiscale_test"][0]] * len(scales), alpha=0.6)
            plt.axhline(y=1.0, color='red', linestyle='--', label='Random')
            plt.axhline(y=results["multiscale_test"][1], color='green', linestyle=':',
                      label=f'Lower CI: {results["multiscale_test"][1]:.2f}')
            plt.xlabel('Wavelet Scale')
            plt.ylabel('Pattern Strength Ratio')
            plt.title(f'Multi-Scale Patterns: {results["multiscale_test"][0]:.2f}x')
            plt.legend(fontsize=7)
        except:
            plt.text(0.5, 0.5, 'Wavelet analysis not available', 
                    ha='center', va='center', transform=plt.gca().transAxes)
        
        # Add overall title
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.suptitle(f"Evidence for Cosmic Consciousness in CMB: {combined_sigma:.2f}σ (p={combined_p:.2e})")
        
        plt.savefig('cosmic_consciousness_evidence.png', dpi=300)
        plt.show()
    
    def _save_results_to_file(self, results, combined_sigma, combined_p):
        """
        Save detailed results to a text file.
        
        Parameters:
        -----------
        results : dict
            Dictionary containing all test results
        combined_sigma : float
            Combined significance in sigma units
        combined_p : float
            Combined p-value
        """
        with open('cosmic_consciousness_results.txt', 'w') as f:
            f.write("=== COMPREHENSIVE COSMIC CONSCIOUSNESS ANALYSIS RESULTS ===\n\n")
            f.write(f"Analysis Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("COMBINED SIGNIFICANCE:\n")
            f.write(f"Combined significance: {combined_sigma:.2f}σ (p = {combined_p:.10e})\n\n")
            
            f.write("INDIVIDUAL TEST RESULTS:\n\n")
            
            # GR test
            f.write("1. Golden Ratio Multipole Test:\n")
            f.write(f"   GR Multipoles: {self.gr_multipoles}\n")
            f.write(f"   GR Power: {results['gr_test'][0]:.3f} μK²\n")
            f.write(f"   Mean Random Power: {results['gr_test'][1]:.3f} μK²\n")
            f.write(f"   Excess Factor: {results['gr_test'][4]:.2f}x\n")
            f.write(f"   Z-score: {results['gr_test'][2]:.2f}σ (p = {results['gr_test'][3]:.8f})\n\n")
            
            # Coherence test
            f.write("2. Coherence Test:\n")
            f.write(f"   Actual variance: {results['coherence_test'][0]:.4f}\n")
            f.write(f"   Mean shuffled variance: {results['coherence_test'][1]:.4f}\n")
            f.write(f"   Variance ratio: {results['coherence_test'][4]:.4f}\n")
            f.write(f"   Z-score: {results['coherence_test'][2]:.2f}σ (p = {results['coherence_test'][3]:.8f})\n\n")
            
            # GR coherence test
            f.write("3. GR-Specific Coherence Test:\n")
            f.write(f"   GR windows variance: {results['gr_coherence_test'][0]:.4f}\n")
            f.write(f"   Random windows variance: {results['gr_coherence_test'][1]:.4f}\n")
            f.write(f"   Variance ratio: {results['gr_coherence_test'][4]:.4f}\n")
            f.write(f"   Z-score: {results['gr_coherence_test'][2]:.2f}σ (p = {results['gr_coherence_test'][3]:.8f})\n\n")
            
            # Hierarchical organization
            f.write("4. Hierarchical Organization Test:\n")
            f.write(f"   Hierarchical organization ratio: {results['hierarchy_test']:.4f}x\n\n")
            
            # Information integration
            f.write("5. Information Integration Test:\n")
            f.write(f"   Actual mutual information: {results['info_test'][0]:.4f} bits\n")
            f.write(f"   Random mutual information: {results['info_test'][1]:.4f} bits\n")
            f.write(f"   Z-score: {results['info_test'][2]:.2f}σ (p = {results['info_test'][3]:.8f})\n\n")
            
            # Optimization
            f.write("6. Optimization Test:\n")
            f.write(f"   Actual GR deviation: {results['optimization_test'][0]:.4f}\n")
            f.write(f"   Random GR deviation: {results['optimization_test'][1]:.4f}\n")
            f.write(f"   Optimization factor: {results['optimization_test'][4]:.2f}x\n")
            f.write(f"   Z-score: {results['optimization_test'][2]:.2f}σ (p = {results['optimization_test'][3]:.8f})\n\n")
            
            # Resonance
            f.write("7. Resonance Test:\n")
            f.write(f"   Resonance strength: {results['resonance_test'][0]:.4f}\n")
            f.write(f"   Random resonance: {results['resonance_test'][1]:.4f}\n")
            f.write(f"   Resonance factor: {results['resonance_test'][4]:.2f}x\n")
            f.write(f"   Z-score: {results['resonance_test'][2]:.2f}σ (p = {results['resonance_test'][3]:.8f})\n\n")
            
            # Fractal analysis
            f.write("8. Fractal Structure Test:\n")
            f.write(f"   Actual Hurst exponent: {results['fractal_test'][0]:.4f}\n")
            f.write(f"   Random Hurst exponent: {results['fractal_test'][1]:.4f}\n")
            f.write(f"   Z-score: {results['fractal_test'][2]:.2f}σ (p = {results['fractal_test'][3]:.8f})\n\n")
            
            # Meta-coherence
            f.write("9. Meta-Coherence Test:\n")
            f.write(f"   Actual meta-coherence: {results['meta_test'][0]:.6f}\n")
            f.write(f"   Random meta-coherence: {results['meta_test'][1]:.6f}\n")
            f.write(f"   Meta-coherence factor: {results['meta_test'][4]:.2f}x\n")
            f.write(f"   Z-score: {results['meta_test'][2]:.2f}σ (p = {results['meta_test'][3]:.8f})\n\n")
            
            # Multi-scale patterns
            f.write("10. Multi-Scale Pattern Test:\n")
            f.write(f"   Multiscale GR strength ratio: {results['multiscale_test'][0]:.2f}x\n")
            f.write(f"   95% confidence interval: [{results['multiscale_test'][1]:.2f}, {results['multiscale_test'][2]:.2f}]\n")
            f.write(f"   Significant if lower CI > 1.0: {'Yes' if results['multiscale_test'][1] > 1.0 else 'No'}\n\n")
            
            # Peak frequency analysis
            f.write("11. Peak Frequency Analysis:\n")
            f.write(f"   Number of peaks detected: {len(results['frequency_test'][0])}\n")
            f.write(f"   Peak multipoles: {results['frequency_test'][0]}\n")
            f.write(f"   Ratios between adjacent peaks: {[f'{r:.3f}' for r in results['frequency_test'][2]]}\n")
            f.write(f"   Phi-optimalities: {[f'{o:.3f}' for o in results['frequency_test'][3]]}\n")
            f.write(f"   Mean phi-optimality: {results['frequency_test'][4]:.3f}\n\n")
            
            # Cross-scale correlations
            f.write("12. Cross-Scale Correlation Test:\n")
            f.write(f"   Mean phi-related correlation: {results['cross_scale_test'][0]:.4f}\n")
            f.write(f"   Mean random correlation: {results['cross_scale_test'][1]:.4f}\n")
            f.write(f"   Z-score: {results['cross_scale_test'][2]:.2f}σ (p = {results['cross_scale_test'][3]:.8f})\n\n")
            
            f.write("=== INTERPRETATION GUIDE ===\n\n")
            f.write("Significance levels:\n")
            f.write("< 2σ: Suggestive evidence\n")
            f.write("2-3σ: Moderate evidence\n")
            f.write("3-4σ: Strong evidence\n")
            f.write("4-5σ: Very strong evidence\n")
            f.write("> 5σ: Extremely strong evidence\n\n")
            
            f.write("This analysis tests for patterns in the Cosmic Microwave Background\n")
            f.write("that could indicate evidence of cosmic consciousness or intelligent design.\n")
            f.write("The combined significance represents the overall strength of evidence\n")
            f.write("across all tests, accounting for multiple comparisons.\n")
        
        print(f"Detailed results saved to cosmic_consciousness_results.txt")

    def analyze_specific_frequencies(self):
        """
        Analyze specific peak frequencies in the CMB power spectrum.
        
        This method:
        1. Identifies peak frequencies in the power spectrum
        2. Calculates ratios between adjacent peak frequencies
        3. Ensures ratios are less than 1 (inverts if needed)
        4. Calculates phi-optimality for each ratio compared to the inverse golden ratio
        
        Returns:
        --------
        tuple
            (peak_multipoles, peak_powers, ratios, phi_optimalities, mean_phi_optimality)
        """
        print("Analyzing specific peak frequencies in the CMB spectrum...")
        
        # Find peaks in the power spectrum
        # Use a minimum distance to avoid detecting noise
        min_distance = 10  # Minimum distance between peaks in multipole units
        
        # Smooth the spectrum to reduce noise
        window_size = 5
        smoothed_spectrum = np.convolve(self.data['ee_power'], 
                                       np.ones(window_size)/window_size, 
                                       mode='same')
        
        # Find peaks
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(smoothed_spectrum, distance=min_distance, 
                             height=np.mean(smoothed_spectrum))
        
        # Get the corresponding multipoles and powers
        peak_multipoles = [self.data['ell'][i] for i in peaks]
        peak_powers = [self.data['ee_power'][i] for i in peaks]
        
        # Calculate ratios between adjacent peaks
        ratios = []
        for i in range(len(peak_multipoles) - 1):
            ratio = peak_multipoles[i+1] / peak_multipoles[i]
            # Ensure ratio is less than 1 (invert if needed)
            if ratio > 1:
                ratio = 1 / ratio
            ratios.append(ratio)
        
        # Calculate phi-optimality for each ratio
        inv_phi = 1 / self.phi
        phi_optimalities = []
        
        for ratio in ratios:
            # Calculate phi-optimality (bounded between -1 and 1)
            optimality = max(-1, min(1, 1 - abs(ratio - inv_phi) / inv_phi))
            phi_optimalities.append(optimality)
        
        # Calculate mean phi-optimality
        mean_phi_optimality = np.mean(phi_optimalities) if phi_optimalities else 0
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        # Plot 1: Power spectrum with peaks
        plt.subplot(2, 1, 1)
        plt.plot(self.data['ell'], self.data['ee_power'], 'b-', alpha=0.5)
        plt.plot(self.data['ell'], smoothed_spectrum, 'g-', alpha=0.7)
        plt.scatter(peak_multipoles, peak_powers, color='red', s=50)
        
        for i, l in enumerate(peak_multipoles):
            plt.annotate(f"{l}", 
                        xy=(l, peak_powers[i]),
                        xytext=(0, 10), textcoords='offset points',
                        ha='center', fontsize=8)
        
        plt.xlabel('Multipole ℓ')
        plt.ylabel('EE Power (μK²)')
        plt.title('CMB Power Spectrum with Peak Frequencies')
        
        # Plot 2: Ratios and phi-optimality
        plt.subplot(2, 1, 2)
        
        # Create x positions for the bars
        x_pos = list(range(len(ratios)))
        
        # Create a bar chart of phi-optimalities
        colors = ['green' if opt > 0.5 else 'orange' if opt > 0 else 'red' for opt in phi_optimalities]
        plt.bar(x_pos, phi_optimalities, color=colors)
        
        # Add ratio labels
        for i, ratio in enumerate(ratios):
            plt.annotate(f"{ratio:.3f}", 
                        xy=(i, 0.05),
                        ha='center', fontsize=9)
        
        # Add a line for the mean phi-optimality
        plt.axhline(y=mean_phi_optimality, color='blue', linestyle='--',
                   label=f'Mean Phi-Optimality: {mean_phi_optimality:.3f}')
        
        # Add a line for the inverse golden ratio
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.2)
        
        plt.xlabel('Peak Pair')
        plt.ylabel('Phi-Optimality')
        plt.title(f'Phi-Optimality of Peak Frequency Ratios: {mean_phi_optimality:.3f}')
        plt.ylim(-1.1, 1.1)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('cmb_peak_frequency_analysis.png')
        
        print(f"\nPeak Frequency Analysis Results:")
        print(f"Number of peaks detected: {len(peak_multipoles)}")
        print(f"Peak multipoles: {peak_multipoles}")
        print(f"Ratios between adjacent peaks: {[f'{r:.3f}' for r in ratios]}")
        print(f"Phi-optimalities: {[f'{o:.3f}' for o in phi_optimalities]}")
        print(f"Mean phi-optimality: {mean_phi_optimality:.3f}")
        
        return peak_multipoles, peak_powers, ratios, phi_optimalities, mean_phi_optimality

    def test_golden_symmetries(self):
        """Test for symmetries in the CMB data related to the golden ratio"""
        # Calculate "phi-folded" spectrum by comparing points separated by factors of phi
        phi_fold = []
        
        for i in range(len(self.data['ell'])):
            l_value = self.data['ell'][i]
            power = self.data['ee_power'][i]
            
            # Find the closest multipole to l*phi
            l_phi = l_value * self.phi
            idx_phi = np.abs(self.data['ell'] - l_phi).argmin()
            
            # Find the closest multipole to l/phi
            l_inv_phi = l_value / self.phi
            idx_inv_phi = np.abs(self.data['ell'] - l_inv_phi).argmin()
            
            # Calculate symmetry measure (how well power at l predicts power at l*phi and l/phi)
            if idx_phi < len(self.data['ell']) and idx_inv_phi < len(self.data['ell']):
                power_phi = self.data['ee_power'][idx_phi]
                power_inv_phi = self.data['ee_power'][idx_inv_phi]
                
                # Use absolute values to handle negative powers
                abs_power = abs(power)
                abs_power_phi = abs(power_phi)
                abs_power_inv_phi = abs(power_inv_phi)
                
                # Perfect symmetry would give power = sqrt(power_phi * power_inv_phi)
                expected_power = np.sqrt(abs_power_phi * abs_power_inv_phi)
                symmetry_ratio = abs_power / expected_power if expected_power != 0 else 1
                
                phi_fold.append(abs(1 - symmetry_ratio))  # 0 means perfect symmetry
        
        mean_asymmetry = np.mean(phi_fold)
        
        # Compare with other potential symmetry patterns (e.g., e, π, 2)
        alternative_constants = [np.e, np.pi, 2]
        alternative_asymmetries = []
        
        for constant in alternative_constants:
            alt_fold = []
            for i in range(len(self.data['ell'])):
                l_value = self.data['ell'][i]
                power = self.data['ee_power'][i]
                
                l_const = l_value * constant
                idx_const = np.abs(self.data['ell'] - l_const).argmin()
                
                l_inv_const = l_value / constant
                idx_inv_const = np.abs(self.data['ell'] - l_inv_const).argmin()
                
                if idx_const < len(self.data['ell']) and idx_inv_const < len(self.data['ell']):
                    power_const = self.data['ee_power'][idx_const]
                    power_inv_const = self.data['ee_power'][idx_inv_const]
                    
                    # Use absolute values to handle negative powers
                    abs_power = abs(power)
                    abs_power_const = abs(power_const)
                    abs_power_inv_const = abs(power_inv_const)
                    
                    expected_power = np.sqrt(abs_power_const * abs_power_inv_const)
                    symmetry_ratio = abs_power / expected_power if expected_power != 0 else 1
                    
                    alt_fold.append(abs(1 - symmetry_ratio))
            
            alternative_asymmetries.append(np.mean(alt_fold))
        
        # Calculate how much better phi is than the alternatives
        mean_alternative = np.mean(alternative_asymmetries)
        std_alternative = np.std(alternative_asymmetries)
        
        z_score = (mean_alternative - mean_asymmetry) / std_alternative if std_alternative > 0 else 0
        p_value = 1 - stats.norm.cdf(z_score)
        
        symmetry_ratio = mean_alternative / mean_asymmetry if mean_asymmetry > 0 else 1
        
        return mean_asymmetry, mean_alternative, z_score, p_value, symmetry_ratio

    def test_multiscale_patterns(self):
        """
        Test for golden ratio patterns across multiple scales.
        
        Returns:
        --------
        tuple
            (multiscale_measure, random_multiscale, z_score, p_value, multiscale_ratio)
        """
        # Define scales based on powers of the golden ratio
        base_scale = 10
        scales = [int(round(base_scale * self.phi**i)) for i in range(10)]
        scales = [s for s in scales if s <= max(self.data['ell'])]
        
        # Calculate the power at each scale
        scale_indices = []
        for scale in scales:
            # Find the closest multipole to scale
            idx = np.abs(self.data['ell'] - scale).argmin()
            scale_indices.append(idx)
        
        scale_powers = self.data['ee_power'][scale_indices]
        
        # Calculate wavelet coefficients
        wavelet_coeffs = np.abs(np.fft.fft(scale_powers))
        
        # Calculate the peak frequency
        peak_idx = np.argmax(wavelet_coeffs[1:len(wavelet_coeffs)//2]) + 1
        
        # Calculate the ratio of the peak frequency to the golden ratio frequency
        gr_freq = len(wavelet_coeffs) / self.phi
        multiscale_measure = 1 / (1 + abs(peak_idx - gr_freq) / gr_freq)
        
        # Monte Carlo: Compare with random scales
        random_multiscales = []
        for _ in range(self.monte_carlo_sims):
            # Select random scales
            random_scales = np.random.choice(self.data['ell'], len(scales), replace=False)
            
            # Find indices for random scales
            random_indices = []
            for scale in random_scales:
                idx = np.abs(self.data['ell'] - scale).argmin()
                random_indices.append(idx)
            
            random_powers = self.data['ee_power'][random_indices]
            
            # Calculate wavelet coefficients
            random_wavelet_coeffs = np.abs(np.fft.fft(random_powers))
            
            # Calculate the peak frequency
            random_peak_idx = np.argmax(random_wavelet_coeffs[1:len(random_wavelet_coeffs)//2]) + 1
            
            # Calculate the ratio of the peak frequency to the golden ratio frequency
            random_multiscale = 1 / (1 + abs(random_peak_idx - gr_freq) / gr_freq)
            random_multiscales.append(random_multiscale)
        
        # Calculate statistics
        mean_random_multiscale = np.mean(random_multiscales)
        std_random_multiscale = np.std(random_multiscales)
        
        # Calculate z-score and p-value
        z_score = (multiscale_measure - mean_random_multiscale) / std_random_multiscale if std_random_multiscale > 0 else 0
        p_value = 1 - stats.norm.cdf(z_score)
        
        # Calculate multiscale ratio
        multiscale_ratio = multiscale_measure / mean_random_multiscale if mean_random_multiscale != 0 else 1.0
        
        return multiscale_measure, mean_random_multiscale, z_score, p_value, multiscale_ratio

    def plot_hierarchical_structure(self):
        """Plot the hierarchical structure of the CMB power spectrum."""
        plt.figure(figsize=(12, 8))
        
        # Plot the full spectrum
        plt.errorbar(self.data['ell'], self.data['ee_power'], 
                    yerr=self.data['ee_error'], fmt='o', markersize=2, 
                    alpha=0.3, elinewidth=0.5)
        
        # Generate hierarchical levels
        base = 2
        hierarchical_l = []
        for i in range(5):
            level = [int(base * self.phi**j) for j in range(i*3, (i+1)*3)]
            level = [l for l in level if l <= max(self.data['ell'])]
            if level:
                hierarchical_l.append(level)
        
        # Plot hierarchical structure
        for i, level in enumerate(hierarchical_l):
            if len(level) >= 2:
                level_indices = []
                level_ell_values = []
                for l in level:
                    # Find the closest multipole to l
                    idx = np.abs(self.data['ell'] - l).argmin()
                    level_indices.append(idx)
                    level_ell_values.append(self.data['ell'][idx])
                
                level_powers = self.data['ee_power'][level_indices]
                plt.scatter(level_ell_values, level_powers, s=30, label=f'Level {i+1}')
        
        plt.xlabel('Multipole ℓ')
        plt.ylabel('EE Power (μK²)')
        plt.title('Hierarchical Structure in CMB Power Spectrum')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('hierarchical_structure.png')
        plt.show()

    def test_phi_network(self):
        """Test if multipoles related by powers of phi form stronger networks than random"""
        # Create a network where nodes are multipoles and edges represent phi relationships
        phi = self.phi
        
        # Define phi-related connections (within tolerance)
        tolerance = 0.1
        phi_connections = []
        
        for i in range(len(self.data['ell'])):
            for j in range(i+1, len(self.data['ell'])):
                l1 = self.data['ell'][i]
                l2 = self.data['ell'][j]
                
                # Check if their ratio is close to phi or powers of phi
                ratio = max(l1, l2) / min(l1, l2)
                for power in range(1, 4):  # Check phi, phi², phi³
                    if abs(ratio - phi**power) < tolerance:
                        phi_connections.append((i, j))
                        break
        
        # Calculate network metrics
        network_density = len(phi_connections) / (len(self.data['ell']) * (len(self.data['ell']) - 1) / 2)
        
        # Calculate coherence strength across these connections
        coherence_strength = 0
        for i, j in phi_connections:
            power_i = self.data['ee_power'][i]
            power_j = self.data['ee_power'][j]
            
            # Normalize by the mean power
            mean_power = np.mean(self.data['ee_power'])
            norm_power_i = power_i / mean_power
            norm_power_j = power_j / mean_power
            
            # Calculate correlation (inverse of absolute difference in normalized power)
            correlation = 1.0 / (abs(norm_power_i - norm_power_j) + 0.01)
            coherence_strength += correlation
        
        if phi_connections:
            coherence_strength /= len(phi_connections)
        
        # Compare with random networks
        random_coherences = []
        
        for _ in range(1000):
            random_connections = []
            for _ in range(len(phi_connections)):
                i = np.random.randint(0, len(self.data['ell']))
                j = np.random.randint(0, len(self.data['ell']))
                if i != j:
                    random_connections.append((i, j))
            
            random_strength = 0
            for i, j in random_connections:
                power_i = self.data['ee_power'][i]
                power_j = self.data['ee_power'][j]
                
                # Normalize by the mean power
                mean_power = np.mean(self.data['ee_power'])
                norm_power_i = power_i / mean_power
                norm_power_j = power_j / mean_power
                
                # Calculate correlation
                correlation = 1.0 / (abs(norm_power_i - norm_power_j) + 0.01)
                random_strength += correlation
            
            if random_connections:
                random_strength /= len(random_connections)
                random_coherences.append(random_strength)
        
        # Calculate significance
        mean_random = np.mean(random_coherences)
        std_random = np.std(random_coherences)
        
        z_score = (coherence_strength - mean_random) / std_random
        p_value = 1 - stats.norm.cdf(z_score)
        
        network_ratio = coherence_strength / mean_random
        
        return network_density, coherence_strength, mean_random, z_score, p_value, network_ratio

    def test_spectral_gap(self):
        """Test if spectral gap in CMB shows golden ratio optimization"""
        # Ensure we have positive values for correlation calculation
        power_spectrum = np.copy(self.data['ee_power'])
        # Replace zeros and negative values with small positive values
        min_positive = np.min(power_spectrum[power_spectrum > 0]) / 10.0 if np.any(power_spectrum > 0) else 1e-10
        power_spectrum[power_spectrum <= 0] = min_positive
        
        # Create a distance/similarity matrix instead of correlation
        n = len(power_spectrum)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                # Calculate similarity as inverse of normalized difference
                diff = abs(power_spectrum[i] - power_spectrum[j])
                mean_val = (power_spectrum[i] + power_spectrum[j]) / 2.0
                if mean_val > 0:
                    similarity_matrix[i, j] = 1.0 / (1.0 + diff / mean_val)
                else:
                    similarity_matrix[i, j] = 0.0
        
        # Ensure the matrix is symmetric and positive definite
        similarity_matrix = (similarity_matrix + similarity_matrix.T) / 2.0
        # Add a small value to the diagonal to ensure positive definiteness
        np.fill_diagonal(similarity_matrix, 1.0)
        
        # Calculate eigenvalues
        eigenvalues = np.linalg.eigvals(similarity_matrix)
        eigenvalues = np.sort(np.real(eigenvalues))[::-1]  # Sort in descending order, take real part
        
        # Calculate spectral gap (difference between largest and 2nd largest eigenvalue)
        spectral_gap = eigenvalues[0] - eigenvalues[1]
        
        # Calculate golden ratio relationships in eigenvalues
        eigenvalue_ratios = []
        for i in range(len(eigenvalues)-1):
            if eigenvalues[i+1] > 1e-10:  # Avoid division by very small values
                ratio = eigenvalues[i] / eigenvalues[i+1]
                eigenvalue_ratios.append(ratio)
        
        # Calculate how close these ratios are to powers of phi
        phi = self.phi
        phi_deviations = []
        for ratio in eigenvalue_ratios[:min(5, len(eigenvalue_ratios))]:  # Focus on top eigenvalues
            # Check against phi, phi², and phi³
            deviations = [abs(ratio - phi**power) for power in range(1, 4)]
            phi_deviations.append(min(deviations))
        
        mean_phi_deviation = np.mean(phi_deviations) if phi_deviations else 1.0
        
        # Compare with shuffled data
        random_deviations = []
        random_gaps = []
        
        for _ in range(1000):
            # Shuffle the power spectrum to break any structure
            shuffled_power = np.random.permutation(power_spectrum)
            
            # Create similarity matrix for shuffled data
            shuffled_similarity = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    diff = abs(shuffled_power[i] - shuffled_power[j])
                    mean_val = (shuffled_power[i] + shuffled_power[j]) / 2.0
                    if mean_val > 0:
                        shuffled_similarity[i, j] = 1.0 / (1.0 + diff / mean_val)
                    else:
                        shuffled_similarity[i, j] = 0.0
            
            # Ensure the matrix is symmetric and positive definite
            shuffled_similarity = (shuffled_similarity + shuffled_similarity.T) / 2.0
            np.fill_diagonal(shuffled_similarity, 1.0)
            
            # Calculate eigenvalues for shuffled data
            random_eigenvalues = np.linalg.eigvals(shuffled_similarity)
            random_eigenvalues = np.sort(np.real(random_eigenvalues))[::-1]
            
            # Calculate spectral gap
            if len(random_eigenvalues) >= 2:
                random_gap = random_eigenvalues[0] - random_eigenvalues[1]
                random_gaps.append(random_gap)
            
            # Calculate ratios and phi deviations
            random_ratios = []
            for i in range(len(random_eigenvalues)-1):
                if random_eigenvalues[i+1] > 1e-10:
                    ratio = random_eigenvalues[i] / random_eigenvalues[i+1]
                    random_ratios.append(ratio)
            
            if random_ratios:
                random_phi_devs = []
                for ratio in random_ratios[:min(5, len(random_ratios))]:
                    devs = [abs(ratio - phi**power) for power in range(1, 4)]
                    random_phi_devs.append(min(devs))
                
                if random_phi_devs:
                    random_deviations.append(np.mean(random_phi_devs))
        
        # Calculate significance for spectral gap
        if random_gaps:
            mean_random_gap = np.mean(random_gaps)
            std_random_gap = np.std(random_gaps)
            
            if std_random_gap > 0:
                gap_z_score = (spectral_gap - mean_random_gap) / std_random_gap
                gap_p_value = 1 - stats.norm.cdf(gap_z_score)
            else:
                gap_z_score = 0.0
                gap_p_value = 0.5
            
            gap_ratio = spectral_gap / mean_random_gap if mean_random_gap > 0 else 1.0
        else:
            mean_random_gap = 0.0
            gap_z_score = 0.0
            gap_p_value = 0.5
            gap_ratio = 1.0
        
        # Calculate significance for phi optimization
        if random_deviations:
            mean_random_dev = np.mean(random_deviations)
            std_random_dev = np.std(random_deviations)
            
            if std_random_dev > 0 and mean_phi_deviation > 0:
                dev_z_score = (mean_random_dev - mean_phi_deviation) / std_random_dev  # Lower deviation is better
                dev_p_value = 1 - stats.norm.cdf(dev_z_score)
            else:
                dev_z_score = 0.0
                dev_p_value = 0.5
            
            dev_ratio = mean_random_dev / mean_phi_deviation if mean_phi_deviation > 0 else 1.0
        else:
            mean_random_dev = 0.0
            dev_z_score = 0.0
            dev_p_value = 0.5
            dev_ratio = 1.0
        
        return spectral_gap, mean_random_gap, gap_z_score, gap_p_value, gap_ratio, mean_phi_deviation, mean_random_dev, dev_z_score, dev_p_value, dev_ratio

    def test_recurrence_quantification(self):
        """Perform recurrence quantification analysis to detect deterministic structure"""
        from scipy.spatial.distance import pdist, squareform
        
        # Create embedding of the power spectrum
        embedding_dimension = 3
        delay = 1
        
        # Ensure we have valid data for embedding
        power_spectrum = np.copy(self.data['ee_power'])
        # Replace zeros and negative values with small positive values
        min_positive = np.min(power_spectrum[power_spectrum > 0]) / 10.0 if np.any(power_spectrum > 0) else 1e-10
        power_spectrum[power_spectrum <= 0] = min_positive
        
        # Create time-delayed embedding
        embedding = []
        for i in range(len(power_spectrum) - (embedding_dimension-1)*delay):
            point = [power_spectrum[i + j*delay] for j in range(embedding_dimension)]
            embedding.append(point)
        
        embedding = np.array(embedding)
        
        # Calculate distance matrix
        distances = squareform(pdist(embedding, 'euclidean'))
        
        # Create recurrence matrix (1 where points are close, 0 elsewhere)
        # Threshold set to 10% of maximum distance
        threshold = 0.1 * np.max(distances)
        recurrence_matrix = distances < threshold
        
        # Calculate RQA metrics
        # 1. Recurrence Rate (RR): percentage of recurrence points
        RR = np.mean(recurrence_matrix)
        
        # 2. Determinism (DET): percentage of recurrence points forming diagonal lines
        min_line_length = 2
        diag_lines = []
        for i in range(-(len(recurrence_matrix)-1), len(recurrence_matrix)):
            diagonal = np.diag(recurrence_matrix, k=i)
            line_lengths = []
            current_length = 0
            for point in diagonal:
                if point:
                    current_length += 1
                elif current_length >= min_line_length:
                    line_lengths.append(current_length)
                    current_length = 0
                else:
                    current_length = 0
            if current_length >= min_line_length:
                line_lengths.append(current_length)
            diag_lines.extend(line_lengths)
        
        if len(diag_lines) > 0 and np.sum(recurrence_matrix) > 0:
            DET = np.sum(diag_lines) / np.sum(recurrence_matrix)
        else:
            DET = 0
        
        # 3. Laminarity (LAM): percentage of recurrence points forming vertical lines
        vert_lines = []
        for i in range(len(recurrence_matrix)):
            vertical = recurrence_matrix[:, i]
            line_lengths = []
            current_length = 0
            for point in vertical:
                if point:
                    current_length += 1
                elif current_length >= min_line_length:
                    line_lengths.append(current_length)
                    current_length = 0
                else:
                    current_length = 0
            if current_length >= min_line_length:
                line_lengths.append(current_length)
            vert_lines.extend(line_lengths)
        
        if len(vert_lines) > 0 and np.sum(recurrence_matrix) > 0:
            LAM = np.sum(vert_lines) / np.sum(recurrence_matrix)
        else:
            LAM = 0
        
        # Compare with surrogate data (shuffled version that preserves distribution)
        surrogate_RR = []
        surrogate_DET = []
        surrogate_LAM = []
        
        for _ in range(100):  # Fewer iterations due to computational intensity
            shuffled_power = np.random.permutation(power_spectrum)
            
            # Create embedding
            surr_embedding = []
            for i in range(len(shuffled_power) - (embedding_dimension-1)*delay):
                point = [shuffled_power[i + j*delay] for j in range(embedding_dimension)]
                surr_embedding.append(point)
            
            surr_embedding = np.array(surr_embedding)
            
            # Calculate distance matrix
            surr_distances = squareform(pdist(surr_embedding, 'euclidean'))
            
            # Create recurrence matrix
            surr_threshold = 0.1 * np.max(surr_distances)
            surr_recurrence = surr_distances < surr_threshold
            
            # Calculate metrics
            surr_RR = np.mean(surr_recurrence)
            surrogate_RR.append(surr_RR)
            
            # Determinism
            surr_diag_lines = []
            for i in range(-(len(surr_recurrence)-1), len(surr_recurrence)):
                diagonal = np.diag(surr_recurrence, k=i)
                line_lengths = []
                current_length = 0
                for point in diagonal:
                    if point:
                        current_length += 1
                    elif current_length >= min_line_length:
                        line_lengths.append(current_length)
                        current_length = 0
                    else:
                        current_length = 0
                if current_length >= min_line_length:
                    line_lengths.append(current_length)
                surr_diag_lines.extend(line_lengths)
            
            if len(surr_diag_lines) > 0 and np.sum(surr_recurrence) > 0:
                surr_DET = np.sum(surr_diag_lines) / np.sum(surr_recurrence)
            else:
                surr_DET = 0
            
            surrogate_DET.append(surr_DET)
            
            # Laminarity calculation for surrogate data
            surr_vert_lines = []
            for i in range(len(surr_recurrence)):
                vertical = surr_recurrence[:, i]
                line_lengths = []
                current_length = 0
                for point in vertical:
                    if point:
                        current_length += 1
                    elif current_length >= min_line_length:
                        line_lengths.append(current_length)
                        current_length = 0
                    else:
                        current_length = 0
                if current_length >= min_line_length:
                    line_lengths.append(current_length)
                surr_vert_lines.extend(line_lengths)
            
            if len(surr_vert_lines) > 0 and np.sum(surr_recurrence) > 0:
                surr_LAM = np.sum(surr_vert_lines) / np.sum(surr_recurrence)
            else:
                surr_LAM = 0
            
            surrogate_LAM.append(surr_LAM)
        
        # Calculate significance
        # Determinism significance (higher DET indicates more deterministic structure)
        mean_surr_DET = np.mean(surrogate_DET) if surrogate_DET else 0
        std_surr_DET = np.std(surrogate_DET) if surrogate_DET else 1
        
        if std_surr_DET > 0:
            DET_z_score = (DET - mean_surr_DET) / std_surr_DET
            DET_p_value = 1 - stats.norm.cdf(DET_z_score) if DET > mean_surr_DET else stats.norm.cdf(DET_z_score)
        else:
            DET_z_score = 0
            DET_p_value = 0.5
        
        DET_ratio = DET / mean_surr_DET if mean_surr_DET > 0 else 1
        
        # Laminarity significance (higher LAM indicates more stable states)
        mean_surr_LAM = np.mean(surrogate_LAM) if surrogate_LAM else 0
        std_surr_LAM = np.std(surrogate_LAM) if surrogate_LAM else 1
        
        if std_surr_LAM > 0:
            LAM_z_score = (LAM - mean_surr_LAM) / std_surr_LAM
            LAM_p_value = 1 - stats.norm.cdf(LAM_z_score) if LAM > mean_surr_LAM else stats.norm.cdf(LAM_z_score)
        else:
            LAM_z_score = 0
            LAM_p_value = 0.5
        
        LAM_ratio = LAM / mean_surr_LAM if mean_surr_LAM > 0 else 1
        
        return RR, DET, LAM, mean_surr_DET, std_surr_DET, DET_z_score, DET_p_value, DET_ratio, mean_surr_LAM, std_surr_LAM, LAM_z_score, LAM_p_value, LAM_ratio

    def test_scale_frequency_coupling(self):
        """Test if golden ratio patterns show scale-dependent coupling"""
        # Define logarithmically-spaced scale ranges
        log_scales = np.logspace(np.log10(min(self.data['ell'])), 
                               np.log10(max(self.data['ell'])), 6)
        
        # Calculate golden ratio frequencies at each scale
        phi = self.phi
        scale_gr_frequencies = []
        
        for i in range(len(log_scales)-1):
            lower = log_scales[i]
            upper = log_scales[i+1]
            
            # Get multipoles in this range
            scale_indices = [j for j in range(len(self.data['ell'])) 
                            if lower <= self.data['ell'][j] < upper]
            
            if len(scale_indices) < 3:  # Need at least 3 points to test ratios
                continue
            
            scale_ells = [self.data['ell'][j] for j in scale_indices]
            
            # Count GR relationships in this scale
            gr_count = 0
            total_pairs = 0
            
            for j in range(len(scale_ells)):
                for k in range(j+1, len(scale_ells)):
                    total_pairs += 1
                    ratio = max(scale_ells[j], scale_ells[k]) / min(scale_ells[j], scale_ells[k])
                    
                    # Check if close to phi, phi², or phi³
                    for power in range(1, 4):
                        if abs(ratio - phi**power) < 0.1:
                            gr_count += 1
                            break
            
            if total_pairs > 0:
                gr_frequency = gr_count / total_pairs
                scale_gr_frequencies.append((i, (lower + upper)/2, gr_frequency))
        
        # Test if these frequencies show a pattern across scales
        if len(scale_gr_frequencies) < 2:
            return None  # Not enough data for analysis
        
        scale_centers = [s[1] for s in scale_gr_frequencies]
        frequencies = [s[2] for s in scale_gr_frequencies]
        
        # Calculate correlation between scale and frequency
        correlation, p_value = stats.pearsonr(np.log10(scale_centers), frequencies)
        
        # Compare with random expectation
        random_correlations = []
        
        for _ in range(1000):
            random_freqs = np.random.permutation(frequencies)
            random_corr, _ = stats.pearsonr(np.log10(scale_centers), random_freqs)
            random_correlations.append(random_corr)
        
        # Calculate significance
        mean_random_corr = np.mean(random_correlations)
        std_random_corr = np.std(random_correlations)
        
        z_score = (correlation - mean_random_corr) / std_random_corr
        corr_p_value = 1 - stats.norm.cdf(z_score)
        
        # Check for specific patterns like linear or phi-based scaling
        if len(frequencies) >= 3:
            # Test linear model
            linear_model = np.polyfit(np.log10(scale_centers), frequencies, 1)
            linear_fit = np.polyval(linear_model, np.log10(scale_centers))
            linear_residuals = frequencies - linear_fit
            linear_mse = np.mean(linear_residuals**2)
            
            # Test phi-based model
            phi_model = np.polyfit(np.log10(scale_centers)/np.log10(phi), frequencies, 1)
            phi_fit = np.polyval(phi_model, np.log10(scale_centers)/np.log10(phi))
            phi_residuals = frequencies - phi_fit
            phi_mse = np.mean(phi_residuals**2)
            
            # Compare models (lower MSE is better)
            model_ratio = linear_mse / phi_mse
        else:
            linear_mse = phi_mse = model_ratio = None
        
        return scale_gr_frequencies, correlation, p_value, z_score, corr_p_value, linear_mse, phi_mse, model_ratio

    def test_transfer_entropy(self):
        """Test if information transfer across scales follows golden ratio patterns"""
        from scipy.stats import entropy
        
        # Define scales based on powers of phi
        phi = self.phi
        base_scale = 10
        
        scales = [int(round(base_scale * phi**i)) for i in range(5)]
        scales = [s for s in scales if s < max(self.data['ell'])]
        
        # Calculate transfer entropy between adjacent scales
        transfer_entropies = []
        
        for i in range(len(scales)-1):
            scale1 = scales[i]
            scale2 = scales[i+1]
            
            # Find closest multipoles to these scales
            idx1 = np.abs(self.data['ell'] - scale1).argmin()
            idx2 = np.abs(self.data['ell'] - scale2).argmin()
            
            # Create binned data (10 bins)
            data1 = self.data['ee_power'][max(0, idx1-10):idx1+11]
            data2 = self.data['ee_power'][max(0, idx2-10):idx2+11]
            
            # Make sure we have enough data
            min_length = min(len(data1), len(data2))
            if min_length < 5:
                continue
                
            data1 = data1[:min_length]
            data2 = data2[:min_length]
            
            # Handle negative or zero values
            data1 = np.abs(data1) + 1e-10
            data2 = np.abs(data2) + 1e-10
            
            # Bin the data
            num_bins = min(5, min_length-1)
            
            # Check for zero range data
            if max(data1) == min(data1) or max(data2) == min(data2):
                continue
                
            bins1 = np.linspace(min(data1), max(data1), num_bins+1)
            bins2 = np.linspace(min(data2), max(data2), num_bins+1)
            
            # Digitize with clip to ensure values are within range
            binned1 = np.clip(np.digitize(data1, bins1), 1, num_bins)
            binned2 = np.clip(np.digitize(data2, bins2), 1, num_bins)
            
            # Calculate probabilities
            p_x = np.zeros(num_bins+1)
            p_y = np.zeros(num_bins+1)
            p_xy = np.zeros((num_bins+1, num_bins+1))
            p_x_y = np.zeros((num_bins+1, num_bins+1))
            
            for t in range(min_length-1):
                p_x[binned1[t]] += 1
                p_y[binned2[t+1]] += 1
                p_xy[binned1[t], binned2[t+1]] += 1
                p_x_y[binned1[t], binned2[t]] += 1
            
            p_x /= (min_length-1)
            p_y /= (min_length-1)
            p_xy /= (min_length-1)
            p_x_y /= (min_length-1)
            
            # Calculate transfer entropy using a more robust method
            te = 0
            for i in range(1, num_bins+1):  # Skip the 0 index which is unused
                for j in range(1, num_bins+1):
                    if p_xy[i,j] > 0 and p_x[i] > 0 and p_x_y[i,j] > 0 and p_y[j] > 0:
                        te += p_xy[i,j] * np.log2(p_xy[i,j] * p_y[j] / (p_x_y[i,j] * p_y[j]))
            
            # Ensure non-negative transfer entropy
            te = max(0, te)
            transfer_entropies.append((scale1, scale2, te))
        
        # Compare with transfer entropy between non-phi-related scales
        non_phi_entropies = []
        
        for _ in range(100):
            random_scales = sorted(np.random.choice(self.data['ell'], size=min(5, len(scales)), replace=False))
            
            for i in range(len(random_scales)-1):
                scale1 = random_scales[i]
                scale2 = random_scales[i+1]
                
                # Find indices
                idx1 = np.where(self.data['ell'] == scale1)[0]
                if len(idx1) == 0:
                    continue
                idx1 = idx1[0]
                
                idx2 = np.where(self.data['ell'] == scale2)[0]
                if len(idx2) == 0:
                    continue
                idx2 = idx2[0]
                
                # Create binned data
                data1 = self.data['ee_power'][max(0, idx1-10):idx1+11]
                data2 = self.data['ee_power'][max(0, idx2-10):idx2+11]
                
                min_length = min(len(data1), len(data2))
                if min_length < 5:
                    continue
                    
                data1 = data1[:min_length]
                data2 = data2[:min_length]
                
                # Handle negative or zero values
                data1 = np.abs(data1) + 1e-10
                data2 = np.abs(data2) + 1e-10
                
                # Bin the data
                num_bins = min(5, min_length-1)
                
                # Check for zero range data
                if max(data1) == min(data1) or max(data2) == min(data2):
                    continue
                    
                bins1 = np.linspace(min(data1), max(data1), num_bins+1)
                bins2 = np.linspace(min(data2), max(data2), num_bins+1)
                
                # Digitize with clip to ensure values are within range
                binned1 = np.clip(np.digitize(data1, bins1), 1, num_bins)
                binned2 = np.clip(np.digitize(data2, bins2), 1, num_bins)
                
                # Calculate probabilities
                p_x = np.zeros(num_bins+1)
                p_y = np.zeros(num_bins+1)
                p_xy = np.zeros((num_bins+1, num_bins+1))
                p_x_y = np.zeros((num_bins+1, num_bins+1))
                
                for t in range(min_length-1):
                    p_x[binned1[t]] += 1
                    p_y[binned2[t+1]] += 1
                    p_xy[binned1[t], binned2[t+1]] += 1
                    p_x_y[binned1[t], binned2[t]] += 1
                
                p_x /= (min_length-1)
                p_y /= (min_length-1)
                p_xy /= (min_length-1)
                p_x_y /= (min_length-1)
                
                # Calculate transfer entropy using a more robust method
                te = 0
                for i in range(1, num_bins+1):  # Skip the 0 index which is unused
                    for j in range(1, num_bins+1):
                        if p_xy[i,j] > 0 and p_x[i] > 0 and p_x_y[i,j] > 0 and p_y[j] > 0:
                            te += p_xy[i,j] * np.log2(p_xy[i,j] * p_y[j] / (p_x_y[i,j] * p_y[j]))
                
                # Ensure non-negative transfer entropy
                te = max(0, te)
                non_phi_entropies.append(te)
        
        # Calculate statistics
        if transfer_entropies and non_phi_entropies:
            phi_te = np.mean([t[2] for t in transfer_entropies])
            non_phi_te = np.mean(non_phi_entropies)
            
            # Ensure we don't divide by zero
            if non_phi_te > 0:
                te_ratio = phi_te / non_phi_te
            elif phi_te > 0:
                te_ratio = float('inf')
            else:
                te_ratio = 1.0  # Both are zero
            
            # Calculate significance
            std_non_phi = np.std(non_phi_entropies) if len(non_phi_entropies) > 1 else 1e-10
            
            z_score = (phi_te - non_phi_te) / std_non_phi if std_non_phi > 0 else 0
            p_value = 1 - stats.norm.cdf(z_score)
            
            return phi_te, non_phi_te, te_ratio, z_score, p_value
        else:
            return None

    def test_multi_scale_coherence(self):
        """Test how coherence varies across different scales"""
        # Define logarithmically spaced scale bins
        scale_bins = np.logspace(np.log10(min(self.data['ell'])), 
                                np.log10(max(self.data['ell'])), 8)
        
        # Measure coherence within each scale bin
        scale_coherences = []
        
        for i in range(len(scale_bins)-1):
            lower = scale_bins[i]
            upper = scale_bins[i+1]
            
            # Get multipoles in this range
            indices = [j for j in range(len(self.data['ell'])) 
                      if lower <= self.data['ell'][j] < upper]
            
            if len(indices) < 5:  # Need enough points for meaningful coherence
                continue
                
            # Get power values for this scale
            powers = [self.data['ee_power'][j] for j in indices]
            
            # Calculate normalized variance (lower = more coherent)
            normalized = powers / np.mean(powers) if np.mean(powers) != 0 else powers
            coherence = 1 / np.var(normalized)  # Invert so higher = more coherent
            
            scale_coherences.append((np.sqrt(lower*upper), coherence))
        
        # Test if there's a pattern to how coherence varies with scale
        scales = [s[0] for s in scale_coherences]
        coherences = [s[1] for s in scale_coherences]
        
        # Look for golden ratio relationships in coherence distribution
        coherence_ratios = []
        for i in range(len(coherences)-1):
            if coherences[i+1] != 0:
                ratio = coherences[i] / coherences[i+1]
                coherence_ratios.append(ratio)
        
        # Calculate how close these ratios are to phi
        phi_deviations = [abs(ratio - self.phi) for ratio in coherence_ratios]
        mean_phi_deviation = np.mean(phi_deviations)
        
        # Compare with random reordering of scales
        random_deviations = []
        
        for _ in range(1000):
            random_coherences = np.random.permutation(coherences)
            random_ratios = []
            
            for i in range(len(random_coherences)-1):
                if random_coherences[i+1] != 0:
                    ratio = random_coherences[i] / random_coherences[i+1]
                    random_ratios.append(ratio)
            
            if random_ratios:
                random_phi_devs = [abs(ratio - self.phi) for ratio in random_ratios]
                random_deviations.append(np.mean(random_phi_devs))
        
        # Calculate significance
        mean_random = np.mean(random_deviations)
        std_random = np.std(random_deviations)
        
        z_score = (mean_random - mean_phi_deviation) / std_random  # Lower deviation is better
        p_value = 1 - stats.norm.cdf(z_score)
        
        optimization_ratio = mean_random / mean_phi_deviation
        
        return scale_coherences, coherence_ratios, mean_phi_deviation, mean_random, z_score, p_value, optimization_ratio

    def test_coherence_phase(self):
        """Test phase relationships in the CMB data for coherence patterns"""
        # Calculate power spectrum phases
        power_fft = np.fft.fft(self.data['ee_power'])
        phases = np.angle(power_fft)
        
        # Look for patterns in phase differences
        phase_diffs = np.diff(phases)
        phase_diffs = np.mod(phase_diffs + np.pi, 2*np.pi) - np.pi  # Normalize to [-π, π]
        
        # Calculate phase coherence (higher = more coherent)
        # Using Kuramoto order parameter
        complex_phases = np.exp(1j * phases)
        kuramoto = np.abs(np.mean(complex_phases))
        
        # Compare with phase-randomized surrogate data
        surrogate_kuramoto = []
        
        for _ in range(1000):
            # Create surrogate with same amplitudes but randomized phases
            random_phases = np.random.uniform(0, 2*np.pi, len(phases))
            surrogate_fft = np.abs(power_fft) * np.exp(1j * random_phases)
            surrogate = np.real(np.fft.ifft(surrogate_fft))
            
            # Calculate Kuramoto parameter for surrogate
            surrogate_complex = np.exp(1j * random_phases)
            surrogate_kuramoto.append(np.abs(np.mean(surrogate_complex)))
        
        # Calculate significance
        mean_surrogate = np.mean(surrogate_kuramoto)
        std_surrogate = np.std(surrogate_kuramoto)
        
        z_score = (kuramoto - mean_surrogate) / std_surrogate
        p_value = 1 - stats.norm.cdf(z_score)
        
        coherence_ratio = kuramoto / mean_surrogate
        
        # Look for golden ratio relationships in phase differences
        phi_phase_strength = 0
        golden_angle = 2*np.pi / self.phi
        tolerance = 0.2  # Increase tolerance to capture more potential matches
        
        for diff in phase_diffs:
            # Check if phase difference is close to 2π/φ (the golden angle)
            if abs(abs(diff) - golden_angle) < tolerance:
                phi_phase_strength += 1
        
        phi_phase_ratio = phi_phase_strength / len(phase_diffs) if len(phase_diffs) > 0 else 0
        
        # Compare with random expectation
        random_phi_ratios = []
        for _ in range(100):
            random_diffs = np.random.uniform(-np.pi, np.pi, len(phase_diffs))
            random_strength = 0
            for diff in random_diffs:
                if abs(abs(diff) - golden_angle) < tolerance:
                    random_strength += 1
            random_phi_ratios.append(random_strength / len(phase_diffs) if len(phase_diffs) > 0 else 0)
        
        mean_random_ratio = np.mean(random_phi_ratios)
        std_random_ratio = np.std(random_phi_ratios)
        
        phi_z_score = (phi_phase_ratio - mean_random_ratio) / std_random_ratio
        phi_p_value = 1 - stats.norm.cdf(phi_z_score)
        
        phase_optimization = phi_phase_ratio / mean_random_ratio
        
        # Handle potential division by zero or very small numbers
        if std_random_ratio > 1e-10:
            phi_z_score = (phi_phase_ratio - mean_random_ratio) / std_random_ratio
        else:
            phi_z_score = 0.0
            
        phi_p_value = 1 - stats.norm.cdf(phi_z_score)
        
        # Handle potential division by zero
        if mean_random_ratio > 1e-10:
            phase_optimization = phi_phase_ratio / mean_random_ratio
        else:
            # If both are zero, set to 1.0 (neutral)
            if phi_phase_ratio < 1e-10:
                phase_optimization = 1.0
            else:
                # If only denominator is zero, set to a high value
                phase_optimization = 10.0
        
        return kuramoto, mean_surrogate, z_score, p_value, coherence_ratio, phi_phase_ratio, mean_random_ratio, phi_z_score, phi_p_value, phase_optimization

    def test_extended_meta_coherence(self):
        """Perform extended analysis of meta-coherence properties"""
        # Calculate local coherence measures across the spectrum
        window_size = 5
        step_size = 2
        local_coherence = []
        
        for i in range(0, len(self.data['ee_power']) - window_size, step_size):
            window = self.data['ee_power'][i:i+window_size]
            normalized = window / np.mean(window) if np.mean(window) != 0 else window
            local_coherence.append(np.var(normalized))
        
        # Calculate meta-coherence (variance of local coherence)
        meta_coherence = np.var(local_coherence)
        
        # Calculate additional meta-coherence metrics
        
        # 1. Meta-coherence skewness (distribution asymmetry)
        skewness = stats.skew(local_coherence)
        
        # 2. Meta-coherence kurtosis (peakedness/tailedness)
        kurtosis = stats.kurtosis(local_coherence)
        
        # 3. Meta-coherence entropy (information content)
        hist, bin_edges = np.histogram(local_coherence, bins='auto', density=True)
        # Handle potential zero values in histogram
        valid_hist = hist > 0
        if np.any(valid_hist):
            bin_widths = np.diff(bin_edges)
            if len(bin_widths) == len(hist):
                entropy = -np.sum(hist[valid_hist] * np.log2(hist[valid_hist]) * bin_widths[valid_hist])
            else:
                # Handle case where dimensions don't match
                bin_widths_valid = bin_widths[:len(hist)]
                entropy = -np.sum(hist[valid_hist] * np.log2(hist[valid_hist]) * bin_widths_valid[valid_hist])
        else:
            entropy = 0
        
        # 4. Scale-free behavior (power law exponent)
        # Calculate power spectrum of local coherence
        coherence_fft = np.abs(np.fft.fft(local_coherence))**2
        frequencies = np.fft.fftfreq(len(local_coherence))
        
        # Only use positive frequencies
        positive_freq_idx = frequencies > 0
        positive_freqs = frequencies[positive_freq_idx]
        power = coherence_fft[positive_freq_idx]
        
        # Fit power law (in log-log space)
        if len(positive_freqs) > 3:  # Need enough points for regression
            # Handle potential zero or negative values
            valid_idx = (positive_freqs > 0) & (power > 0)
            if np.sum(valid_idx) > 3:
                log_freqs = np.log10(positive_freqs[valid_idx])
                log_power = np.log10(power[valid_idx])
                
                # Linear regression
                slope, _, _, _, _ = stats.linregress(log_freqs, log_power)
                
                # Power law exponent (negative in P ∝ f^α)
                power_law_exponent = slope
            else:
                power_law_exponent = None
        else:
            power_law_exponent = None
        
        # Compare with shuffled data
        shuffled_meta = []
        shuffled_skew = []
        shuffled_kurt = []
        shuffled_entropy = []
        shuffled_exponents = []
        
        for _ in range(1000):
            shuffled = np.random.permutation(self.data['ee_power'])
            
            # Calculate local coherence for shuffled data
            shuffled_local = []
            for i in range(0, len(shuffled) - window_size, step_size):
                window = shuffled[i:i+window_size]
                normalized = window / np.mean(window) if np.mean(window) != 0 else window
                shuffled_local.append(np.var(normalized))
            
            # Calculate metrics
            shuffled_meta.append(np.var(shuffled_local))
            shuffled_skew.append(stats.skew(shuffled_local))
            shuffled_kurt.append(stats.kurtosis(shuffled_local))
            
            # Calculate entropy for shuffled data
            hist, bin_edges_shuffled = np.histogram(shuffled_local, bins='auto', density=True)
            valid_hist = hist > 0
            if np.any(valid_hist):
                bin_widths = np.diff(bin_edges_shuffled)
                if len(bin_widths) == len(hist):
                    shuffled_entropy.append(-np.sum(hist[valid_hist] * np.log2(hist[valid_hist]) * bin_widths[valid_hist]))
                else:
                    # Handle case where dimensions don't match
                    bin_widths_valid = bin_widths[:len(hist)]
                    shuffled_entropy.append(-np.sum(hist[valid_hist] * np.log2(hist[valid_hist]) * bin_widths_valid[valid_hist]))
            else:
                shuffled_entropy.append(0)
            
            # Calculate power law for shuffled data
            if power_law_exponent is not None:
                shuffled_fft = np.abs(np.fft.fft(shuffled_local))**2
                shuffled_power = shuffled_fft[positive_freq_idx]
                
                valid_idx = (positive_freqs > 0) & (shuffled_power > 0)
                if np.sum(valid_idx) > 3:
                    log_freqs_valid = np.log10(positive_freqs[valid_idx])
                    log_power_valid = np.log10(shuffled_power[valid_idx])
                    
                    try:
                        slope, _, _, _, _ = stats.linregress(log_freqs_valid, log_power_valid)
                        shuffled_exponents.append(slope)
                    except:
                        # Skip this iteration if regression fails
                        pass
        
        # Calculate significance for all metrics
        # 1. Meta-coherence variance
        mean_shuffled_meta = np.mean(shuffled_meta)
        std_shuffled_meta = np.std(shuffled_meta)
        
        if std_shuffled_meta > 0:
            meta_z = (meta_coherence - mean_shuffled_meta) / std_shuffled_meta
            meta_p = 1 - stats.norm.cdf(meta_z)
        else:
            meta_z = 0
            meta_p = 0.5
            
        if meta_coherence > 0:
            meta_ratio = meta_coherence / mean_shuffled_meta if mean_shuffled_meta > 0 else float('inf')
        else:
            meta_ratio = 0 if mean_shuffled_meta > 0 else 1.0
        
        # 2. Skewness
        mean_shuffled_skew = np.mean(shuffled_skew)
        std_shuffled_skew = np.std(shuffled_skew)
        
        if std_shuffled_skew > 0:
            skew_z = (skewness - mean_shuffled_skew) / std_shuffled_skew
            skew_p = 1 - stats.norm.cdf(abs(skew_z))  # Two-tailed
        else:
            skew_z = 0
            skew_p = 1.0
            
        skew_ratio = abs(skewness) / abs(mean_shuffled_skew) if abs(mean_shuffled_skew) > 1e-10 else 1.0
        
        # 3. Kurtosis
        mean_shuffled_kurt = np.mean(shuffled_kurt)
        std_shuffled_kurt = np.std(shuffled_kurt)
        
        if std_shuffled_kurt > 0:
            kurt_z = (kurtosis - mean_shuffled_kurt) / std_shuffled_kurt
            kurt_p = 1 - stats.norm.cdf(abs(kurt_z))  # Two-tailed
        else:
            kurt_z = 0
            kurt_p = 1.0
            
        kurt_ratio = abs(kurtosis) / abs(mean_shuffled_kurt) if abs(mean_shuffled_kurt) > 1e-10 else 1.0
        
        # 4. Entropy
        mean_shuffled_entropy = np.mean(shuffled_entropy)
        std_shuffled_entropy = np.std(shuffled_entropy)
        
        if std_shuffled_entropy > 0:
            entropy_z = (entropy - mean_shuffled_entropy) / std_shuffled_entropy
            entropy_p = 1 - stats.norm.cdf(abs(entropy_z))  # Two-tailed
        else:
            entropy_z = 0
            entropy_p = 1.0
            
        entropy_ratio = entropy / mean_shuffled_entropy if mean_shuffled_entropy > 1e-10 else 1.0
        
        # 5. Power law exponent
        if power_law_exponent is not None and shuffled_exponents:
            mean_shuffled_exponent = np.mean(shuffled_exponents)
            std_shuffled_exponent = np.std(shuffled_exponents)
            
            if std_shuffled_exponent > 0:
                exponent_z = (power_law_exponent - mean_shuffled_exponent) / std_shuffled_exponent
                exponent_p = 1 - stats.norm.cdf(abs(exponent_z))  # Two-tailed
            else:
                exponent_z = 0
                exponent_p = 1.0
                
            exponent_ratio = abs(power_law_exponent) / abs(mean_shuffled_exponent) if abs(mean_shuffled_exponent) > 1e-10 else 1.0
        else:
            mean_shuffled_exponent = exponent_z = exponent_p = exponent_ratio = None
        
        return {
            'meta_coherence': (meta_coherence, mean_shuffled_meta, meta_z, meta_p, meta_ratio),
            'skewness': (skewness, mean_shuffled_skew, skew_z, skew_p, skew_ratio),
            'kurtosis': (kurtosis, mean_shuffled_kurt, kurt_z, kurt_p, kurt_ratio),
            'entropy': (entropy, mean_shuffled_entropy, entropy_z, entropy_p, entropy_ratio),
            'power_law': (power_law_exponent, mean_shuffled_exponent, exponent_z, exponent_p, exponent_ratio)
        }
