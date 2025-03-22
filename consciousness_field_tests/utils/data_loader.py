#!/usr/bin/env python3
"""
Data loader for Cosmic Consciousness Analysis.
"""

import os
import numpy as np
from astropy.io import fits
import warnings

class DataLoader:
    """Class for loading and preprocessing CMB data."""
    
    def __init__(self, data_dir="planck_data"):
        """
        Initialize the data loader with a data directory.
        
        Parameters:
        -----------
        data_dir : str
            Directory containing Planck CMB data files
        """
        self.data_dir = data_dir
        self.data = {}
        self.phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        self.gr_multipoles = []
        
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
                with fits.open(cov_file) as hdul:
                    self.data['cov_matrix'] = hdul[0].data
                print(f"Loaded covariance matrix with shape {self.data['cov_matrix'].shape}")
            except Exception as e:
                # Try alternate FITS location
                try:
                    alt_cov_file = os.path.join("planck_data", "COM_PowerSpect_CMB-CovMatrix_R3.01.fits")
                    with fits.open(alt_cov_file) as hdul:
                        self.data['cov_matrix'] = hdul[0].data
                    print(f"Loaded covariance matrix from alternate location with shape {self.data['cov_matrix'].shape}")
                except Exception as e2:
                    # Try numpy format
                    try:
                        npy_cov_file = os.path.join(self.data_dir, "cov_matrix.npy")
                        self.data['cov_matrix'] = np.load(npy_cov_file)
                        print(f"Loaded numpy covariance matrix with shape {self.data['cov_matrix'].shape}")
                    except Exception as e3:
                        print(f"Warning: Could not load covariance matrix")
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
    
    def get_golden_ratio_multipoles(self):
        """
        Get the list of multipoles related by the golden ratio.
        
        Returns:
        --------
        list
            List of multipoles related by the golden ratio
        """
        return self.gr_multipoles
    
    def get_data(self):
        """
        Get the loaded data.
        
        Returns:
        --------
        dict
            Dictionary containing the loaded data
        """
        return self.data
    
    def get_power_at_multipole(self, multipole):
        """
        Get the power at a specific multipole.
        
        Parameters:
        -----------
        multipole : int
            Multipole value
            
        Returns:
        --------
        float
            Power at the specified multipole
        """
        # Find the closest multipole
        idx = np.abs(self.data['ell'] - multipole).argmin()
        return self.data['ee_power'][idx]
    
    def get_power_in_range(self, min_multipole, max_multipole):
        """
        Get the power spectrum in a range of multipoles.
        
        Parameters:
        -----------
        min_multipole : int
            Minimum multipole value
        max_multipole : int
            Maximum multipole value
            
        Returns:
        --------
        tuple
            Tuple containing (multipoles, power, error) in the specified range
        """
        # Find indices in the range
        indices = np.where((self.data['ell'] >= min_multipole) & (self.data['ell'] <= max_multipole))[0]
        
        # Extract data
        multipoles = self.data['ell'][indices]
        power = self.data['ee_power'][indices]
        error = self.data['ee_error'][indices]
        
        return multipoles, power, error

# Convenience function to get a data loader instance
def get_data_loader(data_dir=None):
    """
    Get a configured data loader instance.
    
    Parameters:
    -----------
    data_dir : str, optional
        Directory containing Planck CMB data files
        
    Returns:
    --------
    DataLoader
        Configured data loader instance
    """
    if data_dir is None:
        # Default to the planck_data directory in the project root
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'planck_data')
    
    return DataLoader(data_dir=data_dir)
