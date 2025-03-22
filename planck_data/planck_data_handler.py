#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Planck Data Handler
-----------------
Module for importing and preprocessing Planck CMB data.
"""

import os
import numpy as np
import healpy as hp
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from core_framework.constants import DEFAULT_SEED


def download_planck_data(output_dir, map_type='SMICA', resolution='R1'):
    """
    Download Planck CMB data from the Planck Legacy Archive.
    
    Args:
        output_dir (str): Directory to save downloaded data
        map_type (str, optional): Type of map to download. Options: 'SMICA', 'NILC', 'SEVEM', 'Commander'.
            Defaults to 'SMICA'.
        resolution (str, optional): Resolution of the map. Options: 'R1' (low), 'R2' (high).
            Defaults to 'R1'.
            
    Returns:
        str: Path to downloaded file
    """
    import urllib
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define base URL for Planck Legacy Archive
    base_url = "https://pla.esac.esa.int/pla/aio/product-action?MAP.MAP_ID="
    
    # Define map IDs based on type and resolution
    map_ids = {
        'SMICA': {
            'R1': 'COM_CMB_IQU-smica_1024_R2.02_full.fits',
            'R2': 'COM_CMB_IQU-smica_2048_R2.02_full.fits'
        },
        'NILC': {
            'R1': 'COM_CMB_IQU-nilc_1024_R2.02_full.fits',
            'R2': 'COM_CMB_IQU-nilc_2048_R2.02_full.fits'
        },
        'SEVEM': {
            'R1': 'COM_CMB_IQU-sevem_1024_R2.02_full.fits',
            'R2': 'COM_CMB_IQU-sevem_2048_R2.02_full.fits'
        },
        'Commander': {
            'R1': 'COM_CMB_IQU-commander_1024_R2.02_full.fits',
            'R2': 'COM_CMB_IQU-commander_2048_R2.02_full.fits'
        }
    }
    
    # Get map ID
    try:
        map_id = map_ids[map_type][resolution]
    except KeyError:
        raise ValueError("Invalid map_type or resolution. map_type must be one of 'SMICA', 'NILC', 'SEVEM', 'Commander'. resolution must be one of 'R1', 'R2'.")
    
    # Define URL
    url = base_url + map_id
    
    # Define output file path
    output_file = os.path.join(output_dir, map_id)
    
    # Download file if it doesn't exist
    if not os.path.exists(output_file):
        print("Downloading Planck CMB data from {}".format(url))
        print("This may take a while...")
        
        try:
            urllib.request.urlretrieve(url, output_file)
            print("Download complete. File saved to {}".format(output_file))
        except Exception as e:
            print("Error downloading file: {}".format(e))
            return None
    else:
        print("File already exists at {}".format(output_file))
    
    return output_file


def load_planck_map(filepath, field=0):
    """
    Load a Planck CMB map from a FITS file.
    
    Args:
        filepath (str): Path to FITS file
        field (int, optional): Field to load. 0=I (temperature), 1=Q, 2=U (polarization).
            Defaults to 0 (temperature).
            
    Returns:
        ndarray: HEALPix map
    """
    # Check if file exists
    if not os.path.exists(filepath):
        raise FileNotFoundError("File not found: {}".format(filepath))
    
    # Load map
    try:
        cmb_map = hp.read_map(filepath, field=field)
        print("Loaded Planck CMB map with Nside = {}".format(hp.get_nside(cmb_map)))
        return cmb_map
    except Exception as e:
        print("Error loading map: {}".format(e))
        return None


def extract_power_spectrum(cmb_map, lmax=2500):
    """
    Extract the power spectrum from a CMB map.
    
    Args:
        cmb_map (ndarray): HEALPix map
        lmax (int, optional): Maximum multipole. Defaults to 2500.
            
    Returns:
        ndarray: Power spectrum Cl
    """
    # Calculate power spectrum
    cl = hp.anafast(cmb_map, lmax=lmax)
    
    # Convert to Dl = l(l+1)Cl/(2π)
    l = np.arange(len(cl))
    dl = l * (l + 1) * cl / (2 * np.pi)
    
    return dl


def apply_mask(cmb_map, mask_path):
    """
    Apply a mask to a CMB map.
    
    Args:
        cmb_map (ndarray): HEALPix map
        mask_path (str): Path to mask file
            
    Returns:
        ndarray: Masked HEALPix map
    """
    # Load mask
    mask = hp.read_map(mask_path)
    
    # Ensure mask and map have the same Nside
    nside_map = hp.get_nside(cmb_map)
    nside_mask = hp.get_nside(mask)
    
    if nside_map != nside_mask:
        mask = hp.ud_grade(mask, nside_map)
    
    # Apply mask
    masked_map = cmb_map.copy()
    masked_map[mask == 0] = hp.UNSEEN
    
    return masked_map


def extract_1d_slice(cmb_map, nside=None, n_points=4096, seed=DEFAULT_SEED):
    """
    Extract a 1D slice from a CMB map for analysis.
    
    Args:
        cmb_map (ndarray): HEALPix map
        nside (int, optional): HEALPix Nside parameter. Defaults to None (use map's Nside).
        n_points (int, optional): Number of points in the output array. Defaults to 4096.
        seed (int, optional): Random seed for reproducibility. Defaults to DEFAULT_SEED.
            
    Returns:
        ndarray: 1D array of CMB temperature values
    """
    # Set random seed
    np.random.seed(seed)
    
    # Get map's Nside if not provided
    if nside is None:
        nside = hp.get_nside(cmb_map)
    
    # Generate random points on the sphere
    theta = np.arccos(2 * np.random.random(n_points) - 1)
    phi = 2 * np.pi * np.random.random(n_points)
    
    # Convert to pixel indices
    pixels = hp.ang2pix(nside, theta, phi)
    
    # Extract values
    values = cmb_map[pixels]
    
    # Handle masked values
    mask = values != hp.UNSEEN
    values = values[mask]
    
    # Ensure we have enough points
    if len(values) < n_points:
        # If we don't have enough points, resample
        indices = np.random.choice(len(values), n_points, replace=True)
        values = values[indices]
    elif len(values) > n_points:
        # If we have too many points, truncate
        values = values[:n_points]
    
    # Normalize
    values = (values - np.mean(values)) / np.std(values)
    
    return values


def visualize_cmb_map(cmb_map, title="Planck CMB Map", output_path=None):
    """
    Visualize a CMB map.
    
    Args:
        cmb_map (ndarray): HEALPix map
        title (str, optional): Plot title. Defaults to "Planck CMB Map".
        output_path (str, optional): Path to save the plot. Defaults to None.
            
    Returns:
        None
    """
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot map
    hp.mollview(cmb_map, title=title, unit="mK", min=-500, max=500)
    hp.graticule()
    
    # Save or show
    if output_path is not None:
        plt.savefig(output_path, dpi=300)
        plt.close()
        print("Plot saved to {}".format(output_path))
    else:
        plt.show()


def visualize_power_spectrum(cl, title="CMB Power Spectrum", output_path=None):
    """
    Visualize a CMB power spectrum.
    
    Args:
        cl (ndarray): Power spectrum Cl or Dl
        title (str, optional): Plot title. Defaults to "CMB Power Spectrum".
        output_path (str, optional): Path to save the plot. Defaults to None.
            
    Returns:
        None
    """
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Plot power spectrum
    l = np.arange(len(cl))
    plt.plot(l, cl, 'b-')
    
    # Set labels and title
    plt.xlabel(r"Multipole $\ell$")
    plt.ylabel(r"$D_\ell = \ell(\ell+1)C_\ell/2\pi$ [$\mu K^2$]")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    # Set log scale for x-axis
    plt.xscale('log')
    
    # Save or show
    if output_path is not None:
        plt.savefig(output_path, dpi=300)
        plt.close()
        print("Plot saved to {}".format(output_path))
    else:
        plt.show()


def preprocess_for_analysis(cmb_map, n_points=4096, seed=DEFAULT_SEED):
    """
    Preprocess a CMB map for analysis.
    
    Args:
        cmb_map (ndarray): HEALPix map
        n_points (int, optional): Number of points in the output array. Defaults to 4096.
        seed (int, optional): Random seed for reproducibility. Defaults to DEFAULT_SEED.
            
    Returns:
        ndarray: Preprocessed 1D array ready for analysis
    """
    # Extract 1D slice
    data = extract_1d_slice(cmb_map, n_points=n_points, seed=seed)
    
    # Normalize
    data = (data - np.mean(data)) / np.std(data)
    
    return data
