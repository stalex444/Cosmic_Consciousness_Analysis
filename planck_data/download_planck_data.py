#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Planck Data Downloader

This script downloads the necessary Planck CMB data files from the Planck Legacy Archive
for use with the Cosmic Consciousness Analysis framework.
"""

from __future__ import print_function
import os
import sys
import urllib2
import hashlib
import time

# Python 2.7 compatibility
try:
    input = raw_input
except NameError:
    pass

# URLs for Planck data files
PLANCK_DATA_URLS = {
    'TT': 'https://pla.esac.esa.int/pla/aio/product-action?COSMOLOGY.FILE_ID=COM_PowerSpect_CMB-TT-full_R3.01.txt',
    'EE': 'https://pla.esac.esa.int/pla/aio/product-action?COSMOLOGY.FILE_ID=COM_PowerSpect_CMB-EE-full_R3.01.txt',
    'TE': 'https://pla.esac.esa.int/pla/aio/product-action?COSMOLOGY.FILE_ID=COM_PowerSpect_CMB-TE-full_R3.01.txt'
}

# Expected file sizes (in bytes) for verification
EXPECTED_SIZES = {
    'TT': 58000,  # Approximate size
    'EE': 58000,  # Approximate size
    'TE': 58000   # Approximate size
}

# Output directory structure
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'power_spectra')


def create_directories():
    """Create the necessary directories if they don't exist."""
    if not os.path.exists(OUTPUT_DIR):
        print("Creating directory: {}".format(OUTPUT_DIR))
        os.makedirs(OUTPUT_DIR)


def download_file(url, output_path, description):
    """
    Download a file from a URL with progress reporting.
    
    Args:
        url (str): URL to download
        output_path (str): Path to save the file
        description (str): Description of the file for progress reporting
    
    Returns:
        bool: True if download was successful, False otherwise
    """
    try:
        print("Downloading {} data...".format(description))
        
        # Open the URL
        response = urllib2.urlopen(url)
        total_size = int(response.info().getheader('Content-Length', 0))
        
        # Check if file already exists with correct size
        if os.path.exists(output_path):
            if os.path.getsize(output_path) == total_size:
                print("File already exists and has correct size. Skipping download.")
                return True
            else:
                print("File exists but has incorrect size. Re-downloading...")
        
        # Download the file with progress reporting
        bytes_downloaded = 0
        block_size = 8192
        with open(output_path, 'wb') as f:
            while True:
                buffer = response.read(block_size)
                if not buffer:
                    break
                
                bytes_downloaded += len(buffer)
                f.write(buffer)
                
                # Calculate and display progress
                if total_size > 0:
                    percent = bytes_downloaded * 100.0 / total_size
                    status = "\r[{:.2f}%] [{}/{}]".format(
                        percent, bytes_downloaded, total_size)
                    sys.stdout.write(status)
                    sys.stdout.flush()
        
        print("\nDownload complete: {}".format(output_path))
        return True
    
    except Exception as e:
        print("Error downloading {}: {}".format(description, str(e)))
        return False


def verify_file(file_path, expected_size):
    """
    Verify that a file exists and has approximately the expected size.
    
    Args:
        file_path (str): Path to the file
        expected_size (int): Expected file size in bytes
    
    Returns:
        bool: True if file exists and has approximately the expected size
    """
    if not os.path.exists(file_path):
        print("Error: File not found: {}".format(file_path))
        return False
    
    actual_size = os.path.getsize(file_path)
    # Allow for some variation in file size (±10%)
    size_tolerance = 0.1
    size_min = expected_size * (1 - size_tolerance)
    size_max = expected_size * (1 + size_tolerance)
    
    if size_min <= actual_size <= size_max:
        print("File verified: {} ({} bytes)".format(file_path, actual_size))
        return True
    else:
        print("Warning: File size mismatch for {}".format(file_path))
        print("  Expected: approximately {} bytes".format(expected_size))
        print("  Actual: {} bytes".format(actual_size))
        return False


def main():
    """Main function to download and verify Planck data files."""
    print("=" * 80)
    print("Planck CMB Data Downloader")
    print("=" * 80)
    print("This script will download the Planck CMB power spectrum data files")
    print("required for the Cosmic Consciousness Analysis framework.")
    print("\nData will be saved to: {}".format(OUTPUT_DIR))
    print("=" * 80)
    
    # Create necessary directories
    create_directories()
    
    # Ask user which files to download
    download_tt = True
    download_ee = input("\nDownload E-mode polarization (EE) spectrum? (y/n) [n]: ").lower() == 'y'
    download_te = input("Download temperature-polarization (TE) spectrum? (y/n) [n]: ").lower() == 'y'
    
    print("\nStarting downloads...")
    
    # Download TT spectrum (always required)
    tt_path = os.path.join(OUTPUT_DIR, 'COM_PowerSpect_CMB-TT-full_R3.01.txt')
    tt_success = download_file(PLANCK_DATA_URLS['TT'], tt_path, "temperature (TT)")
    
    # Download EE spectrum (optional)
    ee_success = True
    if download_ee:
        ee_path = os.path.join(OUTPUT_DIR, 'COM_PowerSpect_CMB-EE-full_R3.01.txt')
        ee_success = download_file(PLANCK_DATA_URLS['EE'], ee_path, "E-mode polarization (EE)")
    
    # Download TE spectrum (optional)
    te_success = True
    if download_te:
        te_path = os.path.join(OUTPUT_DIR, 'COM_PowerSpect_CMB-TE-full_R3.01.txt')
        te_success = download_file(PLANCK_DATA_URLS['TE'], te_path, "temperature-polarization (TE)")
    
    print("\nVerifying downloads...")
    
    # Verify downloads
    tt_verified = verify_file(tt_path, EXPECTED_SIZES['TT']) if tt_success else False
    ee_verified = verify_file(ee_path, EXPECTED_SIZES['EE']) if download_ee and ee_success else True
    te_verified = verify_file(te_path, EXPECTED_SIZES['TE']) if download_te and te_success else True
    
    # Summary
    print("\n" + "=" * 80)
    print("Download Summary:")
    print("  Temperature (TT) spectrum: {}".format("SUCCESS" if tt_verified else "FAILED"))
    if download_ee:
        print("  E-mode polarization (EE) spectrum: {}".format("SUCCESS" if ee_verified else "FAILED"))
    if download_te:
        print("  Temperature-polarization (TE) spectrum: {}".format("SUCCESS" if te_verified else "FAILED"))
    
    # Final message
    if tt_verified and ee_verified and te_verified:
        print("\nAll downloads completed successfully!")
        print("\nYou can now run the analysis with:")
        print("  python run_analysis.py --all --no-simulated --data-file={}".format(tt_path))
    else:
        print("\nSome downloads failed. Please check the error messages above.")
    
    print("=" * 80)


if __name__ == "__main__":
    main()
