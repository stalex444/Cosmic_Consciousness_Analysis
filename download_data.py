#!/usr/bin/env python
"""
Data Download Script for Cosmic Consciousness Analysis

This script downloads the necessary large data files for the Cosmic Consciousness Analysis project.
Run this script after cloning the repository to download all required data files.
"""

import os
import sys
import argparse
import hashlib
from urllib.request import urlretrieve
import tarfile
import zipfile
import shutil

# Define the data files and their properties
DATA_FILES = {
    "planck_data": {
        "url": "https://zenodo.org/record/XXXXXXX/files/planck_data.tar.gz",  # Replace with actual URL
        "md5": "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX",  # Replace with actual MD5
        "size_mb": 500,  # Approximate size in MB
        "extract_dir": "planck_data",
        "description": "Planck CMB data files"
    },
    "wmap_data": {
        "url": "https://zenodo.org/record/XXXXXXX/files/wmap_data.tar.gz",  # Replace with actual URL
        "md5": "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX",  # Replace with actual MD5
        "size_mb": 300,  # Approximate size in MB
        "extract_dir": "wmap_data",
        "description": "WMAP CMB data files"
    }
    # Add more data files as needed
}

def calculate_md5(file_path):
    """Calculate MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def download_file(url, destination, description):
    """Download a file with progress reporting."""
    def report_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(100, downloaded * 100 / total_size)
        sys.stdout.write(f"\r{description}: {percent:.1f}% ({downloaded / 1024 / 1024:.1f} MB / {total_size / 1024 / 1024:.1f} MB)")
        sys.stdout.flush()
    
    print(f"Downloading {description}...")
    urlretrieve(url, destination, report_progress)
    print()  # New line after progress

def extract_archive(archive_path, extract_dir):
    """Extract a tar.gz or zip archive."""
    print(f"Extracting to {extract_dir}...")
    
    if archive_path.endswith('.tar.gz'):
        with tarfile.open(archive_path) as tar:
            tar.extractall(path=extract_dir)
    elif archive_path.endswith('.zip'):
        with zipfile.ZipFile(archive_path) as zip_ref:
            zip_ref.extractall(extract_dir)
    
    print(f"Extraction complete.")

def download_and_extract(data_key, force=False):
    """Download and extract a specific data file."""
    data_info = DATA_FILES[data_key]
    download_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "downloads")
    extract_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), data_info["extract_dir"])
    
    # Create download directory if it doesn't exist
    os.makedirs(download_dir, exist_ok=True)
    
    # File paths
    file_name = os.path.basename(data_info["url"])
    download_path = os.path.join(download_dir, file_name)
    
    # Check if extraction directory already exists
    if os.path.exists(extract_dir) and not force:
        print(f"{data_info['description']} already exists at {extract_dir}")
        return
    
    # Check if file already downloaded
    if os.path.exists(download_path) and not force:
        print(f"File already downloaded: {download_path}")
        if data_info["md5"] != "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX":  # Skip check for placeholder MD5
            print("Verifying file integrity...")
            if calculate_md5(download_path) == data_info["md5"]:
                print("File integrity verified.")
            else:
                print("File integrity check failed. Re-downloading...")
                os.remove(download_path)
                download_file(data_info["url"], download_path, data_info["description"])
    else:
        # Download the file
        download_file(data_info["url"], download_path, data_info["description"])
    
    # Extract the file
    os.makedirs(extract_dir, exist_ok=True)
    extract_archive(download_path, extract_dir)

def main():
    parser = argparse.ArgumentParser(description="Download data files for Cosmic Consciousness Analysis")
    parser.add_argument("--all", action="store_true", help="Download all data files")
    parser.add_argument("--force", action="store_true", help="Force re-download and extraction")
    
    for data_key in DATA_FILES:
        parser.add_argument(f"--{data_key}", action="store_true", help=f"Download {DATA_FILES[data_key]['description']}")
    
    args = parser.parse_args()
    
    # If no specific arguments provided, download all
    if not any([getattr(args, data_key) for data_key in DATA_FILES]) and not args.all:
        args.all = True
    
    # Download selected or all data files
    for data_key in DATA_FILES:
        if args.all or getattr(args, data_key):
            download_and_extract(data_key, force=args.force)
    
    print("\nData download complete!")
    print("You can now run the analysis scripts.")

if __name__ == "__main__":
    main()
