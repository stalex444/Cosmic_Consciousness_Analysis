#!/usr/bin/env python3
"""
Module for handling Planck mission data for cosmic consciousness analysis.
"""

import os
import numpy as np
from astropy.io import fits
import requests
import tqdm
import matplotlib.pyplot as plt
from astropy.visualization import astropy_mpl_style

# Set up Matplotlib with Astropy style
plt.style.use(astropy_mpl_style)

class PlanckDataHandler:
    """Class for handling Planck mission data."""
    
    def __init__(self, data_dir="planck_data"):
        """Initialize the data handler with a data directory."""
        self.data_dir = data_dir
        self.ensure_data_directory()
        
        # Planck data URLs
        self.data_urls = {
            "cmb_temperature": "https://pla.esac.esa.int/pla/aio/product-action?MAP.MAP_ID=COM_CMB_IQU-smica_1024_R2.02_full.fits",
            "lensing_potential": "https://pla.esac.esa.int/pla/aio/product-action?MAP.MAP_ID=COM_Lensing_Mask_2048_R2.00.fits",
            "dust_temperature": "https://pla.esac.esa.int/pla/aio/product-action?MAP.MAP_ID=HFI_CompMap_ThermalDustModel_2048_R1.20.fits"
        }
        
    def ensure_data_directory(self):
        """Ensure the data directory exists."""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            
    def download_data(self, data_type):
        """
        Download Planck data of a specific type.
        
        Parameters:
        -----------
        data_type : str
            Type of data to download, must be a key in self.data_urls
            
        Returns:
        --------
        str
            Path to the downloaded file
        """
        if data_type not in self.data_urls:
            raise ValueError(f"Unknown data type: {data_type}. Available types: {list(self.data_urls.keys())}")
            
        url = self.data_urls[data_type]
        filename = f"planck_{data_type}.fits"
        filepath = os.path.join(self.data_dir, filename)
        
        if os.path.exists(filepath):
            print(f"File {filename} already exists.")
            return filepath
            
        print(f"Downloading {filename} from Planck archive...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024
        
        with open(filepath, 'wb') as file, tqdm.tqdm(
                desc=filename,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
            for data in response.iter_content(block_size):
                file.write(data)
                bar.update(len(data))
                
        print(f"Downloaded {filename} successfully.")
        return filepath
        
    def load_fits_data(self, filepath):
        """
        Load data from a FITS file.
        
        Parameters:
        -----------
        filepath : str
            Path to the FITS file
            
        Returns:
        --------
        tuple
            (header, data) from the FITS file
        """
        with fits.open(filepath) as hdul:
            # Get the header and data from the first HDU
            header = hdul[0].header
            data = hdul[0].data
            
        return header, data
        
    def visualize_map(self, data, title="Planck Map", cmap="viridis", save_path=None):
        """
        Visualize a Planck map.
        
        Parameters:
        -----------
        data : numpy.ndarray
            Map data to visualize
        title : str
            Title for the plot
        cmap : str
            Colormap to use
        save_path : str, optional
            Path to save the visualization
        """
        plt.figure(figsize=(12, 8))
        plt.imshow(data, cmap=cmap)
        plt.colorbar(label='Temperature (K)')
        plt.title(title)
        plt.grid(False)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.show()
        
    def extract_time_series(self, map_data, coordinates_list):
        """
        Extract time series data from map at specific coordinates.
        
        Parameters:
        -----------
        map_data : numpy.ndarray
            Map data
        coordinates_list : list of tuples
            List of (row, col) coordinates to extract
            
        Returns:
        --------
        numpy.ndarray
            Array of time series data
        """
        time_series = []
        
        for coords in coordinates_list:
            row, col = coords
            # Extract a small patch around the coordinates
            patch_size = 5
            row_start = max(0, row - patch_size//2)
            row_end = min(map_data.shape[0], row + patch_size//2 + 1)
            col_start = max(0, col - patch_size//2)
            col_end = min(map_data.shape[1], col + patch_size//2 + 1)
            
            patch = map_data[row_start:row_end, col_start:col_end]
            # Flatten the patch to create a time series
            time_series.append(patch.flatten())
            
        return np.array(time_series)
        
def main():
    """Main function to demonstrate Planck data handling."""
    handler = PlanckDataHandler()
    print("Planck Data Handler initialized.")
    
    # Note: Downloading actual Planck data requires authentication
    # and the URLs provided are examples. In a real scenario,
    # you would need to register with the Planck Legacy Archive.
    print("Note: To download actual Planck data, you need to register with the Planck Legacy Archive.")
    
    # Instead, let's create some synthetic data for demonstration
    print("Creating synthetic CMB temperature map for demonstration...")
    
    # Create a synthetic CMB-like map (512x512 pixels)
    np.random.seed(42)
    size = 512
    
    # Generate a random field
    k = np.fft.fftfreq(size) * size
    kx, ky = np.meshgrid(k, k)
    k2 = kx**2 + ky**2
    
    # Create a power spectrum that roughly mimics CMB
    power = np.zeros_like(k2)
    power[1:] = k2[1:] ** (-2.1)  # Power law spectrum
    
    # Generate random phases
    phases = np.random.uniform(0, 2*np.pi, size=(size, size))
    
    # Create complex Fourier components
    fourier = np.sqrt(power) * np.exp(1j * phases)
    
    # Inverse FFT to get the map
    cmb_map = np.real(np.fft.ifft2(fourier))
    
    # Scale to typical CMB temperature fluctuations (μK)
    cmb_map = cmb_map * 100 / np.std(cmb_map)
    
    # Save the synthetic map
    synthetic_map_path = os.path.join(handler.data_dir, 'synthetic_cmb.npy')
    np.save(synthetic_map_path, cmb_map)
    print(f"Saved synthetic CMB map to {synthetic_map_path}")
    
    # Visualize the synthetic map
    plt.figure(figsize=(10, 8))
    plt.imshow(cmb_map, cmap='RdBu_r')
    plt.colorbar(label='Temperature fluctuations (μK)')
    plt.title('Synthetic CMB Temperature Map')
    plt.grid(False)
    plt.savefig(os.path.join(handler.data_dir, 'synthetic_cmb_map.png'), dpi=300, bbox_inches='tight')
    
    print("Synthetic CMB map visualization saved to planck_data/synthetic_cmb_map.png")
    
    # Extract some time series data for analysis
    print("Extracting time series data from synthetic map...")
    
    # Define some random coordinates
    np.random.seed(0)
    num_points = 10
    coordinates = [(np.random.randint(0, size), np.random.randint(0, size)) 
                   for _ in range(num_points)]
    
    time_series = handler.extract_time_series(cmb_map, coordinates)
    
    # Save the time series data
    np.save(os.path.join(handler.data_dir, 'synthetic_time_series.npy'), time_series)
    print(f"Saved synthetic time series data to {os.path.join(handler.data_dir, 'synthetic_time_series.npy')}")
    
    print("\nPlanck data handling demonstration complete.")

if __name__ == "__main__":
    main()
