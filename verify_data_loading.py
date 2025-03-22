#!/usr/bin/env python3
"""
Simple script to verify data loading and basic calculations.
This script loads the Planck data and performs minimal calculations to verify everything works.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

def main():
    """Main function to verify data loading."""
    print("Verifying data loading...")
    
    # Define data directory
    data_dir = "planck_data"
    
    # Try to load the EE spectrum data
    try:
        # Import EE power spectrum
        ee_file = os.path.join(data_dir, "power_spectra", "COM_PowerSpect_CMB-EE-binned_R3.02.txt")
        print(f"Looking for file: {os.path.abspath(ee_file)}")
        
        if os.path.exists(ee_file):
            print(f"File exists: {ee_file}")
            
            # Load the data
            ee_data = np.loadtxt(ee_file)
            print(f"Data loaded successfully with shape: {ee_data.shape}")
            
            # Extract ell and power
            ell = ee_data[:, 0]
            ee_power = ee_data[:, 1]
            
            print(f"First 5 multipoles: {ell[:5]}")
            print(f"First 5 power values: {ee_power[:5]}")
            
            # Calculate basic statistics
            mean_power = np.mean(ee_power)
            std_power = np.std(ee_power)
            
            print(f"Mean power: {mean_power}")
            print(f"Std power: {std_power}")
            
            # Create a simple plot
            plt.figure(figsize=(10, 6))
            plt.plot(ell, ee_power, 'b-')
            plt.xlabel('Multipole (ℓ)')
            plt.ylabel('EE Power')
            plt.title('Planck EE Power Spectrum')
            plt.grid(True)
            
            # Save the plot
            output_dir = "data_verification"
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, "ee_power_spectrum.png"))
            plt.close()
            
            print(f"Plot saved to {os.path.join(output_dir, 'ee_power_spectrum.png')}")
            
            # Save the data statistics
            with open(os.path.join(output_dir, "data_stats.txt"), "w") as f:
                f.write("Planck EE Power Spectrum Statistics\n")
                f.write("==================================\n\n")
                f.write(f"Number of multipoles: {len(ell)}\n")
                f.write(f"Multipole range: {ell.min()} - {ell.max()}\n")
                f.write(f"Mean power: {mean_power}\n")
                f.write(f"Std power: {std_power}\n")
                f.write(f"Min power: {ee_power.min()}\n")
                f.write(f"Max power: {ee_power.max()}\n")
            
            print(f"Statistics saved to {os.path.join(output_dir, 'data_stats.txt')}")
            
            print("Data verification completed successfully!")
            return True
        else:
            print(f"File does not exist: {ee_file}")
            return False
    
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()
