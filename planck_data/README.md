# Planck CMB Data

This directory contains utilities for working with data from the Planck Cosmic Microwave Background (CMB) mission. The Planck satellite, operated by the European Space Agency (ESA), has provided the most precise measurements of the CMB to date.

## Data Overview

The Planck mission has produced several data releases, with the latest being the Planck 2018 data release (PR3). The data includes:

- **CMB Power Spectra**: Angular power spectra for temperature and polarization
- **CMB Maps**: Full-sky maps of the CMB temperature and polarization
- **Foreground Maps**: Maps of galactic and extragalactic emission
- **Likelihood Code**: Software for cosmological parameter estimation

For the Cosmic Consciousness Analysis framework, we primarily use the CMB power spectra, particularly the temperature (TT) power spectrum.

## Downloading Planck Data

### Option 1: Manual Download

1. Visit the [Planck Legacy Archive (PLA)](https://pla.esac.esa.int/)
2. Navigate to "Power Spectra" → "CMB Spectra"
3. Download the following files:
   - `COM_PowerSpect_CMB-TT-full_R3.01.txt` (Temperature power spectrum, unbinned)
   - `COM_PowerSpect_CMB-EE-full_R3.01.txt` (Optional: E-mode polarization power spectrum)
   - `COM_PowerSpect_CMB-TE-full_R3.01.txt` (Optional: Temperature-E-mode cross-correlation)

4. Place the downloaded files in the `planck_data/power_spectra/` directory (create it if it doesn't exist)

### Option 2: Automated Download

Run the provided download script:

```bash
# Make sure you're in the project root directory
python planck_data/download_planck_data.py
```

This script will:
1. Create the necessary directories
2. Download the required power spectrum files
3. Verify the integrity of the downloaded files

## Data Format

### Power Spectrum Files

The power spectrum files are ASCII text files with the following columns:

- **Column 1**: Multipole moment (ℓ)
- **Column 2**: Power spectrum value (Dℓ = ℓ(ℓ+1)Cℓ/2π in μK²)
- **Column 3**: Lower error bar
- **Column 4**: Upper error bar

Example of the first few lines of `COM_PowerSpect_CMB-TT-full_R3.01.txt`:
```
# ell D_ell D_ell_low D_ell_up
2 1.25304e+03 7.20285e+02 2.08717e+03
3 1.11169e+03 7.68249e+02 1.56683e+03
4 1.17399e+03 9.08203e+02 1.49201e+03
...
```

## Using Planck Data in the Analysis

To use the Planck data in your analysis:

1. Ensure the data files are in the `planck_data/power_spectra/` directory
2. Run the analysis with the `--no-simulated` flag and specify the data file:

```bash
python run_analysis.py --all --no-simulated --data-file=planck_data/power_spectra/COM_PowerSpect_CMB-TT-full_R3.01.txt
```

The `planck_data_handler.py` module provides functions for loading and preprocessing the Planck data:

- `load_planck_power_spectrum()`: Loads the power spectrum from a file
- `preprocess_planck_data()`: Applies necessary preprocessing steps
- `get_planck_data()`: Returns the processed data ready for analysis

## Additional Resources

- [Planck Legacy Archive](https://pla.esac.esa.int/): Official repository for Planck data
- [Planck 2018 Results](https://www.aanda.org/component/toc/?task=topic&id=320): Scientific papers describing the Planck 2018 data release
- [NASA/IPAC Infrared Science Archive (IRSA)](https://irsa.ipac.caltech.edu/Missions/planck.html): Alternative source for Planck data

## Troubleshooting

If you encounter issues with the Planck data:

1. **Missing Files**: Ensure you've downloaded the correct files and placed them in the right directory
2. **Format Errors**: Check that the files haven't been corrupted during download
3. **Import Errors**: Make sure you have the required dependencies (numpy, scipy) installed

For detailed error messages, run the analysis with the `--verbose` flag:

```bash
python run_analysis.py --all --no-simulated --data-file=planck_data/power_spectra/COM_PowerSpect_CMB-TT-full_R3.01.txt --verbose
```

## Directory Structure

- `power_spectra/`: Contains CMB power spectra files
  - `COM_PowerSpect_CMB-TT-full_R3.01.txt`: Temperature power spectrum (unbinned)
  - `COM_PowerSpect_CMB-EE-full_R3.01.txt`: E-mode polarization power spectrum (unbinned)
  - `COM_PowerSpect_CMB-TE-full_R3.01.txt`: Temperature-E-mode cross-correlation power spectrum (unbinned)
  - `COM_PowerSpect_CMB-low-ell-BB-full_R3.01.txt`: B-mode polarization power spectrum (unbinned)
  - `COM_PowerSpect_CMB-TT-binned_R3.01.txt`: Temperature power spectrum (binned)
  - `COM_PowerSpect_CMB-EE-binned_R3.02.txt`: E-mode polarization power spectrum (binned)
  - `COM_PowerSpect_CMB-TE-binned_R3.02.txt`: Temperature-E-mode cross-correlation power spectrum (binned)

- `maps/`: Contains CMB maps
  - `COM_CMB_IQU-smica_1024_R2.02_full.fits`: Full-sky CMB map (SMICA method)
  - `COM_CMB_IQU-commander_1024_R2.02_full.fits`: Full-sky CMB map (Commander method)

- `likelihood/`: Contains likelihood data
  - `COM_Likelihood_Data-baseline_R3.00.tar.gz`: Full likelihood data package
  - `covariance_matrix.dat`: Covariance matrix extracted from the likelihood package

- Root directory files:
  - `cosmological_parameters.txt`: Best-fit cosmological parameters
  - `base_power_spectrum.txt`: Base power spectrum
  - `COM_PowerSpect_CMB-CovMatrix_R3.01.fits`: Covariance matrix (note: may require Planck Legacy Archive authentication)
  - `planck_data_handler.py`: Module for importing and preprocessing Planck CMB data

## Using the Planck Data Handler

The `planck_data_handler.py` module provides functions for downloading, loading, and preprocessing Planck CMB data for analysis. Here's how to use it:

### 1. Download Planck Data

```python
from planck_data.planck_data_handler import download_planck_data

# Download SMICA map at low resolution
output_dir = "planck_data/maps"
filepath = download_planck_data(output_dir, map_type='SMICA', resolution='R1')
```

### 2. Load a CMB Map

```python
from planck_data.planck_data_handler import load_planck_map

# Load temperature map (field=0)
cmb_map = load_planck_map(filepath, field=0)
```

### 3. Extract Power Spectrum

```python
from planck_data.planck_data_handler import extract_power_spectrum

# Extract power spectrum up to lmax=2500
power_spectrum = extract_power_spectrum(cmb_map, lmax=2500)
```

### 4. Preprocess for Analysis

```python
from planck_data.planck_data_handler import preprocess_for_analysis

# Preprocess map for analysis
data = preprocess_for_analysis(cmb_map, n_points=4096, seed=42)
```

### 5. Visualize Data

```python
from planck_data.planck_data_handler import visualize_cmb_map, visualize_power_spectrum

# Visualize CMB map
visualize_cmb_map(cmb_map, title="Planck CMB Map", output_path="planck_map.png")

# Visualize power spectrum
visualize_power_spectrum(power_spectrum, title="CMB Power Spectrum", output_path="power_spectrum.png")
```

### 6. Using Real Planck Data with the Analysis Framework

```python
# In run_analysis.py
from planck_data.planck_data_handler import load_planck_map, preprocess_for_analysis

# Load and preprocess Planck data
filepath = "planck_data/maps/COM_CMB_IQU-smica_1024_R2.02_full.fits"
cmb_map = load_planck_map(filepath)
data = preprocess_for_analysis(cmb_map, n_points=4096, seed=args.seed)

# Run analysis with real data
analyzer = CosmicConsciousnessAnalyzer(output_dir=args.data_dir)
# ... rest of analysis code ...

## Using Alternative Data Sources

The Cosmic Consciousness Analysis framework is designed to be flexible and can work with various types of spectral or time-series data beyond the Planck CMB data. Here's how to use alternative data sources:

### Custom Data Format Requirements

Any custom data should adhere to the following requirements:

1. **Data Structure**: The data should be a 1D array or a 2D array where the first column represents the x-axis (e.g., multipole moment, frequency, or time) and the second column represents the y-axis (e.g., power, amplitude).

2. **File Format**: The data can be in various formats:
   - ASCII text files (space, tab, or comma-delimited)
   - NumPy `.npy` files
   - HDF5 files
   - FITS files (common in astronomy)

3. **Data Preprocessing**: Ensure your data is:
   - Free from missing values or NaN
   - Properly normalized if necessary
   - Filtered for noise if applicable

### Example: Using WMAP Data

The Wilkinson Microwave Anisotropy Probe (WMAP) provides another source of CMB data:

1. Download WMAP power spectrum data from [Lambda Archive](https://lambda.gsfc.nasa.gov/product/wmap/dr5/powspec_get.cfm)
2. Format the data to match the expected structure (multipole vs. power)
3. Run the analysis with:
   ```bash
   python run_analysis.py --all --no-simulated --data-file=path/to/wmap_data.txt
   ```

### Example: Using Custom Spectral Data

For custom spectral data (e.g., EEG, seismic, or other time-series data):

1. Prepare your data file with columns for frequency/time and power/amplitude
2. Implement a custom data loader in `core_framework/data_handler.py` if needed
3. Run the analysis with:
   ```bash
   python run_analysis.py --all --no-simulated --data-file=path/to/custom_data.txt --custom-loader
   ```

### Implementing a Custom Data Loader

To add support for a new data format:

1. Open `core_framework/data_handler.py`
2. Add a new function following this template:
   ```python
   def load_custom_data(file_path):
       """
       Load data from a custom format.
       
       Args:
           file_path (str): Path to the data file
           
       Returns:
           numpy.ndarray: The loaded data
       """
       # Your loading code here
       # Example:
       import numpy as np
       data = np.loadtxt(file_path, delimiter=',')
       # Process data if needed
       return data
   ```
3. Register your loader in the `load_data` function

## Data Preprocessing Options

The framework offers several preprocessing options that can be applied to any data source:

### Smoothing

Apply smoothing to reduce noise in the data:

```bash
python run_analysis.py --all --data-file=path/to/data.txt --smooth --smooth-window=5
```

### Normalization

Normalize the data to a specific range:

```bash
python run_analysis.py --all --data-file=path/to/data.txt --normalize
```

### Detrending

Remove linear or polynomial trends from the data:

```bash
python run_analysis.py --all --data-file=path/to/data.txt --detrend --detrend-order=1
```

### Frequency Filtering

Apply bandpass, lowpass, or highpass filters:

```bash
python run_analysis.py --all --data-file=path/to/data.txt --filter --filter-type=bandpass --filter-low=10 --filter-high=100
```

## Advanced Data Analysis

For more advanced data analysis scenarios:

### Batch Processing

Process multiple data files in batch mode:

```bash
python run_analysis.py --batch --batch-dir=path/to/data_directory --output-dir=path/to/results
```

### Comparative Analysis

Compare results from different data sources:

```bash
python run_analysis.py --compare --data-files=file1.txt,file2.txt,file3.txt --labels=source1,source2,source3
```

### Custom Test Parameters

Customize test parameters for specific data characteristics:

```bash
python run_analysis.py --all --data-file=path/to/data.txt --custom-params=params.json
```

Where `params.json` contains test-specific parameters:

```json
{
  "golden_ratio_test": {
    "phi_bias": 0.05,
    "n_surrogates": 2000
  },
  "fractal_analysis_test": {
    "scales": [4, 8, 16, 32, 64],
    "method": "dfa"
  }
}
```

## Data Visualization Options

Customize the visualization of your data:

```bash
python run_analysis.py --all --data-file=path/to/data.txt --plot-style=publication --color-scheme=vibrant --dpi=600
```

Available options:
- Plot styles: `default`, `publication`, `presentation`, `minimal`
- Color schemes: `default`, `vibrant`, `pastel`, `grayscale`, `colorblind`
- Output formats: `png`, `pdf`, `svg`, `eps`

For more details on data handling and customization options, refer to the documentation in the `core_framework/data_handler.py` module.

## Notes

The E-mode polarization power spectrum (COM_PowerSpect_CMB-EE-full_R3.01.txt) is particularly important for cosmic consciousness analysis as it contains information about the polarization patterns in the early universe.

The covariance matrix file (COM_PowerSpect_CMB-CovMatrix_R3.01.fits) may require authentication with the Planck Legacy Archive to download directly. An alternative covariance matrix has been extracted from the likelihood package.

## Dependencies

To work with Planck data, you'll need the following Python packages:
- healpy
- astropy
- numpy
- scipy
- matplotlib

You can install them with:
```bash
pip install healpy astropy numpy scipy matplotlib
