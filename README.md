# Cosmic Consciousness Analysis

This project analyzes Cosmic Microwave Background (CMB) data to investigate patterns related to consciousness and universal constants, with a particular focus on the golden ratio.

## Overview

The Cosmic Consciousness Analyzer implements multiple statistical tests to evaluate whether the CMB data exhibits patterns that could indicate evidence of conscious organization or design. The analysis includes:

1. **Golden Ratio Test**: Examines whether multipoles related by the golden ratio (φ) have statistically significant power.
2. **Coherence Analysis Test**: Evaluates if the CMB spectrum shows more coherence than random chance.
3. **GR-Specific Coherence Test**: Tests coherence specifically in golden ratio related regions.
4. **Hierarchical Organization Test**: Checks for hierarchical patterns based on the golden ratio.
5. **Information Integration Test**: Measures mutual information between adjacent spectrum regions.
6. **Scale Transition Test**: Analyzes scale boundaries where organizational principles change.
7. **Resonance Analysis Test**: Tests for resonance patterns in the power spectrum.
8. **Fractal Analysis Test**: Uses the Hurst exponent to evaluate fractal behavior.
9. **Meta-Coherence Test**: Analyzes coherence of local coherence measures.
10. **Transfer Entropy Test**: Measures information flow between scales.

## Features

- **Comprehensive Statistical Analysis**: 10 different statistical tests to evaluate various aspects of cosmic consciousness.
- **Parallel Processing**: Efficient execution of multiple tests simultaneously using multi-core processing.
- **Visualization Dashboard**: Interactive and static visualization tools to interpret and communicate results.
- **Planck CMB Data Support**: Integration with actual Planck satellite CMB data.
- **Customizable Analysis**: Flexible configuration of tests, data sources, and analysis parameters.
- **Python 2.7 Compatible**: Fully compatible with Python 2.7 for legacy system support.

## Directory Structure

The project is organized into the following directory structure:

```
Cosmic_Consciousness_Analysis/
├── analysis/                    # Analysis integration layer
│   ├── __init__.py
│   └── analysis.py              # Main analyzer class
├── core_framework/              # Core framework layer
│   ├── __init__.py
│   ├── base_test.py             # Base test class
│   ├── constants.py             # Mathematical and configuration constants
│   ├── data_handler.py          # Data loading and preprocessing
│   ├── statistics.py            # Statistical functions
│   └── visualization.py         # Visualization utilities
├── tests/                       # Test implementation layer
│   ├── __init__.py
│   ├── coherence_tests/         # Tests related to coherence
│   │   ├── __init__.py
│   │   ├── coherence_analysis_test.py
│   │   ├── gr_specific_coherence_test.py
│   │   └── meta_coherence_test.py
│   ├── information_tests/       # Tests related to information theory
│   │   ├── __init__.py
│   │   ├── information_integration_test.py
│   │   └── transfer_entropy_test.py
│   ├── scale_tests/             # Tests related to scale transitions
│   │   ├── __init__.py
│   │   └── scale_transition_test.py
│   └── structural_tests/        # Tests related to structural properties
│       ├── __init__.py
│       ├── fractal_analysis_test.py
│       ├── golden_ratio_test.py
│       ├── hierarchical_organization_test.py
│       └── resonance_analysis_test.py
├── planck_data/                 # Planck CMB data handling
│   ├── __init__.py
│   ├── planck_data_handler.py   # Functions for loading and processing Planck data
│   └── README.md                # Instructions for downloading Planck data
├── visualization/               # Visualization tools
│   ├── __init__.py
│   └── dashboard.py             # Dashboard for visualizing results
├── results/                     # Directory for storing analysis results
├── run_analysis.py              # Main script to run the analysis
└── requirements.txt             # Project dependencies
```

## Installation

### Prerequisites

- Python 2.7 (required)
- pip (for installing dependencies)
- git (for cloning the repository)

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/Cosmic_Consciousness_Analysis.git
cd Cosmic_Consciousness_Analysis
```

### Step 2: Set Up a Virtual Environment (Optional but Recommended)

```bash
# Install virtualenv if you don't have it
pip install virtualenv

# Create a virtual environment
virtualenv venv

# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

The `requirements.txt` file includes:

```
numpy==1.16.6
scipy==1.2.3
matplotlib==2.2.5
pandas==0.24.2
seaborn==0.9.1
tqdm==4.64.1
```

## Downloading Planck CMB Data

The analysis can be run with simulated data (default) or with actual Planck CMB data.

### Option 1: Using Simulated Data

By default, the analysis uses simulated data. No additional downloads are required.

### Option 2: Using Planck CMB Data

To use actual Planck CMB data:

1. Visit the [Planck Legacy Archive](https://pla.esac.esa.int/)
2. Download the CMB power spectrum data (COM_PowerSpect_CMB-TT-full_R3.01.txt)
3. Place the downloaded file in the `planck_data/` directory

Alternatively, you can use the provided script to download the data:

```bash
python planck_data/download_planck_data.py
```

For more detailed instructions, see the README in the `planck_data/` directory.

## Usage

### Basic Usage

Run the main script to execute tests:

```bash
# Run all tests with simulated data
python run_analysis.py --all

# Run only the Golden Ratio test
python run_analysis.py --golden-ratio

# Run with actual Planck data
python run_analysis.py --all --no-simulated --data-file=planck_data/COM_PowerSpect_CMB-TT-full_R3.01.txt
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--data-dir` | Directory to save results | 'results' |
| `--simulated` | Use simulated data | True |
| `--no-simulated` | Use actual data instead of simulated | False |
| `--data-file` | Path to data file (if not using simulated data) | None |
| `--data-size` | Size of simulated data | 4096 |
| `--seed` | Random seed for reproducibility | 42 |
| `--phi-bias` | Bias factor for golden ratio tests | 0.1 |
| `--golden-ratio` | Run golden ratio test | False |
| `--coherence-analysis` | Run coherence analysis test | False |
| `--gr-specific-coherence` | Run GR-specific coherence test | False |
| `--hierarchical-organization` | Run hierarchical organization test | False |
| `--information-integration` | Run information integration test | False |
| `--scale-transition` | Run scale transition test | False |
| `--resonance-analysis` | Run resonance analysis test | False |
| `--fractal-analysis` | Run fractal analysis test | False |
| `--meta-coherence` | Run meta-coherence test | False |
| `--transfer-entropy` | Run transfer entropy test | False |
| `--all` | Run all tests | False |
| `--visualize` | Generate visualizations | True |
| `--no-visualize` | Don't generate visualizations | False |
| `--report` | Generate detailed reports | True |
| `--no-report` | Don't generate reports | False |
| `--parallel` | Use parallel processing | True |
| `--no-parallel` | Don't use parallel processing | False |
| `--n-jobs` | Number of parallel jobs | -1 (all cores) |

### Examples

```bash
# Run all tests with a larger simulated dataset and specific random seed
python run_analysis.py --all --data-size=8192 --seed=123

# Run only the Golden Ratio and Fractal Analysis tests
python run_analysis.py --golden-ratio --fractal-analysis

# Run with custom output directory
python run_analysis.py --all --data-dir=my_results

# Run without parallel processing (useful for debugging)
python run_analysis.py --all --no-parallel

# Run with a specific number of parallel jobs
python run_analysis.py --all --parallel --n-jobs=4

# Run with verbose output for debugging
python run_analysis.py --all --verbose

# Run with a specific subset of tests
python run_analysis.py --golden-ratio --coherence-analysis --meta-coherence
```

## Advanced Usage

### Using Alternative Data Sources

The framework is designed to be flexible and can analyze various types of spectral or time-series data beyond CMB:

#### Example: EEG Data Analysis

```bash
# Analyze EEG data for consciousness patterns
python run_analysis.py --all --no-simulated --data-file=path/to/eeg_data.txt --custom-loader
```

#### Example: Seismic Data Analysis

```bash
# Analyze seismic data for golden ratio patterns
python run_analysis.py --golden-ratio --resonance-analysis --no-simulated --data-file=path/to/seismic_data.csv
```

#### Example: Financial Time Series

```bash
# Analyze stock market data for fractal patterns
python run_analysis.py --fractal-analysis --hierarchical-organization --no-simulated --data-file=path/to/market_data.csv
```

### Batch Processing

Process multiple files in a single run:

```bash
# Analyze multiple data files
python run_analysis.py --all --batch --batch-dir=path/to/data_directory --output-dir=results/batch_analysis
```

### Comparative Analysis

Compare results from different data sources:

```bash
# Compare multiple data sources
python run_analysis.py --all --compare --data-files=file1.txt,file2.txt,file3.txt --labels=source1,source2,source3
```

### Custom Test Parameters

Customize test parameters for specific data characteristics:

```bash
# Use custom parameters
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

## Data Preprocessing Options

The framework offers several preprocessing options:

```bash
# Apply smoothing to reduce noise
python run_analysis.py --all --data-file=path/to/data.txt --smooth --smooth-window=5

# Normalize the data
python run_analysis.py --all --data-file=path/to/data.txt --normalize

# Remove linear trends
python run_analysis.py --all --data-file=path/to/data.txt --detrend --detrend-order=1

# Apply frequency filtering
python run_analysis.py --all --data-file=path/to/data.txt --filter --filter-type=bandpass --filter-low=10 --filter-high=100
```

## Visualization Options

Customize the visualization of your results:

```bash
# Generate publication-quality figures
python run_analysis.py --all --plot-style=publication --color-scheme=vibrant --dpi=600

# Generate minimal visualizations for quick analysis
python run_analysis.py --all --plot-style=minimal --output-format=pdf
```

Available options:
- Plot styles: `default`, `publication`, `presentation`, `minimal`
- Color schemes: `default`, `vibrant`, `pastel`, `grayscale`, `colorblind`
- Output formats: `png`, `pdf`, `svg`, `eps`

## Implementing Custom Tests

You can extend the framework with custom tests by following these steps:

1. Create a new test class that inherits from `BaseTest` in the appropriate category directory
2. Implement the required methods: `run`, `visualize`, and `report`
3. Register the test in `run_analysis.py`

Example of a custom test implementation:

```python
from core_framework.base_test import BaseTest

class MyCustomTest(BaseTest):
    """
    Custom test implementation.
    """
    
    def __init__(self, data, output_dir=None, **kwargs):
        super(MyCustomTest, self).__init__(data, output_dir, **kwargs)
        self.name = "my_custom_test"
        self.description = "My custom test description"
        
    def run(self):
        """Run the test and return results."""
        # Implement your test logic here
        # ...
        
        # Calculate phi optimality
        self.phi_optimality = self._calculate_phi_optimality()
        
        # Calculate significance
        self.significance = self._calculate_significance()
        
        return {
            'phi_optimality': self.phi_optimality,
            'significance': self.significance,
            # Other results...
        }
        
    def visualize(self):
        """Generate visualizations for the test results."""
        # Implement visualization logic
        # ...
        
    def report(self):
        """Generate a detailed report of the test results."""
        # Implement reporting logic
        # ...
        
    def _calculate_phi_optimality(self):
        """Calculate phi optimality score."""
        # Implement calculation
        # ...
        
    def _calculate_significance(self):
        """Calculate statistical significance."""
        # Implement calculation
        # ...
```

## Technical Details

### Phi Optimality Calculation

The phi optimality score measures how closely a pattern aligns with the golden ratio (φ ≈ 1.618). The score ranges from -1 to 1:

- **1**: Perfect alignment with the golden ratio
- **0**: No particular alignment
- **-1**: Anti-alignment with the golden ratio

The calculation uses the formula:

```
phi_optimality = 1 - 2 * min(|observed_ratio - φ|/φ, 1)
```

### Statistical Significance Testing

Statistical significance is calculated using Monte Carlo simulations:

1. Generate surrogate datasets with the same statistical properties as the original data
2. Run the same test on each surrogate dataset
3. Calculate the p-value as the proportion of surrogate results that are more extreme than the observed result

A p-value < 0.05 is considered statistically significant.

### Combined Significance

The combined significance across multiple tests is calculated using Fisher's method:

```
X = -2 * sum(log(p_i))
```

Where X follows a chi-squared distribution with 2k degrees of freedom (k = number of tests).

## Contributing

Contributions to the Cosmic Consciousness Analysis framework are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Implement your changes
4. Add tests for your changes
5. Commit your changes (`git commit -m 'Add your feature'`)
6. Push to the branch (`git push origin feature/your-feature`)
7. Open a Pull Request

Please ensure your code follows the project's coding style and includes appropriate documentation.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The Planck Collaboration for providing the CMB data
- The scientific community for ongoing research into cosmic consciousness
- Contributors and researchers who have helped develop and refine the statistical methods used in this framework

## Contact

For questions, suggestions, or collaboration opportunities, please open an issue on the repository or contact the project maintainers.

## References

1. Planck Collaboration. (2020). Planck 2018 results. Astronomy & Astrophysics, 641, A1.
2. Penrose, R. (1989). The Emperor's New Mind: Concerning Computers, Minds, and the Laws of Physics. Oxford University Press.
3. Tegmark, M. (2014). Our Mathematical Universe: My Quest for the Ultimate Nature of Reality. Knopf.
4. Tononi, G. (2008). Consciousness as integrated information: a provisional manifesto. The Biological Bulletin, 215(3), 216-242.
5. Hameroff, S., & Penrose, R. (2014). Consciousness in the universe: A review of the 'Orch OR' theory. Physics of Life Reviews, 11(1), 39-78.
