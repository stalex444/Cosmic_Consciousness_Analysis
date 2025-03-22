#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main script to run the cosmic consciousness analysis.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import time

# Add the project root directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core_framework.constants import DEFAULT_SEED
from core_framework.data_handler import load_cmb_data

from tests.coherence_tests.meta_coherence_test import MetaCoherenceTest
from tests.coherence_tests.coherence_analysis_test import CoherenceAnalysisTest
from tests.coherence_tests.gr_specific_coherence_test import GRSpecificCoherenceTest
from tests.information_tests.transfer_entropy_test import TransferEntropyTest
from tests.information_tests.information_integration_test import InformationIntegrationTest
from tests.scale_tests.scale_transition_test import ScaleTransitionTest
from tests.structural_tests.golden_ratio_test import GoldenRatioTest
from tests.structural_tests.fractal_analysis_test import FractalAnalysisTest
from tests.structural_tests.hierarchical_organization_test import HierarchicalOrganizationTest
from tests.structural_tests.resonance_analysis_test import ResonanceAnalysisTest

# Import the analyzer class
from analysis.analysis import CosmicConsciousnessAnalyzer

# Try to import visualization dashboard
try:
    from visualization.dashboard import VisualizationDashboard
    DASHBOARD_AVAILABLE = True
except ImportError:
    print("Warning: Visualization dashboard not available. Install required dependencies.")
    DASHBOARD_AVAILABLE = False

# Try to import Planck data handler
try:
    from planck_data.planck_data_handler import download_planck_data, load_planck_map, preprocess_for_analysis
    PLANCK_AVAILABLE = True
except ImportError:
    print("Warning: Planck data support not available. Install required dependencies.")
    PLANCK_AVAILABLE = False


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run cosmic consciousness analysis")
    
    parser.add_argument("--data-dir", type=str, default="results",
                        help="Directory for data storage and results")
    
    # Data source options
    data_group = parser.add_argument_group('Data Source Options')
    data_source = data_group.add_mutually_exclusive_group()
    data_source.add_argument("--simulated", action="store_true", default=True,
                        help="Use simulated data (default)")
    data_source.add_argument("--planck", action="store_true",
                        help="Use Planck CMB data")
    data_source.add_argument("--custom-data", action="store_true",
                        help="Use custom data file")
    
    # Simulated data options
    sim_group = parser.add_argument_group('Simulated Data Options')
    sim_group.add_argument("--data-size", type=int, default=4096,
                        help="Size of simulated data")
    sim_group.add_argument("--phi-bias", type=float, default=0.1,
                        help="Strength of golden ratio bias in simulated data")
    
    # Planck data options
    if PLANCK_AVAILABLE:
        planck_group = parser.add_argument_group('Planck Data Options')
        planck_group.add_argument("--planck-file", type=str, 
                            help="Path to Planck FITS file")
        planck_group.add_argument("--download-planck", action="store_true",
                            help="Download Planck data if not available")
        planck_group.add_argument("--map-type", type=str, default="SMICA",
                            choices=["SMICA", "NILC", "SEVEM", "Commander"],
                            help="Type of Planck map to download")
        planck_group.add_argument("--resolution", type=str, default="R1",
                            choices=["R1", "R2"],
                            help="Resolution of Planck map (R1=low, R2=high)")
        planck_group.add_argument("--field", type=int, default=0,
                            choices=[0, 1, 2],
                            help="Field to use from Planck map (0=temperature, 1=Q, 2=U)")
    
    # Custom data options
    custom_group = parser.add_argument_group('Custom Data Options')
    custom_group.add_argument("--data-file", type=str,
                        help="Path to custom data file (.npy format)")
    
    # General options
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Random seed for reproducibility")
    
    # Test selection
    parser.add_argument("--meta-coherence", action="store_true",
                        help="Run meta-coherence test")
    parser.add_argument("--coherence-analysis", action="store_true",
                        help="Run coherence analysis test")
    parser.add_argument("--gr-specific-coherence", action="store_true",
                        help="Run golden ratio specific coherence test")
    parser.add_argument("--transfer-entropy", action="store_true",
                        help="Run transfer entropy test")
    parser.add_argument("--information-integration", action="store_true",
                        help="Run information integration test")
    parser.add_argument("--scale-transition", action="store_true",
                        help="Run scale transition test")
    parser.add_argument("--golden-ratio", action="store_true",
                        help="Run golden ratio test")
    parser.add_argument("--fractal-analysis", action="store_true",
                        help="Run fractal analysis test")
    parser.add_argument("--hierarchical-organization", action="store_true",
                        help="Run hierarchical organization test")
    parser.add_argument("--resonance-analysis", action="store_true",
                        help="Run resonance analysis test")
    parser.add_argument("--all", action="store_true",
                        help="Run all tests")
    
    # Output options
    parser.add_argument("--visualize", action="store_true", default=True,
                        help="Generate visualizations")
    parser.add_argument("--report", action="store_true", default=True,
                        help="Generate detailed reports")
    parser.add_argument("--no-parallel", action="store_true", default=False,
                        help="Disable parallel processing (run sequentially)")
    parser.add_argument("--n-jobs", type=int, default=-1,
                        help="Number of parallel jobs (-1 for all cores)")
    
    # Dashboard options
    if DASHBOARD_AVAILABLE:
        dashboard_group = parser.add_argument_group('Dashboard Options')
        dashboard_group.add_argument("--dashboard", action="store_true", default=True,
                                help="Generate visualization dashboard")
        dashboard_group.add_argument("--interactive-dashboard", action="store_true",
                                help="Run interactive dashboard")
        dashboard_group.add_argument("--dashboard-port", type=int, default=8050,
                                help="Port for interactive dashboard")
    
    return parser.parse_args()


def main():
    """Main function to run the analysis."""
    args = parse_arguments()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)
    
    print("=" * 80)
    print("COSMIC CONSCIOUSNESS ANALYSIS")
    print("=" * 80)
    print("Data directory: {}".format(args.data_dir))
    
    # Determine data source
    simulated = args.simulated
    use_planck = args.planck
    use_custom = args.custom_data
    
    if use_planck and not PLANCK_AVAILABLE:
        print("ERROR: Planck data support is not available.")
        print("Please install required dependencies: healpy, astropy")
        return
    
    # Print data source information
    if simulated:
        print("Using simulated data")
        print("Data size: {}".format(args.data_size))
        print("Phi bias: {}".format(args.phi_bias))
    elif use_planck:
        print("Using Planck CMB data")
        if args.planck_file:
            print("Planck file: {}".format(args.planck_file))
        elif args.download_planck:
            print("Downloading Planck data (map type: {}, resolution: {})".format(
                args.map_type, args.resolution))
        else:
            print("ERROR: Either --planck-file or --download-planck must be specified")
            return
        print("Field: {}".format(args.field))
    elif use_custom:
        print("Using custom data file")
        if args.data_file:
            print("Data file: {}".format(args.data_file))
        else:
            print("ERROR: --data-file must be specified when using --custom-data")
            return
    
    print("Random seed: {}".format(args.seed))
    print("Parallel processing: {}".format("Disabled" if args.no_parallel else "Enabled"))
    if not args.no_parallel:
        print("Number of jobs: {}".format("All cores" if args.n_jobs == -1 else args.n_jobs))
    print("=" * 80)
    
    # Load data
    start_time = time.time()
    print("\nLoading data...")
    
    if simulated:
        # Load simulated data
        data = load_cmb_data(simulated=True, seed=args.seed, size=args.data_size)
    elif use_planck:
        # Load Planck data
        if args.planck_file and os.path.exists(args.planck_file):
            # Use existing file
            planck_file = args.planck_file
        elif args.download_planck:
            # Download data
            planck_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "planck_data", "maps")
            planck_file = download_planck_data(planck_dir, map_type=args.map_type, resolution=args.resolution)
            if not planck_file:
                print("ERROR: Failed to download Planck data")
                return
        else:
            print("ERROR: Planck data file not found")
            return
        
        # Load and preprocess Planck data
        data = load_cmb_data(simulated=False, filepath=planck_file, seed=args.seed, 
                            size=args.data_size, field=args.field)
    elif use_custom:
        # Load custom data
        if not os.path.exists(args.data_file):
            print("ERROR: Custom data file not found: {}".format(args.data_file))
            return
        
        data = load_cmb_data(simulated=False, filepath=args.data_file, seed=args.seed)
    
    print("Data loaded in {:.2f} seconds".format(time.time() - start_time))
    print("Data shape: {}".format(data.shape))
    
    # Create analyzer
    analyzer = CosmicConsciousnessAnalyzer(output_dir=args.data_dir)
    
    # Add tests based on command line arguments
    if args.all or args.meta_coherence:
        analyzer.add_test(MetaCoherenceTest(data=data, seed=args.seed))
    
    if args.all or args.coherence_analysis:
        analyzer.add_test(CoherenceAnalysisTest(data=data, seed=args.seed))
    
    if args.all or args.gr_specific_coherence:
        analyzer.add_test(GRSpecificCoherenceTest(data=data, seed=args.seed))
    
    if args.all or args.transfer_entropy:
        analyzer.add_test(TransferEntropyTest(data=data, seed=args.seed))
    
    if args.all or args.information_integration:
        analyzer.add_test(InformationIntegrationTest(data=data, seed=args.seed))
    
    if args.all or args.scale_transition:
        analyzer.add_test(ScaleTransitionTest(data=data, seed=args.seed))
    
    if args.all or args.golden_ratio:
        analyzer.add_test(GoldenRatioTest(data=data, seed=args.seed))
    
    if args.all or args.fractal_analysis:
        analyzer.add_test(FractalAnalysisTest(data=data, seed=args.seed))
    
    if args.all or args.hierarchical_organization:
        analyzer.add_test(HierarchicalOrganizationTest(data=data, seed=args.seed))
    
    if args.all or args.resonance_analysis:
        analyzer.add_test(ResonanceAnalysisTest(data=data, seed=args.seed))
    
    # Run analysis
    print("\nRunning analysis...")
    start_time = time.time()
    
    if not args.no_parallel:
        # Run tests in parallel
        analyzer.run_all_tests_parallel(n_jobs=args.n_jobs)
    else:
        # Run tests sequentially
        analyzer.run_all_tests()
    
    print("Analysis completed in {:.2f} seconds".format(time.time() - start_time))
    
    # Generate report
    if args.report:
        print("\nGenerating report...")
        analyzer.generate_report()
    
    # Generate visualizations
    if args.visualize:
        print("\nGenerating visualizations...")
        analyzer.visualize_results()
    
    # Save results
    analyzer.save_results()
    
    # Generate dashboard
    if DASHBOARD_AVAILABLE and hasattr(args, 'dashboard') and args.dashboard:
        print("\nGenerating visualization dashboard...")
        dashboard = VisualizationDashboard(args.data_dir)
        
        if hasattr(args, 'interactive_dashboard') and args.interactive_dashboard:
            print("Starting interactive dashboard...")
            dashboard.create_interactive_dashboard(port=args.dashboard_port)
        else:
            dashboard_file = dashboard.create_static_dashboard()
            print("Dashboard saved to: {}".format(dashboard_file))
    
    print("\nAll tasks completed successfully.")
    print("Results saved to: {}".format(args.data_dir))


if __name__ == "__main__":
    main()
