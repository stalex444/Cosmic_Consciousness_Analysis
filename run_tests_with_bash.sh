#!/bin/bash
# Script to run cosmic consciousness tests with strict timeouts

set -e

# Create results directory
RESULTS_DIR="results/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

# Log file
LOG_FILE="$RESULTS_DIR/test_log.txt"
touch "$LOG_FILE"

# Function to run a test with timeout
run_test() {
    test_num=$1
    test_name=$2
    timeout_seconds=$3
    
    echo "Running test $test_num: $test_name (timeout: ${timeout_seconds}s)" | tee -a "$LOG_FILE"
    
    # Create test-specific directory
    test_dir="$RESULTS_DIR/test_${test_num}_${test_name// /_}"
    mkdir -p "$test_dir"
    
    # Run the test with timeout
    timeout $timeout_seconds python3 -c "
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import traceback
from datetime import datetime

# Define the golden ratio
phi = (1 + np.sqrt(5)) / 2

def load_data():
    data_dir = 'planck_data'
    ee_file = os.path.join(data_dir, 'power_spectra', 'COM_PowerSpect_CMB-EE-binned_R3.02.txt')
    ee_data = np.loadtxt(ee_file)
    ell = ee_data[:, 0]
    ee_power = ee_data[:, 1]
    return ell, ee_power

def run_test_$test_num():
    print('Running test $test_name...')
    ell, ee_power = load_data()
    mean_power = np.mean(ee_power)
    
    # Test-specific code
    if $test_num == 1:  # Cross-Scale Correlations
        # Define scales separated by powers of phi
        base_scales = [10, 20, 50, 100]
        phi_scales = []
        
        for base in base_scales:
            scale_family = [base]
            for i in range(1, 4):
                scale_family.append(int(round(base * phi**i)))
            phi_scales.append(scale_family)
        
        # Calculate correlations between phi-related scales
        phi_correlations = []
        for family in phi_scales:
            for i in range(len(family)-1):
                scale1 = family[i]
                scale2 = family[i+1]
                
                idx1 = np.abs(ell - scale1).argmin()
                idx2 = np.abs(ell - scale2).argmin()
                
                power1 = ee_power[idx1]
                power2 = ee_power[idx2]
                
                norm_power1 = power1 / mean_power
                norm_power2 = power2 / mean_power
                
                correlation = 1.0 / (abs(norm_power1 - norm_power2) + 0.01)
                phi_correlations.append(correlation)
        
        mean_phi_corr = np.mean(phi_correlations)
        
        # Compare with random scale relationships
        random_correlations = []
        for _ in range(20):  # Reduced Monte Carlo sims
            random_scales = np.random.choice(ell, len(phi_correlations)*2)
            
            for i in range(0, len(random_scales), 2):
                if i+1 < len(random_scales):
                    scale1 = random_scales[i]
                    scale2 = random_scales[i+1]
                    
                    idx1 = np.abs(ell - scale1).argmin()
                    idx2 = np.abs(ell - scale2).argmin()
                    
                    power1 = ee_power[idx1]
                    power2 = ee_power[idx2]
                    
                    norm_power1 = power1 / mean_power
                    norm_power2 = power2 / mean_power
                    
                    correlation = 1.0 / (abs(norm_power1 - norm_power2) + 0.01)
                    random_correlations.append(correlation)
        
        mean_random_corr = np.mean(random_correlations)
        random_std = np.std(random_correlations)
        z_score = (mean_phi_corr - mean_random_corr) / random_std
        p_value = 1 - stats.norm.cdf(z_score)
        
        return mean_phi_corr, mean_random_corr, z_score, p_value
    
    elif $test_num == 2:  # Optimization Test
        galaxy_scales = [200, 500, 800]
        galaxy_indices = [np.abs(ell - scale).argmin() for scale in galaxy_scales]
        galaxy_powers = [ee_power[i] for i in galaxy_indices]
        
        power_ratios = [galaxy_powers[i]/galaxy_powers[i+1] for i in range(len(galaxy_powers)-1)]
        gr_deviations = [abs(ratio - phi) for ratio in power_ratios]
        mean_deviation = np.mean(gr_deviations)
        
        random_deviations = []
        for _ in range(20):
            random_scales = np.random.choice(ell, len(galaxy_scales), replace=False)
            random_indices = [np.abs(ell - scale).argmin() for scale in random_scales]
            random_powers = [ee_power[i] for i in random_indices]
            
            random_ratios = [random_powers[i]/random_powers[i+1] for i in range(len(random_powers)-1)]
            random_gr_deviations = [abs(ratio - phi) for ratio in random_ratios]
            random_deviations.append(np.mean(random_gr_deviations))
        
        mean_random = np.mean(random_deviations)
        std_random = np.std(random_deviations)
        
        z_score = (mean_random - mean_deviation) / std_random
        p_value = stats.norm.cdf(z_score)
        optimization_ratio = mean_random / mean_deviation
        
        return mean_deviation, mean_random, z_score, p_value, optimization_ratio
    
    elif $test_num == 3:  # Golden Symmetries Test
        symmetry_scores = []
        
        for i in range(1, min(50, len(ell) - 1)):
            center_ell = ell[i]
            
            for j in range(1, 3):
                distance = j * 10
                
                left_distance = distance / phi
                right_distance = distance * phi
                
                left_ell = center_ell - left_distance
                right_ell = center_ell + right_distance
                
                left_idx = np.abs(ell - left_ell).argmin()
                right_idx = np.abs(ell - right_ell).argmin()
                
                left_power = ee_power[left_idx]
                right_power = ee_power[right_idx]
                
                norm_left = left_power / mean_power
                norm_right = right_power / mean_power
                
                symmetry = 1.0 / (abs(norm_left - norm_right) + 0.01)
                symmetry_scores.append(symmetry)
        
        mean_symmetry = np.mean(symmetry_scores)
        
        random_symmetries = []
        for _ in range(20):
            random_scores = []
            
            for _ in range(len(symmetry_scores)):
                idx1 = np.random.randint(0, len(ell))
                idx2 = np.random.randint(0, len(ell))
                
                power1 = ee_power[idx1]
                power2 = ee_power[idx2]
                
                norm1 = power1 / mean_power
                norm2 = power2 / mean_power
                
                symmetry = 1.0 / (abs(norm1 - norm2) + 0.01)
                random_scores.append(symmetry)
            
            random_symmetries.append(np.mean(random_scores))
        
        mean_random = np.mean(random_symmetries)
        std_random = np.std(random_symmetries)
        
        z_score = (mean_symmetry - mean_random) / std_random
        p_value = 1 - stats.norm.cdf(z_score)
        
        return mean_symmetry, mean_random, z_score, p_value
    
    elif $test_num == 4:  # Phi Network Test
        phi_pairs = []
        max_ell = min(100, len(ell))
        
        for i in range(max_ell):
            for j in range(i+1, max_ell):
                ratio = ell[j] / ell[i]
                if abs(ratio - phi) < 0.1:
                    phi_pairs.append((i, j))
        
        clustering = len(phi_pairs) / (max_ell * (max_ell - 1) / 2)
        
        random_clusterings = []
        for _ in range(20):
            random_pairs = []
            for _ in range(len(phi_pairs)):
                i = np.random.randint(0, max_ell)
                j = np.random.randint(0, max_ell)
                if i != j:
                    random_pairs.append((min(i, j), max(i, j)))
            
            random_clustering = len(set(random_pairs)) / (max_ell * (max_ell - 1) / 2)
            random_clusterings.append(random_clustering)
        
        mean_random = np.mean(random_clusterings)
        std_random = np.std(random_clusterings)
        
        z_score = (clustering - mean_random) / std_random
        p_value = 1 - stats.norm.cdf(z_score)
        
        return clustering, mean_random, z_score, p_value
    
    elif $test_num == 5:  # Spectral Gap Test
        power_diffs = np.diff(ee_power)
        norm_diffs = power_diffs / mean_power
        mean_diff = np.mean(np.abs(norm_diffs))
        
        random_diffs = []
        for _ in range(20):
            random_power = np.random.permutation(ee_power)
            random_power_diffs = np.diff(random_power)
            random_norm_diffs = random_power_diffs / mean_power
            random_mean_diff = np.mean(np.abs(random_norm_diffs))
            random_diffs.append(random_mean_diff)
        
        mean_random = np.mean(random_diffs)
        std_random = np.std(random_diffs)
        
        z_score = (mean_diff - mean_random) / std_random
        p_value = 1 - stats.norm.cdf(z_score)
        
        return mean_diff, mean_random, z_score, p_value
    
    elif $test_num == 6:  # Multi-Scale Coherence Test
        scales = [10, 20, 50, 100, 200, 500]
        scale_indices = [np.abs(ell - scale).argmin() for scale in scales]
        scale_powers = [ee_power[i] for i in scale_indices]
        
        coherence_scores = []
        for i in range(len(scales)):
            for j in range(i+1, len(scales)):
                power_i = scale_powers[i]
                power_j = scale_powers[j]
                
                norm_power_i = power_i / mean_power
                norm_power_j = power_j / mean_power
                
                coherence = 1.0 / (abs(norm_power_i - norm_power_j) + 0.01)
                coherence_scores.append(coherence)
        
        mean_coherence = np.mean(coherence_scores)
        
        random_coherences = []
        for _ in range(20):
            random_scores = []
            
            for _ in range(len(coherence_scores)):
                idx1 = np.random.randint(0, len(ell))
                idx2 = np.random.randint(0, len(ell))
                
                power1 = ee_power[idx1]
                power2 = ee_power[idx2]
                
                norm1 = power1 / mean_power
                norm2 = power2 / mean_power
                
                coherence = 1.0 / (abs(norm1 - norm2) + 0.01)
                random_scores.append(coherence)
            
            random_coherences.append(np.mean(random_scores))
        
        mean_random = np.mean(random_coherences)
        std_random = np.std(random_coherences)
        
        z_score = (mean_coherence - mean_random) / std_random
        p_value = 1 - stats.norm.cdf(z_score)
        
        return mean_coherence, mean_random, z_score, p_value

def save_results(result, test_dir):
    if result is None:
        return
    
    # Save results to file
    with open(os.path.join(test_dir, 'results.txt'), 'w') as f:
        f.write('$test_name Results\\n')
        f.write('=' * len('$test_name Results') + '\\n\\n')
        
        if isinstance(result, tuple):
            if len(result) >= 4 and all(isinstance(x, (int, float)) for x in result[:4]):
                metric, random_metric, z_score, p_value = result[:4]
                f.write(f'Metric Value: {metric:.6f}\\n')
                f.write(f'Random Expectation: {random_metric:.6f}\\n')
                f.write(f'Z-Score: {z_score:.6f}\\n')
                f.write(f'P-Value: {p_value:.6f}\\n')
                f.write(f'Significant: {p_value < 0.05}\\n')
                
                # Calculate phi-optimality
                if p_value < 1e-10:
                    phi_optimality = 1.0
                elif p_value > 0.9:
                    phi_optimality = -1.0
                else:
                    phi_optimality = 1.0 - 2.0 * p_value
                f.write(f'Phi-Optimality: {phi_optimality:.6f}\\n')
                
                # Create visualization
                plt.figure(figsize=(10, 6))
                plt.bar(['Test Metric', 'Random Expectation'], 
                       [metric, random_metric],
                       color=['gold', 'gray'])
                plt.ylabel('Metric Value')
                plt.title('$test_name Test Results')
                plt.annotate(f'p-value: {p_value:.6f}', xy=(0.5, 0.9), 
                            xycoords='axes fraction', ha='center')
                plt.annotate(f'Significant: {p_value < 0.05}', xy=(0.5, 0.85), 
                            xycoords='axes fraction', ha='center')
                plt.tight_layout()
                plt.savefig(os.path.join(test_dir, '$test_name'.lower().replace(' ', '_') + '.png'))
                plt.close()
            else:
                for i, value in enumerate(result):
                    f.write(f'Result {i+1}: {value}\\n')
        else:
            f.write(f'Result: {result}\\n')

try:
    # Run the test
    result = run_test_$test_num()
    
    # Save results
    save_results(result, '$test_dir')
    
    print('Test completed successfully')
    print(f'Results saved to {os.path.abspath(\"$test_dir\")}')
    
    # Print summary
    if isinstance(result, tuple) and len(result) >= 4:
        print(f'P-Value: {result[3]:.6f}')
        print(f'Significant: {result[3] < 0.05}')
except Exception as e:
    print(f'Error running test: {str(e)}')
    traceback.print_exc()
    sys.exit(1)
" > "$test_dir/output.txt" 2>&1
    
    # Check if the test completed successfully
    if [ $? -eq 0 ]; then
        echo "Test $test_num completed successfully" | tee -a "$LOG_FILE"
        cat "$test_dir/output.txt" | tee -a "$LOG_FILE"
    elif [ $? -eq 124 ]; then
        echo "Test $test_num timed out after $timeout_seconds seconds" | tee -a "$LOG_FILE"
    else
        echo "Test $test_num failed with error code $?" | tee -a "$LOG_FILE"
        cat "$test_dir/output.txt" | tee -a "$LOG_FILE"
    fi
    
    echo "----------------------------------------" | tee -a "$LOG_FILE"
}

# Run all tests with timeouts
run_test 1 "Cross-Scale Correlations" 30
run_test 2 "Optimization" 30
run_test 3 "Golden Symmetries" 30
run_test 4 "Phi Network" 30
run_test 5 "Spectral Gap" 30
run_test 6 "Multi-Scale Coherence" 30

echo "All tests completed. Results saved to $RESULTS_DIR" | tee -a "$LOG_FILE"
