import re
import numpy as np
import matplotlib.pyplot as plt
import os

def extract_iteration_histories(file_path, target_iter=15):
    """
    Reads the log file and extracts the history arrays for the given iteration.
    Handles multi-line wrapping caused by the 80-char textwrap in the logs.
    """
    if not os.path.exists(file_path):
        print(f"Warning: Could not find file {file_path}")
        return []

    with open(file_path, 'r') as f:
        content = f.read()

    pattern = rf"Iter {target_iter}:.*?History:\s*\[(.*?)\]"
    matches = re.findall(pattern, content, flags=re.DOTALL)

    histories = []
    for match in matches:
        clean_string = match.replace('\n', '').strip()
        try:
            float_list = [float(val) for val in clean_string.split(',') if val.strip()]
            histories.append(float_list)
        except ValueError as e:
            print(f"Skipping a malformed array in {file_path}: {e}")
            
    return histories

def plot_variance_comparison(file1_path, file2_path, output_image="variance_plot.png"):
    data1 = extract_iteration_histories(file1_path)
    data2 = extract_iteration_histories(file2_path)

    if not data1 and not data2:
        print("No data found in either file. Exiting.")
        return

    plt.figure(figsize=(10, 6))
    iterations = np.arange(1, 16)

    # Process and plot File 1
    if data1:
        arr1 = np.array(data1)
        # Use Median and Percentiles instead of Mean and Std
        median1 = np.median(arr1, axis=0)
        p25_1 = np.percentile(arr1, 25, axis=0)
        p75_1 = np.percentile(arr1, 75, axis=0)
        
        plt.plot(iterations, median1, label='File 1 (k400) (Median)', color='blue', marker='o')
        plt.fill_between(iterations, p25_1, p75_1, color='blue', alpha=0.2, label='File 1 (IQR: 25th-75th %)')

    # Process and plot File 2
    if data2:
        arr2 = np.array(data2)
        median2 = np.median(arr2, axis=0)
        p25_2 = np.percentile(arr2, 25, axis=0)
        p75_2 = np.percentile(arr2, 75, axis=0)
        
        plt.plot(iterations, median2, label='File 2 (Simple) (Median)', color='red', marker='x')
        plt.fill_between(iterations, p25_2, p75_2, color='red', alpha=0.2, label='File 2 (IQR: 25th-75th %)')

    # Formatting
    plt.title("IG Variance Across Optimization Iterations", fontsize=14)
    plt.xlabel("Iteration Step (1 to 15)", fontsize=12)
    plt.ylabel("Median Gradient Variance", fontsize=12)
    plt.xticks(iterations)
    
    # Revert to linear scientific notation scale
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper left')
    plt.tight_layout()

    # Save and show
    plt.savefig(output_image, dpi=300)
    print(f"Plot successfully saved to {output_image}")

if __name__ == "__main__":
    file_1 = "/home/s2498278/logs/spix/create_sp_1388270.err"
    file_2 = "/home/s2498278/logs/spix/create_sp_1388419.err"
    
    plot_variance_comparison(file_1, file_2, output_image="/home/s2498278/results/vis/variance_comparison.png")