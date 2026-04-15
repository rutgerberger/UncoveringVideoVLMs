"""
Helper file which creates a plot
with the magnitude of gradients
against the step (IG, x axis)
"""

import os
import numpy as np
import matplotlib.pyplot as plt

def parse_log_file(filepath):
    """
    Parses the raw log files and extracts the gradient arrays across all stages.
    """
    if not os.path.exists(filepath):
        print(f"Warning: Could not find {filepath}")
        return [], [], [], []

    with open(filepath, 'r') as f:
        content = f.read()

    # Split the file by the start of the mean_gradients block
    # [1:] skips the text before the first stage starts
    blocks = content.split('mean_gradients: [')[1:]

    mean_grads = []
    top_10_means = []
    reg_means = []
    sparse_counts = []

    def clean_and_extract(raw_str):
        """Removes python/numpy syntax and extracts raw floats."""
        clean = raw_str.replace('np.float32', '').replace('np.int64', '')
        clean = clean.replace('dtype=float32', '').replace('array', '')
        for char in '()[],\n':
            clean = clean.replace(char, ' ')
        return [float(x) for x in clean.split()]

    for block in blocks:
        try:
            # 1. Extract mean_gradients
            parts = block.split('top_10_gradients: [')
            mean_str = parts[0]
            mean_grads.extend(clean_and_extract(mean_str))

            # 2. Extract top_10_gradients
            parts2 = parts[1].split('reg_gradients: [')
            top_10_str = parts2[0]
            # Split by 'array' to get the 10 values per step
            for step_str in top_10_str.split('array'):
                nums = clean_and_extract(step_str)
                if nums: top_10_means.append(np.mean(nums))

            # 3. Extract reg_gradients
            parts3 = parts2[1].split('sparse_gradient_counts: [')
            reg_str = parts3[0]
            for step_str in reg_str.split('array'):
                nums = clean_and_extract(step_str)
                if nums: reg_means.append(np.mean(np.abs(nums)))

            # 4. Extract sparse_gradient_counts
            sparse_str = parts3[1].split(']')[0] 
            sparse_counts.extend(clean_and_extract(sparse_str))
            
        except IndexError:
            # Skip if the block was cut off (e.g., interrupted job)
            continue

    # Truncate to the minimum length in case a job crashed mid-stage
    min_len = min(len(mean_grads), len(top_10_means), len(reg_means), len(sparse_counts))
    return mean_grads[:min_len], top_10_means[:min_len], reg_means[:min_len], sparse_counts[:min_len]


# --- MAIN SCRIPT ---
files = {'2 Frames': '/home/s2498278/data/gradients/k400/2_frames.log', '4 Frames': '/home/s2498278/data/gradients/k400/4_frames.log', '8 Frames': '/home/s2498278/data/gradients/k400/8_frames.log'}
colors = {'2 Frames': 'green', '4 Frames': 'orange', '8 Frames': 'red'}

fig, axs = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('SPIX Optimizer Diagnostics: Gradient Collapse Analysis', fontsize=16)

has_data = False

for label, filepath in files.items():
    mean_grads, top_10, reg_means, sparse = parse_log_file(filepath)
    
    if not mean_grads:
        continue
    
    has_data = True
    steps = np.arange(len(mean_grads))
    
    # Plot 1: Mean VLM Signal
    axs[0, 0].plot(steps, mean_grads, label=f'{label} VLM Signal', color=colors[label])
    # Only plot the regularizer once to avoid clutter (it should be roughly the same for all)
    if label == '8 Frames':
        axs[0, 0].plot(steps, reg_means, label='Regularizer Force', color='black', linestyle='--', alpha=0.6)
    
    # Plot 2: Peak Signal (Top 10)
    axs[0, 1].plot(steps, top_10, label=f'{label}', color=colors[label])
    
    # Plot 3: Sparsity (Dead Tubelets)
    axs[1, 0].plot(steps, sparse, label=f'{label}', color=colors[label])
    
    # Plot 4: Signal-to-Noise Ratio (VLM Signal / Regularizer)
    snr = np.array(mean_grads) / (np.array(reg_means) + 1e-8)
    axs[1, 1].plot(steps, snr, label=f'{label}', color=colors[label])


if has_data:
    # Formatting Plot 1
    axs[0, 0].set_yscale('log')
    axs[0, 0].set_title('VLM Signal vs. Regularizer (Log Scale)')
    axs[0, 0].set_ylabel('Mean Gradient Magnitude')
    axs[0, 0].legend()
    axs[0, 0].grid(True, alpha=0.3)

    # Formatting Plot 2
    axs[0, 1].set_title('Peak Object Signal (Mean of Top 10 Tubelets)')
    axs[0, 1].set_ylabel('Gradient Magnitude')
    axs[0, 1].legend()
    axs[0, 1].grid(True, alpha=0.3)

    # Formatting Plot 3
    axs[1, 0].axhline(y=120, color='gray', linestyle=':', label='Total Tubelets (Max)')
    axs[1, 0].set_title('Gradient Sparsity (Dead Tubelets < 1e-4)')
    axs[1, 0].set_xlabel('Total IG Steps')
    axs[1, 0].set_ylabel('Count')
    axs[1, 0].legend()
    axs[1, 0].grid(True, alpha=0.3)

    # Formatting Plot 4
    axs[1, 1].axhline(y=1.0, color='black', linestyle=':', label='Breakeven (Signal = Reg)')
    axs[1, 1].set_title('Signal-to-Regularizer Ratio')
    axs[1, 1].set_xlabel('Total IG Steps')
    axs[1, 1].set_ylabel('Ratio')
    axs[1, 1].legend()
    axs[1, 1].grid(True, alpha=0.3)

    # Add vertical lines to mark the 3 stages
    # Assuming args.iterations = 15 per stage, so 15, 30, 45, etc.
    # Check max steps to draw lines dynamically
    max_steps = max([len(axs[0,0].lines[i].get_xdata()) for i in range(len(axs[0,0].lines))])
    stage_length = max_steps // 3 if max_steps >= 3 else 15 
    
    for ax in axs.flat:
        ax.axvline(x=stage_length, color='black', linestyle='--', alpha=0.3)
        ax.axvline(x=stage_length*2, color='black', linestyle='--', alpha=0.3)
        ax.set_xlim(0, max_steps)

    plt.tight_layout()
    output_path = '/home/s2498278/results/vis/gradient_diagnostics_k400.png'
    plt.savefig(output_path, dpi=300)
    print(f"Saved diagnostics dashboard to {output_path}")
else:
    print("No data parsed. Please check the log filenames.")