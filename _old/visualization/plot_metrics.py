import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def plot_frame_experiment(jsonl_file, output_path="/home/s2498278/results/vis/metrics_clevrer.png"):
    # Structure: {num_frames: {'magnitudes': [], 'variances': []}}
    data_by_frames = defaultdict(lambda: {'magnitudes': [], 'variances': []})

    with open(jsonl_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            frames = data['num_frames']
            
            # For simplicity, we'll just look at Deletion Stage 1 as the primary indicator
            if "stage_1" in data.get("deletion_metrics", {}):
                metrics = data["deletion_metrics"]["stage_1"]
                
                # Append the mean magnitude
                data_by_frames[frames]['magnitudes'].append(metrics["mean_magnitude"])
                
                # The variance is a list of 15 items per run. We calculate the overall mean 
                # for this run and append it.
                run_mean_variance = np.mean(metrics["mean_variances_history"])
                data_by_frames[frames]['variances'].append(run_mean_variance)

    # Sort the data by number of frames for plotting
    sorted_frames = sorted(data_by_frames.keys())
    
    avg_magnitudes = []
    avg_variances = []
    std_variances = []

    for f in sorted_frames:
        avg_magnitudes.append(np.mean(data_by_frames[f]['magnitudes']))
        
        var_array = np.array(data_by_frames[f]['variances'])
        avg_variances.append(np.mean(var_array))
        std_variances.append(np.std(var_array))

    # --- Plotting with Dual Y-Axes ---
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color1 = 'tab:blue'
    ax1.set_xlabel('Number of Frames', fontsize=12)
    ax1.set_ylabel('Mean Gradient Magnitude', color=color1, fontsize=12)
    # Plot Magnitude
    ax1.plot(sorted_frames, avg_magnitudes, color=color1, marker='o', label='Magnitude', linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color1)
    
    # Ensure x-axis ticks match the frame counts exactly
    ax1.set_xticks(sorted_frames)

    # Create a second y-axis that shares the same x-axis
    ax2 = ax1.twinx()  
    color2 = 'tab:red'
    ax2.set_ylabel('Mean IG Variance', color=color2, fontsize=12)
    
    # Plot Variance with Error Bars
    ax2.errorbar(sorted_frames, avg_variances, yerr=std_variances, color=color2, marker='x', 
                 label='Variance', linewidth=2, capsize=5)
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Set the right y-axis to scientific notation
    ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    plt.title('Effect of Frame Count on Gradients and Variance (Deletion Stage 1)', fontsize=14)
    fig.tight_layout() 
    
    plt.savefig(output_path, dpi=300)
    print(f"Saved experiment plot to {output_path}")

# Run the plotter
plot_frame_experiment("/home/s2498278/results/clevrer/412-2118/frame_experiment_metrics.jsonl")