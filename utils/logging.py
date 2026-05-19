import os
import json
import time
import numpy as np
from textwrap import fill
import sys

def eprint(*args, **kwargs):
    """Helper to log to stderr."""
    sep = ' '
    combined_text = sep.join(str(arg) for arg in args)
    wrapped_lines = [fill(line, width=80) if line.strip() else "" for line in combined_text.splitlines()]
    print("\n".join(wrapped_lines), file=sys.stderr, **kwargs)

def log_frame_metrics(args, metrics):
    """Appends a flat dictionary of metrics to a JSONL file."""
    os.makedirs(args.output_dir, exist_ok=True)
    metrics_file = os.path.join(args.output_dir, "frame_experiment_metrics.jsonl")
    with open(metrics_file, "a") as f:
        f.write(json.dumps(metrics) + "\n")

def log_experiment(args, meta_info, metrics, start_time):
    """
    Dynamically logs an experiment run. 
    meta_info: dict containing string/context info (question, answer, etc.)
    metrics: dict containing numerical evaluation results.
    """
    # Save to JSONL
    combined_data = {**meta_info, **metrics}
    log_frame_metrics(args, combined_data)

    # Print to Console
    eprint("-" * 25)
    eprint(f"Question {meta_info.get('video_index', 0)+1}/{args.num_videos}: {meta_info.get('question_text', '')}")
    eprint(f"Ground Truth: {meta_info.get('ground_truth', '')}")
    eprint(f"Model Answer: {meta_info.get('model_answer', '')}")
    eprint(f"Extracted Keywords: {meta_info.get('keywords', '')} (Positions: {meta_info.get('positions', '')})")
    eprint("-" * 25)
    
    eprint(f"=== {meta_info.get('mode_name', 'STANDARD')} Tubelets Search Log ===")
    eprint(f"Pipeline executed in {(time.time() - start_time):.2f}s")
    
    # Dynamically print all metrics
    for key, value in metrics.items():
        if isinstance(value, float):
            eprint(f"{key}: {value:.5f}")
        else:
            eprint(f"{key}: {value}")
    eprint("\n")

def log_global_metrics(args, all_metrics, prefix=""):
    """
    Dynamically computes the mean for all numeric keys across a list of metric dictionaries
    and saves the summary to a JSON file.
    """
    if not all_metrics:
        return

    summary = {}
    # Find all unique numeric keys across all metric dictionaries
    numeric_keys = {
        key for metrics in all_metrics 
        for key, val in metrics.items() 
        if isinstance(val, (int, float, np.number)) and not isinstance(val, bool)
    }

    # Calculate means dynamically
    for key in numeric_keys:
        values = [m[key] for m in all_metrics if key in m and m[key] is not None]
        if values:
            summary[f"{prefix}avg_{key}"] = float(np.mean(values))
            summary[f"{prefix}std_{key}"] = float(np.std(values))

    # Calculate custom derived metrics if base requirements exist
    p_orig = np.array([m.get('prob_orig', 0) for m in all_metrics])
    p_del = np.array([m.get('prob_del', 0) for m in all_metrics])
    p_b_del = np.array([m.get('prob_baseline_del', 0) for m in all_metrics])
    
    avg_diff_del = np.mean(p_orig - p_del)
    avg_diff_base = np.mean(p_orig - p_b_del)
    if avg_diff_base != 0:
        summary[f"{prefix}prob_del_explained"] = float(avg_diff_del / avg_diff_base)

    # Save to disk
    out_path = os.path.join(args.output_dir, f'{prefix}final_metrics.json')
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=4)
    eprint(f"Global metrics saved to {out_path}")