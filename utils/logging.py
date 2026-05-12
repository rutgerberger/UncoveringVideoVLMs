import sys
import os
import json
import time

import numpy as np
from textwrap import fill

def eprint(*args, **kwargs):
    """Helper to log to stderr."""
    sep = ' '
    combined_text = sep.join(str(arg) for arg in args)
    wrapped_lines = []
    for line in combined_text.splitlines():
        if line.strip() == "":
            wrapped_lines.append("")
        else:
            wrapped_lines.append(fill(line, width=80))
    final_text = "\n".join(wrapped_lines)
    print(final_text, file=sys.stderr, **kwargs)

def log_frame_metrics(args, ivd, metrics):
    os.makedirs(args.output_dir, exist_ok=True)
    metrics_file = os.path.join(args.output_dir, "frame_experiment_metrics.jsonl")
    with open(metrics_file, "a") as f:
        f.write(json.dumps(metrics) + "\n")


def log_experiment(args, log_func, ivd, question_text, ground_truth, model_answer, keywords, positions,
                   prob_orig, prob_baseline_del, prob_baseline_ins, prob_ins, prob_del, 
                   auc_ins_mean, auc_del_mean, auc_ins_std, auc_del_std, iou_score, num_runs,
                   metrics, top_k, fmt_ins, fmt_del, fmt_merged, 
                   selected_ins, selected_del, selected_merged, unique_tubes, k_fraction, mode_name, start_time):
    """Handles single experiment logging"""

    os.makedirs(args.output_dir, exist_ok=True)
    metrics_file = os.path.join(args.output_dir, "frame_experiment_metrics.jsonl")
    experiment_data = {
        "video_index": ivd,
        "num_frames": getattr(args, 'num_frames', 8),
        "num_runs": num_runs,
        "Prob orig": round(prob_orig, 3),
        "Prob_baseline_del": round(prob_baseline_del, 3),
        "Prob_baseline_ins": round(prob_baseline_ins, 3),
        "Prob ins Mean": round(prob_ins, 3),
        "Prob del Mean": round(prob_del, 3),
        "AUC Ins Mean": round(auc_ins_mean, 3),
        "AUC Ins Std": round(auc_ins_std, 3),
        "AUC Del Mean": round(auc_del_mean, 3),
        "AUC Del Std": round(auc_del_std, 3),
        "Infidelity": round(metrics.get("infd", 0.0), 3),
        "IoU (Jaccard)": round(iou_score, 4),
        "metrics": metrics
    }
    
    with open(metrics_file, "a") as f:
        f.write(json.dumps(experiment_data) + "\n")

    log_func("-" * 25)
    log_func(f"Question {ivd+1}/{args.num_videos}: {question_text}")
    log_func(f"Ground Truth: {ground_truth}")
    log_func(f"Model Answer: {model_answer}")
    log_func(f"Extracted Keywords: {keywords} (Positions: {positions})")
    log_func("-" * 25)
    log_func(f"Original probs: {prob_orig:.5f}")
    log_func(f"baseline_del probs: {prob_baseline_del:.5f}")
    log_func(f"baseline_ins probs: {prob_baseline_ins:.5f}\n")

    log_func(f"=== {mode_name} Tubelets Search Log ===")
    log_func(f"Created tubelets and optimized weights in {(time.time() - start_time):.2f}s")
    log_func(f"Final Insertion tubelets (Top {top_k}): {fmt_ins} ({len(selected_ins)}/{len(unique_tubes)})")
    log_func(f"Final Deletion tubelets (Top {top_k}): {fmt_del} ({len(selected_del)}/{len(unique_tubes)})")
    log_func(f"Final Combined tubelets (Top {top_k}): {fmt_merged} ({len(selected_merged)}/{len(unique_tubes)})")
    log_func(f"Prob when Inserting Mask (top {k_fraction*100}%): {prob_ins:.5f} (Diff: {prob_orig - prob_ins:.5f})")
    log_func(f"Prob when Deleting Mask (top {k_fraction*100}%): {prob_del:.5f} (Diff: {prob_orig - prob_del:.5f})")
    log_func(f"\nAUC Ins: {auc_ins_mean:.4f} ± {auc_ins_std:.4f} | Del: {auc_del_mean:.4f} ± {auc_del_std:.4f}")
    log_func(f"Infidelity: {metrics.get('infd', 0.0):.5f} ± {metrics.get('infd_std', 0.0):.5f}") # <--- NEW
    
    if num_runs > 1:
        log_func(f"IoU (Jaccard Similarity) across {num_runs} runs: {iou_score:.4f}")

def log_metrics(args, metrics, prefix=""):
    """dump experiment metrics in log file"""
        
    with open(os.path.join(args.output_dir, f'{prefix}final_metrics.json'), 'w') as f:
        p_orig = np.array([m['prob_orig'] for m in metrics])
        p_del = np.array([m['prob_del'] for m in metrics])
        p_b_del = np.array([m['prob_baseline_del'] for m in metrics])
        p_ins = np.array([m['prob_ins'] for m in metrics])
        p_b_ins = np.array([m['prob_baseline_ins'] for m in metrics])
        
        # Calculate average probability differences
        avg_prob_diff_del_topx = np.mean(p_orig - p_del)
        avg_prob_diff_del_blur = np.mean(p_orig - p_b_del)
        avg_prob_diff_ins_topx = np.mean(p_ins - p_b_ins)
        avg_prob_diff_ins_orig = np.mean(p_orig - p_b_ins)
        
        summary = {
            f"{prefix}avg_auc_ins": float(np.mean([m['auc_ins'] for m in metrics])),
            f"{prefix}avg_auc_del": float(np.mean([m['auc_del'] for m in metrics])),
            f"{prefix}avg_infidelity": float(np.mean([m.get('infd', 0.0) for m in metrics])),
            f"{prefix}avg_iou_score": float(np.mean([m.get('iou_score', 1.0) for m in metrics])),
            f"{prefix}avg_prob_diff_del_topx": float(avg_prob_diff_del_topx),
            f"{prefix}avg_prob_diff_del_blur": float(avg_prob_diff_del_blur),
            f"{prefix}avg_prob_diff_ins_topx": float(avg_prob_diff_ins_topx),
            f"{prefix}avg_prob_diff_ins_orig": float(avg_prob_diff_ins_orig),
            f"{prefix}prob_del_explained": float(avg_prob_diff_del_topx / avg_prob_diff_del_blur) if avg_prob_diff_del_blur != 0 else 0.0,
            f"{prefix}prob_ins_explained": float(avg_prob_diff_ins_topx / avg_prob_diff_ins_orig) if avg_prob_diff_ins_orig != 0 else 0.0,
        }
        json.dump(summary, f, indent=4)