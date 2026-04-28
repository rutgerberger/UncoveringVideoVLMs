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


def log_experiment(args, log_func, ivd, question_text, ground_truth, model_answer, keywords, positions,
                   prob_orig, prob_baseline_del, prob_baseline_ins, prob_ins, prob_del, auc_ins, auc_del,
                   deletion_metrics, insertion_metrics, top_k, fmt_ins, fmt_del, fmt_merged, 
                   selected_ins, selected_del, selected_merged, unique_tubes, k_fraction, mode_name, start_time):

    """Handles single experiment logging"""

    os.makedirs(args.output_dir, exist_ok=True)
    metrics_file = os.path.join(args.output_dir, "frame_experiment_metrics.jsonl")
    experiment_data = {
        "video_index": ivd,
        "num_frames": args.num_frames,
        "Prob orig": round(prob_orig,3),
        "Prob_baseline_del": round(prob_baseline_del,3),
        "Prob_baseline_ins": round(prob_baseline_ins,3),
        "Prob ins": round(prob_ins,3),
        "Prob del": round(prob_del,3),
        "AUC Ins": round(auc_ins,3),
        "AUC Del": round(auc_del,3),
        "deletion_metrics": deletion_metrics,
        "insertion_metrics": insertion_metrics,
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
    log_func(f"Created tubelets and optimized weights in {(time.time() - start_time):.2f}s") # <-- Updated to start_time
    log_func(f"Final Insertion tubelets (Top {top_k}): {fmt_ins} ({len(selected_ins)}/{len(unique_tubes)})")
    log_func(f"Final Deletion tubelets (Top {top_k}): {fmt_del} ({len(selected_del)}/{len(unique_tubes)})")
    log_func(f"Final Combined tubelets (Top {top_k}): {fmt_merged} ({len(selected_merged)}/{len(unique_tubes)})")
    log_func(f"Prob when Inserting Mask (top {k_fraction*100}%): {prob_ins:.5f} (Diff: {prob_orig - prob_ins:.5f})")
    log_func(f"Prob when Deleting Mask (top {k_fraction*100}%): {prob_del:.5f} (Diff: {prob_orig - prob_del:.5f})")

    log_func(f"\n=== Experiment Metrics ===\n")
    log_func(f"Insertion Landscape -> d_eff: {insertion_metrics.get('d_eff', 0):.2f} | Diversity (L1): {insertion_metrics.get('diversity', 0):.4f} (from {insertion_metrics.get('num_top_candidates', 0)} top masks)")
    log_func(f"Deletion Landscape  -> d_eff: {deletion_metrics.get('d_eff', 0):.2f} | Diversity (L1): {deletion_metrics.get('diversity', 0):.4f} (from {deletion_metrics.get('num_top_candidates', 0)} top masks)")
    log_func(f"\nAUC Ins: {auc_ins:.4f} | Del: {auc_del:.4f}")


def log_metrics(args, metrics, prefix=""):
    """dump experiment metrics in log file"""
        
    with open(os.path.join(args.output_dir, f'{prefix}final_metrics.json'), 'w') as f: # <-- Appended prefix to filename to prevent overwrite
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
            f"{prefix}avg_prob_diff_del_topx": float(avg_prob_diff_del_topx),
            f"{prefix}avg_prob_diff_del_blur": float(avg_prob_diff_del_blur),
            f"{prefix}avg_prob_diff_ins_topx": float(avg_prob_diff_ins_topx),
            f"{prefix}avg_prob_diff_ins_orig": float(avg_prob_diff_ins_orig),
            f"{prefix}prob_del_explained": float(avg_prob_diff_del_topx / avg_prob_diff_del_blur) if avg_prob_diff_del_blur != 0 else 0.0,
            f"{prefix}prob_ins_explained": float(avg_prob_diff_ins_topx / avg_prob_diff_ins_orig) if avg_prob_diff_ins_orig != 0 else 0.0,
        }
        json.dump(summary, f, indent=4)