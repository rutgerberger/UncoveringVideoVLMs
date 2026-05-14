import os
import math

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

from itertools import combinations
from PIL import Image
from sklearn.metrics import auc

from .logging import eprint
from .model_utils import get_log_prob, get_score_direct
from .preprocessing import get_baseline_deletion, get_baseline_insertion, apply_universal_mask



def get_prob_drop(args, model, processor, full_ids, baseline_ins_frames, output_ids, frames, tokenizer):
    log_prob_orig = get_log_prob(args, model, processor, full_ids, output_ids, frames, tokenizer=tokenizer)
    log_prob_baseline = get_log_prob(args, model, processor, full_ids, output_ids, baseline_ins_frames, tokenizer=tokenizer)
    # Calculate the actual probability ratio: e^(log_baseline - log_orig)
    prob_ratio = math.exp(log_prob_baseline - log_prob_orig)
    prob_drop = 1.0 - prob_ratio
    return prob_drop


def tv_norm_3d(mask, tv_beta=2):
    """
    Calculates the Total Variation loss for a 3D video mask.
    mask shape expected: (Batch, Channels, Time, Height, Width)
    Includes shape guards to prevent NaN when Time, Height, or Width == 1
    """
    tv_t = torch.mean(torch.abs(mask[:, :, 1:, :, :] - mask[:, :, :-1, :, :]).pow(tv_beta)) if mask.shape[2] > 1 else 0.0
    tv_h = torch.mean(torch.abs(mask[:, :, :, 1:, :] - mask[:, :, :, :-1, :]).pow(tv_beta)) if mask.shape[3] > 1 else 0.0
    tv_w = torch.mean(torch.abs(mask[:, :, :, :, 1:] - mask[:, :, :, :, :-1]).pow(tv_beta)) if mask.shape[4] > 1 else 0.0
    return tv_t + tv_h + tv_w


def evaluate_fitness(
        args, mode, M_scaled, M_vol, M_large, video, baseline, model, positions, full_ids, output_ids, dummy_inputs_orig,
        vocab_stats=None
    ):
    # M_scaled is the ratio of original video to keep.
    # For both deletion and insertion, this is exactly the same base formula.
    video_blended = video * M_scaled + baseline * (1.0 - M_scaled)

    with torch.no_grad(): # Calculate the log-likelihood score from the blended tensor
        score = get_score_direct(video_blended, model, args, full_ids, output_ids, dummy_inputs_orig, positions, vocab_stats=vocab_stats)

    # For joint optimization, we also need to test the exact inverse mask 
    if mode == 'joint':
        video_inverse = baseline * M_scaled + video * (1.0 - M_scaled)
        with torch.no_grad():
            score_inv = get_score_direct(video_inverse, model, args, full_ids, output_ids, dummy_inputs_orig, positions, vocab_stats=vocab_stats)
        score_del, score_ins = score, score_inv
    else:
        score_del = score if mode == 'deletion' else 0.0
        score_ins = score if mode == 'insertion' else 0.0

    tv_penalty_raw = tv_norm_3d(M_vol)
    tv_penalty = tv_penalty_raw.item() if isinstance(tv_penalty_raw, torch.Tensor) else float(tv_penalty_raw)
    
    # Optimization Objectives:
    if mode == 'deletion':
        # Push M towards 0 (drop pixels) while minimizing the log-likelihood
        L1_penalty = torch.mean(torch.abs(1.0 - M_large)).item() 
        fitness = score_del + args.reg_lambda * (L1_penalty + tv_penalty)
        
    elif mode == 'insertion':
        # Push M towards 0 (hide pixels) while MAXIMIZING the log-likelihood (minimizing -score)
        L1_penalty = torch.mean(torch.abs(M_large)).item() 
        fitness = -score_ins + args.reg_lambda * (L1_penalty + tv_penalty)
        
    else: # mode == 'joint'
        # Push M towards 0 while minimizing score_del and maximizing score_ins
        L1_penalty = torch.mean(torch.abs(1.0 - M_large)).item() 
        fitness = score_del - score_ins + args.reg_lambda * (L1_penalty + tv_penalty)

    return float(fitness), score_del, score_ins


def plot_and_save_auc(percentages, ins_curve, del_curve, auc_ins, auc_del, prob_orig, prob_blur, output_dir, ivd):
    """Handles matplotlib generation and disk I/O."""
    plt.figure(figsize=(8, 6))
    plt.plot(percentages, ins_curve, label=f'Insertion (AUC={auc_ins:.3f})', color='green', marker='o')
    plt.plot(percentages, del_curve, label=f'Deletion (AUC={auc_del:.3f})', color='red', marker='x')
    plt.axhline(y=prob_orig, color='gray', linestyle='--', label='Original Prob')
    plt.axhline(y=prob_blur, color='blue', linestyle='--', label='Blurred Prob')
    
    plt.title("Insertion / Deletion Curves")
    plt.xlabel("Fraction of Total Video Pixels Revealed/Masked")
    plt.ylabel("Target Probability")
    plt.legend()
    plt.grid(True)
    
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"{ivd}_auc_curves.png"))
    plt.close()

def evaluate_auc(args, model, processor, tokenizer, full_ids, output_ids, frames, video_array, tubelets, selected_tubes, baseline_ins_arr, baseline_del_arr, ivd=0, positions=None, num_steps=20):
    eprint("\n--- Starting Clean AUC Evaluation ---")
    T, H, W, _ = video_array.shape
    
    baseline_ins_frames = [Image.fromarray(f.astype(np.uint8)) for f in baseline_ins_arr]
    baseline_del_frames = [Image.fromarray(f.astype(np.uint8)) for f in baseline_del_arr]
    
    prob_orig = get_log_prob(args, model, processor, full_ids, output_ids, frames, positions, tokenizer)
    prob_blur = get_log_prob(args, model, processor, full_ids, output_ids, baseline_ins_frames, positions, tokenizer)
    prob_del_base = get_log_prob(args, model, processor, full_ids, output_ids, baseline_del_frames, positions, tokenizer)
    
    # Rank the pixels (Higher value = More Important)
    pixel_ranks = np.zeros((T, H, W), dtype=np.float32)
    num_selected = len(selected_tubes)
    
    for i, t_id in enumerate(selected_tubes):
        pixel_ranks[tubelets == t_id] = num_selected - i
        
    # Add tiny uniform noise to strictly break ties across non-selected background pixels
    np.random.seed(getattr(args, 'manual_seed', 42))
    pixel_ranks += np.random.rand(*pixel_ranks.shape) * 0.5 
    
    # Main Evaluation Loop
    fractions = np.linspace(0.0, 1.0, num_steps + 1)
    ins_curve_raw, del_curve_raw = [], []
    
    for f in fractions:
        # Find the threshold that isolates the top `f` fraction of pixels
        # If f=0 -> thresh = max (mask is entirely False)
        # If f=1 -> thresh = min (mask is entirely True)
        thresh = np.percentile(pixel_ranks, 100.0 * (1.0 - f))
        mask = pixel_ranks >= thresh
        mask_expanded = mask[..., np.newaxis]
        
        # -- Deletion --
        # f=0: Original video. f=1: Baseline masked video.
        del_array = np.where(mask_expanded, baseline_del_arr, video_array).astype(np.uint8)
        frames_del = [Image.fromarray(frm) for frm in del_array]
        p_del = get_log_prob(args, model, processor, full_ids, output_ids, frames_del, positions, tokenizer)
        del_curve_raw.append(p_del)

        # -- Insertion --
        # f=0: Baseline blurred video. f=1: Original video.
        ins_array = np.where(mask_expanded, video_array, baseline_ins_arr).astype(np.uint8)
        frames_ins = [Image.fromarray(frm) for frm in ins_array]
        p_ins = get_log_prob(args, model, processor, full_ids, output_ids, frames_ins, positions, tokenizer)
        ins_curve_raw.append(p_ins)

    # Standard XAI Metric Normalization (Clamped to prevent > 1.0 anomalies)
    norm_ins = np.clip((np.array(ins_curve_raw) - prob_blur) / (prob_orig - prob_blur + 1e-7), 0, 1)
    norm_del = np.clip((np.array(del_curve_raw) - prob_del_base) / (prob_orig - prob_del_base + 1e-7), 0, 1)

    auc_ins = auc(fractions, norm_ins)
    auc_del = auc(fractions, norm_del)

    eprint(f"Final AUC - Insertion: {auc_ins:.4f} | Deletion: {auc_del:.4f}")

    # Plot raw probabilities for readability
    plot_and_save_auc(list(fractions), ins_curve_raw, del_curve_raw, auc_ins, auc_del, prob_orig, prob_blur, args.output_dir, ivd)
    
    return auc_ins, auc_del


def calculate_faithfulness(all_prob_orig, all_prob_del):
    """
    Calculates Faithfulness metric over all samples. Min/max normalization is based on
    the min and max log prob over the dataset. Expects lists or 1D arrays of log-probs
    for the entire dataset (N samples).
    """
    orig_np = np.array(all_prob_orig)
    del_np = np.array(all_prob_del)
    all_probs = np.concatenate([orig_np, del_np])
    f_max = np.max(all_probs)
    f_min = np.min(all_probs)
    
    # Edge case: prevent division by zero
    if f_max == f_min:
        return 1.0
    eprint(f"f_max {f_max} f_min {f_min}")
    eprint(f"orig_prob {orig_np} del_prob {del_np}")
    #Normalization
    orig_norm = (orig_np - f_min) / (f_max - f_min)
    del_norm = (del_np - f_min) / (f_max - f_min)
    eprint(f"orig_norm {orig_norm} del_norm {del_norm}")
    #Calculate final faithfulness
    mean_diff = np.abs(orig_norm - del_norm).mean()
    faithfulness = 1.0 - mean_diff
    
    return float(faithfulness)

def calculate_monotonicity(args, model, processor, tokenizer, full_ids, output_ids, 
                           frames, video_array, tubelets, baseline_del_arr, 
                           ranked_tubelets, positions):
    """
    Calculates Monotonicity (Kendall's tau) by masking out increasing fractions 
    of the most important tubelets.
    """
    # 1. Define the masking ratios (e.g., 10%, 20%, ... 90%)
    ratios = np.linspace(0.1, 0.9, num=9)
    num_unique_tubes = len(np.unique(tubelets))
    
    # 2. Get the baseline prediction (0% masked)
    # Convert log-prob to true prob to match the paper's definition of d_k
    log_prob_orig = get_log_prob(args, model, processor, full_ids, output_ids, frames, positions, tokenizer)
    prob_orig = np.exp(log_prob_orig) 
    
    drops_dk = []
    
    # 3. Iteratively mask top-k features and measure the drop
    for ratio in ratios:
        k_tubes_to_mask = max(1, int(num_unique_tubes * ratio))
        
        # Get the top K most important tubelets
        tubes_to_delete = ranked_tubelets[:k_tubes_to_mask]
        
        # Apply the deletion mask
        masked_frames = apply_universal_mask(video_array, baseline_del_arr, tubelets, tubes_to_delete)
        
        # Evaluate model on masked frames
        log_prob_masked = get_log_prob(args, model, processor, full_ids, output_ids, masked_frames, positions, tokenizer)
        prob_masked = np.exp(log_prob_masked)
        
        # Calculate d_k (Original Prob - Masked Prob)
        d_k = prob_orig - prob_masked
        drops_dk.append(d_k)
        
    # 4. Calculate Kendall's tau between ratios (p_k) and drops (d_k)
    tau, p_value = stats.kendalltau(ratios, drops_dk)
    
    # Handle NaN cases (if the model probability literally never changed)
    if np.isnan(tau):
        tau = 0.000414
        
    return float(tau)

def jaccard_similarity(masks, top_k_fraction=0.25):
    """
    Calculates the average pairwise Jaccard similarity across a list of masks.
    Extracts the top 25% tubelets for each mask before comparing.
    """
    if len(masks) < 2:
        return 1.0

    top_k_sets = []
    
    #First select the tob tubelets
    for mask in masks:
        # Handle numpy arrays (from CMA_ES candidates)
        if isinstance(mask, (list, np.ndarray)):
            num_tubes = len(mask)
            k = max(1, int(num_tubes * top_k_fraction))
            # Argsort returns ascending order, so we take the last k elements
            top_tubes = set(np.argsort(mask)[-k:])
        # Handle dictionaries (the final 'scores' dict returned by process_video)
        elif isinstance(mask, dict):
            num_tubes = len(mask)
            k = max(1, int(num_tubes * top_k_fraction))
            sorted_tubes = sorted(mask.keys(), key=lambda t: mask[t], reverse=True)
            top_tubes = set(sorted_tubes[:k])
        else:
            raise TypeError("Mask must be a dictionary or array-like.")
        top_k_sets.append(top_tubes)

    # Calculate pairwise Jaccard similarity
    jaccard_scores = []
    for set1, set2 in combinations(top_k_sets, 2):
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        jaccard_scores.append(intersection / union if union > 0 else 0.0)

    return float(np.mean(jaccard_scores))

def jaccard_similarity_pixel(masks, top_k_fraction=0.10):
    """
    Calculates the average pairwise Jaccard similarity across a list of masks.
    Extracts the top k_fraction of pixels
    masks: list of mask tensors (T, 1, H, W)
    """
    if len(masks) < 2:
        return 1.0
    with torch.no_grad():
        jaccard_scores = []
        binary_masks = []
        for mask in masks:
            flat_mask = mask.view(-1)
            k = max(1, int(flat_mask.numel() * top_k_fraction)) #Threshold value
            threshold = torch.topk(flat_mask, k).values[-1]
            binarized = mask >= threshold #Binarized top-k
            binary_masks.append(binarized)

        for mask1, mask2 in combinations(binarized, 2):
            intersection = torch.logical_and(mask1, mask2).sum().item()
            union = torch.logical_or(mask1, mask2).sum().item()
            if union > 0:
                jaccard_scores.append(intersection / union)
            else:
                jaccard_scores.append(0.0)
        return float(np.mean(jaccard_scores))