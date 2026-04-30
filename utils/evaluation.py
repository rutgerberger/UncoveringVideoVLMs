import os
import math

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from sklearn.metrics import auc

from .logging import eprint
from .model_utils import get_prob, get_score_direct
from .preprocessing import get_baseline_deletion, get_baseline_insertion



def get_prob_drop(args, model, processor, full_ids, baseline_ins_frames, output_ids, frames, tokenizer):
    log_prob_orig = get_prob(args, model, processor, full_ids, output_ids, frames, tokenizer=tokenizer)
    log_prob_baseline = get_prob(args, model, processor, full_ids, output_ids, baseline_ins_frames, tokenizer=tokenizer)
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
    args, mode, M_scaled, M_vol, M_large, video, baseline, model, positions, full_ids, output_ids, dummy_inputs_orig
):
    # M_scaled is the ratio of original video to keep.
    # For both deletion and insertion, this is exactly the same base formula.
    video_blended = video * M_scaled + baseline * (1.0 - M_scaled)

    with torch.no_grad():
        score = get_score_direct(video_blended, model, args, full_ids, output_ids, dummy_inputs_orig, positions)

    # For joint optimization, we also need to test the exact inverse mask 
    if mode == 'joint':
        video_inverse = baseline * M_scaled + video * (1.0 - M_scaled)
        with torch.no_grad():
            score_inv = get_score_direct(video_inverse, model, args, full_ids, output_ids, dummy_inputs_orig, positions)
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



def evaluate_confidence(model, processor, frames, input_ids, output_ids, is_qwen):
    """
    Evaluates the model's probability for the target output_ids.
    Returns the decoded prediction and the probability score.
    """
    with torch.no_grad():
        if is_qwen:
            inputs = processor(text=[" "], videos=[frames], padding=True, return_tensors="pt", max_pixels=112896).to(model.device)
            forward_kwargs = {
                "input_ids": torch.cat((input_ids, output_ids), dim=1),
                "attention_mask": torch.ones_like(torch.cat((input_ids, output_ids), dim=1)).to(model.device),
                "pixel_values_videos": inputs['pixel_values_videos'].to(dtype=model.dtype),
                "video_grid_thw": inputs['video_grid_thw'],
                "use_cache": False
            }
        else:
            inputs = processor(text=" ", videos=frames, return_tensors="pt").to(model.device)
            forward_kwargs = {
                "input_ids": torch.cat((input_ids, output_ids), dim=1),
                "attention_mask": torch.ones_like(torch.cat((input_ids, output_ids), dim=1)).to(model.device),
                "pixel_values_videos": inputs['pixel_values_videos'].to(dtype=model.dtype),
                "use_cache": False
            }
        outputs = model(**forward_kwargs)
        logits = outputs.logits
        out_len = output_ids.shape[-1]
        target_logits = logits[:, -out_len - 1 : -1, :]
        predicted_ids = torch.argmax(target_logits[:, 0:1, :], dim=-1) #Mmhhh.. what is this?
        probs = F.softmax(target_logits, dim=-1)
        target_probs = probs.gather(dim=-1, index=output_ids.unsqueeze(-1)).squeeze(-1)
        mean_prob = target_probs.mean().item()
        entropy = -torch.sum(probs * torch.log(probs + 1e-7), dim=-1)
        mean_entropy = entropy.mean().item()
    return predicted_ids, mean_prob, mean_entropy


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
    
    prob_orig = get_prob(args, model, processor, full_ids, output_ids, frames, positions, tokenizer)
    prob_blur = get_prob(args, model, processor, full_ids, output_ids, baseline_ins_frames, positions, tokenizer)
    prob_del_base = get_prob(args, model, processor, full_ids, output_ids, baseline_del_frames, positions, tokenizer)
    
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
        p_del = get_prob(args, model, processor, full_ids, output_ids, frames_del, positions, tokenizer)
        del_curve_raw.append(p_del)

        # -- Insertion --
        # f=0: Baseline blurred video. f=1: Original video.
        ins_array = np.where(mask_expanded, video_array, baseline_ins_arr).astype(np.uint8)
        frames_ins = [Image.fromarray(frm) for frm in ins_array]
        p_ins = get_prob(args, model, processor, full_ids, output_ids, frames_ins, positions, tokenizer)
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