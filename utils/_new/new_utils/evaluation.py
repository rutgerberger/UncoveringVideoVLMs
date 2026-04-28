import os

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from sklearn.metrics import auc

from .logging import eprint
from .model_utils import get_prob, get_score_direct
from .preprocessing import get_baseline_deletion, get_baseline_insertion

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
    args, M_scaled, M_vol, M_large, video, baseline, model, positions, full_ids, output_ids, dummy_inputs_orig,
    current_l2_weight=0.0
    ):
    
    video_del = video * M_scaled + baseline * (1.0 - M_scaled)
    video_ins = baseline * M_scaled + video * (1.0 - M_scaled) 

    with torch.no_grad():
        score_del = get_score_direct(video_del, model, args, full_ids, output_ids, dummy_inputs_orig, positions)
        score_ins = get_score_direct(video_ins, model, args, full_ids, output_ids, dummy_inputs_orig, positions)

    tv_penalty_raw = tv_norm_3d(M_vol)
    tv_penalty = tv_penalty_raw.item() if isinstance(tv_penalty_raw, torch.Tensor) else float(tv_penalty_raw)
    L1_penalty = torch.mean(torch.abs(1.0 - M_large)).item() 
    #L2_penalty = torch.mean((1.0 - M_large)**2).item() 

    # Minimize deletion score, maximize insertion score (so minus score_ins)
    fitness = score_del - score_ins + args.reg_lambda * (L1_penalty + tv_penalty)# + (current_l2_weight * L2_penalty)

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

def evaluate_auc(args, model, processor, full_ids, output_ids, frames, video_array, tubelets, selected_tubes, baseline_ins_arr, baseline_del_arr, ivd=0, positions=None, num_steps=20):
    eprint("\n--- Starting Clean AUC Evaluation ---")
    T, H, W, _ = video_array.shape
    
    baseline_ins_frames = [Image.fromarray(f.astype(np.uint8)) for f in baseline_ins_arr]
    baseline_del_frames = [Image.fromarray(f.astype(np.uint8)) for f in baseline_del_arr]
    
    prob_orig = get_prob(args, model, processor, full_ids, output_ids, frames, positions)
    prob_blur = get_prob(args, model, processor, full_ids, output_ids, baseline_ins_frames, positions)
    prob_del_base = get_prob(args, model, processor, full_ids, output_ids, baseline_del_frames, positions)
    
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
        p_del = get_prob(args, model, processor, full_ids, output_ids, frames_del, positions)
        del_curve_raw.append(p_del)

        # -- Insertion --
        # f=0: Baseline blurred video. f=1: Original video.
        ins_array = np.where(mask_expanded, video_array, baseline_ins_arr).astype(np.uint8)
        frames_ins = [Image.fromarray(frm) for frm in ins_array]
        p_ins = get_prob(args, model, processor, full_ids, output_ids, frames_ins, positions)
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

# -- For iGOS++


def evaluate_auc_pixel(args, model, processor, full_ids, output_ids, frames, continuous_mask, ivd=0, positions=None):
    eprint("\n--- Starting iGOS AUC Evaluation (Reference Match) ---")
    video_array = np.stack([np.array(img) for img in frames])
    T, H, W, C = video_array.shape
    num_pixels = T * H * W
    
    step = max(1, num_pixels // 50)
    
    baseline_ins = get_baseline_insertion(args, video_array)
    baseline_del = get_baseline_deletion(args, video_array)
    
    prob_orig = get_prob(args, model, processor, full_ids, output_ids, frames, positions)
    frames_blur = [Image.fromarray(f) for f in baseline_ins]
    prob_blur = get_prob(args, model, processor, full_ids, output_ids, frames_blur, positions)
    
    # Curves track raw probabilities for the plotting function
    del_curve = [prob_orig]
    ins_curve = [prob_blur]
    index = [0.0]
    
    # Broadcast and sort pixels
    pixel_ranks = np.tile(continuous_mask, (T, 1, 1))
    np.random.seed(args.manual_seed)
    pixel_ranks += np.random.rand(*pixel_ranks.shape) * 1e-5
    sorted_ranks = np.sort(pixel_ranks, axis=None)[::-1] 

    for pixels in range(0, num_pixels, step):
        # Calculate exactly how many pixels are evaluated in this step
        current_step_size = step if pixels + step < num_pixels else num_pixels - pixels
        
        idx = min(pixels + current_step_size - 1, num_pixels - 1)
        thresh = sorted_ranks[idx]
        
        current_mask = pixel_ranks >= thresh
        mask_expanded = current_mask[..., np.newaxis] 
        
        # --- Deletion ---
        del_array = np.where(mask_expanded, baseline_del, video_array).astype(np.uint8)
        frames_del = [Image.fromarray(f) for f in del_array]
        p_del = get_prob(args, model, processor, full_ids, output_ids, frames_del, positions)
        del_curve.append(p_del) 
        
        # --- Insertion ---
        ins_array = np.where(mask_expanded, video_array, baseline_ins).astype(np.uint8)
        frames_ins = [Image.fromarray(f) for f in ins_array]
        p_ins = get_prob(args, model, processor, full_ids, output_ids, frames_ins, positions)
        ins_curve.append(p_ins) 
        
        index.append((pixels + current_step_size) / num_pixels)

    # --- NEW: Normalize curves before AUC calculation (Matching evaluate_auc) ---
    norm_ins_curve = [(p - prob_blur) / (prob_orig - prob_blur + 1e-7) for p in ins_curve]
    norm_del_curve = [(p - prob_blur) / (prob_orig - prob_blur + 1e-7) for p in del_curve]

    # Compute Normalized AUC using sklearn
    auc_del = auc(index, norm_del_curve)
    auc_ins = auc(index, norm_ins_curve)

    eprint(f"Final Normalized AUC - Insertion: {auc_ins:.4f} | Deletion: {auc_del:.4f}")
    
    # Generate the visual curves using the unified plotting function
    plot_and_save_auc(
        percentages=index, 
        ins_curve=ins_curve, 
        del_curve=del_curve, 
        auc_ins=auc_ins, 
        auc_del=auc_del, 
        prob_orig=prob_orig, 
        prob_blur=prob_blur, 
        output_dir=args.output_dir, 
        ivd=ivd
    )
    
    return auc_ins, auc_del