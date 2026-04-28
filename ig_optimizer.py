import os
import gc
import heapq
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import cv2
import yake

from scipy.ndimage import center_of_mass
import torchvision.transforms.functional as TF 
import torch.nn.functional as F

from utils import *
from method_helpers import *

SAVE_INTERMEDIATE_VISUALS = True


def optimize_tubelet_weights(
        args, model, tokenizer, processor, full_ids, output_ids, frames, baseline_frames, 
        tubelets, positions, mode='deletion', stage="init"
    ):
    """
    Gradually move the weights of the tubelets 
    to optimize the insertion / deletion loss.
    """
    #-- Initialize the raw weight tensor <-inf, inf>
    num_tubes = int(tubelets.max()) + 1
    if mode == 'deletion': 
        W_raw = torch.full((num_tubes,), 2.0, dtype=torch.float32, device=model.device, requires_grad=True)
    else: 
        W_raw = torch.full((num_tubes,), -2.0, dtype=torch.float32, device=model.device, requires_grad=True)
    tubelets_tensor = torch.tensor(tubelets, device=model.device, dtype=torch.long)
    is_qwen = getattr(args, 'model', '') == 'qwen'
    optimizer = torch.optim.Adam([W_raw], lr=args.lr)
    
    #-- Efficiently obtain required inputs for the model and resizing factors (for downscaling)
    packed_inputs = get_rescale_and_dummys(model, processor, frames, baseline_frames, is_qwen, tubelets)
    (dummy_inputs_orig, pixels_orig, pixels_base,
    target_T, target_H, target_W, t_dim_index,
    crop_top, crop_left, new_H, new_W, T_orig, H_orig, W_orig) = packed_inputs
    
    #-- Lists to keep track of gradients and variances (debugging purposes)
    top_10_gradients = []
    mean_gradients = []
    reg_gradients = []
    sparse_gradient_counts = []
    mean_variances = []

    gamma = 0.2  # The decay rate
    lambda_2 = 0.5  # The initial strength of the L2 smoothing
    #-- How much steps we want to take
    for i in range(args.iterations):
        optimizer.zero_grad() # Reset gradients
        W = torch.sigmoid(W_raw) 
        M_high_res = W[tubelets_tensor].unsqueeze(1).float() 
        
        #-- Resizing and cropping
        M_resized_tv = F.interpolate(M_high_res, size=(new_H, new_W), mode='bilinear', align_corners=False)
        M_cropped_tv = M_resized_tv[:, :, crop_top:crop_top+target_H, crop_left:crop_left+target_W]
        M_vol_tv = M_cropped_tv.permute(1, 0, 2, 3).unsqueeze(0) 
        if target_T != T_orig: 
            M_vol_tv = F.interpolate(M_vol_tv, size=(target_T, target_H, target_W), mode='trilinear', align_corners=False)
            
        #-- Compute Regularization Loss (Outside the IG loop, more efficient)
        loss_tv = tv_norm_3d(M_vol_tv, tv_beta=2)
        if mode == 'deletion':
            # Mask initialized to 1. Penalize moving away from 1.
            loss_l1 = torch.mean(torch.abs(1.0 - W))
            loss_l2 = torch.mean((1.0 - W)**2)
        else:
            # Mask initialized to 0. Penalize moving away from 0.
            loss_l1 = torch.mean(torch.abs(W))
            loss_l2 = torch.mean((W)**2)
        current_l2_weight = lambda_2 * np.exp(-gamma * i)
        reg_loss = args.reg_lambda * (loss_l1 + loss_tv) + (current_l2_weight * loss_l2)
        reg_loss.backward() 
        reg_grads = W_raw.grad.detach().cpu().numpy() 
        reg_gradients.append(reg_grads.copy())
        #-- Clean Up
        del W, M_high_res, M_resized_tv, M_cropped_tv, M_vol_tv, reg_loss
        
        # --- State tracking for independent gradients ---
        prev_accumulated_grad = reg_grads.copy()
        step_independent_grads = []
        raw_accumulated = None # Fallback initialization
        
        # --- Main Integrated Gradients Loop ---
        for step in range(1, args.ig_steps + 1):
            alpha = step / args.ig_steps
            W_step = torch.sigmoid(W_raw)
            M_high_res_step = W_step[tubelets_tensor].unsqueeze(1).float() 
            M_low_res_step = rescale_mask(
                M_high_res_step, new_H, new_W, crop_top, crop_left, 
                target_H, target_W, target_T, T_orig, is_qwen, t_dim_index
            )
            
            # Create the fused multiplier (M * alpha)
            #fused_mask = M_low_res_step * alpha
            # Interpolate between the pure baseline and the original video
            #pixels_interval = pixels_orig * fused_mask + pixels_base * (1.0 - fused_mask)
            # At alpha = 0.0, mask = 1.0: pixels : fused_mask = 0.0, pixels_orig = 0.0, pixels_base = 1.0
            # At alpha = 1.0, mask = 1.0: pixels : fused_mask = 1.0, pixels_orig = 1.0, pixels_base = 0.0
            # At alpha = 0.0, mask = 0.0: pixels : fused_mask = 0.0, pixels_orig = 0.0, pixels_base = 1.0
            # At alpha = 1.0, mask = 0.0: pixels : fused_mask = 0.0, pixels_orig = 0.0, pixels_base = 1.0

            if mode == 'deletion':
                # INITIALIZATION: M ~ 1.0
                # ANCHOR (alpha=0): pixels_base
                # TARGET (alpha=1): pixels_orig
                fused_mask = M_low_res_step * alpha
                pixels_interval = pixels_orig * fused_mask + pixels_base * (1.0 - fused_mask)

            else: # mode == 'insertion'
                # INITIALIZATION: M ~ 0.0
                # ANCHOR (alpha=0): pixels_orig
                # TARGET (alpha=1): pixels_base
                fused_mask = (1.0 - M_low_res_step) * alpha
                pixels_interval = pixels_base * fused_mask + pixels_orig * (1.0 - fused_mask)

            # Call modified calculate_gradient
            predicted_text, target_text, raw_accumulated, mean_prob = calculate_gradient(
                model, tokenizer, W_raw, pixels_interval, full_ids, 
                output_ids, positions, mode, args, 
                is_qwen, dummy_inputs_orig
            )
            
            # --- Calculate Independent Gradients ---
            independent_grad = raw_accumulated - prev_accumulated_grad
            prev_accumulated_grad = raw_accumulated.copy()
            step_independent_grads.append(independent_grad.copy())

            if i % 5 == 0 and SAVE_INTERMEDIATE_VISUALS:  
                save_folder = os.path.join(args.output_dir, f'gradients/{stage}')
                os.makedirs(save_folder, exist_ok=True)
                
                mask_hr = W_step[tubelets_tensor].detach().cpu().numpy() 
                faded_hr_frames = []
                
                for t_idx in range(T_orig):
                    orig_f = np.array(frames[t_idx]).astype(np.float32)
                    base_f = np.array(baseline_frames[t_idx]).astype(np.float32)
                    m = mask_hr[t_idx, ..., np.newaxis] 
                    if mode == 'deletion':
                        fused_mask = m * alpha
                        faded_f = orig_f * fused_mask + base_f * (1.0 - fused_mask)
                    else: # mode == 'insertion'
                        fused_mask = (1.0 - m) * alpha
                        faded_f = base_f * fused_mask + orig_f * (1.0 - fused_mask)
                    faded_hr_frames.append(np.clip(faded_f, 0, 255).astype(np.uint8))
                
                if is_qwen:
                    debug_tensors = [torch.from_numpy(f).permute(2, 0, 1).float() / 255.0 for f in faded_hr_frames]
                    grid = torchvision.utils.make_grid(debug_tensors, nrow=len(debug_tensors), padding=2)
                    save_path = os.path.join(save_folder, f"debug_blend_alpha_{alpha:.2f}.png")
                    torchvision.utils.save_image(grid, save_path)
                else:
                    debug_save_pixels_interval(
                        pixels_tensor=pixels_interval, 
                        orig_tensor=pixels_orig, 
                        output_dir=save_folder, 
                        filename=f"debug_blend_alpha_{alpha:.2f}.png",
                        t_dim_index=t_dim_index
                    )
                
                # Visualize the INDEPENDENT gradients for this specific step
                visualize_gradients(
                    gradients=independent_grad, 
                    frames=faded_hr_frames, 
                    tubelets=tubelets, 
                    save_folder=save_folder, 
                    step=step, 
                    title=f"Independent_Grad_Iter_{i+1}_Alpha_{alpha:.2f}"
                )

        # --- Calculate and Visualize Variance Map ---
        grad_matrix = np.stack(step_independent_grads) 
        variance_array = np.var(grad_matrix, axis=0)
        variance_scores = {t: float(v) for t, v in enumerate(variance_array)}
        
        # NEW: Log and track the average variance across all tubelets for this iteration
        current_mean_var = float(np.mean(variance_array))
        mean_variances.append(current_mean_var)
        history_str = ", ".join([f"{v:.2e}" for v in mean_variances])
        
        eprint(f"Stage {stage} Iter {i+1}: Confidence in '{target_text}': {mean_prob:.4f}")
        eprint(f"Stage {stage} Iter {i+1}: Mean IG Variance: {current_mean_var:.2e} | History: [{history_str}]")
        
        save_folder = os.path.join(args.output_dir, f'gradients/{stage}')
        os.makedirs(save_folder, exist_ok=True)
        
        if i % 5 == 0 and SAVE_INTERMEDIATE_VISUALS:
            save_folder = os.path.join(args.output_dir, f'gradients/{stage}')
            os.makedirs(save_folder, exist_ok=True)
            variance_save_path = os.path.join(save_folder, f"Variance_Heatmap_Iter_{i+1}.gif")
            base_video_array = np.stack([np.array(img) for img in frames]) 
            visualize_heatmap(
                video_array=base_video_array,
                tubelet_labels=tubelets,
                tubelet_scores=variance_scores,
                output_path=variance_save_path,
                alpha=0.7 
            )

        # --- RESTORE ORIGINAL METRICS ---
        real_grads = raw_accumulated - reg_grads
        real_grads_abs = np.abs(real_grads)
        top_10_gradients.append(np.sort(real_grads_abs)[-10:])
        mean_gradients.append(np.mean(real_grads_abs))  
        sparse_gradient_counts.append(np.sum(real_grads_abs < 1e-4))
        
        # Update the weights of the tubelets 
        optimizer.step()
        if (i+1) % 10 == 0:
            eprint(f"Iter {i+1}/{args.iterations} | TV: {loss_tv.item():.4f} | L1: {loss_l1.item():.4f}")
            
        del loss_tv, loss_l1
        torch.cuda.empty_cache()
        
    # -- Generate final output weights
    final_weights = torch.sigmoid(W_raw).detach().cpu().numpy()
    if mode == 'deletion':
        init_w = sigmoid(2.0)
        scores = {t: float(init_w - w) for t, w in enumerate(final_weights)}
    else:
        init_w = sigmoid(-2.0) 
        scores = {t: float(w - init_w) for t, w in enumerate(final_weights)}
        
    max_score = max(scores.values()) 
    if max_score > 0.01: 
        dynamic_threshold = max_score * 0.5
        selected_tubelets = [t for t, s in scores.items() if s >= dynamic_threshold]
    else:
        selected_tubelets = []
    if len(selected_tubelets) == 0:
        eprint(f"Warning: Model gradients were dead. Falling back to top 5%.")
        ranked_all = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
        selected_tubelets = ranked_all[:max(1, int(num_tubes * 0.05))]
    
    del pixels_orig, pixels_base
    del dummy_inputs_orig
    del optimizer, W_raw
    torch.cuda.empty_cache()
    gc.collect()

    overall_mean_magnitude = float(np.mean(mean_gradients)) if mean_gradients else 0.0
    metrics = {
        "mean_magnitude": overall_mean_magnitude,
        "mean_variances_history": mean_variances
    }

    return selected_tubelets, scores, metrics

from sklearn.cluster import KMeans