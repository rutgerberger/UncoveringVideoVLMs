from utils import *

import numpy as np
import torch
import torch.nn.functional as F
import cv2

import yake
import heapq

from scipy.ndimage import center_of_mass
import torchvision.transforms.functional as TF 


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# For Gradient Based search

def tv_norm_3d(mask, tv_beta=2):
    """
    Calculates the Total Variation loss for a 3D video mask.
    mask shape expected: (Batch, Channels, Time, Height, Width)
    """
    # Temporal TV (Smoothness across frames)
    tv_t = torch.mean(torch.abs(mask[:, :, 1:, :, :] - mask[:, :, :-1, :, :]).pow(tv_beta))
    # Spatial TV Height (Smoothness vertically)
    tv_h = torch.mean(torch.abs(mask[:, :, :, 1:, :] - mask[:, :, :, :-1, :]).pow(tv_beta))
    # Spatial TV Width (Smoothness horizontally)
    tv_w = torch.mean(torch.abs(mask[:, :, :, :, 1:] - mask[:, :, :, :, :-1]).pow(tv_beta))
    
    return tv_t + tv_h + tv_w

def optimize_tubelet_weights(
        args, model, processor, full_ids, output_ids, frames, baseline_frames, 
        tubelets, positions, mode='deletion'
    ):
    num_tubes = int(tubelets.max()) + 1
    
    if mode == 'deletion': 
        W_raw = torch.full((num_tubes,), 2.0, dtype=torch.float32, device=model.device, requires_grad=True)
    else: 
        W_raw = torch.full((num_tubes,), -2.0, dtype=torch.float32, device=model.device, requires_grad=True)
        
    tubelets_tensor = torch.tensor(tubelets, device=model.device, dtype=torch.long)

    # ---------------------------------------------------------
    # OPTIMIZATION: Process the videos ONCE outside the loop
    # ---------------------------------------------------------
    with torch.no_grad():
        dummy_inputs_orig = processor(text=" ", videos=frames, return_tensors="pt").to(model.device)
        pixels_orig = dummy_inputs_orig['pixel_values_videos'].detach()
        
        dummy_inputs_base = processor(text=" ", videos=baseline_frames, return_tensors="pt").to(model.device)
        pixels_base = dummy_inputs_base['pixel_values_videos'].detach()
        
        target_tensor_shape = pixels_orig.shape
        
        # Handle (B, C, T, H, W) vs (B, T, C, H, W)
        if target_tensor_shape[1] == 3: 
            target_C, target_T, target_H, target_W = target_tensor_shape[1:5]
            t_dim_index = 2
        else:
            target_T, target_C, target_H, target_W = target_tensor_shape[1:5]
            t_dim_index = 1

    optimizer = torch.optim.Adam([W_raw], lr=args.lr)
    
    for i in range(args.iterations):
        optimizer.zero_grad()
        
        W = torch.sigmoid(W_raw)
        
        # Create continuous mask in original resolution: (Time, Channels=1, Height, Width)
        M_high_res = W[tubelets_tensor].unsqueeze(1).float() 
        
        # Downsample the mask directly to the LLM's target resolution
        # Shape: (T, 1, target_H, target_W)
        M_low_res_base = F.interpolate(M_high_res, size=(target_H, target_W), mode='bilinear', align_corners=False)
        
        # Reshape mask to broadcast across the pre-processed pixels for BLENDING
        if t_dim_index == 1:
            # HuggingFace standard: (Batch=1, Time=T, Channels=1, Height, Width)
            M_low_res = M_low_res_base.unsqueeze(0) 
        else:
            # Alternate standard: (Batch=1, Channels=1, Time=T, Height, Width)
            M_low_res = M_low_res_base.permute(1, 0, 2, 3).unsqueeze(0) 

        # Blend the pre-processed pixel latent tensors
        pixels_final = pixels_orig * M_low_res + pixels_base * (1.0 - M_low_res)
        
        # Reshape mask for TV Loss
        # tv_norm_3d ALWAYS expects: (Batch=1, Channels=1, Time=T, Height, Width)
        mask_for_tv = M_low_res_base.permute(1, 0, 2, 3).unsqueeze(0)
            
        loss_tv = tv_norm_3d(mask_for_tv, tv_beta=2)
        loss_l1 = torch.mean(torch.abs(1.0 - W)) if mode == 'deletion' else torch.mean(torch.abs(W))
        
        reg_loss = args.reg_lambda * (loss_l1 + loss_tv)
        reg_loss.backward(retain_graph=True) 

        for step in range(1, args.ig_steps + 1):
            alpha = step / args.ig_steps 
            pixels_interval = pixels_final * alpha + (pixels_base * (1.0 - alpha))
            
            forward_kwargs = {
                "input_ids": full_ids,
                "attention_mask": torch.ones_like(full_ids).to(model.device), 
                "pixel_values_videos": pixels_interval.to(dtype=model.dtype),
                "use_cache": True
            }
            outputs = model(**forward_kwargs)

            logits = outputs.logits 
            out_len = output_ids.shape[-1]
            target_logits = logits[:, -out_len - 1 : -1, :] 
            probs = F.softmax(target_logits, dim=-1) 
            target_probs = probs.gather(dim=-1, index=output_ids.unsqueeze(-1)).squeeze(-1)

            if positions is not None and len(positions) > 0:
                target_probs = target_probs[0, positions] 
            mean_prob = target_probs.mean()
            log_prob = torch.log(mean_prob + 1e-7) 

            if mode == 'deletion':
                main_loss = log_prob / args.ig_steps 
            else:
                main_loss = -log_prob / args.ig_steps 
            
            retain = (step < args.ig_steps)
            main_loss.backward(retain_graph=retain) 

        optimizer.step()
        if (i+1) % 10 == 0:
            eprint(f"Iter {i+1}/{args.iterations} | TV: {loss_tv.item():.4f} | L1: {loss_l1.item():.4f}")

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
        dynamic_threshold = max_score * 0.2
        selected_tubelets = [t for t, s in scores.items() if s >= dynamic_threshold]
    else:
        selected_tubelets = []

    if len(selected_tubelets) == 0:
        eprint(f"Warning: Model gradients were dead. Falling back to top 5%.")
        ranked_all = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
        selected_tubelets = ranked_all[:max(1, int(num_tubes * 0.05))]
        
    return selected_tubelets, scores