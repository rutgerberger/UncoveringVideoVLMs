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

# -- For Lazy Greedy Search

def initialize_lazy_greedy_queues(args, model, processor, full_ids, output_ids, video_array, tubelets, unique_tubes, baseline_ins, baseline_del, prob_orig, prob_base, positions):
    queue_ins = []
    queue_del = []
    
    for tube in unique_tubes:
        # -- Gather probabilities for insertion
        frames_ins = apply_universal_mask(video_array, baseline_ins, tubelets, [tube])
        prob_ins = get_prob(args, model, processor, full_ids, output_ids, frames_ins, positions)
        
        # -- Gather probabilities for deletion
        keep_tubes = [t for t in unique_tubes if t != tube]
        frames_del = apply_universal_mask(video_array, baseline_del, tubelets, keep_tubes)
        prob_del = get_prob(args, model, processor, full_ids, output_ids, frames_del, positions)
        
        # -- Populate separate queues
        heapq.heappush(queue_ins, (-(prob_ins - prob_base), tube))
        heapq.heappush(queue_del, (-(prob_orig - prob_del), tube))
        
    return queue_ins, queue_del



def run_lazy_greedy_search(args, model, processor, full_ids, output_ids, 
        video_array, tubelets, unique_tubes, queue, mode, baseline_ins, 
        baseline_del, prob_orig, prob_base, centroids_dict, positions):
    """
    requires:
             args, model, processor: common sense
             full_ids, output_ids:   ids of full prompt and output
             video_array:            original frames
             tubelets, unique_tubes: set of tubelets (should be similar)
             queue:                  initialized heapq
             mode:                   'insertion' or 'deletion'
             baseline_ins/del:       baseline frames (blurred / constant)
             prob_orig:              list of probabilities (per token ID)
             prob_base:              same but then for blurred vid
             centroids_dict:         precomputed centroids for distance penalty
             positions:              keyword positions
    returns: 
             selected_tubes:         optimal set of tubelets which maximizes 
                                     the objective function, based on mode
    """
    selected_tubes = []
    tubelet_scores = {}
    current_total_score = 0.0 
    
    # -- Main Loop --
    for step in range(len(unique_tubes)):
        best_tube = None
        best_marginal_gain = -float('inf')
        best_absolute_score = -float('inf')
        prob_ins = 0.0
        prob_del = 0.0

        while queue:
            # -- Prerequisites: insertion / deletion probs
            _, candidate = heapq.heappop(queue)
            candidate_set = selected_tubes + [candidate]
            if mode == 'insertion':
                frames_ins = apply_universal_mask(video_array, baseline_ins, tubelets, candidate_set)
                prob_ins = get_prob(args, model, processor, full_ids, output_ids, frames_ins, positions)
            else:
                keep_tubes = [t for t in unique_tubes if t not in candidate_set]
                frames_del = apply_universal_mask(video_array, baseline_del, tubelets, keep_tubes)
                prob_del = get_prob(args, model, processor, full_ids, output_ids, frames_del, positions)

            # -- Optional: distance penalty
            dist_penalty = get_distance_penalty(candidate, selected_tubes, centroids_dict)
            
            # -- Score Reflection (Match Objective Function)
            if mode == 'insertion':
                actual_absolute_score = (prob_ins - prob_base) - (args.L1 * dist_penalty)
            else:
                actual_absolute_score = (prob_orig - prob_del) - (args.L1 * dist_penalty)
                
            actual_marginal_gain = actual_absolute_score - current_total_score
            
            # -- Queue is empty: done
            if not queue:
                best_tube = candidate
                best_marginal_gain = actual_marginal_gain
                best_absolute_score = actual_absolute_score
                break
                
            next_best_upper_bound_gain = -queue[0][0] - current_total_score

            # -- Calculated Gain is actually > Next Gain (Lazy Greedy): use this one
            if actual_marginal_gain >= next_best_upper_bound_gain:
                best_tube = candidate
                best_marginal_gain = actual_marginal_gain
                best_absolute_score = actual_absolute_score
                break
            # -- If not update the score
            else:
                heapq.heappush(queue, (-actual_absolute_score, candidate))
                
        tubelet_scores[best_tube] = max(0.0, best_marginal_gain)
        current_total_score = best_absolute_score
        selected_tubes.append(best_tube)
        
        # -- Logging and ratio calculation based on mode
        if mode == 'insertion':
            eprint(f"Ins Step {step+1}: Tube {best_tube} | Gain: {best_marginal_gain:.4f} | Total: {current_total_score:.4f} | P_ins: {prob_ins:.4f}")
            objective_ratio = prob_ins / max(prob_orig, 1e-5)
        else:
            eprint(f"Del Step {step+1}: Tube {best_tube} | Gain: {best_marginal_gain:.4f} | Total: {current_total_score:.4f} | P_orig-P_del: {(prob_orig - prob_del):.4f}")
            objective_ratio = (prob_orig - prob_del) / max(prob_orig, 1e-5)

        # -- Early Stopping
        if len(selected_tubes) > len(unique_tubes) * 0.25 or objective_ratio >= 0.90:
            if best_marginal_gain < 1e-4:
                eprint(f"Early stopping triggered at step {step+1}: Sufficient confidence reached and gain flatlined.")
                break
                
    return selected_tubes, tubelet_scores



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
    """
    Mode = deletion
        Start with original video (?), goal is to find 
        which pixels we need to remove to drop the
        probabilities to 0.
    Mode = insertion
        Start with blurred video, goal is to find
        which pixels we need to insert to increase
        the probabilities.
    """
    num_tubes = int(tubelets.max()) + 1
    #-- Initialization of masks \all X \in {-2.0,2.0} dependent on Deletion / Insertion
    if mode == 'deletion': 
        W_raw = torch.full((num_tubes,), 2.0, dtype=torch.float32, device=model.device, requires_grad=True)
    else: 
        W_raw = torch.full((num_tubes,), -2.0, dtype=torch.float32, device=model.device, requires_grad=True)
    tubelets_tensor = torch.tensor(tubelets, device=model.device, dtype=torch.long) #For masking gradients
    #-- Because we optimize w.r.t. output logits, we need tensors to contain computational graph
    rgb_orig = torch.stack([TF.to_tensor(img.convert('RGB')) for img in frames]).to(model.device)
    rgb_base = torch.stack([TF.to_tensor(img.convert('RGB')) for img in baseline_frames]).to(model.device)

    # -- Figure out what the model expects by running the processor once (no gradients)
    with torch.no_grad():
        dummy_inputs = processor(text=" ", videos=frames, return_tensors="pt").to(model.device)
        target_tensor_shape = dummy_inputs['pixel_values_videos'].shape
        
        # Dynamically handle (B, C, T, H, W) vs (B, T, C, H, W)
        if target_tensor_shape[1] == 3: # (B, C, T, H, W)
            target_C, target_T, target_H, target_W = target_tensor_shape[1:5]
            t_dim_index = 2
        else: # (B, T, C, H, W) -> standard for Hugging Face Video-LLaVA
            target_T, target_C, target_H, target_W = target_tensor_shape[1:5]
            t_dim_index = 1

    #-- Model normalization constants (Standard OpenAI CLIP values used by LLaVA/Qwen)
    mean_vals = [0.48145466, 0.4578275, 0.40821073]
    std_vals = [0.26862954, 0.26130258, 0.27577711]
    
    # Adjust views so they align with the expected tensor format for broadcasting
    if t_dim_index == 1:
        mean = torch.tensor(mean_vals).view(1, 1, 3, 1, 1).to(model.device)
        std = torch.tensor(std_vals).view(1, 1, 3, 1, 1).to(model.device)
    else:
        mean = torch.tensor(mean_vals).view(1, 3, 1, 1, 1).to(model.device)
        std = torch.tensor(std_vals).view(1, 3, 1, 1, 1).to(model.device)
    
    optimizer = torch.optim.Adam([W_raw], lr=args.lr)
    
    #-- We take 'args.iterations' number of steps using IG
    for i in range(args.iterations):
        optimizer.zero_grad()
        #-- Masking. Mask with tubelets, blend original video with baseline video
        W = torch.sigmoid(W_raw)
        M_high_res = W[tubelets_tensor] 
        M_high_res = M_high_res.unsqueeze(1) #Insert new dimension: (T, H, W) --> (T, C, H, W)
        blended_rgb = rgb_orig * M_high_res + rgb_base * (1.0 - M_high_res) # 'Real' space

        #-- Moving to latent space
        #   Take blended_rgb (created video tensor) and transform it into the
        #   exact shape and format that the vision model expects for its
        #   pixel_value_videos input
        _, _, H_orig, W_orig = blended_rgb.shape
        ratio = max(target_H / H_orig, target_W / W_orig)
        new_H, new_W = int(H_orig * ratio), int(W_orig * ratio) # First we preserve the original aspect ratio
        blended_resized = F.interpolate(blended_rgb, size=(new_H, new_W), mode='bilinear', align_corners=False)
        crop_top = (new_H - target_H) // 2 #Here, we calculate starting pixel coordinates for the crop
        crop_left = (new_W - target_W) // 2 #Find exact offset needed to trim an equal amount from both sides
        blended_cropped = blended_resized[:, :, crop_top:crop_top+target_H, crop_left:crop_left+target_W] #Cut the video
        blended_cropped = blended_cropped.permute(1, 0, 2, 3).unsqueeze(0) #(T,C,H,W) --> (1,C,T,H,W)
        # F.interpolate (trilinear) ALWAYS expects (N, C, D, H, W), which is why we keep it as (1, C, T, H, W) here
        if blended_cropped.shape[2] != target_T:
            blended_cropped = F.interpolate(blended_cropped, size=(target_T, target_H, target_W), mode='trilinear', align_corners=False)
        if t_dim_index == 1:
            blended_cropped = blended_cropped.permute(0, 2, 1, 3, 4) # (1, C, T, H, W) -> (1, T, C, H, W)
        pixels_final = (blended_cropped - mean) / std
        
        #-- Calculating regularizations (TV norm, L1 norm)
        mask_for_tv = M_high_res.permute(1, 0, 2, 3).unsqueeze(0)
        loss_tv = tv_norm_3d(mask_for_tv, tv_beta=2)
        loss_l1 = torch.mean(torch.abs(1.0 - W)) if mode == 'deletion' else torch.mean(torch.abs(W))
        reg_loss = (args.L1_lambda * loss_l1) + (args.TV_lambda * loss_tv)
        reg_loss.backward(retain_graph=True)  #-- Add these gradients to the computational graph already

        #-- The main integrated gradients loop: sum the gradient over ~ig_steps
        for step in range(1, args.ig_steps + 1):
            #-- Forward the model with a blurred version (where mask is true) combined with the original version
            alpha = step / args.ig_steps #The amount of 'blur'
            pixels_interval = pixels_final * alpha + (dummy_inputs['pixel_values_videos'].detach() * (1.0 - alpha))
            forward_kwargs = {
                "input_ids": full_ids,
                "attention_mask": torch.ones_like(full_ids).to(model.device), 
                "pixel_values_videos": pixels_interval.to(dtype=model.dtype),
                "use_cache": True
            }
            outputs = model(**forward_kwargs)

            #-- Obtain probabilities
            logits = outputs.logits 
            out_len = output_ids.shape[-1]
            target_logits = logits[:, -out_len - 1 : -1, :] 
            probs = F.softmax(target_logits, dim=-1) 
            target_probs = probs.gather(dim=-1, index=output_ids.unsqueeze(-1)).squeeze(-1)

            #-- Filter on keyword positions
            if positions is not None and len(positions) > 0:
                target_probs = target_probs[0, positions] 
            mean_prob = target_probs.mean()
            log_prob = torch.log(mean_prob + 1e-7) # Steeper direction to take... (99.98 > 1.0 larger diff.)

            #-- The goal is to maximize / minimize the log probs dependent on the mode
            if mode == 'deletion':
                main_loss = log_prob / args.ig_steps 
            else:
                main_loss = -log_prob / args.ig_steps  #We're optimizing towards INCREASING
            
            retain = (step < args.ig_steps)
            main_loss.backward(retain_graph=retain) # Accumulates gradients across steps

        optimizer.step()
        if (i+1) % 10 == 0:
            eprint(f"Iter {i+1}/{args.iterations} | TV: {loss_tv.item():.4f} | L1: {loss_l1.item():.4f}")

    # -- Generate final output weights
    final_weights = torch.sigmoid(W_raw).detach().cpu().numpy()
    # Relative movement instead of absolute values (final evaluation)
    if mode == 'deletion':
        init_w = sigmoid(2.0) # ~0.8808
        scores = {t: float(init_w - w) for t, w in enumerate(final_weights)}
    else:
        init_w = sigmoid(-2.0) # ~0.1192
        scores = {t: float(w - init_w) for t, w in enumerate(final_weights)}
        
    # Dynamic Thresholding (Relative to the strongest signal)
    max_score = max(scores.values())
    
    # If the strongest signal moved at least a tiny bit
    if max_score > 0.01: 
        # We keep any tubelet that moved at least 20% as much as the most-moved tubelet
        dynamic_threshold = max_score * 0.2
        selected_tubelets = [t for t, s in scores.items() if s >= dynamic_threshold]
    else:
        selected_tubelets = []

    # Fallback (Only if gradients completely died)
    if len(selected_tubelets) == 0:
        eprint(f"Warning: Model gradients were dead. Falling back to top 5%.")
        ranked_all = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
        selected_tubelets = ranked_all[:max(1, int(num_tubes * 0.05))]
    #eprint(max_score, dynamic_threshold, selected_tubelets, scores)
    return selected_tubelets, scores


# def optimize_tubelet_weights(
#             args, 
#             model, 
#             processor, 
#             full_ids, 
#             output_ids, 
#             frames, 
#             baseline_frames, 
#             tubelets, 
#             positions, 
#             mode='deletion', 
#             args.iterations=50, 
#             lr=0.05, 
#             L1_lambda=0.05
#     ):
#     """
#     Alternative to run lazy greedy search.
#         Instead of performing lazy greedy over a set,
#         this function performs SGD over a vector of
#         weights per tubelet. 

#     Mode = deletion
#         Start with original video, goal is to find 
#         which pixels we need to remove to drop the
#         probabilities to 0.

#     Mode = insertion
#         Start with blurred video, goal is to find
#         which pixels we need to insert to increase
#         the probabilities.
#     """
#     num_tubes = int(tubelets.max()) + 1
    
#     # -- Initialize Weights (based on mode)
#     # 1. Initialize UNBOUNDED weights for Sigmoid
#     if mode == 'deletion': 
#         # sigmoid(2.0) ~ 0.88 (Starts mostly visible, pushing towards 0)
#         W_raw = torch.full((num_tubes,), 2.0, dtype=torch.float32, device=model.device, requires_grad=True)
#     else: 
#         # sigmoid(-2.0) ~ 0.11 (Starts mostly blurred, pushing towards 1)
#         W_raw = torch.full((num_tubes,), -2.0, dtype=torch.float32, device=model.device, requires_grad=True)

#     # Keep tubelets at ORIGINAL resolution
#     tubelets_tensor = torch.tensor(tubelets, device=model.device, dtype=torch.long)
    
#     # -- Pre-process frames into latent tensors (Done outside loop to save compute)
#     inputs_orig = processor(text=" ", videos=frames, return_tensors="pt").to(model.device)
#     inputs_base = processor(text=" ", videos=baseline_frames, return_tensors="pt").to(model.device)
#     pixels_orig = inputs_orig['pixel_values_videos'].detach()
#     pixels_base = inputs_base['pixel_values_videos'].detach()
    
#     # Assuming pixels_orig shape is: (Batch, Channels, Time, Height, Width)
#     target_T = pixels_orig.shape[2]
#     target_H = pixels_orig.shape[3]
#     target_W = pixels_orig.shape[4]
    
#     optimizer = torch.optim.Adam([W_raw], lr=lr)
    
#     # -- Main Loop
#     for i in range(args.iterations):
#         optimizer.zero_grad()
#         # -- Create Mask / Adjust dimensions
#         # Pass through Sigmoid to get continuous [0, 1] weights
#         W = torch.sigmoid(W_raw)
#         # Map weights to the ORIGINAL high-res spatial dimensions
#         M_high_res = W[tubelets_tensor] # Shape: (T_orig, H_orig, W_orig)
#         # Add batch and channel dims for 3D interpolation -> (1, 1, T, H, W)
#         M_high_res = M_high_res.unsqueeze(0).unsqueeze(0) 
        
#         # Trilinear interpolation - smooth mask boundaries
#         M_expanded = F.interpolate(
#             M_high_res, 
#             size=(target_T, target_H, target_W), 
#             mode='trilinear', 
#             align_corners=False
#         )
        
#         # -- Mix Tensors
#         pixels_blended = pixels_orig * M_expanded + pixels_base * (1.0 - M_expanded)
        
#         # -- Differentiable Forward Pass
#         forward_kwargs = {
#             "input_ids": full_ids,
#             "attention_mask": torch.ones_like(full_ids).to(model.device),
#             "pixel_values_videos": pixels_blended.to(dtype=model.dtype),
#             "use_cache": True
#         }
#         outputs = model(**forward_kwargs)
        
#         # -- Calculate target probability
#         logits = outputs.logits 
#         out_len = output_ids.shape[-1]
#         target_logits = logits[:, -out_len - 1 : -1, :] 
#         probs = F.softmax(target_logits, dim=-1) 
#         target_probs = probs.gather(dim=-1, index=output_ids.unsqueeze(-1)).squeeze(-1)
        
#         if positions is not None and len(positions) > 0:
#             target_probs = target_probs[0, positions] 
    
#         # -- Calculate Loss (specific to mode)
#         # Log-Probabilities to prevent vanishing gradients
#         mean_prob = target_probs.mean()
#         log_prob = torch.log(mean_prob + 1e-7)
        
#         if mode == 'deletion':
#             loss_main = log_prob # Minimize log probability (of keywords)
#             loss_l1 = torch.mean(torch.abs(1.0 - W)) # Penalize W moving away from 1
#         else:
#             loss_main = -log_prob # Maximize log probability (of keywords)
#             loss_l1 = torch.mean(torch.abs(W)) # Penalize W moving away from 0
            
#         total_loss = loss_main + (L1_lambda * loss_l1)
        
#         # -- Update weights
#         total_loss.backward()
#         optimizer.step()
            
#     # Detach and get final sigmoid weights
#     final_weights = torch.sigmoid(W_raw).detach().cpu().numpy()
    
#     # -- Convert weights to scores
#     if mode == 'deletion':
#         # Invert weights for scoring (1.0 = highly important/deleted, 0.0 = ignored)
#         scores = {t: 1.0 - w for t, w in enumerate(final_weights)}
#     else:
#         # Raw weights act as scores (1.0 = highly important/inserted, 0.0 = ignored)
#         scores = {t: float(w) for t, w in enumerate(final_weights)}
        
#     # -- Binarize the continuous weights (Thresholding)
#     # We consider a tubelet "selected" if the optimizer pushed its score past 0.5
#     threshold = 0.5
#     selected_tubelets = [t for t, s in scores.items() if s > threshold]
    
#     # -- Fallback Safety Net
#     # If the L1 penalty was very aggressive and pushed everything below 0.5,
#     # we fall back to selecting the top 10% of tubelets so the downstream pipeline doesn't crash.
#     if len(selected_tubelets) == 0:
#         eprint(f"Warning: No tubelets crossed the {threshold} threshold. Falling back to top 10%.")
#         ranked_all = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
#         selected_tubelets = ranked_all[:max(1, int(num_tubes * 0.1))]
        
#     return selected_tubelets, scores




# -- For Distance Penalties




def precompute_tubelet_centroids(tubelets, unique_tubes):
    """
    Calculates the normalized 3D centroid (t, y, x) for every tubelet.
    Returns a dictionary mapping tube ID -> numpy array of (t, y, x).
    """
    T, H, W = tubelets.shape
    # Calculate center of mass for all unique tubes at once
    # center_of_mass returns a list of (t, y, x) tuples
    centroids_list = center_of_mass(np.ones_like(tubelets), tubelets, unique_tubes)
    centroids_dict = {}
    for tube_id, (t, y, x) in zip(unique_tubes, centroids_list):
        # Normalize to [0, 1] so time and space scales don't distort the distance
        norm_t = t / T if T > 1 else 0.0
        norm_y = y / H
        norm_x = x / W
        centroids_dict[tube_id] = np.array([norm_t, norm_y, norm_x])
    return centroids_dict



def get_distance_penalty(candidate_tube, selected_tubes, centroids_dict):
    """
    Calculates the minimum distance from the candidate to the already selected group.
    Returns 0 if no tubes are selected yet.
    """
    if not selected_tubes:
        return 0.0 # First tubelet has no penalty
    candidate_pos = centroids_dict[candidate_tube]
    # Get positions of all currently selected tubes
    selected_positions = np.array([centroids_dict[t] for t in selected_tubes])
    # Calculate Euclidean distance from candidate to all selected tubes
    distances = np.linalg.norm(selected_positions - candidate_pos, axis=1)
    # We penalize based on the distance to the *closest* selected tube.
    # This encourages the candidate to "attach" to the existing blob.
    min_distance = np.min(distances)
    return min_distance