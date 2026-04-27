from utils import *
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

SAVE_INTERMEDIATE_VISUALS = True

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def debug_save_pixels_interval(pixels_tensor, orig_tensor, output_dir, t_dim_index, filename="debug_blend_alpha_0.5.png"):
    """
    Extracts the VLM's pixel input tensor, un-normalizes it using the original 
    video's min/max to preserve the alpha-blend fading effect, and saves it.
    """
    pt = pixels_tensor.detach().cpu().float()
    pt_orig = orig_tensor.detach().cpu().float()
    # Realign dimensions to (Time, Channels, Height, Width)
    if t_dim_index == 1:
        frames = pt[0] 
        orig_frames = pt_orig[0]
    else:
        frames = pt[0].permute(1, 0, 2, 3) 
        orig_frames = pt_orig[0].permute(1, 0, 2, 3)
    
    #-- Calculate min/max from the original frames
    # --> Locks the "exposure" so faded/blurred frames actually look faded.
    amin = orig_frames.amin(dim=(0, 1, 2, 3), keepdim=True) # original absolute minimum tensor value
    amax = orig_frames.amax(dim=(0, 1, 2, 3), keepdim=True) # original absolute maximum tensor value
    frames -= amin
    frames /= (amax - amin) + 1e-5
    frames = torch.clip(frames, 0, 1) # Prevent out-of-bounds from baseline weirdness
    # Stitch frames horizontally into a single image
    grid = torchvision.utils.make_grid(frames, nrow=frames.shape[0], padding=2)
    # Save to disk
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, filename)
    torchvision.utils.save_image(grid, save_path)


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

    
def _rescale_mask(mask, new_H, new_W, crop_top, crop_left, target_H, target_W, target_T, T_orig, is_qwen, t_dim_index):
    M_resized_step = F.interpolate(mask, size=(new_H, new_W), mode='bilinear', align_corners=False)
    M_cropped_step = M_resized_step[:, :, crop_top:crop_top+target_H, crop_left:crop_left+target_W]
    M_vol_step = M_cropped_step.permute(1, 0, 2, 3).unsqueeze(0) # (1, 1, T, H, W)
    
    if target_T != T_orig:
        M_vol_step = F.interpolate(M_vol_step, size=(target_T, target_H, target_W), mode='trilinear', align_corners=False)
    if is_qwen:
        M_low_res_step = M_vol_step.view(-1, 1) 
    else:
        if t_dim_index == 1:
            M_low_res_step = M_vol_step.permute(0, 2, 1, 3, 4) # (1, T, 1, H, W)
        else:
            M_low_res_step = M_vol_step # (1, 1, T, H, W)
    return M_low_res_step

    
def _get_rescale_and_dummys(model, processor, frames, baseline_frames, is_qwen, tubelets):
    """
    Given a huggingface VLM, its corresponding processor,
    a list of frames, and a list of baseline frames, this
    function returns
    - dummy inputs original (original video)
    - dummy inputs baseline (baseline video)
    - pixels_orig:
    - pixels_base:
    - grid_thw:
    - 
    """
    with torch.no_grad():
        if is_qwen:
            dummy_inputs_orig = processor(text=[" "], videos=[frames], padding=True, return_tensors="pt", max_pixels=112896).to(model.device)
            dummy_inputs_base = processor(text=[" "], videos=[baseline_frames], padding=True, return_tensors="pt", max_pixels=112896).to(model.device)
            pixels_orig = dummy_inputs_orig['pixel_values_videos'].detach()
            pixels_base = dummy_inputs_base['pixel_values_videos'].detach()
            
            grid_thw = dummy_inputs_orig['video_grid_thw'][0] 
            target_T, target_H, target_W = grid_thw[0].item(), grid_thw[1].item(), grid_thw[2].item()
            # Qwen's tensor is strictly 2D: (total_patches, patch_features)
            target_C = pixels_orig.shape[1] 
            t_dim_index = -1 # Placeholder, standard 5D indexing does not apply
        else:
            dummy_inputs_orig = processor(text=" ", videos=frames, return_tensors="pt").to(model.device)
            dummy_inputs_base = processor(text=" ", videos=baseline_frames, return_tensors="pt").to(model.device)
            pixels_orig = dummy_inputs_orig['pixel_values_videos'].detach()
            pixels_base = dummy_inputs_base['pixel_values_videos'].detach()
            
            target_tensor_shape = pixels_orig.shape
            if target_tensor_shape[1] == 3: 
                target_C, target_T, target_H, target_W = target_tensor_shape[1:5]
                t_dim_index = 2
            else:
                target_T, target_C, target_H, target_W = target_tensor_shape[1:5]
                t_dim_index = 1
                
    #-- Cropping: Precalculate exact HF geometry scaling
    T_orig, H_orig, W_orig = tubelets.shape
    ratio = max(target_H / float(H_orig), target_W / float(W_orig))
    new_H, new_W = int(H_orig * ratio), int(W_orig * ratio)
    crop_top = (new_H - target_H) // 2
    crop_left = (new_W - target_W) // 2
    return dummy_inputs_orig, pixels_orig, pixels_base, target_T, target_H, target_W, t_dim_index, crop_top, crop_left, new_H, new_W, T_orig, H_orig, W_orig


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


def _calculate_gradient(model, tokenizer, W_raw, pixels_interval, full_ids, 
                        output_ids, positions, mode, args, 
                        is_qwen, dummy_inputs_orig):
    """
    Calculates the gradients of each weight of the tubelets.
    Returns the raw accumulated gradient so the main loop can calculate independent steps.
    """
    # Feed to model
    forward_kwargs = {
        "input_ids": full_ids,
        "attention_mask": torch.ones_like(full_ids).to(model.device), 
        "pixel_values_videos": pixels_interval.to(dtype=model.dtype),
        "use_cache": False
    }
    if is_qwen:
        forward_kwargs["video_grid_thw"] = dummy_inputs_orig["video_grid_thw"]
    
    outputs = model(**forward_kwargs)
    
    #-- Define the objective probabilities
    logits = outputs.logits
    out_len = output_ids.shape[-1]
    target_logits = logits[:, -out_len - 1 : -1, :]
    
    predicted_ids = torch.argmax(target_logits[:, 0:1, :], dim=-1)
    predicted_text = tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
    target_text = tokenizer.decode(output_ids[0], skip_special_tokens=True) # Assuming batch is 1
    
    probs = F.softmax(target_logits, dim=-1) 
    target_probs = probs.gather(dim=-1, index=output_ids.unsqueeze(-1)).squeeze(-1)
    if positions is not None and len(positions) > 0:
        target_probs = target_probs[0, positions] 
    mean_prob = target_probs.mean()
    mean_prob_val = mean_prob.item()
    log_prob = torch.log(mean_prob + 1e-7) 
    
    #-- Objective is dependend on the mode
    if mode == 'deletion':
        main_loss = log_prob / args.ig_steps 
    else:
        main_loss = -log_prob / args.ig_steps 
    main_loss.backward()
    
    raw_accumulated_grads = W_raw.grad.detach().cpu().numpy()
    
    del outputs, logits, target_logits, probs, target_probs, main_loss
    torch.cuda.empty_cache()
    gc.collect()
    
    return predicted_text, target_text, raw_accumulated_grads.copy(), mean_prob_val
    

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
    packed_inputs = _get_rescale_and_dummys(model, processor, frames, baseline_frames, is_qwen, tubelets)
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
            M_low_res_step = _rescale_mask(
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
            predicted_text, target_text, raw_accumulated, mean_prob = _calculate_gradient(
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

def create_super_tubelets(video_array, tubelets, n_clusters=12, mode='spatial'):
    """
    Clusters N tubelets into K super-tubelets.
    mode: 'spatial' (groups by physical proximity) or 'appearance' (groups by color)
    """
    num_tubes = int(tubelets.max()) + 1
    features = np.zeros((num_tubes, 3)) 
    for i in range(num_tubes):
        mask = (tubelets == i)
        if not np.any(mask): 
            continue
            
        if mode == 'appearance':
            # Mean RGB color
            features[i] = video_array[mask].mean(axis=0)[:3]
        elif mode == 'spatial':
            # 3D Centroid (T, Y, X)
            coords = np.argwhere(mask)
            features[i] = coords.mean(axis=0) 

    # Normalize features so K-Means treats all dimensions equally
    features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-5)
    # Cluster into Super-Tubelets
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto').fit(features)
    labels = kmeans.labels_ # Maps sub_id -> super_id
    # Create the 3D Super-Tubelet mask
    super_tubelets = labels[tubelets]
    
    return super_tubelets, labels