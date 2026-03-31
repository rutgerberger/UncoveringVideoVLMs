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



def _calculate_gradient(model, tokenizer, W_raw, pixels_interval, full_ids, 
                          output_ids, positions, mode, args, 
                          is_qwen, dummy_inputs_orig, reg_grads):
    """

    Given a value of alpha in [0, 1], and a mask,
    this function calculates the gradients of each
    weight of the tubelets.

    Intuitively, alpha determines how much of the baseline
    video is still visible. 

    When measuring deletion, we want to start with an 
    original video. At alpha = 0: we get the complete
    original video (so masking has no effect). For
    alpha = 0.5: the bixels are mixed between the
    original video and the masked video. At this pointe
    we measure 'how does the weight of this mask affect
    the (negative) probability of the final output tokens'.
    
    If this is positive, we want to increase the weight -
        apparently, if we would increase the weight, the
        model is less certain of its answer.

    The other way around counts for insertion.

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
    predicted_ids = torch.argmax(target_logits, dim=-1)

    predicted_text = tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
    target_text = tokenizer.decode(output_ids[0], skip_special_tokens=True) # Assuming batch is 1

    probs = F.softmax(target_logits, dim=-1) 
    target_probs = probs.gather(dim=-1, index=output_ids.unsqueeze(-1)).squeeze(-1)
    if positions is not None and len(positions) > 0:
        target_probs = target_probs[0, positions] 
    mean_prob = target_probs.mean()
    log_prob = torch.log(mean_prob + 1e-7) 

    #-- Objective is dependend on the mode
    if mode == 'deletion':
        main_loss = log_prob / args.ig_steps 
    else:
        main_loss = -log_prob / args.ig_steps 
    main_loss.backward()
    
    # Logging Model Gradients
    real_grads = W_raw.grad.detach().cpu().numpy() - reg_grads
    #real_grads = np.abs(real_grads)

    # Cleanup (memory management)
    #del W_step, M_high_res_step, M_resized_step, M_cropped_step, M_vol_step, M_low_res_ste
    del outputs, logits, target_logits, probs, target_probs, main_loss
    torch.cuda.empty_cache()
    gc.collect()

    return predicted_text, target_text, real_grads.copy()
    
def optimize_tubelet_weights(
        args, model, tokenizer, processor, full_ids, output_ids, frames, baseline_frames, 
        tubelets, positions, mode='deletion', stage="init"
    ):
    """
    Gradually move the weights of the tubelets 
    to optimize the insertion / deletion loss.
    
    deletion: baseline_frames constant
    insertion: baseline_frames constant/blurred
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
    
    #-- Lists to keep track of gradients (debugging purposes)
    top_10_gradients = []
    mean_gradients = []
    reg_gradients = []
    sparse_gradient_counts = []
    
    #-- How much steps we want to take
    for i in range(args.iterations):
        optimizer.zero_grad() # Reset gradients
        W = torch.sigmoid(W_raw) # Sigmoid(W) \in [0,1] (this is what we want, but optimizing in <-inf, inf> is easier)
        M_high_res = W[tubelets_tensor].unsqueeze(1).float() # Replace tubelet IDs with weights, add extra dim

        #-- Resizing and cropping
        M_resized_tv = F.interpolate(M_high_res, size=(new_H, new_W), mode='bilinear', align_corners=False)
        # VLMs do perform center crop to go to target_H and target_W
        M_cropped_tv = M_resized_tv[:, :, crop_top:crop_top+target_H, crop_left:crop_left+target_W]
        M_vol_tv = M_cropped_tv.permute(1, 0, 2, 3).unsqueeze(0) # Shape: (1, 1, T, H, W) for 3D TV norm
        if target_T != T_orig: #safety check
            M_vol_tv = F.interpolate(M_vol_tv, size=(target_T, target_H, target_W), mode='trilinear', align_corners=False)
            
        #-- Compute Regularization Loss Outside the IG loop
        loss_tv = tv_norm_3d(M_vol_tv, tv_beta=2)
        loss_l1 = torch.mean(torch.abs(1.0 - W)) if mode == 'deletion' else torch.mean(torch.abs(W))
        reg_loss = args.reg_lambda * (loss_l1 + loss_tv)
        reg_loss.backward() # frees this small subgraph (save computation / mem)
        reg_grads = W_raw.grad.detach().cpu().numpy() # Log Reg Gradients
        reg_gradients.append(reg_grads.copy())
        del W, M_high_res, M_resized_tv, M_cropped_tv, M_vol_tv, reg_loss

        # --- Main Integrated Gradients Loop ---
        for step in range(1, args.ig_steps + 1):
            alpha = step / args.ig_steps
            #Create mask
            W_step = torch.sigmoid(W_raw)
            M_high_res_step = W_step[tubelets_tensor].unsqueeze(1).float() # Replace tubelet IDs with weights, add extra dim
            M_low_res_step = _rescale_mask(
                M_high_res_step, new_H, new_W, crop_top, crop_left, 
                target_H, target_W, target_T, T_orig, is_qwen, t_dim_index
            )
            # Get pixel values
            #
            # pixels_orig : original video (video)
            # pixels_base : baseline (constant video, or blurred)
            # pixels_final is a mix:
            #   M_low_res_step==
            #                Rescaled mask to work on the model's input space.
            #                This gives values between 0 and 1
            #   If M == 1: we see the complete original video
            #   If M == 0: we see the baseline video
            #
            # For deletion, the mask is initialized around 0.9.
            #   the first round, pixels_final = a slightly less bright original video
            #   the situation we are aiming at: 
            #       'decrease the weights at important parts'
            #        --> this leads to 'hiding' parts of the original input
            #        --> and introducing 'constant' parts which do not tell us much
            #   say the second round, some weights have decreased --> see above
            #
            # The end-point of the IG path shows this masked video
            #       the start-point should show a completely visible video
            #       shouldn't this be a path from a constant video as well?
            #       why do we do original video inside interval?
            # 
            # The IG Baseline (alpha = 0): This is the starting state of the Integrated Gradients integral. 
            # In IG, the baseline represents the state of "zero intervention" or the reference state before
            # your current parameters take effect.
            #
            # For the deletion metric, the primary objective is to find out which tubelets, when removed,
            # destroy the model's confidence the fastest
            #
            # Start of Path (alpha=0): pixels_orig (The completely unmasked video).
            # This is the "zero intervention" state where the mask has not been applied yet. 
            # The model is fully confident.
            #
            # End of Path (alpha=1): pixels_final (The video with your current mask M applied,
            # where areas with low weights are replaced by pixels_base).
            #
            # Likewise for insertion

            pixels_final = pixels_orig * M_low_res_step + pixels_base * (1.0 - M_low_res_step)
            if mode == 'deletion':
                pixels_interval = pixels_final * alpha + pixels_orig * (1.0 - alpha)
            else:
                pixels_interval = pixels_final * alpha + pixels_base * (1.0 - alpha)
            
            predicted_text, target_text, real_grads = _calculate_gradient(model, tokenizer, W_raw, pixels_interval, full_ids, 
                          output_ids, positions, mode, args, 
                          is_qwen, dummy_inputs_orig, reg_grads)

            if i % 5 == 0:  # For debugging and logging
                save_folder = os.path.join(args.output_dir, f'gradients/{stage}')
                os.makedirs(save_folder, exist_ok=True)

                # Show what the model exactly saw at this interval
                debug_save_pixels_interval(
                    pixels_tensor=pixels_interval, 
                    orig_tensor=pixels_orig, 
                    output_dir=save_folder, 
                    filename=f"debug_blend_alpha_{alpha:.2f}.png",
                    t_dim_index=t_dim_index
                )

                # To align with the uncropped tubelets, we must fade the original high-res frames,
                # rather than trying to un-crop and stretch the VLM's square tensor.
                mask_hr = W_step[tubelets_tensor].detach().cpu().numpy() # Shape: (T_orig, H_orig, W_orig)
                faded_hr_frames = []
                for t_idx in range(T_orig):
                    # Convert original and baseline frames to float arrays for blending
                    orig_f = np.array(frames[t_idx]).astype(np.float32)
                    base_f = np.array(baseline_frames[t_idx]).astype(np.float32)
                    # The mask_hr array is a 3D volume with the shape (Time,Height,Width).
                    # By using t_idx, we are grabbing a single frame at timestep t_idx.
                    # ... (Ellipsis) is shorthand for "include all remaining dimensions" 
                    # (in this case, all Height and Width pixels).
                    m = mask_hr[t_idx, ..., np.newaxis] # Add channel dim (H, W, 1)
                    # Recreate the Integrated Gradients alpha blend in high-res
                    final_f = orig_f * m + base_f * (1.0 - m)
                    # Blend this guy in here, using similar logic as before
                    if mode == 'deletion':
                        faded_f = final_f * alpha + orig_f * (1.0 - alpha)
                    else:
                        faded_f = final_f * alpha + base_f * (1.0 - alpha)
                    faded_hr_frames.append(np.clip(faded_f, 0, 255).astype(np.uint8))

                # Visualize how the gradients exactly look like at this interval
                eprint(f"Stage {stage} Iter {i+1}: Predicted {predicted_text} | Target {target_text}")
                visualize_gradients(
                    gradients=real_grads, 
                    frames=faded_hr_frames, 
                    tubelets=tubelets, 
                    save_folder=save_folder, 
                    step=step, 
                    title=f"IG_Iter_{i+1}_Alpha_{alpha:.2f}"
                )


            real_grads = np.abs(real_grads)
            top_10_gradients.append(np.sort(real_grads)[-10:])
            mean_gradients.append(np.mean(real_grads))  
            sparse_gradient_counts.append(np.sum(real_grads < 1e-4))

        #-- Update the weights of the tubelets 
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
    
    #print(f"mean_gradients: {mean_gradients}  \n top_10_gradients: {top_10_gradients} \n reg_gradients: {reg_gradients} \n sparse_gradient_counts: {sparse_gradient_counts} ")

    # --- FUNCTION EXIT CLEANUP ---
    del pixels_orig, pixels_base
    del dummy_inputs_orig
    del optimizer, W_raw
    torch.cuda.empty_cache()
    gc.collect()

    return selected_tubelets, scores