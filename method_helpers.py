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

    

def rescale_mask(mask, new_H, new_W, crop_top, crop_left, target_H, target_W, target_T, T_orig, is_qwen, t_dim_index):
    """
    Resizes mask to input size of the VLM and returns both the formatted 
    VLM tensor and the 5D volumetric tensor for TV norm calculation.
    """
    M_resized_step = F.interpolate(mask, size=(new_H, new_W), mode='bilinear', align_corners=False)
    M_cropped_step = M_resized_step[:, :, crop_top:crop_top+target_H, crop_left:crop_left+target_W]
    M_vol_step = M_cropped_step.permute(1, 0, 2, 3).unsqueeze(0) # (1, 1, T, H, W)
    
    if target_T != T_orig:
        M_vol_step = F.interpolate(M_vol_step, size=(target_T, target_H, target_W), mode='trilinear', align_corners=False)
    
    if is_qwen:
        M_scaled = M_vol_step.view(-1, 1) 
    else:
        if t_dim_index == 1:
            M_scaled = M_vol_step.permute(0, 2, 1, 3, 4) # (1, T, 1, H, W)
        else:
            M_scaled = M_vol_step # (1, 1, T, H, W)

    return M_scaled, M_vol_step


def get_rescale_and_dummys(model, processor, frames, baseline_frames, is_qwen, tubelets):
    """
    Given a huggingface VLM, its corresponding processor,
    a list of frames, and a list of baseline frames, this
    function returns dummy inputs and geometric scaling parameters.
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
                
    # Cropping: Precalculate exact HF geometry scaling
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


def calculate_gradient(model, tokenizer, W_raw, pixels_interval, full_ids, 
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