"""
igos_frame_wise.py
Frame-by-frame iGOS++ adaptation for Video VLMs.
"""

import math
import time
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import os
import json
import numpy as np
import sys
import cv2
import matplotlib.pyplot as plt

from textwrap import fill


from qwen_vl_utils import process_vision_info

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
    

def exp_decay(init, iter, gamma=0.2):
    return init * math.exp(-gamma * iter)

def get_token_probs_tensor(args, model, full_ids, output_ids, pixel_values, positions=None, is_qwen=False, video_grid_thw=None):
    forward_kwargs = {
        "input_ids": full_ids,
        "attention_mask": torch.ones_like(full_ids).to(model.device),
        "pixel_values_videos": pixel_values,
        "use_cache": True
    }
    
    if is_qwen and video_grid_thw is not None:
        forward_kwargs["video_grid_thw"] = video_grid_thw
        
    outputs = model(**forward_kwargs)
    logits = outputs.logits  
    out_len = output_ids.shape[-1] 
    target_logits = logits[:, -out_len - 1 : -1, :] 
    probs = F.softmax(target_logits, dim=-1) 
    target_probs = probs.gather(dim=-1, index=output_ids.unsqueeze(-1)).squeeze(-1)
    
    if target_probs.dim() == 0:
        target_probs = target_probs.unsqueeze(0)
    if positions is not None and len(positions) > 0:
        target_probs = target_probs[0, positions] 
        
    return target_probs

def process_vid_qwen(processor, frames, prompt=" ", apply_chat_template=False, fps=1.0, max_pixels=112896):
    """
    Universal wrapper for Qwen's vision processor to prevent feature/token mismatches.
    """
    messages = [
        {"role": "user", "content": [
            {"type": "video", "video": frames, "fps": fps},
            {"type": "text", "text": prompt},
        ]},
    ]
    image_inputs, video_inputs = process_vision_info(messages)
    
    # Only format with the <|im_start|> tags if we are explicitly asking for an answer
    if apply_chat_template:
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        text_inputs = [text]
    else:
        text_inputs = [prompt]
        
    inputs = processor(
        text=text_inputs, 
        images=image_inputs, 
        videos=video_inputs, 
        padding=True, 
        return_tensors="pt", 
        max_pixels=max_pixels
    )
    
    return inputs

def phi_tensor(img_tensor, baseline_tensor, mask):
    return img_tensor * mask + baseline_tensor * (1.0 - mask)

def spatio_temporal_bilateral_tv(video, mask, tv_beta=2, sigma=0.01, lambda_t=0.5):
    """
    video: (T, C, H, W) or (1, T, C, H, W) or 2D flattened sequence (Qwen)
    mask: (T, 1, size, size)
    lambda_t: weight for temporal smoothness
    """
    # Squeeze batch dimension from video if present (e.g., 1, T, C, H, W -> T, C, H, W)
    if video.dim() == 5 and video.shape[0] == 1:
        video = video.squeeze(0)
        
    if video.dim() == 4 and video.shape[1] == mask.shape[0]: 
        video = video.permute(1, 0, 2, 3)
        
    if video.dim() < 4:
        dh_mask = torch.abs(mask[:, :, :-1, :] - mask[:, :, 1:, :]).pow(tv_beta)
        dw_mask = torch.abs(mask[:, :, :, :-1] - mask[:, :, :, 1:]).pow(tv_beta)
        spatial_tv = torch.mean(dh_mask) + torch.mean(dw_mask)
        
        # Temporal TV
        if mask.shape[0] > 1: 
            dt_mask = torch.abs(mask[:-1, :, :, :] - mask[1:, :, :, :]).pow(tv_beta)
            temporal_tv = torch.mean(dt_mask)
        else:
            temporal_tv = 0.0
        return spatial_tv + (lambda_t * temporal_tv)

    # Upscale mask to match video resolution for element-wise operations
    up_mask = F.interpolate(mask, size=(video.shape[-2], video.shape[-1]), mode='bilinear', align_corners=False)
    dh_mask = torch.abs(up_mask[:, :, :-1, :] - up_mask[:, :, 1:, :]).pow(tv_beta)
    dw_mask = torch.abs(up_mask[:, :, :, :-1] - up_mask[:, :, :, 1:]).pow(tv_beta)
    # Extract edges from the video (for Bilateral weighting)
    dh_img = torch.exp(-(video[:, :, :-1, :] - video[:, :, 1:, :]).mean(dim=1, keepdim=True)**2 / sigma)
    dw_img = torch.exp(-(video[:, :, :, :-1] - video[:, :, :, 1:]).mean(dim=1, keepdim=True)**2 / sigma)
    
    spatial_tv = torch.mean(dh_img * dh_mask) + torch.mean(dw_img * dw_mask)

    # Temporal TV (T dimension) to prevent flickering
    if up_mask.shape[0] > 1: # If T > 1
        dt_mask = torch.abs(up_mask[:-1, :, :, :] - up_mask[1:, :, :, :]).pow(tv_beta)
        dt_img = torch.exp(-(video[:-1, :, :, :] - video[1:, :, :, :]).mean(dim=1, keepdim=True)**2 / sigma)
        temporal_tv = torch.mean(dt_img * dt_mask)
    else:
        temporal_tv = 0.0
        
    return spatial_tv + (lambda_t * temporal_tv)

def integrated_gradient_framewise(args, model, full_ids, output_ids, img_tensor, base_tensor, mask_input, num_iter, positions, target_H, target_W, target_T, is_qwen=False, video_grid_thw=None):
    intervals = torch.linspace(1/num_iter, 1, num_iter, device=img_tensor.device).view(-1, 1, 1, 1, 1)
    total_loss = 0.0
    
    for alpha in intervals:
        if isinstance(mask_input, tuple):
            mask = mask_input[0] * mask_input[1]
        else:
            mask = mask_input
        # Mask is (T, 1, size, size) -> Upscale to (T, 1, target_H, target_W)
        up_mask = F.interpolate(mask, size=(target_H, target_W), mode='bilinear', align_corners=False)

        if is_qwen:
            # Qwen expects a flat sequence of patches (T * H * W, 1)
            interval_mask = up_mask.reshape(-1, 1) * alpha.view(1)
        else:
            # Standard video expects (1, 1, T, H, W)
            up_mask_5d = up_mask.permute(1, 0, 2, 3).unsqueeze(0) 
            interval_mask = up_mask_5d * alpha
            
        blended = phi_tensor(img_tensor, base_tensor, interval_mask)
        blended = blended + torch.randn_like(blended) * 0.2
        
        probs = get_token_probs_tensor(args, model, full_ids, output_ids, blended, positions, is_qwen=is_qwen, video_grid_thw=video_grid_thw)
        
        # Optimize log-probs to prevent gradient saturation
        step_loss = torch.log(probs + 1e-7).sum() / num_iter
        step_loss.backward()
        total_loss += step_loss.item()
        
    return total_loss

def video_iGOS_pp(
        args, model, processor, full_ids, output_ids, frames, baseline_frames, positions,
        size=32, iterations=15, ig_iter=6, L1=1, L2=1, L3=20, lr=10):

    is_qwen = getattr(args, 'model', '') == 'qwen'

    if is_qwen:
        inputs_orig = process_vid_qwen(processor, frames, prompt=" ", max_pixels=112896).to(model.device)
        inputs_base = process_vid_qwen(processor, baseline_frames, prompt=" ", max_pixels=112896).to(model.device)
        video_grid_thw = inputs_orig['video_grid_thw']
        target_T, target_H, target_W = video_grid_thw[0][0].item(), video_grid_thw[0][1].item(), video_grid_thw[0][2].item()
    else:
        inputs_orig = processor(text=" ", videos=frames, return_tensors="pt").to(model.device)
        inputs_base = processor(text=" ", videos=baseline_frames, return_tensors="pt").to(model.device)
        video_grid_thw = None
        shape = inputs_orig['pixel_values_videos'].shape
        target_H, target_W = shape[-2], shape[-1]
        target_T = shape[2] if len(shape) == 5 else 1

    image = inputs_orig['pixel_values_videos'].to(dtype=model.dtype).detach()
    baseline = inputs_base['pixel_values_videos'].to(dtype=model.dtype).detach()

    # Frame-wise initialization: (target_T, 1, size, size)
    masks_del = Variable(torch.ones((target_T, 1, size, size), dtype=torch.float32, device=model.device), requires_grad=True)
    masks_ins = Variable(torch.ones((target_T, 1, size, size), dtype=torch.float32, device=model.device), requires_grad=True)

    cita_d = torch.zeros_like(masks_del).to(model.device)
    cita_i = torch.zeros_like(masks_ins).to(model.device)

    def regularization_loss(masks, current_L2):
        loss_l1 = L1 * torch.mean(torch.abs(1 - masks).view(masks.shape[0], -1), dim=1)
        loss_tv = L3 * spatio_temporal_bilateral_tv(image, masks)
        loss_l2 = current_L2 * torch.sum((1 - masks)**2, dim=[1, 2, 3])
        return loss_l1, loss_tv, loss_l2

    eprint("\nStarting video_iGOS++ Frame-wise Optimization...")
    for i in range(iterations):        
    # Combined Deletion IG (Pass as a tuple)
        loss_comb_del = integrated_gradient_framewise(args, model, full_ids, output_ids, image, baseline, (masks_del, masks_ins), ig_iter, positions, target_H, target_W, target_T, is_qwen, video_grid_thw)
        total_grads1 = masks_del.grad.clone() if masks_del.grad is not None else torch.zeros_like(masks_del)
        total_grads2 = masks_ins.grad.clone() if masks_ins.grad is not None else torch.zeros_like(masks_ins)
        if masks_del.grad is not None: masks_del.grad.zero_()
        if masks_ins.grad is not None: masks_ins.grad.zero_()

        # Combined Insertion IG (Pass as a tuple)
        loss_comb_ins = integrated_gradient_framewise(args, model, full_ids, output_ids, baseline, image, (masks_del, masks_ins), ig_iter, positions, target_H, target_W, target_T, is_qwen, video_grid_thw)
        total_grads1 -= masks_del.grad.clone()
        total_grads2 -= masks_ins.grad.clone()
        masks_del.grad.zero_()
        masks_ins.grad.zero_()

        # Individual Deletion IG (Pass normally)
        loss_del = integrated_gradient_framewise(args, model, full_ids, output_ids, image, baseline, masks_del, ig_iter, positions, target_H, target_W, target_T, is_qwen, video_grid_thw)
        total_grads1 += masks_del.grad.clone()
        masks_del.grad.zero_()

        # Individual Insertion IG (Pass normally)
        loss_ins = integrated_gradient_framewise(args, model, full_ids, output_ids, baseline, image, masks_ins, ig_iter, positions, target_H, target_W, target_T, is_qwen, video_grid_thw)
        total_grads2 -= masks_ins.grad.clone()
        masks_ins.grad.zero_()

        total_grads1 /= 2
        total_grads2 /= 2

        # Regularization (Calculate it here since regularization uses a single backward pass)
        comb_mask = masks_del * masks_ins
        current_L2 = exp_decay(L2, i, getattr(args, 'gamma', 0.2))
        loss_l1, loss_tv, loss_l2_val = regularization_loss(comb_mask, current_L2)
        losses = loss_l1.sum() + loss_tv + loss_l2_val.sum()
        losses.backward()
        
        total_grads1 += masks_del.grad.clone()
        total_grads2 += masks_ins.grad.clone()

        # Optimizer Step (NAG)
        momentum = getattr(args, 'momentum', 3)
        e = i / (i + momentum)
        cita_d_p, cita_i_p = cita_d, cita_i
        cita_d = masks_del.data - lr * total_grads1
        cita_i = masks_ins.data - lr * total_grads2
        masks_del.data = cita_d + e * (cita_d - cita_d_p)
        masks_ins.data = cita_i + e * (cita_i - cita_i_p)

        masks_del.grad.zero_()
        masks_ins.grad.zero_()
        masks_del.data.clamp_(0, 1)
        masks_ins.data.clamp_(0, 1)

        print(f"iGOS++ Iter: {i} | loss_del: {loss_del:.4f} | loss_ins: {loss_ins:.4f} | TV: {loss_tv.item():.4f}")

    final_mask = masks_del * masks_ins
    return final_mask, target_H, target_W, target_T, video_grid_thw, image, baseline

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


def evaluate_auc_pixel_framewise(args, model, processor, full_ids, output_ids, final_mask, image_tensor, baseline_tensor, positions, target_H, target_W, target_T, is_qwen, video_grid_thw, size=32):
    """
    Evaluates AUC by globally sorting pixels across the entire Spatio-Temporal mask.
    Mathematically aligned with iGOS++ and constrained to [0, 1].
    """
    with torch.no_grad():
        num_pixels = target_T * size * size
        step = max(1, num_pixels // 50) 

        og_probs = get_token_probs_tensor(args, model, full_ids, output_ids, image_tensor, positions, is_qwen, video_grid_thw)
        blur_probs = get_token_probs_tensor(args, model, full_ids, output_ids, baseline_tensor, positions, is_qwen, video_grid_thw)
        
        prob_orig = og_probs.mean().item()
        prob_blur = blur_probs.mean().item()

        del_curve_raw, ins_curve_raw = [prob_orig], [prob_blur]
        fractions = [0.0]

        # true_mask starts as all 1s (determines the blend ratio)
        true_mask = torch.ones((1, num_pixels), device=model.device)

        # Globally sort all pixels in the video (descending = highest gradients first)
        elements = torch.argsort(final_mask.view(1, -1), dim=1, descending=True)

        for pixels in range(0, num_pixels, step):
            indices = elements[:, pixels:pixels+step].squeeze(0)
            true_mask[0, indices] = 0 # Mask out the most important pixels

            # Reshape back to (T, 1, size, size) and upsample
            spatial_mask = true_mask.view(target_T, 1, size, size)
            up_mask = F.interpolate(spatial_mask, size=(target_H, target_W), mode='bilinear', align_corners=False)
            
            if is_qwen:
                final_applied_mask = up_mask.reshape(-1, 1)
            else:
                final_applied_mask = up_mask.permute(1, 0, 2, 3).unsqueeze(0)

            # -- Deletion Score --
            # Mask goes from 1 to 0 -> Image goes from Original to Baseline
            del_image = phi_tensor(image_tensor, baseline_tensor, final_applied_mask).to(dtype=model.dtype)
            del_probs = get_token_probs_tensor(args, model, full_ids, output_ids, del_image, positions, is_qwen, video_grid_thw)
            del_curve_raw.append(del_probs.mean().item())

            # -- Insertion Score --
            # Mask goes from 1 to 0 -> Image goes from Baseline to Original
            ins_image = phi_tensor(baseline_tensor, image_tensor, final_applied_mask).to(dtype=model.dtype)
            ins_probs = get_token_probs_tensor(args, model, full_ids, output_ids, ins_image, positions, is_qwen, video_grid_thw)
            ins_curve_raw.append(ins_probs.mean().item())
            
            current_fraction = min((pixels + step) / num_pixels, 1.0)
            fractions.append(current_fraction)

        # Standard Normalization with strict [0, 1] clamping
        norm_del = np.clip((np.array(del_curve_raw) - prob_blur) / (prob_orig - prob_blur + 1e-7), 0.0, 1.0)
        norm_ins = np.clip((np.array(ins_curve_raw) - prob_blur) / (prob_orig - prob_blur + 1e-7), 0.0, 1.0)

        # Mathematical AUC integration via Trapezoidal rule
        auc_del = np.trapezoid(norm_del, x=fractions)
        auc_ins = np.trapezoid(norm_ins, x=fractions)

    return auc_ins, auc_del, del_curve_raw, ins_curve_raw, fractions, prob_orig, prob_blur

def save_video_heatmaps(final_mask, frames, target_H, target_W, outdir, ivd):
    """
    Upscales the final spatio-temporal mask and saves per-frame heatmaps.
    frames: list of numpy arrays or PIL Images representing the original RGB video.
    """
    os.makedirs(outdir, exist_ok=True)

    # 1. Get the actual dimensions of the original video frames
    orig_H, orig_W = np.array(frames[0]).shape[:2]

    # 2. Upscale the (T, 1, size, size) mask to the ORIGINAL video resolution
    up_mask = F.interpolate(final_mask, size=(orig_H, orig_W), mode='bilinear', align_corners=False)
    up_mask = up_mask.squeeze(1).cpu().detach().numpy() # Shape: (T, orig_H, orig_W)

    # Global Min-Max Normalization across the whole sequence to maintain temporal relativity
    mask_min = up_mask.min()
    mask_max = up_mask.max()
    up_mask = (up_mask - mask_min) / (mask_max - mask_min + 1e-7)

    for t, mask_frame in enumerate(up_mask):
        frame = np.array(frames[t]) # Convert PIL to NumPy if needed
        
        # Generate Jet Colormap
        heatmap = cv2.applyColorMap(np.uint8(255 * mask_frame), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) # Matplotlib expects RGB
        heatmap = np.float32(heatmap) / 255.0
        
        frame_norm = np.float32(frame) / 255.0

        # Overlay formula (50% heat, 50% original)
        overlay = 0.5 * heatmap + 0.5 * frame_norm
        overlay = np.clip(overlay, 0.0, 1.0)

        # Save to disk
        plt.imsave(os.path.join(outdir, f'{ivd}_frame_{t:03d}_heatmap.jpg'), heatmap)
        plt.imsave(os.path.join(outdir, f'{ivd}_frame_{t:03d}_overlay.jpg'), overlay)
    
def run_igos(args, model, processor, full_ids, output_ids, frames, baseline_frames, positions, ivd):
    start = time.time()
    eprint("Running iGOS++ Frame-wise Optimization...")
    
    final_mask, target_H, target_W, target_T, video_grid_thw, img_tensor, base_tensor = video_iGOS_pp(
        args, model, processor, full_ids, output_ids, frames, baseline_frames, positions,
        lr=10, L1=1, L2=1, L3=20, size=32, iterations=15, ig_iter=6
    )
    
    is_qwen = getattr(args, 'model', '') == 'qwen'
    
    auc_ins, auc_del, del_curve, ins_curve, fractions, prob_orig, prob_blur = evaluate_auc_pixel_framewise(
        args, model, processor, full_ids, output_ids, final_mask, img_tensor, base_tensor, 
        positions, target_H, target_W, target_T, is_qwen, video_grid_thw, size=32
    )
    
    plot_and_save_auc(
        percentages=fractions, 
        ins_curve=ins_curve, 
        del_curve=del_curve, 
        auc_ins=auc_ins, 
        auc_del=auc_del, 
        prob_orig=prob_orig, 
        prob_blur=prob_blur, 
        output_dir=args.output_dir, 
        ivd=ivd
    )

    save_video_heatmaps(
        final_mask=final_mask, 
        frames=frames, 
        target_H=target_H, 
        target_W=target_W, 
        outdir=os.path.join(args.output_dir, f"heatmaps_vid_{ivd}"), 
        ivd=ivd
    )

    eprint(f"iGOS++ Time: {(time.time() - start):.2f}s")
    eprint(f"AUC iGOS++ | Ins: {auc_ins:.4f} | Del: {auc_del:.4f}")
    
    experiment_data = {
        "video_index": ivd,
        "AUC_Del": auc_del,
        "AUC_Ins": auc_ins
    }
    
    with open(os.path.join(args.output_dir, "igos_framewise_metrics.jsonl"), "a") as f:
        f.write(json.dumps(experiment_data) + "\n")
        
    return auc_ins, auc_del