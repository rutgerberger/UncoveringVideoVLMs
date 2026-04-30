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

def phi_tensor(img_tensor, baseline_tensor, mask):
    return img_tensor * mask + baseline_tensor * (1.0 - mask)

def spatio_temporal_bilateral_tv(video, mask, tv_beta=2, sigma=0.01, lambda_t=0.5):
    """
    video: (T, C, H, W) or (1, T, C, H, W)
    mask: (T, 1, size, size)
    lambda_t: weight for temporal smoothness
    """
    # Squeeze batch dimension from video if present (e.g., 1, T, C, H, W -> T, C, H, W)
    if video.dim() == 5 and video.shape[0] == 1:
        video = video.squeeze(0)
        
    # Upscale mask to match video resolution for element-wise operations
    up_mask = F.interpolate(mask, size=(video.shape[-2], video.shape[-1]), mode='bilinear', align_corners=False)
    dh_mask = torch.abs(up_mask[:, :, :-1, :] - up_mask[:, :, 1:, :]).pow(tv_beta)
    dw_mask = torch.abs(up_mask[:, :, :, :-1] - up_mask[:, :, :, 1:]).pow(tv_beta)
    
    # Extract edges from the video (for Bilateral weighting)
    dh_img = torch.exp(-(video[:, :, :-1, :] - video[:, :, 1:, :]).mean(dim=1, keepdim=True)**2 / sigma)
    dw_img = torch.exp(-(video[:, :, :, :-1] - video[:, :, :, 1:]).mean(dim=1, keepdim=True)**2 / sigma)
    
    spatial_tv = torch.mean(dh_img * dh_mask) + torch.mean(dw_img * dw_mask)

    #Temporal TV (T dimension) to prevent flickering
    if up_mask.shape[0] > 1: # If T > 1
        dt_mask = torch.abs(up_mask[:-1, :, :, :] - up_mask[1:, :, :, :]).pow(tv_beta)
        dt_img = torch.exp(-(video[:-1, :, :, :] - video[1:, :, :, :]).mean(dim=1, keepdim=True)**2 / sigma)
        temporal_tv = torch.mean(dt_img * dt_mask)
    else:
        temporal_tv = 0.0
        
    return spatial_tv + (lambda_t * temporal_tv)

def integrated_gradient_framewise(args, model, full_ids, output_ids, img_tensor, base_tensor, mask, num_iter, positions, target_H, target_W, target_T, is_qwen=False, video_grid_thw=None):
    intervals = torch.linspace(1/num_iter, 1, num_iter, device=img_tensor.device).view(-1, 1, 1, 1, 1)
    total_loss = 0.0
    
    for alpha in intervals:
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
        inputs_orig = processor(text=[" "], videos=[frames], padding=True, return_tensors="pt", max_pixels=112896).to(model.device)
        inputs_base = processor(text=[" "], videos=[baseline_frames], padding=True, return_tensors="pt", max_pixels=112896).to(model.device)
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

    print("\nStarting video_iGOS++ Frame-wise Optimization...")
    for i in range(iterations):
        comb_mask = masks_del * masks_ins
        
        # Combined Deletion IG
        loss_comb_del = integrated_gradient_framewise(args, model, full_ids, output_ids, image, baseline, comb_mask, ig_iter, positions, target_H, target_W, target_T, is_qwen, video_grid_thw)
        total_grads1 = masks_del.grad.clone() if masks_del.grad is not None else torch.zeros_like(masks_del)
        total_grads2 = masks_ins.grad.clone() if masks_ins.grad is not None else torch.zeros_like(masks_ins)
        if masks_del.grad is not None: masks_del.grad.zero_()
        if masks_ins.grad is not None: masks_ins.grad.zero_()

        # Combined Insertion IG
        loss_comb_ins = integrated_gradient_framewise(args, model, full_ids, output_ids, baseline, image, comb_mask, ig_iter, positions, target_H, target_W, target_T, is_qwen, video_grid_thw)
        total_grads1 -= masks_del.grad.clone()
        total_grads2 -= masks_ins.grad.clone()
        masks_del.grad.zero_()
        masks_ins.grad.zero_()

        # Individual Deletion IG
        loss_del = integrated_gradient_framewise(args, model, full_ids, output_ids, image, baseline, masks_del, ig_iter, positions, target_H, target_W, target_T, is_qwen, video_grid_thw)
        total_grads1 += masks_del.grad.clone()
        masks_del.grad.zero_()

        # Individual Insertion IG
        loss_ins = integrated_gradient_framewise(args, model, full_ids, output_ids, baseline, image, masks_ins, ig_iter, positions, target_H, target_W, target_T, is_qwen, video_grid_thw)
        total_grads2 -= masks_ins.grad.clone()
        masks_ins.grad.zero_()

        total_grads1 /= 2
        total_grads2 /= 2

        # Regularization
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


def evaluate_auc_pixel_framewise(args, model, processor, full_ids, output_ids, final_mask, image_tensor, baseline_tensor, positions, target_H, target_W, target_T, is_qwen, video_grid_thw, size=32):
    """
    Evaluates AUC by globally sorting pixels across the entire Spatio-Temporal mask.
    final_mask: (T, 1, size, size) tensor.
    """
    with torch.no_grad():
        num_pixels = target_T * size * size
        step = max(1, num_pixels // 50) 

        og_probs = get_token_probs_tensor(args, model, full_ids, output_ids, image_tensor, positions, is_qwen, video_grid_thw)
        blur_probs = get_token_probs_tensor(args, model, full_ids, output_ids, baseline_tensor, positions, is_qwen, video_grid_thw)
        og_scores = og_probs.mean().item()
        blur_scores = blur_probs.mean().item()

        del_curve, ins_curve = [og_scores], [blur_scores]
        index_curve = [0.0]

        true_mask = torch.ones((1, num_pixels), device=model.device)
        del_auc_score, ins_auc_score = 0.0, 0.0

        # Globally sort all pixels in the video
        elements = torch.argsort(final_mask.view(1, -1), dim=1, descending=True)

        for pixels in range(0, num_pixels, step):
            indices = elements[:, pixels:pixels+step].squeeze(0)
            true_mask[0, indices] = 0 

            # Reshape back to (T, 1, size, size)
            spatial_mask = true_mask.view(target_T, 1, size, size)
            up_mask = F.interpolate(spatial_mask, size=(target_H, target_W), mode='bilinear', align_corners=False)
            
            if is_qwen:
                final_applied_mask = up_mask.reshape(-1, 1)
            else:
                final_applied_mask = up_mask.permute(1, 0, 2, 3).unsqueeze(0)

            # Deletion Score
            del_image = phi_tensor(image_tensor, baseline_tensor, final_applied_mask).to(dtype=model.dtype)
            del_probs = get_token_probs_tensor(args, model, full_ids, output_ids, del_image, positions, is_qwen, video_grid_thw)
            d_score = del_probs.mean().item()
            del_curve.append(d_score)
            norm_d = (d_score - blur_scores) / (og_scores - blur_scores + 1e-7)
            del_auc_score += norm_d * step if pixels + step < num_pixels else num_pixels - pixels

            # Insertion Score
            ins_image = phi_tensor(baseline_tensor, image_tensor, final_applied_mask).to(dtype=model.dtype)
            ins_probs = get_token_probs_tensor(args, model, full_ids, output_ids, ins_image, positions, is_qwen, video_grid_thw)
            i_score = ins_probs.mean().item()
            ins_curve.append(i_score)
            norm_i = (i_score - blur_scores) / (og_scores - blur_scores + 1e-7)
            ins_auc_score += norm_i * step if pixels + step < num_pixels else num_pixels - pixels
            
            index_curve.append((pixels + step) / num_pixels)

        del_auc_score /= num_pixels
        ins_auc_score /= num_pixels

    return ins_auc_score, del_auc_score, del_curve, ins_curve, index_curve

def run_igos(args, model, processor, full_ids, output_ids, frames, baseline_frames, positions, ivd):
    start = time.time()
    print("Running iGOS++ Frame-wise Optimization...")
    
    final_mask, target_H, target_W, target_T, video_grid_thw, img_tensor, base_tensor = video_iGOS_pp(
        args, model, processor, full_ids, output_ids, frames, baseline_frames, positions,
        lr=10, L1=1, L2=1, L3=20, size=32, iterations=15, ig_iter=6
    )
    
    is_qwen = getattr(args, 'model', '') == 'qwen'
    auc_ins, auc_del, del_curve, ins_curve, idx_curve = evaluate_auc_pixel_framewise(
        args, model, processor, full_ids, output_ids, final_mask, img_tensor, base_tensor, 
        positions, target_H, target_W, target_T, is_qwen, video_grid_thw, size=32
    )
    
    print(f"iGOS++ Time: {(time.time() - start):.2f}s")
    print(f"AUC iGOS++ | Ins: {auc_ins:.4f} | Del: {auc_del:.4f}")
    
    experiment_data = {
        "video_index": ivd,
        "AUC_Del": auc_del,
        "AUC_Ins": auc_ins
    }
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "igos_framewise_metrics.jsonl"), "a") as f:
        f.write(json.dumps(experiment_data) + "\n")
        
    return auc_ins, auc_del