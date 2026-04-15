import torch
import torch.nn.functional as F

def get_token_probs_tensor(args, model, full_ids, output_ids, pixel_values, positions=None, is_qwen=False, video_grid_thw=None):
    """
    A differentiable version of your get_token_probs. 
    Accepts pre-computed pixel_values so gradients can flow back to the mask.
    """
    forward_kwargs = {
        "input_ids": full_ids,
        "attention_mask": torch.ones_like(full_ids).to(model.device),
        "pixel_values_videos": pixel_values,
        "use_cache": True
    }
    
    # Inject Qwen's positional grid if applicable
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
    """Composes image and baseline via a continuous mask."""
    # mask handles standard broadcasting OR flattened Qwen patch broadcasting
    return img_tensor * mask + baseline_tensor * (1 - mask)


def tv_norm_video(mask, beta=2):
    """Simple Spatial Total Variation norm for the mask."""
    # mask shape: (1, 1, H, W)
    a = torch.mean(torch.abs(mask[:, :, :-1, :] - mask[:, :, 1:, :]).pow(beta))
    b = torch.mean(torch.abs(mask[:, :, :, :-1] - mask[:, :, :, 1:]).pow(beta))
    return a + b

def bilateral_tv_norm_video(image, mask, beta=2, sigma=0.01):
    """Bilateral TV: mask shape (1, 1, H, W), image shape (1, 3, T, H, W)"""
    # Get target spatial dimensions from the image
    target_H, target_W = image.shape[-2], image.shape[-1]
    
    # Upscale the mask to match the image resolution
    up_mask = F.interpolate(mask, size=(target_H, target_W), mode='bilinear', align_corners=False)
    
    # Since the broadcast handles the temporal dimension T and channel dimension automatically,
    # we just need to calculate the spatial differences.
    
    # Mask spatial differences
    mask_h = torch.abs(up_mask[..., :-1, :] - up_mask[..., 1:, :]).pow(beta)
    mask_w = torch.abs(up_mask[..., :, :-1] - up_mask[..., :, 1:]).pow(beta)
    
    # Image spatial differences (mean across color channels to get intensity diff)
    img_h = (image[..., :-1, :] - image[..., 1:, :]).mean(dim=1, keepdim=True).pow(2)
    img_w = (image[..., :, :-1] - image[..., :, 1:]).mean(dim=1, keepdim=True).pow(2)
    
    # Average across the temporal dimension if your image has one (T)
    if img_h.dim() == 5:
        img_h = img_h.mean(dim=2, keepdim=True)
        img_w = img_w.mean(dim=2, keepdim=True)
    
    # Bilateral weighting: smooth image areas (img_h ~ 0) strongly penalize mask differences
    bil_h = torch.mean(torch.exp(-img_h / sigma) * mask_h)
    bil_w = torch.mean(torch.exp(-img_w / sigma) * mask_w)
    
    return bil_h + bil_w

def integrated_gradient_video(args, model, full_ids, output_ids, img_tensor, base_tensor, mask, num_iter, positions, target_H, target_W, target_T=1, is_qwen=False, video_grid_thw=None):
    """
    Calculates the integrated gradient while heavily managing VRAM.
    """
    intervals = torch.linspace(1/num_iter, 1, num_iter, device=img_tensor.device).view(-1, 1, 1, 1, 1)
    total_loss = 0.0
    
    for alpha in intervals:
        # 1. Upscale the mask inside the loop so we don't need retain_graph=True
        up_mask = F.interpolate(mask, size=(target_H, target_W), mode='bilinear', align_corners=False)
        
        if is_qwen:
            # Qwen patches are flattened, so we expand the mask temporally and flatten it
            up_mask_5d = up_mask.unsqueeze(2).expand(-1, -1, target_T, -1, -1)
            interval_mask = up_mask_5d.reshape(-1, 1) * alpha.view(1)
        else:
            # Standard 5D tensor broadcast (1, 1, 1, H, W)
            up_mask_5d = up_mask.unsqueeze(2) 
            interval_mask = up_mask_5d * alpha
            
        blended = phi_tensor(img_tensor, base_tensor, interval_mask)

        noise_tensor = torch.randn_like(blended) * 0.2
        blended = blended + noise_tensor
        
        # 2. Forward Pass (now passing is_qwen and video_grid_thw)
        probs = get_token_probs_tensor(args, model, full_ids, output_ids, blended, positions, is_qwen=is_qwen, video_grid_thw=video_grid_thw)
        step_loss = torch.log(probs + 1e-7).sum() / num_iter
        
        # 3. Backward Pass
        step_loss.backward()
        
        total_loss += step_loss.item()
        
    return total_loss
