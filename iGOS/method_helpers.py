import torch
import torch.nn.functional as F

def get_token_probs_tensor(args, model, full_ids, output_ids, pixel_values, positions=None):
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
    # mask is expected to be [1, 1, 1, H, W], video tensors are [1, 3, T, H, W]
    return img_tensor * mask + baseline_tensor * (1 - mask)

def tv_norm_video(mask, beta=2):
    """Simple Spatial Total Variation norm for the mask."""
    # mask shape: (1, 1, H, W)
    a = torch.mean(torch.abs(mask[:, :, :-1, :] - mask[:, :, 1:, :]).pow(beta))
    b = torch.mean(torch.abs(mask[:, :, :, :-1] - mask[:, :, :, 1:]).pow(beta))
    return a + b

def integrated_gradient_video(args, model, full_ids, output_ids, img_tensor, base_tensor, mask, num_iter, positions, target_H, target_W):
    """
    Calculates the integrated gradient while heavily managing VRAM.
    By doing step-loss.backward() inside the loop, we prevent PyTorch 
    from trying to hold multiple LLM computational graphs in memory at once.
    """
    intervals = torch.linspace(1/num_iter, 1, num_iter, device=img_tensor.device).view(-1, 1, 1, 1, 1)
    total_loss = 0.0
    
    for alpha in intervals:
        # 1. Upscale the mask inside the loop so we don't need retain_graph=True
        up_mask = F.interpolate(mask, size=(target_H, target_W), mode='bilinear', align_corners=False)
        up_mask_5d = up_mask.unsqueeze(2) # Shape -> (1, 1, 1, H, W)
        
        interval_mask = up_mask_5d * alpha
        blended = phi_tensor(img_tensor, base_tensor, interval_mask)
        
        # 2. Forward Pass
        probs = get_token_probs_tensor(args, model, full_ids, output_ids, blended, positions)
        step_loss = torch.log(probs + 1e-7).sum() / num_iter
        
        # 3. Backward Pass immediately flushes the massive LLM computation graph!
        # Gradients naturally accumulate in mask.grad
        step_loss.backward()
        
        total_loss += step_loss.item()
        
    return total_loss