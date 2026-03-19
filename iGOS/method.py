import math
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from .method_helpers import integrated_gradient_video, tv_norm_video

def exp_decay(init, iter, gamma=0.2):
    """Exact exponential decay function from original helpers."""
    return init * math.exp(-gamma * iter)

def iGOS_p(
        args,
        model,
        processor,
        full_ids,
        output_ids,
        frames,
        baseline_frames,
        positions,
        init_mask=None,
        size=28,
        iterations=15,
        ig_iter=20,
        L1=1,
        L2=1,
        L3=20,
        lr=1000,
        opt='NAG',
        **kwargs):

    # 1. Prepare Differentiable Tensors
    inputs_orig = processor(text=" ", videos=frames, return_tensors="pt")
    inputs_base = processor(text=" ", videos=baseline_frames, return_tensors="pt")
    
    image = inputs_orig['pixel_values_videos'].to(model.device, dtype=model.dtype).detach()
    baseline = inputs_base['pixel_values_videos'].to(model.device, dtype=model.dtype).detach()
    
    target_H, target_W = image.shape[-2], image.shape[-1]

    # Exact Regularization Match
    def regularization_loss(masks, current_L2):
        loss_l1 = L1 * torch.mean(torch.abs(1-masks).view(masks.shape[0],-1), dim=1)
        loss_tv = L3 * tv_norm_video(masks)
        loss_l2 = current_L2 * torch.sum((1 - masks)**2, dim=[1, 2, 3])
        return loss_l1, loss_tv, loss_l2

    # 2. Setup Masks exactly like original
    masks = torch.ones((1,1,size,size), dtype=torch.float32, device=model.device)
    if init_mask is not None:
        masks = masks * init_mask.to(model.device)
    masks = Variable(masks, requires_grad=True)

    if opt == 'NAG':
        cita = torch.zeros(1).to(model.device)

    losses_del, losses_ins, losses_l1, losses_tv, losses_l2 = [], [], [], [], []
    
    # 3. Optimization Loop
    for i in range(iterations):
        total_grads = torch.zeros_like(masks).to(model.device)

        # Deletion integrated gradient
        loss_del = integrated_gradient_video(args, model, full_ids, output_ids, image, baseline, masks, ig_iter, positions, target_H, target_W)
        total_grads += masks.grad.clone()
        masks.grad.zero_()

        # Insertion integrated gradient
        loss_ins = integrated_gradient_video(args, model, full_ids, output_ids, baseline, image, masks, ig_iter, positions, target_H, target_W)
        total_grads += -masks.grad.clone()
        masks.grad.zero_()

        # Exponential decay for L2 matching original
        gamma = getattr(args, 'gamma', 0.2)
        current_L2 = exp_decay(L2, i, gamma)
        
        # Regularization matching original
        loss_l1_val, loss_tv_val, loss_l2_val = regularization_loss(masks, current_L2)
        losses = loss_l1_val + loss_tv_val + loss_l2_val
        losses.sum().backward()
        
        total_grads += masks.grad.clone()
        masks.grad.zero_()

        # Optimizer exactly as original
        if opt == 'NAG':
            momentum = getattr(args, 'momentum', 3)
            e = i / (i + momentum) 
            cita_p = cita
            cita = masks.data - lr * total_grads
            masks.data = cita + e * (cita - cita_p)

        losses_del.append(loss_del)
        losses_ins.append(loss_ins)
        losses_l1.append(loss_l1_val.item())
        losses_tv.append(loss_tv_val.item())
        losses_l2.append(loss_l2_val.item())

        print(f'iteration: {i} lr: {lr:.4f} loss_del: {loss_del:.4f}, loss_ins: {loss_ins:.4f}, loss_l1: {loss_l1_val.item():.4f}, loss_tv: {loss_tv_val.item():.4f}, loss_l2: {loss_l2_val.item():.4f}')
        
        masks.grad.zero_()
        masks.data.clamp_(0, 1)

    return masks, losses_del, losses_ins, losses_l1, losses_tv, losses_l2, None, None