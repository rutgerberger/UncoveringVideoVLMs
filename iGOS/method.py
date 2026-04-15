import math
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from .method_helpers import integrated_gradient_video, tv_norm_video, bilateral_tv_norm_video

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
        size=32,
        iterations=5,
        ig_iter=6,
        L1=1,
        L2=1,
        L3=20,
        lr=10,
        opt='NAG',
        **kwargs):

    is_qwen = getattr(args, 'model', '') == 'qwen'

    # 1. Prepare Differentiable Tensors
    if is_qwen:
        # Qwen requires explicit max_pixels and padding to form the grid
        inputs_orig = processor(text=[" "], videos=[frames], padding=True, return_tensors="pt", max_pixels=112896).to(model.device)
        inputs_base = processor(text=[" "], videos=[baseline_frames], padding=True, return_tensors="pt", max_pixels=112896).to(model.device)
        video_grid_thw = inputs_orig['video_grid_thw']
        
        # Extract dimensions from Qwen's grid
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

    # Exact Regularization Match
    def regularization_loss(masks, current_L2):
        loss_l1 = L1 * torch.mean(torch.abs(1-masks).view(masks.shape[0],-1), dim=1)
        
        # If Qwen, 'image' is flat, so bilateral TV will crash. Use standard mask TV instead.
        if is_qwen:
            loss_tv = L3 * tv_norm_video(masks)
        else:
            loss_tv = L3 * bilateral_tv_norm_video(image, masks)
            
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

        # Deletion integrated gradient (Pass down target_T, is_qwen, and video_grid_thw)
        loss_del = integrated_gradient_video(args, model, full_ids, output_ids, image, baseline, masks, ig_iter, positions, target_H, target_W, target_T, is_qwen, video_grid_thw)
        total_grads += masks.grad.clone()
        masks.grad.zero_()

        # Insertion integrated gradient
        loss_ins = integrated_gradient_video(args, model, full_ids, output_ids, baseline, image, masks, ig_iter, positions, target_H, target_W, target_T, is_qwen, video_grid_thw)
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

def perform_igos():
    
        eprint("Running iGOS Continuous Pixel Optimization...")
        baseline_ins = get_baseline_insertion(args, video_array)
        frames_ins_base = [Image.fromarray(f.astype(np.uint8)) for f in baseline_ins]
        
        # Unpack the exact tuple return
        raw_mask, l_del, l_ins, l_l1, l_tv, l_l2, _, _ = iGOS_p(
            args, model, processor, full_ids, output_ids, frames, frames_ins_base, positions,
            lr=10, L1=0.5, L2=1, L3=20, size=32 # Explicitly using original hyperparameters
        )
        
        is_qwen = getattr(args, 'model', '') == 'qwen'
        packed_inputs = _get_rescale_and_dummys(model, processor, frames, frames_ins_base, is_qwen, tubelets)
        (_, _, _, target_T, target_H, target_W, t_dim_index, crop_top, crop_left, new_H, new_W, T_orig, H_orig, W_orig) = packed_inputs
        # Interpolate 28x28 mask to the VLM's tensor crop size (e.g., 224x224)
        tensor_mask = torch.nn.functional.interpolate(raw_mask.data, size=(target_H, target_W), mode='bilinear', align_corners=False)
        # Create an empty canvas representing the resized image before center cropping
        canvas = torch.zeros((1, 1, new_H, new_W), device=tensor_mask.device)
        # Paste the mask into the exact crop location
        canvas[:, :, crop_top:crop_top+target_H, crop_left:crop_left+target_W] = tensor_mask
        # Resize the canvas back down to the original raw video dimensions
        final_mask = torch.nn.functional.interpolate(canvas, size=(H_orig, W_orig), mode='bilinear', align_corners=False)
        # Squeeze to 2D shape (H, W) for the evaluator and visualizer
        igos_mask_numpy = final_mask.squeeze().cpu().numpy()
        
        if getattr(args, 'save_visuals', True):
            eprint(f"{ivd+1}/{args.num_videos}: Saving iGOS Heatmaps...")
            save_igos_heatmaps(
                video_array, 
                igos_mask_numpy, 
                args.output_dir, 
                prefix=f"{ivd}_{file_prefix}igos"
            )

        # Evaluate using the pixel evaluator
        auc_ins, auc_del = evaluate_auc_pixel(
            args, model, processor, full_ids, output_ids, frames, igos_mask_numpy, 
            ivd=ivd, positions=positions)   
        
        log_func(f"iGOS Time: {(time.time() - start):.2f}s")
        log_func(f"AUC iGOS - Ins: {auc_ins:.4f} | Del: {auc_del:.4f}")
        
        experiment_data = {
            "video_index": ivd,
            "num_frames": args.num_frames,
            "AUC Union - Del": auc_del,
            "AUC Union - Ins": auc_ins
        }
        os.makedirs(args.output_dir, exist_ok=True)
        metrics_file = os.path.join(args.output_dir, "frame_experiment_metrics.jsonl")

        with open(metrics_file, "a") as f:
            f.write(json.dumps(experiment_data) + "\n")
        
        return auc_ins, auc_del