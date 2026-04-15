from sklearn.ensemble import RandomForestRegressor
import cma
import torch
import numpy as np
import torch.nn.functional as F
# Assuming utils is imported for eprint
from utils import *


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


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


def _get_rescale_and_dummys(model, processor, frames, baseline_frames, is_qwen, tubelets):
    """
    Given a huggingface VLM, its corresponding processor,
    a list of frames, and a list of baseline frames, this
    function returns dummy inputs and scaling geometries.
    """
    with torch.no_grad():
        if is_qwen:
            dummy_inputs_orig = processor(text=[" "], videos=[frames], padding=True, return_tensors="pt", max_pixels=112896).to(model.device)
            dummy_inputs_base = processor(text=[" "], videos=[baseline_frames], padding=True, return_tensors="pt", max_pixels=112896).to(model.device)
            pixels_orig = dummy_inputs_orig['pixel_values_videos'].detach()
            pixels_base = dummy_inputs_base['pixel_values_videos'].detach()
            
            grid_thw = dummy_inputs_orig['video_grid_thw'][0] 
            target_T, target_H, target_W = grid_thw[0].item(), grid_thw[1].item(), grid_thw[2].item()
            target_C = pixels_orig.shape[1] 
            t_dim_index = -1 
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


def evaluate_fitness(
    args, mode, M, M_vol, M_large, video, baseline, model, positions, full_ids, output_ids, dummy_inputs_orig
    ):
    
    if mode == 'deletion': # M = what we want to keep
        video_input = video * M + baseline * (1.0 - M)
    else: # M = what we want to reveal
        video_input = baseline * (1.0 - M) + video * M
    
    with torch.no_grad():
        forward_kwargs = {
            "input_ids": full_ids,
            "attention_mask": torch.ones_like(full_ids).to(model.device),
            "pixel_values_videos": video_input.to(dtype=model.dtype),
            "use_cache": False
        }
        if getattr(args, 'model', '') == 'qwen':
            forward_kwargs["video_grid_thw"] = dummy_inputs_orig["video_grid_thw"]
            
        outputs = model(**forward_kwargs)
        logits = outputs.logits
        out_len = output_ids.shape[-1]
        target_logits = logits[:, -out_len - 1 : -1, :]
        probs = torch.nn.functional.softmax(target_logits, dim=-1)
        target_probs = probs.gather(dim=-1, index=output_ids.unsqueeze(-1)).squeeze(-1)
        
        if positions is not None and len(positions) > 0:
            target_probs = target_probs[0, positions]
        
        # NaN protection guard
        if torch.isnan(target_probs).any():
            mean_prob = 1.0 if mode == 'deletion' else 0.0
        else:
            mean_prob = target_probs.mean().item()

    # The float(...) type-cast fix
    tv_penalty_raw = tv_norm_3d(M_vol)
    tv_penalty = tv_penalty_raw.item() if isinstance(tv_penalty_raw, torch.Tensor) else float(tv_penalty_raw)
    
    # Apply unified args.reg_lambda logic
    if mode == 'deletion': 
        L1_penalty = torch.mean(1.0 - M_large).item() 
        fitness = mean_prob + args.reg_lambda * (L1_penalty + tv_penalty)
    else: 
        L1_penalty = torch.mean(M_large).item()
        fitness = -mean_prob + args.reg_lambda * (L1_penalty + tv_penalty)

    return float(fitness), mean_prob


def SAEA_CMA_ES(
    args, model, processor, tokenizer, full_ids, output_ids, 
    frames_orig, frames_base, tubelets, positions=None, mode='deletion'
    ):
    """
    Surrogate-Assisted Evolutionary Algorithm with CMA-ES.
    """
    num_tubes = int(tubelets.max()) + 1
    tubelets_tensor = torch.tensor(tubelets, device=model.device, dtype=torch.long)
    is_qwen = getattr(args, 'model', '') == 'qwen'
    
    # Pre-calculate inputs
    packed_inputs = _get_rescale_and_dummys(
        model, processor, frames_orig, frames_base, is_qwen, tubelets
    )
    (dummy_inputs_orig, video, baseline, target_T, target_H, target_W, 
     t_dim_index, crop_top, crop_left, new_H, new_W, T_orig, H_orig, W_orig) = packed_inputs

    # Initialize CMA-ES
    W_init = 2.0 if mode == 'deletion' else -2.0
    mean_init = np.full(num_tubes, W_init)
    sigma_0 = 1.0

    es = cma.CMAEvolutionStrategy(mean_init, sigma_0, {
        'maxiter': getattr(args, 'iterations', 150), 
        'popsize': getattr(args, 'popsize', 20),
        'bounds': [-5.0, 5.0]
    })

    # Initialize Surrogate Model
    surrogate = RandomForestRegressor(n_estimators=50, random_state=42)
    X_history = []
    y_history = []
    
    # Frequency of calling the true VLM to update the surrogate
    surrogate_update_freq = getattr(args, 'surrogate_freq', 5) 
    vlm_calls = 0
    generation = 0

    eprint(f"\n--- Starting SAEA CMA-ES Optimization ({mode}) ---")

    # Main optimization loop
    while not es.stop():
        candidates = es.ask()
        fitnesses = []
        
        #-- Ground Truth Evaluation (Real VLM)
        if generation % surrogate_update_freq == 0 or len(X_history) < es.popsize:
            candidates_tensor = torch.tensor(np.stack(candidates), dtype=torch.float32, device=model.device)
            
            for i in range(len(candidates)):
                W = candidates_tensor[i]
                M_large = torch.sigmoid(W)[tubelets_tensor].unsqueeze(1).float()
                
                # Unpack both the VLM mask and Volumetric mask
                M_scaled, M_vol = rescale_mask(M_large, new_H, new_W, crop_top, crop_left, target_H, target_W, target_T, T_orig, is_qwen, t_dim_index)
                
                # Retrieve fitness and confidence
                fitness, confidence = evaluate_fitness(args, mode, M_scaled, M_vol, M_large, video, baseline, model, positions, full_ids, output_ids, dummy_inputs_orig)
                
                fitnesses.append(fitness)
                vlm_calls += 1
            
            # Update surrogate archive and retrain
            X_history.extend(candidates)
            y_history.extend(fitnesses)
            surrogate.fit(np.array(X_history), np.array(y_history))
            eval_type = "TRUE VLM"
            
        #-- Surrogate Evaluation (Random Forest)
        else:
            fitnesses = surrogate.predict(np.array(candidates)).tolist()
            eval_type = "SURROGATE"

        es.tell(candidates, fitnesses)
        es.disp()
        
        best_fitness = np.min(fitnesses)
        eprint(f"Gen {generation} | {eval_type} | Best Fitness: {best_fitness:.4f}")
        generation += 1
    
    # Fallback to mean_init to prevent NoneType errors if optimizations halted early
    W_best = es.result.xbest if es.result.xbest is not None else mean_init
    W_final = sigmoid(W_best)

    if mode == 'deletion':
        scores = {t: float(1 - W_final[t]) for t in range(num_tubes)}
    else:
        scores = {t: float(W_final[t]) for t in range(num_tubes)}

    # Metrics & Return
    max_score = max(scores.values())
    threshold = max_score * 0.20 if max_score > 0.01 else 0.0
    selected_tubelets = [t for t, s in scores.items() if s >= threshold]
    metrics = {
        "cma_evals": es.result.evaluations,
        "vlm_calls": vlm_calls,
        "best_fitness": es.result.fbest
    }
    
    del video, baseline, dummy_inputs_orig
    torch.cuda.empty_cache()

    return selected_tubelets, scores, metrics