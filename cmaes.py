import cma
import torch
import numpy as np
import torch.nn.functional as F
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
        
        if torch.isnan(target_probs).any():
            mean_prob = 1.0 if mode == 'deletion' else 0.0
        else:
            mean_prob = target_probs.mean().item()

    tv_penalty_raw = tv_norm_3d(M_vol)
    tv_penalty = tv_penalty_raw.item() if isinstance(tv_penalty_raw, torch.Tensor) else float(tv_penalty_raw)
    
    if mode == 'deletion': # minimize probability 
        L1_penalty = torch.mean(1.0 - M_large).item() # Penalize dropping 
        fitness = mean_prob + args.reg_lambda * (L1_penalty + tv_penalty)
    else: # maximize probability 
        L1_penalty = torch.mean(M_large).item() # Penalize revealing 
        fitness = -mean_prob + args.reg_lambda * (L1_penalty + tv_penalty)

    return float(fitness), mean_prob


def CMA_ES(
    args, model, processor, tokenizer, full_ids, output_ids, 
    frames_orig, frames_base, tubelets, positions=None, mode='deletion'
    ):
    """
    Input:
        model
        processor
        tokenizer
        full_ids
        output_ids
        frames_orig
        frames_base
        tubelets
        positions
    Output:
        selected_tubelets: set of selected tubelets
        scores: corresponding saliency scores
        metrics: for logging purposes (e.g. variance )
    """
    num_tubes = int(tubelets.max()) + 1
    tubelets_tensor = torch.tensor(tubelets, device=model.device, dtype=torch.long)
    is_qwen = getattr(args, 'model', '') == 'qwen' # Qwen requires different handling of input
    
    # Pre-calculate inputs
    packed_inputs = _get_rescale_and_dummys(
        model, processor, frames_orig, frames_base, is_qwen, tubelets
    )
    (dummy_inputs_orig, video, baseline, target_T, target_H, target_W, 
     t_dim_index, crop_top, crop_left, new_H, new_W, T_orig, H_orig, W_orig) = packed_inputs

    # Initialize algorithm
    W_init = 2.0 if mode == 'deletion' else -2.0
    mean_init = np.full(num_tubes, W_init)
    sigma_0 = 1.0

    es = cma.CMAEvolutionStrategy(mean_init, sigma_0, {
        'maxiter': getattr(args, 'iterations', 150), 
        'popsize': getattr(args, 'popsize', 20),
        'bounds': [-5.0, 5.0]
    })

    generation = 0

    # Main optimization loop
    while not es.stop():
        candidates = es.ask()
        candidates_tensor = torch.tensor(np.stack(candidates), dtype=torch.float32, device=model.device) #(popsize, num_tubes)
        fitnesses = []
        confidences = []
        for i in range(len(candidates)):
            W = candidates_tensor[i]
            M_large = torch.sigmoid(W)[tubelets_tensor].unsqueeze(1).float()
            
            M_scaled, M_vol = rescale_mask(M_large, new_H, new_W, crop_top, crop_left, target_H, target_W, target_T, T_orig, is_qwen, t_dim_index)
            
            fitness, confidence = evaluate_fitness(args, mode, M_scaled, M_vol, M_large, video, baseline, model, positions, full_ids, output_ids, dummy_inputs_orig)
            fitnesses.append(fitness)
            confidences.append(confidence)

        es.tell(candidates, fitnesses)
        best_idx = np.argmin(fitnesses)
        es.disp()
        eprint(f"Gen {generation} | Best Fit: {fitnesses[best_idx]:.4f} | Conf: {confidences[best_idx]:.4f}")
        generation += 1
    
    # Fallback to mean_init if no valid evaluations occurred to prevent NoneType TypeError
    W_best = es.result.xbest if es.result.xbest is not None else mean_init
    W_final = sigmoid(W_best)

    #Log the change of the weights
    if mode == 'deletion':
        scores = {t: float(1.0 - W_final[t]) for t in range(num_tubes)}
    else:
        scores = {t: float(W_final[t]) for t in range(num_tubes)}

    # Metrics & Return
    max_score = max(scores.values())
    threshold = max_score * 0.10 if max_score > 0.01 else 0.0
    selected_tubelets = [t for t, s in scores.items() if s >= threshold]
    
    metrics = {
        "cma_evals": es.result.evaluations,
        "best_fitness": es.result.fbest
    }

    return selected_tubelets, scores, metrics

def SimulatedAnnealing_optimization(
    args, model, processor, tokenizer, full_ids, output_ids, 
    frames_orig, frames_base, tubelets, positions=None, mode='deletion'
):
    """
    Optimizes a binary mask using Simulated Annealing, tuned for a strict 
    budget of 250 VLM evaluations.
    """
    num_tubes = int(tubelets.max()) + 1
    tubelets_tensor = torch.tensor(tubelets, device=model.device, dtype=torch.long)
    is_qwen = getattr(args, 'model', '') == 'qwen'
    
    # Pre-calculate inputs and HF geometry scaling
    packed_inputs = _get_rescale_and_dummys(
        model, processor, frames_orig, frames_base, is_qwen, tubelets
    )
    (dummy_inputs_orig, video, baseline, target_T, target_H, target_W, 
     t_dim_index, crop_top, crop_left, new_H, new_W, T_orig, H_orig, W_orig) = packed_inputs

    # --- SA Hyperparameters (Tuned for 250 budget) ---
    iterations = getattr(args, 'iterations', 250)
    initial_temp = 1.0
    final_temp = 0.01
    cooling_rate = (final_temp / initial_temp) ** (1.0 / iterations)
    
    # --- Smart Initialization ---
    # For deletion: start with 90% of the video intact (mostly 1s).
    # For insertion: start with 90% of the video masked (mostly 0s).
    if mode == 'deletion':
        current_mask = (torch.rand(num_tubes, device=model.device) > 0.1).float()
    else:
        current_mask = (torch.rand(num_tubes, device=model.device) > 0.9).float()
    
    # Evaluate Initial State
    M_large = current_mask[tubelets_tensor].unsqueeze(1).float()
    M_scaled, M_vol = rescale_mask(M_large, new_H, new_W, crop_top, crop_left, 
                                   target_H, target_W, target_T, T_orig, is_qwen, t_dim_index)
    
    current_fitness, current_conf = evaluate_fitness(
        args, mode, M_scaled, M_vol, M_large, video, baseline, 
        model, positions, full_ids, output_ids, dummy_inputs_orig
    )
    
    best_mask = current_mask.clone()
    best_fitness = current_fitness
    current_temp = initial_temp

    # --- Target Thresholds for Early Stopping ---
    # Stop early if target is sufficiently destroyed (deletion) or recovered (insertion)
    target_confidence_del = 0.05 
    target_confidence_ins = 0.90 

    eprint(f"Starting SA | Mode: {mode} | Initial Temp: {initial_temp} | Iterations: {iterations}")

    for generation in range(iterations):
        # 1. Dynamic Neighborhood: Flip more bits early on, fewer bits later
        progress = generation / iterations
        max_flips = 4 if progress < 0.3 else (2 if progress < 0.7 else 1)
        num_flips = torch.randint(1, max_flips + 1, (1,)).item()
        
        neighbor_mask = current_mask.clone()
        flip_indices = torch.randperm(num_tubes)[:num_flips]
        neighbor_mask[flip_indices] = 1.0 - neighbor_mask[flip_indices] # Flip 0 to 1, or 1 to 0
        
        # 2. Evaluate Neighbor
        M_large_neigh = neighbor_mask[tubelets_tensor].unsqueeze(1).float()
        M_scaled_neigh, M_vol_neigh = rescale_mask(M_large_neigh, new_H, new_W, crop_top, crop_left, 
                                                   target_H, target_W, target_T, T_orig, is_qwen, t_dim_index)
        
        neighbor_fitness, neighbor_conf = evaluate_fitness(
            args, mode, M_scaled_neigh, M_vol_neigh, M_large_neigh, video, baseline, 
            model, positions, full_ids, output_ids, dummy_inputs_orig
        )
        
        # 3. Early Stopping Check
        if mode == 'deletion' and neighbor_conf < target_confidence_del:
            eprint(f"Target confidence destroyed (<{target_confidence_del}) at pass {generation}. Stopping early.")
            best_mask = neighbor_mask
            best_fitness = neighbor_fitness
            break
        elif mode == 'insertion' and neighbor_conf > target_confidence_ins:
            eprint(f"Target confidence recovered (>{target_confidence_ins}) at pass {generation}. Stopping early.")
            best_mask = neighbor_mask
            best_fitness = neighbor_fitness
            break

        # 4. Acceptance Logic (Metropolis Criterion)
        delta = neighbor_fitness - current_fitness
        
        if delta < 0:
            # Better solution: accept always
            accept = True
        else:
            # Worse solution: accept with probability exp(-delta / Temp)
            acceptance_probability = torch.exp(-torch.tensor(delta) / current_temp).item()
            accept = np.random.rand() < acceptance_probability
            
        if accept:
            current_mask = neighbor_mask
            current_fitness = neighbor_fitness
            
            # Update global best
            if current_fitness < best_fitness:
                best_fitness = current_fitness
                best_mask = current_mask.clone()
                eprint(f"Gen {generation} | NEW BEST | Fit: {best_fitness:.4f} | Conf: {neighbor_conf:.4f} | Temp: {current_temp:.4f}")

        # 5. Cool down
        current_temp *= cooling_rate

    # --- Finalization ---
    W_final = best_mask.cpu().numpy()
    
    # Calculate scores based on the mode.
    # For deletion: 1.0 means we kept it, 0.0 means we deleted it. We score what was deleted.
    # For insertion: 1.0 means we inserted it, 0.0 means we left it blank. We score what was inserted.
    if mode == 'deletion':
        scores = {t: float(1.0 - W_final[t]) for t in range(num_tubes)}
    else:
        scores = {t: float(W_final[t]) for t in range(num_tubes)}

    # Since it's a binary mask, everything scored 1.0 was selected.
    selected_tubelets = [t for t, s in scores.items() if s > 0.5]
    
    metrics = {
        "sa_evals": generation + 1,  # Records actual evals if stopped early
        "best_fitness": best_fitness
    }

    return selected_tubelets, scores, metrics