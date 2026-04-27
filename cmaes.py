import cma
import torch
import numpy as np
import torch.nn.functional as F

from utils import *
from method_helpers import *


def evaluate_fitness(
    args, mode, M, M_vol, M_large, video, baseline, model, positions, full_ids, output_ids, dummy_inputs_orig,
    anchor_base=0.0, anchor_orig=1.0
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
            mean_prob = anchor_base if mode == 'deletion' else anchor_orig
        else:
            mean_prob = target_probs.mean().item()

    tv_penalty_raw = tv_norm_3d(M_vol)
    tv_penalty = tv_penalty_raw.item() if isinstance(tv_penalty_raw, torch.Tensor) else float(tv_penalty_raw)
    
    """
    if mode == 'deletion': # minimize probability towards baseline
        L1_penalty = torch.mean(1.0 - M_large).item() # Penalize dropping 
        diff_loss = abs(mean_prob - anchor_base)
        fitness = diff_loss + args.reg_lambda * (L1_penalty + tv_penalty)
    else: # maximize probability towards original
        L1_penalty = torch.mean(M_large).item() # Penalize revealing 
        diff_loss = abs(mean_prob - anchor_orig)
        fitness = diff_loss + args.reg_lambda * (L1_penalty + tv_penalty)
    """

    if mode == 'deletion': # minimize probability 
        L1_penalty = torch.mean(1.0 - M_large).item() # Penalize dropping 
        fitness = mean_prob + args.reg_lambda * (L1_penalty + tv_penalty)
    else: # maximize probability 
        L1_penalty = torch.mean(M_large).item() # Penalize revealing 
        fitness = -mean_prob + args.reg_lambda * (L1_penalty + tv_penalty)

    return float(fitness), mean_prob


def CMA_ES(
    args, model, processor, tokenizer, full_ids, output_ids, 
    frames_orig, frames_base, tubelets, positions=None, mode='deletion',
    max_iters=None, initial_weights=None, active_tubes=None, popsize=None
    ):
    """
    Performs main optimization using CMA-ES optimizers
    - Finds ~ optimal weights according to objective (min/max confidence)
    - Returns metrics and found weights
    """
    num_tubes = int(tubelets.max()) + 1
    tubelets_tensor = torch.tensor(tubelets, device=model.device, dtype=torch.long)
    is_qwen = getattr(args, 'model', '') == 'qwen'
    
    packed_inputs = _get_rescale_and_dummys(
        model, processor, frames_orig, frames_base, is_qwen, tubelets
    )
    (dummy_inputs_orig, video, baseline, target_T, target_H, target_W, 
     t_dim_index, crop_top, crop_left, new_H, new_W, T_orig, H_orig, W_orig) = packed_inputs

    # Pre-calculate base and original anchors dynamically
    W_zeros = torch.full((num_tubes,), -100.0, device=model.device) 
    M_large_zeros = torch.sigmoid(W_zeros)[tubelets_tensor].unsqueeze(1).float()
    M_scaled_zeros, M_vol_zeros = rescale_mask(M_large_zeros, new_H, new_W, crop_top, crop_left, target_H, target_W, target_T, T_orig, is_qwen, t_dim_index)
    _, anchor_base = evaluate_fitness(args, mode, M_scaled_zeros, M_vol_zeros, M_large_zeros, video, baseline, model, positions, full_ids, output_ids, dummy_inputs_orig)

    W_ones = torch.full((num_tubes,), 100.0, device=model.device) 
    M_large_ones = torch.sigmoid(W_ones)[tubelets_tensor].unsqueeze(1).float()
    M_scaled_ones, M_vol_ones = rescale_mask(M_large_ones, new_H, new_W, crop_top, crop_left, target_H, target_W, target_T, T_orig, is_qwen, t_dim_index)
    _, anchor_orig = evaluate_fitness(args, mode, M_scaled_ones, M_vol_ones, M_large_ones, video, baseline, model, positions, full_ids, output_ids, dummy_inputs_orig)

    if initial_weights is not None:
        mean_init = np.array(initial_weights, dtype=np.float64)
    else:
        W_init = 2.0 if mode == 'deletion' else -2.0
        mean_init = np.full(num_tubes, W_init, dtype=np.float64)

    sigma_0 = 1.0
    iters = max_iters if max_iters is not None else getattr(args, 'iterations', 150)
    es_popsize = popsize if popsize is not None else getattr(args, 'popsize', 20)

    es = cma.CMAEvolutionStrategy(mean_init, sigma_0, {
        'maxiter': iters, 
        'popsize': es_popsize,
        'bounds': [-5.0, 5.0]
    })

    generation = 0
    active_set = set(active_tubes) if active_tubes is not None else None

    while not es.stop():
        candidates = es.ask()
        
        if active_set is not None:
            for c in candidates:
                for t in range(num_tubes):
                    if t not in active_set:
                        c[t] = mean_init[t]

        candidates_tensor = torch.tensor(np.stack(candidates), dtype=torch.float32, device=model.device)
        fitnesses = []
        confidences = []
        for i in range(len(candidates)):
            W = candidates_tensor[i]
            M_large = torch.sigmoid(W)[tubelets_tensor].unsqueeze(1).float()
            
            M_scaled, M_vol = rescale_mask(M_large, new_H, new_W, crop_top, crop_left, target_H, target_W, target_T, T_orig, is_qwen, t_dim_index)
            
            fitness, confidence = evaluate_fitness(
                args, mode, M_scaled, M_vol, M_large, video, baseline, model, 
                positions, full_ids, output_ids, dummy_inputs_orig, 
                anchor_base=anchor_base, anchor_orig=anchor_orig
            )
            fitnesses.append(fitness)
            confidences.append(confidence)

        es.tell(candidates, fitnesses)
        best_idx = np.argmin(fitnesses)
        es.disp()
        eprint(f"Gen {generation} | Best Fit: {fitnesses[best_idx]:.4f} | Conf: {confidences[best_idx]:.4f}")
        generation += 1
    
    W_best = es.result.xbest if es.result.xbest is not None else mean_init
    W_final = sigmoid(W_best)

    if mode == 'deletion':
        scores = {t: float(1.0 - W_final[t]) for t in range(num_tubes)}
    else:
        scores = {t: float(W_final[t]) for t in range(num_tubes)}

    max_score = max(scores.values()) if scores else 0.0
    threshold = max_score * 0.10 if max_score > 0.01 else 0.0 # Needs to be doing something at least
    selected_tubelets = [t for t, s in scores.items() if s >= threshold]

    # es.D contains the square roots of the eigenvalues of the covariance matrix
    eigenvalues = es.D ** 2
    sum_eig = np.sum(eigenvalues)
    sum_sq_eig = np.sum(eigenvalues ** 2)
    d_eff = float((sum_eig ** 2) / sum_sq_eig) if sum_sq_eig > 0 else 0.0

    gen_best_fit = np.min(fitnesses)
    # Apply sigmoid to the mask - we compare actual mask differences [0, 1], not unbounded logits [-5, 5]
    top_masks = [sigmoid(c) for i, c in enumerate(candidates) if fitnesses[i] <= gen_best_fit * 1.10]
    if len(top_masks) > 1:
        dists = [np.mean(np.abs(top_masks[i] - top_masks[j])) 
                 for i in range(len(top_masks)) for j in range(i + 1, len(top_masks))]
        diversity = float(np.mean(dists))
    else:
        diversity = 0.0

    metrics = {
        "cma_evals": es.result.evaluations,
        "best_fitness": es.result.fbest,
        "d_eff": d_eff,
        "diversity": diversity,
        "num_top_candidates": len(top_masks)
    }
    
    return selected_tubelets, scores, metrics