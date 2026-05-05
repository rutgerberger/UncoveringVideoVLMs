import os
import gc

import cma
import numpy as np
import torch
import torchvision
import torch.nn.functional as F

from PIL import Image

from utils.preprocessing import rescale_mask, create_super_tubelets, unpack_super_weights, get_baseline_insertion, get_baseline_deletion
from utils.logging import eprint
from utils.evaluation import tv_norm_3d, evaluate_fitness, evaluate_confidence, jaccard_similarity
from utils.model_utils import get_rescale_and_dummys, sigmoid, calculate_gradient
from utils.visualization import visualize_gradients, visualize_heatmap, debug_save_pixels_interval

SAVE_INTERMEDIATE_VISUALS = False



def CMA_ES(
        args, model, processor, tokenizer, full_ids, output_ids, 
        frames_orig, frames_base, tubelets, positions=None, mode='joint',
        max_iters=None, initial_weights=None, active_tubes=None, popsize=None
    ):
    """
    Performs Universal CMA-ES optimization
    - mode='joint': Minimizes deletion and maximizes insertion simultaneously.
    - mode='deletion': Only minimizes deletion log-likelihood.
    - mode='insertion': Only maximizes insertion log-likelihood.
    """
    
    sigma_0 = 1.0 #HP
    iters = max_iters if max_iters is not None else getattr(args, 'iterations', 150)
    es_popsize = popsize if popsize is not None else getattr(args, 'popsize', 20)

    num_tubes = int(tubelets.max()) + 1
    tubelets_tensor = torch.tensor(tubelets, device=model.device, dtype=torch.long)
    is_qwen = getattr(args, 'model', '') == 'qwen'
    
    #We need these attributes to directly feed the model (without slow HF preprocessor)
    packed_inputs = get_rescale_and_dummys(
        model, processor, frames_orig, frames_base, is_qwen, tubelets
    )
    (dummy_inputs_orig, video, baseline, target_T, target_H, target_W, 
     t_dim_index, crop_top, crop_left, new_H, new_W, T_orig, H_orig, W_orig) = packed_inputs

    # Init depending on mode
    if initial_weights is not None:
        mean_init = np.array(initial_weights, dtype=np.float64)
    else:
        w_init = -2.0 if mode == 'insertion' else 2.0
        mean_init = np.full(num_tubes, w_init, dtype=np.float64)

    es = cma.CMAEvolutionStrategy(mean_init, sigma_0, {
        'maxiter': iters, 'popsize': es_popsize, 'bounds': [-5.0, 5.0]
    })

    generation = 0
    active_set = set(active_tubes) if active_tubes is not None else None

    #ES loop
    while not es.stop():
        candidates = es.ask()
        
        if active_set is not None:
            for c in candidates:
                for t in range(num_tubes):
                    if t not in active_set:
                        c[t] = mean_init[t]

        candidates_tensor = torch.tensor(np.stack(candidates), dtype=torch.float32, device=model.device)
        fitnesses, confidences = [], []
        
        for i in range(len(candidates)): #Evaluation
            W = candidates_tensor[i] # Map weights to tubelets
            M_large = torch.sigmoid(W)[tubelets_tensor].unsqueeze(1).float() # Raw weights to [0,1]
            #Rescale the mask to target dimension of the VLM
            M_scaled, M_vol = rescale_mask(M_large, new_H, new_W, crop_top, crop_left, target_H, target_W, target_T, T_orig, is_qwen, t_dim_index)
            
            #Use these to evaluate the fitness
            fitness, score_del, score_ins = evaluate_fitness(
                args, mode, M_scaled, M_vol, M_large, video, baseline, model, 
                positions, full_ids, output_ids, dummy_inputs_orig
            )
            fitnesses.append(fitness)
            confidences.append((score_del, score_ins))

        es.tell(candidates, fitnesses) #Update
        best_idx = np.argmin(fitnesses)
        es.disp()
        
        best_del, best_ins = confidences[best_idx]
        eprint(f"Gen {generation} | Best Fit: {fitnesses[best_idx]:.4f} | Conf (Del/Ins): {best_del:.4f} / {best_ins:.4f}")
        generation += 1
    
    W_best = es.result.xbest if es.result.xbest is not None else mean_init
    W_final = sigmoid(W_best)

    # Invert mask for Deletion and Joint to get Saliency (0-1). For insertion, W is already saliency.
    if mode == 'insertion':
        scores = {t: float(W_final[t]) for t in range(num_tubes)}
    else:
        scores = {t: float(1.0 - W_final[t]) for t in range(num_tubes)}

    eigenvalues = es.D ** 2
    sum_eig = np.sum(eigenvalues)
    sum_sq_eig = np.sum(eigenvalues ** 2)
    d_eff = float((sum_eig ** 2) / sum_sq_eig) if sum_sq_eig > 0 else 0.0

    gen_best_fit = np.min(fitnesses)
    top_masks = [sigmoid(c) for i, c in enumerate(candidates) if fitnesses[i] <= gen_best_fit * 1.10]
    if len(top_masks) > 1:
        # Calculate Jaccard similarity over the top 25% of tubelets within the population
        population_jaccard = jaccard_similarity(top_masks, top_k_fraction=0.25)
    else:
        population_jaccard = 1.0 # 100% similarity if only one mask qualifies

    metrics = {
        "cma_evals": es.result.evaluations,
        "best_fitness": es.result.fbest,
        "d_eff": d_eff,
        "population_jaccard": population_jaccard,
        "num_top_candidates": len(top_masks)
    }
    
    return scores, metrics


def process_video(args, model, tokenizer, processor, output_ids, full_ids, frames, tubelets, baseline_ins, baseline_del, positions=None):
    # Expects args.mask_mode = 'joint', 'separate', 'deletion', or 'insertion'
    mask_mode = getattr(args, 'mask_mode', 'joint') 
    
    frames_orig = [Image.fromarray(np.array(img).astype(np.uint8)) for img in frames]
    frames_del_base = [Image.fromarray(f.astype(np.uint8)) for f in baseline_del]
    frames_ins_base = [Image.fromarray(f.astype(np.uint8)) for f in baseline_ins]
    
    # Internal helper to handle Hierarchical vs Standard seamlessly for both objectives (joint vs separate)
    def run_cmaes_for_mode(opt_mode, base_frames):
        use_hierarchical = getattr(args, 'use_hierarchical', False)
        total_iters = getattr(args, 'iterations', 100)
        popsize = getattr(args, 'popsize', 22)
        
        if not use_hierarchical: #Just a single loop of CMA-ES
            eprint(f"\n=== Standard CMA-ES ({opt_mode.upper()}) ===")
            return CMA_ES(
                args, model, processor, tokenizer, full_ids, output_ids, frames_orig, base_frames, 
                tubelets, positions=positions, mode=opt_mode, max_iters=total_iters, popsize=popsize
            )

        eprint(f"\n=== Hierarchical Stage 1: Coarse ({opt_mode.upper()}) ===")
        video_array = np.stack([np.array(img) for img in frames])
        n_super = getattr(args, 'super_clusters', 12)
        cluster_mode = getattr(args, 'cluster_mode', 'spatial')
        super_tubelets, sub_to_super = create_super_tubelets(video_array, tubelets, n_clusters=n_super, mode=cluster_mode)
        
        #First part over super clusters
        super_scores, _ = CMA_ES(
            args, model, processor, tokenizer, full_ids, output_ids, frames_orig, base_frames, 
            super_tubelets, positions=positions, mode=opt_mode, max_iters=total_iters // 2, popsize=popsize
        )
        
        #Boil down to smaller clusters
        init_w, active_tubes = unpack_super_weights(args, tubelets, sub_to_super, super_scores, opt_mode)
            
        eprint(f"\n=== Hierarchical Stage 2: Fine ({opt_mode.upper()}) ===")
        return CMA_ES(
            args, model, processor, tokenizer, full_ids, output_ids, frames_orig, base_frames, 
            tubelets, positions=positions, mode=opt_mode, max_iters=total_iters // 2, 
            initial_weights=init_w, active_tubes=active_tubes, popsize=popsize
        )

    if mask_mode == 'separate':
        scores_del, metrics_del = run_cmaes_for_mode('deletion', frames_del_base)
        scores_ins, metrics_ins = run_cmaes_for_mode('insertion', frames_ins_base)
        # Hadamard Product merge
        merged_scores = {t: scores_del[t] * scores_ins.get(t, 0.0) for t in scores_del}
        if getattr(args, 'normalize_weights', False):
            max_val = max(merged_scores.values()) or 1.0
            merged_scores = {t: s / max_val for t, s in merged_scores.items()}
        ranked_tubelets = sorted(merged_scores.keys(), key=lambda t: merged_scores[t], reverse=True)
        return ranked_tubelets, merged_scores, {"del": metrics_del, "ins": metrics_ins}
        
    else: # 'joint', 'deletion', 'insertion'
        base = frames_ins_base if mask_mode == 'insertion' else frames_del_base
        scores, metrics = run_cmaes_for_mode(mask_mode, base)
        
        if getattr(args, 'normalize_weights', False):
            max_val = max(scores.values()) or 1.0
            scores = {t: s / max_val for t, s in scores.items()}
            
        ranked_tubelets = sorted(scores.keys(), key=lambda t: scores[t], reverse=True)
        return ranked_tubelets, scores, metrics