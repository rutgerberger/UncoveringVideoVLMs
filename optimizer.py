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
from utils.evaluation import tv_norm_3d, evaluate_fitness, jaccard_similarity
from utils.model_utils import get_rescale_and_dummys, sigmoid, get_vocab_stats
from utils.visualization import visualize_gradients, visualize_heatmap, debug_save_pixels_interval

def _get_initial_state(mode, num_tubes, initial_weights, active_tubes):
    """Determines the starting mean vector and the active constraints."""
    if initial_weights is not None:
        mean_init = np.array(initial_weights, dtype=np.float64)
    else:
        w_init = -2.0 if mode == 'insertion' else 2.0
        mean_init = np.full(num_tubes, w_init, dtype=np.float64)
    active_set = set(active_tubes) if active_tubes is not None else None
    return mean_init, active_set


def _build_evaluator(
    args, model, mode, tubelets_tensor, packed_inputs, is_qwen, positions, 
    full_ids, output_ids, vocab_stats, active_set, mean_init, num_tubes
):
    """Returns a self-contained function to evaluate a single candidate vector."""
    (dummy_inputs_orig, video, baseline, target_T, target_H, target_W, 
     t_dim_index, crop_top, crop_left, new_H, new_W, T_orig, H_orig, W_orig) = packed_inputs

    def evaluate_candidate(candidate_weights):
        # Enforce active set constraint
        if active_set is not None:
            for t in range(num_tubes):
                if t not in active_set:
                    candidate_weights[t] = mean_init[t]
        # Convert to Mask Tensor
        W = torch.tensor(candidate_weights, dtype=torch.float32, device=model.device)
        M_large = torch.sigmoid(W)[tubelets_tensor].unsqueeze(1).float()
        # Rescale for the VLM
        M_scaled, M_vol = rescale_mask(
            M_large, new_H, new_W, crop_top, crop_left, 
            target_H, target_W, target_T, T_orig, is_qwen, t_dim_index
        )
        fitness, score_del, score_ins = evaluate_fitness(
            args, mode, M_scaled, M_vol, M_large, video, baseline, model, 
            positions, full_ids, output_ids, dummy_inputs_orig, vocab_stats=vocab_stats
        )
        return float(fitness), score_del, score_ins

    return evaluate_candidate


def run_bipop_cmaes(mean_init, sigma_0, iters, popsize, args, evaluate_fn):
    """Runs the advanced BIPOP-CMA-ES multi-restart strategy."""
    eprint(f"--- Running with BIPOP-CMA-ES ---")
    
    # fmin2 requires a function that returns a single scalar float (fitness)
    def objective_function(x):
        fitness, _, _ = evaluate_fn(x)
        return fitness
    
    max_evals = iters * popsize * 5

    options = {
        'maxiter': iters,
        'popsize': popsize,
        'bounds': [-5.0, 5.0],
        'seed': getattr(args, 'manual_seed', 42),
        'verbose': -1,
        'maxfevals': max_evals
    }
    res = cma.fmin2(objective_function, mean_init, sigma_0, options, restarts=5, bipop=True)
    
    W_best = res[1].result.xbest if res[1].result.xbest is not None else mean_init
    
    metrics = {
        "cma_evals": res[1].result.evaluations,    # Corrected attribute path
        "best_fitness": res[1].result.fbest,       # Corrected attribute path
        "d_eff": 0.0, 
        "population_jaccard": 1.0,
        "num_top_candidates": 1
    }
    return W_best, metrics


def run_standard_cmaes(mean_init, sigma_0, iters, popsize, args, evaluate_fn):
    """Runs the standard step-by-step CMA-ES loop."""
    es = cma.CMAEvolutionStrategy(mean_init, sigma_0, {
        'maxiter': iters, 'popsize': popsize, 'bounds': [-5.0, 5.0], 'seed': getattr(args, 'manual_seed', 42)
    })

    generation = 0
    while not es.stop():
        candidates = es.ask()
        
        fitnesses, confidences = [], []
        for c in candidates:
            fit, score_del, score_ins = evaluate_fn(c)
            fitnesses.append(fit)
            confidences.append((score_del, score_ins))

        es.tell(candidates, fitnesses)
        best_idx = np.argmin(fitnesses)
        es.disp()
        
        best_del, best_ins = confidences[best_idx]
        eprint(f"Gen {generation} | Best Fit: {fitnesses[best_idx]:.4f} | Raw Scores (Del/Ins): {best_del:.4f} / {best_ins:.4f}")
        generation += 1

    W_best = es.result.xbest if es.result.xbest is not None else mean_init

    # Compute Diversity/Convergence Metrics
    eigenvalues = es.D ** 2
    sum_sq_eig = np.sum(eigenvalues ** 2)
    d_eff = float((np.sum(eigenvalues) ** 2) / sum_sq_eig) if sum_sq_eig > 0 else 0.0

    gen_best_fit = np.min(fitnesses)
    top_masks = [sigmoid(c) for i, c in enumerate(candidates) if fitnesses[i] <= gen_best_fit * 1.10]
    population_jaccard = jaccard_similarity(top_masks, top_k_fraction=0.25) if len(top_masks) > 1 else 1.0

    metrics = {
        "cma_evals": es.result.evaluations,
        "best_fitness": es.result.fbest,
        "d_eff": d_eff,
        "population_jaccard": population_jaccard,
        "num_top_candidates": len(top_masks)
    }
    return W_best, metrics


def compute_final_scores(W_best, mode, num_tubes):
    """Converts the raw optimized weights into visual saliency scores."""
    W_final = sigmoid(W_best)
    if mode == 'insertion':
        return {t: float(W_final[t]) for t in range(num_tubes)}
    else:
        # Invert mask for Deletion and Joint to get Saliency (0-1)
        return {t: float(1.0 - W_final[t]) for t in range(num_tubes)}


def CMA_ES(
        args, model, processor, tokenizer, full_ids, output_ids, 
        frames_orig, frames_base, tubelets, positions=None, mode='joint',
        max_iters=None, initial_weights=None, active_tubes=None, popsize=None,
        vocab_stats=None
    ):
    """
    Performs Universal Optimization using either standard CMA-ES or BIPOP-CMA-ES.
    """
    
    # Parse HP & Environments
    sigma_0 = getattr(args, 'sigma_init', 1.0)
    iters = max_iters if max_iters is not None else getattr(args, 'iterations', 150)
    es_popsize = popsize if popsize is not None else getattr(args, 'popsize', 20)
    is_qwen = getattr(args, 'model', '') == 'qwen'
    
    num_tubes = int(tubelets.max()) + 1
    tubelets_tensor = torch.tensor(tubelets, device=model.device, dtype=torch.long)
    
    packed_inputs = get_rescale_and_dummys(model, processor, frames_orig, frames_base, is_qwen, tubelets)
    mean_init, active_set = _get_initial_state(mode, num_tubes, initial_weights, active_tubes)

    # Build Evaluation function
    evaluate_fn = _build_evaluator(
        args, model, mode, tubelets_tensor, packed_inputs, is_qwen, positions, 
        full_ids, output_ids, vocab_stats, active_set, mean_init, num_tubes
    )

    # Optimizer dependent on arg
    if getattr(args, 'use_bipop', False):
        best_weights, metrics = run_bipop_cmaes(mean_init, sigma_0, iters, es_popsize, args, evaluate_fn)
    else:
        best_weights, metrics = run_standard_cmaes(mean_init, sigma_0, iters, es_popsize, args, evaluate_fn)

    scores = compute_final_scores(best_weights, mode, num_tubes)
    
    return scores, metrics

# import os
# import gc

# import cma
# import numpy as np
# import torch
# import torchvision
# import torch.nn.functional as F

# from PIL import Image

# from utils.preprocessing import rescale_mask, create_super_tubelets, unpack_super_weights, get_baseline_insertion, get_baseline_deletion
# from utils.logging import eprint
# from utils.evaluation import tv_norm_3d, evaluate_fitness, jaccard_similarity
# from utils.model_utils import get_rescale_and_dummys, sigmoid, get_vocab_stats
# from utils.visualization import visualize_gradients, visualize_heatmap, debug_save_pixels_interval

# SAVE_INTERMEDIATE_VISUALS = False



# def CMA_ES(
#         args, model, processor, tokenizer, full_ids, output_ids, 
#         frames_orig, frames_base, tubelets, positions=None, mode='joint',
#         max_iters=None, initial_weights=None, active_tubes=None, popsize=None,
#         vocab_stats=None
#     ):
#     """
#     Performs Universal CMA-ES optimization
#     - mode='joint': Minimizes deletion and maximizes insertion simultaneously.
#     - mode='deletion': Only minimizes deletion log-likelihood.
#     - mode='insertion': Only maximizes insertion log-likelihood.
#     """
    
#     sigma_0 = getattr(args, 'sigma_init', 1.0) #HP
#     # We can't use these directly as args because we also have Curriculum Strategy
#     iters = max_iters if max_iters is not None else getattr(args, 'iterations', 150)
#     es_popsize = popsize if popsize is not None else getattr(args, 'popsize', 20)

#     num_tubes = int(tubelets.max()) + 1
#     tubelets_tensor = torch.tensor(tubelets, device=model.device, dtype=torch.long)
#     is_qwen = getattr(args, 'model', '') == 'qwen'
    
#     #We need these attributes to directly feed the model (without slow HF preprocessor)
#     packed_inputs = get_rescale_and_dummys(
#         model, processor, frames_orig, frames_base, is_qwen, tubelets
#     )
#     (dummy_inputs_orig, video, baseline, target_T, target_H, target_W, 
#      t_dim_index, crop_top, crop_left, new_H, new_W, T_orig, H_orig, W_orig) = packed_inputs

#     # Init depending on mode
#     if initial_weights is not None:
#         mean_init = np.array(initial_weights, dtype=np.float64)
#     else:
#         w_init = -2.0 if mode == 'insertion' else 2.0
#         mean_init = np.full(num_tubes, w_init, dtype=np.float64)

#     es = cma.CMAEvolutionStrategy(mean_init, sigma_0, {
#         'maxiter': iters, 'popsize': es_popsize, 'bounds': [-5.0, 5.0], 'seed': getattr(args, 'manual_seed', 42)
#     })

#     generation = 0
#     active_set = set(active_tubes) if active_tubes is not None else None

#     #ES loop
#     while not es.stop():
#         candidates = es.ask()
        
#         if active_set is not None:
#             for c in candidates:
#                 for t in range(num_tubes):
#                     if t not in active_set:
#                         c[t] = mean_init[t]

#         candidates_tensor = torch.tensor(np.stack(candidates), dtype=torch.float32, device=model.device)
#         fitnesses, confidences = [], []
        
#         for i in range(len(candidates)): #Evaluation
#             W = candidates_tensor[i] # Map weights to tubelets
#             M_large = torch.sigmoid(W)[tubelets_tensor].unsqueeze(1).float() # Raw weights to [0,1]
#             #Rescale the mask to target dimension of the VLM
#             M_scaled, M_vol = rescale_mask(M_large, new_H, new_W, crop_top, crop_left, target_H, target_W, target_T, T_orig, is_qwen, t_dim_index)
            
#             #Use these to evaluate the fitness
#             fitness, score_del, score_ins = evaluate_fitness(
#                 args, mode, M_scaled, M_vol, M_large, video, baseline, model, 
#                 positions, full_ids, output_ids, dummy_inputs_orig, vocab_stats=vocab_stats
#             )
#             fitnesses.append(fitness)
#             confidences.append((score_del, score_ins))

#         es.tell(candidates, fitnesses) #Update
#         best_idx = np.argmin(fitnesses)
#         es.disp()
        
#         best_del, best_ins = confidences[best_idx]
#         eprint(f"Gen {generation} | Best Fit: {fitnesses[best_idx]:.4f} | Raw Scores (Del/Ins): {best_del:.4f} / {best_ins:.4f}")
#         generation += 1
    
#     W_best = es.result.xbest if es.result.xbest is not None else mean_init
#     W_final = sigmoid(W_best)

#     # Invert mask for Deletion and Joint to get Saliency (0-1). For insertion, W is already saliency.
#     if mode == 'insertion':
#         scores = {t: float(W_final[t]) for t in range(num_tubes)}
#     else:
#         scores = {t: float(1.0 - W_final[t]) for t in range(num_tubes)}

#     eigenvalues = es.D ** 2
#     sum_eig = np.sum(eigenvalues)
#     sum_sq_eig = np.sum(eigenvalues ** 2)
#     d_eff = float((sum_eig ** 2) / sum_sq_eig) if sum_sq_eig > 0 else 0.0

#     gen_best_fit = np.min(fitnesses)
#     top_masks = [sigmoid(c) for i, c in enumerate(candidates) if fitnesses[i] <= gen_best_fit * 1.10]
#     if len(top_masks) > 1:
#         # Calculate Jaccard similarity over the top 25% of tubelets within the population
#         population_jaccard = jaccard_similarity(top_masks, top_k_fraction=0.25)
#     else:
#         population_jaccard = 1.0 # 100% similarity if only one mask qualifies

#     metrics = {
#         "cma_evals": es.result.evaluations,
#         "best_fitness": es.result.fbest,
#         "d_eff": d_eff,
#         "population_jaccard": population_jaccard,
#         "num_top_candidates": len(top_masks)
#     }
    
#     return scores, metrics


def process_video(args, model, tokenizer, processor, output_ids, full_ids, frames, tubelets, baseline_ins, baseline_del, positions=None):
    # Expects args.mask_mode = 'joint', 'separate', 'deletion', or 'insertion'
    mask_mode = getattr(args, 'mask_mode', 'joint') 
    
    frames_orig = [Image.fromarray(np.array(img).astype(np.uint8)) for img in frames]
    frames_del_base = [Image.fromarray(f.astype(np.uint8)) for f in baseline_del]
    frames_ins_base = [Image.fromarray(f.astype(np.uint8)) for f in baseline_ins]

    # These scores are required for normalizing the target logits
    # Which unbiases the optimization loop
    vocab_mus, vocab_sigmas = get_vocab_stats(
        args, model, processor, full_ids, output_ids, frames, positions
    )
    vocab_stats = (vocab_mus, vocab_sigmas)

    # Internal helper to handle Hierarchical vs Standard seamlessly for both objectives (joint vs separate)
    def run_cmaes_for_mode(opt_mode, base_frames):
        use_hierarchical = getattr(args, 'use_hierarchical', False)
        total_iters = getattr(args, 'iterations', 100)
        popsize = getattr(args, 'popsize', 22)
        
        if not use_hierarchical: #Just a single loop of CMA-ES
            eprint(f"\n=== Standard CMA-ES ({opt_mode.upper()}) ===")
            return CMA_ES(
                args, model, processor, tokenizer, full_ids, output_ids, frames_orig, base_frames, 
                tubelets, positions=positions, mode=opt_mode, max_iters=total_iters, popsize=popsize,
                vocab_stats=vocab_stats
            )

        eprint(f"\n=== Hierarchical Stage 1: Coarse ({opt_mode.upper()}) ===")
        video_array = np.stack([np.array(img) for img in frames])
        n_super = getattr(args, 'super_clusters', 12)
        cluster_mode = getattr(args, 'cluster_mode', 'spatial')
        super_tubelets, sub_to_super = create_super_tubelets(video_array, tubelets, n_clusters=n_super, mode=cluster_mode)
        
        #First part over super clusters
        super_scores, _ = CMA_ES(
            args, model, processor, tokenizer, full_ids, output_ids, frames_orig, base_frames, 
            super_tubelets, positions=positions, mode=opt_mode, max_iters=total_iters // 2, popsize=popsize, vocab_stats=vocab_stats
        )
        
        #Boil down to smaller clusters
        init_w, active_tubes = unpack_super_weights(args, tubelets, sub_to_super, super_scores, opt_mode)
            
        eprint(f"\n=== Hierarchical Stage 2: Fine ({opt_mode.upper()}) ===")
        return CMA_ES(
            args, model, processor, tokenizer, full_ids, output_ids, frames_orig, base_frames, 
            tubelets, positions=positions, mode=opt_mode, max_iters=total_iters // 2, 
            initial_weights=init_w, active_tubes=active_tubes, popsize=popsize, vocab_stats=vocab_stats
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