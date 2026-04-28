import os
import gc

import cma
import numpy as np
import torch
import torchvision
import torch.nn.functional as F

from PIL import Image

from new_utils.preprocessing import rescale_mask, create_super_tubelets, get_baseline_insertion, get_baseline_deletion
from new_utils.logging import eprint
from new_utils.evaluation import tv_norm_3d, evaluate_fitness, evaluate_confidence
from new_utils.model_utils import get_rescale_and_dummys, sigmoid, calculate_gradient
from new_utils.visualization import visualize_gradients, visualize_heatmap, debug_save_pixels_interval

SAVE_INTERMEDIATE_VISUALS = False



def CMA_ES(
        args, model, processor, tokenizer, full_ids, output_ids, 
        frames_orig, frames_base, tubelets, positions=None,
        max_iters=None, initial_weights=None, active_tubes=None, popsize=None
    ):
    """
    Performs Joint CMA-ES optimization (iGOS++ style)
    - Finds optimal weights minimizing deletion and maximizing insertion simultaneously.
    """
    num_tubes = int(tubelets.max()) + 1
    tubelets_tensor = torch.tensor(tubelets, device=model.device, dtype=torch.long)
    is_qwen = getattr(args, 'model', '') == 'qwen'
    
    packed_inputs = get_rescale_and_dummys(
        model, processor, frames_orig, frames_base, is_qwen, tubelets
    )
    (dummy_inputs_orig, video, baseline, target_T, target_H, target_W, 
     t_dim_index, crop_top, crop_left, new_H, new_W, T_orig, H_orig, W_orig) = packed_inputs

    # Initialize at W = 2.0 (Sigmoid(2.0) ~ 0.88 -> Mostly Background)
    if initial_weights is not None:
        mean_init = np.array(initial_weights, dtype=np.float64)
    else:
        mean_init = np.full(num_tubes, 2.0, dtype=np.float64)

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
            
            fitness, score_del, score_ins = evaluate_fitness(
                args, M_scaled, M_vol, M_large, video, baseline, model, 
                positions, full_ids, output_ids, dummy_inputs_orig
            )
            fitnesses.append(fitness)
            confidences.append((score_del, score_ins))

        es.tell(candidates, fitnesses)
        best_idx = np.argmin(fitnesses)
        es.disp()
        
        best_del, best_ins = confidences[best_idx]
        eprint(f"Gen {generation} | Best Fit: {fitnesses[best_idx]:.4f} | Conf (Del/Ins): {best_del:.4f} / {best_ins:.4f}")
        generation += 1
    
    W_best = es.result.xbest if es.result.xbest is not None else mean_init
    W_final = sigmoid(W_best)

    # Invert the mask: M=0 was salient, so we do 1.0 - M to get a 0-1 saliency score
    scores = {t: float(1.0 - W_final[t]) for t in range(num_tubes)}

    max_score = max(scores.values()) if scores else 0.0
    threshold = max_score * 0.10 if max_score > 0.01 else 0.0 
    selected_tubelets = [t for t, s in scores.items() if s >= threshold]

    eigenvalues = es.D ** 2
    sum_eig = np.sum(eigenvalues)
    sum_sq_eig = np.sum(eigenvalues ** 2)
    d_eff = float((sum_eig ** 2) / sum_sq_eig) if sum_sq_eig > 0 else 0.0

    gen_best_fit = np.min(fitnesses)
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


def process_video(args, model, tokenizer, processor, input_ids, output_ids, frames, tubelets, baseline_ins, baseline_del, positions=None):
    """
    Wrapper around CMA-ES Optimization (iGOS++ style objective).
    Performs standard optimization by default, or Two-Stage Curriculum Learning 
    if `args.use_hierarchical` is set to True.
    """
    
    # Configuration
    use_hierarchical = getattr(args, 'use_hierarchical', False)
    total_iters = getattr(args, 'iterations', 100)
    init_pop_size = getattr(args, 'popsize', 22)
    
    # Base conversions
    frames_del_base = [Image.fromarray(f.astype(np.uint8)) for f in baseline_del]
    frames_orig = [Image.fromarray(np.array(img).astype(np.uint8)) for img in frames]
    
    full_ids = torch.cat((input_ids, output_ids), dim=1)
    
    # Setup variables for the fine-grained run
    init_w = None
    active_tubes = None
    final_iters = total_iters # Defaults to full iterations for standard mode
    
    if use_hierarchical:
        eprint("\n=== Starting Hierarchical CMA-ES (Stage 1: Coarse) ===")
        video_array = np.stack([np.array(img) for img in frames])
        n_super = getattr(args, 'super_clusters', 12)
        cluster_mode = getattr(args, 'cluster_mode', 'spatial')
        freeze_losers = getattr(args, 'freeze_losers', False)
        
        # Split iterations in half for the two stages
        final_iters = total_iters // 2
        
        eprint(f"Building Super-Tubelets ({n_super} clusters, mode: {cluster_mode})")
        super_tubelets, sub_to_super_map = create_super_tubelets(video_array, tubelets, n_clusters=n_super, mode=cluster_mode)
        
        eprint(f"Coarse Optimization: Joint Objective")
        _, super_scores, _ = CMA_ES(
            args, model, processor, tokenizer, full_ids, output_ids, frames_orig, frames_del_base, 
            super_tubelets, positions=positions, max_iters=final_iters, popsize=init_pop_size
        )
        
        # Unpack Super-Weights to Sub-Weights using Inverse Sigmoid (Logit)
        num_sub_tubes = int(tubelets.max()) + 1
        init_w = np.zeros(num_sub_tubes)
        
        for sub_id in range(num_sub_tubes):
            super_id = sub_to_super_map[sub_id]
            # CMA_ES returns Saliency (S = 1.0 - M). Therefore M = 1.0 - S.
            # We initialize CMA-ES with the logit of M.
            s = np.clip(super_scores.get(super_id, 0.0), 1e-4, 1 - 1e-4)
            m = 1.0 - s
            init_w[sub_id] = np.log(m / (1.0 - m))
            
        # Optional: Freeze the bottom 50% performing coarse regions
        if freeze_losers:
            eprint("Freezing bottom 50% of coarse regions...")
            med = np.median(list(super_scores.values()))
            active_tubes = [sub_id for sub_id in range(num_sub_tubes) if super_scores[sub_to_super_map[sub_id]] >= med]
            
    else:
        eprint("\n=== Starting Standard CMA-ES (Joint Objective) ===")

    # --- Final / Standard Optimization Stage ---
    stage_name = "Stage 2: Fine-grained " if use_hierarchical else ""
    eprint(f"{stage_name}Optimization: Joint Objective")
    
    selected_tubelets, final_scores, metrics = CMA_ES(
        args, model, processor, tokenizer, full_ids, output_ids, frames_orig, frames_del_base, 
        tubelets, positions=positions, max_iters=final_iters, 
        initial_weights=init_w, active_tubes=active_tubes, popsize=init_pop_size
    )
    
    # Optional Normalization
    if getattr(args, 'normalize_weights', False):
        eprint("Normalizing weights...")
        max_score = max(final_scores.values()) or 1.0
        final_scores = {t: s / max_score for t, s in final_scores.items()}

    # Rank
    selected_ranked = sorted(final_scores.keys(), key=lambda t: final_scores[t], reverse=True)

    # Return the unified results duplicated to satisfy main.py's legacy (Ins/Del) unpacking
    return selected_ranked, final_scores, metrics


def optimize_tubelet_weights(
        args, model, tokenizer, processor, full_ids, output_ids, frames, baseline_frames, 
        tubelets, positions, mode='deletion', stage="init"
    ):
    """
    Gradually move the weights of the tubelets 
    to optimize the insertion / deletion loss.
    """
    #-- Initialize the raw weight tensor <-inf, inf>
    num_tubes = int(tubelets.max()) + 1
    if mode == 'deletion': 
        W_raw = torch.full((num_tubes,), 2.0, dtype=torch.float32, device=model.device, requires_grad=True)
    else: 
        W_raw = torch.full((num_tubes,), -2.0, dtype=torch.float32, device=model.device, requires_grad=True)
    tubelets_tensor = torch.tensor(tubelets, device=model.device, dtype=torch.long)
    is_qwen = getattr(args, 'model', '') == 'qwen'
    optimizer = torch.optim.Adam([W_raw], lr=args.lr)
    
    #-- Efficiently obtain required inputs for the model and resizing factors (for downscaling)
    packed_inputs = get_rescale_and_dummys(model, processor, frames, baseline_frames, is_qwen, tubelets)
    (dummy_inputs_orig, pixels_orig, pixels_base,
    target_T, target_H, target_W, t_dim_index,
    crop_top, crop_left, new_H, new_W, T_orig, H_orig, W_orig) = packed_inputs
    
    #-- Lists to keep track of gradients and variances (debugging purposes)
    top_10_gradients = []
    mean_gradients = []
    reg_gradients = []
    sparse_gradient_counts = []
    mean_variances = []

    gamma = 0.2  # The decay rate
    lambda_2 = 0.5  # The initial strength of the L2 smoothing
    #-- How much steps we want to take
    for i in range(args.iterations):
        optimizer.zero_grad() # Reset gradients
        W = torch.sigmoid(W_raw) 
        M_high_res = W[tubelets_tensor].unsqueeze(1).float() 
        
        #-- Resizing and cropping
        M_resized_tv = F.interpolate(M_high_res, size=(new_H, new_W), mode='bilinear', align_corners=False)
        M_cropped_tv = M_resized_tv[:, :, crop_top:crop_top+target_H, crop_left:crop_left+target_W]
        M_vol_tv = M_cropped_tv.permute(1, 0, 2, 3).unsqueeze(0) 
        if target_T != T_orig: 
            M_vol_tv = F.interpolate(M_vol_tv, size=(target_T, target_H, target_W), mode='trilinear', align_corners=False)
            
        #-- Compute Regularization Loss (Outside the IG loop, more efficient)
        loss_tv = tv_norm_3d(M_vol_tv, tv_beta=2)
        if mode == 'deletion':
            # Mask initialized to 1. Penalize moving away from 1.
            loss_l1 = torch.mean(torch.abs(1.0 - W))
            loss_l2 = torch.mean((1.0 - W)**2)
        else:
            # Mask initialized to 0. Penalize moving away from 0.
            loss_l1 = torch.mean(torch.abs(W))
            loss_l2 = torch.mean((W)**2)
        current_l2_weight = lambda_2 * np.exp(-gamma * i)
        reg_loss = args.reg_lambda * (loss_l1 + loss_tv) + (current_l2_weight * loss_l2)
        reg_loss.backward() 
        reg_grads = W_raw.grad.detach().cpu().numpy() 
        reg_gradients.append(reg_grads.copy())
        #-- Clean Up
        del W, M_high_res, M_resized_tv, M_cropped_tv, M_vol_tv, reg_loss
        
        # --- State tracking for independent gradients ---
        prev_accumulated_grad = reg_grads.copy()
        step_independent_grads = []
        raw_accumulated = None # Fallback initialization
        
        # --- Main Integrated Gradients Loop ---
        for step in range(1, args.ig_steps + 1):
            alpha = step / args.ig_steps
            W_step = torch.sigmoid(W_raw)
            M_high_res_step = W_step[tubelets_tensor].unsqueeze(1).float() 
            M_low_res_step = rescale_mask(
                M_high_res_step, new_H, new_W, crop_top, crop_left, 
                target_H, target_W, target_T, T_orig, is_qwen, t_dim_index
            )
            
            # Create the fused multiplier (M * alpha)
            #fused_mask = M_low_res_step * alpha
            # Interpolate between the pure baseline and the original video
            #pixels_interval = pixels_orig * fused_mask + pixels_base * (1.0 - fused_mask)
            # At alpha = 0.0, mask = 1.0: pixels : fused_mask = 0.0, pixels_orig = 0.0, pixels_base = 1.0
            # At alpha = 1.0, mask = 1.0: pixels : fused_mask = 1.0, pixels_orig = 1.0, pixels_base = 0.0
            # At alpha = 0.0, mask = 0.0: pixels : fused_mask = 0.0, pixels_orig = 0.0, pixels_base = 1.0
            # At alpha = 1.0, mask = 0.0: pixels : fused_mask = 0.0, pixels_orig = 0.0, pixels_base = 1.0

            if mode == 'deletion':
                # INITIALIZATION: M ~ 1.0
                # ANCHOR (alpha=0): pixels_base
                # TARGET (alpha=1): pixels_orig
                fused_mask = M_low_res_step * alpha
                pixels_interval = pixels_orig * fused_mask + pixels_base * (1.0 - fused_mask)

            else: # mode == 'insertion'
                # INITIALIZATION: M ~ 0.0
                # ANCHOR (alpha=0): pixels_orig
                # TARGET (alpha=1): pixels_base
                fused_mask = (1.0 - M_low_res_step) * alpha
                pixels_interval = pixels_base * fused_mask + pixels_orig * (1.0 - fused_mask)

            # Call modified calculate_gradient
            predicted_text, target_text, raw_accumulated, mean_prob = calculate_gradient(
                model, tokenizer, W_raw, pixels_interval, full_ids, 
                output_ids, positions, mode, args, 
                is_qwen, dummy_inputs_orig
            )
            
            # --- Calculate Independent Gradients ---
            independent_grad = raw_accumulated - prev_accumulated_grad
            prev_accumulated_grad = raw_accumulated.copy()
            step_independent_grads.append(independent_grad.copy())

            if i % 5 == 0 and SAVE_INTERMEDIATE_VISUALS:  
                save_folder = os.path.join(args.output_dir, f'gradients/{stage}')
                os.makedirs(save_folder, exist_ok=True)
                
                mask_hr = W_step[tubelets_tensor].detach().cpu().numpy() 
                faded_hr_frames = []
                
                for t_idx in range(T_orig):
                    orig_f = np.array(frames[t_idx]).astype(np.float32)
                    base_f = np.array(baseline_frames[t_idx]).astype(np.float32)
                    m = mask_hr[t_idx, ..., np.newaxis] 
                    if mode == 'deletion':
                        fused_mask = m * alpha
                        faded_f = orig_f * fused_mask + base_f * (1.0 - fused_mask)
                    else: # mode == 'insertion'
                        fused_mask = (1.0 - m) * alpha
                        faded_f = base_f * fused_mask + orig_f * (1.0 - fused_mask)
                    faded_hr_frames.append(np.clip(faded_f, 0, 255).astype(np.uint8))
                
                if is_qwen:
                    debug_tensors = [torch.from_numpy(f).permute(2, 0, 1).float() / 255.0 for f in faded_hr_frames]
                    grid = torchvision.utils.make_grid(debug_tensors, nrow=len(debug_tensors), padding=2)
                    save_path = os.path.join(save_folder, f"debug_blend_alpha_{alpha:.2f}.png")
                    torchvision.utils.save_image(grid, save_path)
                else:
                    debug_save_pixels_interval(
                        pixels_tensor=pixels_interval, 
                        orig_tensor=pixels_orig, 
                        output_dir=save_folder, 
                        filename=f"debug_blend_alpha_{alpha:.2f}.png",
                        t_dim_index=t_dim_index
                    )
                
                # Visualize the INDEPENDENT gradients for this specific step
                visualize_gradients(
                    gradients=independent_grad, 
                    frames=faded_hr_frames, 
                    tubelets=tubelets, 
                    save_folder=save_folder, 
                    step=step, 
                    title=f"Independent_Grad_Iter_{i+1}_Alpha_{alpha:.2f}"
                )

        # --- Calculate and Visualize Variance Map ---
        grad_matrix = np.stack(step_independent_grads) 
        variance_array = np.var(grad_matrix, axis=0)
        variance_scores = {t: float(v) for t, v in enumerate(variance_array)}
        
        # NEW: Log and track the average variance across all tubelets for this iteration
        current_mean_var = float(np.mean(variance_array))
        mean_variances.append(current_mean_var)
        history_str = ", ".join([f"{v:.2e}" for v in mean_variances])
        
        eprint(f"Stage {stage} Iter {i+1}: Confidence in '{target_text}': {mean_prob:.4f}")
        eprint(f"Stage {stage} Iter {i+1}: Mean IG Variance: {current_mean_var:.2e} | History: [{history_str}]")
        
        save_folder = os.path.join(args.output_dir, f'gradients/{stage}')
        os.makedirs(save_folder, exist_ok=True)
        
        if i % 5 == 0 and SAVE_INTERMEDIATE_VISUALS:
            save_folder = os.path.join(args.output_dir, f'gradients/{stage}')
            os.makedirs(save_folder, exist_ok=True)
            variance_save_path = os.path.join(save_folder, f"Variance_Heatmap_Iter_{i+1}.gif")
            base_video_array = np.stack([np.array(img) for img in frames]) 
            visualize_heatmap(
                video_array=base_video_array,
                tubelet_labels=tubelets,
                tubelet_scores=variance_scores,
                output_path=variance_save_path,
                alpha=0.7 
            )

        # --- RESTORE ORIGINAL METRICS ---
        real_grads = raw_accumulated - reg_grads
        real_grads_abs = np.abs(real_grads)
        top_10_gradients.append(np.sort(real_grads_abs)[-10:])
        mean_gradients.append(np.mean(real_grads_abs))  
        sparse_gradient_counts.append(np.sum(real_grads_abs < 1e-4))
        
        # Update the weights of the tubelets 
        optimizer.step()
        if (i+1) % 10 == 0:
            eprint(f"Iter {i+1}/{args.iterations} | TV: {loss_tv.item():.4f} | L1: {loss_l1.item():.4f}")
            
        del loss_tv, loss_l1
        torch.cuda.empty_cache()
        
    # -- Generate final output weights
    final_weights = torch.sigmoid(W_raw).detach().cpu().numpy()
    if mode == 'deletion':
        init_w = sigmoid(2.0)
        scores = {t: float(init_w - w) for t, w in enumerate(final_weights)}
    else:
        init_w = sigmoid(-2.0) 
        scores = {t: float(w - init_w) for t, w in enumerate(final_weights)}
        
    max_score = max(scores.values()) 
    if max_score > 0.01: 
        dynamic_threshold = max_score * 0.5
        selected_tubelets = [t for t, s in scores.items() if s >= dynamic_threshold]
    else:
        selected_tubelets = []
    if len(selected_tubelets) == 0:
        eprint(f"Warning: Model gradients were dead. Falling back to top 5%.")
        ranked_all = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
        selected_tubelets = ranked_all[:max(1, int(num_tubes * 0.05))]
    
    del pixels_orig, pixels_base
    del dummy_inputs_orig
    del optimizer, W_raw
    torch.cuda.empty_cache()
    gc.collect()

    overall_mean_magnitude = float(np.mean(mean_gradients)) if mean_gradients else 0.0
    metrics = {
        "mean_magnitude": overall_mean_magnitude,
        "mean_variances_history": mean_variances
    }

    return selected_tubelets, scores, metrics

def spix_gradient_iterative(args, model, tokenizer, processor, input_ids, output_ids, frames, tubelets, positions=None, max_stages=2, discount_factor=0.75, index=1):
    """
    Optimize two separate masks. 'Whack-a-Mole Optimization'
        - We run a while loop until the model's confidence is destroyed or max_stages is hit.
        - Saliency scores are normalized per stage and decayed over time.
    """
    video_array = np.stack([np.array(img) for img in frames])
    full_ids = torch.cat((input_ids, output_ids), dim=1)
    is_qwen = getattr(args, 'model', '') == 'qwen'
    
    target_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # -- Generate Base Canvases
    baseline_ins = get_baseline_insertion(args, video_array) 
    baseline_del = get_baseline_deletion(args, video_array)  

    # -- State Tracking
    accumulated_ins = set()
    accumulated_del = set()
    final_scores_ins = {}
    final_scores_del = {}
    deletion_metrics = {}
    insertion_metrics = {}

    # -- Get the "True" Original Confidence once to base thresholds on
    frames_orig = [Image.fromarray(f.astype(np.uint8)) for f in video_array]
    _, initial_orig_conf, initial_orig_ent = evaluate_confidence(model, processor, frames_orig, input_ids, output_ids, is_qwen)
    # Dynamic thresholds based on the unmasked video's confidence
    min_confidence_del = 0.8 * initial_orig_conf # Stop deletion when confidence drops to 80% of original (AND entr).
    max_confidence_ins = 0.8 * initial_orig_conf # Stop insertion when we recover 80% of original confidence
    target_entropy = 1.50 * initial_orig_ent # Entropy increases by 50%


    eprint(f"\n--- Starting Iterative Deletion Optimization ---")
    current_video_del = video_array.copy()
    stage = 1
    
    while stage <= max_stages:
        eprint(f"\n>> [Deletion Stage {stage}/{max_stages}]")
        frames_current_video = [Image.fromarray(f.astype(np.uint8)) for f in current_video_del]
        frames_del_base = [Image.fromarray(f.astype(np.uint8)) for f in baseline_del]

        # Check current model confidence before optimizing
        pred_ids, current_conf, current_ent = evaluate_confidence(model, processor, frames_current_video, input_ids, output_ids, is_qwen)
        pred_text = tokenizer.decode(pred_ids[0], skip_special_tokens=True)
        eprint(f"Current State -> Prediction: '{pred_text}' | Entropy {current_ent:.4f} | Confidence in '{target_text}': {current_conf:.4f}")
        
        if current_conf <= min_confidence_del and current_ent >= target_entropy:
            eprint(f"Target destroyed! Conf ({current_conf:.4f} <= {min_confidence_del:.4f}) AND Ent ({current_ent:.4f} >= {target_entropy:.4f}). Stopping Deletion.")
            break

        # Main Weight Optimization Loop
        selected, scores, metrics = optimize_tubelet_weights(
            args, model, tokenizer, processor, full_ids, output_ids, frames_current_video, frames_del_base, 
            tubelets, positions, mode='deletion', stage=f"{index}-{stage}-deletion")
        # selected, scores, metrics = SimulatedAnnealing_optimization(
        #     args, model, processor, tokenizer, full_ids, output_ids, 
        #     frames_current_video, frames_del_base, tubelets, positions=None, mode='deletion'
        # )

        eprint(f"Selected tubelets {selected}")
        if not selected:
            eprint("No new tubelets found. Stopping deletion stages early.")
            break

        deletion_metrics[f"stage_{stage}"] = metrics
        
        # Normalize and Decay
        stage_max = max(scores.values()) if scores else 1.0
        stage_min = min(scores.values()) if scores else 0.0
        current_discount = discount_factor ** (stage - 1)
        for t, s in scores.items():
            #normalized_s = (s - stage_min) / (stage_max - stage_min + 1e-7)
            decayed_s = s * current_discount
            final_scores_del[t] = max(final_scores_del.get(t, 0), decayed_s) # Gradients above 0

        # Filtering & Recording
        new_selected = [t for t in selected if t not in accumulated_del]
        accumulated_del.update(new_selected)

        # Update the target video: completely mask the newly selected tubelets with the baseline
        mask = np.isin(tubelets, list(accumulated_del))[..., np.newaxis]
        current_video_del = np.where(mask, baseline_del, video_array)
        stage += 1

    eprint(f"\n--- Starting Iterative Insertion Optimization ---")
    current_baseline_ins = baseline_ins.copy()
    stage = 1

    while stage <= max_stages:
        eprint(f"\n>> [Insertion Stage {stage}/{max_stages}]")
        # For insertion, the "baseline" we pass to the model gets progressively more original pixels revealed
        frames_current_ins = [Image.fromarray(f.astype(np.uint8)) for f in current_baseline_ins]
        
        # Check current model confidence before optimizing
        pred_ids, current_conf, current_ent = evaluate_confidence(model, processor, frames_current_ins, input_ids, output_ids, is_qwen)
        pred_text = tokenizer.decode(pred_ids[0], skip_special_tokens=True)
        eprint(f"Current State -> Prediction: '{pred_text}' | Entropy {current_ent:.4f} | Confidence in '{target_text}': {current_conf:.4f}")
        
        if current_conf > max_confidence_ins and stage > 1:
            eprint(f"Confidence recovered above threshold ({max_confidence_ins:.4f}). Stopping Insertion.")
            break

        # Main Weight Optimization Loop
        # Note: frames_orig is constant (the target), frames_current_ins acts as the blurred/constant baseline
        selected, scores, metrics = optimize_tubelet_weights(
            args, model, tokenizer, processor, full_ids, output_ids, frames_orig, frames_current_ins, 
            tubelets, positions, mode='insertion', stage=f"{index}-{stage}-insertion")
        # selected, scores, metrics = SimulatedAnnealing_optimization(
        #     args, model, processor, tokenizer, full_ids, output_ids, 
        #     frames_orig, frames_current_ins, tubelets, positions=None, mode='insertion'
        # )

        if not selected:
            eprint("No new tubelets found. Stopping insertion stages early.")
            break

        insertion_metrics[f"stage_{stage}"] = metrics
        # Normalize and Decay (for saliency scores)
        stage_max = max(scores.values()) if scores else 1.0
        stage_min = min(scores.values()) if scores else 0.0
        current_discount = discount_factor ** (stage - 1)
        for t, s in scores.items():
            normalized_s = (s - stage_min) / (stage_max - stage_min + 1e-7)
            decayed_s = normalized_s * current_discount
            final_scores_ins[t] = max(final_scores_ins.get(t, 0), decayed_s)

        # Filtering & Recording
        new_selected = [t for t in selected if t not in accumulated_ins]
        accumulated_ins.update(new_selected)

        # Update the target video: completely reveal the newly selected tubelets by replacing baseline with original
        mask = np.isin(tubelets, list(accumulated_ins))[..., np.newaxis]
        current_baseline_ins = np.where(mask, video_array, baseline_ins)
        
        stage += 1

    return list(accumulated_ins), list(accumulated_del), final_scores_ins, final_scores_del, insertion_metrics, deletion_metrics 