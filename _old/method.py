from utils import *
from method_helpers import *

from PIL import Image
import numpy as np
import torch

import heapq
import random
import itertools
import gc

from cmaes import CMA_ES
from ig_optimizer import optimize_tubelet_weights



# def match_keywords(output_list, kw_ids):
#     """Helper to find the sublist kw_ids within output_list and return the indices."""
#     kw_len = len(kw_ids)
#     for i in range(len(output_list) - kw_len + 1):
#         if output_list[i : i + kw_len] == kw_ids:
#             return list(range(i, i + kw_len))
#     return []

# def find_keywords(args, model, processor, input_ids, output_ids, frames, baseline_ins_frames, output_text, tokenizer=None, use_yake=False, special_ids=None):
#     """
#     Finds the words that change mostly when comparing to a baseline video.
#     - baseline insertion: we are comparing to a blurred version preferably
#     """
#     if special_ids is None:
#         special_ids = []
#     seq_len = output_ids.shape[-1]
    
#     if seq_len <= 4:
#         clean_output_ids = output_ids
#         if output_ids[0, -1] == tokenizer.eos_token_id:
#             clean_output_ids = output_ids[:, :-1] 
            
#         positions = list(range(clean_output_ids.shape[-1]))
#         keywords = [tokenizer.decode(idx).strip() for idx in clean_output_ids[0]]
#         valid_indices = [i for i, kw in enumerate(keywords) if kw]
#         positions = [positions[i] for i in valid_indices]
#         keywords = [keywords[i] for i in valid_indices]
        
#     else: 
#         if use_yake:
#             import yake
#             num_words = len(output_text.split())
#             keywords_num = 3 if num_words <= 10 else num_words // 4
#             kw_extractor = yake.KeywordExtractor(lan="en", n=2, dedupLim=0.2, top=keywords_num, features=None)
#             extracted = kw_extractor.extract_keywords(output_text)
#             kw_strings = [kw[0] for kw in extracted]
#             positions = []
#             keywords = []
#             for kw in kw_strings:
#                 kw_ids = tokenizer.encode(kw, add_special_tokens=False)
#                 matched_pos = match_keywords(output_ids[0].tolist(), kw_ids)
#                 if matched_pos:
#                     positions.extend(matched_pos)
#                     keywords.extend([tokenizer.decode(output_ids[0][p]).strip() for p in matched_pos])
#         else:
#             full_prompt = torch.cat((input_ids, output_ids), dim=1)
#             probs = get_token_probs(args, model, processor, full_prompt, output_ids, frames)
#             probs_blur = get_token_probs(args, model, processor, full_prompt, output_ids, baseline_ians_frames)
            
#             eps = 1e-7
#             probs_safe = torch.clamp(probs, min=eps)
#             probs_blur_safe = torch.clamp(probs_blur, min=eps)
            
#             condition = (
#                 (torch.log(probs_safe) - torch.log(probs_blur_safe) > 1.0) & 
#                 (probs > 0.001) & 
#                 (~torch.isin(output_ids[0], torch.tensor(special_ids, device=probs.device)))
#             )
#             positions = torch.where(condition)[0].tolist()
#             keywords = [tokenizer.decode(output_ids[0][idx]).strip() for idx in positions]
            
#     return positions, keywords
    
def spix_rise_perturbation(args, model, tokenizer, processor, input_ids, output_ids, frames, tubelets, positions=None, num_masks=50, p=0.5):
    """
    Randomized Input Sampling for Explanation (RISE).
    Generates `num_masks` random binary masks, evaluates the raw model logit, 
    and assigns importance based on the expected marginal contribution of each tubelet.
    """
    eprint(f"\n--- Starting RISE Perturbation ({num_masks} Masks) ---")
    
    video_array = np.stack([np.array(img) for img in frames])
    full_ids = torch.cat((input_ids, output_ids), dim=1)
    is_qwen = getattr(args, 'model', '') == 'qwen'
    target_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # -- Generate Baseline (background video when masked)
    baseline_ins = get_baseline_insertion(args, video_array) 
    baseline_frames = [Image.fromarray(f.astype(np.uint8)) for f in baseline_ins]
    num_tubes = int(tubelets.max()) + 1
    tubelets_tensor = torch.tensor(tubelets, device=model.device, dtype=torch.long)

    # -- Fast Tensor Setup
    packed_inputs = get_rescale_and_dummys(model, processor, frames, baseline_frames, is_qwen, tubelets)
    (dummy_inputs_orig, pixels_orig, pixels_base,
     target_T, target_H, target_W, t_dim_index,
     crop_top, crop_left, new_H, new_W, T_orig, H_orig, W_orig) = packed_inputs

    # -- Tracking arrays
    sum_score_vis = np.zeros(num_tubes, dtype=np.float32)
    count_vis = np.zeros(num_tubes, dtype=np.float32)
    sum_score_hid = np.zeros(num_tubes, dtype=np.float32)
    count_hid = np.zeros(num_tubes, dtype=np.float32)

    forward_kwargs = {
        "input_ids": full_ids,
        "attention_mask": torch.ones_like(full_ids).to(model.device), 
        "use_cache": False
    }
    if is_qwen:
        forward_kwargs["video_grid_thw"] = dummy_inputs_orig["video_grid_thw"]

    # -- Main Loop --
    for k in range(1, num_masks + 1):
        # Generate Random Binary Mask
        # W_step is 1 (visible) with probability p, and 0 (hidden) with probability 1-p
        W_step = torch.bernoulli(torch.full((num_tubes,), p, device=model.device))
        W_step_np = W_step.cpu().numpy()
        # Upscale and Format Mask to match Video Tensor
        M_high_res_step = W_step[tubelets_tensor].unsqueeze(1).float() 
        M_low_res_step = rescale_mask(
            M_high_res_step, new_H, new_W, crop_top, crop_left, 
            target_H, target_W, target_T, T_orig, is_qwen, t_dim_index
        )
        # Blend Tensors (on GPU)
        pixels_final = pixels_orig * M_low_res_step + pixels_base * (1.0 - M_low_res_step)
        forward_kwargs["pixel_values_videos"] = pixels_final.to(dtype=model.dtype)

        # Forward Pass (No Gradients Required ==> Faster =)) )
        with torch.no_grad():
            outputs = model(**forward_kwargs)
            logits = outputs.logits
            
            #extract raw logits (no softmax, prevent saturation)
            out_len = output_ids.shape[-1]
            target_logits = logits[:, -out_len - 1 : -1, :]
            target_logits_gathered = target_logits.gather(dim=-1, index=output_ids.unsqueeze(-1)).squeeze(-1)
            if positions is not None and len(positions) > 0:
                target_logits_gathered = target_logits_gathered[0, positions]
            
            #raw un-squashed confidence score of the current random mask
            score = target_logits_gathered.mean().item()

        #accumulate scores
        sum_score_vis += W_step_np * score
        count_vis += W_step_np
        sum_score_hid += (1.0 - W_step_np) * score
        count_hid += (1.0 - W_step_np)

        if k % 10 == 0:
            eprint(f">> RISE Iteration {k}/{num_masks} | Current Mask Mean Logit: {score:.4f}")

    # -- Scoring Calculation --
    # Expected logit when tubelet is visible
    E_vis = sum_score_vis / np.maximum(count_vis, 1.0)
    # Expected logit when tubelet is hidden
    E_hid = sum_score_hid / np.maximum(count_hid, 1.0)
    
    # The true importance of a tubelet is how much it increases the logit when visible
    # compared to when it is hidden. (Zero-centered automatically!)
    importance = E_vis - E_hid

    # -- Format for Framework --
    #We only care about tubelets that positively contribute to the target text
    scores_dict = {t: float(importance[t]) for t in range(num_tubes)}
    scores_ins = {t: max(0.0, s) for t, s in scores_dict.items()}
    scores_del = scores_ins.copy() # they are both equal, haha
    max_score = max(scores_ins.values())
    
    #Thresholding: Keep tubelets that provide at least 20% of the max observed contribution
    if max_score > 1e-4: 
        dynamic_threshold = max_score * 0.20 
        selected_tubelets = [t for t, s in scores_ins.items() if s >= dynamic_threshold]
    else:
        selected_tubelets = []

    if len(selected_tubelets) == 0:
        eprint(f"Warning: Variance was dead. Falling back to top 5%.")
        ranked_all = sorted(scores_ins.keys(), key=lambda k: scores_ins[k], reverse=True)
        selected_tubelets = ranked_all[:max(1, int(num_tubes * 0.05))]
    # Cleanup
    del pixels_orig, pixels_base, dummy_inputs_orig
    torch.cuda.empty_cache()
    gc.collect()
    return selected_tubelets, selected_tubelets, scores_ins, scores_del

def spix_cmaes(args, model, tokenizer, processor, input_ids, output_ids, frames, tubelets, baseline_ins, baseline_del, positions=None):
    """
    Goal: perform CMA-ES / Optimize Weights / Return weights (no merging yet)
    """
    full_ids = torch.cat((input_ids, output_ids), dim=1)
    
    # We pass the PIL images of the baselines to CMA_ES
    frames_ins_base = [Image.fromarray(f.astype(np.uint8)) for f in baseline_ins]
    frames_del_base = [Image.fromarray(f.astype(np.uint8)) for f in baseline_del]
    frames_orig = [Image.fromarray(np.array(img).astype(np.uint8)) for img in frames]

    eprint("--- Starting CMA-ES Optimization (Deletion) ---")
    _, scores_del, deletion_metrics = CMA_ES(
            args, model, processor, tokenizer, full_ids, output_ids, 
            frames_orig, frames_del_base, tubelets, positions=positions, mode='deletion'
    )
    
    eprint("--- Starting CMA-ES Optimization (Insertion) ---")
    _, scores_ins, insertion_metrics = CMA_ES(
            args, model, processor, tokenizer, full_ids, output_ids, 
            frames_orig, frames_ins_base, tubelets, positions=positions, mode='insertion'
    )

    if getattr(args, 'normalize_weights', False):
        eprint("Normalizing weights...")
        # Min-max normalization per dictionary
        max_ins = max(scores_ins.values()) or 1.0
        max_del = max(scores_del.values()) or 1.0
        scores_ins = {t: s / max_ins for t, s in scores_ins.items()}
        scores_del = {t: s / max_del for t, s in scores_del.items()}

    # Rank tubelets based on scores (Descending)
    selected_ins = sorted(scores_ins.keys(), key=lambda t: scores_ins[t], reverse=True)
    selected_del = sorted(scores_del.keys(), key=lambda t: scores_del[t], reverse=True)

    return selected_ins, selected_del, scores_ins, scores_del, insertion_metrics, deletion_metrics

def spix_hierarchical_cmaes(args, model, tokenizer, processor, input_ids, output_ids, frames, tubelets, baseline_ins, baseline_del, positions=None):
    """
    Two-Stage Curriculum Learning Optimization.
    Stage 1: Coarse optimization on Super-Tubelets.
    Stage 2: Fine optimization on unpacked Sub-Tubelets.
    """
    full_ids = torch.cat((input_ids, output_ids), dim=1)
    frames_ins_base = [Image.fromarray(f.astype(np.uint8)) for f in baseline_ins]
    frames_del_base = [Image.fromarray(f.astype(np.uint8)) for f in baseline_del]
    frames_orig = [Image.fromarray(np.array(img).astype(np.uint8)) for img in frames]
    video_array = np.stack([np.array(img) for img in frames])

    # Configuration
    n_super = getattr(args, 'super_clusters', 12)
    cluster_mode = getattr(args, 'cluster_mode', 'spatial') # 'spatial' or 'appearance'
    freeze_losers = getattr(args, 'freeze_losers', False)
    total_iters = getattr(args, 'iterations', 100)
    stage_iters = total_iters // 2
    init_pop_size = getattr(args, 'popsize', 22)

    eprint(f"--- Building Super-Tubelets ({n_super} clusters, mode: {cluster_mode}) ---")
    super_tubelets, sub_to_super_map = create_super_tubelets(video_array, tubelets, n_clusters=n_super, mode=cluster_mode)

    #First we do coarse optimization
    eprint(f"Starting Deletion Optimization (Coarse)")
    _, super_scores_del, _ = CMA_ES(
        args, model, processor, tokenizer, full_ids, output_ids, frames_orig, frames_del_base, 
        super_tubelets, positions=positions, mode='deletion', max_iters=stage_iters, popsize=init_pop_size
    )
    
    eprint(f"Starting Insertion Optimization (Coarse)")
    _, super_scores_ins, _ = CMA_ES(
        args, model, processor, tokenizer, full_ids, output_ids, frames_orig, frames_ins_base, 
        super_tubelets, positions=positions, mode='insertion', max_iters=stage_iters, popsize=init_pop_size
    )

    #Update the smaller tubelets
    num_sub_tubes = int(tubelets.max()) + 1
    
    #Unpack Super-Weights to Sub-Weights
    #We must convert the 0-1 saliency scores back into the raw unbounded weights CMA-ES uses 
    #(Inverse Sigmoid / Logit function). CMA-ES initialized deletion at 2.0 and insertion at -2.0.
    init_weights_del = np.zeros(num_sub_tubes)
    init_weights_ins = np.zeros(num_sub_tubes)
    
    for sub_id in range(num_sub_tubes):
        super_id = sub_to_super_map[sub_id]
        # Deletion score is (1 - sigmoid(W)), so W = -logit(score)
        s_del = np.clip(super_scores_del.get(super_id, 0.5), 1e-4, 1 - 1e-4)
        init_weights_del[sub_id] = -np.log(s_del / (1 - s_del))
        # Insertion score is sigmoid(W), so W = logit(score)
        s_ins = np.clip(super_scores_ins.get(super_id, 0.5), 1e-4, 1 - 1e-4)
        init_weights_ins[sub_id] = np.log(s_ins / (1 - s_ins))

    #Freezing (Optional)
    active_tubes_del = None
    active_tubes_ins = None
    if freeze_losers:
        eprint("Freezing bottom 50% of coarse regions...")
        # Only keep sub-tubelets whose parent score was above the median
        med_del = np.median(list(super_scores_del.values()))
        active_tubes_del = [sub_id for sub_id in range(num_sub_tubes) if super_scores_del[sub_to_super_map[sub_id]] >= med_del]
        
        med_ins = np.median(list(super_scores_ins.values()))
        active_tubes_ins = [sub_id for sub_id in range(num_sub_tubes) if super_scores_ins[sub_to_super_map[sub_id]] >= med_ins]

    #Final stage: fine-grained optimization
    
    eprint(f"Starting Deletion Optimization (fine-grained)")
    _, final_scores_del, deletion_metrics = CMA_ES(
        args, model, processor, tokenizer, full_ids, output_ids, frames_orig, frames_del_base, 
        tubelets, positions=positions, mode='deletion', max_iters=stage_iters, 
        initial_weights=init_weights_del, active_tubes=active_tubes_del, popsize=init_pop_size
    )
    
    eprint(f"Starting Insertion Optimization (fine-grained)")
    _, final_scores_ins, insertion_metrics = CMA_ES(
        args, model, processor, tokenizer, full_ids, output_ids, frames_orig, frames_ins_base, 
        tubelets, positions=positions, mode='insertion', max_iters=stage_iters,
        initial_weights=init_weights_ins, active_tubes=active_tubes_ins, popsize=init_pop_size
    )

    # Rank and Return
    selected_ins = sorted(final_scores_ins.keys(), key=lambda t: final_scores_ins[t], reverse=True)
    selected_del = sorted(final_scores_del.keys(), key=lambda t: final_scores_del[t], reverse=True)

    return selected_ins, selected_del, final_scores_ins, final_scores_del, insertion_metrics, deletion_metrics


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
        # selected, scores, metrics = optimize_tubelet_weights(
        #     args, model, tokenizer, processor, full_ids, output_ids, frames_current_video, frames_del_base, 
        #     tubelets, positions, mode='deletion', stage=f"{index}-{stage}-deletion")
        selected, scores, metrics = SimulatedAnnealing_optimization(
            args, model, processor, tokenizer, full_ids, output_ids, 
            frames_current_video, frames_del_base, tubelets, positions=None, mode='deletion'
        )

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
        # selected, scores, metrics = optimize_tubelet_weights(
        #     args, model, tokenizer, processor, full_ids, output_ids, frames_orig, frames_current_ins, 
        #     tubelets, positions, mode='insertion', stage=f"{index}-{stage}-insertion")
        selected, scores, metrics = SimulatedAnnealing_optimization(
            args, model, processor, tokenizer, full_ids, output_ids, 
            frames_orig, frames_current_ins, tubelets, positions=None, mode='insertion'
        )

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

def frame_redundancy(args, model, processor, input_ids, output_ids, frames, compute_interactions=True):
    """
    Calculates frame importance using Monte Carlo Shapley approximation 
    to capture temporal dependencies and redundancies.

    Handling frame redundancy:
        If Frame 4 and Frame 5 are nearly identical, algorithm handles this
        Whenever Frame 4 is added after Frame 5 in a random permutation, its
        marginal gain will be near zero (because the VLM already got the 
        information from Frame 5). Over M iterations, they share the credit 
        accurately.

    Handling synergy: 
        If a specific action requires both Frame 2 and Frame 6 to be understood 
        by the VLM, neither will score highly alone. But when the second one is
        added to the subset containing the first, the probability spikes.
        Implemented equation ensures gain is distributed across both.

    Interaction matrix:
        If the number is significantly <0:
            Frame i and j are redundant. 
            The VLM gets less value from having both of them together than 
            it expects from adding them individually.
        If the number is significantly >0: 
            Frame i and j are synergistic. 
            They unlock a higher probability together than they do apart.
        If the number is around 0:
            They are mostly independent of one another.

    We are looking for:
        - Which frames are redundant?
        - Which frames have a strong correlation?
        - Which frames are independently very important?
    """
    def get_subset_prob(subset):
        """ Helper to get probability of a specific subset easily """
        sub_frames = [frames[idx] if idx in subset else blank_frame for idx in range(N)]
        return get_prob(args, model, processor, full_ids, output_ids, sub_frames)

    N = len(frames)
    #Higher M = better approximation, but slower.
    M = getattr(args, 'mc_samples', 50) 
    full_ids = torch.cat((input_ids, output_ids), dim=1)
    #Keep expected sequence length for the VLM.
    #Replace frames not currently in our subset with a blank frame.
    if hasattr(frames[0], 'size'): # Assuming PIL Image
        blank_frame = Image.new('RGB', frames[0].size, (0, 0, 0))
    else:
        blank_frame = np.zeros_like(frames[0])
    shapley_values = {i: 0.0 for i in range(N)}
    interaction_matrix = np.zeros((N, N)) #Matrix to store pairwise interactions
    #First we precompute v(∅) - the baseline probability with a completely masked video
    empty_frames = [blank_frame for _ in range(N)]
    prob_empty = get_prob(args, model, processor, full_ids, output_ids, empty_frames)
    
    #Monte Carlo Approximation Loop
    for m in range(M):
        eprint(f"Calculating scores for Monte Carlo Sample {m+1}/{M}")
        perm = list(range(N)) #Random purmutation of frame indices
        random.shuffle(perm)
        current_subset_indices = set()
        current_prob = prob_empty
        for i in perm: #Now add frames one by one according to the random permutation
            current_subset_indices.add(i)
            # Build the temporal sequence: actual frame if in subset, else blank
            current_frames = [
                frames[idx] if idx in current_subset_indices else blank_frame 
                for idx in range(N)
            ]
            new_prob = get_prob(args, model, processor, full_ids, output_ids, current_frames)
            marginal_gain = new_prob - current_prob # Calculate Marginal Gain: v(S U {i}) - v(S)
            shapley_values[i] += marginal_gain #Accumulate
            current_prob = new_prob

        if compute_interactions: #Pairwise Interaction Calculation
            eprint(f"Computing interactions. ({m+1}/{M})")
            # We evaluate pairs to map out synergy vs redundancy
            for i, j in itertools.combinations(range(N), 2):
                # Create a random subset S excluding i and j
                available_frames = [k for k in range(N) if k != i and k != j]
                subset_size = random.randint(0, len(available_frames))
                S = set(random.sample(available_frames, subset_size))
                # Calculate the 4 necessary probabilities
                v_S = get_subset_prob(S)
                v_S_i = get_subset_prob(S.union({i}))
                v_S_j = get_subset_prob(S.union({j}))
                v_S_i_j = get_subset_prob(S.union({i, j}))
                # I(i, j) calculation
                interaction = v_S_i_j - v_S_i - v_S_j + v_S
                # Accumulate symmetrically
                interaction_matrix[i, j] += interaction
                interaction_matrix[j, i] += interaction 
                
    for i in range(N): #average Marginal Gains (as required for equation)
        shapley_values[i] /= M
    if compute_interactions:
        interaction_matrix /= M
    #Finally we sort them by shapley value (higher = more important / less redundant)
    sorted_frames = sorted(shapley_values.items(), key=lambda item: item[1], reverse=True)
    return shapley_values, sorted_frames, interaction_matrix