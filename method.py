from utils import *
from method_helpers import *

from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F

import heapq
import random
import itertools

PROB_DELTA = 0.01

# def necessity_score(prob_orig, prob_del):
#     return prob_orig - prob_del

# def sufficiency_score(prob_ins, prob_empty):
#     return prob_ins - prob_empty


def spix(args, model, processor, input_ids, output_ids, frames, tubelets):
    """
    Depreciated. Not up to date with additions (e.g. find keywords, dynamic scaling)

    Exhaustive, greedy optimization over the tubelets.
    Initializes empty candidate set S. Then in each
    step loops over ALL tubelets to find the most
    informative tubelet to add. With highest necessity
    and insight scores. O(n * k) complexity.
    """
    unique_tubes = np.unique(tubelets)
    full_ids = torch.cat((input_ids, output_ids), dim=1)
    prob_orig = get_prob(args, model, processor, full_ids, output_ids, frames)
    
    selected_tubes = []
    remaining_tubes = list(unique_tubes)
    
    for step in range(min(args.k, len(unique_tubes))):
        best_tube = None
        best_score = -float('inf')

        for tube in remaining_tubes:
            candidate_set = selected_tubes + [tube]
            
            frames_ins = apply_mask(frames, tubelets, candidate_set)
            prob_ins = get_prob(args, model, processor, full_ids, output_ids, frames_ins)
            
            keep_tubes = [t for t in unique_tubes if t not in candidate_set]
            frames_del = apply_mask(frames, tubelets, keep_tubes)
            prob_del = get_prob(args, model, processor, full_ids, output_ids, frames_del)
            
            necessity = prob_orig - prob_del
            score = prob_ins + necessity
            
            if score > best_score:
                best_score = score
                best_tube = tube
                
        selected_tubes.append(best_tube)
        remaining_tubes.remove(best_tube)
        eprint(f"Step {step+1}/{args.k}: Selected tube {best_tube} | Obj Score: {best_score:.4f}")
        
    return selected_tubes

import heapq
def spix_optimized(args, model, processor, input_ids, output_ids, frames, tubelets, positions=None, opt_mode='combined', use_dynamic_lambda=True):
    unique_tubes = np.unique(tubelets)
    full_ids = torch.cat((input_ids, output_ids), dim=1)
    video_array = np.stack([np.array(img) for img in frames])
    
    prob_orig = get_prob(args, model, processor, full_ids, output_ids, frames, positions)
    
    blurred_video = precompute_blurred_video(video_array) 
    frames_blur_base = [Image.fromarray(frame.astype(np.uint8)) for frame in blurred_video]
    prob_blur = get_prob(args, model, processor, full_ids, output_ids, frames_blur_base, positions)

    selected_tubes = []
    tubelet_scores = {}
    current_total_score = 0.0 
    queue = []
    
    # Use blur_mask_fast directly as function expects numpy, converting down below
    apply_mask_func = lambda v_arr, tubes, active: apply_blur_mask_fast(v_arr, tubes, active) if args.mask_type == 'blur' else apply_mask_fast(v_arr, tubes, active)
    
    eprint("Precomputing tubelet centroids for distance penalty...")
    centroids_dict = precompute_tubelet_centroids(tubelets, unique_tubes)
    
    eprint(f"Performing initial evaluation for Lazy Greedy (Mode: {opt_mode})...")
    
    initial_evals = []
    max_ins_gain = 0.0
    max_del_gain = 0.0
    
    for tube in unique_tubes:
        frames_ins = [Image.fromarray(f) for f in apply_mask_func(video_array, tubelets, [tube])]
        prob_ins = get_prob(args, model, processor, full_ids, output_ids, frames_ins, positions)
        
        keep_tubes = [t for t in unique_tubes if t != tube] 
        frames_del = [Image.fromarray(f) for f in apply_mask_func(video_array, tubelets, keep_tubes)]
        prob_del = get_prob(args, model, processor, full_ids, output_ids, frames_del, positions) 
        
        ins_gain = max(0, prob_ins - prob_blur)
        del_gain = max(0, prob_orig - prob_del)
        max_ins_gain = max(max_ins_gain, ins_gain)
        max_del_gain = max(max_del_gain, del_gain)
        initial_evals.append((tube, prob_ins, prob_del))

    dynamic_lambda = 1.0
    if use_dynamic_lambda and opt_mode == 'combined':
        safe_del_gain = max(1e-5, max_del_gain)
        dynamic_lambda = max_ins_gain / safe_del_gain
        eprint(f"Computed Dynamic Lambda: {dynamic_lambda:.4f} (Max Ins: {max_ins_gain:.4f}, Max Del: {max_del_gain:.4f})")

    for tube, p_ins, p_del in initial_evals:
        if opt_mode == 'insertion':
            score = p_ins
        elif opt_mode == 'deletion':
            score = prob_orig - p_del
        else:
            score = p_ins + dynamic_lambda * (prob_orig - p_del)
        heapq.heappush(queue, (-score, tube)) 

    for step in range(len(unique_tubes)):
        best_tube = None
        best_marginal_gain = -float('inf')
        best_absolute_score = -float('inf')
        
        while queue:
            _, candidate = heapq.heappop(queue)
            candidate_set = selected_tubes + [candidate]
            frames_ins = [Image.fromarray(f) for f in apply_mask_func(video_array, tubelets, candidate_set)]
            prob_ins = get_prob(args, model, processor, full_ids, output_ids, frames_ins, positions)
            
            keep_tubes = [t for t in unique_tubes if t not in candidate_set]
            frames_del = [Image.fromarray(f) for f in apply_mask_func(video_array, tubelets, keep_tubes)]
            prob_del = get_prob(args, model, processor, full_ids, output_ids, frames_del, positions)
            
            dist_penalty = get_distance_penalty(candidate, selected_tubes, centroids_dict)
            
            if opt_mode == 'insertion':
                actual_absolute_score = prob_ins - (args.L1 * dist_penalty)
            elif opt_mode == 'deletion':
                actual_absolute_score = (prob_orig - prob_del) - (args.L1 * dist_penalty)
            else:
                actual_absolute_score = prob_ins + dynamic_lambda * (prob_orig - prob_del) - (args.L1 * dist_penalty)
                
            actual_marginal_gain = actual_absolute_score - current_total_score
            
            if not queue:
                best_tube = candidate
                best_marginal_gain = actual_marginal_gain
                best_absolute_score = actual_absolute_score
                break
                
            next_best_upper_bound_gain = -queue[0][0] - current_total_score
   
            if actual_marginal_gain >= next_best_upper_bound_gain:
                best_tube = candidate
                best_marginal_gain = actual_marginal_gain
                best_absolute_score = actual_absolute_score
                break
            else:
                heapq.heappush(queue, (-actual_absolute_score, candidate))
                
        tubelet_scores[best_tube] = max(0.0, best_marginal_gain)
        current_total_score = best_absolute_score
        selected_tubes.append(best_tube)
        
        eprint(f"Step {step+1}: Tube {best_tube} | Gain: {best_marginal_gain:.4f} | Total: {current_total_score:.4f} | P_ins: {prob_ins:.4f} | P_orig-P_del: {(prob_orig - prob_del):.4f}")
    
        if best_marginal_gain < 1e-4:
            eprint(f"Early stopping triggered at step {step+1}: Marginal gain flatlined.")
            break
            
    return selected_tubes, tubelet_scores

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