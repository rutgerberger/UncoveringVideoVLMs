from utils import *
from method_helpers import *

from PIL import Image
import numpy as np
import torch

import heapq
import random
import itertools


def spix_gradient_iterative(args, model, tokenizer, processor, input_ids, output_ids, frames, tubelets, positions=None, stages=3, iters_per_stage=20, index=1):
    """
    Optimize two separate masks. 'Whack-a-Mole Optimization'
        - We run gradient descent in multiple stages (with args.iterations per stage)
        - Each stage, highest scoring tublets are masked (or revealed for insertion)
        - We force the optimizer to find secondary and tertiary evidence in subsequent stages
    """
    video_array = np.stack([np.array(img) for img in frames])
    full_ids = torch.cat((input_ids, output_ids), dim=1)
    
    # -- Generate Base Canvases
    baseline_ins = get_baseline_insertion(args, video_array) 
    baseline_del = get_baseline_deletion(args, video_array)  

    # -- State Tracking
    accumulated_ins = set()
    accumulated_del = set()
    final_scores_ins = {}
    final_scores_del = {}

    # -- Insetion Optimization
    eprint(f"\n--- Starting Iterative Insertion Optimization ({args.stages} Stages) ---")
    current_baseline_ins = baseline_ins.copy()
    for stage in range(1, args.stages + 1):
        eprint(f"\n>> [Insertion Stage {stage}/{args.stages}]")
        #Update baseline (shows previously found tubelets)
        frames_ins_base = [Image.fromarray(f.astype(np.uint8)) for f in current_baseline_ins]
        
        # -- Main Weight Optimization Loop
        selected, scores = optimize_tubelet_weights(
            args, model, tokenizer, processor, full_ids, output_ids, frames, frames_ins_base, 
            tubelets, positions, mode='insertion', stage=f"{index}-{stage}-insertion") 

        # -- Evaluation. Record highest score a tubelet achieved across all stages
        for t, s in scores.items():
            final_scores_ins[t] = max(final_scores_ins.get(t, 0), s)
        # -- Filtering & Recording Inside Set
        new_selected = [t for t in selected if t not in accumulated_ins]
        if not new_selected:
            eprint("No new tubelets found. Stopping insertion stages early.")
            break
        accumulated_ins.update(new_selected)

        # -- Update Basline: Reveal newly selected tubelets
        mask = np.isin(tubelets, list(accumulated_ins))[..., np.newaxis]
        current_baseline_ins = np.where(mask, video_array, baseline_ins)


    # -- Deletion Optimization
    eprint(f"\n--- Starting Iterative Deletion Optimization ({args.stages} Stages) ---")
    current_video_del = video_array.copy()
    for stage in range(1, args.stages + 1):
        eprint(f"\n>> [Deletion Stage {stage}/{args.stages}]")
        #Update baseline (hides previously found tubelets)
        frames_current_video = [Image.fromarray(f.astype(np.uint8)) for f in current_video_del]
        frames_del_base = [Image.fromarray(f.astype(np.uint8)) for f in baseline_del]

        # -- Main Weight Optimization Loop
        selected, scores = optimize_tubelet_weights(
            args, model, tokenizer, processor, full_ids, output_ids, frames_current_video, frames_del_base, 
            tubelets, positions, mode='deletion', stage=f"{index}-{stage}-deletion")

        # -- Evaluation. Record highest score a tubelet achieved across all stages
        for t, s in scores.items():
            final_scores_del[t] = max(final_scores_del.get(t, 0), s)
        # -- Filtering & Recording Inside Set
        new_selected = [t for t in selected if t not in accumulated_del]
        if not new_selected:
            eprint("No new tubelets found. Stopping deletion stages early.")
            break
        accumulated_del.update(new_selected)

        # -- Update the target video: mask the newly selected tubelets
        mask = np.isin(tubelets, list(accumulated_del))[..., np.newaxis]
        current_video_del = np.where(mask, baseline_del, video_array)

    return list(accumulated_ins), list(accumulated_del), final_scores_ins, final_scores_del



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