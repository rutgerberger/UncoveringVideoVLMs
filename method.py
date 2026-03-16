from utils import *
from method_helpers import *

from PIL import Image
import numpy as np
import torch

import heapq
import random
import itertools

PROB_DELTA = 0.01

def spix_optimized(args, model, processor, input_ids, output_ids, frames, tubelets, positions=None):
    """
    Optimizes two separate masks (Insertion and Deletion) simultaneously.
    Shares the initial O(N) evaluation to save computation, then diverges 
    into two separate lazy greedy searches.
    """
    unique_tubes = np.unique(tubelets)
    full_ids = torch.cat((input_ids, output_ids), dim=1)
    video_array = np.stack([np.array(img) for img in frames])
    
    baseline_ins = get_baseline_insertion(args, video_array) 
    baseline_del = get_baseline_deletion(args, video_array)  
    frames_base = [Image.fromarray(f.astype(np.uint8)) for f in baseline_ins]
    
    prob_orig = get_prob(args, model, processor, full_ids, output_ids, frames, positions)
    prob_base = get_prob(args, model, processor, full_ids, output_ids, frames_base, positions)

    eprint("Precomputing tubelet centroids for distance penalty...")
    centroids_dict = precompute_tubelet_centroids(tubelets, unique_tubes)
    
    # -- Initial evaluation round
    eprint("Performing shared initial evaluation for Lazy Greedy...")
    queue_ins, queue_del = initialize_lazy_greedy_queues(
        args, model, processor, full_ids, output_ids, video_array, 
        tubelets, unique_tubes, baseline_ins, baseline_del, 
        prob_orig, prob_base, positions
    )

    # -- Insertion Optimization Loop
    eprint("\n--- Starting INSERTION Optimization ---")
    selected_ins, scores_ins = run_lazy_greedy_search(
        args, model, processor, full_ids, output_ids, video_array, 
        tubelets, unique_tubes, queue_ins, 'insertion', 
        baseline_ins, baseline_del, prob_orig, prob_base, 
        centroids_dict, positions
    )

    # -- Deletion Optimization Loop
    eprint("\n--- Starting DELETION Optimization ---")
    selected_del, scores_del = run_lazy_greedy_search(
        args, model, processor, full_ids, output_ids, video_array, 
        tubelets, unique_tubes, queue_del, 'deletion', 
        baseline_ins, baseline_del, prob_orig, prob_base, 
        centroids_dict, positions
    )

    return selected_ins, selected_del, scores_ins, scores_del


def spix_gradient(args, model, processor, input_ids, output_ids, frames, tubelets, positions=None):
    """
    Optimizes two separate masks (Insertion and Deletion) simultaneously 
    using Continuous Gradient Descent over Spatio-Temporal Tubelets.
    """
    video_array = np.stack([np.array(img) for img in frames])
    full_ids = torch.cat((input_ids, output_ids), dim=1)
    
    # -- Generate Baselines
    baseline_ins = get_baseline_insertion(args, video_array) 
    baseline_del = get_baseline_deletion(args, video_array)  
    
    # -- Convert numpy baselines to PIL for processor compatibility
    frames_ins_base = [Image.fromarray(f.astype(np.uint8)) for f in baseline_ins]
    frames_del_base = [Image.fromarray(f.astype(np.uint8)) for f in baseline_del]

    # -- Run Insertion Optimization
    eprint("\n--- Starting Continuous INSERTION Optimization ---")
    selected_ins, scores_ins = optimize_tubelet_weights(
        args, model, processor, full_ids, output_ids, frames, frames_ins_base, 
        tubelets, positions, mode='insertion', iterations=50, lr=0.05, L1_lambda=0.2
    )

    # -- Run Deletion Optimization
    eprint("\n--- Starting Continuous DELETION Optimization ---")
    selected_del, scores_del = optimize_tubelet_weights(
        args, model, processor, full_ids, output_ids, frames, frames_del_base, 
        tubelets, positions, mode='deletion', iterations=50, lr=0.05, L1_lambda=0.2
    )

    return selected_ins, selected_del, scores_ins, scores_del

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