PROB_DELTA = 0.01

# -- For Lazy Greedy Search

def initialize_lazy_greedy_queues(args, model, processor, full_ids, output_ids, video_array, tubelets, unique_tubes, baseline_ins, baseline_del, prob_orig, prob_base, positions):
    queue_ins = []
    queue_del = []
    
    for tube in unique_tubes:
        # -- Gather probabilities for insertion
        frames_ins = apply_universal_mask(video_array, baseline_ins, tubelets, [tube])
        prob_ins = get_prob(args, model, processor, full_ids, output_ids, frames_ins, positions)
        
        # -- Gather probabilities for deletion
        keep_tubes = [t for t in unique_tubes if t != tube]
        frames_del = apply_universal_mask(video_array, baseline_del, tubelets, keep_tubes)
        prob_del = get_prob(args, model, processor, full_ids, output_ids, frames_del, positions)
        
        # -- Populate separate queues
        heapq.heappush(queue_ins, (-(prob_ins - prob_base), tube))
        heapq.heappush(queue_del, (-(prob_orig - prob_del), tube))
        
    return queue_ins, queue_del



def run_lazy_greedy_search(args, model, processor, full_ids, output_ids, 
        video_array, tubelets, unique_tubes, queue, mode, baseline_ins, 
        baseline_del, prob_orig, prob_base, centroids_dict, positions):
    """
    requires:
             args, model, processor: common sense
             full_ids, output_ids:   ids of full prompt and output
             video_array:            original frames
             tubelets, unique_tubes: set of tubelets (should be similar)
             queue:                  initialized heapq
             mode:                   'insertion' or 'deletion'
             baseline_ins/del:       baseline frames (blurred / constant)
             prob_orig:              list of probabilities (per token ID)
             prob_base:              same but then for blurred vid
             centroids_dict:         precomputed centroids for distance penalty
             positions:              keyword positions
    returns: 
             selected_tubes:         optimal set of tubelets which maximizes 
                                     the objective function, based on mode
    """
    selected_tubes = []
    tubelet_scores = {}
    current_total_score = 0.0 
    
    # -- Main Loop --
    for step in range(len(unique_tubes)):
        best_tube = None
        best_marginal_gain = -float('inf')
        best_absolute_score = -float('inf')
        prob_ins = 0.0
        prob_del = 0.0

        while queue:
            # -- Prerequisites: insertion / deletion probs
            _, candidate = heapq.heappop(queue)
            candidate_set = selected_tubes + [candidate]
            if mode == 'insertion':
                frames_ins = apply_universal_mask(video_array, baseline_ins, tubelets, candidate_set)
                prob_ins = get_prob(args, model, processor, full_ids, output_ids, frames_ins, positions)
            else:
                keep_tubes = [t for t in unique_tubes if t not in candidate_set]
                frames_del = apply_universal_mask(video_array, baseline_del, tubelets, keep_tubes)
                prob_del = get_prob(args, model, processor, full_ids, output_ids, frames_del, positions)

            # -- Optional: distance penalty
            dist_penalty = get_distance_penalty(candidate, selected_tubes, centroids_dict)
            
            # -- Score Reflection (Match Objective Function)
            if mode == 'insertion':
                actual_absolute_score = (prob_ins - prob_base) - (args.L1 * dist_penalty)
            else:
                actual_absolute_score = (prob_orig - prob_del) - (args.L1 * dist_penalty)
                
            actual_marginal_gain = actual_absolute_score - current_total_score
            
            # -- Queue is empty: done
            if not queue:
                best_tube = candidate
                best_marginal_gain = actual_marginal_gain
                best_absolute_score = actual_absolute_score
                break
                
            next_best_upper_bound_gain = -queue[0][0] - current_total_score

            # -- Calculated Gain is actually > Next Gain (Lazy Greedy): use this one
            if actual_marginal_gain >= next_best_upper_bound_gain:
                best_tube = candidate
                best_marginal_gain = actual_marginal_gain
                best_absolute_score = actual_absolute_score
                break
            # -- If not update the score
            else:
                heapq.heappush(queue, (-actual_absolute_score, candidate))
                
        tubelet_scores[best_tube] = max(0.0, best_marginal_gain)
        current_total_score = best_absolute_score
        selected_tubes.append(best_tube)
        
        # -- Logging and ratio calculation based on mode
        if mode == 'insertion':
            eprint(f"Ins Step {step+1}: Tube {best_tube} | Gain: {best_marginal_gain:.4f} | Total: {current_total_score:.4f} | P_ins: {prob_ins:.4f}")
            objective_ratio = prob_ins / max(prob_orig, 1e-5)
        else:
            eprint(f"Del Step {step+1}: Tube {best_tube} | Gain: {best_marginal_gain:.4f} | Total: {current_total_score:.4f} | P_orig-P_del: {(prob_orig - prob_del):.4f}")
            objective_ratio = (prob_orig - prob_del) / max(prob_orig, 1e-5)

        # -- Early Stopping
        if len(selected_tubes) > len(unique_tubes) * 0.25 or objective_ratio >= 0.90:
            if best_marginal_gain < 1e-4:
                eprint(f"Early stopping triggered at step {step+1}: Sufficient confidence reached and gain flatlined.")
                break
                
    return selected_tubes, tubelet_scores


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



# -- For Distance Penalties




def precompute_tubelet_centroids(tubelets, unique_tubes):
    """
    Calculates the normalized 3D centroid (t, y, x) for every tubelet.
    Returns a dictionary mapping tube ID -> numpy array of (t, y, x).
    """
    T, H, W = tubelets.shape
    # Calculate center of mass for all unique tubes at once
    # center_of_mass returns a list of (t, y, x) tuples
    centroids_list = center_of_mass(np.ones_like(tubelets), tubelets, unique_tubes)
    centroids_dict = {}
    for tube_id, (t, y, x) in zip(unique_tubes, centroids_list):
        # Normalize to [0, 1] so time and space scales don't distort the distance
        norm_t = t / T if T > 1 else 0.0
        norm_y = y / H
        norm_x = x / W
        centroids_dict[tube_id] = np.array([norm_t, norm_y, norm_x])
    return centroids_dict



def get_distance_penalty(candidate_tube, selected_tubes, centroids_dict):
    """
    Calculates the minimum distance from the candidate to the already selected group.
    Returns 0 if no tubes are selected yet.
    """
    if not selected_tubes:
        return 0.0 # First tubelet has no penalty
    candidate_pos = centroids_dict[candidate_tube]
    # Get positions of all currently selected tubes
    selected_positions = np.array([centroids_dict[t] for t in selected_tubes])
    # Calculate Euclidean distance from candidate to all selected tubes
    distances = np.linalg.norm(selected_positions - candidate_pos, axis=1)
    # We penalize based on the distance to the *closest* selected tube.
    # This encourages the candidate to "attach" to the existing blob.
    min_distance = np.min(distances)
    return min_distance