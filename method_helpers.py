import numpy as np
import cv2
from scipy.ndimage import center_of_mass
import yake
import torch
import torch.nn.functional as F

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


def apply_mask_fast(video_array, tubelets, active_tubes):
    """Optimized masking returning a raw Numpy array directly."""
    mask = np.isin(tubelets, active_tubes)[..., np.newaxis]
    masked_array = (video_array * mask).astype(np.uint8)
    return masked_array


def apply_blur_mask_fast(video_array, blurred_video, tubelets, active_tubes):
    """
    Optimized masking returning a raw Numpy array directly.
    Selected tubelets show original pixels, unselected show blurred pixels.
    """
    # Create the boolean mask (T, H, W) and add the channel dimension (T, H, W, 1)
    mask = np.isin(tubelets, active_tubes)[..., np.newaxis]
    masked_array = np.where(mask, video_array, blurred_video)
    return masked_array.astype(np.uint8)

def precompute_blurred_video(video_array, kernel_size=(51, 51)):
    """
    Precomputes a heavily blurred version of the video to save time during the greedy search.
    Kernel size should be odd. Larger kernel = heavier blur.
    """
    blurred_video = np.empty_like(video_array)
    for i in range(len(video_array)):
        # Apply Gaussian blur. 0 means OpenCV automatically calculates standard deviation
        blurred_video[i] = cv2.GaussianBlur(video_array[i], kernel_size, 0)
    return blurred_video

def get_token_probs(args, model, processor, full_ids, output_ids, frames):
    """
    Calculates the token-wise probability of the target 'output_ids' given frames.
    Supports both Qwen2.5-VL and Video-LLaVA architectures.
    Replaces the old pred_probs function.
    """
    if args.model == 'qwen':
        inputs = processor(text=[" "], videos=[frames], padding=True, return_tensors="pt")
    else:
        inputs = processor(text=" ", videos=frames, return_tensors="pt")
        
    pixel_values = inputs['pixel_values_videos'].to(model.device, dtype=model.dtype)
    
    # Safety Check: Ensure 5D tensor (Batch, Time, Channels, Height, Width)
    if pixel_values.dim() == 4:
        pixel_values = pixel_values.unsqueeze(0)
        
    forward_kwargs = {
        "input_ids": full_ids,
        "attention_mask": torch.ones_like(full_ids).to(model.device),
        "pixel_values_videos": pixel_values,
        "use_cache": True
    }

    # Conditionally add Qwen's specific temporal grid tensor if it exists
    if 'video_grid_thw' in inputs:
        forward_kwargs['video_grid_thw'] = inputs['video_grid_thw'].to(model.device)
        
    with torch.no_grad():
        outputs = model(**forward_kwargs)
        
    logits = outputs.logits  # Shape: [batch_size, seq_len, vocab_size]
    out_len = output_ids.shape[-1]
    
    target_logits = logits[:, -out_len - 1 : -1, :] 
    probs = F.softmax(target_logits, dim=-1) 
    target_probs = probs.gather(dim=-1, index=output_ids.unsqueeze(-1)).squeeze(-1)
    
    if target_probs.dim() == 0:
        target_probs = target_probs.unsqueeze(0)
        
    return target_probs.squeeze(0) # Return 1D tensor of token probabilities

# NOTE: Added `processor` to the signature
def find_keywords(args, model, processor, input_ids, output_ids, frames, blur_frames, output_text, tokenizer=None, use_yake=False, special_ids=None):
    # Ensure special_ids is defined, default to empty list if not passed
    if special_ids is None:
        special_ids = []
    seq_len = output_ids.shape[-1]
    
    # short outputs (<= 4 tokens)
    if seq_len <= 4:
        # Clean the EOS token if it exists
        clean_output_ids = output_ids
        if output_ids[0, -1] == tokenizer.eos_token_id:
            clean_output_ids = output_ids[:, :-1] 
            
        # Map every valid token ID to its own position and string
        positions = list(range(clean_output_ids.shape[-1]))
        keywords = [tokenizer.decode(idx).strip() for idx in clean_output_ids[0]]
        # Filter out empty strings caused by special tokens
        valid_indices = [i for i, kw in enumerate(keywords) if kw]
        positions = [positions[i] for i in valid_indices]
        keywords = [keywords[i] for i in valid_indices]
        
    # long outputs (> 4 tokens)
    else: 
        if use_yake:
            import yake
            num_words = len(output_text.split())
            keywords_num = 3 if num_words <= 10 else num_words // 4
            kw_extractor = yake.KeywordExtractor(lan="en", n=2, dedupLim=0.2, top=keywords_num, features=None)
            extracted = kw_extractor.extract_keywords(output_text)
            kw_strings = [kw[0] for kw in extracted]
            positions = []
            keywords = []
            for kw in kw_strings:
                kw_ids = tokenizer.encode(kw, add_special_tokens=False)
                # Assuming match_keywords is a helper function defined elsewhere
                matched_pos = match_keywords(output_ids[0].tolist(), kw_ids)
                if matched_pos:
                    positions.extend(matched_pos)
                    # Use the actual decoded tokens to maintain 1:1 mapping
                    keywords.extend([tokenizer.decode(output_ids[0][p]).strip() for p in matched_pos])
                    
        else:
            # Full visual-grounding logic
            full_prompt = torch.cat((input_ids, output_ids), dim=1)
            
            # Using the new get_token_probs method
            probs = get_token_probs(args, model, processor, full_prompt, output_ids, frames)
            probs_blur = get_token_probs(args, model, processor, full_prompt, output_ids, blur_frames)
            
            # Avoid torch.log(0) by clamping probabilities to a tiny number
            eps = 1e-7
            probs_safe = torch.clamp(probs, min=eps)
            probs_blur_safe = torch.clamp(probs_blur, min=eps)
            
            # Condition 1: Log prob difference > 1.0 (Clear frames +-2.7x more likely)
            # Condition 2: Base prob must be > 0.001 (Prevents random noise from passing)
            # Condition 3: Not a special token
            condition = (
                (torch.log(probs_safe) - torch.log(probs_blur_safe) > 1.0) & 
                (probs > 0.001) & 
                (~torch.isin(output_ids[0], torch.tensor(special_ids, device=probs.device)))
            )
            positions = torch.where(condition)[0].tolist()
            keywords = [tokenizer.decode(output_ids[0][idx]).strip() for idx in positions]
            
    return positions, keywords