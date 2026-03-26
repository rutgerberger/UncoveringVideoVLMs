import torch
import torch.nn.functional as F
import os
import cv2
import time
import sys
import requests
import random
import heapq
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
from scipy.interpolate import interp1d
from scipy.ndimage import center_of_mass
from skimage.segmentation import slic, mark_boundaries
from sklearn.metrics import auc
from decord import VideoReader, cpu
from PIL import Image
from textwrap import fill
import torch.utils.checkpoint as checkpoint
from io import BytesIO

from qwen_vl_utils import process_vision_info

DEFAULT_VIDEO_TOKEN = "<video>"
NUM_FRAMES = 32

def eprint(*args, **kwargs):
    """Helper to log to stderr."""
    sep = ' '
    combined_text = sep.join(str(arg) for arg in args)
    wrapped_lines = []
    for line in combined_text.splitlines():
        if line.strip() == "":
            wrapped_lines.append("")
        else:
            wrapped_lines.append(fill(line, width=80))
    final_text = "\n".join(wrapped_lines)
    print(final_text, file=sys.stderr, **kwargs)

def timestamp_to_sec(timestamp):
    h, m, s = timestamp.split(':')
    return float(h) * 3600 + float(m) * 60 + float(s)

def match_keywords(output_list, kw_ids):
    """Helper to find the sublist kw_ids within output_list and return the indices."""
    kw_len = len(kw_ids)
    for i in range(len(output_list) - kw_len + 1):
        if output_list[i : i + kw_len] == kw_ids:
            return list(range(i, i + kw_len))
    return []


def precompute_blurred_video(video_array, kernel_fraction=0.3, passes=3):
    """
    Precomputes an EXTREMELY blurred version of the video.
    For simple animations, solid colors bleed through easily, so we use
    a dynamic massive kernel, explicit high sigma, and multiple blur passes.
    """
    T, H, W, C = video_array.shape
    k_size = int(min(H, W) * kernel_fraction)
    k_size = k_size + 1 if k_size % 2 == 0 else k_size  # Must be odd
    k_size = max(51, k_size)  # Fallback to at least 51x51
    sigma = k_size / 2.0 
    
    blurred_video = np.empty_like(video_array)
    for i in range(T):
        frame = video_array[i]
        #Multiple Passes for total detail destruction
        for _ in range(passes):
            frame = cv2.GaussianBlur(frame, (k_size, k_size), sigmaX=sigma, sigmaY=sigma)
        blurred_video[i] = frame
    return blurred_video


def get_baseline_insertion(args, video_array):
    """Returns the background canvas used for Insertion metrics (Blur or White)."""
    if args.insertion_mask_type == 'blur':
        return precompute_blurred_video(video_array)
    else:
        return np.full_like(video_array, 255) # White canvas

def get_baseline_deletion(args, video_array):
    """Returns the masking canvas used for Deletion metrics (White)."""
    return np.full_like(video_array, 255)

def apply_universal_mask(foreground_array, background_array, tubelets, active_tubes):
    """
    Universal blender.
    Where tubelets are 'active', shows foreground_array.
    Where tubelets are NOT 'active', shows background_array.
    Returns a list of PIL Images ready for the VLM.
    
    Insertion:
    - Active tubelets: original video, else: blurred
    Deletion:
    - Active tubelets: masked out, else: original video
    """
    mask = np.isin(tubelets, active_tubes)[..., np.newaxis]
    masked_array = np.where(mask, foreground_array, background_array).astype(np.uint8)
    return [Image.fromarray(f) for f in masked_array]


def get_data(args, row):
    is_tgif = getattr(args, 'dataset', '') == 'TGIF'
    is_imagenet = getattr(args, 'dataset', '') == 'imagenet'
    start_sec = 0
    end_sec = 1
    # -- ImageNet Data Handling
    if is_imagenet:
        question_text = "If you were a classifier trained on ImageNet, what object is in this video? Answer with only the class name."
        cur_prompt = question_text
        options_prompt = ""
        qs = cur_prompt
        correct_idx = row['label_name'].split(',')[0] 
        img = row['image'].convert('RGB')
        video = [img for _ in range(NUM_FRAMES)]
        eprint("================================")
        eprint(f"Question: {question_text}")
        eprint(f"Ground Truth: {correct_idx}")
        eprint("================================")
        return video, qs, cur_prompt, correct_idx
    # -- It was not ImageNet. TGIF / SIMPLE dataset.
    if is_tgif or getattr(args, 'dataset', '') == 'simple':
        video_filename = f"{row['video_name']}.mp4" 
        if is_tgif:
            video_path = os.path.join(args.video_folder, "mp4", video_filename)
        else:
            video_path = os.path.join(args.video_folder, video_filename)
        question_text = row['question']
        cur_prompt = question_text
        options_prompt = "N/A (Open-ended)"
        qs = cur_prompt + "\nAnswer with as few words as possible."
        correct_idx = row['answer']
        apply_slice = False
    # -- HD-EPIC handling
    else:
        video_input = row['inputs']['video 1']
        video_id = video_input['id']
        participant_id = video_id.split('-')[0]
        video_filename = f"{video_id}.mp4" 
        video_path = os.path.join(args.video_folder, participant_id, video_filename)
        question_text = row['question']
        choices_list = row['choices']
        correct_idx = row['correct_idx']
        options_prompt = ""
        for i, option in enumerate(choices_list):
            option_str = ", ".join(option) if isinstance(option, list) else str(option)
            options_prompt += f"\n{i}. {option_str}"
        cur_prompt = f"{question_text}{options_prompt}"
        qs = cur_prompt + "\nAnswer with only the INDEX of the correct answer."
        apply_slice = args.apply_slice
        if apply_slice:
            start_sec = timestamp_to_sec(video_input['start_time'])
            end_sec = timestamp_to_sec(video_input['end_time'])

    vr = VideoReader(video_path, ctx=cpu(0))
    fps = vr.get_avg_fps()
    total_frames = len(vr)
    if apply_slice:
        start_idx = int(round(start_sec * fps))
        end_idx = int(round(end_sec * fps))
        end_idx = min(end_idx, total_frames)
    else:
        start_idx = 0
        end_idx = total_frames - 1
    num_frames = NUM_FRAMES 
    video = []
    if total_frames > 0 and end_idx > start_idx:
        indices = np.linspace(start_idx, end_idx - 1, num_frames).astype(int)
        batch_data = vr.get_batch(indices).asnumpy()
        video = [Image.fromarray(frame) for frame in batch_data]
    if getattr(args, 'random_shuffle', False):
        random.shuffle(video)
    eprint("================================\n")
    eprint(f"Question: {question_text}")
    eprint(f"Options: {options_prompt}")
    eprint(f"Ground Truth: {correct_idx}")
    eprint("\n================================")
    return video, qs, cur_prompt, correct_idx
    
def generate_tubelets(video, args):
    video_array = np.stack([np.array(img) for img in video]) 
    T, H, W, C = video_array.shape
    if getattr(args, 'use_slic', True):
        tubelet_labels = slic(
            video_array, n_segments=300, compactness=10, channel_axis=-1, spacing=[2, 1, 1]
        )
    else:
        grid_size = 8
        y_indices = np.clip((np.arange(H) / (H / grid_size)).astype(int), 0, grid_size - 1)
        x_indices = np.clip((np.arange(W) / (W / grid_size)).astype(int), 0, grid_size - 1)
        grid_y, grid_x = np.meshgrid(y_indices, x_indices, indexing='ij')
        frame_labels = grid_y * grid_size + grid_x
        tubelet_labels = np.tile(frame_labels, (T, 1, 1))
    return video_array, tubelet_labels

def generate_tubelets_optimized(video, args, downsample_factor=0.5):
    """
    AI-generated function to optimize the SLIC clustering algorithm.
    """
    video_array = np.stack([np.array(img) for img in video]) 
    T, H, W, C = video_array.shape
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if getattr(args, 'use_slic', True):
        # Downsampling: move to GPU, reshape to [Batch, Channels, Depth(T), Height, Width]
        video_tensor = torch.from_numpy(video_array).permute(3, 0, 1, 2).unsqueeze(0).float().to(device)
        down_H, down_W = int(H * downsample_factor), int(W * downsample_factor)
        video_down = F.interpolate(video_tensor, size=(T, down_H, down_W), mode='trilinear', align_corners=False)
        # And then retrieve back to CPU for skimage
        video_for_slic = video_down.squeeze(0).permute(1, 2, 3, 0).cpu().numpy().astype(video_array.dtype)
        # SLIC: fast because the volume is tiny and connectivity is off
        tubelet_labels_down = slic(
            video_for_slic, 
            n_segments=120,              # Note: Within SWAG-V, it was found 120 is optimal for video
            compactness=10, 
            channel_axis=-1, 
            spacing=[2, 1, 1],
            max_num_iter=4,              # Relaxed convergence
            enforce_connectivity=False   # Bypasses the slowest CPU step
        )
        # Upsampling on GPU: push labels back to GPU - [Batch, Channels, Depth, Height, Width]
        labels_tensor = torch.from_numpy(tubelet_labels_down).unsqueeze(0).unsqueeze(0).float().to(device)
        # We use 'nearest' so we don't blend integer class IDs (e.g., tubelet 1 and 3 don't become tubelet 2)
        labels_up = F.interpolate(labels_tensor, size=(T, H, W), mode='nearest')
        tubelet_labels = labels_up.squeeze().cpu().numpy().astype(int)
    else:
        # Fallback grid logic
        grid_size = 8
        y_indices = np.clip((np.arange(H) / (H / grid_size)).astype(int), 0, grid_size - 1)
        x_indices = np.clip((np.arange(W) / (W / grid_size)).astype(int), 0, grid_size - 1)
        grid_y, grid_x = np.meshgrid(y_indices, x_indices, indexing='ij')
        frame_labels = grid_y * grid_size + grid_x
        tubelet_labels = np.tile(frame_labels, (T, 1, 1))
    return video_array, tubelet_labels

def generate_qwen(args, model, processor, prompt: str, frames):
    from qwen_vl_utils import process_vision_info
    messages = [
        {"role": "user", "content": [
            {"type": "video", "video": frames, "fps": 1.0},
            {"type": "text", "text": prompt},
        ]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to(model.device)
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=getattr(args, 'max_new_tokens', 128))
    input_length = inputs.input_ids.shape[1]
    output_ids = generated_ids[:, input_length:]
    output_text = processor.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return inputs.input_ids, output_ids, output_text[0]

def generate(args, model, tokenizer, inputs):
    input_ids = inputs['input_ids'].to(model.device)
    pixel_values_videos = inputs['pixel_values_videos'].to(model.device, dtype=torch.float16)
    output_ids = model.generate(
        input_ids, pixel_values_videos=pixel_values_videos,
        do_sample=True if args.temperature > 0 else False,
        temperature=args.temperature, top_p=args.top_p,
        num_beams=args.num_beams, max_new_tokens=args.max_new_tokens,
        use_cache=True)
    output_ids = output_ids[:, input_ids.shape[1]:]
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return input_ids, output_ids, output_text

def get_model_response(args, model, processor, tokenizer, prompt, frames):
    if args.model == 'qwen':
        return generate_qwen(args, model, processor, prompt, frames)
    else:
        inputs = processor(text=prompt, videos=frames, return_tensors='pt')
        return generate(args, model, tokenizer, inputs)

def get_token_probs(args, model, processor, full_ids, output_ids, frames):
    if args.model == 'qwen':
        inputs = processor(text=[" "], videos=[frames], padding=True, return_tensors="pt")
    else:
        inputs = processor(text=" ", videos=frames, return_tensors="pt")
        
    pixel_values = inputs['pixel_values_videos'].to(model.device, dtype=model.dtype)
    if pixel_values.dim() == 4:
        pixel_values = pixel_values.unsqueeze(0)
        
    forward_kwargs = {
        "input_ids": full_ids,
        "attention_mask": torch.ones_like(full_ids).to(model.device),
        "pixel_values_videos": pixel_values,
        "use_cache": True
    }
    if 'video_grid_thw' in inputs:
        forward_kwargs['video_grid_thw'] = inputs['video_grid_thw'].to(model.device)
        
    with torch.no_grad():
        outputs = model(**forward_kwargs)
        
    logits = outputs.logits 
    out_len = output_ids.shape[-1]
    target_logits = logits[:, -out_len - 1 : -1, :] 
    probs = F.softmax(target_logits, dim=-1) 
    target_probs = probs.gather(dim=-1, index=output_ids.unsqueeze(-1)).squeeze(-1)
    
    if target_probs.dim() == 0:
        target_probs = target_probs.unsqueeze(0)
    return target_probs.squeeze(0) 

def get_prob(args, model, processor, full_ids, output_ids, frames, positions=None):
    """Calculates mean probability of the output_ids; optionally filtered by positions."""
    if args.model == 'qwen':
        inputs = processor(text=[" "], videos=[frames], padding=True, return_tensors="pt")
    else:
        inputs = processor(text=" ", videos=frames, return_tensors="pt")
    pixel_values = inputs['pixel_values_videos'].to(model.device, dtype=model.dtype)
    if pixel_values.dim() == 4:
        pixel_values = pixel_values.unsqueeze(0)
    forward_kwargs = {
        "input_ids": full_ids,
        "attention_mask": torch.ones_like(full_ids).to(model.device),
        "pixel_values_videos": pixel_values,
        "use_cache": True
    }
    if 'video_grid_thw' in inputs:
        forward_kwargs['video_grid_thw'] = inputs['video_grid_thw'].to(model.device)
        
    with torch.no_grad():
        outputs = model(**forward_kwargs)
    logits = outputs.logits  
    out_len = output_ids.shape[-1] 
    target_logits = logits[:, -out_len - 1 : -1, :] 
    probs = F.softmax(target_logits, dim=-1) 
    target_probs = probs.gather(dim=-1, index=output_ids.unsqueeze(-1)).squeeze(-1)
    
    if target_probs.dim() == 0:
        target_probs = target_probs.unsqueeze(0)
    if positions is not None and len(positions) > 0:
        target_probs = target_probs[0, positions] 
        
    return target_probs.mean().item()

def create_description(args, model, processor, frames, tokenizer):
    if args.model == 'qwen':
        prompt = "Describe this video. Besides the scenery, explain what kind of actions you see."
        _, _, description = generate_qwen(args, model, processor, prompt, frames)
    else:
        prompt = f"USER: {DEFAULT_VIDEO_TOKEN}\nDescribe this video. Besides the scenery, explain what kind of actions you see. ASSISTANT:"
        inputs = processor(text=prompt, videos=frames, return_tensors='pt')
        _, _, description = generate(args, model, tokenizer, inputs)
    return description

# =========================
# Visualization Definitions
# =========================

def save_igos_heatmaps(video_array, mask_numpy, outdir, prefix, alpha=0.5):
    """
    Saves heatmaps and overlays for the continuous masks generated by iGOS.
    video_array: (T, H, W, C) numpy array of original frames (0-255)
    mask_numpy: (H, W) numpy array of the upscaled iGOS mask
    """
    os.makedirs(outdir, exist_ok=True)
    
    # Normalize the mask to 0-1 for the colormap
    mask_min, mask_max = mask_numpy.min(), mask_numpy.max()
    if mask_max - mask_min > 0:
        norm_mask = (mask_numpy - mask_min) / (mask_max - mask_min)
    else:
        norm_mask = mask_numpy

    # Create the heatmap (0-1 float32 scale)
    heatmap_uint8 = np.uint8(255 * norm_mask)
    heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    visualized_frames = []

    # Apply the heatmap to every frame in the video array
    for i, frame in enumerate(video_array):
        image_float = frame.astype(np.float32) / 255.0
        
        # Overlay: 50% heatmap + 50% image (matching your provided code)
        overlay = alpha * heatmap + (1.0 - alpha) * image_float
        overlay = np.clip(overlay, 0.0, 1.0)
        
        # Save individual frames (optional, but matches your original request)
        plt.imsave(os.path.join(outdir, f'{prefix}_frame{i}_heatmap.jpg'), heatmap)
        plt.imsave(os.path.join(outdir, f'{prefix}_frame{i}_overlay.jpg'), overlay)
        
        # Accumulate frames for a GIF
        overlay_uint8 = np.uint8(255 * overlay)
        visualized_frames.append(Image.fromarray(overlay_uint8))

    # Save as GIF for consistency with your other methods
    if visualized_frames:
        visualized_frames[0].save(
            os.path.join(outdir, f'{prefix}_overlay.gif'),
            save_all=True,
            append_images=visualized_frames[1:],
            duration=250,
            loop=0
        )

def visualize_spix(video_array, baseline_array, tubelets, selected_tubes, output_path):
    masked_frames = apply_universal_mask(video_array, baseline_array, tubelets, selected_tubes)
    masked_frames[0].save(
        output_path, save_all=True, append_images=masked_frames[1:], duration=250, loop=0
    )

def visualize_frames(frames, output_path):
    frames[0].save(
        output_path, save_all=True, append_images=frames[1:], duration=250, loop=0
    )

def visualize_tubelets(video_array, tubelet_labels, output_path):
    visualized_video = []
    for i in range(len(video_array)):
        frame = video_array[i]
        label_slice = tubelet_labels[i]
        boundary_img = mark_boundaries(frame, label_slice)
        boundary_img_uint8 = (boundary_img * 255).astype(np.uint8)
        visualized_video.append(Image.fromarray(boundary_img_uint8))
    visualized_video[0].save(
        output_path, save_all=True, append_images=visualized_video[1:], duration=250, loop=0 
    )
    print(f"Saved visualization to {output_path}")

def visualize_heatmap(video_array, tubelet_labels, tubelet_scores, output_path, alpha=0.7, blur_fraction=0.15, gamma=1.0):
    visualized_video = []
    max_id = tubelet_labels.max()
    score_map = np.zeros(max_id + 1, dtype=np.float32)
    
    # Clip negative scores to 0 (we only care about positive signal)
    for tid, score in tubelet_scores.items():
        score_map[tid] = max(0.0, score)
        
    # Normalize the scores so the maximum value is exactly 1.0
    mask_max = score_map.max()
    if mask_max > 0:
        score_map = score_map / mask_max
        
    # Optional: Apply gamma to boost mid-tones if needed
    #score_map = np.power(score_map, gamma)
    heatmap_mask = score_map[tubelet_labels]
    H, W = video_array[0].shape[:2]
    k_size = int(min(H, W) * blur_fraction)
    k_size = k_size + 1 if k_size % 2 == 0 else k_size 
    k_size = max(15, k_size) 

    for i in range(len(video_array)):
        image = video_array[i].astype(np.float32) # Ensure float for smooth blending
        mask_frame = heatmap_mask[i]
        # Blur the 0-1 mask
        blurred_mask = cv2.GaussianBlur(mask_frame, (k_size, k_size), 0)
        # Convert to 0-255 for the colormap
        heatmap_uint8 = np.uint8(255 * blurred_mask)
        heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32)
        # We scale the alpha by the blurred mask itself.
        # Where the mask is 0, dynamic_alpha is 0 (100% original image).
        # Where the mask is 1, dynamic_alpha is `alpha` (e.g., 60% heatmap, 40% image).
        dynamic_alpha = blurred_mask[..., np.newaxis] * alpha
        overlay = (1.0 - dynamic_alpha) * image + dynamic_alpha * heatmap
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        visualized_video.append(Image.fromarray(overlay))
        
    visualized_video[0].save(
        output_path, save_all=True, append_images=visualized_video[1:], duration=250, loop=0
    )
    print(f"Saved heatmap visualization to {output_path}")

def visualize_interaction_matrix(interaction_matrix, output_path):
    plt.figure(figsize=(10, 8))
    max_val = np.max(np.abs(interaction_matrix))
    if max_val == 0:
        max_val = 1.0 
    sns.heatmap(
        interaction_matrix, cmap="RdYlGn", center=0, vmin=-max_val, vmax=max_val,
        annot=True, fmt=".3f", linewidths=.5,
        cbar_kws={'label': 'Interaction Index (Red=Redundant (lim -2), Green=Synergistic (lim 2))'}
    )
    plt.title("Frame Pairwise Interactions (Shapley)")
    plt.xlabel("Frame Index J")
    plt.ylabel("Frame Index I")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def find_keywords(args, model, processor, input_ids, output_ids, frames, baseline_ins_frames, output_text, tokenizer=None, use_yake=False, special_ids=None):
    """
    Finds the words that change mostly when comparing to a baseline video.
    - baseline insertion: we are comparing to a blurred version preferably
    """
    if special_ids is None:
        special_ids = []
    seq_len = output_ids.shape[-1]
    
    if seq_len <= 4:
        clean_output_ids = output_ids
        if output_ids[0, -1] == tokenizer.eos_token_id:
            clean_output_ids = output_ids[:, :-1] 
            
        positions = list(range(clean_output_ids.shape[-1]))
        keywords = [tokenizer.decode(idx).strip() for idx in clean_output_ids[0]]
        valid_indices = [i for i, kw in enumerate(keywords) if kw]
        positions = [positions[i] for i in valid_indices]
        keywords = [keywords[i] for i in valid_indices]
        
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
                matched_pos = match_keywords(output_ids[0].tolist(), kw_ids)
                if matched_pos:
                    positions.extend(matched_pos)
                    keywords.extend([tokenizer.decode(output_ids[0][p]).strip() for p in matched_pos])
        else:
            full_prompt = torch.cat((input_ids, output_ids), dim=1)
            probs = get_token_probs(args, model, processor, full_prompt, output_ids, frames)
            probs_blur = get_token_probs(args, model, processor, full_prompt, output_ids, baseline_ins_frames)
            
            eps = 1e-7
            probs_safe = torch.clamp(probs, min=eps)
            probs_blur_safe = torch.clamp(probs_blur, min=eps)
            
            condition = (
                (torch.log(probs_safe) - torch.log(probs_blur_safe) > 1.0) & 
                (probs > 0.001) & 
                (~torch.isin(output_ids[0], torch.tensor(special_ids, device=probs.device)))
            )
            positions = torch.where(condition)[0].tolist()
            keywords = [tokenizer.decode(output_ids[0][idx]).strip() for idx in positions]
            
    return positions, keywords


## AUC CURVES ##
def save_curves(del_curve, ins_curve, index, auc_del, auc_ins, ivd, output_dir):
    """
    Plots and saves the Insertion and Deletion AUC curves.
    """
    # Extract baseline probabilities from the first index of the curves
    prob_orig = del_curve[0]
    prob_blur = ins_curve[0]

    plt.figure(figsize=(8, 6))
    
    # Plot curves
    plt.plot(index, ins_curve, label=f'Insertion (AUC={auc_ins:.3f})', color='green', marker='o')
    plt.plot(index, del_curve, label=f'Deletion (AUC={auc_del:.3f})', color='red', marker='x')
    
    # Add horizontal reference lines
    plt.axhline(y=prob_orig, color='gray', linestyle='--', label='Original Prob')
    plt.axhline(y=prob_blur, color='blue', linestyle='--', label='Blurred Prob')
    
    # Formatting
    plt.title("Insertion / Deletion Curves")
    plt.xlabel("Fraction of Total Video Pixels Revealed/Masked")
    plt.ylabel("Target Probability")
    plt.legend()
    plt.grid(True)
    
    # Save to directory
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f"{ivd}_auc_curves.png")
    plt.savefig(plot_path)
    plt.close()

def evaluate_auc(args, model, processor, full_ids, output_ids, frames, tubelets, selected_tubes, ivd=0, positions=None, num_steps=20):
    """
    To generate insertion and deletion curves, we gradually remove / delete pixels based on
    the weight assigned within the framework earlier on. Tubelets with high importance
    are removed earlier (greedy). Pixels within tubelets are randomly removed (smooth curves)
    """

    eprint("\n--- Starting AUC Evaluation (Pixel-wise) ---")
    video_array = np.stack([np.array(img) for img in frames])
    T, H, W, C = video_array.shape
    num_pixels = T * H * W
    
    #-- Get baseline videos to compare to
    baseline_ins = get_baseline_insertion(args, video_array)
    baseline_del = get_baseline_deletion(args, video_array)
    prob_orig = get_prob(args, model, processor, full_ids, output_ids, frames, positions)
    frames_blur = [Image.fromarray(f) for f in baseline_ins]
    prob_blur = get_prob(args, model, processor, full_ids, output_ids, frames_blur, positions)
    
    ins_curve = [prob_blur]
    del_curve = [prob_orig]
    percentages = [0.0]
    
    pixel_ranks = np.zeros((T, H, W), dtype=np.float32)
    num_selected = len(selected_tubes)
  
    #-- Assign base scores: best tubelet gets highest score, unselected stay at 0
    #   selected_tubes is already sorted based on saliency
    for i, t_id in enumerate(selected_tubes):
        pixel_ranks[tubelets == t_id] = num_selected - i
        
    # Because noise is < 1.0, it never violates the ordering BETWEEN tubelets,
    # but it randomly shuffles the pixels WITHIN the same tubelet.
    np.random.seed(args.manual_seed)
    tubelet_noise = np.random.rand(int(tubelets.max()) + 1) * 0.5  
    pixel_ranks += tubelet_noise[tubelets]
    sorted_ranks = np.sort(pixel_ranks, axis=None)[::-1] # Descending
    
    for step in range(1, num_steps + 1):
        fraction = step / num_steps
        # Find the rank threshold for this exact pixel percentage
        idx = int(fraction * num_pixels) - 1
        idx = max(0, min(idx, num_pixels - 1))
        thresh = sorted_ranks[idx]
        
        # Create boolean mask for the top N% of pixels
        current_mask = pixel_ranks >= thresh
        mask_expanded = current_mask[..., np.newaxis] # Shape: (T, H, W, 1)
        
        # Insertion: Top pixels show original video, rest show blur
        ins_array = np.where(mask_expanded, video_array, baseline_ins).astype(np.uint8)
        frames_ins = [Image.fromarray(f) for f in ins_array]
        p_ins = get_prob(args, model, processor, full_ids, output_ids, frames_ins, positions)
        ins_curve.append(p_ins)
        
        # Deletion: Top pixels are blacked out, rest show original video
        del_array = np.where(mask_expanded, baseline_del, video_array).astype(np.uint8)
        frames_del = [Image.fromarray(f) for f in del_array]
        p_del = get_prob(args, model, processor, full_ids, output_ids, frames_del, positions)
        del_curve.append(p_del)
        
        percentages.append(fraction)

    # Compute AUC
    auc_ins = auc(percentages, ins_curve)
    auc_del = auc(percentages, del_curve)
    eprint(f"Final Pixel AUC - Insertion: {auc_ins:.4f} | Deletion: {auc_del:.4f}")

    # Plot and Save
    plt.figure(figsize=(8, 6))
    plt.plot(percentages, ins_curve, label=f'Insertion (AUC={auc_ins:.3f})', color='green', marker='o')
    plt.plot(percentages, del_curve, label=f'Deletion (AUC={auc_del:.3f})', color='red', marker='x')
    plt.axhline(y=prob_orig, color='gray', linestyle='--', label='Original Prob')
    plt.axhline(y=prob_blur, color='blue', linestyle='--', label='Blurred Prob')
    
    plt.title(f"Insertion / Deletion Curves")
    plt.xlabel("Fraction of Total Video Pixels Revealed/Masked")
    plt.ylabel("Target Probability")
    plt.legend()
    plt.grid(True)
    
    os.makedirs(args.output_dir, exist_ok=True)
    plot_path = os.path.join(args.output_dir, f"{ivd}_auc_curves.png")
    plt.savefig(plot_path)
    plt.close()
    
    return auc_ins, auc_del

def evaluate_auc_pixel(args, model, processor, full_ids, output_ids, frames, continuous_mask, ivd=0, positions=None):
    eprint("\n--- Starting iGOS AUC Evaluation (Reference Match) ---")
    video_array = np.stack([np.array(img) for img in frames])
    T, H, W, C = video_array.shape
    num_pixels = T * H * W
    
    step = max(1, num_pixels // 50)
    
    baseline_ins = get_baseline_insertion(args, video_array)
    baseline_del = get_baseline_deletion(args, video_array)
    
    og_scores = get_prob(args, model, processor, full_ids, output_ids, frames, positions)
    frames_blur = [Image.fromarray(f) for f in baseline_ins]
    blur_scores = get_prob(args, model, processor, full_ids, output_ids, frames_blur, positions)
    
    # Curves track raw probabilities for the unnormalized AUC
    del_curve = [og_scores]
    ins_curve = [blur_scores]
    index = [0.0]
    
    # Broadcast and sort pixels
    pixel_ranks = np.tile(continuous_mask, (T, 1, 1))
    np.random.seed(args.manual_seed)
    pixel_ranks += np.random.rand(*pixel_ranks.shape) * 1e-5
    sorted_ranks = np.sort(pixel_ranks, axis=None)[::-1] 

    for pixels in range(0, num_pixels, step):
        # Calculate exactly how many pixels are evaluated in this step
        current_step_size = step if pixels + step < num_pixels else num_pixels - pixels
        
        idx = min(pixels + current_step_size - 1, num_pixels - 1)
        thresh = sorted_ranks[idx]
        
        current_mask = pixel_ranks >= thresh
        mask_expanded = current_mask[..., np.newaxis] 
        
        # --- Deletion ---
        del_array = np.where(mask_expanded, baseline_del, video_array).astype(np.uint8)
        frames_del = [Image.fromarray(f) for f in del_array]
        p_del = get_prob(args, model, processor, full_ids, output_ids, frames_del, positions)
        del_curve.append(p_del) 
        
        # --- Insertion ---
        ins_array = np.where(mask_expanded, video_array, baseline_ins).astype(np.uint8)
        frames_ins = [Image.fromarray(f) for f in ins_array]
        p_ins = get_prob(args, model, processor, full_ids, output_ids, frames_ins, positions)
        ins_curve.append(p_ins) 
        
        index.append((pixels + current_step_size) / num_pixels)

    # Compute Unnormalized AUC using sklearn (matching your first function perfectly)
    auc_del = auc(index, del_curve)
    auc_ins = auc(index, ins_curve)

    eprint(f"Final unnormalized AUC - Insertion: {auc_ins:.4f} | Deletion: {auc_del:.4f}")
    
    # Generate the visual curves
    save_curves(del_curve, ins_curve, index, auc_del, auc_ins, ivd, args.output_dir)
    
    return auc_ins, auc_del