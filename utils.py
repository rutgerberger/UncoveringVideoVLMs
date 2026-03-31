import os
import torch
import torch.nn.functional as F
import cv2
cv2.setNumThreads(0)
import gc

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

import concurrent.futures

DEFAULT_VIDEO_TOKEN = "<video>"

def _run_slic_isolated(video_down_float, n_segments, compactness):
    """
    This runs in a completely separate OS process. 
    PyTorch locks cannot reach here.
    """
    # Force single thread in the child process just to be absolutely safe
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    
    from skimage.segmentation import slic
    
    return slic(
        video_down_float, 
        n_segments=n_segments,
        compactness=compactness, 
        channel_axis=-1, 
        spacing=[2, 1, 1],
        max_num_iter=4,              
        enforce_connectivity=False   
    )

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
    #return np.zeros_like(video_array)
    return np.full_like(video_array, 255) # Constant white video

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
    num_frames = args.num_frames
    start_sec = 0
    end_sec = 1
    
    # --- GLOBAL CAP --- 
    # 384x384 keeps detail but strictly limits tokens to prevent OOM
    max_dim = 384 

    # -- ImageNet Data Handling
    if getattr(args, 'dataset', '') == 'imagenet':
        question_text = "If you were a classifier trained on ImageNet, what object is in this video? Answer with only the class name."
        cur_prompt = question_text
        options_prompt = ""
        qs = cur_prompt
        correct_idx = row['label_name'].split(',')[0] 
        img = row['image'].convert('RGB')
        
        # Cap ImageNet resolution
        img.thumbnail((max_dim, max_dim))
        video = [img for _ in range(num_frames)]
        
        eprint("================================")
        eprint(f"Question: {question_text}")
        eprint(f"Ground Truth: {correct_idx}")
        eprint("================================")
        return video, qs, cur_prompt, correct_idx
        
    # -- It was not ImageNet. TGIF / SIMPLE dataset.
    if getattr(args, 'dataset', '') == 'TGIF' or getattr(args, 'dataset', '') == 'simple':
        video_filename = f"{row['video_name']}.mp4" 
        if getattr(args, 'dataset', '') == 'TGIF':
            video_path = os.path.join(args.video_folder, "mp4", video_filename)
        else:
            video_path = os.path.join(args.video_folder, video_filename)
        question_text = row['question']
        cur_prompt = question_text
        options_prompt = "N/A (Open-ended)"
        qs = cur_prompt + "\nAnswer with as few words as possible."
        correct_idx = row['answer']
        apply_slice = False
        
    elif getattr(args, 'dataset', '') == 'k400':
        yt_id = row['youtube_id']
        start_time = str(row['time_start']).zfill(6)
        end_time = str(row['time_end']).zfill(6)
        video_filename = f"{yt_id}_{start_time}_{end_time}.mp4"
        video_path = os.path.join(args.video_folder, "val", video_filename)
        question_text = "What type of activity is happening in this video? Answer with only the action class name."
        cur_prompt = question_text
        options_prompt = "N/A (Open-ended)"
        qs = cur_prompt
        correct_idx = row['label']
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

    video = []
    
    if total_frames > 0 and end_idx > start_idx:
        indices = np.linspace(start_idx, end_idx - 1, num_frames).astype(int)
        batch_data = vr.get_batch(indices).asnumpy()
        
        # --- ROOT CAUSE FIX: Cap Video Frame Resolution ---
        for frame in batch_data:
            img = Image.fromarray(frame)
            img.thumbnail((max_dim, max_dim))
            video.append(img)
            
    if getattr(args, 'random_shuffle', False):
        random.shuffle(video)
        
    eprint("================================\n")
    eprint(f"Question: {question_text}")
    eprint(f"Options: {options_prompt}")
    eprint(f"Ground Truth: {correct_idx}")
    eprint("\n================================")
    
    return video, qs, cur_prompt, correct_idx

# -- VLM utils

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
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt", max_pixels=112896)
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
        inputs = processor(text=[" "], videos=[frames], padding=True, return_tensors="pt", max_pixels=112896)
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
        inputs = processor(text=[" "], videos=[frames], padding=True, return_tensors="pt", max_pixels=112896)
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
    
def generate_tubelets_optimized(video, args, downsample_factor=0.5):
    video_array = np.stack([np.array(img) for img in video]) 
    T, H, W, C = video_array.shape
    
    if getattr(args, 'use_slic', True):
        # 1. CPU Downsampling via OpenCV
        if downsample_factor < 1.0:
            down_H, down_W = int(H * downsample_factor), int(W * downsample_factor)
            video_down = np.stack([
                cv2.resize(frame, (down_W, down_H), interpolation=cv2.INTER_LINEAR) 
                for frame in video_array
            ])
        else:
            video_down = video_array.copy()

        # Cast to float32 [0, 1] to keep the memory footprint tiny
        video_down_float = (video_down.astype(np.float32) / 255.0)

        # We perform SLIC clustering in an isolated process (otherwise this would lead into deadlocks)
        n_seg = getattr(args, 'n_segments', 120)
        comp = getattr(args, 'compactness', 10)
        with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
            # Submit the job to the isolated process
            future = executor.submit(_run_slic_isolated, video_down_float, n_seg, comp)
            # This will wait for the child process to return the result
            tubelet_labels_down = future.result() 
        
        # 3. CPU Upsampling via OpenCV
        if downsample_factor < 1.0:
            labels_uint16 = tubelet_labels_down.astype(np.uint16)
            tubelet_labels = np.stack([
                cv2.resize(label_frame, (W, H), interpolation=cv2.INTER_NEAREST) 
                for label_frame in labels_uint16
            ]).astype(int)
        else:
            tubelet_labels = tubelet_labels_down

    else:
        grid_size = 8
        y_indices = np.clip((np.arange(H) / (H / grid_size)).astype(int), 0, grid_size - 1)
        x_indices = np.clip((np.arange(W) / (W / grid_size)).astype(int), 0, grid_size - 1)
        grid_y, grid_x = np.meshgrid(y_indices, x_indices, indexing='ij')
        frame_labels = grid_y * grid_size + grid_x
        tubelet_labels = np.tile(frame_labels, (T, 1, 1))
        
    return video_array, tubelet_labels

# -- Visualization methods

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
        heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET )
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

def visualize_gradients(gradients, frames, tubelets, save_folder, step, title="IG_Step", alpha=0.6, blur_fraction=0.08):
    """
    Overlays tubelet gradients onto original frames using cv2/PIL for fast GIF generation.
    Blue = Negative (suppresses output), Red = Positive (supports output).
    Unimportant areas (zero gradient) remain the original grayscale frame.
    """
    #-- Map 1D tubelet gradients back to the 3D video space (T, H, W)
    if torch.is_tensor(gradients):
        gradients = gradients.detach().cpu().numpy()
    if torch.is_tensor(tubelets):
        tubelets = tubelets.detach().cpu().numpy()
        
    spatial_grads = gradients[tubelets].astype(np.float32)
    T, H, W = spatial_grads.shape
    
    #-- Determine symmetric limits for the diverging colormap
    vmax = np.max(np.abs(spatial_grads))
    if vmax == 0: vmax = 1e-9 # Prevent division by zero
    
    #-- Dynamic blur kernel size based on frame resolution
    k_size = int(min(H, W) * blur_fraction)
    k_size = k_size + 1 if k_size % 2 == 0 else k_size 
    k_size = max(5, k_size) 

    visualized_video = []
    cmap = plt.get_cmap('coolwarm') # Still the easiest way to get Blue-White-Red

    for t in range(T):
        #-- Extract and format the base image
        image = frames[t]
        if torch.is_tensor(image):
            image = image.detach().cpu().numpy()
            if image.ndim == 3 and image.shape[0] == 3: # If channels first (C, H, W)
                image = np.transpose(image, (1, 2, 0))
        else:
            # ---> FIX: Convert PIL Image (or other types) to NumPy array <---
            image = np.array(image)
                
        # Ensure image is in 0-255 float range for blending
        if image.max() <= 1.0:
            image = image * 255.0
        image = image.astype(np.float32)
        
        #-- Convert base image to 3-channel grayscale so the Red/Blue heatmap pops
        # Check if grayscale already (shape (H, W)) to prevent index errors
        if image.ndim == 2:
            image = image[..., np.newaxis]
            
        gray_image = np.mean(image, axis=-1, keepdims=True)
        gray_image = np.repeat(gray_image, 3, axis=-1)
        
        #-- Process the gradient mask
        grad_frame = spatial_grads[t]
        
        # Blur the raw gradients to smooth the blocky tubelets
        blurred_grad = cv2.GaussianBlur(grad_frame, (k_size, k_size), 0)
        
        # Calculate absolute magnitude for the alpha mask (0.0 to 1.0)
        magnitude = np.abs(blurred_grad) / vmax
        magnitude = np.clip(magnitude, 0, 1)
        
        # Normalize gradients to [0, 1] for the colormap (-vmax -> 0.0, 0 -> 0.5, +vmax -> 1.0)
        norm_grad = (blurred_grad / vmax + 1.0) / 2.0
        norm_grad = np.clip(norm_grad, 0, 1)
        
        # Apply colormap to get RGB overlay (drop the alpha channel from cmap)
        heatmap_rgba = cmap(norm_grad)
        heatmap_rgb = (heatmap_rgba[..., :3] * 255.0).astype(np.float32)
        
        #-- Perform dynamic alpha blending
        # Where magnitude is 0, alpha is 0 (100% original frame)
        # Where magnitude is 1, alpha is `alpha` (e.g., 60% heatmap, 40% frame)
        dynamic_alpha = magnitude[..., np.newaxis] * alpha
        overlay = (1.0 - dynamic_alpha) * gray_image + dynamic_alpha * heatmap_rgb
        
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        visualized_video.append(Image.fromarray(overlay))

    # --- SAVE AS GIF ---
    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, f"{title}_step_{step:03d}.gif")
    
    visualized_video[0].save(
        save_path, 
        save_all=True, 
        append_images=visualized_video[1:], 
        duration=250, # 4 FPS 
        loop=0
    )

# -- Keyword finder (based on 'Where do VLMs look at')

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


# -- AUC Curves
def _compute_pixel_ranks(T, H, W, tubelets, selected_tubes, seed):
    """AUC helper, computes pixel importance and adds smooth noise for AUC generation."""
    pixel_ranks = np.zeros((T, H, W), dtype=np.float32)
    num_selected = len(selected_tubes)
    
    for i, t_id in enumerate(selected_tubes):
        pixel_ranks[tubelets == t_id] = num_selected - i
    
    np.random.seed(seed)
    tubelet_noise = np.random.rand(int(tubelets.max()) + 1) * 0.5  
    pixel_ranks += tubelet_noise[tubelets]
    sorted_ranks = np.sort(pixel_ranks, axis=None)[::-1]
    
    return pixel_ranks, sorted_ranks

def _fast_tensor_forward(model, forward_kwargs, blended_pixels, output_ids, positions):
    """AUC helper, runs optimized forward pass on the GPU."""
    with torch.no_grad():
        outputs = model(**forward_kwargs, pixel_values_videos=blended_pixels)
        logits = outputs.logits[:, -output_ids.shape[-1] - 1 : -1, :]
        probs = F.softmax(logits, dim=-1).gather(dim=-1, index=output_ids.unsqueeze(-1)).squeeze(-1)
        
        if probs.dim() == 0:
            probs = probs.unsqueeze(0)
        if positions is not None and len(positions) > 0:
            probs = probs[0, positions]
            
        prob_val = probs.mean().item()
        
        del outputs, logits # Aggressive cache clearing
        return prob_val

def _plot_and_save_auc(percentages, ins_curve, del_curve, auc_ins, auc_del, prob_orig, prob_blur, output_dir, ivd):
    """Handles matplotlib generation and disk I/O."""
    plt.figure(figsize=(8, 6))
    plt.plot(percentages, ins_curve, label=f'Insertion (AUC={auc_ins:.3f})', color='green', marker='o')
    plt.plot(percentages, del_curve, label=f'Deletion (AUC={auc_del:.3f})', color='red', marker='x')
    plt.axhline(y=prob_orig, color='gray', linestyle='--', label='Original Prob')
    plt.axhline(y=prob_blur, color='blue', linestyle='--', label='Blurred Prob')
    
    plt.title("Insertion / Deletion Curves")
    plt.xlabel("Fraction of Total Video Pixels Revealed/Masked")
    plt.ylabel("Target Probability")
    plt.legend()
    plt.grid(True)
    
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"{ivd}_auc_curves.png"))
    plt.close()

def evaluate_auc(args, model, processor, full_ids, output_ids, frames, tubelets, selected_tubes, ivd=0, positions=None, num_steps=20):
    eprint("\n--- Starting AUC Evaluation (Pixel-wise) ---")
    video_array = np.stack([np.array(img) for img in frames])
    T, H, W, _ = video_array.shape
    num_pixels = T * H * W
    
    # Get baselines & calculate starting bounds
    baseline_ins = get_baseline_insertion(args, video_array)
    baseline_del = get_baseline_deletion(args, video_array)
    frames_blur = [Image.fromarray(f) for f in baseline_ins]
    frames_del_base = [Image.fromarray(f) for f in baseline_del]
    
    prob_orig = get_prob(args, model, processor, full_ids, output_ids, frames, positions)
    prob_blur = get_prob(args, model, processor, full_ids, output_ids, frames_blur, positions)
    
    # Compute pixel ranks and initialize curves
    pixel_ranks, sorted_ranks = _compute_pixel_ranks(T, H, W, tubelets, selected_tubes, args.manual_seed)
    ins_curve, del_curve, percentages = [prob_blur], [prob_orig], [0.0]
    
    # Pre-process Tensors to GPU
    is_qwen = getattr(args, 'model', '') == 'qwen'
    
    if is_qwen:
        orig_inputs = processor(text=[" "], videos=[frames], padding=True, return_tensors="pt", max_pixels=112896)
        blur_inputs = processor(text=[" "], videos=[frames_blur], padding=True, return_tensors="pt", max_pixels=112896)
        del_inputs = processor(text=[" "], videos=[frames_del_base], padding=True, return_tensors="pt", max_pixels=112896)
    else:
        orig_inputs = processor(text=" ", videos=frames, return_tensors="pt")
        blur_inputs = processor(text=" ", videos=frames_blur, return_tensors="pt")
        del_inputs = processor(text=" ", videos=frames_del_base, return_tensors="pt")
        
    pixels_orig = orig_inputs['pixel_values_videos'].to(model.device, dtype=model.dtype).detach()
    pixels_blur = blur_inputs['pixel_values_videos'].to(model.device, dtype=model.dtype).detach()
    pixels_del_base = del_inputs['pixel_values_videos'].to(model.device, dtype=model.dtype).detach()
    
    # --- MODEL AGNOSTIC SHAPE EXTRACTION ---
    target_shape = pixels_orig.shape
    if is_qwen:
        # Extract temporal and spatial grid from Qwen's metadata
        grid_thw = orig_inputs['video_grid_thw'][0]
        target_T, target_H, target_W = grid_thw[0].item(), grid_thw[1].item(), grid_thw[2].item()
    else:
        # Fallback to standard Video-LLaVA 5D extraction
        if pixels_orig.dim() == 4:
            pixels_orig = pixels_orig.unsqueeze(0)
            pixels_blur = pixels_blur.unsqueeze(0)
            pixels_del_base = pixels_del_base.unsqueeze(0)
            target_shape = pixels_orig.shape
            
        t_dim_index, target_H, target_W = (2, target_shape[3], target_shape[4]) if target_shape[1] == 3 else (1, target_shape[3], target_shape[4])
        
    forward_kwargs = {"input_ids": full_ids, "attention_mask": torch.ones_like(full_ids).to(model.device), "use_cache": True}
    if 'video_grid_thw' in orig_inputs:
        forward_kwargs['video_grid_thw'] = orig_inputs['video_grid_thw'].to(model.device)
        
    torch.cuda.empty_cache()
    gc.collect()
    
    # Evaluation Loop (Optimized)
    for step in range(1, num_steps + 1):
        now = time.time()
        eprint(f"step {step}/{num_steps}")
        fraction = step / num_steps
        
        idx = max(0, min(int(fraction * num_pixels) - 1, num_pixels - 1))
        current_mask = pixel_ranks >= sorted_ranks[idx]
        
        # --- MODEL AGNOSTIC MASK INTERPOLATION ---
        mask_tensor = torch.tensor(current_mask, dtype=torch.float32, device=model.device) # Shape: (T, H, W)
        
        if is_qwen:
            # Format as 5D volume for trilinear interpolation: (1, 1, T, H, W)
            mask_vol = mask_tensor.unsqueeze(0).unsqueeze(0)
            mask_low_res_vol = F.interpolate(mask_vol, size=(target_T, target_H, target_W), mode='nearest')
            # Flatten to 1D sequence to broadcast against Qwen's patch sequence
            mask_low_res = mask_low_res_vol.view(-1, 1)
        else:
            # Format as (T, 1, H, W) for standard spatial interpolation
            mask_spatial = mask_tensor.unsqueeze(1)
            mask_low_res_base = F.interpolate(mask_spatial, size=(target_H, target_W), mode='nearest')
            mask_low_res = mask_low_res_base.unsqueeze(0) if t_dim_index == 1 else mask_low_res_base.permute(1, 0, 2, 3).unsqueeze(0)

        # GPU Blending
        pixels_ins = pixels_orig * mask_low_res + pixels_blur * (1.0 - mask_low_res)
        pixels_del = pixels_del_base * mask_low_res + pixels_orig * (1.0 - mask_low_res)
        
        # GPU Inference
        ins_curve.append(_fast_tensor_forward(model, forward_kwargs, pixels_ins, output_ids, positions))
        del_curve.append(_fast_tensor_forward(model, forward_kwargs, pixels_del, output_ids, positions))
        percentages.append(fraction)
        eprint(f"Done in {time.time() - now:.2f}s")

        # Cleanup
        del mask_tensor, mask_low_res
        del pixels_ins, pixels_del
        if is_qwen:
            del mask_vol, mask_low_res_vol
        else:
            del mask_spatial, mask_low_res_base

    # Final Metrics & Plotting
    auc_ins, auc_del = auc(percentages, ins_curve), auc(percentages, del_curve)
    eprint(f"Final Pixel AUC - Insertion: {auc_ins:.4f} | Deletion: {auc_del:.4f}")
    _plot_and_save_auc(percentages, ins_curve, del_curve, auc_ins, auc_del, prob_orig, prob_blur, args.output_dir, ivd)
    
    del pixels_orig, pixels_blur, pixels_del_base
    del orig_inputs, blur_inputs, del_inputs
    torch.cuda.empty_cache()

    return auc_ins, auc_del

# -- For iGOS++

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