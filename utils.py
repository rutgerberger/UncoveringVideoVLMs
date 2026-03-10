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
NUM_FRAMES = 4

def eprint(*args, **kwargs):
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

def precompute_tubelet_centroids(tubelets, unique_tubes):
    T, H, W = tubelets.shape
    centroids_list = center_of_mass(np.ones_like(tubelets), tubelets, unique_tubes)
    centroids_dict = {}
    for tube_id, (t, y, x) in zip(unique_tubes, centroids_list):
        norm_t = t / T if T > 1 else 0.0
        norm_y = y / H
        norm_x = x / W
        centroids_dict[tube_id] = np.array([norm_t, norm_y, norm_x])
    return centroids_dict

def get_distance_penalty(candidate_tube, selected_tubes, centroids_dict):
    if not selected_tubes:
        return 0.0 
    candidate_pos = centroids_dict[candidate_tube]
    selected_positions = np.array([centroids_dict[t] for t in selected_tubes])
    distances = np.linalg.norm(selected_positions - candidate_pos, axis=1)
    min_distance = np.min(distances)
    return min_distance

def apply_mask_fast(video_array, tubelets, active_tubes):
    mask = np.isin(tubelets, active_tubes)[..., np.newaxis]
    masked_array = (video_array * mask).astype(np.uint8)
    return masked_array

def apply_blur_mask_fast(video_array, tubelets, active_tubes):
    blurred_video = precompute_blurred_video(video_array)
    mask = np.isin(tubelets, active_tubes)[..., np.newaxis]
    masked_array = np.where(mask, video_array, blurred_video)
    return masked_array.astype(np.uint8)

def apply_constant_mask(frames_array, tubelets, active_tubes):
    """Sets the unselected tubelets to black (0) for strict deletion metrics."""
    mask = np.isin(tubelets, active_tubes)[..., np.newaxis]
    masked_array = frames_array * mask
    return [Image.fromarray(frame.astype(np.uint8)) for frame in masked_array]

def apply_mask(frames, tubelets, active_tubes):
    video_array = np.stack([np.array(img) for img in frames])
    mask = np.isin(tubelets, active_tubes)[..., np.newaxis]
    masked_array = video_array * mask
    return [Image.fromarray(frame.astype(np.uint8)) for frame in masked_array]

def precompute_blurred_video(video_array, kernel_size=(51, 51)):
    blurred_video = np.empty_like(video_array)
    for i in range(len(video_array)):
        blurred_video[i] = cv2.GaussianBlur(video_array[i], kernel_size, 0)
    return blurred_video

def get_data(args, row):
    is_tgif = getattr(args, 'dataset', '') == 'TGIF'
    start_sec = 0
    end_sec = 1
    if is_tgif or getattr(args, 'dataset', '') == 'simple':
        video_filename = f"{row['video_name']}.mp4" 
        if is_tgif:
            video_path = os.path.join(args.video_folder, "mp4", video_filename)
        else:
            video_path = os.path.join(args.video_folder, video_filename)
        question_text = row['question']
        cur_prompt = question_text
        options_prompt = "N/A (Open-ended)"
        qs = cur_prompt + "\nAnswer concisely."
        correct_idx = row['answer']
        apply_slice = False
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
    """Calculates mean probability; optionally filtered by positions."""
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

def visualize_spix(frames, tubelets, selected_tubes, output_path):
    masked_frames = apply_mask(frames, tubelets, selected_tubes)
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

def visualize_heatmap(video_array, tubelet_labels, tubelet_scores, output_path, alpha=0.5, blur_fraction=0.15, gamma=1.0):
    visualized_video = []
    max_id = tubelet_labels.max()
    score_map = np.zeros(max_id + 1, dtype=np.float32)
    for tid, score in tubelet_scores.items():
        score_map[tid] = score
    active_scores = score_map[score_map > 0]
    
    heatmap_mask = score_map[tubelet_labels]
    mask_max = heatmap_mask.max()

    H, W = video_array[0].shape[:2]
    k_size = int(min(H, W) * blur_fraction)
    k_size = k_size + 1 if k_size % 2 == 0 else k_size 
    k_size = max(15, k_size) 

    for i in range(len(video_array)):
        image = video_array[i]
        mask_frame = heatmap_mask[i]
        blurred_mask = cv2.GaussianBlur(mask_frame, (k_size, k_size), 0)
        heatmap_uint8 = np.uint8(255 * blurred_mask)
        heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = (1.0 - alpha) * image + alpha * heatmap
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

def find_keywords(args, model, processor, input_ids, output_ids, frames, blur_frames, output_text, tokenizer=None, use_yake=False, special_ids=None):
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
            probs_blur = get_token_probs(args, model, processor, full_prompt, output_ids, blur_frames)
            
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

def evaluate_auc(args, model, processor, full_ids, output_ids, frames, tubelets, selected_tubes, positions=None, num_steps=20):
    eprint("\n--- Starting AUC Evaluation ---")
    video_array = np.stack([np.array(img) for img in frames])
    blurred_video = precompute_blurred_video(video_array)
    
    prob_orig = get_prob(args, model, processor, full_ids, output_ids, frames, positions)
    prob_blur = get_prob(args, model, processor, full_ids, output_ids, [Image.fromarray(f) for f in blurred_video], positions)
    
    chunk_size = max(1, len(selected_tubes) // num_steps)
    
    ins_curve = [prob_blur]
    del_curve = [prob_orig]
    percentages = [0.0]
    
    current_ins_tubes = []
    current_del_tubes = list(np.unique(tubelets)) 
    
    for i in range(0, len(selected_tubes), chunk_size):
        chunk = selected_tubes[i : i + chunk_size]
        
        # Insertion evaluation (blur background)
        current_ins_tubes.extend(chunk)
        frames_ins = [Image.fromarray(f) for f in apply_blur_mask_fast(video_array, tubelets, current_ins_tubes)]
        p_ins = get_prob(args, model, processor, full_ids, output_ids, frames_ins, positions)
        ins_curve.append(p_ins)
        
        # Deletion evaluation (black background per RISE logic)
        for t in chunk:
            if t in current_del_tubes:
                current_del_tubes.remove(t)
        frames_del = apply_constant_mask(video_array, tubelets, current_del_tubes)
        p_del = get_prob(args, model, processor, full_ids, output_ids, frames_del, positions)
        del_curve.append(p_del)
        
        percentages.append(min(1.0, (i + chunk_size) / len(selected_tubes)))

    auc_ins = auc(percentages, ins_curve)
    auc_del = auc(percentages, del_curve)
    eprint(f"Final AUC - Insertion: {auc_ins:.4f} | Deletion: {auc_del:.4f}")

    plt.figure(figsize=(8, 6))
    plt.plot(percentages, ins_curve, label=f'Insertion (AUC={auc_ins:.3f})', color='green', marker='o')
    plt.plot(percentages, del_curve, label=f'Deletion (AUC={auc_del:.3f})', color='red', marker='x')
    plt.axhline(y=prob_orig, color='gray', linestyle='--', label='Original Prob')
    plt.axhline(y=prob_blur, color='blue', linestyle='--', label='Blurred Prob')
    
    plt.title(f"XAI Fidelity Curves (Mode: {getattr(args, 'opt_mode', 'combined')})")
    plt.xlabel("Percentage of Tubelets Revealed / Masked")
    plt.ylabel("Target Probability")
    plt.legend()
    plt.grid(True)
    
    # Store directly in out_dir from main loop
    os.makedirs(args.output_dir, exist_ok=True)
    plot_path = os.path.join(args.output_dir, "auc_curves.png")
    plt.savefig(plot_path)
    plt.close()
    
    return auc_ins, auc_del