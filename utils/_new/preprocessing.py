"""
Utilities with regards to preprocessing data
"""

import cv2
cv2.setNumThreads(0)
import os
import re
import random
import concurrent.futures

import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image
from decord import VideoReader, cpu
from sklearn.cluster import KMeans

from logging import eprint

def timestamp_to_sec(timestamp):
    h, m, s = timestamp.split(':')
    return float(h) * 3600 + float(m) * 60 + float(s)

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
    return precompute_blurred_video(video_array)
    return np.full_like(video_array, 255) # Constant white video

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

def generate_tubelets_optimized(video, args, downsample_factor=0.5):
    video_array = np.stack([np.array(img) for img in video]) 
    T, H, W, C = video_array.shape
    
    if getattr(args, 'use_slic', True):
        # CPU Downsampling via OpenCV
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

        # Perform SLIC clustering in an isolated process (otherwise this would lead into deadlocks)
        n_seg = getattr(args, 'n_segments', 120)
        comp = getattr(args, 'compactness', 10)
        with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
            # Submit the job to the isolated process
            future = executor.submit(_run_slic_isolated, video_down_float, n_seg, comp)
            # This will wait for the child process to return the result
            tubelet_labels_down = future.result() 
        
        # CPU Upsampling via OpenCV
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

def create_super_tubelets(video_array, tubelets, n_clusters=12, mode='spatial'):
    """
    Clusters N tubelets into K super-tubelets.
    mode: 'spatial' (groups by physical proximity) or 'appearance' (groups by color)
    """
    num_tubes = int(tubelets.max()) + 1
    features = np.zeros((num_tubes, 3)) 
    for i in range(num_tubes):
        mask = (tubelets == i)
        if not np.any(mask): 
            continue
            
        if mode == 'appearance':
            # Mean RGB color
            features[i] = video_array[mask].mean(axis=0)[:3]
        elif mode == 'spatial':
            # 3D Centroid (T, Y, X)
            coords = np.argwhere(mask)
            features[i] = coords.mean(axis=0) 

    # Normalize features so K-Means treats all dimensions equally
    features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-5)
    # Cluster into Super-Tubelets
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto').fit(features)
    labels = kmeans.labels_ # Maps sub_id -> super_id
    # Create the 3D Super-Tubelet mask
    super_tubelets = labels[tubelets]
    
    return super_tubelets, labels

def get_data(args, row):
    num_frames = args.num_frames
    start_sec = 0
    end_sec = 1
    
    # --- GLOBAL CAP --- 
    # 384x384 keeps detail but strictly limits tokens to prevent OOM
    max_dim = 384 

    # -- ImageNet Data Handling
    if getattr(args, 'dataset', '') == 'imagenet':
        question_text = "If you were a classifier trained on ImageNet, what object is in this video? Answer with only the class name (use spaces to separate multiple words)."
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
    elif getattr(args, 'dataset', '') == 'clevrer':
        video_filename = row.get('video_filename', '')
        video_path = None
        match = re.search(r'\d+', video_filename)
        if match:
            vid_num = int(match.group())
            lower_bound = (vid_num // 1000) * 1000
            upper_bound = lower_bound + 1000
            subfolder_name = f"video_{lower_bound}-{upper_bound}"
            fast_path = os.path.join(args.video_folder, subfolder_name, video_filename)
            if os.path.exists(fast_path):
                video_path = fast_path

        # Fallback to os.walk ONLY if the mathematical path fails or file is missing
        if video_path is None:
            for root, dirs, files in os.walk(args.video_folder):
                if video_filename in files:
                    video_path = os.path.join(root, video_filename)
                    break
                    
        # Absolute fallback if neither worked
        if video_path is None:
            video_path = os.path.join(args.video_folder, video_filename)

        # Parse question data
        q_data = row['questions'][0] # first question in the scene
        question_text = q_data['question']
        q_type = q_data.get('question_type', 'descriptive')
        options_prompt = ""
        
        # Descriptive questions have a direct string answer
        if q_type == 'descriptive' or 'choices' not in q_data:
            correct_idx = q_data.get('answer', '')
            cur_prompt = question_text
            qs = cur_prompt + "\nAnswer with as few words as possible."
        else:
            correct_choices = []
            for i, choice in enumerate(q_data['choices']):
                options_prompt += f"\n{i}. {choice['choice']}"
                if choice.get('answer') == 'correct':
                    correct_choices.append(str(i))
            # CLEVRER multiple-choice questions can have multiple correct answers.
            # We join them by comma (e.g., "0, 2")
            correct_idx = ", ".join(correct_choices)
            cur_prompt = f"{question_text}{options_prompt}"
            qs = cur_prompt + "\nAnswer with only the INDEX of the correct answer(s)."
            
        apply_slice = False

    elif getattr(args, 'dataset', '') == 'k400':
        yt_id = row['youtube_id']
        start_time = str(row['time_start']).zfill(6)
        end_time = str(row['time_end']).zfill(6)
        video_filename = f"{yt_id}_{start_time}_{end_time}.mp4"
        video_path = os.path.join(args.video_folder, "val", video_filename)
        question_text = "What type of activity is happening in this video? Answer with only the action class name (use spaces to separate multiple words)"
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

def rescale_mask(mask, new_H, new_W, crop_top, crop_left, target_H, target_W, target_T, T_orig, is_qwen, t_dim_index):
    """
    Resizes mask to input size of the VLM and returns both the formatted 
    VLM tensor and the 5D volumetric tensor for TV norm calculation.
    """
    M_resized_step = F.interpolate(mask, size=(new_H, new_W), mode='bilinear', align_corners=False)
    M_cropped_step = M_resized_step[:, :, crop_top:crop_top+target_H, crop_left:crop_left+target_W]
    M_vol_step = M_cropped_step.permute(1, 0, 2, 3).unsqueeze(0) # (1, 1, T, H, W)
    
    if target_T != T_orig:
        M_vol_step = F.interpolate(M_vol_step, size=(target_T, target_H, target_W), mode='trilinear', align_corners=False)
    
    if is_qwen:
        M_scaled = M_vol_step.view(-1, 1) 
    else:
        if t_dim_index == 1:
            M_scaled = M_vol_step.permute(0, 2, 1, 3, 4) # (1, T, 1, H, W)
        else:
            M_scaled = M_vol_step # (1, 1, T, H, W)

    return M_scaled, M_vol_step
