import torch.nn.functional as F
import cv2
cv2.setNumThreads(0)
import gc
import re
import os

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
from itertools import combinations

from qwen_vl_utils import process_vision_info

import concurrent.futures

from . import vis_utils
from . import model_utils
from . import cv_utils

import os
import json

DEFAULT_VIDEO_TOKEN = "<video>"


def timestamp_to_sec(timestamp):
    h, m, s = timestamp.split(':')
    return float(h) * 3600 + float(m) * 60 + float(s)


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

def log_experiment(args, log_func, ivd, question_text, ground_truth, model_answer, keywords, positions,
                   prob_orig, prob_baseline_del, prob_baseline_ins, prob_ins, prob_del, auc_ins, auc_del,
                   deletion_metrics, insertion_metrics, top_k, fmt_ins, fmt_del, fmt_merged, 
                   selected_ins, selected_del, selected_merged, unique_tubes, k_fraction, mode_name, start_time):

    """Handles single experiment logging"""

    os.makedirs(args.output_dir, exist_ok=True)
    metrics_file = os.path.join(args.output_dir, "frame_experiment_metrics.jsonl")
    experiment_data = {
        "video_index": ivd,
        "num_frames": args.num_frames,
        "Prob orig": round(prob_orig,3),
        "Prob_baseline_del": round(prob_baseline_del,3),
        "Prob_baseline_ins": round(prob_baseline_ins,3),
        "Prob ins": round(prob_ins,3),
        "Prob del": round(prob_del,3),
        "AUC Ins": round(auc_ins,3),
        "AUC Del": round(auc_del,3),
        "deletion_metrics": deletion_metrics,
        "insertion_metrics": insertion_metrics,
    }
    
    with open(metrics_file, "a") as f:
        f.write(json.dumps(experiment_data) + "\n")

    log_func("-" * 25)
    log_func(f"Question {ivd+1}/{args.num_videos}: {question_text}")
    log_func(f"Ground Truth: {ground_truth}")
    log_func(f"Model Answer: {model_answer}")
    log_func(f"Extracted Keywords: {keywords} (Positions: {positions})")
    log_func("-" * 25)
    log_func(f"Original probs: {prob_orig:.5f}")
    log_func(f"baseline_del probs: {prob_baseline_del:.5f}")
    log_func(f"baseline_ins probs: {prob_baseline_ins:.5f}\n")

    log_func(f"=== {mode_name} Tubelets Search Log ===")
    log_func(f"Created tubelets and optimized weights in {(time.time() - start_time):.2f}s") # <-- Updated to start_time
    log_func(f"Final Insertion tubelets (Top {top_k}): {fmt_ins} ({len(selected_ins)}/{len(unique_tubes)})")
    log_func(f"Final Deletion tubelets (Top {top_k}): {fmt_del} ({len(selected_del)}/{len(unique_tubes)})")
    log_func(f"Final Combined tubelets (Top {top_k}): {fmt_merged} ({len(selected_merged)}/{len(unique_tubes)})")
    log_func(f"Prob when Inserting Mask (top {k_fraction*100}%): {prob_ins:.5f} (Diff: {prob_orig - prob_ins:.5f})")
    log_func(f"Prob when Deleting Mask (top {k_fraction*100}%): {prob_del:.5f} (Diff: {prob_orig - prob_del:.5f})")

    log_func(f"\n=== Experiment Metrics ===\n")
    log_func(f"Insertion Landscape -> d_eff: {insertion_metrics.get('d_eff', 0):.2f} | Diversity (L1): {insertion_metrics.get('diversity', 0):.4f} (from {insertion_metrics.get('num_top_candidates', 0)} top masks)")
    log_func(f"Deletion Landscape  -> d_eff: {deletion_metrics.get('d_eff', 0):.2f} | Diversity (L1): {deletion_metrics.get('diversity', 0):.4f} (from {deletion_metrics.get('num_top_candidates', 0)} top masks)")
    log_func(f"\nAUC Ins: {auc_ins:.4f} | Del: {auc_del:.4f}")


def log_metrics(args, metrics, prefix=""):
    """dump experiment metrics in log file"""
        
    with open(os.path.join(args.output_dir, f'{prefix}final_metrics.json'), 'w') as f: # <-- Appended prefix to filename to prevent overwrite
        p_orig = np.array([m['prob_orig'] for m in metrics])
        p_del = np.array([m['prob_del'] for m in metrics])
        p_b_del = np.array([m['prob_baseline_del'] for m in metrics])
        p_ins = np.array([m['prob_ins'] for m in metrics])
        p_b_ins = np.array([m['prob_baseline_ins'] for m in metrics])
        
        # Calculate average probability differences
        avg_prob_diff_del_topx = np.mean(p_orig - p_del)
        avg_prob_diff_del_blur = np.mean(p_orig - p_b_del)
        avg_prob_diff_ins_topx = np.mean(p_ins - p_b_ins)
        avg_prob_diff_ins_orig = np.mean(p_orig - p_b_ins)
        
        summary = {
            f"{prefix}avg_auc_ins": float(np.mean([m['auc_ins'] for m in metrics])),
            f"{prefix}avg_auc_del": float(np.mean([m['auc_del'] for m in metrics])),
            f"{prefix}avg_prob_diff_del_topx": float(avg_prob_diff_del_topx),
            f"{prefix}avg_prob_diff_del_blur": float(avg_prob_diff_del_blur),
            f"{prefix}avg_prob_diff_ins_topx": float(avg_prob_diff_ins_topx),
            f"{prefix}avg_prob_diff_ins_orig": float(avg_prob_diff_ins_orig),
            f"{prefix}prob_del_explained": float(avg_prob_diff_del_topx / avg_prob_diff_del_blur) if avg_prob_diff_del_blur != 0 else 0.0,
            f"{prefix}prob_ins_explained": float(avg_prob_diff_ins_topx / avg_prob_diff_ins_orig) if avg_prob_diff_ins_orig != 0 else 0.0,
        }
        json.dump(summary, f, indent=4)


