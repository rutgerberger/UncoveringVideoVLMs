import os
import torch
import torch.nn.functional as F
import cv2
cv2.setNumThreads(0)
import gc
import re

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

from . import io_utils
from . import model_utils
from . import vis_utils

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
    return get_baseline_insertion(args, video_array)
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


