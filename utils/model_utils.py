import torch.nn.functional as F
import cv2
cv2.setNumThreads(0)
import gc
import re

import torch

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

from . import io_utils
from . import cv_utils
from . import vis_utils

import concurrent.futures

DEFAULT_VIDEO_TOKEN = "<video>"

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

def get_prob(args, model, processor, full_ids, output_ids, frames, positions=None, tokenizer=None):
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
    
    if tokenizer and positions is not None and len(positions) > 0:
        decoded_targets = [tokenizer.decode(idx) for idx in output_ids[0]]
        eprint("\n--- TOKEN ALIGNMENT SANITY CHECK ---")
        for pos in positions:
            token_str = decoded_targets[pos]
            prob_val = target_probs[0, pos].item()
            eprint(f"Position {pos} | Token: '{token_str}' | Extracted Prob: {prob_val:.4f}")
        eprint("------------------------------------\n")

    if target_probs.dim() == 0:
        target_probs = target_probs.unsqueeze(0)
    if positions is not None and len(positions) > 0:
        target_probs = target_probs[0, positions] 
    return torch.log(target_probs + 1e-7).sum().item()
    #sreturn target_probs.mean().item()



def match_keywords(output_list, kw_ids):
    """Helper to find the sublist kw_ids within output_list and return the indices."""
    kw_len = len(kw_ids)
    for i in range(len(output_list) - kw_len + 1):
        if output_list[i : i + kw_len] == kw_ids:
            return list(range(i, i + kw_len))
    return []


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