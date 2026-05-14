import gc

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

import numpy as np

from qwen_vl_utils import process_vision_info

DEFAULT_VIDEO_TOKEN = "<video>"

from .logging import eprint

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# ==========================================
# Build original output & input
# ==========================================

def process_vid_qwen(processor, frames, prompt=" ", apply_chat_template=False, fps=1.0, max_pixels=112896):
    """
    Universal wrapper for Qwen's vision processor to prevent feature/token mismatches.
    """
    messages = [
        {"role": "user", "content": [
            {"type": "video", "video": frames, "fps": fps},
            {"type": "text", "text": prompt},
        ]},
    ]
    image_inputs, video_inputs = process_vision_info(messages)
    
    # Only format with the <|im_start|> tags if we are explicitly asking for an answer
    if apply_chat_template:
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        text_inputs = [text]
    else:
        text_inputs = [prompt]
        
    inputs = processor(
        text=text_inputs, 
        images=image_inputs, 
        videos=video_inputs, 
        padding=True, 
        return_tensors="pt", 
        max_pixels=max_pixels
    )
    
    return inputs    

def generate_qwen(args, model, processor, prompt: str, frames):
    inputs = process_vid_qwen(processor, frames, prompt=prompt, apply_chat_template=True)
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


# ==========================================
# During optimization, we use optimized func.
# to generate output & build input
# ==========================================

def _get_target_logits(model, forward_kwargs, output_ids):
    """Executes the model and slices the logits for the output text."""
    with torch.no_grad():
        outputs = model(**forward_kwargs)
    out_len = output_ids.shape[-1]
    return outputs.logits[:, -out_len - 1 : -1, :]


def _build_kwargs_processor(args, model, processor, full_ids, frames):
    """Prepares model inputs using the Hugging Face processor."""

    if getattr(args, 'model', '') == 'qwen':
        inputs = process_vid_qwen(processor, frames)
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
        
    return forward_kwargs

def _build_kwargs_direct(args, model, full_ids, vid_input, dummy_inputs_orig):
    """Prepares model inputs directly from a tensor (bypassing the processor)."""
    forward_kwargs = {
        "input_ids": full_ids,
        "attention_mask": torch.ones_like(full_ids).to(model.device),
        "pixel_values_videos": vid_input.to(dtype=model.dtype),
        "use_cache": False
    }
    
    if getattr(args, 'model', '') == 'qwen' and dummy_inputs_orig is not None:
        forward_kwargs["video_grid_thw"] = dummy_inputs_orig.get("video_grid_thw")
        
    return forward_kwargs


# ==========================================
# Processing based on positions
# ==========================================

def _gather_and_filter(tensor, output_ids, positions=None):
    """Gathers values for specific output_ids and optionally filters by positions."""
    gathered = tensor.gather(dim=-1, index=output_ids.unsqueeze(-1)).squeeze(-1)
    if gathered.dim() == 0:
        gathered = gathered.unsqueeze(0)
    if positions is not None and len(positions) > 0:
        gathered = gathered[0, positions]
    return gathered


# ==========================================
# Public Functions 
# ==========================================

def get_logits(args, model, processor, full_ids, output_ids, frames):
    """Returns the raw output logit scores (full distribution) of the model."""
    kwargs = _build_kwargs_processor(args, model, processor, full_ids, frames)
    return _get_target_logits(model, kwargs, output_ids)


def get_log_prob(args, model, processor, full_ids, output_ids, frames, positions=None, tokenizer=None):
    """Calculates log probability of the output_ids; optionally filtered by positions."""
    target_logits = get_logits(args, model, processor, full_ids, output_ids, frames)
    probs = F.softmax(target_logits, dim=-1) 
    target_probs = _gather_and_filter(probs, output_ids, positions)
    return torch.log(target_probs + 1e-7).sum().item()


def get_prob_array(args, model, processor, full_ids, output_ids, frames, positions=None):
    """Returns an array of probability scores for the output_ids; optionally filtered by positions."""
    target_logits = get_logits(args, model, processor, full_ids, output_ids, frames)
    probs = F.softmax(target_logits, dim=-1)
    return _gather_and_filter(probs, output_ids, positions)


def get_vocab_stats(args, model, processor, full_ids, output_ids, frames, positions):
    """
    Returns the mean (mu) and standard deviation (sigma) of the entire 
    vocabulary distribution for the specified target positions.
    """
    full_logits = get_logits(args, model, processor, full_ids, output_ids, frames)
    if positions is not None and len(positions) > 0:
        full_logits = full_logits[:, positions, :]
    # Calculate statistics across the vocab dimension (dim=-1)
    mus = full_logits.mean(dim=-1).squeeze(0)   # Shape: [len(positions)]
    sigmas = full_logits.std(dim=-1).squeeze(0) # Shape: [len(positions)]

    return mus, sigmas


def get_score_direct(vid_input, model, args, full_ids, output_ids, dummy_inputs_orig, positions=None, vocab_stats=None):
    """
    Directly calculates the Z-scored raw logit fitness from a video tensor,
    bypassing the processor and avoiding Softmax squashing.
    """
    kwargs = _build_kwargs_direct(args, model, full_ids, vid_input, dummy_inputs_orig)
    target_logits = _get_target_logits(model, kwargs, output_ids)
    raw_token_logits = _gather_and_filter(target_logits, output_ids, positions)
    if vocab_stats is not None:
        mus, sigmas = vocab_stats
        # Ensure tensors are on the same device
        mus = mus.to(raw_token_logits.device)
        sigmas = sigmas.to(raw_token_logits.device)
        # Z = (X - mu) / sigma
        z_scores = (raw_token_logits - mus) / (sigmas + 1e-7)
        return z_scores.sum().item()
    return raw_token_logits.sum().item() #fallback


def get_rescale_and_dummys(model, processor, frames, baseline_frames, is_qwen, tubelets):
    """
    Given a huggingface VLM, its corresponding processor,
    a list of frames, and a list of baseline frames, this
    function returns dummy inputs and geometric scaling parameters.
    """
    with torch.no_grad():
        if is_qwen:
            dummy_inputs_orig = process_vid_qwen(processor, frames).to(model.device)
            dummy_inputs_base = process_vid_qwen(processor, baseline_frames).to(model.device)

            pixels_orig = dummy_inputs_orig['pixel_values_videos'].detach()
            pixels_base = dummy_inputs_base['pixel_values_videos'].detach()
            
            grid_thw = dummy_inputs_orig['video_grid_thw'][0] 
            target_T, target_H, target_W = grid_thw[0].item(), grid_thw[1].item(), grid_thw[2].item()
            target_C = pixels_orig.shape[1] 
            t_dim_index = -1
        else:
            dummy_inputs_orig = processor(text=" ", videos=frames, return_tensors="pt").to(model.device)
            dummy_inputs_base = processor(text=" ", videos=baseline_frames, return_tensors="pt").to(model.device)
            pixels_orig = dummy_inputs_orig['pixel_values_videos'].detach()
            pixels_base = dummy_inputs_base['pixel_values_videos'].detach()
            
            target_tensor_shape = pixels_orig.shape
            if target_tensor_shape[1] == 3: 
                target_C, target_T, target_H, target_W = target_tensor_shape[1:5]
                t_dim_index = 2
            else:
                target_T, target_C, target_H, target_W = target_tensor_shape[1:5]
                t_dim_index = 1

                
    # Cropping: Precalculate exact HF geometry scaling
    T_orig, H_orig, W_orig = tubelets.shape
    ratio = max(target_H / float(H_orig), target_W / float(W_orig))
    new_H, new_W = int(H_orig * ratio), int(W_orig * ratio)
    crop_top = (new_H - target_H) // 2
    crop_left = (new_W - target_W) // 2
    
    return dummy_inputs_orig, pixels_orig, pixels_base, target_T, target_H, target_W, t_dim_index, crop_top, crop_left, new_H, new_W, T_orig, H_orig, W_orig


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
                matched_pos = match_keywords(output_ids[0].tolist(), kw_ids) # Assuming this is a helper you have
                if matched_pos:
                    positions.extend(matched_pos)
                    keywords.extend([tokenizer.decode(output_ids[0][p]).strip() for p in matched_pos])
        else:
            full_ids = torch.cat((input_ids, output_ids), dim=1)
            
            probs = get_prob_array(args, model, processor, full_ids, output_ids, frames).squeeze(0)
            probs_blur = get_prob_array(args, model, processor, full_ids, output_ids, baseline_ins_frames).squeeze(0)
            
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