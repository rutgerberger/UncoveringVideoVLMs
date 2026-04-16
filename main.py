import sys
from textwrap import fill

def eprint(*args, **kwargs):
    sep = ' '
    combined_text = sep.join(str(arg) for arg in args)
    wrapped_lines = [fill(line, width=80) if line.strip() else "" for line in combined_text.splitlines()]
    print("\n".join(wrapped_lines), file=sys.stderr, **kwargs)

eprint("Loading libraries 1/3...")
import os
import gc
import json
import time
import pickle
import random
import datetime
from dotenv import load_dotenv

eprint("Loading libraries 2/3...")
import torch
import numpy as np
import pandas as pd
from PIL import Image
from skimage.segmentation import slic, mark_boundaries
from datasets import load_dataset
from transformers import VideoLlavaForConditionalGeneration, VideoLlavaProcessor
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

eprint("Importing files 3/3...")
from utils import *
from method_helpers import *
from method_helpers import _get_rescale_and_dummys
from method import (
    spix_cmaes, 
    spix_gradient_iterative, 
    spix_rise_perturbation, 
    frame_redundancy, 
    spix_simulated_annealing
)
from args import init_args
from iGOS.method import iGOS_p, perform_igos

load_dotenv()
DEFAULT_VIDEO_TOKEN = "<video>"


def xai_method(args, model, tokenizer, processor, input_ids, output_ids, frames, tubelets, baseline_ins_arr, baseline_del_arr, positions, ivd):
    """
    Returns the XAI method of preference (for testing)
    """
    method = getattr(args, 'method', 'sa')
    if method == 'cmaes':
        return spix_cmaes(
            args, model, tokenizer, processor, input_ids, output_ids, 
            frames, tubelets, baseline_ins_arr, baseline_del_arr, positions=positions
        )
    elif method == 'gradient_iterative':
        return spix_gradient_iterative(
            args, model, tokenizer, processor, input_ids, output_ids, frames, tubelets,
            positions=positions, max_stages=getattr(args, 'max_stages', 2), index=ivd
        )
    else:
        return spix_simulated_annealing(
             args, model, tokenizer, processor, input_ids, output_ids, frames, tubelets,
             positions=positions
        )

def run_xai_pipeline(args, model, processor, tokenizer, frames, video_array, tubelets, 
                     baseline_ins_arr, baseline_del_arr, baseline_ins_frames, baseline_del_frames, 
                     special_ids, input_ids, output_ids, question_text, ground_truth, model_answer, 
                     target_text, ivd, log_func, mode_name, file_prefix):
    """
    Runs XAI pipeline and handles logging
    """
    start = time.time()
    full_ids = torch.cat((input_ids, output_ids), dim=1)
    unique_tubes = np.unique(tubelets)

    eprint(f"{ivd+1}/{args.num_videos}: Selecting Important Tubelets ({mode_name}).")
    positions, keywords = find_keywords(
        args, model, processor, input_ids, output_ids, frames, baseline_ins_frames, 
        target_text, tokenizer=tokenizer, use_yake=args.use_yake, special_ids=special_ids
    )

    if getattr(args, 'method', '') == 'igos':
        auc_ins, auc_del = perform_igos()
        return auc_ins, auc_del, auc_ins, auc_del

    prob_orig = get_prob(args, model, processor, full_ids, output_ids, frames, positions)
    prob_baseline_del = get_prob(args, model, processor, full_ids, output_ids, baseline_del_frames, positions)
    prob_baseline_ins = get_prob(args, model, processor, full_ids, output_ids, baseline_ins_frames, positions)

    eprint(f"{ivd+1}/{args.num_videos}: Optimizing weights")
    selected_ins, selected_del, scores_ins, scores_del, insertion_metrics, deletion_metrics = xai_method(
        args, model, tokenizer, processor, input_ids, output_ids, frames, tubelets, 
        baseline_ins_arr, baseline_del_arr, positions, ivd
    )

    #Unite the masks (both intersection and union)
    # raw_union = set(selected_ins) | set(selected_del)
    # raw_inter = set(selected_ins) & set(selected_del)
    # selected_union, _ = merge_masks_borda(selected_ins, selected_del, list(raw_union))
    # selected_inter, _ = merge_masks_borda(selected_ins, selected_del, list(raw_inter))
    scores_merged = {}
    for t in unique_tubes:
        # Multiply the continuous scores (Logical AND / Intersection)
        scores_merged[t] = scores_ins.get(t, 0.0) * scores_del.get(t, 0.0)
    selected_merged = sorted(list(unique_tubes), key=lambda t: scores_merged[t], reverse=True)

    k_fraction = 0.20
    k_tubes = max(1, int(len(unique_tubes) * k_fraction))

    top_final = selected_merged[:k_tubes]
    frames_ins = apply_universal_mask(video_array, baseline_ins_arr, tubelets, top_final)
    prob_ins = get_prob(args, model, processor, full_ids, output_ids, frames_ins, positions)
    
    keep_tubes_del = [t for t in unique_tubes if t not in top_final]
    frames_del = apply_universal_mask(video_array, baseline_del_arr, tubelets, keep_tubes_del)
    prob_del = get_prob(args, model, processor, full_ids, output_ids, frames_del, positions)

    top_k = min(5, len(selected_ins))
    fmt_ins = {t: f"{scores_ins.get(t, 0):.4f}" for t in selected_ins[:top_k]}
    fmt_del = {t: f"{scores_del.get(t, 0):.4f}" for t in selected_del[:top_k]}
    fmt_merged = {t: f"{scores_merged.get(t, 0):.4f}" for t in selected_merged[:top_k]}

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
    log_func(f"Created tubelets in {(time.time() - start):.2f}s")
    log_func(f"Final Insertion tubelets (Top {top_k}): {fmt_ins} ({len(selected_ins)}/{len(unique_tubes)})")
    log_func(f"Final Deletion tubelets (Top {top_k}): {fmt_del} ({len(selected_del)}/{len(unique_tubes)})")
    log_func(f"Final Combined tubelets (Top {top_k}): {fmt_merged} ({len(selected_merged)}/{len(unique_tubes)})")
    log_func(f"Prob when Inserting Mask (top 20%): {prob_ins:.5f} (Diff: {prob_orig - prob_ins:.5f})")
    log_func(f"Prob when Deleting Mask (top 20%): {prob_del:.5f} (Diff: {prob_orig - prob_del:.5f})")

    auc_ins, auc_del = evaluate_auc(
        args, model, processor, full_ids, output_ids, frames, video_array, tubelets, 
        selected_merged, baseline_ins_arr, baseline_del_arr, ivd=f"{ivd}_merged", positions=positions
    )
    
    log_func(f"\nAUC Ins: {auc_ins:.4f} | Del: {auc_del:.4f}")

    os.makedirs(args.output_dir, exist_ok=True)
    metrics_file = os.path.join(args.output_dir, "frame_experiment_metrics.jsonl")
    experiment_data = {
        "video_index": ivd,
        "num_frames": args.num_frames,
        "deletion_metrics": deletion_metrics,
        "insertion_metrics": insertion_metrics,
        "AUC Ins": auc_ins, "AUC Del": auc_del,
    }
    with open(metrics_file, "a") as f:
        f.write(json.dumps(experiment_data) + "\n")

    if getattr(args, 'save_visuals', True):
        eprint(f"{ivd+1}/{args.num_videos}: Visualizing the Masked Tubelets ({mode_name}).")
        # Binary cutout of top 20%
        keep_tubes_vis = [t for t in unique_tubes if t not in top_final]
        visualize_spix(video_array, baseline_del_arr, tubelets, keep_tubes_vis, os.path.join(args.output_dir, f"{ivd}_{file_prefix}mask_cutout.gif"))
        # Continuous heatmaps
        visualize_heatmap(video_array, tubelets, scores_ins, os.path.join(args.output_dir, f"{ivd}_{file_prefix}highlight_ins.gif"))
        visualize_heatmap(video_array, tubelets, scores_del, os.path.join(args.output_dir, f"{ivd}_{file_prefix}highlight_del.gif"))
        visualize_heatmap(video_array, tubelets, scores_merged, os.path.join(args.output_dir, f"{ivd}_{file_prefix}highlight_merged.gif"))
        
    return auc_ins, auc_del

def explain_vid(data, model, processor, args, tokenizer):
    now = datetime.datetime.now()
    setting = f'{now.month}{now.day}-{now.hour}{now.minute}'
    out_dir = os.path.join(args.output_dir, setting)
    args.output_dir = out_dir
    os.makedirs(args.output_dir, exist_ok=True)
    
    with open(os.path.join(out_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    log_file = os.path.join(args.output_dir, "log.txt")
    def log(msg):
        with open(log_file, "a") as f:
            f.write(str(msg) + "\n")

    num_videos = min(len(data), args.num_videos) if args.num_videos > 0 else len(data)
    global_metrics = {k: [] for k in ["auc_ins", "auc_del", "gt_auc_ins", "gt_auc_del"]}
    
    for ivd in range(1,num_videos):
        eprint(f"\n{ivd+1}/{num_videos}: Retrieving data.")
        start = time.time()
        idx = random.randint(0, len(data)-1) if args.dataset == 'k400' else ivd
        
        try:
            frames, qs, cur_prompt, correct_idx = get_data(args, data[idx])
        except Exception as e:
            eprint(f"Error loading row {idx}: {e}")
            continue
            
        eprint(f"Retrieved data in {time.time() - start:.2f}s")
        ground_truth_label = str(correct_idx)

        video_array, tubelets = generate_tubelets_optimized(frames, args)
        prompt = f"USER: {DEFAULT_VIDEO_TOKEN}\n{qs}. ASSISTANT:"
        input_ids, output_ids, output_text = get_model_response(args, model, processor, tokenizer, prompt, frames)

        baseline_ins_arr = get_baseline_insertion(args, video_array)
        baseline_del_arr = get_baseline_deletion(args, video_array)
        baseline_ins_frames = [Image.fromarray(f.astype(np.uint8)) for f in baseline_ins_arr]
        baseline_del_frames = [Image.fromarray(f.astype(np.uint8)) for f in baseline_del_arr]
        special_ids = [idx for idx in [tokenizer.pad_token_id, tokenizer.eos_token_id, tokenizer.bos_token_id] if idx is not None]

        if getattr(args, 'save_visuals', True):
            eprint(f"Visualizing baseline frames for video {ivd}...")
            visualize_frames(baseline_ins_frames, os.path.join(args.output_dir, f"{ivd}_baseline_insertion.gif"))
            visualize_frames(baseline_del_frames, os.path.join(args.output_dir, f"{ivd}_baseline_deletion.gif"))

        metrics = run_xai_pipeline(
            args, model, processor, tokenizer, frames, video_array, tubelets, 
            baseline_ins_arr, baseline_del_arr, baseline_ins_frames, baseline_del_frames, special_ids,
            input_ids, output_ids, qs, ground_truth_label, output_text, output_text, ivd, log, 
            mode_name="STANDARD", file_prefix=""
        )

        global_metrics["auc_ins"].append(metrics[0])
        global_metrics["auc_del"].append(metrics[1])

        if getattr(args, 'gt_forcing', False):    
            gt_ids = tokenizer(ground_truth_label, return_tensors='pt', add_special_tokens=False).input_ids.to(model.device)
            if gt_ids is not None:
                res_gt = run_xai_pipeline(
                    args, model, processor, tokenizer, frames, video_array, tubelets, 
                    baseline_ins_arr, baseline_del_arr, baseline_ins_frames, baseline_del_frames, special_ids,
                    input_ids, gt_ids, qs, ground_truth_label, output_text, ground_truth_label, ivd, log, 
                    mode_name="GROUND TRUTH", file_prefix="gt_"
                )
                global_metrics["gt_auc_ins"].append(res_gt[0])
                global_metrics["gt_auc_del"].append(res_gt[1])

        gc.collect()
        torch.cuda.empty_cache()

    with open(os.path.join(args.output_dir, 'final_metrics.json'), 'w') as f:
        summary = {k: float(np.mean(v)) if v else 0.0 for k, v in global_metrics.items()}
        json.dump(summary, f, indent=4)
    eprint(f"\nExperiment Complete. Mean metrics saved to {args.output_dir}/final_metrics.json")

if __name__ == "__main__":
    args = init_args()
    eprint(f"args:\n {json.dumps(vars(args), indent=2)}")

    torch.manual_seed(args.manual_seed)
    random.seed(args.manual_seed)
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)

    eprint("Initializing Model...")
    if args.model == 'qwen':
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype=torch.bfloat16, device_map="auto")
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    else:
        model = VideoLlavaForConditionalGeneration.from_pretrained("LanguageBind/Video-LLaVA-7B-hf", dtype=torch.float16, device_map="auto")
        processor = VideoLlavaProcessor.from_pretrained("LanguageBind/Video-LLaVA-7B-hf")
    
    tokenizer = processor.tokenizer
    for param in model.parameters(): param.requires_grad = False
    model.gradient_checkpointing = True
    eprint("Model Mapping:", model.hf_device_map)

    eprint("Loading Dataset...")
    if getattr(args, 'dataset', '') == 'TGIF':
        q_path, a_path = os.path.join(args.data_path, 'test_q.json'), os.path.join(args.data_path, 'test_a.json')
        q_data, a_dict = json.load(open(q_path)), {i['question_id']: i for i in json.load(open(a_path))}
        data = [{**q, **a_dict[q['question_id']]} for q in q_data if q['question_id'] in a_dict]
    
    elif getattr(args, 'dataset', '') == 'imagenet':
        os.makedirs(args.video_folder, exist_ok=True)
        cache_path = os.path.join(args.video_folder, f"imagenet_5k_seed{args.manual_seed}.pkl")
        if os.path.exists(cache_path):
            data = pickle.load(open(cache_path, 'rb'))
        else:
            dataset = load_dataset("ILSVRC/imagenet-1k", split="validation", streaming=True)
            n_images = 10
            dataset = dataset.shuffle(seed=args.manual_seed, buffer_size=n_images*10).take(n_images)
            labels_feature = load_dataset("ILSVRC/imagenet-1k", split="validation", streaming=True).info.features['label']
            data = [{'image': row['image'], 'label_name': labels_feature.int2str(row['label'])} for row in dataset]
            pickle.dump(data, open(cache_path, 'wb'))
    
    elif args.data_path.endswith('npy') or getattr(args, 'dataset', '') == 'moving_mnist':
        raw_mnist = np.load(args.data_path)
        if raw_mnist.shape[0] == 20 and len(raw_mnist.shape) == 4:
            raw_mnist = np.transpose(raw_mnist, (1, 0, 2, 3))
        data = [{
            'is_numpy_video': True, 'video_array': np.stack((raw_mnist[i],)*3, axis=-1),
            'question': "Describe the movement of the digits in this video.", 
            'answer': "Two digits are bouncing off the walls.", 'video_index': i
        } for i in range(min(len(raw_mnist), getattr(args, 'num_videos', len(raw_mnist))))]
    
    elif args.data_path.endswith('csv'):
        data = pd.read_csv(args.data_path).to_dict(orient='records')
    elif args.data_path.endswith('jsonl'):
        data = [json.loads(q) for q in open(args.data_path, "r")]
    elif args.data_path.endswith('json'):
        raw = json.load(open(args.data_path))
        data = list(raw.values()) if isinstance(raw, dict) else raw
    elif args.data_path.endswith('pkl'):
        data = pickle.load(open(args.data_path, 'rb'))
    else:
        data = load_dataset(args.data_path, "val")["val"].to_pandas().to_dict(orient="records")
        
    explain_vid(data, model, processor, args, tokenizer)