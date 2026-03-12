import torch
import numpy as np
import pandas as pd
import json
import datetime
import time
import os
import pickle
import random

from skimage.segmentation import slic, mark_boundaries
from PIL import Image
from datasets import load_dataset
from transformers import VideoLlavaForConditionalGeneration, VideoLlavaProcessor
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

from utils import *
from method import spix, spix_optimized, frame_redundancy
from args import init_args

DEFAULT_VIDEO_TOKEN = "<video>"
MAX_VIDEOS = 5

from dotenv import load_dotenv
load_dotenv()

def run_xai_pipeline(args, model, processor, tokenizer, frames, video_array, tubelets, baseline_ins_frames, special_ids, 
                     input_ids, output_ids, target_text, ivd, log_func, mode_name, file_prefix):
    """
    This block runs the following pipeline:
        Keyword Extraction -> SPIX -> Explicit Logging -> AUC Evaluation -> Visualization.
        
    returns the auc metrics.
    """
    start = time.time()
    # -- Finding Keywords and Probabilities
    eprint(f"{ivd+1}/{MAX_VIDEOS}: Selecting Important Tubelets ({mode_name}).")
    positions, keywords = find_keywords(
        args, model, processor, input_ids, output_ids, frames, baseline_ins_frames, 
        target_text, tokenizer=tokenizer, use_yake=args.use_yake, special_ids=special_ids
    )
    log_func(f"[{mode_name}] Extracted Keywords: {keywords} at positions {positions}")
    full_ids = torch.cat((input_ids, output_ids), dim=1)
    prob_orig = get_prob(args, model, processor, full_ids, output_ids, frames, positions)

    # -- Obtaining Tubelets
    final_tubelets, tubelet_scores = spix_optimized(
        args, model, processor, input_ids, output_ids, frames, tubelets,
        positions=positions, opt_mode=args.opt_mode, use_dynamic_lambda=args.use_dynamic_lambda
    )

    # -- Baseline Videos: What Happens to Video \ Tubelets
    baseline_ins = get_baseline_insertion(args, video_array)
    baseline_del = get_baseline_deletion(args, video_array)
    # -- Calculate explicit probability drops (logging purposes)
    # Insertion: Foreground is original video, Background is baseline_ins
    #TODO investigate this logic
    frames_ins = apply_universal_mask(video_array, baseline_ins, tubelets, final_tubelets)
    prob_ins = get_prob(args, model, processor, full_ids, output_ids, frames_ins, positions)
    # Deletion: Foreground is original video, Background is baseline_del
    unique_tubes = np.unique(tubelets)
    tubes_to_keep = [t for t in unique_tubes if t not in final_tubelets]
    frames_del = apply_universal_mask(video_array, baseline_del, tubelets, tubes_to_keep)
    prob_del = get_prob(args, model, processor, full_ids, output_ids, frames_del, positions)
    # -- Logging
    log_func(f"\n=== {mode_name} TUBELETS SEARCH LOG ===")
    log_func(f"Created tubelets in {time.time() - start:.2f}s")
    log_func(f"Final tubelets: {final_tubelets} ({len(final_tubelets)}/{len(unique_tubes)})")
    log_func(f"Prob Original: {prob_orig:.5f}")
    log_func(f"Prob Insertion: {prob_ins:.5f} (Diff: {prob_orig - prob_ins:.5f})")
    log_func(f"Prob Deletion: {prob_del:.5f} (Diff: {prob_orig - prob_del:.5f})")
    # -- Metric calculation
    auc_ins, auc_del = evaluate_auc(
        args, model, processor, full_ids, output_ids, frames, tubelets, final_tubelets, ivd=ivd,
        positions=positions, num_steps=20
    )
    log_func(f"[{mode_name}] AUC Insertion: {auc_ins:.4f}")
    log_func(f"[{mode_name}] AUC Deletion: {auc_del:.4f}")
    # -- Saving gifs for visualization
    if getattr(args, 'save_visuals', True):
        eprint(f"{ivd+1}/{MAX_VIDEOS}: Visualizing the Masked Tubelets ({mode_name}).")
        tubes_to_vis = [t for t in unique_tubes if t not in final_tubelets]
        visualize_spix(video_array, baseline_del, tubelets, tubes_to_vis, os.path.join(args.output_dir, f"{ivd}_{file_prefix}mask.gif"))
        visualize_heatmap(video_array, tubelets, tubelet_scores, os.path.join(args.output_dir, f"{ivd}_{file_prefix}highlight.gif"))
        
    return auc_ins, auc_del

def explain_vid(data, model, processor, args, tokenizer):
    """
    Holds the main loop with logging logic for method.
    """
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

    num_videos = min(len(data), MAX_VIDEOS) if MAX_VIDEOS > 0 else len(data)
    
    global_metrics = {
        "auc_ins": [], "auc_del": [],
        "gt_auc_ins": [], "gt_auc_del": []
    }
    # -- Main Loop - performs per given video
    for ivd in range(num_videos):
        # -- Data Processing / Logging
        eprint(f"{ivd+1}/{num_videos}: Retrieving data.")
        start = time.time()
        row = data[ivd]
        frames, qs, cur_prompt, correct_idx = get_data(args, row)
        log(f"\n\n================================")
        log(f"Question {ivd+1}/{num_videos} \n{qs}")
        eprint(f"Retrieved data in {time.time() - start}s")
        if getattr(args, 'dataset', '') != 'imagenet':
            video_desc = create_description(args, model, processor, frames, tokenizer)
            log(f"Video Description: {video_desc}")
        
        # -- Creation of Video Tubelets
        start = time.time()
        eprint(f"{ivd+1}/{num_videos}: Generating tubelets.")
        video_array, tubelets = generate_tubelets(frames, args)
        eprint(f"Generated tubelets in {time.time() - start}s")
        
        if getattr(args, 'save_visuals', True):
            visualize_frames(frames, os.path.join(args.output_dir, f"{ivd}_frames.gif"))
            visualize_tubelets(video_array, tubelets, os.path.join(args.output_dir, f"{ivd}_slic.gif"))
        
        eprint(f"{ivd+1}/{num_videos}: Acquiring Model Output.")
        prompt = f"USER: {DEFAULT_VIDEO_TOKEN}\n{qs}. ASSISTANT:"
        input_ids, output_ids, output_text = get_model_response(args, model, processor, tokenizer, prompt, frames)
        log(f"Model Answer: {output_text}")

        # Precompute constants used for both standard and GT pipelines
        baseline_ins = get_baseline_insertion(args, video_array)
        baseline_ins_frames = [Image.fromarray(f) for f in baseline_ins]
        special_ids = [tokenizer.pad_token_id, tokenizer.eos_token_id, tokenizer.bos_token_id]
        special_ids = [idx for idx in special_ids if idx is not None]

        # -- First Step: Standard Evaluation on Given Answer
        auc_ins, auc_del = run_xai_pipeline(
            args, model, processor, tokenizer, frames, video_array, tubelets, baseline_ins_frames, special_ids,
            input_ids, output_ids, output_text, ivd, log, mode_name="STANDARD", file_prefix=""
        )
        global_metrics["auc_ins"].append(auc_ins)
        global_metrics["auc_del"].append(auc_del)

        # -- Second Step: Evaluation on Ground Truth Tokens
        ground_truth = str(correct_idx)
        log(f"Ground Truth Label: {ground_truth}")
        gt_ids = tokenizer(ground_truth, return_tensors='pt', add_special_tokens=False).input_ids.to(model.device)

        if getattr(args, 'gt_forcing', False) and gt_ids is not None:    
            auc_ins_gt, auc_del_gt = run_xai_pipeline(
                args, model, processor, tokenizer, frames, video_array, tubelets, baseline_ins_frames, special_ids,
                input_ids, gt_ids, ground_truth, ivd, log, mode_name="GROUND TRUTH", file_prefix="gt_"
            )
            global_metrics["gt_auc_ins"].append(auc_ins_gt)
            global_metrics["gt_auc_del"].append(auc_del_gt)

    # Final Aggregation Log
    with open(os.path.join(args.output_dir, 'final_metrics.json'), 'w') as f:
        summary = {
            "mean_auc_ins": np.mean(global_metrics["auc_ins"]) if global_metrics["auc_ins"] else 0,
            "mean_auc_del": np.mean(global_metrics["auc_del"]) if global_metrics["auc_del"] else 0,
            "mean_gt_auc_ins": np.mean(global_metrics["gt_auc_ins"]) if global_metrics["gt_auc_ins"] else 0,
            "mean_gt_auc_del": np.mean(global_metrics["gt_auc_del"]) if global_metrics["gt_auc_del"] else 0,
        }
        json.dump(summary, f, indent=4)
    eprint(f"\nExperiment Complete. Mean metrics saved to {args.output_dir}/final_metrics.json")

if __name__ == "__main__":
    args = init_args()
    eprint(f"args:\n {args}")

    torch.manual_seed(args.manual_seed)
    random.seed(args.manual_seed)
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)

    if args.model == 'qwen':
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype=torch.bfloat16, device_map="auto")
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    else:
        model = VideoLlavaForConditionalGeneration.from_pretrained("LanguageBind/Video-LLaVA-7B-hf", dtype=torch.float16, device_map="auto")
        processor = VideoLlavaProcessor.from_pretrained("LanguageBind/Video-LLaVA-7B-hf")
    tokenizer = processor.tokenizer

    for param in model.parameters():
        param.requires_grad = False
    model.gradient_checkpointing = True

    if getattr(args, 'dataset', '') == 'TGIF':
        q_path = os.path.join(args.data_path, 'test_q.json')
        a_path = os.path.join(args.data_path, 'test_a.json')
        q_data = json.load(open(q_path))
        a_data = json.load(open(a_path))
        a_dict = {item['question_id']: item for item in a_data}
        data = []
        for q in q_data:
            if q['question_id'] in a_dict:
                merged_row = {**q, **a_dict[q['question_id']]}
                data.append(merged_row)
    elif getattr(args, 'dataset', '') == 'imagenet':
        os.makedirs(args.video_folder, exist_ok=True)
        cache_path = os.path.join(args.video_folder, f"imagenet_5k_seed{args.manual_seed}.pkl")
        # If we already downloaded it, load it instantly
        if os.path.exists(cache_path):
            eprint(f"Loading cached ImageNet subset from {cache_path}")
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
        else:
            eprint("Streaming ImageNet validation set from Hugging Face (skipping Train set)...")
            # streaming=True is the magic word that skips the 150GB train download
            dataset = load_dataset("ILSVRC/imagenet-1k", split="validation", streaming=True)
            # Shuffle using a buffer. The validation set is 50k images.
            n_images = 10
            dataset = dataset.shuffle(seed=args.manual_seed, buffer_size=n_images*10).take(n_images)
            # Extract the class label mapping from the dataset info
            info = load_dataset("ILSVRC/imagenet-1k", split="validation", streaming=True).info
            labels_feature = info.features['label']
            data = []
            eprint(f"Fetching {n_images} images... (This will take a few minutes)")
            for i, row in enumerate(dataset):
                data.append({
                    'image': row['image'], 
                    'label_name': labels_feature.int2str(row['label'])
                })
                if (i + 1) % 500 == 0:
                    eprint(f"Fetched {i + 1}/5000 images...")
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            eprint(f"Saved ImageNet subset to {cache_path}")
    elif args.data_path.endswith('csv'):
        df = pd.read_csv(args.data_path)
        data = df.to_dict(orient='records')
    elif args.data_path.endswith('jsonl'):
        data = [json.loads(q) for q in open(args.data_path, "r")]
    elif args.data_path.endswith('json'):
        data = json.load(open(args.data_path))
        if isinstance(data, dict):
            data = list(data.values())
    elif args.data_path.endswith('pkl'):
        data = pickle.load(open(args.data_path, 'rb'))
    else:
        data = load_dataset(args.data_path, "val")["val"].to_pandas()
        data = data.to_dict(orient="records")
        
    explain_vid(data, model, processor, args, tokenizer)