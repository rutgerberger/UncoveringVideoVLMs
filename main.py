import torch
import numpy as np
import pandas as pd
import json
import datetime
import time
import os
import gc
import pickle
import random

from skimage.segmentation import slic, mark_boundaries
from PIL import Image
from datasets import load_dataset
from transformers import VideoLlavaForConditionalGeneration, VideoLlavaProcessor
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

from utils import *
from method import spix_gradient_iterative, frame_redundancy
from args import init_args

from iGOS.method import iGOS_p
from utils import evaluate_auc_pixel

DEFAULT_VIDEO_TOKEN = "<video>"
MAX_VIDEOS = 20

from dotenv import load_dotenv
load_dotenv()

def run_xai_pipeline(args, model, processor, tokenizer, frames, video_array, tubelets, baseline_ins_frames, special_ids, 
                     input_ids, output_ids, target_text, ivd, log_func, mode_name, file_prefix):
    """
    This runs the following pipeline:
        Keyword Extraction -> SPIX -> Explicit Logging -> AUC Evaluation -> Visualization.

    returns the auc metrics.
    """
    start = time.time()
    
    # -- Finding Keywords and Probabilities
    eprint(f"{ivd+1}/{args.num_videos}: Selecting Important Tubelets ({mode_name}).")
    positions, keywords = find_keywords(
        args, model, processor, input_ids, output_ids, frames, baseline_ins_frames, 
        target_text, tokenizer=tokenizer, use_yake=args.use_yake, special_ids=special_ids
    )
    log_func(f"Extracted Keywords: {keywords} at positions {positions}")
    full_ids = torch.cat((input_ids, output_ids), dim=1)

    if getattr(args, 'method', 'spix') == 'igos':
        eprint("Running iGOS Continuous Pixel Optimization...")
        baseline_ins = get_baseline_insertion(args, video_array)
        frames_ins_base = [Image.fromarray(f.astype(np.uint8)) for f in baseline_ins]
        
        # Unpack the exact tuple return
        raw_mask, l_del, l_ins, l_l1, l_tv, l_l2, _, _ = iGOS_p(
            args, model, processor, full_ids, output_ids, frames, frames_ins_base, positions,
            lr=1000, L1=1, L2=1, L3=20, size=28 # Explicitly using original hyperparameters
        )
        
        # Scale the raw mask up to video resolution for the evaluator
        target_H, target_W = video_array.shape[1], video_array.shape[2]
        up_mask = torch.nn.functional.interpolate(raw_mask.data, size=(target_H, target_W), mode='bilinear', align_corners=False)
        igos_mask_numpy = up_mask.squeeze().cpu().numpy()
        
        if getattr(args, 'save_visuals', True):
            eprint(f"{ivd+1}/{args.num_videos}: Saving iGOS Heatmaps...")
            save_igos_heatmaps(
                video_array, 
                igos_mask_numpy, 
                args.output_dir, 
                prefix=f"{ivd}_{file_prefix}igos"
            )

        # Evaluate using the pixel evaluator
        auc_ins, auc_del = evaluate_auc_pixel(
            args, model, processor, full_ids, output_ids, frames, igos_mask_numpy, 
            ivd=ivd, positions=positions)#, num_steps=20
        
        #)
        
        log_func(f"iGOS Time: {(time.time() - start):.2f}s")
        log_func(f"AUC iGOS - Ins: {auc_ins:.4f} | Del: {auc_del:.4f}")
        
        return auc_ins, auc_del, auc_ins, auc_del

    prob_orig = get_prob(args, model, processor, full_ids, output_ids, frames, positions)

    # -- Main method is called here to acquire the selected tubelets and scores
    selected_ins, selected_del, scores_ins, scores_del = spix_gradient_iterative(
        args, model, tokenizer, processor, input_ids, output_ids, frames, tubelets,
        positions=positions, stages=3, iters_per_stage=20, index=ivd
    )
    # -- Calculating Union and Intersection Masks
    unique_tubes = np.unique(tubelets)
    raw_union = set(selected_ins) | set(selected_del)
    raw_inter = set(selected_ins) & set(selected_del)
    # Scores are calculated similarly
    scores_union = {t: max(scores_ins.get(t, 0), scores_del.get(t, 0)) for t in raw_union}
    scores_inter = {t: min(scores_ins.get(t, 0), scores_del.get(t, 0)) for t in raw_inter}
    # Sort the lists descending by their new scores so evaluate_auc ranks them correctly
    selected_union = sorted(list(raw_union), key=lambda t: scores_union[t], reverse=True)
    selected_inter = sorted(list(raw_inter), key=lambda t: scores_inter[t], reverse=True)

    # -- Baseline Videos: What Happens to video
    baseline_ins = get_baseline_insertion(args, video_array)
    baseline_del = get_baseline_deletion(args, video_array)
    
    # -- Calculate probability drops (logging purposes)
    # Insertion: Foreground is original video, Background is baseline_ins
    frames_ins = apply_universal_mask(video_array, baseline_ins, tubelets, selected_ins)
    prob_ins = get_prob(args, model, processor, full_ids, output_ids, frames_ins, positions)
    # Deletion: Foreground is original video (keep_tubes), Background is masked video
    keep_tubes_del = [t for t in unique_tubes if t not in selected_del]
    frames_del = apply_universal_mask(video_array, baseline_del, tubelets, keep_tubes_del)
    prob_del = get_prob(args, model, processor, full_ids, output_ids, frames_del, positions)

    # -- Logging
    log_func(f"\n=== {mode_name} TUBELETS SEARCH LOG ===")
    log_func(f"Created tubelets in {(time.time() - start):.2f}s")
    log_func(f"Final Insertion tubelets: {selected_ins} ({len(selected_ins)}/{len(unique_tubes)})")
    log_func(f"Final Deletion tubelets: {selected_del} ({len(selected_del)}/{len(unique_tubes)})")
    log_func(f"Union mask size: {len(selected_union)} | Intersection mask size: {len(selected_inter)}")
    log_func(f"Prob Original: {prob_orig:.5f}")
    log_func(f"Prob Insertion: {prob_ins:.5f} (Diff: {prob_orig - prob_ins:.5f})")
    log_func(f"Prob Deletion: {prob_del:.5f} (Diff: {prob_orig - prob_del:.5f})")

    # -- Metric calculation
    auc_ins_union, auc_del_union = evaluate_auc(args, model, processor, full_ids, output_ids, frames, tubelets, selected_union, ivd=f"{ivd}_union", positions=positions)#, num_steps=20)
    auc_ins_inter, auc_del_inter = evaluate_auc(args, model, processor, full_ids, output_ids, frames, tubelets, selected_inter, ivd=f"{ivd}_inter", positions=positions)#, num_steps=20)
    
    log_func(f"AUC Union - Ins: {auc_ins_union:.4f} | Del: {auc_del_union:.4f}")
    log_func(f"AUC Inter - Ins: {auc_ins_inter:.4f} | Del: {auc_del_inter:.4f}")

    # -- Saving gifs for visualization
    if getattr(args, 'save_visuals', True):
        eprint(f"{ivd+1}/{args.num_videos}: Visualizing the Masked Tubelets ({mode_name}).")
        # Mask Visualization (using union mask as the default masking visual)
        keep_tubes_vis = [t for t in unique_tubes if t not in selected_union]
        visualize_spix(video_array, baseline_del, tubelets, keep_tubes_vis, os.path.join(args.output_dir, f"{ivd}_{file_prefix}mask_union.gif"))
        # Heatmaps Vis
        visualize_heatmap(video_array, tubelets, scores_ins, os.path.join(args.output_dir, f"{ivd}_{file_prefix}highlight_ins.gif"))
        visualize_heatmap(video_array, tubelets, scores_del, os.path.join(args.output_dir, f"{ivd}_{file_prefix}highlight_del.gif"))
        visualize_heatmap(video_array, tubelets, scores_union, os.path.join(args.output_dir, f"{ivd}_{file_prefix}highlight_union.gif"))
        visualize_heatmap(video_array, tubelets, scores_inter, os.path.join(args.output_dir, f"{ivd}_{file_prefix}highlight_inter.gif"))
        
    return auc_ins_union, auc_del_union, auc_ins_inter, auc_del_inter

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

    num_videos = min(len(data), args.num_videos) if args.num_videos > 0 else len(data)
    global_metrics = {
        "auc_ins_union": [], "auc_del_union": [],
        "auc_ins_inter": [], "auc_del_inter": [],
        "gt_auc_ins_union": [], "gt_auc_del_union": [],
        "gt_auc_ins_inter": [], "gt_auc_del_inter": []
    }
    
    # -- Main Loop - performs per given video
    for ivd in range(num_videos):
        # -- Data Processing / Logging
        eprint(f"{ivd+1}/{num_videos}: Retrieving data.")
        start = time.time()
        id = random.randint(0,len(data)-1)
        row = data[id]
        try:
            frames, qs, cur_prompt, correct_idx = get_data(args, row)
        except:
            continue
        log(f"\n\n=== Question {ivd+1}/{num_videos} ===\n{qs}")
        eprint(f"Retrieved data in {time.time() - start}s")
        if getattr(args, 'dataset', '') != 'imagenet':
            video_desc = create_description(args, model, processor, frames, tokenizer)
            log(f"Video Description: {video_desc}")
        
        # -- Creation of Video Tubelets
        start = time.time()
        eprint(f"{ivd+1}/{num_videos}: Generating tubelets.")
        video_array, tubelets = generate_tubelets_optimized(frames, args)
        eprint(f"Generated tubelets in {time.time() - start}s")
        
        # -- Gathering Original Model Output
        eprint(f"{ivd+1}/{num_videos}: Acquiring Model Output.")
        prompt = f"USER: {DEFAULT_VIDEO_TOKEN}\n{qs}. ASSISTANT:"
        input_ids, output_ids, output_text = get_model_response(args, model, processor, tokenizer, prompt, frames)
        log(f"Model Answer: {output_text}")
        eprint(f"Model Answer: {output_text}")

        # -- Precompute constants used for both standard and GT pipelines
        baseline_ins = get_baseline_insertion(args, video_array)
        baseline_ins_frames = [Image.fromarray(f) for f in baseline_ins]
        special_ids = [tokenizer.pad_token_id, tokenizer.eos_token_id, tokenizer.bos_token_id]
        special_ids = [idx for idx in special_ids if idx is not None]

        # -- Runs standard tubelet selection & vis. pipeline with model's answer
        auc_ins_union, auc_del_union, auc_ins_inter, auc_del_inter = run_xai_pipeline(
            args, model, processor, tokenizer, frames, video_array, tubelets, baseline_ins_frames, special_ids,
            input_ids, output_ids, output_text, ivd, log, mode_name="STANDARD", file_prefix=""
        )
        global_metrics["auc_ins_union"].append(auc_ins_union)
        global_metrics["auc_del_union"].append(auc_del_union)
        global_metrics["auc_ins_inter"].append(auc_ins_inter)
        global_metrics["auc_del_inter"].append(auc_del_inter)

        # -- Runs tubelet & visualization selection pipeline with ground truth answer
        ground_truth = str(correct_idx)
        gt_ids = tokenizer(ground_truth, return_tensors='pt', add_special_tokens=False).input_ids.to(model.device)
        log(f"Ground Truth Label: {ground_truth}")

        if getattr(args, 'gt_forcing', False) and gt_ids is not None:    
            auc_ins_union_gt, auc_del_union_gt, auc_ins_inter_gt, auc_del_inter_gt = run_xai_pipeline(
                args, model, processor, tokenizer, frames, video_array, tubelets, baseline_ins_frames, special_ids,
                input_ids, gt_ids, ground_truth, ivd, log, mode_name="GROUND TRUTH", file_prefix="gt_"
            )
            global_metrics["gt_auc_ins_union"].append(auc_ins_union_gt)
            global_metrics["gt_auc_del_union"].append(auc_del_union_gt)
            global_metrics["gt_auc_ins_inter"].append(auc_ins_inter_gt)
            global_metrics["gt_auc_del_inter"].append(auc_del_inter_gt)
        # Garbage Collector
        gc.collect()
        torch.cuda.empty_cache()

    # -- Final Aggregation Log
    with open(os.path.join(args.output_dir, 'final_metrics.json'), 'w') as f:
        summary = {
            "mean_auc_ins_union": np.mean(global_metrics["auc_ins_union"]) if global_metrics["auc_ins_union"] else 0,
            "mean_auc_del_union": np.mean(global_metrics["auc_del_union"]) if global_metrics["auc_del_union"] else 0,
            "mean_auc_ins_inter": np.mean(global_metrics["auc_ins_inter"]) if global_metrics["auc_ins_inter"] else 0,
            "mean_auc_del_inter": np.mean(global_metrics["auc_del_inter"]) if global_metrics["auc_del_inter"] else 0,
            "gt_auc_ins_union": np.mean(global_metrics["gt_auc_ins_union"]) if global_metrics["gt_auc_ins_union"] else 0,
            "gt_auc_del_union": np.mean(global_metrics["gt_auc_del_union"]) if global_metrics["gt_auc_del_union"] else 0,
            "gt_auc_ins_inter": np.mean(global_metrics["gt_auc_ins_inter"]) if global_metrics["gt_auc_ins_inter"] else 0,
            "gt_auc_del_inter": np.mean(global_metrics["gt_auc_del_inter"]) if global_metrics["gt_auc_del_inter"] else 0,
        }
        json.dump(summary, f, indent=4)
    eprint(f"\nExperiment Complete. Mean metrics saved to {args.output_dir}/final_metrics.json")

if __name__ == "__main__":
    eprint("Hello world")
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
    eprint("Model Mapping:", model.hf_device_map)

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