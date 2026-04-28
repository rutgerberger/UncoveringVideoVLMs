from new_utils.logging import eprint

eprint("Loading libraries 1/3...")
import os
import gc
import json
import time
import pickle
import random
import datetime

eprint("Loading libraries 2/3...")
import torch
import numpy as np
import pandas as pd

from PIL import Image
from dotenv import load_dotenv
from datasets import load_dataset

from transformers import VideoLlavaForConditionalGeneration, VideoLlavaProcessor
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

eprint("Importing files 3/3...")

from new_utils import *
from optimizer import process_video, spix_gradient_iterative
from args import init_args

load_dotenv()
DEFAULT_VIDEO_TOKEN = "<video>"

def xai_method(args, model, tokenizer, processor, input_ids, output_ids, frames, tubelets, baseline_ins_arr, baseline_del_arr, positions, ivd):
    """
    Returns the XAI method of preference (for testing)
    """
    method = getattr(args, 'method', 'cmaes')
    if method == 'cmaes':
        return process_video(
            args, model, tokenizer, processor, input_ids, output_ids, 
            frames, tubelets, baseline_ins_arr, baseline_del_arr, positions=positions
        )
    else:
        return spix_gradient_iterative(
            args, model, tokenizer, processor, input_ids, output_ids, frames, tubelets,
            positions=positions, max_stages=getattr(args, 'max_stages', 2), index=ivd
        )



def explain_vid(args, model, processor, tokenizer, frames, video_array, tubelets, 
                     baseline_ins_arr, baseline_del_arr, baseline_ins_frames, baseline_del_frames, 
                     special_ids, input_ids, output_ids, question_text, ground_truth, model_answer, 
                     target_text, ivd, log_func, mode_name, file_prefix):
    """
    Runs XAI pipeline for a single video and handles logging
    """
    start = time.time()

    eprint(f"{ivd+1}/{args.num_videos}: Selecting Keywords ({mode_name}).")
    positions, keywords = find_keywords(
        args, model, processor, input_ids, output_ids, frames, baseline_ins_frames, 
        target_text, tokenizer=tokenizer, use_yake=args.use_yake, special_ids=special_ids
    )

    if getattr(args, 'method', '') == 'igos':
        auc_ins, auc_del = perform_igos()
        return auc_ins, auc_del, auc_ins, auc_del

    eprint(f"{ivd+1}/{args.num_videos}: Optimizing Tubelet Weights")
    
    # Unpack the unified mask outputs (Assuming process_video returns 3 items now: selected, scores, metrics)
    selected_tubes, scores, metrics = xai_method(
        args, model, tokenizer, processor, input_ids, output_ids, frames, tubelets, 
        baseline_ins_arr, baseline_del_arr, positions, ivd
    )

    # Evaluation (calculating metrics, etc.)
    full_ids = torch.cat((input_ids, output_ids), dim=1)
    prob_orig = get_prob(args, model, processor, full_ids, output_ids, frames, positions=positions, tokenizer=tokenizer)
    prob_baseline_del = get_prob(args, model, processor, full_ids, output_ids, baseline_del_frames, positions=positions, tokenizer=tokenizer)
    prob_baseline_ins = get_prob(args, model, processor, full_ids, output_ids, baseline_ins_frames, positions=positions, tokenizer=tokenizer)

    # ... for evaluation, we show what happens when deleting / inserting our mask!
    unique_tubes = np.unique(tubelets)
    k_fraction = 0.25
    k_tubes = max(1, int(len(unique_tubes) * k_fraction))

    top_final = selected_tubes[:k_tubes]
    frames_ins = apply_universal_mask(video_array, baseline_ins_arr, tubelets, top_final)
    prob_ins = get_prob(args, model, processor, full_ids, output_ids, frames_ins, positions=positions, tokenizer=tokenizer)
    
    keep_tubes_del = [t for t in unique_tubes if t not in top_final]
    frames_del = apply_universal_mask(video_array, baseline_del_arr, tubelets, keep_tubes_del)
    prob_del = get_prob(args, model, processor, full_ids, output_ids, frames_del, positions=positions, tokenizer=tokenizer)

    top_k = min(5, len(selected_tubes))
    fmt_scores = {t: f"{scores.get(t, 0):.4f}" for t in selected_tubes[:top_k]}
 
    # Get AUC metrics
    auc_ins, auc_del = evaluate_auc(
        args, model, processor, full_ids, output_ids, frames, video_array, tubelets, 
        selected_tubes, baseline_ins_arr, baseline_del_arr, ivd=f"{ivd}_mask", positions=positions
    )
    
    #All the data will be logged in log.txt
    log_experiment(args, log_func, ivd, question_text, ground_truth, model_answer, keywords, positions,
        prob_orig, prob_baseline_del, prob_baseline_ins, prob_ins, prob_del, auc_ins, auc_del,
        metrics, metrics, top_k, fmt_scores, fmt_scores, fmt_scores, 
        selected_tubes, selected_tubes, selected_tubes, unique_tubes, k_fraction, mode_name, start
    )

    if getattr(args, 'save_visuals', True):
        eprint(f"{ivd+1}/{args.num_videos}: Visualizing the Masked Tubelets ({mode_name}).")
        # Binary cutout of top 20%
        keep_tubes_vis = [t for t in unique_tubes if t not in top_final]
        visualize_spix(video_array, baseline_del_arr, tubelets, keep_tubes_vis, os.path.join(args.output_dir, f"{ivd}_{file_prefix}mask_cutout.gif"))
        # Continuous heatmaps
        visualize_heatmap(video_array, tubelets, scores, os.path.join(args.output_dir, f"{ivd}_{file_prefix}highlight_mask.gif"))

    return {
        "auc_ins": auc_ins,
        "auc_del": auc_del,
        "prob_orig": prob_orig,
        "prob_baseline_del": prob_baseline_del,
        "prob_baseline_ins": prob_baseline_ins,
        "prob_ins": prob_ins,
        "prob_del": prob_del
    }

def explain_data(data, model, processor, args, tokenizer):
    """
    param data: pre-loaded list of data points (dict) - packed with get_data util
    param model: HF VLM model
    param processor: processor alongside the model
    param args: run arguments
    param tokenizer: "  "  alongside the model
    for each datapoint, runs XAI pipeline
    """
    # First we set up logging structure
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
    # For evaluation
    num_videos = min(len(data), args.num_videos) if args.num_videos > 0 else len(data)
    global_metrics = []
    global_gt_metrics = []
    
    if args.randomize_data == True:
        random.shuffle(data)

    for ivd in range(num_videos):
        eprint(f"\n{ivd+1}/{num_videos}: Retrieving data.")
        start = time.time()

        #Load current question, frame and ground truth (if avail)
        try:
            frames, qs, cur_prompt, correct_idx = get_data(args, data[ivd])
            ground_truth_label = str(correct_idx)
            prompt = f"USER: {DEFAULT_VIDEO_TOKEN}\n{qs}. ASSISTANT:"
        except Exception as e:
            eprint(f"Error loading row {ivd}: {e}")
            continue
        eprint(f"Retrieved data in {time.time() - start:.2f}s")

        #Get original answer of the model (with token IDs)
        input_ids, output_ids, output_text = get_model_response(args, model, processor, tokenizer, prompt, frames)
        special_ids = [idx for idx in [tokenizer.pad_token_id, tokenizer.eos_token_id, tokenizer.bos_token_id] if idx is not None]

        #Obtain baseline videos (and save if desired)- we do it here once (saves time)
        eprint(f"{ivd+1}/{args.num_videos}: Creating Tubelets and Baseline Frames.")
        video_array, tubelets = generate_tubelets_optimized(frames, args)
        baseline_ins_arr = get_baseline_insertion(args, video_array)
        baseline_del_arr = get_baseline_deletion(args, video_array)
        baseline_ins_frames = [Image.fromarray(f.astype(np.uint8)) for f in baseline_ins_arr]
        baseline_del_frames = [Image.fromarray(f.astype(np.uint8)) for f in baseline_del_arr]
        if getattr(args, 'save_visuals', True):
            eprint(f"Visualizing baseline frames for video {ivd}...")
            visualize_frames(baseline_ins_frames, os.path.join(args.output_dir, f"{ivd}_baseline_insertion.gif"))
            visualize_frames(baseline_del_frames, os.path.join(args.output_dir, f"{ivd}_baseline_deletion.gif"))

        #Main pipeline
        metrics = explain_vid(
            args, model, processor, tokenizer, frames, video_array, tubelets, 
            baseline_ins_arr, baseline_del_arr, baseline_ins_frames, baseline_del_frames, special_ids,
            input_ids, output_ids, qs, ground_truth_label, output_text, output_text, ivd, log, 
            mode_name="STANDARD", file_prefix=""
        )
        global_metrics.append(metrics)

        if getattr(args, 'gt_forcing', False):    
            gt_ids = tokenizer(ground_truth_label, return_tensors='pt', add_special_tokens=False).input_ids.to(model.device)
            if gt_ids is not None:
                gt_metrics = explain_vid(
                    args, model, processor, tokenizer, frames, video_array, tubelets, 
                    baseline_ins_arr, baseline_del_arr, baseline_ins_frames, baseline_del_frames, special_ids,
                    input_ids, gt_ids, qs, ground_truth_label, output_text, ground_truth_label, ivd, log, 
                    mode_name="GROUND TRUTH", file_prefix="gt_"
                )
                global_gt_metrics.append(gt_metrics)

        gc.collect()
        torch.cuda.empty_cache()

    if global_metrics:
        log_metrics(args, global_metrics)
    if global_gt_metrics:
        log_metrics(args, global_gt_metrics, prefix="GT-")

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
        
    explain_data(data, model, processor, args, tokenizer)