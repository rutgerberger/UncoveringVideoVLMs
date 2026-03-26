"""
Separate file to experiment with creating tubelets.
Creates visualizations of the tubelets we make, but
also logs explicit timing etc.
"""

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

from PIL import Image
from datasets import load_dataset

from utils import *
from args import init_args

DEFAULT_VIDEO_TOKEN = "<video>"
MAX_VIDEOS = 20

from dotenv import load_dotenv
load_dotenv()


def experiment_tubelets(data, args):
    """
    Loops through the videos, generates tubelets efficiently, tracks time, and visualizes.
    """
    now = datetime.datetime.now()
    setting = f'tubelet_exp_{now.month}{now.day}-{now.hour}{now.minute}'
    out_dir = os.path.join(args.output_dir, setting)
    args.output_dir = out_dir
    os.makedirs(args.output_dir, exist_ok=True)
    
    with open(os.path.join(out_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    log_file = os.path.join(args.output_dir, "log.txt")
    def log(msg):
        with open(log_file, "a") as f:
            f.write(str(msg) + "\n")
        print(msg) # Print to console as well

    num_videos = min(len(data), MAX_VIDEOS) if MAX_VIDEOS > 0 else len(data)
    time_records = []
    
    log(f"Starting Tubelet Experiment for {num_videos} videos...")

    # -- Main Loop
    for ivd in range(num_videos):
        log(f"\n--- Processing Video {ivd+1}/{num_videos} ---")
        
        # -- Data Retrieval
        start_data = time.time()
        row = data[ivd]
        frames, qs, cur_prompt, correct_idx = get_data(args, row)
        log(f"Data retrieved in {time.time() - start_data:.4f}s")
        
        # -- Tubelet Generation & Timing
        start_tubelets = time.time()
        video_array, tubelets = generate_tubelets_optimized(frames, args, downsample_factor=0.5)
        elapsed_time = time.time() - start_tubelets
        time_records.append(elapsed_time)
        
        unique_segments = len(np.unique(tubelets))
        log(f"SUCCESS: Generated {unique_segments} unique tubelets in {elapsed_time:.4f} seconds.")

        # -- Visualization
        vis_path = os.path.join(args.output_dir, f"video_{ivd}_tubelets.gif")
        visualize_tubelets(video_array, tubelets, vis_path)
        log(f"Saved visualization to: {vis_path}")

        # Garbage Collection
        gc.collect()
        torch.cuda.empty_cache()

    # -- Final Reporting
    mean_time = np.mean(time_records)
    log("\n=== Experiment Complete ===")
    log(f"Total Videos Processed: {num_videos}")
    log(f"Average Tubelet Generation Time: {mean_time:.4f} seconds")
    
    with open(os.path.join(args.output_dir, 'timing_results.json'), 'w') as f:
        json.dump({"times": time_records, "mean_time": mean_time}, f, indent=4)


if __name__ == "__main__":
    args = init_args()
    print(f"args:\n {args}")

    torch.manual_seed(args.manual_seed)
    random.seed(args.manual_seed)

    # Dataset Parsing block
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
        if os.path.exists(cache_path):
            print(f"Loading cached ImageNet subset from {cache_path}")
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
        else:
            print("Streaming ImageNet validation set from Hugging Face...")
            dataset = load_dataset("ILSVRC/imagenet-1k", split="validation", streaming=True)
            n_images = 10
            dataset = dataset.shuffle(seed=args.manual_seed, buffer_size=n_images*10).take(n_images)
            info = load_dataset("ILSVRC/imagenet-1k", split="validation", streaming=True).info
            labels_feature = info.features['label']
            data = []
            for i, row in enumerate(dataset):
                data.append({
                    'image': row['image'], 
                    'label_name': labels_feature.int2str(row['label'])
                })
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
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
        
    # Trigger the experiment
    experiment_tubelets(data, args)