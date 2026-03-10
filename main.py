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
# Assuming spix, apply_mask, spix_optimized, frame_redundancy, and AUC methods 
# are imported from your method/utils files as defined.
from args import init_args

DEFAULT_VIDEO_TOKEN = "<video>"
MAX_VIDEOS = 5
WANT_RANDOM = False

def explain_vid(data, model, processor, args, tokenizer):
    """
    Generates 
    - heatmap visualizations for each entry in the data
    - shapley (interaction) values per frame
    - AUC curves for XAI evaluation
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

    num_videos = min(len(data), MAX_VIDEOS)
    for ivd in range(num_videos):
        eprint(f"{ivd+1}/{num_videos}: Retrieving data.")
        start = time.time()
        row = data[ivd]
        frames, qs, cur_prompt, correct_idx = get_data(args, row)
        log(f"\n\n=== Question {ivd+1}/{num_videos} ===\n{qs}")
        eprint(f"Retrieved data in {time.time() - start}s")
        video_desc = create_description(args, model, processor, frames, tokenizer)
        log(f"Video Description: {video_desc}")
        
        start = time.time()
        eprint(f"{ivd+1}/{num_videos}: Generating tubelets.")
        video_array, tubelets = generate_tubelets(frames, args)
        eprint(f"Generated tubelets in {time.time() - start}s")
        
        eprint(f"{ivd+1}/{num_videos}: Creating visualizations.")
        visualize_frames(frames, os.path.join(args.output_dir, f"{ivd}_frames.gif"))
        visualize_tubelets(video_array, tubelets, os.path.join(args.output_dir, f"{ivd}_slic.gif"))
        
        eprint(f"{ivd+1}/{num_videos}: Acquiring Model Output.")
        prompt = f"USER: {DEFAULT_VIDEO_TOKEN}\n{qs}. ASSISTANT:"
        input_ids, output_ids, output_text = get_model_response(args, model, processor, tokenizer, prompt, frames)
        log(f"Model Answer: {output_text}")

        # // FIND KEYWORDS FOR GENERATED ANSWER 
        blurred_video_np = precompute_blurred_video(video_array)
        blur_frames = [Image.fromarray(f) for f in blurred_video_np]
        special_ids = [tokenizer.pad_token_id, tokenizer.eos_token_id, tokenizer.bos_token_id]
        special_ids = [idx for idx in special_ids if idx is not None]
        
        positions, keywords = find_keywords(
            args, model, processor, input_ids, output_ids, frames, blur_frames, 
            output_text, tokenizer=tokenizer, use_yake=args.use_yake, special_ids=special_ids
        )
        log(f"Extracted Keywords: {keywords} at positions {positions}")
        # //

        gt_ids = None
        # Teacher forcing: Use ground truth instead of generated answer
        ground_truth = str(correct_idx)
        gt_ids = tokenizer(ground_truth, return_tensors='pt', add_special_tokens=False).input_ids.to(model.device)
        log(f"Ground Truth: {ground_truth}")

        # Analysis on frame level redundancies
        if args.frame_analysis:
            eprint(f"{ivd+1}/{num_videos}: Running Frame-Level Shapley & Interaction Analysis...")
            start = time.time()
            shapley_values, sorted_frames, interaction_matrix = frame_redundancy(
                args, model, processor, input_ids, output_ids, frames, compute_interactions=True
            )
            log(f"\n\n=== SHAPLEY FRAME ANALYSIS LOG ===\n")
            log(f"Computed Shapley values in {time.time() - start:.2f}s")
            log(f"Sorted Frames (Highest = Most Important/Least Redundant):")
            for frame_idx, shapley_val in sorted_frames:
                log(f"  Frame {frame_idx}: {shapley_val:.5f}")
            matrix_out_path = os.path.join(args.output_dir, f"{ivd}_interaction_matrix.png")
            visualize_interaction_matrix(interaction_matrix, matrix_out_path)
            eprint(f"Saved Interaction Matrix heatmap to {matrix_out_path}")

        start = time.time()

        if WANT_RANDOM:
            full_ids = torch.cat((input_ids, output_ids), dim=1)
            prob_orig = get_prob(args, model, processor, full_ids, output_ids, frames, positions)
            unique_tubes = np.unique(tubelets)
            tubes_to_mask = np.random.choice(unique_tubes, size=int(len(unique_tubes)*0.2), replace=False)
            final_tubelets = [t for t in unique_tubes if t not in tubes_to_mask]
            masked_frames = apply_mask(frames, tubelets, final_tubelets)
            prob_masked = get_prob(args, model, processor, full_ids, output_ids, masked_frames, positions)
            eprint(f"Prob orig: {prob_orig:.5f}, Prob masked: {prob_masked:.5f}, Diff: {prob_orig - prob_masked:.5f}")
        else:
            eprint(f"{ivd+1}/{num_videos}: Selecting Important Tubelets.")
            full_ids = torch.cat((input_ids, output_ids), dim=1)
            prob_orig = get_prob(args, model, processor, full_ids, output_ids, frames, positions)

            final_tubelets, tubelet_scores = spix_optimized(
                args, model, processor, input_ids, output_ids, frames, tubelets,
                positions=positions, opt_mode=args.opt_mode, use_dynamic_lambda=args.use_dynamic_lambda
            )
            
            frames_ins = apply_mask(frames, tubelets, final_tubelets)
            prob_ins = get_prob(args, model, processor, full_ids, output_ids, frames_ins, positions)
            unique_tubes = np.unique(tubelets)
            tubes_del = [t for t in unique_tubes if t not in final_tubelets]
            frames_del = apply_mask(frames, tubelets, tubes_del)
            prob_del = get_prob(args, model, processor, full_ids, output_ids, frames_del, positions)
            
            log(f"\n\n=== TUBELETS SEARCH LOG ===\n")
            log(f"Created tubelets in {time.time() - start}s")
            log(f"Final tubelets: {final_tubelets} ({len(final_tubelets)}/{len(unique_tubes)})")
            log(f"Prob Original: {prob_orig:.5f}")
            log(f"Prob Insertion: {prob_ins:.5f} (Diff: {prob_orig - prob_ins:.5f})")
            log(f"Prob Deletion: {prob_del:.5f} (Diff: {prob_orig - prob_del:.5f})")

            auc_ins, auc_del = evaluate_auc(
                args, model, processor, full_ids, output_ids, frames, tubelets, final_tubelets, 
                positions=positions, num_steps=20
            )
            log(f"AUC Insertion: {auc_ins:.4f}")
            log(f"AUC Deletion: {auc_del:.4f}")
            # Rename the plot so it doesn't overwrite if GT is also run
            os.rename(os.path.join(args.output_dir, "auc_curves.png"), os.path.join(args.output_dir, f"{ivd}_auc_curves.png"))

            eprint(f"{ivd+1}/{MAX_VIDEOS}: Visualizing the Masked Tubelets.")
            tubes_to_vis = [t for t in unique_tubes if t not in final_tubelets]
            visualize_spix(frames, tubelets, tubes_to_vis, os.path.join(args.output_dir, f"{ivd}_mask.gif"))
            visualize_heatmap(video_array, tubelets, tubelet_scores, os.path.join(args.output_dir, f"{ivd}_highlight.gif"))
            eprint(f"{ivd+1}/{MAX_VIDEOS}: Done.")

        # GROUND TRUTH FORCING
        start = time.time()
        if gt_ids is not None:    
            eprint(f"{ivd+1}/{num_videos}: Ground truth forcing.")
            output_ids = gt_ids

            gt_positions, gt_keywords = find_keywords(
                args, model, processor, input_ids, output_ids, frames, blur_frames, 
                ground_truth, tokenizer=tokenizer, use_yake=args.use_yake, special_ids=special_ids
            )
            log(f"[GT] Extracted Keywords: {gt_keywords} at positions {gt_positions}")
            
            eprint(f"{ivd+1}/{num_videos}: Selecting Important Tubelets (GT).")
            full_ids = torch.cat((input_ids, output_ids), dim=1)
            prob_orig = get_prob(args, model, processor, full_ids, output_ids, frames, gt_positions)
            
            final_tubelets, tubelet_scores = spix_optimized(
                args, model, processor, input_ids, output_ids, frames, tubelets,
                positions=gt_positions, opt_mode=args.opt_mode, use_dynamic_lambda=args.use_dynamic_lambda
            )
            
            frames_ins = apply_mask(frames, tubelets, final_tubelets)
            prob_ins = get_prob(args, model, processor, full_ids, output_ids, frames_ins, gt_positions)
            unique_tubes = np.unique(tubelets)
            tubes_del = [t for t in unique_tubes if t not in final_tubelets]
            frames_del = apply_mask(frames, tubelets, tubes_del)
            prob_del = get_prob(args, model, processor, full_ids, output_ids, frames_del, gt_positions)
            
            log(f"\n\n=== GROUND TRUTH FORCING LOG ===\n")
            log(f"Created tubelets in {time.time() - start}s")
            log(f"Final tubelets: {final_tubelets} ({len(final_tubelets)}/{len(unique_tubes)})")
            log(f"Prob Original: {prob_orig:.5f}")
            log(f"Prob Insertion: {prob_ins:.5f} (Diff: {prob_orig - prob_ins:.5f})")
            log(f"Prob Deletion: {prob_del:.5f} (Diff: {prob_orig - prob_del:.5f})")

            auc_ins_gt, auc_del_gt = evaluate_auc(
                args, model, processor, full_ids, output_ids, frames, tubelets, final_tubelets, 
                positions=gt_positions, num_steps=20
            )
            log(f"[GT] AUC Insertion: {auc_ins_gt:.4f}")
            log(f"[GT] AUC Deletion: {auc_del_gt:.4f}")
            os.rename(os.path.join(args.output_dir, "auc_curves.png"), os.path.join(args.output_dir, f"{ivd}_gt_auc_curves.png"))

            eprint(f"{ivd+1}/{MAX_VIDEOS}: Visualizing the Masked Tubelets (GT).")
            tubes_to_vis = [t for t in unique_tubes if t not in final_tubelets]
            visualize_spix(frames, tubelets, tubes_to_vis, os.path.join(args.output_dir, f"{ivd}_gt_mask.gif"))
            visualize_heatmap(video_array, tubelets, tubelet_scores, os.path.join(args.output_dir, f"{ivd}_gt_highlight.gif"))
            eprint(f"{ivd+1}/{MAX_VIDEOS}: Done.")

if __name__ == "__main__":
    args = init_args()
    eprint(f"args:\n {args}")

    torch.manual_seed(args.manual_seed)
    random.seed(args.manual_seed)
    #disable torch init
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
        import os
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