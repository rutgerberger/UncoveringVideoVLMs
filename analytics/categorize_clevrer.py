import os
import json
import argparse

def create_orthogonal_bins(input_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading data from {input_path}...")
    with open(input_path, 'r') as f:
        data = json.load(f)
        
    # Dictionary to hold our flattened, binned questions
    bins = {
        "1_static_attribute": [],
        "2_temporal_action": [],
        "3_relational_grounding": [],
        "4_deep_causal": [],
        "5_forward_physics": [],
        "6_counterfactual": []
    }
    
    dropped_count = 0
    total_questions = 0

    for scene in data:
        vid_filename = scene['video_filename']
        scene_idx = scene['scene_index']
        
        for q in scene['questions']:
            total_questions += 1
            prog = q.get('program', [])
            q_type = q.get('question_type', '')
            
            # Define Categorial Signatures based on program logic
            is_static = (q_type == 'descriptive') and ('filter_stationary' in prog)
            is_temporal = (q_type == 'descriptive') and any(x in prog for x in ['filter_in', 'filter_out', 'filter_moving'])
            is_relational = (q_type == 'descriptive') and ('filter_collision' in prog) and ('get_col_partner' in prog)
            is_causal = (q_type == 'explanatory') and ('filter_ancestor' in prog)
            is_forward = (q_type == 'predictive') and ('unseen_events' in prog)
            is_counterfactual = (q_type == 'counterfactual') and ('get_counterfact' in prog)
            
            # Check for Orthogonality (Must match exactly ONE category)
            matches = [is_static, is_temporal, is_relational, is_causal, is_forward, is_counterfactual]
            
            if sum(matches) == 1:
                # Format exactly as your `get_data` and `explain_data` functions expect
                datapoint = {
                    "video_filename": vid_filename,
                    "scene_index": scene_idx,
                    "question_id": q['question_id'],
                    "question": q['question'],
                    "question_type": q_type,
                    "question_subtype": q.get('question_subtype', ''),
                    "program": prog,
                    "answer": q.get('answer', ''),  # for descriptive
                    "choices": q.get('choices', []) # for multi-choice
                }
                
                if is_static: bins["1_static_attribute"].append(datapoint)
                elif is_temporal: bins["2_temporal_action"].append(datapoint)
                elif is_relational: bins["3_relational_grounding"].append(datapoint)
                elif is_causal: bins["4_deep_causal"].append(datapoint)
                elif is_forward: bins["5_forward_physics"].append(datapoint)
                elif is_counterfactual: bins["6_counterfactual"].append(datapoint)
            else:
                # Ambiguous, overlapping, or non-matching questions are dropped
                dropped_count += 1

    print("\n--- Binning Summary ---")
    print(f"Total questions processed: {total_questions}")
    for bin_name, binned_data in bins.items():
        out_file = os.path.join(output_dir, f"{bin_name}.json")
        with open(out_file, 'w') as f:
            json.dump(binned_data, f, indent=4)
        print(f"[{bin_name}]: {len(binned_data)} questions saved to {out_file}")
        
    print(f"\nDropped {dropped_count} questions to enforce strict orthogonality.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bin CLEVRER dataset for XAI analysis")
    parser.add_argument("--input_path", type=str, required=True, help="Path to train.json or val.json")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save binned JSONs")
    args = parser.parse_args()
    
    create_orthogonal_bins(args.input_path, args.output_dir)