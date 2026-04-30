import argparse
import json
import pandas as pd
import sys
from pathlib import Path

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Analyze AUC scores from multiple JSONL files.")
    parser.add_argument('files', nargs='+', help='A list of JSONL filenames to process')
    args = parser.parse_args()

    data = []
    #print(args.files)
    for file in args.files:
        try:
            with open(file, 'r') as f:
                short_name = Path(file).parent.name 

                for line in f:
                    line = line.strip()
                    if line:
                        record = json.loads(line)
                        
                        # Extract the scores
                        auc_ins = record.get('AUC Ins', pd.NA)
                        auc_del = record.get('AUC Del', pd.NA)
                        
                        # If the value is not NA and is less than 0, replace it with pd.NA
                        if pd.notna(auc_ins) and (float(auc_ins) < 0 or float(auc_ins) > 1):
                            auc_ins = pd.NA
                        if pd.notna(auc_del) and (float(auc_del) < 0 or float(auc_del) > 1):
                            auc_del = pd.NA

                        data.append({
                            'File': short_name, 
                            'video_index': record['video_index'],
                            'AUC_Ins': auc_ins,
                            'AUC_Del': auc_del
                        })
        except FileNotFoundError:
            print(f"Warning: File '{file}' not found. Skipping.")
        except json.JSONDecodeError:
            print(f"Warning: File '{file}' contains invalid JSON. Skipping line.")

    if not data:
        print("No valid data found in the provided files.")
        sys.exit(1)

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # 2. Side-by-Side Comparison & Mean
    print("\n" + "="*50)
    print(" SIDE-BY-SIDE AUC COMPARISON")
    print("="*50)
    
    # Pivot the table to have video_index as rows, and files as columns
    pivot_df = df.pivot(index='video_index', columns='File', values=['AUC_Ins', 'AUC_Del'])
    
    # Compute the mean for each column (pd.NA values are automatically ignored in the mean)
    mean_row = pivot_df.mean().to_frame().T
    mean_row.index = ['MEAN']
    
    # Combine the data and the mean row
    combined_df = pd.concat([pivot_df, mean_row])
    
    # Fill NaN values with a placeholder for cleaner printing
    combined_df = combined_df.fillna("N/A")
    print(combined_df.to_string())
    print("\n")


    # 3. Highlight the VERY WELL and VERY BAD videos per file
    print("="*50)
    print(" BEST (WELL) & WORST (BAD) VIDEOS PER FILE")
    print("="*50)
    
    for file in args.files:
        short_name = Path(file).parent.name 
        
        # Filter data using the short_name!
        file_data = df[df['File'] == short_name].dropna(subset=['AUC_Ins', 'AUC_Del'])
        
        if file_data.empty:
            continue
            
        # Find the index of the max and min values for AUC Ins
        best_ins = file_data.loc[file_data['AUC_Ins'].idxmax()]
        worst_ins = file_data.loc[file_data['AUC_Ins'].idxmin()]
        
        # Find the index of the max and min values for AUC Del
        best_del = file_data.loc[file_data['AUC_Del'].idxmin()] 
        worst_del = file_data.loc[file_data['AUC_Del'].idxmax()]
        
        print(f"► FILE: {short_name}")
        print(f"  [AUC Ins] 🟢 BEST : Video {int(best_ins['video_index']):>2} (Score: {best_ins['AUC_Ins']:.3f})")
        print(f"            🔴 WORST: Video {int(worst_ins['video_index']):>2} (Score: {worst_ins['AUC_Ins']:.3f})")
        print(f"  [AUC Del] 🟢 BEST : Video {int(best_del['video_index']):>2} (Score: {best_del['AUC_Del']:.3f})")
        print(f"            🔴 WORST: Video {int(worst_del['video_index']):>2} (Score: {worst_del['AUC_Del']:.3f})")
        print("-" * 50)

if __name__ == "__main__":
    main()