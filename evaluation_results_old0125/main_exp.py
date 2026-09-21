import os
import json
import pandas as pd

# 1. Configuration
BASE_PATH = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/diffusion/evaluation_results/"
TARGET_MODELS = ["old_niu_1b_100b_2node", "niu_1B_100b_daum_2node", "niu_1B_100b_daum_loose_2node", "niu_1B_tight_2node","niu_main_exp"]
OUTPUT_FILE = "xiaorong.xlsx"

def format_score(value):
    """Converts 0.3427 -> 34.28"""
    if value is None:
        return None
    try:
        return round(float(value) * 100, 2)
    except:
        return None

def extract_all_data():
    raw_rows = []

    # Iterate through models
    for model_name in TARGET_MODELS:
        model_dir = os.path.join(BASE_PATH, model_name)
        
        if not os.path.exists(model_dir):
            print(f"Skipping missing directory: {model_name}")
            continue

        # Walk through directory
        for root, dirs, files in os.walk(model_dir):
            if "index.json" in files:
                json_path = os.path.join(root, "index.json")
                
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        content = json.load(f)
                        
                    # Handle list format
                    if isinstance(content, list) and len(content) > 0:
                        data = content[0]
                    else:
                        continue

                    metrics = data.get("metrics", {})
                    params = data.get("params", {})
                    
                    # 1. Determine Task Name
                    task_name = params.get("task", os.path.basename(root))

                    # 2. Determine Checkpoint
                    rel_path = os.path.relpath(root, model_dir)
                    path_parts = rel_path.split(os.sep)
                    checkpoint = "Main (No Checkpoint)"
                    for part in path_parts:
                        if "checkpoint-" in part:
                            checkpoint = part
                            break
                    
                    # 3. Extract Scores (Handling TruthfulQA Special Case)
                    # For TruthfulQA, we use mc2_prob_mass_score as the value for all metrics
                    mc2_score = metrics.get("mc2_prob_mass_score")
                    
                    # Get raw values or fallback to mc2_score if specific metric is missing
                    # This ensures TruthfulQA shows up in all 3 sheets
                    val_acc = metrics.get("acc")
                    if val_acc is None and mc2_score is not None:
                        val_acc = mc2_score

                    val_acc_norm_token = metrics.get("acc_norm_token")
                    if val_acc_norm_token is None and mc2_score is not None:
                        val_acc_norm_token = mc2_score

                    val_acc_norm_char = metrics.get("acc_norm_char")
                    if val_acc_norm_char is None and mc2_score is not None:
                        val_acc_norm_char = mc2_score

                    # Store raw data
                    raw_rows.append({
                        "Model": model_name,
                        "Checkpoint": checkpoint,
                        "Task": task_name,
                        "acc": format_score(val_acc),
                        "acc_norm_token": format_score(val_acc_norm_token),
                        "acc_norm_char": format_score(val_acc_norm_char)
                    })

                except Exception as e:
                    print(f"Error parsing {json_path}: {e}")
    
    return raw_rows

def create_pivot_df(raw_data, value_column):
    """
    Creates a pivot table (Matrix) for easier reading.
    Rows: Model, Checkpoint
    Cols: Task
    Values: Score
    """
    df = pd.DataFrame(raw_data)
    if df.empty:
        return pd.DataFrame()
    
    # Create pivot table
    pivot_df = df.pivot_table(
        index=["Model", "Checkpoint"], 
        columns="Task", 
        values=value_column,
        aggfunc='first' # Should be unique, but take first if duplicates
    )
    return pivot_df

def main():
    data = extract_all_data()
    
    if not data:
        print("No data found.")
        return

    # Create the 3 DataFrames for the 3 sheets
    df_acc = create_pivot_df(data, "acc")
    df_token = create_pivot_df(data, "acc_norm_token")
    df_char = create_pivot_df(data, "acc_norm_char")

    # Save to Excel
    with pd.ExcelWriter(OUTPUT_FILE, engine='openpyxl') as writer:
        df_acc.to_excel(writer, sheet_name='ACC')
        df_token.to_excel(writer, sheet_name='ACC Norm Token')
        df_char.to_excel(writer, sheet_name='ACC Norm Char')

    print(f"Done! File saved to: {OUTPUT_FILE}")
    print(f"Processed {len(data)} experiment result files.")

if __name__ == "__main__":
    main()