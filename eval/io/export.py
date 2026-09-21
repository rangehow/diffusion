# eval/io/export.py
"""
Export utilities for evaluation results.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd


def parse_run_key(key: str) -> Optional[Dict[str, Any]]:
    """
    Parse summary.json key into components.
    
    Example: "hellaswag_0shot_causal" -> {'task': 'hellaswag', 'fewshot': 0, 'type': 'causal'}
    """
    parts = key.split('_')
    try:
        task = parts[0]
        shot_part = next((p for p in parts if "shot" in p), "0shot")
        fewshot = int(shot_part.replace("shot", ""))
        return {'task': task, 'fewshot': fewshot}
    except (IndexError, ValueError):
        return None


def gather_results(results_dir: Path) -> List[Dict[str, Any]]:
    """
    Collect all results from summary.json files.
    
    Args:
        results_dir: Root directory containing model results
        
    Returns:
        List of flattened result dictionaries
    """
    all_data = []
    summary_files = list(results_dir.rglob('summary.json'))
    
    if not summary_files:
        print(f"No summary.json files found in {results_dir}")
        return []
    
    print(f"Found {len(summary_files)} summary files")
    
    for summary_file in summary_files:
        model_name = summary_file.parent.name
        print(f"  Processing: {model_name}")
        
        try:
            with open(summary_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for run_key, details in data.items():
                parsed = parse_run_key(run_key)
                if not parsed:
                    continue
                
                record = {
                    'model_name': model_name,
                    'task_name': parsed['task'],
                    'num_fewshot': parsed['fewshot'],
                }
                record.update(details.get('metrics', {}))
                all_data.append(record)
                
        except json.JSONDecodeError:
            print(f"  Error: Could not parse {summary_file}")
        except Exception as e:
            print(f"  Error processing {summary_file}: {e}")
    
    return all_data


def create_pivot_table(data: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Create a pivot table from flattened results.
    
    Rows: model names
    Columns: task metrics
    """
    if not data:
        return pd.DataFrame()
    
    df = pd.DataFrame(data)
    
    # Create display name for tasks
    df['task_display'] = df.apply(
        lambda row: f"{row['task_name']} ({row['num_fewshot']}-shot)",
        axis=1
    )
    
    # Find metric columns
    meta_cols = {'model_name', 'task_name', 'num_fewshot', 'task_display'}
    metric_cols = [c for c in df.columns if c not in meta_cols]
    
    if not metric_cols:
        print("Warning: No metric columns found")
        return pd.DataFrame()
    
    # Create pivot table
    pivot = df.pivot_table(
        index='model_name',
        columns='task_display',
        values=metric_cols,
    )
    
    # Reorder columns: (task, metric) instead of (metric, task)
    pivot = pivot.swaplevel(0, 1, axis=1)
    pivot.sort_index(axis=1, level=0, inplace=True)
    
    return pivot


def export_to_excel(
    results_dir: str,
    output_file: str = "evaluation_summary.xlsx",
) -> Optional[Path]:
    """
    Export all results to an Excel file.
    
    Args:
        results_dir: Directory containing evaluation results
        output_file: Output Excel filename
        
    Returns:
        Path to created Excel file, or None if failed
    """
    results_path = Path(results_dir)
    output_path = Path(output_file)
    
    if not results_path.is_dir():
        print(f"Error: {results_path} is not a directory")
        return None
    
    # Gather and process data
    flat_data = gather_results(results_path)
    
    if not flat_data:
        print("No data to export")
        return None
    
    # Create pivot table
    table = create_pivot_table(flat_data)
    
    if table.empty:
        print("Could not create summary table")
        return None
    
    # Save to Excel
    try:
        table.to_excel(output_path)
        print(f"✓ Exported to: {output_path.absolute()}")
        return output_path
    except Exception as e:
        print(f"✗ Error saving Excel: {e}")
        return None