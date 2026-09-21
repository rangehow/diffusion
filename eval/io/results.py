# eval/io/results.py
"""
Results persistence and management.
"""

import json
import os
import uuid
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, TYPE_CHECKING

from ..config import ModelType

if TYPE_CHECKING:
    from ..engine.evaluator import EvaluationResult


class ResultsManager:
    """
    Manages saving and organizing evaluation results using a manifest (index) system.
    
    Results are organized in a hierarchical structure:
        output_dir/
            model_name/
                task_name/
                    index.json              # Registry of runs and parameters
                    {uuid}.results.json     # Detailed results
                    {uuid}.params.json      # Run configuration
    """
    
    def __init__(self, output_dir: str, model_path: str, task_name: str):
        """
        Initialize results manager.
        
        Args:
            output_dir: Base output directory
            model_path: Path to the model being evaluated
            task_name: Name of the task
        """
        self.output_dir = Path(output_dir)
        self.model_path = model_path
        self.task_name = task_name
        self.task_dir = self._prepare_task_dir()
        self.index_path = self.task_dir / "index.json"
    
    def _prepare_task_dir(self) -> Path:
        """
        Create and return the task-specific output directory.
        """
        path = Path(self.model_path)
        
        # Handle trailing slash
        if path.name == '':
            path = path.parent
        
        # Check if this is a checkpoint
        if path.name.startswith('checkpoint-'):
            experiment_name = path.parent.name
            checkpoint_name = path.name
            model_dir = self.output_dir / experiment_name / checkpoint_name
        else:
            model_dir = self.output_dir / path.name
        
        # Create directory structure: model/task
        task_dir = model_dir / self.task_name
        task_dir.mkdir(parents=True, exist_ok=True)
        
        return task_dir

    def _normalize_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize parameters for consistent comparison (sort keys, handle non-serializable).
        """
        # Round-trip through JSON to handle Enums and non-standard types transparently
        # sort_keys=True ensures parameter order doesn't affect the signature
        return json.loads(json.dumps(params, sort_keys=True, default=str))

    def _load_index(self) -> List[Dict[str, Any]]:
        """Load the index.json manifest."""
        if not self.index_path.exists():
            return []
        try:
            with open(self.index_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"[WARNING] Could not read index file {self.index_path}: {e}")
            return []

    def _save_index(self, index: List[Dict[str, Any]]):
        """Atomic save of the index."""
        temp_path = self.index_path.with_suffix('.tmp')
        try:
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(index, f, indent=2)
            os.replace(temp_path, self.index_path)
        except Exception as e:
            print(f"[ERROR] Failed to update index: {e}")
            if temp_path.exists():
                os.remove(temp_path)

    def find_existing_run(self, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Check if a run with the given parameters already exists.
        
        Args:
            params: Dictionary of parameters to check
            
        Returns:
            Index entry if found, None otherwise
        """
        index = self._load_index()
        target_params = self._normalize_params(params)
        
        for entry in index:
            # Check if parameters match
            if entry.get('params') == target_params:
                # Verify the actual result file exists
                result_file = self.task_dir / f"{entry['id']}.results.json"
                if result_file.exists():
                    return entry
        
        return None

    def load_result(self, run_id: str) -> Optional["EvaluationResult"]:
        """
        Load a specific result by its run ID.
        """
        # Local import to avoid circular dependency
        from ..engine.evaluator import EvaluationResult
        
        path = self.task_dir / f"{run_id}.results.json"
        if not path.exists():
            return None
            
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            return EvaluationResult(
                task_name=self.task_name,
                metrics=data['metrics'],
                detailed_results=data.get('detailed_results', []),
                metadata=data.get('metadata', {})
            )
        except Exception as e:
            print(f"[ERROR] Failed to load result {run_id}: {e}")
            return None
    
    def save(self, result: "EvaluationResult", params: Dict[str, Any]) -> Path:
        """
        Save evaluation results.
        
        1. Generates a unique run ID (UUID)
        2. Saves detailed results to {uuid}.results.json
        3. Saves parameters to {uuid}.params.json (for readability)
        4. Updates index.json
        
        Args:
            result: EvaluationResult to save
            params: Parameters defining this run
            
        Returns:
            Path to the detailed results file
        """
        # Generate short UUID
        run_id = uuid.uuid4().hex[:8]
        
        results_file = self.task_dir / f"{run_id}.results.json"
        params_file = self.task_dir / f"{run_id}.params.json"
        
        # 1. Save Params (Readable file)
        normalized_params = self._normalize_params(params)
        try:
            with open(params_file, 'w', encoding='utf-8') as f:
                json.dump(normalized_params, f, indent=2)
        except Exception as e:
            print(f"✗ Error saving params file: {e}")

        # 2. Save Detailed Results
        result_data = {
            "metadata": result.metadata,
            "metrics": result.metrics,
            "detailed_results": result.detailed_results,
        }
        
        try:
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, indent=2, ensure_ascii=False)
            print(f"✓ Detailed results saved: {results_file}")
        except Exception as e:
            print(f"✗ Error saving detailed results: {e}")
            raise
        
        # 3. Update Index
        index = self._load_index()
        
        # Remove old entry if exact params match (overwrite logic behavior)
        index = [i for i in index if i.get('params') != normalized_params]
        
        new_entry = {
            "id": run_id,
            "timestamp_utc": result.metadata.get('timestamp_utc'),
            "metrics": {k: v for k, v in result.metrics.items() if 'correct' not in k},
            "params": normalized_params
        }
        
        index.append(new_entry)
        self._save_index(index)
        
        return results_file