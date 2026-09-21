# eval/dataset_paths.py
"""
Configuration for local dataset paths.
Maps TASK_NAME to local directory paths.
"""

import os

LOCAL_DATASETS = {
    "mmlu": "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/BERT_TRAINING_SERVICE/platform/dataset/cais/mmlu/main",
    "hellaswag": "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/BERT_TRAINING_SERVICE/platform/dataset/Rowan/hellaswag/main",
    "sciq": "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/BERT_TRAINING_SERVICE/platform/dataset/sciq/main",
    "winogrande": "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/BERT_TRAINING_SERVICE/platform/dataset/winogrande/main",
    "piqa": "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/BERT_TRAINING_SERVICE/platform/dataset/baber/piqa/main",
    
    # Fixed: Use TASK_NAME as keys
    "arc_easy": "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/BERT_TRAINING_SERVICE/platform/dataset/allenai/ai2_arc/main",
    "arc_challenge": "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/BERT_TRAINING_SERVICE/platform/dataset/allenai/ai2_arc/main",
    "commonsense_qa": "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/BERT_TRAINING_SERVICE/platform/dataset/tau/commonsense_qa/main",
    "truthfulqa_mc1": "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/BERT_TRAINING_SERVICE/platform/dataset/truthful_qa/main",
    "truthfulqa_mc2": "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/BERT_TRAINING_SERVICE/platform/dataset/truthful_qa/main",
    "mmlu_redux": "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/BERT_TRAINING_SERVICE/platform/dataset/edinburgh-dawg/mmlu-redux-2.0/main",
    "mmlu_redux_corrected": "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/BERT_TRAINING_SERVICE/platform/dataset/edinburgh-dawg/mmlu-redux-2.0/main",
    "mmlu_redux_ok": "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/BERT_TRAINING_SERVICE/platform/dataset/edinburgh-dawg/mmlu-redux-2.0/main",
    "mmlu_redux_all": "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/BERT_TRAINING_SERVICE/platform/dataset/edinburgh-dawg/mmlu-redux-2.0/main"
}

def get_local_path(task_name: str):
    """
    Get the local path for a task if configured and exists.
    """
    path = LOCAL_DATASETS.get(task_name)
    
    # Handle sub-task cases like "mmlu:anatomy" -> look up "mmlu"
    if not path and ":" in task_name:
        parent_task = task_name.split(":")[0]
        path = LOCAL_DATASETS.get(parent_task)
        
    if path and os.path.exists(path):
        return path
    return None