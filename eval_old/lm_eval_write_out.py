import argparse
import logging
import os
import random
import json  # 1. Import the json library

import numpy as np

from lm_eval import tasks
from lm_eval.evaluator_utils import get_task_list
from lm_eval.tasks import TaskManager
from lm_eval.utils import join_iters


eval_logger = logging.getLogger(__name__)


# The EXAMPLE_DIVIDER is no longer needed, but we can leave it for reference
# EXAMPLE_DIVIDER = "!!@@##@@!! -- Example {i}\n"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_base_path", "--output_path", required=True)
    parser.add_argument("--tasks", default="all_tasks")
    parser.add_argument("--sets", type=str, default="test")  # example: val,test
    parser.add_argument("--num_fewshot", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_examples", type=int, default=0)
    parser.add_argument(
        "--include_path",
        type=str,
        default=None,
        help="Additional path to include if there are external tasks to include.",
    )
    parser.add_argument(
        "--verbosity",
        type=str,
        default="INFO",
        help="Log error when tasks are not registered.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    # Set up logging
    logging.basicConfig(level=getattr(logging, args.verbosity))

    if args.include_path is not None:
        eval_logger.info(f"Including path: {args.include_path}")

    task_manager = TaskManager(args.verbosity, include_path=args.include_path)

    if args.tasks == "all_tasks":
        task_names = task_manager.all_tasks
    else:
        task_names = args.tasks.split(",")
    task_dict = tasks.get_task_dict(task_names, task_manager)

    os.makedirs(args.output_base_path, exist_ok=True)
    for task in [x.task for x in get_task_list(task_dict)]:
        task_name = task.config.task
        rnd = random.Random()
        rnd.seed(args.seed)

        iters = []

        for set_name in args.sets.split(","):
            docs = None
            if set_name == "train" and task.has_training_docs():
                docs = task.training_docs()
            if set_name == "val" and task.has_validation_docs():
                docs = task.validation_docs()
            if set_name == "test" and task.has_test_docs():
                docs = task.test_docs()
            if docs is not None:
                iters.append(docs)

        if len(iters) == 0:
            raise ValueError(
                f"Passed --sets '{args.sets}' but this task has no splits which match. Please specify a different --sets value."
            )

        docs = join_iters(iters)

        # *** MODIFIED SECTION START ***

        # 2. Define the output file path with a .jsonl extension
        output_file_path = os.path.join(args.output_base_path, f"{task_name}.jsonl")
        eval_logger.info(f"Writing prompts for task '{task_name}' to {output_file_path}...")

        with open(output_file_path, "w", encoding="utf8") as f:
            # Determine the document iterator
            doc_iterator = (
                zip(range(args.num_examples), docs)
                if args.num_examples > 0
                else enumerate(docs)
            )

            for i, doc in doc_iterator:
                # Generate the few-shot context (the prompt)
                ctx = task.fewshot_context(
                    doc=doc,
                    num_fewshot=args.num_fewshot,
                )

                # 3. Create a dictionary and write it as a JSON line
                output_record = {"prompt": ctx}
                f.write(json.dumps(output_record, ensure_ascii=False) + "\n")

        # *** MODIFIED SECTION END ***

if __name__ == "__main__":
    main()