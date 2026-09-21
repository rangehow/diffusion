"""
Generate Countdown and Sudoku datasets for reasoning task evaluation.
Follows the Ye et al. (2024) "Beyond Autoregression" setting.

Countdown: Given N numbers and a target, find arithmetic steps to reach the target.
Sudoku:    Given a 9x9 grid with blanks (0), fill in the solution.

Output format: HuggingFace datasets saved to disk with columns: {"text": "prompt [SEP] response"}
"""

import os
import json
import random
import argparse
from itertools import combinations
from typing import List, Tuple, Optional, Set
from copy import deepcopy

import datasets


# ============================================================================
# Countdown Task
# ============================================================================

def evaluate_expression(a: int, op: str, b: int) -> Optional[int]:
    """Evaluate a single arithmetic operation. Returns None if invalid."""
    if op == '+':
        return a + b
    elif op == '-':
        result = a - b
        return result if result > 0 else None  # Must be positive
    elif op == '*':
        return a * b
    elif op == '/':
        if b == 0 or a % b != 0:
            return None  # Must divide evenly
        return a // b
    return None


def solve_countdown(numbers: List[int], target: int, max_solutions: int = 1) -> List[List[str]]:
    """
    Find arithmetic expressions using the given numbers to reach the target.
    Returns a list of solution step lists, each step like "a+b=c".
    Uses recursive search (DFS).
    """
    solutions = []
    
    def search(available: List[int], steps: List[str]):
        if len(solutions) >= max_solutions:
            return
        
        if target in available:
            solutions.append(list(steps))
            return
        
        if len(available) < 2:
            return
        
        # Try all pairs and operations
        for i in range(len(available)):
            for j in range(len(available)):
                if i == j:
                    continue
                a, b = available[i], available[j]
                for op in ['+', '-', '*', '/']:
                    result = evaluate_expression(a, op, b)
                    if result is None or result <= 0:
                        continue
                    
                    # Build new available list
                    remaining = [available[k] for k in range(len(available)) if k != i and k != j]
                    remaining.append(result)
                    
                    step = f"{a}{op}{b}={result}"
                    steps.append(step)
                    search(remaining, steps)
                    steps.pop()
                    
                    if len(solutions) >= max_solutions:
                        return
    
    search(numbers, [])
    return solutions


def generate_countdown_instance(n_numbers: int, target_range: Tuple[int, int] = (10, 100),
                                 number_range: Tuple[int, int] = (1, 25)) -> Optional[dict]:
    """Generate a single valid Countdown instance with n_numbers.
    
    Format matches Ye et al. (2024) exactly:
      prompt:   "86,28,13,31,96"  (comma-separated numbers, target is last)
      response: "86+28=114,31-13=18,114-18=96"  (comma-separated steps)
    """
    max_attempts = 200
    
    for _ in range(max_attempts):
        numbers = [random.randint(number_range[0], number_range[1]) for _ in range(n_numbers)]
        target = random.randint(target_range[0], target_range[1])
        
        solutions = solve_countdown(numbers, target, max_solutions=1)
        if solutions:
            steps_str = ",".join(solutions[0])  # No spaces: "86+28=114,31-13=18,114-18=96"
            # Ye et al. format: numbers + target as comma-separated list
            prompt = ",".join(map(str, numbers)) + "," + str(target)
            response = steps_str
            return {"prompt": prompt, "response": response}
    
    return None


def generate_countdown_dataset(n_numbers: int, n_train: int, n_test: int, seed: int = 42):
    """Generate Countdown dataset for a specific difficulty level."""
    random.seed(seed)
    
    all_data = []
    seen_prompts: Set[str] = set()
    total_needed = n_train + n_test
    attempts = 0
    max_attempts = total_needed * 20
    
    print(f"Generating CD{n_numbers}: {n_train} train + {n_test} test ...")
    
    while len(all_data) < total_needed and attempts < max_attempts:
        instance = generate_countdown_instance(n_numbers)
        attempts += 1
        
        if instance and instance["prompt"] not in seen_prompts:
            seen_prompts.add(instance["prompt"])
            all_data.append(instance)
            
            if len(all_data) % 5000 == 0:
                print(f"  Generated {len(all_data)}/{total_needed} instances ({attempts} attempts)")
    
    if len(all_data) < total_needed:
        print(f"  WARNING: Only generated {len(all_data)}/{total_needed} instances after {attempts} attempts")
        # Adjust splits proportionally
        n_test = min(n_test, len(all_data) // 10)
        n_train = len(all_data) - n_test
    
    random.shuffle(all_data)
    train_data = all_data[:n_train]
    test_data = all_data[n_train:n_train + n_test]
    
    return train_data, test_data


# ============================================================================
# Sudoku Task
# ============================================================================

def is_valid_sudoku_placement(grid, row, col, num):
    """Check if placing num at (row, col) is valid."""
    # Check row
    if num in grid[row]:
        return False
    # Check column
    if any(grid[r][col] == num for r in range(9)):
        return False
    # Check 3x3 box
    box_row, box_col = 3 * (row // 3), 3 * (col // 3)
    for r in range(box_row, box_row + 3):
        for c in range(box_col, box_col + 3):
            if grid[r][c] == num:
                return False
    return True


def solve_sudoku(grid):
    """Solve a Sudoku puzzle using backtracking. Returns True if solved."""
    for row in range(9):
        for col in range(9):
            if grid[row][col] == 0:
                for num in range(1, 10):
                    if is_valid_sudoku_placement(grid, row, col, num):
                        grid[row][col] = num
                        if solve_sudoku(grid):
                            return True
                        grid[row][col] = 0
                return False
    return True


def generate_complete_sudoku():
    """Generate a complete valid Sudoku grid."""
    grid = [[0]*9 for _ in range(9)]
    
    # Fill diagonal 3x3 boxes first (they're independent)
    for box in range(3):
        nums = list(range(1, 10))
        random.shuffle(nums)
        for i in range(3):
            for j in range(3):
                grid[box*3 + i][box*3 + j] = nums[i*3 + j]
    
    # Solve the rest
    solve_sudoku(grid)
    return grid


def generate_sudoku_puzzle(n_blanks: int = 40) -> Tuple[str, str]:
    """
    Generate a Sudoku puzzle and its solution.
    Returns (puzzle_string, solution_string) where each is 81 chars.
    """
    solution = generate_complete_sudoku()
    puzzle = deepcopy(solution)
    
    # Remove n_blanks cells
    positions = [(r, c) for r in range(9) for c in range(9)]
    random.shuffle(positions)
    
    for r, c in positions[:n_blanks]:
        puzzle[r][c] = 0
    
    # No spaces between digits — our custom char-level tokenizer handles
    # each digit as a separate token, matching Ye et al. exactly
    puzzle_str = ''.join(str(puzzle[r][c]) for r in range(9) for c in range(9))
    solution_str = ''.join(str(solution[r][c]) for r in range(9) for c in range(9))
    
    return puzzle_str, solution_str


def generate_sudoku_dataset(n_train: int, n_test: int, n_blanks: int = 40, seed: int = 42):
    """Generate Sudoku dataset."""
    random.seed(seed)
    
    all_data = []
    seen: Set[str] = set()
    total_needed = n_train + n_test
    
    print(f"Generating Sudoku: {n_train} train + {n_test} test (blanks={n_blanks}) ...")
    
    while len(all_data) < total_needed:
        puzzle_str, solution_str = generate_sudoku_puzzle(n_blanks)
        
        if puzzle_str not in seen:
            seen.add(puzzle_str)
            prompt = puzzle_str  # Just the digits, no prefix (matches Ye et al.)
            response = solution_str
            all_data.append({"prompt": prompt, "response": response})
            
            if len(all_data) % 5000 == 0:
                print(f"  Generated {len(all_data)}/{total_needed} instances")
    
    random.shuffle(all_data)
    train_data = all_data[:n_train]
    test_data = all_data[n_train:n_train + n_test]
    
    return train_data, test_data


# ============================================================================
# Dataset Creation
# ============================================================================

SEP_TOKEN = " [SEP] "

def format_for_hf(data: List[dict]) -> datasets.Dataset:
    """Convert list of {prompt, response} dicts to a HF Dataset with 'text' column."""
    texts = [d["prompt"] + SEP_TOKEN + d["response"] for d in data]
    return datasets.Dataset.from_dict({"text": texts})


def format_for_hf_seq2seq(data: List[dict]) -> datasets.Dataset:
    """Convert to HF Dataset keeping prompt and response columns separately."""
    return datasets.Dataset.from_dict({
        "prompt": [d["prompt"] for d in data],
        "response": [d["response"] for d in data],
        "text": [d["prompt"] + SEP_TOKEN + d["response"] for d in data],
    })


def main():
    parser = argparse.ArgumentParser(description="Generate Countdown and Sudoku datasets")
    parser.add_argument("--output_dir", type=str, default="reasoning_tasks/data")
    parser.add_argument("--tasks", nargs="+", default=["cd3", "cd4", "cd5", "sudoku"],
                        choices=["cd3", "cd4", "cd5", "sudoku"])
    parser.add_argument("--cd_train_size", type=int, default=500000)
    parser.add_argument("--cd_test_size", type=int, default=1000)
    parser.add_argument("--sudoku_train_size", type=int, default=100000)
    parser.add_argument("--sudoku_test_size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    task_configs = {
        "cd3": {"type": "countdown", "n_numbers": 3, "n_train": args.cd_train_size, "n_test": args.cd_test_size},
        "cd4": {"type": "countdown", "n_numbers": 4, "n_train": args.cd_train_size, "n_test": args.cd_test_size},
        "cd5": {"type": "countdown", "n_numbers": 5, "n_train": args.cd_train_size, "n_test": args.cd_test_size},
        "sudoku": {"type": "sudoku", "n_train": args.sudoku_train_size, "n_test": args.sudoku_test_size},
    }
    
    for task_name in args.tasks:
        cfg = task_configs[task_name]
        
        if cfg["type"] == "countdown":
            train_data, test_data = generate_countdown_dataset(
                n_numbers=cfg["n_numbers"],
                n_train=cfg["n_train"],
                n_test=cfg["n_test"],
                seed=args.seed + hash(task_name) % 10000
            )
        else:  # sudoku
            train_data, test_data = generate_sudoku_dataset(
                n_train=cfg["n_train"],
                n_test=cfg["n_test"],
                seed=args.seed
            )
        
        # Save with both formats
        train_ds = format_for_hf_seq2seq(train_data)
        test_ds = format_for_hf_seq2seq(test_data)
        
        train_path = os.path.join(args.output_dir, f"{task_name}_train")
        test_path = os.path.join(args.output_dir, f"{task_name}_test")
        
        train_ds.save_to_disk(train_path)
        test_ds.save_to_disk(test_path)
        
        print(f"  Saved {task_name}: train={len(train_ds)} -> {train_path}")
        print(f"  Saved {task_name}: test={len(test_ds)} -> {test_path}")
        
        # Show a few examples
        print(f"\n  Sample {task_name} examples:")
        for i in range(min(3, len(test_ds))):
            print(f"    [{i}] prompt:   {test_ds[i]['prompt']}")
            print(f"         response: {test_ds[i]['response']}")
        print()


if __name__ == "__main__":
    main()
