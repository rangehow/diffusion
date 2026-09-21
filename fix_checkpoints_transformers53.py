#!/usr/bin/env python3
"""
Fix all checkpoint modeling_llada.py, configuration_llada.py, and config.json files
for transformers >= 5.3 compatibility.

Bugs:
  1. `_tied_weights_keys` had wrong paths and ff_out doesn't exist when weight_tying=True.
  2. `all_tied_weights_keys` property override conflicts with transformers 5.3's instance attribute.
  3. `tie_weights()` tried to assign ff_out = wte which crashes when ff_out doesn't exist.
  4. `LLaDAModelLM.__init__` never called `self.post_init()`, so `all_tied_weights_keys`
     instance attribute is never set → AttributeError in mark_tied_weights_as_initialized.
  5. `config.json` missing `tie_word_embeddings: false`, so HF defaults to True and tries
     to auto-manage tying we handle ourselves.
  6. `configuration_llada.py` missing `tie_word_embeddings=False` default.

Fixes applied:
  - _tied_weights_keys = None
  - Remove all_tied_weights_keys property
  - tie_weights() → no-op
  - Add self.post_init() at end of LLaDAModelLM.__init__
  - config.json: tie_word_embeddings = False
  - configuration_llada.py: all_kwargs.setdefault("tie_word_embeddings", False)
"""

import os
import re
import json
import argparse
from pathlib import Path


def fix_modeling_file(filepath: str, dry_run: bool = False) -> list[str]:
    """Fix a single modeling_llada.py file. Returns list of changes made."""
    changes = []
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    original = content
    
    # === Fix 1: _tied_weights_keys ===
    old_patterns = [
        r'_tied_weights_keys\s*=\s*\{[^}]*\}',
        r'_tied_weights_keys\s*=\s*\[[^\]]*\]',
    ]
    for pattern in old_patterns:
        if re.search(pattern, content):
            content = re.sub(pattern, '_tied_weights_keys = None', content)
            changes.append("Fixed _tied_weights_keys → None")
    
    # === Fix 2: Remove all_tied_weights_keys property ===
    prop_patterns = [
        (
            r'\n    @property\n'
            r'    def all_tied_weights_keys\(self\):\n'
            r'(?:        """[^"]*"""\n)?'
            r'        return self\._tied_weights_keys\n'
        ),
        (
            r'\n    @property\n'
            r'    def all_tied_weights_keys\(self\):\n'
            r'        """.*?"""\n'
            r'        return self\._tied_weights_keys\n'
        ),
    ]
    for pattern in prop_patterns:
        if re.search(pattern, content, re.DOTALL):
            content = re.sub(pattern, '\n', content, flags=re.DOTALL)
            changes.append("Removed all_tied_weights_keys property")
            break
    
    # === Fix 3: Fix tie_weights ===
    tie_pattern = (
        r'def tie_weights\(self(?:,\s*\*\*kwargs)?\):\n'
        r'        if self\.config\.weight_tying:\n'
        r'            self\.model\.transformer\.ff_out = self\.model\.transformer\.wte\n'
    )
    new_tie = (
        'def tie_weights(self, **kwargs):\n'
        '        # Weight tying is handled in forward() via F.linear(x, wte.weight).\n'
        '        pass\n'
    )
    if re.search(tie_pattern, content):
        content = re.sub(tie_pattern, new_tie, content)
        changes.append("Fixed tie_weights → no-op")
    
    # === Fix 4: Add self.post_init() to __init__ ===
    if 'self.post_init()' not in content:
        # Simple and robust: find "        else:\n            self.model = model\n"
        # (the last line of __init__) and append post_init after it.
        # 
        # Two patterns to handle:
        #   Pattern A (if/else):
        #       else:
        #           self.model = model
        #   Pattern B (single assignment):
        #       self.model = LLaDAModel(...)
        #
        # We look for the else branch first (more specific), then fall back.
        
        target = '        else:\n            self.model = model\n'
        if target in content:
            content = content.replace(
                target,
                target +
                '        # NOTE (transformers >= 5.3): post_init() sets required instance\n'
                '        # attributes (all_tied_weights_keys, etc.) and calls init_weights().\n'
                '        self.post_init()\n',
                1  # replace first occurrence only
            )
            changes.append("Added self.post_init() to __init__")
        else:
            # Fallback: find the model assignment in the if branch
            # Look for "self.model = LLaDAModel(" followed by a blank line or next def
            model_assign = re.search(
                r'(            self\.model = LLaDAModel\([^)]+\))\n',
                content
            )
            if model_assign:
                old = model_assign.group(0)
                content = content.replace(
                    old,
                    old +
                    '        # NOTE (transformers >= 5.3): post_init() sets required instance\n'
                    '        # attributes (all_tied_weights_keys, etc.) and calls init_weights().\n'
                    '        self.post_init()\n',
                    1
                )
                changes.append("Added self.post_init() to __init__ (fallback pattern)")
    
    if content != original and not dry_run:
        with open(filepath, 'w') as f:
            f.write(content)
    
    return changes


def fix_config_json(filepath: str, dry_run: bool = False) -> list[str]:
    """Fix a single config.json file."""
    changes = []
    
    with open(filepath, 'r') as f:
        config = json.load(f)
    
    if config.get('tie_word_embeddings') is not False:
        old_val = config.get('tie_word_embeddings')
        config['tie_word_embeddings'] = False
        changes.append(f"Set tie_word_embeddings=False (was {old_val!r})")
        
        if not dry_run:
            with open(filepath, 'w') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
                f.write('\n')
    
    return changes


def fix_configuration_file(filepath: str, dry_run: bool = False) -> list[str]:
    """Fix a single configuration_llada.py file."""
    changes = []
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    original = content
    
    # === Fix: tie_word_embeddings default ===
    if 'tie_word_embeddings' not in content:
        if 'super().__init__(**all_kwargs)' in content:
            content = content.replace(
                '        super().__init__(**all_kwargs)',
                '        all_kwargs.setdefault("tie_word_embeddings", False)\n'
                '        super().__init__(**all_kwargs)'
            )
            changes.append("Added tie_word_embeddings=False default")
    
    # === Fix: use_cache attribute ===
    # Ensure __init__ explicitly accepts and stores use_cache so
    # self.config.use_cache always exists (transformers >= 5.3 relies on it).
    if 'use_cache' not in content:
        if 'super().__init__(**all_kwargs)' in content:
            content = content.replace(
                '        super().__init__(**all_kwargs)',
                '        all_kwargs.setdefault("use_cache", False)\n'
                '        super().__init__(**all_kwargs)'
            )
            changes.append("Added use_cache=False default")
    
    if content != original and not dry_run:
        with open(filepath, 'w') as f:
            f.write(content)
    
    return changes


def main():
    parser = argparse.ArgumentParser(description="Fix LLaDA checkpoints for transformers >= 5.3")
    parser.add_argument(
        "--model_output_dir",
        default="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/diffusion/model_output",
        help="Root directory containing model checkpoints",
    )
    parser.add_argument(
        "--hf_cache_dir",
        default="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/.cache/modules/transformers_modules",
        help="HuggingFace trust_remote_code cache directory",
    )
    parser.add_argument("--dry_run", action="store_true", help="Preview changes without writing")
    parser.add_argument("--verbose", action="store_true", help="Print all changes")
    args = parser.parse_args()
    
    # Scan both model_output AND the HF cache for files to fix
    scan_dirs = [Path(args.model_output_dir)]
    hf_cache = Path(args.hf_cache_dir)
    if hf_cache.exists():
        scan_dirs.append(hf_cache)
    
    modeling_files = []
    config_jsons = []
    config_pys = []
    for d in scan_dirs:
        modeling_files.extend(sorted(d.rglob("modeling_llada.py")))
        config_jsons.extend(sorted(d.rglob("config.json")))
        config_pys.extend(sorted(d.rglob("configuration_llada.py")))
    
    print(f"Found {len(modeling_files)} modeling_llada.py files")
    print(f"Found {len(config_pys)} configuration_llada.py files")
    print(f"Found {len(config_jsons)} config.json files")
    
    if args.dry_run:
        print("\n*** DRY RUN — no files will be modified ***\n")
    
    counts = {"modeling": 0, "config_py": 0, "config_json": 0}
    
    for fp in modeling_files:
        changes = fix_modeling_file(str(fp), dry_run=args.dry_run)
        if changes:
            counts["modeling"] += 1
            if args.verbose:
                for c in changes:
                    print(f"  [modeling]  {fp}: {c}")
    
    for fp in config_pys:
        changes = fix_configuration_file(str(fp), dry_run=args.dry_run)
        if changes:
            counts["config_py"] += 1
            if args.verbose:
                for c in changes:
                    print(f"  [conf.py]  {fp}: {c}")
    
    for fp in config_jsons:
        changes = fix_config_json(str(fp), dry_run=args.dry_run)
        if changes:
            counts["config_json"] += 1
            if args.verbose:
                for c in changes:
                    print(f"  [conf.json] {fp}: {c}")
    
    print(f"\nFixed {counts['modeling']}/{len(modeling_files)} modeling_llada.py")
    print(f"Fixed {counts['config_py']}/{len(config_pys)} configuration_llada.py")
    print(f"Fixed {counts['config_json']}/{len(config_jsons)} config.json")
    
    if args.dry_run:
        print("\n*** DRY RUN complete — re-run without --dry_run to apply ***")


if __name__ == "__main__":
    main()
