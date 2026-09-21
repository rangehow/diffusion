#!/usr/bin/env python3
"""
fix_openreview_latex.py — Fix LaTeX math rendering for OpenReview/marked.js

OpenReview pipeline: marked (v15) → DOMPurify → MathJax
marked has NO math extension, so CommonMark rules destroy LaTeX before MathJax sees it.

7 bug categories, 23 rules applied in order (order matters).
Idempotent — running twice produces the same output.
"""

import re
import sys
import argparse


def find_math_spans(text):
    """Find all $...$ and $$...$$ spans, returning (start, end) tuples."""
    spans = []
    i = 0
    while i < len(text):
        if text[i] == '$':
            # Check for $$
            if i + 1 < len(text) and text[i + 1] == '$':
                # Display math $$...$$
                end = text.find('$$', i + 2)
                if end != -1:
                    spans.append((i, end + 2))
                    i = end + 2
                    continue
            else:
                # Inline math $...$
                # Find matching closing $ (not preceded by \)
                j = i + 1
                found = False
                while j < len(text):
                    if text[j] == '$' and (j == 0 or text[j - 1] != '\\'):
                        spans.append((i, j + 1))
                        i = j + 1
                        found = True
                        break
                    j += 1
                if found:
                    continue
        i += 1
    return spans


def fix_math_content(m):
    """Apply all fixes to a single math span's inner content."""
    content = m

    # === Category 1: Backslash eating (CommonMark eats \ before ASCII punctuation) ===

    # Rule 1-4: Thin/thick spaces — \, \; \: \! → mkern equivalents
    # These are single-char non-letter commands; no lookahead needed
    content = re.sub(r'\\,', r'\\mkern{3}mu', content)
    content = re.sub(r'\\;', r'\\mkern{5}mu', content)
    content = re.sub(r'\\:', r'\\mkern{4}mu', content)
    content = re.sub(r'\\!', r'\\mkern{-3}mu', content)

    # Rule 5-6: \{ \} → \lbrace \rbrace (skip if already \lbrace/\rbrace)
    content = re.sub(r'\\{(?!\\)', r'\\lbrace ', content)
    content = re.sub(r'\\}(?!\\)', r'\\rbrace ', content)

    # Rule 7: \| → \Vert (skip if already \Vert)
    content = re.sub(r'\\\|(?![a-zA-Z])', r'\\Vert ', content)

    # === Category 2: Bare * → Markdown emphasis ===
    # Rule 8: *  in superscript/subscript → \ast
    content = re.sub(r'(?<=[\^_])\*', r'\\ast', content)
    # Rule 9: standalone * not part of \ast or ** → \ast
    content = re.sub(r'(?<!\\)(?<!\*)\*(?!\*|\\)', r'\\ast ', content)

    # === Category 3: Bare < or > → HTML tag parsing ===
    # Rule 10: < followed by letter → \lt
    content = re.sub(r'<(?=[a-zA-Z])', r'\\lt ', content)
    # Rule 11: > that could cause issues (bare > in math)
    # Only fix bare > not part of \gt, \geq etc.
    # content = re.sub(r'(?<!\\)>(?=[a-zA-Z0-9 ])', r'\\gt ', content)

    # === Category 4: }_ → italic trigger ===
    # Rule 12: }_ → }\mkern0mu_  (the space-less zero-width kern prevents emphasis)
    # Must check it's not already fixed
    content = re.sub(r'\}(?:\\mkern0mu)?_', r'}\\mkern0mu_', content)

    # === Category 5: \rbrace _ / \Vert _ / \lbrace _ → cross-block emphasis ===
    # After rules above convert \} → \rbrace, the pattern \rbrace _{sub} has a space
    # before _ which makes it a left-flanking delimiter in marked's emphasis parser.
    # Rule 13: \rbrace _ → \rbrace\mkern0mu_
    content = re.sub(r'\\rbrace\s+_', r'\\rbrace\\mkern0mu_', content)
    # Rule 14: \Vert _ → \Vert\mkern0mu_
    content = re.sub(r'\\Vert\s+_', r'\\Vert\\mkern0mu_', content)
    # Rule 15: \lbrace _ → \lbrace\mkern0mu_
    content = re.sub(r'\\lbrace\s+_', r'\\lbrace\\mkern0mu_', content)

    # === Category 6: \underbrace{A}_{B} → \underset{B}{\underbrace{A}} ===
    content = re.sub(
        r'\\underbrace\{([^}]+)\}\\mkern0mu_\{([^}]+)\}',
        r'\\underset{\2}{\\underbrace{\1}}',
        content
    )

    # === Category 7: |X| → \lvert X \rvert ===
    # Only fix bare | used as delimiters (heuristic: | at start/end of known patterns)
    # This is tricky — skip aggressive fixing, only do obvious cases
    # e.g. |\mathcal{V}| → \lvert\mathcal{V}\rvert
    content = re.sub(
        r'\|\\(mathcal|mathbf|mathbb|mathrm)\{([^}]+)\}\|',
        r'\\lvert\\\1{\2}\\rvert',
        content
    )

    return content


def fix_openreview_latex(text):
    """Fix all math spans in the text. Non-math content is left untouched."""
    spans = find_math_spans(text)
    if not spans:
        return text

    result = []
    prev_end = 0
    for start, end in spans:
        # Add non-math text as-is
        result.append(text[prev_end:start])

        # Determine delimiter
        raw = text[start:end]
        if raw.startswith('$$') and raw.endswith('$$'):
            delim = '$$'
            inner = raw[2:-2]
        else:
            delim = '$'
            inner = raw[1:-1]

        fixed_inner = fix_math_content(inner)
        result.append(f'{delim}{fixed_inner}{delim}')
        prev_end = end

    result.append(text[prev_end:])
    return ''.join(result)


def run_tests():
    """Self-tests for the fix rules."""
    tests = [
        # Category 1: backslash eating
        (r'$a\,b$', r'$a\mkern{3}mub$'),
        (r'$\{x\}$', r'$\lbrace x\rbrace $'),
        # Category 2: bare *
        (r'$p^*$', r'$p^\ast$'),
        # Category 3: bare <
        (r'$i<n$', r'$i\lt n$'),
        # Category 4: }_
        (r'$\mathbf{x}_n$', r'$\mathbf{x}\mkern0mu_n$'),
        # No math — unchanged
        ('Hello **world**', 'Hello **world**'),
    ]
    passed = 0
    for inp, expected in tests:
        result = fix_openreview_latex(inp)
        status = '✓' if result == expected else '✗'
        if status == '✗':
            print(f'  {status} Input:    {inp}')
            print(f'    Expected: {expected}')
            print(f'    Got:      {result}')
        else:
            print(f'  {status} {inp} → {result}')
            passed += 1
    print(f'\n{passed}/{len(tests)} tests passed.')


def main():
    parser = argparse.ArgumentParser(description='Fix LaTeX for OpenReview rendering')
    parser.add_argument('input', nargs='?', help='Input markdown file')
    parser.add_argument('-o', '--output', help='Output file (default: stdout)')
    parser.add_argument('--check', action='store_true', help='Dry-run: show what would change')
    parser.add_argument('--test', action='store_true', help='Run self-tests')
    args = parser.parse_args()

    if args.test:
        run_tests()
        return

    if not args.input:
        parser.error('Input file required (or use --test)')

    with open(args.input, 'r') as f:
        text = f.read()

    fixed = fix_openreview_latex(text)

    if args.check:
        if fixed == text:
            print('No changes needed.')
        else:
            # Show diff
            import difflib
            diff = difflib.unified_diff(
                text.splitlines(keepends=True),
                fixed.splitlines(keepends=True),
                fromfile=args.input,
                tofile=f'{args.input} (fixed)',
            )
            sys.stdout.writelines(diff)
        return

    if args.output:
        with open(args.output, 'w') as f:
            f.write(fixed)
        print(f'Fixed output written to {args.output}')
    else:
        sys.stdout.write(fixed)


if __name__ == '__main__':
    main()
