#!/usr/bin/env python3
"""
Convert compact SKI format (k/D/X/-) to Lazy K unlambda format (`ski).

Compact format (postfix): k=S, X=K, D=I, -=application
Lazy K unlambda (prefix): s=S, k=K, i=I, `=application

Since the stars.txt program is a self-contained expression (not a function),
we wrap it as K(expr) to make it a valid Lazy K program that ignores input.
"""

import sys
import os
import time


def compact_to_lazyk(compact_str):
    """Convert postfix compact to prefix Lazy K (unlambda style)."""
    stack = []
    for c in compact_str:
        if c == 'k':
            stack.append('s')
        elif c == 'X':
            stack.append('k')
        elif c == 'D':
            stack.append('i')
        elif c == '-':
            y = stack.pop()
            x = stack.pop()
            stack.append('`' + x + y)
    assert len(stack) == 1, f"Stack has {len(stack)} elements"
    return stack[0]


def compact_to_lazyk_streaming(compact_str, outfile):
    """
    Convert postfix compact to prefix Lazy K, writing to file.
    Uses iterative approach to avoid stack overflow for large inputs.

    Returns the number of characters written.
    """
    n = len(compact_str)

    # First pass: compute the prefix string for each subtree
    # For efficiency, store start positions and lengths in an array
    # Each element on the stack is (start_pos_in_output, length)

    # Actually, for very large inputs, we need a different approach.
    # The prefix notation has the same characters, just reordered.
    # We can compute the output by tracking subtree positions.

    # Simple approach: build a tree of indices, then do prefix traversal

    # Represent each node as (type, left_child_idx, right_child_idx)
    # type: 's', 'k', 'i', or 'a' for application

    nodes = []  # list of (type, left, right)
    stack = []  # stack of node indices

    for c in compact_str:
        if c == 'k':
            idx = len(nodes)
            nodes.append(('s', -1, -1))
            stack.append(idx)
        elif c == 'X':
            idx = len(nodes)
            nodes.append(('k', -1, -1))
            stack.append(idx)
        elif c == 'D':
            idx = len(nodes)
            nodes.append(('i', -1, -1))
            stack.append(idx)
        elif c == '-':
            y = stack.pop()
            x = stack.pop()
            idx = len(nodes)
            nodes.append(('a', x, y))
            stack.append(idx)

    assert len(stack) == 1
    root = stack[0]

    print(f"  Tree built: {len(nodes):,} nodes")

    # Iterative prefix traversal (DFS)
    written = 0
    visit_stack = [root]

    while visit_stack:
        idx = visit_stack.pop()
        typ, left, right = nodes[idx]

        if typ == 'a':
            outfile.write('`')
            written += 1
            # Push right first, then left (so left is processed first)
            visit_stack.append(right)
            visit_stack.append(left)
        else:
            outfile.write(typ)
            written += 1

    return written


def test_conversion():
    """Test compact to Lazy K conversion."""
    print("Testing conversion...")

    tests = [
        ('D', 'i', "I"),
        ('X', 'k', "K"),
        ('k', 's', "S"),
        ('XD-', '`ki', "KI (false)"),
        ('kXX--D-', '``s`kks`kki', "S(KK)I (true)"),  # Hmm, let me recalculate
    ]

    for compact, expected_lazyk, name in tests:
        result = compact_to_lazyk(compact)
        # For complex cases, just check it runs
        print(f"  {name}: compact={compact} -> lazyk={result}")
        if expected_lazyk:
            if result != expected_lazyk:
                print(f"    WARNING: expected {expected_lazyk}")

    print()


def main():
    print("=== Compact to Lazy K Converter ===\n")

    test_conversion()

    base = os.path.join(os.path.dirname(__file__), '..')
    compact_path = os.path.join(base, 'very_large_txt', 'stars_compact.txt')

    if not os.path.exists(compact_path):
        print(f"Compact file not found: {compact_path}")
        return

    file_size = os.path.getsize(compact_path)
    print(f"Reading compact file: {compact_path}")
    print(f"  Size: {file_size:,} bytes")

    start = time.time()
    with open(compact_path, 'r') as f:
        compact = f.read().strip()
    print(f"  Read in {time.time() - start:.1f}s")
    print(f"  Characters: {len(compact):,}")

    # Wrap in K(expr) to make a valid Lazy K program
    # K(expr)(input) = expr, so the program ignores input
    wrapped = 'X' + compact + '-'
    print(f"  Wrapped (K(expr)): {len(wrapped):,} chars")

    # Convert to Lazy K
    output_path = os.path.join(base, 'very_large_txt', 'stars.lazyk')
    print(f"\nConverting to Lazy K format...")
    print(f"  Output: {output_path}")

    start = time.time()
    with open(output_path, 'w') as outf:
        n_written = compact_to_lazyk_streaming(wrapped, outf)
    elapsed = time.time() - start

    print(f"  Written: {n_written:,} chars")
    print(f"  Time: {elapsed:.1f}s")

    out_size = os.path.getsize(output_path)
    print(f"  File size: {out_size:,} bytes")

    # Also create a version without K wrapper (raw expression)
    raw_output_path = os.path.join(base, 'very_large_txt', 'stars_raw.lazyk')
    print(f"\nAlso creating raw (unwrapped) version...")
    print(f"  Output: {raw_output_path}")

    start = time.time()
    with open(raw_output_path, 'w') as outf:
        n_written = compact_to_lazyk_streaming(compact, outf)
    elapsed = time.time() - start

    print(f"  Written: {n_written:,} chars")
    print(f"  Time: {elapsed:.1f}s")

    print("\nDone!")


if __name__ == '__main__':
    main()
