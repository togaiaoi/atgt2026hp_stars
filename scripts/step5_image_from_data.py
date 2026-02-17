#!/usr/bin/env python3
"""
Step 5: Convert data part to image.

Based on syugasato's approach:
- Extract leaf characters from the data
- Map to black/white pixels
- Create image with 4096 width

Try multiple approaches:
1. All leaves from entire compact file
2. Leaves from item 9 (largest data item)
3. Quadtree interpretation
"""

import os
import sys
import struct

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def count_leaves(compact):
    """Count leaf types in compact string."""
    counts = {'k': 0, 'X': 0, 'D': 0, '-': 0}
    for c in compact:
        counts[c] = counts.get(c, 0) + 1
    return counts


def leaves_only(compact):
    """Extract just the leaf characters from compact string."""
    return [c for c in compact if c in ('k', 'X', 'D')]


def write_pbm(filename, width, height, pixels):
    """Write a PBM (portable bitmap) file. pixels is list of 0/1."""
    with open(filename, 'w') as f:
        f.write(f"P1\n{width} {height}\n")
        for y in range(height):
            row = pixels[y * width : (y + 1) * width]
            f.write(' '.join(str(p) for p in row) + '\n')


def write_pgm(filename, width, height, pixels):
    """Write a PGM file. pixels is list of 0-255."""
    with open(filename, 'wb') as f:
        header = f"P5\n{width} {height}\n255\n"
        f.write(header.encode())
        f.write(bytes(pixels))


def compact_tree_leaves_inorder(compact):
    """
    Parse compact string to tree and do inorder traversal.
    Returns list of leaf characters in tree order (left-to-right).
    """
    stack = []
    for c in compact:
        if c in ('k', 'X', 'D'):
            stack.append([c])  # leaf as single-element list
        elif c == '-':
            right = stack.pop()
            left = stack.pop()
            stack.append(left + right)  # concatenate in-order
    if stack:
        return stack[0]
    return []


def decode_quadtree(compact, max_depth=12):
    """
    Try to interpret compact string as a quadtree.
    At each level, we have either:
    - A leaf (S, K, I) = uniform color region
    - An application node = split into subregions

    For a true quadtree: each split divides into 4 sub-quadrants.
    But SKI trees are binary, so we need to figure out the grouping.

    The diamond pattern from the reference:
    diamond COND QA QB QC QD
    = 5-argument function

    If the data directly encodes a quadtree in SKI:
    - true/false leaf = solid color (black/white)
    - application chain = recursion

    This is speculative. Let me try a simpler approach.
    """
    pass


def try_image_from_leaves(compact, name, output_dir, widths=[4096, 2048, 1024, 512, 256, 128]):
    """Try creating images from leaf sequence with different widths."""
    lvs = leaves_only(compact)
    n_leaves = len(lvs)
    counts = count_leaves(compact)

    print(f"\n=== {name} ===")
    print(f"  Total chars: {len(compact):,}")
    print(f"  Leaves: {n_leaves:,} (S={counts.get('k',0):,}, K={counts.get('X',0):,}, I={counts.get('D',0):,})")
    print(f"  Apps: {counts.get('-',0):,}")

    if n_leaves == 0:
        print("  No leaves!")
        return

    # Try different mappings
    mappings = {
        # mapping name: (black_chars, description)
        'K_black': (set('X'), "K=black, S,I=white"),
        'S_black': (set('k'), "S=black, K,I=white"),
        'I_black': (set('D'), "I=black, S,K=white"),
        'SK_black': (set('kX'), "S,K=black, I=white"),
        'SI_black': (set('kD'), "S,I=black, K=white"),
        'KI_black': (set('XD'), "K,I=black, S=white"),
    }

    for width in widths:
        if n_leaves < width:
            continue
        height = n_leaves // width
        actual_pixels = width * height
        if actual_pixels < 100:
            continue

        print(f"\n  Width={width}: {width}x{height} ({actual_pixels:,} pixels, {n_leaves - actual_pixels} unused)")

        for map_name, (black_set, desc) in mappings.items():
            pixels = [1 if c in black_set else 0 for c in lvs[:actual_pixels]]
            n_black = sum(pixels)
            pct_black = n_black / actual_pixels * 100

            # Only save if the ratio isn't extreme
            if 5 < pct_black < 95:
                fname = f"{name}_{width}x{height}_{map_name}.pgm"
                fpath = os.path.join(output_dir, fname)
                # Convert to grayscale: 0=black, 255=white
                gray = [0 if p else 255 for p in pixels]
                write_pgm(fpath, width, height, gray)
                print(f"    {desc}: {pct_black:.1f}% black -> {fname}")


def try_inorder_image(compact, name, output_dir, width=4096):
    """Try creating image from inorder tree traversal."""
    print(f"\n=== {name} (inorder traversal) ===")

    if len(compact) > 1000000:
        print("  Too large for tree-based traversal")
        return

    lvs = compact_tree_leaves_inorder(compact)
    n_leaves = len(lvs)
    print(f"  Inorder leaves: {n_leaves:,}")

    if n_leaves < width:
        return

    height = n_leaves // width
    actual_pixels = width * height
    if actual_pixels < 100:
        return

    # Try K=black (most promising for image data)
    pixels = [0 if c == 'X' else 255 for c in lvs[:actual_pixels]]
    fname = f"{name}_inorder_{width}x{height}.pgm"
    fpath = os.path.join(output_dir, fname)
    write_pgm(fpath, width, height, pixels)
    print(f"  K=black: saved {fname}")


def main():
    print("=== Step 5: Image from Data ===\n")

    base = os.path.join(os.path.dirname(__file__), '..')
    ext_dir = os.path.join(base, 'extracted')
    output_dir = os.path.join(base, 'images')
    os.makedirs(output_dir, exist_ok=True)

    # === 1. Item 9 (largest data item) ===
    item9_path = os.path.join(ext_dir, 'data_items', 'item_09.txt')
    if os.path.exists(item9_path):
        with open(item9_path, 'r') as f:
            item9 = f.read().strip()
        try_image_from_leaves(item9, "item09", output_dir,
                             widths=[4096, 2048, 1024, 512, 256, 128, 64])
        try_inorder_image(item9, "item09", output_dir, width=512)

    # === 2. Data chain ===
    dc_path = os.path.join(ext_dir, 'data_chain.txt')
    if os.path.exists(dc_path):
        with open(dc_path, 'r') as f:
            dc = f.read().strip()
        try_image_from_leaves(dc, "data_chain", output_dir,
                             widths=[4096, 2048, 1024, 512, 256, 128])

    # === 3. RIGHT subtree ===
    right_path = os.path.join(ext_dir, 'right.txt')
    if os.path.exists(right_path):
        with open(right_path, 'r') as f:
            right = f.read().strip()
        try_image_from_leaves(right, "right", output_dir,
                             widths=[4096, 2048, 1024])

    # === 4. Entire compact file ===
    compact_path = os.path.join(base, 'very_large_txt', 'stars_compact.txt')
    if os.path.exists(compact_path):
        with open(compact_path, 'r') as f:
            compact = f.read().strip()
        try_image_from_leaves(compact, "full", output_dir,
                             widths=[4096, 2048, 1024])

    # === 5. RIGHT_FUNC (the H decoder) ===
    rf_path = os.path.join(ext_dir, 'right_func.txt')
    if os.path.exists(rf_path):
        with open(rf_path, 'r') as f:
            rf = f.read().strip()
        try_image_from_leaves(rf, "right_func", output_dir,
                             widths=[4096, 2048, 1024])

    print("\n\nDone! Check images/ directory for output.")


if __name__ == '__main__':
    main()
