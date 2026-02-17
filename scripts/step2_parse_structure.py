#!/usr/bin/env python3
"""
Step 2: Parse compact string and extract structure

Reads stars_compact.txt (k/D/X/- format) and:
1. Finds top-level split: (LEFT RIGHT)
2. Extracts LEFT and RIGHT as substrings
3. Analyzes RIGHT's structure: ((B^n H) DATA_CHAIN)
4. Extracts DATA_CHAIN items
5. Saves each part to separate files
"""

import os
import sys
import io
import json

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


def find_subtree_end(s, start=0):
    """
    Find where the first complete subtree ends in compact postorder string.
    A complete subtree has balance=1 (one more leaf than applications).
    Returns the index after the last character of the subtree.
    """
    balance = 0
    i = start
    while i < len(s):
        c = s[i]
        if c in ('k', 'D', 'X'):
            balance += 1
        elif c == '-':
            balance -= 1
        i += 1
        if balance == 1:
            return i
    raise ValueError(f"No complete subtree found starting at {start}")


def split_application(s):
    """
    Split a compact string representing (LEFT RIGHT) into LEFT and RIGHT.
    The string is: LEFT_str + RIGHT_str + '-'
    """
    if not s.endswith('-'):
        raise ValueError("String doesn't end with '-' (not an application)")

    # Remove the trailing '-' (top-level application marker)
    inner = s[:-1]

    # Find the end of the first complete subtree (LEFT)
    left_end = find_subtree_end(inner, 0)

    left = inner[:left_end]
    right = inner[left_end:]

    # Verify right is also a complete subtree
    verify_end = find_subtree_end(right, 0)
    if verify_end != len(right):
        raise ValueError(f"RIGHT is not a single complete subtree (end={verify_end}, len={len(right)})")

    return left, right


def count_leaves(s):
    """Count the number of leaves (k/D/X) in a compact string"""
    return sum(1 for c in s if c in ('k', 'D', 'X'))


def count_apps(s):
    """Count the number of applications (-) in a compact string"""
    return s.count('-')


def is_leaf(s):
    """Check if the string is a single leaf"""
    return len(s) == 1 and s in ('k', 'D', 'X')


def tree_to_ski(s):
    """Parse compact string into SKI tree using stack algorithm"""
    stack = []
    for c in s:
        if c == 'k':
            stack.append('S')
        elif c == 'D':
            stack.append('I')
        elif c == 'X':
            stack.append('K')
        elif c == '-':
            y = stack.pop()
            x = stack.pop()
            stack.append((x, y))
    assert len(stack) == 1
    return stack[0]


def ski_to_str(tree, depth=0):
    """Pretty print SKI tree"""
    if isinstance(tree, str):
        return tree
    left, right = tree
    l = ski_to_str(left)
    r = ski_to_str(right)
    return f"({l} {r})"


def extract_data_chain(s):
    """
    Extract items from DATA_CHAIN.
    DATA_CHAIN is a linked list of applications.
    """
    items = []
    remaining = s
    while remaining:
        if is_leaf(remaining):
            items.append(remaining)
            break
        # remaining = (left right) = left_str + right_str + '-'
        left, right = split_application(remaining)
        items.append(left)
        remaining = right
    return items


def detect_b_tower(s):
    """
    Detect B^n composition tower.
    B = S(KS)K, so B^n has a specific pattern.
    B = S(KS)K in compact: kXk--X-
    B^n(f) = S(K(S(K(...(S(K f))...))))

    Looking for the pattern: the tower applies S(K ...) repeatedly.
    """
    # Parse the tree
    tree = tree_to_ski(s)

    # Count nesting depth of S(K ...) pattern
    depth = 0
    current = tree
    while isinstance(current, tuple):
        left, right = current
        # Check if this is S(K(something))(...) = ((S (K x)) y)
        if isinstance(left, tuple):
            ll, lr = left
            if ll == 'S' and isinstance(lr, tuple) and lr[0] == 'K':
                depth += 1
                # Next level is inside the K: lr[1]
                # But the right side is the next part of the tower or the function
                current = right
                continue
        break

    return depth, current


def analyze_structure(compact_str):
    """Main analysis: parse the compact string and extract all components"""
    total_len = len(compact_str)
    total_leaves = count_leaves(compact_str)
    total_apps = count_apps(compact_str)

    print(f"Total length: {total_len:,} chars")
    print(f"  Leaves (k/D/X): {total_leaves:,}")
    print(f"  Applications (-): {total_apps:,}")
    print(f"  Valid tree: {total_leaves - total_apps == 1}")
    print()

    # Step 1: Split top-level (LEFT RIGHT)
    print("=== Top-level split ===")
    left, right = split_application(compact_str)
    print(f"LEFT:  {len(left):>12,} chars  ({len(left)/total_len*100:.1f}%)")
    print(f"RIGHT: {len(right):>12,} chars  ({len(right)/total_len*100:.1f}%)")
    print()

    # Analyze LEFT
    print("=== LEFT analysis ===")
    analyze_subtree("LEFT", left)
    print()

    # Analyze LEFT structure: ((S X)(K Y))
    try:
        left_tree = tree_to_ski(left)
        print(f"LEFT as SKI (truncated): {ski_to_str(left_tree)[:200]}...")
        # Check if it's ((S X)(K Y))
        if isinstance(left_tree, tuple):
            ll, lr = left_tree
            if isinstance(ll, tuple) and ll[0] == 'S':
                x_tree = ll[1]
                if isinstance(lr, tuple) and lr[0] == 'K':
                    y_tree = lr[1]
                    print(f"LEFT matches ((S X)(K Y)) pattern!")
                    print(f"  X size: {len(ski_to_compact(x_tree)):,} chars")
                    print(f"  Y size: {len(ski_to_compact(y_tree)):,} chars")
    except Exception as e:
        print(f"  LEFT tree analysis failed: {e}")
    print()

    # Analyze RIGHT
    print("=== RIGHT analysis ===")
    analyze_subtree("RIGHT", right)

    # Split RIGHT into function part and data part
    print("\nSplitting RIGHT into ((B^n H) DATA_CHAIN)...")
    right_func, right_data = split_application(right)
    print(f"  Function part: {len(right_func):,} chars")
    print(f"  Data part:     {len(right_data):,} chars")

    # Extract DATA_CHAIN items
    print("\n=== DATA_CHAIN analysis ===")
    items = extract_data_chain(right_data)
    print(f"Number of items: {len(items)}")
    print()

    for i, item in enumerate(items):
        item_leaves = count_leaves(item)
        item_apps = count_apps(item)
        prefix = item[:80] + ('...' if len(item) > 80 else '')
        print(f"  Item {i:2d}: {len(item):>8,} chars  "
              f"({item_leaves} leaves, {item_apps} apps)  {prefix}")

    # Analyze the function part of RIGHT
    print("\n=== RIGHT function part analysis ===")
    print(f"Detecting B^n tower...")
    try:
        # The function part is ((B^n H))
        # It could be structured as a series of S(K ...) applications
        bf_left, bf_right = split_application(right_func)
        print(f"  Left (B^n or similar): {len(bf_left):,} chars")
        print(f"  Right (H or operand): {len(bf_right):,} chars")
    except Exception as e:
        print(f"  Could not split function part: {e}")

    return {
        'left': left,
        'right': right,
        'right_func': right_func,
        'right_data': right_data,
        'data_items': items,
    }


def ski_to_compact(tree):
    """Convert SKI tree back to compact string"""
    if isinstance(tree, str):
        if tree == 'S':
            return 'k'
        elif tree == 'K':
            return 'X'
        elif tree == 'I':
            return 'D'
        else:
            return tree
    left, right = tree
    return ski_to_compact(left) + ski_to_compact(right) + '-'


def analyze_subtree(name, s):
    """Print basic analysis of a compact subtree"""
    leaves = count_leaves(s)
    apps = count_apps(s)
    k_count = s.count('X')
    d_count = s.count('D')
    s_count = s.count('k')
    print(f"  Length: {len(s):,}")
    print(f"  Leaves: {leaves:,} (S:{s_count:,} K:{k_count:,} I:{d_count:,})")
    print(f"  Applications: {apps:,}")
    print(f"  First 100 chars: {s[:100]}")


def decode_scott_number(s):
    """
    Decode a Scott-encoded number from compact string.
    Numbers are pair chains: pair(bit, rest) where bit is true/false.
    true = S(KK)I = kXX--D-
    false = KI = XD-
    pair(A,B) = S(KK)(S(SI(KA))(KB)) = kXX--kkD-X<A>---X<B>---
    nil = KI = XD-
    """
    tree = tree_to_ski(s)
    return decode_scott_number_tree(tree)


def decode_scott_number_tree(tree):
    """Decode a Scott-encoded number from an SKI tree"""
    bits = []
    current = tree
    while True:
        # Check if current is a pair
        pair_result = try_decode_pair(current)
        if pair_result is None:
            break
        bit_tree, rest_tree = pair_result

        # Decode the bit
        if is_true(bit_tree):
            bits.append(1)
        elif is_false(bit_tree):
            bits.append(0)
        else:
            break

        current = rest_tree

    if not bits:
        return None

    # Convert LSB-first bits to number
    value = 0
    for i, bit in enumerate(bits):
        value += bit * (2 ** i)
    return value


def try_decode_pair(tree):
    """
    Try to decode tree as pair(A, B).
    pair(A,B) = S(KK)(S(SI(KA))(KB))
    = ((S (K K)) ((S ((S I) (K A))) (K B)))
    """
    if not isinstance(tree, tuple):
        return None

    left, right = tree
    # left should be (S (K K))
    if not (isinstance(left, tuple) and left[0] == 'S'
            and isinstance(left[1], tuple) and left[1] == ('K', 'K')):
        return None

    # right should be ((S ((S I) (K A))) (K B))
    if not isinstance(right, tuple):
        return None
    r_left, r_right = right

    # r_right should be (K B)
    if not (isinstance(r_right, tuple) and r_right[0] == 'K'):
        return None
    b = r_right[1]

    # r_left should be (S ((S I) (K A)))
    if not (isinstance(r_left, tuple) and r_left[0] == 'S'):
        return None
    inner = r_left[1]

    # inner should be ((S I) (K A))
    if not isinstance(inner, tuple):
        return None
    i_left, i_right = inner

    # i_left should be (S I)
    if not (isinstance(i_left, tuple) and i_left == ('S', 'I')):
        return None

    # i_right should be (K A)
    if not (isinstance(i_right, tuple) and i_right[0] == 'K'):
        return None
    a = i_right[1]

    return a, b


def is_true(tree):
    """Check if tree represents true = S(KK)I = ((S (K K)) I)"""
    if isinstance(tree, tuple):
        left, right = tree
        if right == 'I' and isinstance(left, tuple):
            if left == ('S', ('K', 'K')):
                return True
    return False


def is_false(tree):
    """Check if tree represents false/nil = KI = (K I)"""
    return isinstance(tree, tuple) and tree == ('K', 'I')


def decode_scott_list(s):
    """Decode a Scott-encoded list of numbers"""
    tree = tree_to_ski(s)
    numbers = []
    current = tree
    while True:
        pair_result = try_decode_pair(current)
        if pair_result is None:
            break
        head, tail = pair_result
        # head should be a Scott-encoded number
        num = decode_scott_number_tree(head)
        if num is not None:
            numbers.append(num)
        else:
            numbers.append(f"<unknown: {ski_to_str(head)[:50]}>")
        current = tail
    return numbers


if __name__ == '__main__':
    print("=== Step 2: Parse Structure ===\n")

    compact_path = os.path.join(os.path.dirname(__file__), '..', 'very_large_txt', 'stars_compact.txt')

    if not os.path.exists(compact_path):
        print(f"Compact file not found: {compact_path}")
        print("Run step1_compress.py first.")
        sys.exit(1)

    print(f"Reading {compact_path}...")
    with open(compact_path, 'r') as f:
        compact = f.read()
    print(f"Loaded {len(compact):,} characters\n")

    result = analyze_structure(compact)

    # Save extracted parts
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'extracted')
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'left.txt'), 'w') as f:
        f.write(result['left'])
    print(f"\nSaved LEFT to extracted/left.txt ({len(result['left']):,} chars)")

    with open(os.path.join(output_dir, 'right.txt'), 'w') as f:
        f.write(result['right'])
    print(f"Saved RIGHT to extracted/right.txt ({len(result['right']):,} chars)")

    with open(os.path.join(output_dir, 'right_func.txt'), 'w') as f:
        f.write(result['right_func'])
    print(f"Saved RIGHT function part to extracted/right_func.txt ({len(result['right_func']):,} chars)")

    with open(os.path.join(output_dir, 'data_chain.txt'), 'w') as f:
        f.write(result['right_data'])
    print(f"Saved DATA_CHAIN to extracted/data_chain.txt ({len(result['right_data']):,} chars)")

    # Save individual data items
    items_dir = os.path.join(output_dir, 'data_items')
    os.makedirs(items_dir, exist_ok=True)
    for i, item in enumerate(result['data_items']):
        with open(os.path.join(items_dir, f'item_{i:02d}.txt'), 'w') as f:
            f.write(item)

    print(f"Saved {len(result['data_items'])} data items to extracted/data_items/")

    # Try to decode known data items
    print("\n=== Decoding data items ===")
    for i, item in enumerate(result['data_items']):
        if len(item) <= 2000 and len(item) > 1:
            try:
                # Try as number
                num = decode_scott_number(item)
                if num is not None:
                    print(f"  Item {i:2d}: number = {num}")
                    continue
            except Exception:
                pass

            try:
                # Try as list of numbers
                nums = decode_scott_list(item)
                if nums:
                    print(f"  Item {i:2d}: list = {nums}")
                    continue
            except Exception:
                pass

            print(f"  Item {i:2d}: {len(item)} chars (could not decode)")
        elif len(item) == 1:
            if item == 'X':
                print(f"  Item {i:2d}: K (constant)")
            elif item == 'D':
                print(f"  Item {i:2d}: I (identity)")
            elif item == 'k':
                print(f"  Item {i:2d}: S (substitution)")
        else:
            print(f"  Item {i:2d}: {len(item):,} chars (too large to decode directly)")
