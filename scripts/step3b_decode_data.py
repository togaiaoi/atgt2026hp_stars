#!/usr/bin/env python3
"""
Step 3b: Decode DATA_CHAIN items using tree-level pattern matching.

The data items have layered structure:
  B^n(SI(K(V))) where V is the actual value

After stripping all wrappers, V is decoded as:
  - pair(A, B) = S(KK)(S(SI(KA))(KB))  -> extract A, B
  - true = S(KK)I                        -> bit 1
  - false = KI                           -> bit 0 / nil
  - K                                    -> constant K
  - I                                    -> identity
  - Numbers: pair chains of bits (LSB-first Scott encoding)
"""

import os
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.setrecursionlimit(50000)


def compact_to_tree(s):
    """Parse compact string to tree. Leaves are strings, apps are tuples."""
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
    assert len(stack) == 1, f"Expected 1 on stack, got {len(stack)}"
    return stack[0]


def tree_to_compact(tree):
    """Convert tree back to compact string."""
    if isinstance(tree, str):
        return {'S': 'k', 'K': 'X', 'I': 'D'}[tree]
    left, right = tree
    return tree_to_compact(left) + tree_to_compact(right) + '-'


def tree_to_str(tree, depth=0):
    """Pretty-print tree as SKI expression."""
    if isinstance(tree, str):
        return tree
    left, right = tree
    l = tree_to_str(left)
    r = tree_to_str(right)
    if isinstance(right, tuple):
        return f"({l} ({r}))"
    return f"({l} {r})"


def tree_size(tree):
    """Count total nodes (leaves + applications)."""
    if isinstance(tree, str):
        return 1
    return 1 + tree_size(tree[0]) + tree_size(tree[1])


# === Wrapper stripping ===

def strip_b_layers(tree):
    """
    Strip B^n composition layers: S(K(S(K(...S(K(V))...))))
    Each B level looks like: (S (K inner)) applied to next_arg
    Actually B^n(f) = ((S (K f)) next)... the tree structure is:
    ((S (K inner)) arg) at each level
    """
    n = 0
    current = tree
    while isinstance(current, tuple):
        left, right = current
        if isinstance(left, tuple):
            ll, lr = left
            if ll == 'S' and isinstance(lr, tuple) and lr[0] == 'K':
                # This is ((S (K inner)) arg)  -- but this is B applied, not B wrapping
                # Actually B^n wrapping in the data chain items is:
                # S(K(S(K(...V...))))  — nested inside K applications
                # Not applied to anything
                break
        break

    # Alternative: B^n wrapper as kXkX...V...---- pattern
    # Strip by checking tree = (S (K inner))
    current = tree
    while isinstance(current, tuple) and len(current) == 2:
        left, right = current
        if left == 'S' and isinstance(right, tuple) and right[0] == 'K':
            # (S (K inner)) = one B level
            n += 1
            current = right[1]  # inner
        else:
            break

    return n, current


def strip_si_k_wrapper(tree):
    """
    Strip S(SI(K(V))) wrapper.
    Tree structure: (S ((S I) (K V)))
    """
    if not isinstance(tree, tuple):
        return None
    left, right = tree
    if left != 'S':
        return None
    if not isinstance(right, tuple):
        return None
    rl, rr = right
    # rl should be (S I)
    if not (isinstance(rl, tuple) and rl == ('S', 'I')):
        return None
    # rr should be (K V)
    if not (isinstance(rr, tuple) and rr[0] == 'K'):
        return None
    return rr[1]  # V


def full_unwrap(tree):
    """Strip all wrapper layers: B^n then SI(K(V))."""
    b_level, inner = strip_b_layers(tree)
    si_result = strip_si_k_wrapper(inner)
    if si_result is not None:
        return b_level, True, si_result
    return b_level, False, inner


# === Scott encoding patterns ===

def is_true(tree):
    """Check if tree is true = S(KK)I = ((S (K K)) I)"""
    if isinstance(tree, tuple):
        left, right = tree
        if right == 'I' and isinstance(left, tuple):
            ll, lr = left
            if ll == 'S' and isinstance(lr, tuple) and lr == ('K', 'K'):
                return True
    return False


def is_false(tree):
    """Check if tree is false/nil = KI = (K I)"""
    return isinstance(tree, tuple) and tree == ('K', 'I')


def try_decode_pair(tree):
    """
    Decode pair(A, B) = S(KK)(S(SI(KA))(KB))
    Tree: ((S (K K)) ((S ((S I) (K A))) (K B)))
    Returns (A, B) or None.
    """
    if not isinstance(tree, tuple):
        return None
    left, right = tree

    # left = (S (K K))
    if not (isinstance(left, tuple) and left[0] == 'S'
            and isinstance(left[1], tuple) and left[1] == ('K', 'K')):
        return None

    # right = ((S ((S I) (K A))) (K B))
    if not isinstance(right, tuple):
        return None
    r_left, r_right = right

    # r_right = (K B)
    if not (isinstance(r_right, tuple) and r_right[0] == 'K'):
        return None
    b_val = r_right[1]

    # r_left = (S ((S I) (K A)))
    if not (isinstance(r_left, tuple) and r_left[0] == 'S'):
        return None
    inner = r_left[1]

    # inner = ((S I) (K A))
    if not isinstance(inner, tuple):
        return None
    i_left, i_right = inner

    if not (isinstance(i_left, tuple) and i_left == ('S', 'I')):
        return None
    if not (isinstance(i_right, tuple) and i_right[0] == 'K'):
        return None
    a_val = i_right[1]

    return a_val, b_val


def try_decode_pair_inner(tree):
    """
    Decode the INNER pair form: S(SI(KA))(KB)
    This is the pair encoding WITHOUT the S(KK) wrapper.
    Tree: ((S ((S I) (K A))) (K B))
    Returns (A, B) or None.
    """
    if not isinstance(tree, tuple):
        return None
    left, right = tree

    # left = (S ((S I) (K A)))
    if not (isinstance(left, tuple) and left[0] == 'S'):
        return None
    inner = left[1]

    # inner = ((S I) (K A))
    if not isinstance(inner, tuple):
        return None
    i_left, i_right = inner

    if not (isinstance(i_left, tuple) and i_left == ('S', 'I')):
        return None
    if not (isinstance(i_right, tuple) and i_right[0] == 'K'):
        return None
    a_val = i_right[1]

    # right = (K B)
    if not (isinstance(right, tuple) and right[0] == 'K'):
        return None
    b_val = right[1]

    return a_val, b_val


def decode_number(tree):
    """
    Decode a Scott-encoded number from pair chain.
    Number = pair(bit, rest) where bit is true(1)/false(0)
    Terminal: false/nil = KI
    """
    bits = []
    current = tree
    max_iter = 1000

    for _ in range(max_iter):
        # Try standard pair
        pair_result = try_decode_pair(current)
        if pair_result is None:
            # Try inner pair form
            pair_result = try_decode_pair_inner(current)
        if pair_result is None:
            break

        bit_tree, rest_tree = pair_result

        if is_true(bit_tree):
            bits.append(1)
        elif is_false(bit_tree):
            bits.append(0)
        elif bit_tree == 'K':
            # K alone might represent true (extensionally equivalent)
            bits.append(1)
        else:
            # Unknown bit value
            compact = tree_to_compact(bit_tree)
            print(f"    Unknown bit: {compact[:50]}")
            break

        current = rest_tree

    if not bits:
        return None

    # Verify terminal is nil/false
    if not (is_false(current) or current == ('K', 'I')):
        pass  # Non-standard termination, still try to decode

    # LSB-first to number
    value = 0
    for i, bit in enumerate(bits):
        value += bit * (2 ** i)
    return value


def decode_list(tree):
    """
    Decode a Scott-encoded list of values.
    List = pair(head, tail) chain, terminated by false/nil.
    """
    items = []
    current = tree
    max_iter = 1000

    for _ in range(max_iter):
        pair_result = try_decode_pair(current)
        if pair_result is None:
            pair_result = try_decode_pair_inner(current)
        if pair_result is None:
            break

        head, tail = pair_result
        items.append(head)
        current = tail

    return items, current


def decode_number_list(tree):
    """Decode a list of Scott-encoded numbers."""
    items, terminal = decode_list(tree)
    numbers = []
    for item in items:
        num = decode_number(item)
        if num is not None:
            numbers.append(num)
        else:
            compact = tree_to_compact(item)
            numbers.append(f"?({compact[:30]})")
    return numbers


def analyze_tree(tree, prefix=""):
    """Analyze and describe a tree structure."""
    if isinstance(tree, str):
        return f"{prefix}{tree}"

    # Check for known patterns
    if is_true(tree):
        return f"{prefix}TRUE (S(KK)I)"
    if is_false(tree):
        return f"{prefix}FALSE/NIL (KI)"
    if tree == ('K', 'K'):
        return f"{prefix}(K K)"

    pair_result = try_decode_pair(tree)
    if pair_result:
        a, b = pair_result
        return f"{prefix}PAIR({analyze_tree(a)}, {analyze_tree(b)})"

    pair_inner = try_decode_pair_inner(tree)
    if pair_inner:
        a, b = pair_inner
        return f"{prefix}PAIR_INNER({analyze_tree(a)}, {analyze_tree(b)})"

    num = decode_number(tree)
    if num is not None:
        return f"{prefix}NUMBER({num})"

    compact = tree_to_compact(tree)
    if len(compact) <= 50:
        return f"{prefix}TREE({compact})"
    return f"{prefix}TREE({len(compact)} chars)"


def main():
    print("=== Step 3b: Decode DATA_CHAIN Items ===\n")

    items_dir = os.path.join(os.path.dirname(__file__), '..', 'extracted', 'data_items')
    if not os.path.exists(items_dir):
        print(f"Data items directory not found: {items_dir}")
        print("Run step2_parse_structure.py first.")
        sys.exit(1)

    # Process each item (only the base items, not _unwrapped)
    print("=== Processing individual items ===\n")
    for i in range(25):
        fname = f"item_{i:02d}.txt"
        fpath = os.path.join(items_dir, fname)
        if not os.path.exists(fpath):
            print(f"  Item {i:2d}: NOT FOUND")
            continue

        with open(fpath, 'r') as f:
            compact = f.read().strip()

        if len(compact) > 50000:
            print(f"  Item {i:2d}: {len(compact):,} chars (too large for tree parsing)")
            continue

        tree = compact_to_tree(compact)
        b_level, has_si, inner = full_unwrap(tree)

        wrapper_desc = f"B^{b_level}"
        if has_si:
            wrapper_desc += "+SI(K)"

        # Analyze the inner value
        if isinstance(inner, str):
            print(f"  Item {i:2d}: [{len(compact):>6} chars] {wrapper_desc} -> {inner}")
            continue

        # Try to decode as number
        num = decode_number(inner)
        if num is not None:
            print(f"  Item {i:2d}: [{len(compact):>6} chars] {wrapper_desc} -> NUMBER({num})")
            continue

        # Try to decode as list of numbers
        numbers = decode_number_list(inner)
        if numbers and len(numbers) > 0 and all(isinstance(n, int) for n in numbers):
            print(f"  Item {i:2d}: [{len(compact):>6} chars] {wrapper_desc} -> LIST({len(numbers)} items): {numbers}")
            continue

        # Try as pair
        pair_result = try_decode_pair(inner)
        if pair_result:
            a, b = pair_result
            a_desc = analyze_tree(a)
            b_desc = analyze_tree(b)
            print(f"  Item {i:2d}: [{len(compact):>6} chars] {wrapper_desc} -> PAIR({a_desc}, {b_desc})")
            continue

        # Fallback
        inner_compact = tree_to_compact(inner)
        inner_preview = inner_compact[:80] + ('...' if len(inner_compact) > 80 else '')
        print(f"  Item {i:2d}: [{len(compact):>6} chars] {wrapper_desc} -> UNKNOWN({len(inner_compact)} chars) {inner_preview}")

    # Now handle item 9 specially (too large for full tree parsing)
    print("\n=== Item 9 (large) ===")
    item9_path = os.path.join(items_dir, 'item_09.txt')
    with open(item9_path, 'r') as f:
        item9 = f.read().strip()
    print(f"  Size: {len(item9):,} chars")
    print(f"  First 200 chars: {item9[:200]}")

    # Try to detect B^n wrapper from the compact string directly
    n_layers = 0
    s = item9
    while s.startswith('kX') and len(s) > 4:
        # Check balance of inner string
        candidate = s[2:]
        # Find if removing one -- from end is valid
        if candidate.endswith('--'):
            inner = candidate[:-2]
            balance = 0
            valid = True
            for c in inner:
                if c in ('k', 'D', 'X'):
                    balance += 1
                elif c == '-':
                    balance -= 1
                if balance < 0:
                    valid = False
                    break
            if valid and balance == 1:
                s = inner
                n_layers += 1
                continue
        break

    print(f"  B^{n_layers} wrapper detected")
    if n_layers > 0:
        print(f"  Inner size after B^{n_layers}: {len(s):,} chars")
        print(f"  Inner first 200 chars: {s[:200]}")

    # Check for SI(K(V)) wrapper in compact string
    # S(SI(K(V))) = k + kD-X<V>-- + - = kkD-X<V>---
    if s.startswith('kkD-X') and s.endswith('---'):
        print(f"  SI(K(V)) wrapper detected")
        # The inner V is between position 5 and -3
        # But need to find exact boundary via tree parsing
        # For large items, use string-level approach
        # Actually, just strip the prefix and suffix
        # kkD-X<V>--- means V starts at position 5, and we need to find where V ends
        # by balance counting
        inner_start = 5
        # Count from the left after position 5 to find the complete subtree
        balance = 0
        v_end = inner_start
        for j in range(inner_start, len(s)):
            c = s[j]
            if c in ('k', 'D', 'X'):
                balance += 1
            elif c == '-':
                balance -= 1
            if balance == 1:
                v_end = j + 1
                # Check remaining characters are ---
                remaining = s[v_end:]
                if remaining == '---':
                    v_str = s[inner_start:v_end]
                    print(f"  V (inner value): {len(v_str):,} chars")
                    print(f"  V first 200 chars: {v_str[:200]}")
                    # Save
                    output_dir = os.path.join(os.path.dirname(__file__), '..', 'extracted')
                    with open(os.path.join(output_dir, 'item_09_value.txt'), 'w') as f:
                        f.write(v_str)
                    print(f"  Saved to extracted/item_09_value.txt")
                break


    # Decode item 2 in detail (33-number list)
    print("\n=== Item 2 detailed decode ===")
    item2_path = os.path.join(items_dir, 'item_02.txt')
    with open(item2_path, 'r') as f:
        item2 = f.read().strip()

    tree2 = compact_to_tree(item2)
    b_level, has_si, inner2 = full_unwrap(tree2)
    print(f"  B^{b_level}, SI(K): {has_si}")

    if inner2:
        numbers = decode_number_list(inner2)
        print(f"  Decoded numbers ({len(numbers)}): {numbers}")

    # Decode item 19 in detail (5-number list)
    print("\n=== Item 19 detailed decode ===")
    item19_path = os.path.join(items_dir, 'item_19.txt')
    with open(item19_path, 'r') as f:
        item19 = f.read().strip()

    tree19 = compact_to_tree(item19)
    b_level, has_si, inner19 = full_unwrap(tree19)
    print(f"  B^{b_level}, SI(K): {has_si}")

    if inner19:
        numbers = decode_number_list(inner19)
        print(f"  Decoded numbers ({len(numbers)}): {numbers}")

    # Decode item 6 in detail
    print("\n=== Item 6 detailed decode ===")
    item6_path = os.path.join(items_dir, 'item_06.txt')
    with open(item6_path, 'r') as f:
        item6 = f.read().strip()

    tree6 = compact_to_tree(item6)
    b_level, has_si, inner6 = full_unwrap(tree6)
    print(f"  B^{b_level}, SI(K): {has_si}")
    print(f"  Inner tree: {analyze_tree(inner6)}")


if __name__ == '__main__':
    main()
