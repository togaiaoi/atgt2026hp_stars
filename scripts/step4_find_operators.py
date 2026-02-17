#!/usr/bin/env python3
"""
Step 4: Find known operator patterns in the decoder.

The GM hint says: "Compare operators like + and - with the converted tree to restore the decoder."

This script:
1. Extracts X and Y from LEFT = ((S X)(K Y))
2. Searches for known operator patterns in the decoder components
3. Reports findings
"""

import os
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.setrecursionlimit(50000)

# Known operator patterns in compact notation
# From the reference: SKI notation -> compact (S->k, K->X, I->D)
KNOWN_PATTERNS = {
    # Combinators
    'B (S(KS)K)': 'kXk--X-',
    'C (S(S(KS)(S(KK)S))(KK))': None,  # complex, skip

    # Boolean values
    'true (S(KK)I)': 'kXX--D-',
    'false (KI)': 'XD-',

    # Operators from reference
    'not': 'kkD-XXD----XkXX--D---',
    'format (I)': 'D',
    'recursive (Y)': 'kkkXk--kXX--D---XkD-D----kkkXk--kXX--D---XkD-D----',

    # Pair encoding: pair(A,B) = S(KK)(S(SI(KA))(KB))
    # The S(KK) part: kXX--
    'S(KK) prefix': 'kXX--',

    # SI(K_) pattern: kD-X
    'SI(K_) prefix': 'kD-X',

    # B^n tower pattern: S(K(S(K(...)))) = kXkX...
    'B tower prefix': 'kXkX',
}

# Also search for the half of the Y combinator
KNOWN_PATTERNS['Y half'] = 'kkkXk--kXX--D---XkD-D----'


def find_subtree_end(s, start=0):
    """Find end of first complete subtree from position start."""
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
    return len(s)


def split_application(s):
    """Split (LEFT RIGHT) into LEFT, RIGHT."""
    inner = s[:-1]  # remove trailing -
    balance = 0
    for i in range(len(inner) - 1, -1, -1):
        c = inner[i]
        if c in ('k', 'D', 'X'):
            balance -= 1
        elif c == '-':
            balance += 1
        if balance == -1:
            return inner[:i], inner[i:]
    raise ValueError("Could not split")


def extract_left_components(left_compact):
    """Extract X and Y from LEFT = ((S X)(K Y)) = kX---XY---."""
    # LEFT = ((S X) (K Y)) in tree form
    # The compact encoding is:
    #   encode(App(App(S,X), App(K,Y)))
    #   = encode(App(S,X)) + encode(App(K,Y)) + '-'
    #   = (k + encode(X) + -) + (X + encode(Y) + -) + -
    #   = k<X_compact>-X<Y_compact>--

    # Split LEFT into ((S X), (K Y))
    sx_str, ky_str = split_application(left_compact)
    print(f"  (S X) part: {len(sx_str):,} chars")
    print(f"  (K Y) part: {len(ky_str):,} chars")

    # (S X) = k<X_compact>-  ->  remove k prefix and - suffix
    # Actually, (S X) compact = encode(S) + encode(X) + '-' = 'k' + X_compact + '-'
    assert sx_str[0] == 'k', f"Expected 'k' prefix, got '{sx_str[0]}'"
    assert sx_str[-1] == '-', f"Expected '-' suffix, got '{sx_str[-1]}'"
    x_compact = sx_str[1:-1]

    # (K Y) = encode(K) + encode(Y) + '-' = 'X' + Y_compact + '-'
    assert ky_str[0] == 'X', f"Expected 'X' prefix, got '{ky_str[0]}'"
    assert ky_str[-1] == '-', f"Expected '-' suffix, got '{ky_str[-1]}'"
    y_compact = ky_str[1:-1]

    return x_compact, y_compact


def search_patterns(data, name, patterns, max_report=20):
    """Search for known patterns in a compact string."""
    print(f"\n--- Searching in {name} ({len(data):,} chars) ---")
    for pat_name, pat in patterns.items():
        if pat is None:
            continue
        count = 0
        positions = []
        start = 0
        while True:
            pos = data.find(pat, start)
            if pos == -1:
                break
            count += 1
            if len(positions) < max_report:
                positions.append(pos)
            start = pos + 1

        if count > 0:
            pos_str = ', '.join(str(p) for p in positions[:10])
            if count > 10:
                pos_str += f', ... (+{count - 10} more)'
            print(f"  {pat_name}: {count:,} occurrences  [{pat}]")
            print(f"    Positions: {pos_str}")
        else:
            print(f"  {pat_name}: 0 occurrences")


def analyze_structure(compact, name, depth=0, max_depth=5):
    """Analyze the top-level structure of a compact expression."""
    if depth >= max_depth:
        return
    indent = "  " * depth
    if len(compact) <= 3:
        labels = {'k': 'S', 'D': 'I', 'X': 'K'}
        if compact in labels:
            print(f"{indent}{name}: {labels[compact]}")
        else:
            print(f"{indent}{name}: {compact}")
        return

    if compact[-1] != '-':
        # Not an application, just a leaf
        labels = {'k': 'S', 'D': 'I', 'X': 'K'}
        print(f"{indent}{name}: {labels.get(compact, compact[:50])}")
        return

    try:
        left, right = split_application(compact)
        print(f"{indent}{name}: ({len(compact):,} chars) = App(left={len(left):,}, right={len(right):,})")

        # Check if left is a known combinator
        if left == 'k':
            print(f"{indent}  -> S applied to: [{len(right):,} chars]")
            analyze_structure(right, f"S_arg", depth + 1, max_depth)
        elif left == 'X':
            print(f"{indent}  -> K applied to: [{len(right):,} chars]")
            analyze_structure(right, f"K_arg", depth + 1, max_depth)
        elif left == 'D':
            print(f"{indent}  -> I applied to: [{len(right):,} chars]")
        else:
            analyze_structure(left, "func", depth + 1, max_depth)
            analyze_structure(right, "arg", depth + 1, max_depth)
    except Exception as e:
        print(f"{indent}{name}: ERROR splitting ({e})")


def main():
    print("=== Step 4: Find Operator Patterns ===\n")

    base = os.path.join(os.path.dirname(__file__), '..')

    # Read LEFT
    left_path = os.path.join(base, 'extracted', 'left.txt')
    with open(left_path, 'r') as f:
        left = f.read().strip()
    print(f"LEFT: {len(left):,} chars")

    # Extract X and Y from LEFT
    print("\nExtracting X and Y from LEFT = ((S X)(K Y))...")
    x_compact, y_compact = extract_left_components(left)
    print(f"  X: {len(x_compact):,} chars")
    print(f"  Y: {len(y_compact):,} chars")

    # Save X and Y
    ext_dir = os.path.join(base, 'extracted')
    with open(os.path.join(ext_dir, 'left_x.txt'), 'w') as f:
        f.write(x_compact)
    with open(os.path.join(ext_dir, 'left_y.txt'), 'w') as f:
        f.write(y_compact)
    print(f"  Saved to extracted/left_x.txt and extracted/left_y.txt")

    # Analyze top-level structure of X and Y
    print("\n=== X structure (top levels) ===")
    analyze_structure(x_compact, "X", max_depth=8)

    print("\n=== Y structure (top levels) ===")
    analyze_structure(y_compact, "Y", max_depth=8)

    # Search for known patterns in X
    search_patterns(x_compact, "X", KNOWN_PATTERNS)

    # Search for known patterns in Y
    search_patterns(y_compact, "Y", KNOWN_PATTERNS)

    # Search in full LEFT
    search_patterns(left, "LEFT", KNOWN_PATTERNS)

    # Also read the right_func and search (but skip if too large)
    right_func_path = os.path.join(base, 'extracted', 'right_func.txt')
    if os.path.exists(right_func_path):
        rf_size = os.path.getsize(right_func_path)
        print(f"\nright_func.txt: {rf_size:,} bytes")
        if rf_size < 50_000_000:
            print("Searching for patterns in right_func (this may take a moment)...")
            with open(right_func_path, 'r') as f:
                right_func = f.read().strip()

            # Search for distinctive patterns only (skip common ones)
            distinctive = {k: v for k, v in KNOWN_PATTERNS.items()
                          if v and len(v) >= 10}
            search_patterns(right_func, "RIGHT_FUNC", distinctive)

            # Also analyze top-level structure of right_func
            print("\n=== RIGHT_FUNC structure (top levels) ===")
            try:
                rf_left, rf_right = split_application(right_func)
                print(f"  Left (B tower?): {len(rf_left):,} chars")
                print(f"  Right (H): {len(rf_right):,} chars")

                # Analyze the B tower
                print(f"\n  B tower first 200 chars: {rf_left[:200]}")

                # Count B tower depth
                s = rf_left
                b_depth = 0
                while len(s) > 4:
                    try:
                        inner_l, inner_r = split_application(s)
                        if inner_l[0] == 'k' and len(inner_l) > 2:
                            # Check if inner_l = S(K(...))
                            if inner_l[1] == 'X':
                                # inner_l = kX<inner>- = S(K(inner))
                                b_depth += 1
                                s = inner_r
                                continue
                        break
                    except:
                        break
                print(f"  B tower depth: {b_depth}")
                if b_depth > 0:
                    print(f"  H (after B tower): {len(s):,} chars")
                    print(f"  H first 200 chars: {s[:200]}")

                    # Save H
                    with open(os.path.join(ext_dir, 'h_decoder.txt'), 'w') as f:
                        f.write(s)
                    print(f"  Saved H to extracted/h_decoder.txt")

            except Exception as e:
                print(f"  Error analyzing right_func: {e}")

    # Now let's also try to analyze the right_func's B^n(H) structure
    # by looking at the compact string directly
    print("\n=== Analyzing B tower in right_func ===")
    with open(right_func_path, 'r') as f:
        rf_head = f.read(1000)
    print(f"  First 200 chars: {rf_head[:200]}")

    # Count kX prefix repetitions
    pos = 0
    kx_count = 0
    while pos + 1 < len(rf_head) and rf_head[pos:pos+2] == 'kX':
        kx_count += 1
        pos += 2
    print(f"  Leading kX count: {kx_count}")
    if kx_count > 0:
        print(f"  After kX*{kx_count}: {rf_head[pos:pos+100]}")


if __name__ == '__main__':
    main()
