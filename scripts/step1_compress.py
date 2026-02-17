#!/usr/bin/env python3
"""
Step 1: Compress stars.txt (l/- format) to compact (k/D/X/- format)

Key insight: K, I, S patterns in l/- notation can be distinguished by their
first 3-4 characters:
  I starts with: ll-   (27 chars total)
  K starts with: lll-  (21 chars total)
  S starts with: llll  (31 chars total)
  Application:   -     (1 char)

So we can tokenize the l/- string left-to-right without building a tree.
This is O(n) time and O(1) memory (plus output buffer).
"""

import os
import sys
import io
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# SKI combinator patterns in l/- format
K_STR = 'lll--lll--l-l-l-l-l--'          # 21 chars, K (named)  -> compact 'X'
I_STR = 'll-l-lll--lll--l-l-l-l-l---'    # 27 chars, I (equals) -> compact 'D'
S_STR = 'llll-ll-ll---llll-ll-------l-l-' # 31 chars, S (the_call) -> compact 'k'

K_LEN = len(K_STR)  # 21
I_LEN = len(I_STR)  # 27
S_LEN = len(S_STR)  # 31


def validate_patterns():
    """Verify pattern prefixes are distinguishable"""
    assert K_LEN == 21
    assert I_LEN == 27
    assert S_LEN == 31
    # Verify distinguishing prefixes
    assert I_STR[:3] == 'll-'   # I starts with ll-
    assert K_STR[:4] == 'lll-'  # K starts with lll-
    assert S_STR[:4] == 'llll'  # S starts with llll
    # Verify they don't conflict
    assert I_STR[2] == '-'      # 3rd char distinguishes I from K/S
    assert K_STR[3] == '-'      # 4th char distinguishes K from S
    assert S_STR[3] == 'l'      # 4th char of S is l
    print("  Pattern prefixes verified: I=ll-, K=lll-, S=llll")
    print(f"  K: {K_LEN} chars")
    print(f"  I: {I_LEN} chars")
    print(f"  S: {S_LEN} chars")


def compress(input_path, output_path):
    """
    Compress l/- encoding to compact k/D/X/- encoding using tokenizer approach.

    At each position in the l/- string:
    - If '-': output '-' (compact application), advance 1
    - If 'l': determine which combinator by prefix:
      - "ll-..." -> I pattern (27 chars) -> output 'D'
      - "lll-..." -> K pattern (21 chars) -> output 'X'
      - "llll..." -> S pattern (31 chars) -> output 'k'
    """
    file_size = os.path.getsize(input_path)
    start_time = time.time()

    # Read entire file into memory (~400MB)
    print(f"Reading {file_size:,} bytes...")
    with open(input_path, 'r') as f:
        data = f.read()
    read_time = time.time() - start_time
    print(f"Read complete ({read_time:.1f}s)")

    pos = 0
    n = len(data)
    output = []
    s_count = 0
    k_count = 0
    i_count = 0
    app_count = 0
    last_report = 0

    print("Compressing...")
    while pos < n:
        c = data[pos]

        if c == '-':
            output.append('-')
            app_count += 1
            pos += 1

        elif c == 'l':
            # Determine combinator type by prefix
            if pos + 2 < n and data[pos + 2] == '-':
                # Prefix "ll-" -> I pattern
                end = pos + I_LEN
                if end > n or data[pos:end] != I_STR:
                    print(f"ERROR: Expected I pattern at pos {pos}")
                    if end <= n:
                        print(f"  Got:      {data[pos:end]}")
                    print(f"  Expected: {I_STR}")
                    sys.exit(1)
                output.append('D')
                i_count += 1
                pos = end

            elif pos + 3 < n and data[pos + 3] == '-':
                # Prefix "lll-" -> K pattern
                end = pos + K_LEN
                if end > n or data[pos:end] != K_STR:
                    print(f"ERROR: Expected K pattern at pos {pos}")
                    if end <= n:
                        print(f"  Got:      {data[pos:end]}")
                    print(f"  Expected: {K_STR}")
                    sys.exit(1)
                output.append('X')
                k_count += 1
                pos = end

            elif pos + 3 < n and data[pos + 3] == 'l':
                # Prefix "llll" -> S pattern
                end = pos + S_LEN
                if end > n or data[pos:end] != S_STR:
                    print(f"ERROR: Expected S pattern at pos {pos}")
                    if end <= n:
                        print(f"  Got:      {data[pos:end]}")
                    print(f"  Expected: {S_STR}")
                    sys.exit(1)
                output.append('k')
                s_count += 1
                pos = end

            else:
                print(f"ERROR: Unrecognized pattern at pos {pos}")
                print(f"  Context: ...{data[max(0,pos-10):pos+40]}...")
                sys.exit(1)

        elif c in ('\n', '\r', ' ', '\t'):
            # Skip whitespace (e.g., trailing newline)
            pos += 1

        else:
            print(f"ERROR: Unexpected character '{c}' (ord={ord(c)}) at pos {pos}")
            sys.exit(1)

        # Progress reporting every ~20M input chars
        if pos - last_report >= 20 * 1024 * 1024:
            last_report = pos
            elapsed = time.time() - start_time
            pct = pos / n * 100
            rate = pos / elapsed / 1024 / 1024 if elapsed > 0 else 0
            total_out = s_count + k_count + i_count + app_count
            print(f"  {pct:5.1f}%  pos={pos:>12,}/{n:,}  "
                  f"out={total_out:>10,}  S={s_count:,} K={k_count:,} I={i_count:,}  "
                  f"{rate:.1f} MB/s")

    elapsed = time.time() - start_time
    output_str = ''.join(output)
    total_leaves = s_count + k_count + i_count

    print(f"\n=== Compression complete ===")
    print(f"  Input:  {n:>12,} chars ({n/1024/1024:.1f} MB)")
    print(f"  Output: {len(output_str):>12,} chars ({len(output_str)/1024/1024:.1f} MB)")
    print(f"  Ratio:  {n/len(output_str):.1f}x")
    print(f"  Time:   {elapsed:.1f}s")
    print(f"  S (k):  {s_count:>12,}")
    print(f"  K (X):  {k_count:>12,}")
    print(f"  I (D):  {i_count:>12,}")
    print(f"  Apps:   {app_count:>12,}")
    print(f"  Total:  {total_leaves + app_count:>12,} (leaves + apps)")

    # Verify: leaves - apps should equal 1 (single tree)
    if total_leaves - app_count != 1:
        print(f"\n  WARNING: leaves({total_leaves}) - apps({app_count}) = {total_leaves - app_count} (expected 1)")
    else:
        print(f"  Verified: single valid tree (leaves - apps = 1)")

    # Compare with reference values
    ref_s, ref_k, ref_i, ref_out = 6_708_653, 5_658_387, 2_875_571, 30_485_221
    print(f"\n  Reference comparison:")
    print(f"    S: {s_count:,} (ref: {ref_s:,}) {'OK' if s_count == ref_s else 'MISMATCH'}")
    print(f"    K: {k_count:,} (ref: {ref_k:,}) {'OK' if k_count == ref_k else 'MISMATCH'}")
    print(f"    I: {i_count:,} (ref: {ref_i:,}) {'OK' if i_count == ref_i else 'MISMATCH'}")
    print(f"    Output: {len(output_str):,} (ref: {ref_out:,}) {'OK' if len(output_str) == ref_out else 'MISMATCH'}")

    # Write output
    print(f"\nWriting output to {output_path}...")
    with open(output_path, 'w') as f:
        f.write(output_str)
    print("Done.")


def compress_string(s):
    """Compress a small l/- string to compact (for testing)"""
    pos = 0
    n = len(s)
    result = []
    while pos < n:
        c = s[pos]
        if c == '-':
            result.append('-')
            pos += 1
        elif c == 'l':
            if pos + 2 < n and s[pos + 2] == '-':
                assert s[pos:pos + I_LEN] == I_STR, f"Bad I at {pos}: {s[pos:pos+I_LEN]}"
                result.append('D')
                pos += I_LEN
            elif pos + 3 < n and s[pos + 3] == '-':
                assert s[pos:pos + K_LEN] == K_STR, f"Bad K at {pos}: {s[pos:pos+K_LEN]}"
                result.append('X')
                pos += K_LEN
            elif pos + 3 < n and s[pos + 3] == 'l':
                assert s[pos:pos + S_LEN] == S_STR, f"Bad S at {pos}: {s[pos:pos+S_LEN]}"
                result.append('k')
                pos += S_LEN
            else:
                raise ValueError(f"Unknown pattern at {pos}: {s[pos:pos+10]}")
        else:
            raise ValueError(f"Unexpected char '{c}' at {pos}")
    return ''.join(result)


def expand_compact(compact):
    """Expand compact k/D/X/- to full l/- (for testing)"""
    parts = []
    for c in compact:
        if c == 'k':
            parts.append(S_STR)
        elif c == 'D':
            parts.append(I_STR)
        elif c == 'X':
            parts.append(K_STR)
        elif c == '-':
            parts.append('-')
    return ''.join(parts)


def test_small():
    """Test compression on small known examples"""
    print("Running small tests...")

    # Test 1: Single K
    assert compress_string(K_STR) == 'X'
    print("  K -> X  OK")

    # Test 2: Single I
    assert compress_string(I_STR) == 'D'
    print("  I -> D  OK")

    # Test 3: Single S
    assert compress_string(S_STR) == 'k'
    print("  S -> k  OK")

    # Test 4: (K I) = XD-
    ki = K_STR + I_STR + '-'
    assert compress_string(ki) == 'XD-'
    print("  (K I) -> XD-  OK")

    # Test 5: S(KK)I = kXX--D- (true)
    skki = S_STR + K_STR + K_STR + '-' + '-' + I_STR + '-'
    result = compress_string(skki)
    assert result == 'kXX--D-', f"got: {result}"
    print("  S(KK)I -> kXX--D-  OK")

    # Test 6: Round-trip l.2 answer
    target = 'kkD-XXD----XXD---'
    expanded = expand_compact(target)
    recompressed = compress_string(expanded)
    assert recompressed == target
    print(f"  Round-trip '{target}'  OK")

    # Test 7: Round-trip number 3 encoding
    target3 = 'kXX--kkD-XkXX--D----XkXX--kkD-XkXX--D----XkXX--kkD-XXD----XXD----------'
    expanded3 = expand_compact(target3)
    recompressed3 = compress_string(expanded3)
    assert recompressed3 == target3
    print(f"  Round-trip number 3  OK")

    # Test 8: Verify first bytes of stars.txt match S pattern
    stars_path = os.path.join(os.path.dirname(__file__), '..', 'very_large_txt', 'stars.txt')
    if os.path.exists(stars_path):
        with open(stars_path, 'r') as f:
            head = f.read(200)
        assert head[:S_LEN] == S_STR, f"stars.txt doesn't start with S"
        print(f"  stars.txt starts with S pattern  OK")

    print("All tests passed!\n")


if __name__ == '__main__':
    print("=== Step 1: Compress stars.txt ===\n")

    print("Validating patterns...")
    validate_patterns()
    print()

    test_small()

    input_path = os.path.join(os.path.dirname(__file__), '..', 'very_large_txt', 'stars.txt')
    output_path = os.path.join(os.path.dirname(__file__), '..', 'very_large_txt', 'stars_compact.txt')

    if not os.path.exists(input_path):
        print(f"Input file not found: {input_path}")
        sys.exit(1)

    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Size:   {os.path.getsize(input_path):,} bytes\n")

    compress(input_path, output_path)
