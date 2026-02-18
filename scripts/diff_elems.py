#!/usr/bin/env python3
"""Diff elem[0], elem[1], elem[2] decompilations to find differences."""
import re
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def tokenize(s):
    """Split decompiled lambda into tokens."""
    tokens = []
    i = 0
    while i < len(s):
        c = s[i]
        if c in '()':
            tokens.append(c)
            i += 1
        elif c == 'λ':
            # Lambda: λx0. or λ_0.
            j = i + 1
            while j < len(s) and s[j] != '.':
                j += 1
            tokens.append(s[i:j+1])
            i = j + 1
        elif c.isalpha() or c == '_' or c == '«':
            j = i
            while j < len(s) and (s[j].isalnum() or s[j] in '_«»'):
                j += 1
            tokens.append(s[i:j])
            i = j
        elif c == ' ':
            i += 1
        elif c == '.':
            tokens.append('.')
            i += 1
        else:
            tokens.append(c)
            i += 1
    return tokens

def find_cons_data(s):
    """Find the big cons(...) data block and return its position."""
    # Find the start of the repeated cons chain
    pattern = "cons(cons(true, cons(true, cons(false, false))), cons(cons(true, cons(false, cons(true"
    pos = s.find(pattern)
    return pos

def split_at_data(s):
    """Split the decompiled text at the cons data block."""
    pattern = "cons(cons(true, cons(true, cons(false, false))), cons(cons(true, cons(false, cons(true, cons(false, false)))), cons(cons(true, cons(false, cons(false, cons(false, cons(true, cons(false, false)))))), cons(cons(false, false), cons(cons(true, cons(false, cons(true, cons(false, false)))), cons(cons(true, cons(true, cons(false, false))), cons(cons(true, cons(false, cons(true, cons(false, false)))), cons(cons(true, cons(false, cons(false, cons(false, cons(true, cons(false, false)))))), cons(cons(false, false), cons(cons(true, cons(false, cons(true, cons(false, false)))), cons(cons(true, cons(true, cons(false, false))), cons(cons(true, cons(false, cons(true, cons(false, false)))), cons(cons(true, cons(false, cons(false, cons(false, cons(true, cons(false, false)))))), cons(cons(false, false), cons(cons(true, cons(false, cons(true, cons(false, false)))), cons(cons(true, cons(true, cons(false, false))), cons(cons(true, cons(false, cons(true, cons(false, false)))), cons(cons(true, cons(false, cons(false, cons(false, cons(true, cons(false, false)))))), cons(cons(false, false), cons(cons(true, cons(false, cons(true, cons(false, false)))), cons(cons(true, cons(true, cons(false, false))), cons(cons(true, cons(false, cons(true, cons(false, false)))), cons(cons(true, cons(false, cons(false, cons(false, cons(true, cons(false, false)))))), cons(cons(false, false), cons(cons(true, cons(false, cons(true, cons(false, false)))), false))))))))))))))))))))))))))"

    pos = s.find(pattern)
    if pos == -1:
        return s, "", ""
    before = s[:pos]
    after = s[pos + len(pattern):]
    return before, "«KEY_DATA»", after

# Read files
files = [
    r"d:\github\atgt2026hp_stars\images\item09_elem0_decompile.txt",
    r"d:\github\atgt2026hp_stars\images\item09_elem1_decompile.txt",
    r"d:\github\atgt2026hp_stars\images\item09_elem2_decompile.txt",
]

texts = []
for f in files:
    with open(f, 'r', encoding='utf-8') as fh:
        texts.append(fh.read().strip())

# Split at the key data block
print("=" * 80)
print("ANALYSIS: Splitting each element at the shared cons data block")
print("=" * 80)

parts = []
for i, t in enumerate(texts):
    before, data, after = split_at_data(t)
    parts.append((before, data, after))
    print(f"\n--- elem[{i}] ---")
    print(f"Before data: {len(before)} chars")
    print(f"After data:  {len(after)} chars")
    print(f"Before (last 200): ...{before[-200:]}")
    print(f"After  (first 200): {after[:200]}...")

# Now compare "after" parts more carefully
print("\n" + "=" * 80)
print("COMPARING 'AFTER' SECTIONS (after the key data block)")
print("=" * 80)

for i in range(3):
    print(f"\n--- elem[{i}] after ---")
    print(f"Length: {len(parts[i][2])}")
    print(f"Content: {parts[i][2]}")

# Compare "before" parts
print("\n" + "=" * 80)
print("COMPARING 'BEFORE' SECTIONS (before the key data block)")
print("=" * 80)

for i in range(3):
    print(f"\n--- elem[{i}] before ---")
    print(f"Length: {len(parts[i][0])}")
    # Show first 500 chars
    print(f"First 500: {parts[i][0][:500]}")
    print(f"Last 500:  {parts[i][0][-500:]}")

# Tokenize and find differences
print("\n" + "=" * 80)
print("TOKEN-LEVEL DIFF: Comparing elem[0] vs elem[1] vs elem[2]")
print("=" * 80)

# Replace the common data block with a placeholder
normalized = []
for i, t in enumerate(texts):
    before, data, after = split_at_data(t)
    normalized.append(before + "«KEY_DATA»" + after)

toks = [tokenize(n) for n in normalized]
print(f"\nToken counts: elem[0]={len(toks[0])}, elem[1]={len(toks[1])}, elem[2]={len(toks[2])}")

# Find first difference
max_len = max(len(t) for t in toks)
for pos in range(max_len):
    vals = []
    for i in range(3):
        if pos < len(toks[i]):
            vals.append(toks[i][pos])
        else:
            vals.append("<END>")
    if len(set(vals)) > 1:
        # Show context around the difference
        start = max(0, pos - 5)
        end = min(max_len, pos + 10)
        print(f"\nFirst difference at token position {pos}:")
        for i in range(3):
            context = toks[i][start:min(end, len(toks[i]))]
            print(f"  elem[{i}]: ...{' '.join(context)}...")
        break

# Find ALL differences
print("\n--- All differing token positions ---")
diffs = []
for pos in range(max_len):
    vals = []
    for i in range(3):
        if pos < len(toks[i]):
            vals.append(toks[i][pos])
        else:
            vals.append("<END>")
    if len(set(vals)) > 1:
        diffs.append((pos, vals))

print(f"Total differing positions: {len(diffs)}")
for pos, vals in diffs[:50]:  # Show first 50
    context_before = ""
    for i in range(3):
        if pos > 0 and pos - 1 < len(toks[i]):
            context_before = toks[i][max(0,pos-3):pos]
    print(f"  pos {pos}: [{vals[0]}] [{vals[1]}] [{vals[2]}]  context: {' '.join(str(x) for x in context_before)}")
