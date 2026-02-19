#!/usr/bin/env python3
"""Compare elem_0, elem_1, elem_2 to find their differences."""

e0 = open('d:/github/atgt2026hp_stars/extracted/data_items/elem_0.txt').read().strip()
e1 = open('d:/github/atgt2026hp_stars/extracted/data_items/elem_1.txt').read().strip()
e2 = open('d:/github/atgt2026hp_stars/extracted/data_items/elem_2.txt').read().strip()
print(f'elem_0: {len(e0)} chars')
print(f'elem_1: {len(e1)} chars')
print(f'elem_2: {len(e2)} chars')

# Find differences
min_len = min(len(e0), len(e1), len(e2))
diffs = []
for i in range(min_len):
    chars = set([e0[i], e1[i], e2[i]])
    if len(chars) > 1:
        diffs.append(i)

# Also check tail differences
max_len = max(len(e0), len(e1), len(e2))
print(f'Min length: {min_len}, Max length: {max_len}')
print(f'Total differing positions (within min length): {len(diffs)}')
if diffs:
    print(f'Diff positions: {diffs}')
    for d in diffs:
        ctx_s = max(0, d - 10)
        ctx_e = min(min_len, d + 20)
        print(f'  pos {d}:')
        print(f'    e0: {e0[ctx_s:ctx_e]}')
        print(f'    e1: {e1[ctx_s:ctx_e]}')
        print(f'    e2: {e2[ctx_s:ctx_e]}')

# Check if the tails differ
if len(e0) != len(e1) or len(e0) != len(e2):
    print(f'\nTail differences:')
    print(f'  e0 end: {e0[-30:]}')
    print(f'  e1 end: {e1[-30:]}')
    print(f'  e2 end: {e2[-30:]}')
