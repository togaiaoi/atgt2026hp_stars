#!/usr/bin/env python3
"""Analyze the Y-combinator sections and their nesting in SE field.
The goal is to understand the section chain: how each section feeds into the next."""

import re

data = open('d:/github/atgt2026hp_stars/extracted/left_x_lambda.txt', 'r').read()

# Field boundaries
FIELDS = {
    'COND': (8, 13970),
    'NW': (13971, 21568),
    'NE': (21569, 21764),
    'SW': (21765, 21864),
    'SE': (21865, 45038),
}

# Let's now carefully parse the structure
# The SE field is: \x4468. \x4470. x4470 (BIG) (STEP)
# BIG_SECTION starts with outermost Y-comb:
#   \x4474. \x4476. x4474 (x4476 x4476) (\x4482. x4474 (x4482 x4482))
#   This is Y-combinator: Y(x4474) using x4476 as omega
#   Result: x4474 = the recursive function, x4476 = thunk for self-application

# The BODY of the outermost Y-comb is:
#   \x4488. \x4489. \x4490. \x4495. \x4496.  ...
# x4488 = the outermost recursive function (self-reference via Y)
# x4489, x4490, x4495, x4496 = additional arguments

# Then inside, there's ANOTHER Y-comb: x4500 (x4502 x4502)...
# This is applied to (\x4514...) as body and (\x4528...) as init

# Let me trace the structure differently: look at what variables from the
# outer Y-comb body are used later

se_inner = data[21866:45037]

# Let me identify sections by finding the CHAIN PATTERN:
# Each section ends with: result(false)(false)(false)(prev_section_data)(1)
# In the file, this would appear as applications of false/true/numbers
# followed by references to x4489/x4490/x4488 (the outer body's params)

# Let me search for references to x4489 and x4490 (the "data" arguments)
print("=== References to outer body params ===")
for var in ['x4488', 'x4489', 'x4490', 'x4495', 'x4496']:
    refs = [(m.start(), m.end()) for m in re.finditer(r'\b' + var + r'\b', se_inner)]
    print(f"  {var}: {len(refs)} references at offsets {[r[0] for r in refs]}")

print()

# Let me trace the chain of sections by looking at how variables flow
# I'll look at the section breaks

# Each section appears to use a Y-combinator and produce a result
# that's fed into the next section

# Let's identify sections by the depth-1 Y-combinators (outermost level)
# From the depth analysis:
# Y#0 (x4474) at depth 1 - the outermost Y-comb
# Y#1 (x4500) at depth 2 - nested inside the body
# Y#2 (x4541) at depth 3 - further nested
# etc.

# But the sections chain via the pattern at the end
# Let me look at specific text around the x4489/x4490 references

print("=== Context around x4489/x4490 references ===")
for var in ['x4489', 'x4490']:
    for m in re.finditer(r'\b' + var + r'\b', se_inner):
        pos = m.start()
        # Skip the definition (lambda binding)
        if pos < 120:
            continue
        ctx = se_inner[max(0,pos-60):min(len(se_inner),pos+60)]
        print(f"  {var} at {pos}: ...{ctx}...")
        print()

# Now let's look at the overall nesting structure differently
# The SE field's BIG_SECTION can be viewed as:
# Y(outer_rec)(SECTION_A)(SECTION_B)
# where the outer Y-comb's body takes: rec, section_a_result, section_b_result, arg3, arg4
# and builds the final quadtree

# IMPORTANT: Let me look at the SE field from a higher level
# by examining what x4470 receives
# SE = \data.\selector. selector(MAIN_QUAD)(STEP_FUNC)
# STEP_FUNC = \step. step(step(step(data, 2048), 256), 32)

# MAIN_QUAD is the Y-comb construction
# Let me understand what it produces

# Looking at the BIG_SECTION again more carefully:
# The outer Y-comb is applied to a BODY and then to INIT_ARGS
# Y(\x4488. \x4489. \x4490. \x4495. \x4496. INNER_BODY) INIT1 INIT2 INIT3 INIT4

# Let me find the INIT_ARGS
# After the body of the outer Y-comb, we should find the initial arguments

# The structure is:
# \x4474. \x4476. x4474(x4476 x4476)(\x4482. x4474(x4482 x4482))(BODY)(INIT1)(INIT2)(INIT3)(INIT4)

# Let me find the BODY and INIT args by parsing from the start
big_start = se_inner.index('(') + 1
# Skip: \x4474. \x4476. x4474 (x4476 x4476) (\x4482. x4474 (x4482 x4482))
# Then comes the body and init args

# Let me find the end of the Y-comb preamble
# After the Y-preamble (\x4482. x4474 (x4482 x4482))
preamble_end_marker = "x4474 (x4482 x4482))"
idx = se_inner.index(preamble_end_marker) + len(preamble_end_marker)
remaining = se_inner[idx:]
print(f"\n=== After outermost Y-comb preamble (offset {idx}) ===")
print(f"Next 200 chars: {repr(remaining[:200])}")

# Now parse the arguments to the Y-comb
# First comes the BODY (a parenthesized expression), then INIT args
pos = 0
while pos < len(remaining) and remaining[pos] == ' ':
    pos += 1

args = []
while pos < len(remaining):
    c = remaining[pos]
    if c == '(':
        depth = 1
        start = pos
        pos += 1
        while pos < len(remaining) and depth > 0:
            if remaining[pos] == '(':
                depth += 1
            elif remaining[pos] == ')':
                depth -= 1
            pos += 1
        args.append((start, pos))
    elif c == ')':
        break
    elif c == ' ':
        pos += 1
    elif c.isalpha() or c == 'x':
        # bare variable
        start = pos
        while pos < len(remaining) and remaining[pos] not in ' ()':
            pos += 1
        args.append((start, pos))
    else:
        pos += 1

print(f"\nY-comb applied to {len(args)} arguments:")
for i, (s, e) in enumerate(args):
    size = e - s
    text = remaining[s:e]
    label = "BODY" if i == 0 else f"INIT{i}"
    print(f"  {label}: size={size}")
    print(f"    Start: {repr(text[:120])}")
    if size > 120:
        print(f"    End:   {repr(text[-120:])}...")
    print()

# The second INIT arg should be the step function
# which is (\x8737. ...)
