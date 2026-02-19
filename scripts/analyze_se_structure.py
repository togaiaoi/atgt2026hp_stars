#!/usr/bin/env python3
"""Analyze the SE field structure of left_x_lambda.txt."""

import re

data = open('d:/github/atgt2026hp_stars/extracted/left_x_lambda.txt', 'r').read()
se = data[21866:45037]

print(f"SE field size: {len(se)}")
print(f"SE starts: {repr(se[:80])}")
print()

# SE = \x4468. \x4470. x4470 (BIG_SECTION) (STEP_FUNC)
# x4468 = data parameter
# x4470 = selector/handler
# BIG_SECTION = big Y-combinator recursive construction
# STEP_FUNC = step(step(step(data, 2048), 256), 32)

# The structure is: SE takes (data, selector) and returns selector(Y_RESULT)(STEP_RESULT)
# where Y_RESULT is the recursive computation
# and STEP_RESULT is the step function applied to data

# Now let's understand the BIG_SECTION's internal Y-combinator structure
# It has 23 Y-combinators total

# The Y-combinator positions within SE:
y_positions = []
pattern = r'x(\d+) \(x(\d+) x\2\)'
for m in re.finditer(pattern, se):
    # Verify this is a Y-comb by checking for the (\xP. xN (xP xP)) part
    after = se[m.end():]
    var_f = 'x' + m.group(1)
    expected_prefix = f" (\\x"
    if after.startswith(expected_prefix):
        # Find the var
        rest = after[len(expected_prefix):]
        dot_pos = rest.index('.')
        var_w = 'x' + rest[:dot_pos]
        check = f". {var_f} ({var_w} {var_w}))"
        if rest[dot_pos:dot_pos+len(check)] == check:
            y_positions.append(m.start())

print(f"Y-combinator positions in SE: {len(y_positions)}")
for i, pos in enumerate(y_positions):
    ctx = se[max(0,pos-30):pos+60]
    print(f"  Y#{i}: offset {pos}")
    # print(f"    ctx: {repr(ctx)}")

# Now let's understand the nesting by looking at the depth of each Y-comb
# We need to track parenthesis depth at each Y-comb position
depths = []
depth = 0
y_idx = 0
for i, c in enumerate(se):
    if c == '(':
        depth += 1
    elif c == ')':
        depth -= 1
    if y_idx < len(y_positions) and i == y_positions[y_idx]:
        depths.append(depth)
        y_idx += 1

print("\nY-combinator depths:")
for i, (pos, d) in enumerate(zip(y_positions, depths)):
    var_match = re.match(r'x(\d+)', se[pos:])
    var = var_match.group(0) if var_match else '?'
    print(f"  Y#{i}: depth={d}, pos={pos}, var={var}")

# Now let's identify the SECTIONS
# Look for the pattern that ends each section: result(false)(false)(false)(prev)(1)
# In the lambda notation, false = (\xN. \xM. xM)
# Let's find sequences of false applications

# Actually, let me look for the chaining pattern
# Each section seems to end with something applied to the previous section
# and followed by constants like false/true/numbers

# Let me look at the end of BIG_SECTION for the section termination pattern
# The big section ends with: ... x4489 x4490) x4490) x4489))
print("\nBIG_SECTION ending pattern:")
# Find where the big section's body references x4489, x4490
for m in re.finditer(r'x4489|x4490|x4488', se):
    pos = m.start()
    ctx = se[max(0,pos-20):pos+20]
    print(f"  {m.group()} at {pos}: ...{ctx}...")

# Let me now look at the 'false' and 'true' patterns that appear between sections
# These would be the result(false)(false)(false)(prev)(1) pattern
print("\n\nLooking for the chaining constants between Y-comb sections...")
# Find all standalone false patterns: (\xN. \xM. xM) at various depths
# and true patterns: (\xN. \xM. xN)

# Let me look at what happens around x4489 and x4490 (the BIG_SECTION body params)
# x4488 = the outer Y-comb recursive function (not used?)
# x4489 = ?
# x4490 = ?

# Actually, let me re-examine the structure
# BIG_SECTION = Y-comb(BODY)(INIT_ARGS...)
# where BODY = \rec.\arg1.\arg2.\arg3.\arg4. ...
print("\n\nFirst 500 chars of BIG_SECTION body (after Y-comb):")
# Find the body which starts with \x4488.
body_idx = se.index('\\x4488.')
print(se[body_idx:body_idx+500])

# Count the params
params = []
pos = body_idx
while se[pos] == '\\':
    dot = se.index('.', pos)
    var = se[pos+1:dot]
    params.append(var)
    pos = dot + 1
    while pos < len(se) and se[pos] == ' ':
        pos += 1
print(f"\nBody params: {params}")
print(f"Body starts at: {repr(se[pos:pos+100])}")
