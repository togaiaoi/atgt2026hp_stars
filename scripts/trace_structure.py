#!/usr/bin/env python3
"""Trace the detailed structure of how the three sections chain together.

Key observations:
- Section 1 ends with: ...x4534) x4496 x4495) x4489
- Section 2 starts right after with: (\x5928.\x5929. ...)
- Section 2 ends with: ...x5929) x4490
- Then: (\x7319.\x7320. x7319 K (x7327...FALSE FALSE)))) (x4488 ...)
- Section 3 starts after x4488 call

The outer body is: \x4488.\x4489.\x4490.\x4495.\x4496. SECTION1_END x4489 SECTION2_END x4490 SECTION3_CONSTRUCTION
And then the final return: ... x4489 x4490) x4490) x4489))

Let me understand this as:
- x4488 = outer rec (Y-fixed recursive self)
- x4489 = arg_a
- x4490 = arg_b
- x4495 = arg_c
- x4496 = arg_d

Section 1 computes something and applies it to x4489
Section 2 computes something and applies it to x4490
Section 3 uses x4488(rec) and returns a result involving x4489 and x4490
"""

import re

data = open('d:/github/atgt2026hp_stars/extracted/left_x_lambda.txt', 'r').read()
se = data[21866:45037]

# Let me trace the exact structure at each junction point
# by tracking paren depth and identifying the application structure

# At junction 1 (around offset ~7356):
# "...x4534) x4496 x4495) x4489 (\x5928..."
# The ') x4496' closes a paren, then x4496 is the next arg
# Let me parse this more carefully

# Let me look at exactly what the body does
# After params \x4488.\x4489.\x4490.\x4495.\x4496:
# There are more params: \x4500.\x4502
# Then Y-comb: x4500(x4502 x4502)(\x4508. x4500(x4508 x4508))
# This creates a recursive function, applied to body and init

# The whole structure of the OUTER_BODY is:
# \x4488.\x4489.\x4490.\x4495.\x4496.
#   Y2(\x4500)(...body1...) (...init1...)
#   SECTION1_RESULT
#   ...
#   applied to various args including x4489, x4490, etc.

# Let me count exact paren structure
# I'll create an annotated view showing depth changes

print("=== Detailed parenthesis structure of SE body ===")
print("(showing depth at key variable references)")
print()

# Find all references to outer body variables
outer_vars = ['x4488', 'x4489', 'x4490', 'x4495', 'x4496', 'x4468', 'x4470']
depth = 0
for i, c in enumerate(se):
    if c == '(':
        depth += 1
    elif c == ')':
        depth -= 1

    # Check if this position starts any outer variable
    for v in outer_vars:
        if se[i:i+len(v)] == v:
            # Make sure it's a word boundary
            before = se[i-1] if i > 0 else ' '
            after_c = se[i+len(v)] if i+len(v) < len(se) else ' '
            if (before in ' (.\\') and (after_c in ' ).'):
                ctx = se[max(0,i-30):i+len(v)+30]
                print(f"  depth={depth:2d} at offset {i:5d}: {v} | ...{ctx}...")

# Also look for the result pattern
# result(false)(false)(false)(prev_section)(1)
# In the lambda text: something(KI)(KI)(KI)(prev)(K(pair(true, pair(false, nil))))

# Let me look at what x4534 is (it appears at junction 1)
print("\n\n=== What is x4534? ===")
for m in re.finditer(r'\\x4534\.', se):
    pos = m.start()
    ctx = se[max(0,pos-20):pos+100]
    print(f"  Defined at offset {pos}: ...{ctx}...")

# And x5929 (it appears at junction 2)
print("\n=== What is x5929? ===")
for m in re.finditer(r'\\x5929\.', se):
    pos = m.start()
    ctx = se[max(0,pos-20):pos+100]
    print(f"  Defined at offset {pos}: ...{ctx}...")

# And x7344 (appears at final return)
print("\n=== What is x7344? ===")
for m in re.finditer(r'\\x7344\.', se):
    pos = m.start()
    ctx = se[max(0,pos-20):pos+100]
    print(f"  Defined at offset {pos}: ...{ctx}...")
