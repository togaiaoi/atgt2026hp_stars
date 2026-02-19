#!/usr/bin/env python3
"""Analyze how the Y-combinator sections chain together in the SE field.

Key insight from previous analysis:
- The outermost Y-comb's BODY takes params: x4488, x4489, x4490, x4495, x4496
  x4488 = outer recursive self-reference
  x4489, x4490, x4495, x4496 = data/state arguments
- Inside the body, there are more Y-combinators (nested)
- The references to x4489 are at offsets 7356 and 21594/21614
- The references to x4490 are at offsets 14411 and 21600/21607
- x4488 (outer recursion) is referenced at offset 14539

The end of the body shows: ... x7344) x4489 x4490) x4490) x4489))
This means the body RETURNS something that uses x4489 and x4490.
"""

import re

data = open('d:/github/atgt2026hp_stars/extracted/left_x_lambda.txt', 'r').read()
se = data[21866:45037]

# The SE field is: \x4468. \x4470. x4470 (BIG) (STEP)
# BIG = Y(\outer_body)
# outer_body = \x4488.\x4489.\x4490.\x4495.\x4496. INNER
# INNER starts with ANOTHER Y-comb: \x4500.\x4502. x4500(x4502 x4502)...

# Let me trace the structure by finding the section boundaries
#
# Looking at the reference pattern:
# x4496 referenced at 7343 (near x4489's ref at 7356)
# x4495 referenced at 7349
# x4489 referenced at 7356
# This cluster at ~7350 is where SECTION_1 ends and feeds into SECTION_2
#
# x4490 referenced at 14411 - where SECTION_2 ends
# x4488 referenced at 14539 - outer recursion call!
#
# Then x4489 x4490 at 21594-21614 is the final return

# Let me look at the key junction points

def show_context(text, pos, before=100, after=100):
    s = max(0, pos - before)
    e = min(len(text), pos + after)
    return text[s:e]

print("=== JUNCTION 1: around offset 7340-7400 (x4496, x4495, x4489) ===")
print(show_context(se, 7350, 200, 200))
print()

print("=== JUNCTION 2: around offset 14400-14550 (x4490, x4488) ===")
print(show_context(se, 14470, 200, 200))
print()

print("=== FINAL RETURN: around offset 21590-21620 ===")
print(show_context(se, 21600, 200, 100))
print()

# Let me also trace what happens between sections
# Between junction 1 (~7350) and junction 2 (~14410)
# There should be SECTION_2 content

# And between junction 2 (~14540) and the final return (~21594)
# There should be SECTION_3 content

# Let me identify the section boundaries more precisely
# Section 1: from ~130 (after outer body params) to ~7360
# Section 2: from ~7360 to ~14540
# Section 3: from ~14540 to ~21594

sections = [
    ("Section 1", 130, 7360),
    ("Section 2", 7360, 14540),
    ("Section 3", 14540, 21600),
]

for name, s, e in sections:
    text = se[s:e]
    y_count = len(re.findall(r'x\d+ \(x\d+ x\d+\)', text))
    print(f"{name}: offset {s}-{e}, size {e-s}, ~Y-combs: {y_count}")
    print(f"  Start: {repr(text[:120])}")
    print(f"  End:   {repr(text[-120:])}")
    print()

# Now let me understand the connection pattern at junction 1 more precisely
# At offset ~7350: ... x4534) x4496 x4495) x4489 ...
# This means something is applied to x4534, then to x4496, x4495, then to x4489
# Wait - it's nested application: (((... x4534) x4496 x4495) x4489 ...)
# Or maybe: RESULT x4534) is closed, then x4496 x4495) closes something else

# Let me track parenthesis depth more carefully around junction 1
junction = 7350
depth = 0
for i in range(junction):
    if se[i] == '(':
        depth += 1
    elif se[i] == ')':
        depth -= 1

print(f"Paren depth at junction 1 (offset {junction}): {depth}")

junction = 14470
depth = 0
for i in range(junction):
    if se[i] == '(':
        depth += 1
    elif se[i] == ')':
        depth -= 1

print(f"Paren depth at junction 2 (offset {junction}): {depth}")

junction = 21594
depth = 0
for i in range(junction):
    if se[i] == '(':
        depth += 1
    elif se[i] == ')':
        depth -= 1

print(f"Paren depth at final return (offset {junction}): {depth}")
