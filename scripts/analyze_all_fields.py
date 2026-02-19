#!/usr/bin/env python3
"""Analyze all 5 fields of the left_x diamond 5-tuple.

Focus: understand COND and NW in detail, as they mirror SE's structure.
COND has 16 Y-combinators and NW has 9, so they have similar section structures.
"""

import re

data = open('d:/github/atgt2026hp_stars/extracted/left_x_lambda.txt', 'r').read()

# Field boundaries (including outer parens)
FIELD_BOUNDS = [
    ('COND', 8, 13970),
    ('NW', 13971, 21568),
    ('NE', 21569, 21764),
    ('SW', 21765, 21864),
    ('SE', 21865, 45038),
]

# Strip outer parens for each field
def get_field(name):
    for n, s, e in FIELD_BOUNDS:
        if n == name:
            return data[s+1:e-1]  # strip outer parens
    return None

# Analyze COND field
cond = get_field('COND')
print("=" * 70)
print("=== COND FIELD (13960 chars, 16 Y-combinators) ===")
print(f"Starts: {repr(cond[:120])}")
print(f"Ends:   {repr(cond[-120:])}")
print()

# COND starts with: \x8. \x9. \x13. \x15. ...
# So COND takes 4 arguments: x8, x9, x13, x15
# Then x13(x15 x15)(\x21. x13(x21 x21)) is Y-combinator
# So: \x8.\x9. Y(\x13)(BODY)(ARGS...)

# Find params of COND
params = []
pos = 0
while cond[pos] == '\\':
    dot = cond.index('.', pos)
    var = cond[pos+1:dot]
    params.append(var)
    pos = dot + 1
    while pos < len(cond) and cond[pos] == ' ':
        pos += 1

print(f"COND params: {params}")
# After params, should see Y-combinator or body
print(f"After params: {repr(cond[pos:pos+100])}")
print()

# Find all references to x8 and x9 in COND
for v in ['x8', 'x9']:
    refs = []
    for m in re.finditer(r'\b' + v + r'\b', cond):
        # Skip definition
        if m.start() < 20:
            continue
        refs.append(m.start())
    print(f"  {v}: {len(refs)} references at {refs}")
print()

# The structure of COND:
# \x8.\x9. x13(x15 x15)(\x21. x13(x21 x21))(BODY)(ARGS)
# Here x13 is the Y-combinator's recursive function parameter
# x15 is the Z-combinator thunk

# Wait - COND has 4 params: x8, x9, x13, x15
# Then the body starts with: x13 (x15 x15) (\x21. x13 (x21 x21))
# This is Y-combinator structure applied to x13!
# So x13 IS the function being fixed-pointed
# And x15 is the self-application thunk

# But x13 and x15 are params to COND!
# That means COND = \x8.\x9.\f.\z. f(z z)(\w. f(w w)) BODY ARGS
# So COND expects to receive the Y-combinator function and thunk as args 3 and 4

# Hmm, that's unusual. Let me re-read.
# Actually: COND = \x8.\x9. Y_RESULT
# where Y_RESULT uses x8 and x9
# And x13, x15 are part of the Y-combinator preamble
# Let me check: is x13 (x15 x15) using x13 and x15 as the Y-comb params?

# \x8. \x9. \x13. \x15. x13 (x15 x15) (\x21. x13 (x21 x21)) (...)
# Yes! \x13.\x15 are the Y-comb's f and z params
# So Y(f) where f = x13: x13(x15 x15)(\x21. x13(x21 x21)) gives the fixed point

# But wait, COND receives x8 and x9 as external params
# Then the Y-combinator is evaluated inside using x13 and x15

# Actually, \x8.\x9 are the COND's own params (data and key?)
# Then \x13.\x15 start the Y-combinator
# So COND = \data.\key. Y(\rec_body) INIT_ARGS...

# The Y-comb body would be: what comes after the Y-preamble
# Let me find it
preamble_end = "x13 (x21 x21))"
idx = cond.index(preamble_end) + len(preamble_end)
print(f"After Y-preamble in COND: {repr(cond[idx:idx+200])}")
print()

# Now analyze NW
nw = get_field('NW')
print("=" * 70)
print("=== NW FIELD (7595 chars, 9 Y-combinators) ===")
print(f"Starts: {repr(nw[:120])}")
print(f"Ends:   {repr(nw[-120:])}")
print()

# NW starts with: \x2914. \x2916. x2914 (x2916 x2916) (\x2922. x2914 (x2922 x2922))
# So NW = \f.\z. f(z z)(\w. f(w w)) BODY ARGS
# NW takes 2 params that are the Y-combinator's f and z
# This means NW IS a Y-combinator expression (it IS the recursive function)

# Find the body after Y-preamble
preamble_end_nw = "x2914 (x2922 x2922))"
idx_nw = nw.index(preamble_end_nw) + len(preamble_end_nw)
print(f"After Y-preamble in NW: {repr(nw[idx_nw:idx_nw+200])}")
print()

# NW body params
body_start = nw[idx_nw:].lstrip()
params_nw = []
pos = 0
while body_start[pos] == '(':
    pos += 1  # skip opening paren
while body_start[pos] == '\\':
    dot = body_start.index('.', pos)
    var = body_start[pos+1:dot]
    params_nw.append(var)
    pos = dot + 1
    while pos < len(body_start) and body_start[pos] == ' ':
        pos += 1
print(f"NW body params: {params_nw}")
print()

# Now let me look at how COND and SE relate
# Key insight: COND is the pixel color at the current level
# NW is the northwest quadrant subtree
# SE is the southeast quadrant subtree (biggest)
# NE is a list-reversal/accumulation function
# SW is a NOT-reflection

# The 5-tuple left_x = \f. f(COND)(NW)(NE)(SW)(SE)
# This IS the quadtree diamond
# When evaluated as a quadtree:
# COND(data)(key) = the pixel color at this resolution
# NW(args) = the NW sub-quadtree
# etc.

# But SE = \data.\selector. selector(BIG_RESULT)(STEP)
# So SE takes data and a selector
# STEP = \step. step(step(step(data, 2048), 256), 32)
# BIG_RESULT is the recursive Y-comb construction

# The "selector" in SE means: when you apply SE to data,
# you get \selector. selector(quadtree)(step)
# Then when you apply the result to a specific selector,
# you get either the quadtree or the step function

# IMPORTANT REALIZATION:
# The diamond 5-tuple is NOT a simple quadtree!
# It's a FUNCTION that takes some arguments (data? key?)
# and PRODUCES a quadtree

# The COND field takes \x8.\x9 = (data, key) as params
# The NW field is a Y-combinator (takes no external params, it IS the recursive func)
# The SE field takes \x4468.\x4470 = (data, selector) as params

# So the image diamond is actually:
# \f. f(COND)(NW)(NE)(SW)(SE)
# where each field is a function that takes arguments from the evaluation context

# When the I/O interpreter asks for the image, it provides the image data node
# to left_x, and left_x's fields process it

print("=" * 70)
print("=== SUMMARY ===")
print("left_x = lambda f. f(COND)(NW)(NE)(SW)(SE)")
print()
print("COND = \\data.\\key. Y(rec_body)(init_args)  -- takes data+key, returns bool")
print("  -> pixel color computation")
print("  -> 16 Y-combinators (deep recursive structure)")
print()
print("NW = Y(rec_body)(args)  -- IS a recursive computation")
print("  -> NW quadrant subtree generator")
print("  -> 9 Y-combinators")
print()
print("NE = Y(fold_body)(nil)  -- list reversal/accumulation")
print("  -> only 1 Y-combinator")
print()
print("SW = \\self.\\x. self(NOT(x))(x)  -- NOT-reflection")
print("  -> coordinate reflection (flips axis)")
print()
print("SE = \\data.\\selector. selector(Y_RESULT)(STEP)")
print("  -> SE quadrant + step function for depth levels")
print("  -> 23 Y-combinators (the main computation)")
print("  -> STEP = step(step(step(data, 2048), 256), 32)")
