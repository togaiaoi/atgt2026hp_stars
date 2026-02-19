#!/usr/bin/env python3
"""Trace NW field variable usage."""

import re

data = open('d:/github/atgt2026hp_stars/extracted/left_x_lambda.txt', 'r').read()
nw = data[13972:21567]

# Find definitions and uses of key variables
for v in ['x2928', 'x2929', 'x2930', 'x2934', 'x2935', 'x2939', 'x2940',
          'x2954', 'x2955', 'x2999']:
    uses = []
    defs = []
    for m in re.finditer(v, nw):
        pos = m.start()
        # Check if definition (preceded by backslash)
        if pos > 0 and nw[pos-1] == '\\':
            defs.append(pos)
        else:
            uses.append(pos)
    if defs or uses:
        print(f"{v}: defined at {defs}, used at {uses}")

print()

# Now look at how the NW body uses its params
# NW body starts at the first arg after Y-preamble
# Body = (\x2928. \x2929. \x2930. x2929 (\x2934. \x2935. ...
print("NW body structure (first 300 chars after Y-preamble):")
body_start = nw.index("x2914 (x2922 x2922))") + len("x2914 (x2922 x2922))")
print(nw[body_start:body_start+300])
print()

# The body takes \x2928.\x2929.\x2930
# Then: x2929 (\x2934.\x2935. x2930 (\x2939.\x2940. ...))
# x2929 is applied to a lambda -> x2929 is a pair2 (cons case handler)
# The lambda is \x2934.\x2935. x2930 (\x2939.\x2940. ...)
# Inside, x2930 is applied to another lambda
# So x2930 is also a pair2

# This means:
# NW body = \rec.\list1.\list2.
#   list1(\item1.\rest1.
#     list2(\item2.\rest2.
#       PROCESS(item1, rest1, item2, rest2, rec)))

# So NW processes TWO lists in parallel!

# NW ending:
# ... x2934 x2939) (x2928 x2935 x2940)) (\x4397.\x4398. x4398)) (x2930 sel3 K)
#
# x2934 = item from list1, x2939 = item from list2
# x2935 = rest of list1, x2940 = rest of list2
# x2928 x2935 x2940 = rec(rest1)(rest2) = recursive call on tails
#
# The result is some computation on item1(x2934) and item2(x2939),
# combined with the recursive result on the tails (x2928 x2935 x2940)
# and then applied to false and (x2930 sel3 K)
#
# Wait - x2930 sel3 K at the very end
# x2930 = list2 (the second list parameter)
# sel3 = \a.\b.\c.\d.\e. e = 5th-element selector? No, sel3 selects 4th
# Actually (\x4403.\x4404.\x4405.\x4406. x4406) takes 4 args, returns 4th
# And K = (\x4408.\x4409. x4408) = true

# So x2930(sel3)(K) = list2(4-arg-selector)(true)
# If list2 is pair2: pair2(sel3)(K) = sel3(fst)(snd) = ... 4th element
# But sel3 takes 4 args and we only give it 2. Hmm.

# Actually for pair2: pair2(f)(g) = f(A)(B)
# So list2(sel3)(K) = sel3(A)(B) where A,B are pair2's data
# sel3 takes 4 args. sel3(A)(B) still needs 2 more args.
# This would be partially applied.

# OR: x2930 = list2 is not a pair2 at all - it could be a different structure
# Let me reconsider. x2930 is the THIRD parameter of the Y-comb body

# Y(\rec.\list1.\list2. body) means:
# The initial call provides list1 and list2
# But NW is just: Y(\body) with no initial args shown
# So NW = Y(body) = body applied to itself
# The result is a function expecting list1 and list2

print("NW ending context (last 400 chars):")
print(nw[-400:])
