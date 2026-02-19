#!/usr/bin/env python3
"""
Now I can reconstruct the complete flow of the SE field.

SE = \data.\selector. selector (Y1_RESULT) (STEP_FUNC)
where:
  STEP_FUNC = \step. step(step(step(data, 2048), 256), 32)

Y1_RESULT = Y(\outer_rec.\arg_a.\arg_b.\arg_c.\arg_d.
    let sec1_result = SECTION_1_BODY(arg_d)(arg_c) in  -- applied to x4534, then x4496, x4495
    sec1_result arg_a                                    -- applied to x4489
    (\sec2_var1.\sec2_var2.                              -- x5928, x5929
        let sec2_result = SECTION_2_BODY(sec2_var2) in
        sec2_result arg_b                                -- applied to x4490
        (CONNECTOR                                       -- some constant/bool structure
         (outer_rec                                      -- x4488 recursive call!
            (\sec3_var1.\sec3_var2.                       -- x7343, x7344
                let sec3_result = SECTION_3_BODY(sec3_var2) in
                sec3_result arg_a arg_b                   -- applied to x4489, x4490 at depth 4
            ) arg_b                                       -- x4490 at depth 3
         ) arg_a                                          -- x4489 at depth 2
        )
    )
)

Wait, that doesn't match the depth pattern. Let me re-derive from the data.

Depth at key points:
- x4488, x4489, x4490, x4495, x4496 defined at depth 2 (outer body params)
- x4496 used at depth 3, offset 7343
- x4495 used at depth 3, offset 7349
- x4489 used at depth 2, offset 7356 (one level out from x4496/x4495)
- x4490 used at depth 3, offset 14411
- x4488 used at depth 3, offset 14539
- x4489 used at depth 4, offset 21594
- x4490 used at depth 4, offset 21600
- x4490 used at depth 3, offset 21607
- x4489 used at depth 2, offset 21614

The ending: ... x7344) x4489 x4490) x4490) x4489))

Reading right-to-left from the end:
  )) closes SE and BIG_SECTION parens
  x4489) -- at depth 2, closes with the big section
  x4490) -- at depth 3
  x4489 x4490) -- at depth 4

So the structure is:
  (((something x7344) x4489 x4490) x4490) x4489

Which is:  result = (((INNER x7344) x4489 x4490) x4490) x4489
Or:        (INNER x7344 x4489 x4490) x4490 x4489
Which is:  INNER(x7344)(x4489)(x4490)(x4490)(x4489)

Hmm, that's 5 arguments applied to INNER.
INNER is the result of Section 3.
x7344 is a section 3 variable.

Actually, let me re-read the ending text more carefully:
"...x7344) x4489 x4490) x4490) x4489))"

The parens:
- "...x7344)" -- closes some group containing x7344
- " x4489 x4490)" -- closes another group with x4489 and x4490 as args
- " x4490)" -- closes yet another group with x4490
- " x4489))" -- closes two more groups with x4489

So it's structured like:
  (... (... (... x7344) x4489 x4490) x4490) x4489

= (((RESULT3 x7344) x4489 x4490) x4490) x4489

This means RESULT3 takes 5 args: x7344, x4489, x4490, x4490, x4489
Which is: RESULT3(sec3_data)(arg_a)(arg_b)(arg_b)(arg_a)

This looks like constructing a diamond/5-tuple:
diamond(COND)(NW)(NE)(SW)(SE) = \f. f(COND)(NW)(NE)(SW)(SE)

So the 5 args would be: COND=x7344, NW=x4489, NE=x4490, SW=x4490, SE=x4489
That means: COND=section3_data, NW=arg_a, NE=arg_b, SW=arg_b, SE=arg_a

But that seems too simple. Let me check with actual diamond construction.
A diamond node at the top level would be:
\f. f(COND)(NW)(NE)(SW)(SE)

But the result of the Y-comb is not a lambda - it's an APPLICATION.
The result is: RESULT3 applied to 5 args.

Wait - RESULT3 is the LAST Y-comb section's output.
And it takes x7344 (section 3 data) and then 4 quadrant args.

Actually, I think RESULT3 is a DIAMOND CONSTRUCTOR:
result(cond)(nw)(ne)(sw)(se) = \f. f(cond)(nw)(ne)(sw)(se)

So RESULT3 is building a diamond node where:
  COND = x7344 (the section 3 computed value)
  NW = x4489 (arg_a)
  NE = x4490 (arg_b)
  SW = x4490 (arg_b)
  SE = x4489 (arg_a)

This is symmetric! NW=SE=arg_a, NE=SW=arg_b
That makes sense for a recursive quadtree construction.

Now let me also look at junction 1 ending:
"...x4534) x4496 x4495) x4489"
= ((...x4534) x4496 x4495) applied, then result applied to x4489

And junction 2:
"...x5929) x4490"
= (...x5929) applied to x4490

So:
SECTION_1 computes RESULT1 = (...(INNER1 x4534) x4496 x4495)
Then RESULT1 is applied to x4489 (arg_a)
Then the continuation is (\x5928.\x5929. SECTION_2...)

So after SECTION_1:
  RESULT1(x4489)(\x5928.\x5929. SECTION_2...)
  = SECTION_1_RESULT applied to arg_a and a continuation

Then inside SECTION_2:
  RESULT2(x5929)(x4490)(CONNECTOR (x4488 SECTION_3_CALL))

And SECTION_3 uses x4488 (outer recursion).

This means the outer Y-combinator does actually recurse!
x4488 is called with new arguments at offset 14539.

Let me understand what x4488 is called WITH:
"(x4488 (\x7343. \x7344. ...) ...)"
So x4488 receives a new function as first argument.
"""

# Let me now trace the EXACT arguments passed to the outer recursion
import re

data = open('d:/github/atgt2026hp_stars/extracted/left_x_lambda.txt', 'r').read()
se = data[21866:45037]

# At offset 14539: (x4488 (\x7343. \x7344. ...) ...)
# x4488 takes 4 args: x4489, x4490, x4495, x4496 (excluding self x4488)
# But actually Y(f) means x4488 = f(x4488), so when we call x4488,
# we pass the 4 remaining args

# Let me find what comes after x4488 at offset 14539
after_x4488 = se[14543:]  # after "x4488"
print("After x4488 recursive call:")
# Parse the arguments
pos = 0
while pos < len(after_x4488) and after_x4488[pos] == ' ':
    pos += 1

depth = 0
args = []
arg_start = None
for i in range(pos, len(after_x4488)):
    c = after_x4488[i]
    if c == '(' and depth == 0:
        arg_start = i
        depth = 1
    elif c == '(':
        depth += 1
    elif c == ')':
        depth -= 1
        if depth == 0 and arg_start is not None:
            args.append(after_x4488[arg_start:i+1])
            arg_start = None
    elif depth == 0 and c in 'x':
        # bare variable
        j = i
        while j < len(after_x4488) and after_x4488[j] not in ' ()':
            j += 1
        args.append(after_x4488[i:j])
        break
    if len(args) >= 5:
        break

print(f"x4488 is called with {len(args)} arguments:")
for i, a in enumerate(args):
    print(f"  arg{i}: {a[:200]}{'...' if len(a) > 200 else ''}")
    print()

# Also show what's BEFORE x4488 to understand the context
before_x4488 = se[14400:14545]
print("\nContext before x4488 call:")
print(before_x4488)

# And let me look at junction 1 more carefully
print("\n=== Junction 1 detail ===")
# "...x4534) x4496 x4495) x4489 (\x5928..."
# Let me trace back to find the beginning of the expression that ends with x4534
# x4534 is defined at offset 303 in a lambda: \x4533. \x4534. ...

# At junction 1, x4534 is USED at depth 3 (offset ~7330)
# Let me find the exact position of x4534 usage near junction 1
for m in re.finditer(r'\bx4534\b', se):
    if m.start() > 7000:
        ctx = se[max(0,m.start()-30):m.start()+30]
        print(f"  x4534 at {m.start()}: {ctx}")
