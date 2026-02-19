#!/usr/bin/env python3
"""Analyze the common Y-combinator section pattern.

Each section in COND, NW, and SE follows a similar recursive pattern.
The goal is to understand what each Y-combinator section computes
and how sections chain together.

From the analysis:
- SE has 3 major "sections" (separated by junction points where outer vars appear)
- Each section has ~7-8 Y-combinators inside
- The outermost Y-comb of each section takes params like:
    \rec.\arg1.\arg2.\...
  and processes two parallel lists (arg1 and arg2 are pair2 cons cells)

Key hypothesis: Each section is a "zip-with" operation over two lists,
building a new quadtree level from the data.
"""

import re

data = open('d:/github/atgt2026hp_stars/extracted/left_x_lambda.txt', 'r').read()

# Let me analyze the INNERMOST Y-combinator pattern
# In SE section 1, the first Y-comb after the outer one is at offset 146:
# x4500 (x4502 x4502) (\x4508. x4500 (x4508 x4508))
# Body after: (\x4514. \x4515. \x4516. x4516 (\x4520. \x4521. x4514 x4520 x4521) x4515)
# Init: (\x4528. \x4529. x4529)

# Let me decode this inner Y-comb:
# Y(\x4514.\x4515.\x4516.
#     x4516(\x4520.\x4521. x4514 x4520 x4521)(x4515))
# (KI)  -- init = false
#
# Rename: Y(\rec.\acc.\list.
#     list(\fst.\snd. rec fst snd)(acc))
# (false)
#
# This is: rec(false)(list) = list(\fst.\snd. rec(fst)(snd))(false)
# If list=nil: returns false (=acc)
# If list=cons(A,B): returns rec(A)(B) -- replaces acc with A, continues with B
#
# Wait - this isn't accumulating, it's just extracting the LAST element!
# rec(false)(cons(a, cons(b, cons(c, nil))))
# = cons_handler(a)(cons(b,cons(c,nil)))
# = rec(a)(cons(b,cons(c,nil)))
# = cons_handler(b)(cons(c,nil))
# = rec(b)(cons(c,nil))
# = rec(c)(nil)
# = nil_handler = c (the acc at this point)
# Actually no: rec(c)(nil) = nil(\f.\s. rec f s)(c) = c
# So this returns the LAST element of the list!

print("Inner Y-comb #1 in SE Section 1:")
print("  Y(\\rec.\\acc.\\list. list(\\fst.\\snd. rec(fst)(snd))(acc))(false)")
print("  = fold that replaces acc with current fst, continuing with snd")
print("  = returns LAST element of the list")
print()

# Next Y-comb at offset 327 in SE:
# \x4533. \x4534. \x4541. \x4543. x4541 (x4543 x4543) (\x4549. x4541 (x4549 x4549))
# Body: \x4555. \x4556. \x4557. \x4558. \x4559. \x4560. x4559 (\x4564. \x4565. x4560 (...))
#
# This is more complex - it takes 6 params after rec
# x4555 = rec2, x4556 = ?, x4557 = ?, x4558 = ?, x4559 = ?, x4560 = ?

# Let me look at the text
se = data[21866:45037]
# Find the body of Y-comb #2 in SE
y2_start = se.index('\\x4555.')
print(f"Y-comb #2 body starts at SE offset {y2_start}")
y2_text = se[y2_start:y2_start+600]
print(f"Body text: {y2_text}")
print()

# \x4555.\x4556.\x4557.\x4558.\x4559.\x4560.
# x4559(\x4564.\x4565. x4560(
#   \x4569.\x4570.\x4575.\x4576.\x4577.\x4578.
#   x4578(\x4584.\x4585. x4584(\x4590. x4590 FALSE TRUE)(\x4600. x4600) x4585
#     (\x4605.\x4606. x4605(\x4611. x4611 FALSE TRUE)(\x4621. x4621) x4606 x4575 x4576) x4577)
#   (x4575 (x4576 K x4577) (x4576 x4577 KI))
#   x4564 x4569 x4558
#   (\x4647.\x4648.\x4649.\x4650. x4649 x4647 (x4555 x4564 x4569 x4648 x4565 x4570)))

# This is the main computation! It takes:
# x4555 = outer rec
# x4556-x4560 = various params
# x4559 = something being decomposed (pair2 cons case)
# x4560 = another value
# The inner function processes bits of a number

# The pattern x4584(\x4590. x4590 FALSE TRUE) means:
# x4584 is a boolean: if true -> FALSE, if false -> TRUE
# This is NOT(x4584)!

# And (x4575 (x4576 K x4577) (x4576 x4577 KI))
# x4576 is probably a pair. x4576(K)(x4577) = pair_fst(x4576)(x4577)
# and x4576(x4577)(KI) would be... x4576 applied to x4577 then to KI

# This is getting very complex. Let me instead focus on the overall structure.

print("=" * 70)
print("OVERALL DECODER STRUCTURE SUMMARY")
print("=" * 70)
print()
print("1. left_x = diamond(COND, NW, NE, SW, SE) = Church 5-tuple")
print("   Applied as: left_x(selector) = selector(COND)(NW)(NE)(SW)(SE)")
print()
print("2. When the I/O interpreter gets p1=1, p2=2 (image output):")
print("   - data_node = I/O step 4's Q payload")
print("   - The image is: data_node applied to diamond selectors")
print("   - data_node = left_x partially applied to item_09 data")
print()
print("3. SE field structure:")
print("   SE = \\data.\\selector. selector(QUAD_BUILDER)(STEP_FUNC)")
print("   STEP_FUNC = \\step. step(step(step(data, 2048), 256), 32)")
print("   QUAD_BUILDER = Y(\\rec.\\arg_a.\\arg_b.\\arg_c.\\arg_d. BODY)")
print()
print("4. The three sections in QUAD_BUILDER's body:")
print("   Section 1 (offsets 130-7360): Processes two lists in parallel")
print("     - 8 Y-combs: list operations, bit processing, XOR/equality")
print("     - Ends with: ... x4534) x4496 x4495) x4489")
print("     - x4534 = intermediate result, x4496/x4495 = outer params")
print("     - Result applied to x4489 (arg_a)")
print()
print("   Section 2 (offsets 7360-14540): Similar parallel list processing")
print("     - 7 Y-combs: similar structure to Section 1")
print("     - Ends with: ... x5929) x4490 CONNECTOR")
print("     - Result applied to x4490 (arg_b)")
print("     - CONNECTOR includes a boolean constant")
print()
print("   Section 3 (offsets 14540-21600): Uses outer recursion x4488")
print("     - 7 Y-combs: builds the final diamond node")
print("     - x4488 called with 2 args: (section3_func)(x4490)")
print("     - Final return: RESULT(x7344)(x4489)(x4490)(x4490)(x4489)")
print("     - This constructs diamond(cond=x7344, nw=a, ne=b, sw=b, se=a)")
print("     - NOTE: nw==se and ne==sw (symmetric quadtree!)")
print()
print("5. The step function creates depth levels:")
print("   step(step(step(initial_data, 2048), 256), 32)")
print("   11 + 8 + 5 = 24 additional depth levels")
print("   Each step(tree, N) expands the quadtree by N resolution")
print()
print("6. NW field: Y(\\rec.\\list1.\\list2. BODY)")
print("   Processes TWO lists in parallel (list1=data, list2=data)")
print("   Similar Y-comb sections as SE (9 Y-combs)")
print("   Returns: ... (x2928 x2935 x2940)) false) (x2930 sel3 K)")
print("   Where x2928=rec, x2935=rest1, x2940=rest2")
print()
print("7. COND field: \\data.\\key. Y(BODY)")
print("   Takes data and key parameters")
print("   16 Y-combs for complex pixel color computation")
print("   Uses x8 (data) and x9 (key) at the end")
print()
print("8. NE field: Y(\\rec.\\acc.\\list. list_fold)(nil)")
print("   List reversal function")
print()
print("9. SW field: \\self.\\x. self(NOT(x))(x)")
print("   NOT-reflection for coordinate transformation")
print()
print("10. Operators:")
print("    arg0: Equality test on Scott numbers")
print("    arg1: XOR-like operation on numbers")
print("    arg2: List reversal")
print("    arg3: Conditional right-shift (if odd, return n>>1, else return b)")
print("    arg4: Complex binary operation (30K chars)")
