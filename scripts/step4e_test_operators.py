#!/usr/bin/env python3
"""
Step 4e: Test the 5 operators from X using the SKI graph evaluator.

Apply each operator to known inputs and observe outputs.
"""

import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.setrecursionlimit(200000)

sys.path.insert(0, os.path.dirname(__file__))
from ski_eval import (
    compact_to_graph, whnf, follow, make_app,
    make_s, make_k, make_i, make_true_g, make_false_g,
    make_pair_g, make_nil_g, make_scott_num_g,
    eval_app, eval_apps, decode_bool_graph, decode_scott_num_graph,
    decode_pair_fst, decode_pair_snd, graph_to_compact, Node
)
from step4c_extract_operators import extract_s_spine_args, tree_to_compact
from step3_reverse_bracket import compact_to_ski_tree


def describe_whnf(node, depth=0):
    """Describe a WHNF node for debugging."""
    if depth > 5:
        return "..."
    node = follow(node)
    if node.tag == 0:  # APP
        f = describe_whnf(node.a, depth + 1)
        a = describe_whnf(node.b, depth + 1)
        return f"({f} {a})"
    if node.tag == 1:  # S
        return "S"
    if node.tag == 2:  # K
        return "K"
    if node.tag == 3:  # I
        return "I"
    if node.tag == 4:  # S1
        return f"(S {describe_whnf(node.a, depth+1)})"
    if node.tag == 5:  # S2
        return f"(S {describe_whnf(node.a, depth+1)} {describe_whnf(node.b, depth+1)})"
    if node.tag == 6:  # K1
        return f"(K {describe_whnf(node.a, depth+1)})"
    return f"?{node.tag}"


def test_op(op_graph, args_desc, args, fuel=2000000):
    """Apply operator to args, try decoding result."""
    # Build graph copies for each arg
    expr = op_graph
    for arg in args:
        expr = make_app(expr, arg)

    _, steps = whnf(expr, fuel)
    result = follow(expr)

    # Try various decodings
    result_desc = describe_whnf(result, 0)
    if len(result_desc) > 200:
        result_desc = result_desc[:200] + "..."

    # Check boolean
    # Make a copy for bool test (don't disturb the result)
    b = decode_bool_graph(result, fuel // 10)
    if b is True:
        print(f"  {args_desc} -> TRUE  [{steps} steps]")
        return "TRUE"
    elif b is False:
        print(f"  {args_desc} -> FALSE  [{steps} steps]")
        return "FALSE"

    # Check Scott number
    n, nsteps = decode_scott_num_graph(result, fuel // 10)
    if n is not None and n >= 0:
        print(f"  {args_desc} -> NUMBER({n})  [{steps}+{nsteps} steps]")
        return n

    print(f"  {args_desc} -> {result_desc}  [{steps} steps]")
    return result_desc


def main():
    print("=== Step 4e: Test Operators with SKI Evaluator ===\n")

    base = os.path.join(os.path.dirname(__file__), '..')
    ext_dir = os.path.join(base, 'extracted')

    # Read and parse X
    with open(os.path.join(ext_dir, 'left_x.txt'), 'r') as f:
        x_compact = f.read().strip()

    x_tree = compact_to_ski_tree(x_compact)
    x_args_trees = extract_s_spine_args(x_tree)

    # Convert to compact strings and build graphs
    op_compacts = []
    for i, arg in enumerate(x_args_trees):
        c = tree_to_compact(arg)
        op_compacts.append(c)
        print(f"Operator {i}: {len(c):,} compact chars")

    print()

    # === Test arg3 (57 chars - smallest) ===
    print("=" * 60)
    print("=== arg3 (57 chars) ===")
    print(f"Compact: {op_compacts[3]}")
    op3 = compact_to_graph(op_compacts[3])

    # Test with booleans
    test_op(compact_to_graph(op_compacts[3]), "arg3(true, S)", [make_true_g(), make_s()])
    test_op(compact_to_graph(op_compacts[3]), "arg3(false, S)", [make_false_g(), make_s()])

    # Test with scott numbers
    for n in [0, 1, 2, 3, 4, 5, 6, 7]:
        test_op(compact_to_graph(op_compacts[3]), f"arg3({n}, K)",
                [make_scott_num_g(n), make_k()])

    # Test: what does arg3 do with pairs?
    p_tf = make_pair_g(make_true_g(), make_false_g())
    test_op(compact_to_graph(op_compacts[3]), "arg3(pair(T,F), K)",
            [p_tf, make_k()])

    p_ft = make_pair_g(make_false_g(), make_true_g())
    test_op(compact_to_graph(op_compacts[3]), "arg3(pair(F,T), K)",
            [p_ft, make_k()])

    # === Test arg2 (243 chars - Y combinator) ===
    print()
    print("=" * 60)
    print("=== arg2 (243 chars) ===")

    # arg2 is a recursive function (Y combinator). Test with lists/numbers
    for n in [0, 1, 2, 3, 5, 7]:
        test_op(compact_to_graph(op_compacts[2]), f"arg2({n})",
                [make_scott_num_g(n)])

    # Test with nil
    test_op(compact_to_graph(op_compacts[2]), "arg2(nil)",
            [make_nil_g()])

    # Test with pair of scott nums
    p = make_pair_g(make_scott_num_g(1), make_pair_g(make_scott_num_g(2), make_nil_g()))
    test_op(compact_to_graph(op_compacts[2]), "arg2([1, 2])", [p])

    # === Test arg0 (20K chars) ===
    print()
    print("=" * 60)
    print("=== arg0 (20,219 chars) ===")

    # Test with 2 Scott number args (arg0 takes 2 args based on lambda)
    for a, b in [(0, 0), (1, 0), (0, 1), (1, 1), (2, 1), (1, 2),
                  (3, 2), (2, 3), (5, 3), (3, 5), (4, 4), (7, 3)]:
        test_op(compact_to_graph(op_compacts[0]), f"arg0({a}, {b})",
                [make_scott_num_g(a), make_scott_num_g(b)])

    # === Test arg1 (10K chars) ===
    print()
    print("=" * 60)
    print("=== arg1 (10,409 chars) ===")

    # arg1 is Y combinator applied. Try different arg counts.
    # First with 1 arg:
    for n in [0, 1, 2, 3]:
        test_op(compact_to_graph(op_compacts[1]), f"arg1({n})",
                [make_scott_num_g(n)])

    # Try with 2 args:
    for a, b in [(0, 0), (1, 0), (0, 1), (1, 1), (2, 1), (3, 2)]:
        test_op(compact_to_graph(op_compacts[1]), f"arg1({a}, {b})",
                [make_scott_num_g(a), make_scott_num_g(b)])

    # Try with 3 args (from lambda: 3-param recursive function):
    for a, b, c in [(1, 1, 0), (2, 1, 0), (1, 0, 1)]:
        test_op(compact_to_graph(op_compacts[1]), f"arg1({a}, {b}, {c})",
                [make_scott_num_g(a), make_scott_num_g(b), make_scott_num_g(c)])

    # === Test arg4 (30K chars) ===
    print()
    print("=" * 60)
    print("=== arg4 (30,463 chars) ===")

    # arg4 takes 1 arg first (from lambda)
    for n in [0, 1, 2, 3]:
        test_op(compact_to_graph(op_compacts[4]), f"arg4({n})",
                [make_scott_num_g(n)])

    # 2 args
    for a, b in [(0, 0), (1, 0), (0, 1), (1, 1), (2, 3)]:
        test_op(compact_to_graph(op_compacts[4]), f"arg4({a}, {b})",
                [make_scott_num_g(a), make_scott_num_g(b)])


if __name__ == '__main__':
    main()
