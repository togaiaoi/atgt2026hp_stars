#!/usr/bin/env python3
"""
Step 4b: Convert decoder components X and Y to lambda expressions.

LEFT = ((S X)(K Y)), where LEFT(R) = X(R)(Y).
X and Y contain the decoder logic.

We convert them to lambda and analyze the structure,
looking for known operator patterns.
"""

import os
import sys
import io

if hasattr(sys.stdout, 'buffer') and not sys.stdout.closed:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.setrecursionlimit(100000)

# Import from step3
sys.path.insert(0, os.path.dirname(__file__))
from step3_reverse_bracket import (
    compact_to_ski_tree, revert, reset_var_counter, lambda_to_str,
    lambda_depth, count_lambdas, Var, Lam, App, Atom
)


def compact_to_lambda(compact_str, name="expr"):
    """Convert a compact string to lambda expression."""
    print(f"  Parsing {name} ({len(compact_str):,} chars) to SKI tree...")
    tree = compact_to_ski_tree(compact_str)
    print(f"  Running reverse bracket abstraction...")
    reset_var_counter()
    result = revert(tree)
    lam_str = lambda_to_str(result)
    depth = lambda_depth(result)
    n_lam = count_lambdas(result)
    print(f"  Result: depth={depth}, lambdas={n_lam}, str_len={len(lam_str):,}")
    return result, lam_str


def analyze_lambda_structure(expr, name="", depth=0, max_depth=6):
    """Analyze the structure of a lambda expression."""
    indent = "  " * depth

    if isinstance(expr, Var):
        print(f"{indent}{name}: var {expr.name}")
        return
    if isinstance(expr, Atom):
        print(f"{indent}{name}: atom {expr.name}")
        return
    if isinstance(expr, Lam):
        # Count consecutive lambdas
        lam_count = 0
        vars_list = []
        current = expr
        while isinstance(current, Lam):
            lam_count += 1
            vars_list.append(current.var.name)
            current = current.body
        vars_str = ' '.join(vars_list[:10])
        if len(vars_list) > 10:
            vars_str += f' ... (+{len(vars_list)-10} more)'
        print(f"{indent}{name}: \\{vars_str}. [body]")
        if depth < max_depth:
            analyze_lambda_structure(current, "body", depth + 1, max_depth)
        return
    if isinstance(expr, App):
        # Collect application spine
        func = expr.func
        args = [expr.arg]
        while isinstance(func, App):
            args.append(func.arg)
            func = func.func
        args.reverse()
        func_desc = ""
        if isinstance(func, Var):
            func_desc = f"var {func.name}"
        elif isinstance(func, Atom):
            func_desc = f"atom {func.name}"
        elif isinstance(func, Lam):
            func_desc = f"lambda (...)"
        else:
            func_desc = f"complex"
        print(f"{indent}{name}: apply {func_desc} to {len(args)} args")
        if depth < max_depth:
            if isinstance(func, Lam):
                analyze_lambda_structure(func, "func", depth + 1, max_depth)
            for i, arg in enumerate(args[:5]):
                arg_str = lambda_to_str(arg)
                if len(arg_str) <= 80:
                    print(f"{indent}  arg{i}: {arg_str}")
                else:
                    analyze_lambda_structure(arg, f"arg{i}", depth + 1, max_depth)
            if len(args) > 5:
                print(f"{indent}  ... (+{len(args)-5} more args)")
        return


def find_recursive_patterns(expr, path="", depth=0, max_depth=20, results=None):
    """Find patterns like (f f) which indicate self-application / recursion."""
    if results is None:
        results = []
    if depth > max_depth:
        return results

    if isinstance(expr, App):
        # Check for (f f) pattern - self-application
        if isinstance(expr.func, Var) and isinstance(expr.arg, Var):
            if expr.func.name == expr.arg.name:
                results.append((path, f"self-app: {expr.func.name}"))

        find_recursive_patterns(expr.func, path + ".func", depth + 1, max_depth, results)
        find_recursive_patterns(expr.arg, path + ".arg", depth + 1, max_depth, results)
    elif isinstance(expr, Lam):
        find_recursive_patterns(expr.body, path + ".body", depth + 1, max_depth, results)

    return results


def main():
    print("=== Step 4b: Lambda Decoder Analysis ===\n")

    base = os.path.join(os.path.dirname(__file__), '..')
    ext_dir = os.path.join(base, 'extracted')

    # Read X and Y
    x_path = os.path.join(ext_dir, 'left_x.txt')
    y_path = os.path.join(ext_dir, 'left_y.txt')

    if not os.path.exists(x_path):
        print("Run step4_find_operators.py first to extract X and Y.")
        sys.exit(1)

    with open(x_path, 'r') as f:
        x_compact = f.read().strip()
    with open(y_path, 'r') as f:
        y_compact = f.read().strip()

    print(f"X: {len(x_compact):,} chars")
    print(f"Y: {len(y_compact):,} chars")

    # Convert X to lambda
    print(f"\n=== Converting X to lambda ===")
    try:
        x_lambda, x_str = compact_to_lambda(x_compact, "X")
        # Save
        with open(os.path.join(ext_dir, 'left_x_lambda.txt'), 'w') as f:
            f.write(x_str)
        print(f"  Saved to extracted/left_x_lambda.txt")

        print(f"\n  X lambda structure:")
        analyze_lambda_structure(x_lambda, "X", max_depth=5)

        # Find self-application patterns (recursion)
        recs = find_recursive_patterns(x_lambda, max_depth=30)
        if recs:
            print(f"\n  Self-application patterns in X: {len(recs)}")
            for path, desc in recs[:10]:
                print(f"    {desc} at {path}")

    except Exception as e:
        print(f"  Error converting X: {e}")
        import traceback
        traceback.print_exc()

    # Convert Y to lambda
    print(f"\n=== Converting Y to lambda ===")
    try:
        y_lambda, y_str = compact_to_lambda(y_compact, "Y")
        with open(os.path.join(ext_dir, 'left_y_lambda.txt'), 'w') as f:
            f.write(y_str)
        print(f"  Saved to extracted/left_y_lambda.txt")

        print(f"\n  Y lambda structure:")
        analyze_lambda_structure(y_lambda, "Y", max_depth=5)

        recs = find_recursive_patterns(y_lambda, max_depth=30)
        if recs:
            print(f"\n  Self-application patterns in Y: {len(recs)}")
            for path, desc in recs[:10]:
                print(f"    {desc} at {path}")

    except Exception as e:
        print(f"  Error converting Y: {e}")
        import traceback
        traceback.print_exc()

    # Show the first part of X and Y lambda strings for manual inspection
    print(f"\n=== X lambda (first 2000 chars) ===")
    print(x_str[:2000] if 'x_str' in dir() else "(not available)")

    print(f"\n=== Y lambda (first 2000 chars) ===")
    print(y_str[:2000] if 'y_str' in dir() else "(not available)")


if __name__ == '__main__':
    main()
