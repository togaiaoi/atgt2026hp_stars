#!/usr/bin/env python3
"""
Step 4c: Extract individual operator definitions from X and Y.

X = lam x1. x1 arg0 arg1 arg2 arg3 arg4
In SKI: X = S(S(S(S(SI(K arg0))(K arg1))(K arg2))(K arg3))(K arg4)

We parse the tree and walk the S-spine to extract arg0-arg4.
Then we convert each to lambda and try to identify them.
"""

import os
import sys
import io

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.setrecursionlimit(100000)

sys.path.insert(0, os.path.dirname(__file__))
from step3_reverse_bracket import (
    compact_to_ski_tree, revert, reset_var_counter, lambda_to_str,
    Var, Lam, App, Atom
)


def tree_to_compact(tree):
    """Convert an SKI tree back to compact string."""
    if isinstance(tree, Atom):
        if tree.name == 'S':
            return 'k'
        elif tree.name == 'K':
            return 'X'
        elif tree.name == 'I':
            return 'D'
        else:
            return '?'
    elif isinstance(tree, App):
        return tree_to_compact(tree.func) + tree_to_compact(tree.arg) + '-'
    else:
        return '?'


def tree_size(tree):
    """Count nodes in tree."""
    if isinstance(tree, (Atom, Var)):
        return 1
    elif isinstance(tree, App):
        return 1 + tree_size(tree.func) + tree_size(tree.arg)
    elif isinstance(tree, Lam):
        return 1 + tree_size(tree.body)
    return 1


def extract_s_spine_args(tree):
    """
    Extract arguments from S-spine application pattern.
    X = S(S(S(...(SI(K arg0))(K arg1))...)(K argN-1))(K argN)

    Walk down: tree = S(inner)(K argN) -> collect argN, recurse on inner
    Until we hit SI(K arg0) at the core.
    """
    args = []
    current = tree

    while True:
        # current should be App(App(S, inner), App(K, arg))
        if not isinstance(current, App):
            break
        if not isinstance(current.func, App):
            break

        outer_func = current.func  # App(S, inner)
        outer_arg = current.arg    # App(K, arg_i) or just arg_i

        # Check if outer_func.func is S
        if isinstance(outer_func.func, Atom) and outer_func.func.name == 'S':
            inner = outer_func.arg

            # outer_arg should be App(K, arg_i)
            if isinstance(outer_arg, App) and isinstance(outer_arg.func, Atom) and outer_arg.func.name == 'K':
                arg_i = outer_arg.arg
                args.append(arg_i)
                current = inner
            else:
                # Maybe the last arg isn't wrapped in K
                args.append(outer_arg)
                current = inner
        else:
            break

    # At the core: should be SI(K arg0) or I or similar
    # SI(K arg0) = App(App(S, I), App(K, arg0))
    if isinstance(current, App) and isinstance(current.func, App):
        if (isinstance(current.func.func, Atom) and current.func.func.name == 'S' and
            isinstance(current.func.arg, Atom) and current.func.arg.name == 'I'):
            # Core is S I (K arg0)
            core_arg = current.arg
            if isinstance(core_arg, App) and isinstance(core_arg.func, Atom) and core_arg.func.name == 'K':
                args.append(core_arg.arg)
            else:
                args.append(core_arg)

    args.reverse()
    return args


def identify_pattern(compact_str):
    """Try to identify known compact patterns."""
    known = {
        'D': 'I (identity)',
        'X': 'K (const)',
        'k': 'S',
        'XD-': 'false (KI = \\a.\\b.b)',
        'kXX--D-': 'true (S(KK)I = \\a.\\b.a)',
        'kXk--X-': 'B (compose = S(KS)K)',
        'kkD-XXD----XkXX--D---': 'not',
        'kXk--kXX--D---': 'B true = \\f.\\x.\\y.f x',
        # Y combinator half
        'kkkXk--kXX--D---XkD-D----': 'Y_half (\\f. f(\\x. f(x x)))',
    }
    if compact_str in known:
        return known[compact_str]

    # Check prefix patterns
    if compact_str.startswith('kkkXk--kXX--D---'):
        return 'Y combinator applied to ...'

    return None


def analyze_lambda_deeply(expr, name="", max_depth=3):
    """Analyze lambda structure to identify operator behavior."""
    # Count params
    params = []
    body = expr
    while isinstance(body, Lam):
        params.append(body.var.name)
        body = body.body

    if params:
        print(f"  {name}: \\{' '.join(params[:20])}{'...' if len(params)>20 else ''}. <body>")

    # Analyze body structure
    if isinstance(body, Var):
        print(f"    body = {body.name}")
    elif isinstance(body, App):
        # Collect application spine
        func = body.func
        args = [body.arg]
        while isinstance(func, App):
            args.append(func.arg)
            func = func.func
        args.reverse()

        if isinstance(func, Var):
            print(f"    body = {func.name} applied to {len(args)} args")
            if len(args) <= 8:
                for i, a in enumerate(args):
                    a_str = lambda_to_str(a)
                    if len(a_str) <= 120:
                        print(f"      arg{i}: {a_str}")
                    else:
                        # Go one level deeper
                        inner_params = []
                        inner_body = a
                        while isinstance(inner_body, Lam):
                            inner_params.append(inner_body.var.name)
                            inner_body = inner_body.body
                        if inner_params:
                            inner_str = lambda_to_str(inner_body)
                            if len(inner_str) <= 200:
                                print(f"      arg{i}: \\{' '.join(inner_params[:10])}. {inner_str[:200]}")
                            else:
                                print(f"      arg{i}: \\{' '.join(inner_params[:10])}. <{len(a_str)} chars>")
                        else:
                            print(f"      arg{i}: <{len(a_str)} chars>")
        elif isinstance(func, Lam):
            print(f"    body = (\\...) applied to {len(args)} args")


def check_known_lambda_patterns(lam_str):
    """Check if a lambda string matches known patterns."""
    patterns = {
        # Basic
        '\\x1. \\x2. x1': 'TRUE (\\a.\\b.a)',
        '\\x1. \\x2. x2': 'FALSE (\\a.\\b.b)',
        '\\x1. x1': 'IDENTITY',
        '\\x1. \\x2. x1 x2': 'CHURCH 1 (\\f.\\x. f x)',
        '\\x1. \\x2. \\x3. x1 x3 (x2 x3)': 'S combinator',
        # NOT: \x.\a.\b. x b a
        '\\x1. \\x2. \\x3. x1 x3 x2': 'NOT (flip args)',
    }
    for pat, name in patterns.items():
        if lam_str == pat:
            return name

    # Partial patterns
    if lam_str.startswith('\\x1. \\x2. x1 ('):
        return 'Possible pair/cons pattern'
    if '(x1 x1)' in lam_str or 'x1 (x1' in lam_str:
        return 'Contains self-application (recursion)'

    return None


def main():
    print("=== Step 4c: Extract Operators from X and Y ===\n")

    base = os.path.join(os.path.dirname(__file__), '..')
    ext_dir = os.path.join(base, 'extracted')

    # Read X
    x_path = os.path.join(ext_dir, 'left_x.txt')
    with open(x_path, 'r') as f:
        x_compact = f.read().strip()

    print(f"X: {len(x_compact):,} chars")

    # Parse X into tree
    print("Parsing X to tree...")
    x_tree = compact_to_ski_tree(x_compact)

    # Extract S-spine arguments
    print("Extracting S-spine arguments from X...")
    x_args = extract_s_spine_args(x_tree)
    print(f"Found {len(x_args)} arguments\n")

    for i, arg in enumerate(x_args):
        compact = tree_to_compact(arg)
        size = tree_size(arg)
        known = identify_pattern(compact)

        print(f"=== X arg{i}: {len(compact):,} compact chars, {size} nodes ===")
        if known:
            print(f"  Identified: {known}")

        if len(compact) <= 200:
            print(f"  Compact: {compact}")
        else:
            print(f"  Compact prefix: {compact[:100]}...")
            print(f"  Compact suffix: ...{compact[-100:]}")

        # Convert to lambda
        if len(compact) <= 100000:
            try:
                reset_var_counter()
                lam = revert(arg)
                lam_str = lambda_to_str(lam)

                # Check known patterns
                known_lam = check_known_lambda_patterns(lam_str)
                if known_lam:
                    print(f"  Lambda match: {known_lam}")

                if len(lam_str) <= 500:
                    print(f"  Lambda: {lam_str}")
                else:
                    print(f"  Lambda ({len(lam_str):,} chars): {lam_str[:300]}...")

                # Deeper analysis
                analyze_lambda_deeply(lam, f"arg{i}")

                # Save
                out_path = os.path.join(ext_dir, f'x_arg{i}.txt')
                with open(out_path, 'w') as f:
                    f.write(f"compact: {compact}\n")
                    f.write(f"lambda: {lam_str}\n")

            except Exception as e:
                print(f"  Lambda conversion error: {e}")

        print()

    # Also analyze Y
    print("\n=== Analyzing Y ===")
    y_path = os.path.join(ext_dir, 'left_y.txt')
    with open(y_path, 'r') as f:
        y_compact = f.read().strip()

    print(f"Y: {len(y_compact):,} chars")

    # Check if Y has the same S-spine structure
    y_tree = compact_to_ski_tree(y_compact)

    # Convert Y to lambda and analyze top-level structure
    try:
        reset_var_counter()
        y_lam = revert(y_tree)
        y_str = lambda_to_str(y_lam)
        print(f"Y lambda: {len(y_str):,} chars")

        # Analyze Y's structure
        analyze_lambda_deeply(y_lam, "Y")

        # Y's top-level: \x1. body
        if isinstance(y_lam, Lam):
            body = y_lam.body
            if isinstance(body, App):
                # \x1. (func arg)
                func = body.func
                arg = body.arg

                func_str = lambda_to_str(func)
                arg_str = lambda_to_str(arg)
                print(f"\n  Y = \\x1. (F)(A) where")
                print(f"    F: {func_str[:200]}{'...' if len(func_str)>200 else ''}")
                print(f"    A: {arg_str[:200]}{'...' if len(arg_str)>200 else ''}")
    except Exception as e:
        print(f"  Error: {e}")

    # Summary
    print("\n\n=== SUMMARY ===")
    print("LEFT = ((S X)(K Y))")
    print("LEFT(R) = X(R)(Y)")
    print(f"X = \\r. r arg0 arg1 arg2 arg3 arg4")
    print(f"So LEFT(R) = R(arg0)(arg1)(arg2)(arg3)(arg4)(Y)")
    print(f"\nThe 5+1 arguments passed to the program are:")
    for i, arg in enumerate(x_args):
        compact = tree_to_compact(arg)
        known = identify_pattern(compact)
        print(f"  arg{i}: {len(compact):,} chars {f'= {known}' if known else ''}")
    print(f"  Y: {len(y_compact):,} chars")


if __name__ == '__main__':
    main()
