#!/usr/bin/env python3
"""
Step 4d: Evaluate operators with test inputs to identify their behavior.

Uses beta-reduction on lambda expressions to test operator functions
with known inputs (numbers, booleans, lists).
"""

import os
import sys
import copy

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.setrecursionlimit(100000)

sys.path.insert(0, os.path.dirname(__file__))
from step3_reverse_bracket import (
    compact_to_ski_tree, revert, reset_var_counter, lambda_to_str,
    Var, Lam, App, Atom
)


# === Beta reduction engine ===

_fresh = 0
def fresh():
    global _fresh
    _fresh += 1
    return f"_v{_fresh}"


def free_vars(expr):
    """Get free variables in expression."""
    if isinstance(expr, Var):
        return {expr.name}
    if isinstance(expr, Atom):
        return set()
    if isinstance(expr, Lam):
        return free_vars(expr.body) - {expr.var.name}
    if isinstance(expr, App):
        return free_vars(expr.func) | free_vars(expr.arg)
    return set()


def substitute(expr, var_name, value):
    """Substitute var_name with value in expr, avoiding capture."""
    if isinstance(expr, Var):
        return value if expr.name == var_name else expr
    if isinstance(expr, Atom):
        return expr
    if isinstance(expr, Lam):
        if expr.var.name == var_name:
            return expr  # Shadowed
        if expr.var.name in free_vars(value):
            # Alpha-rename to avoid capture
            new_name = fresh()
            new_var = Var(new_name)
            renamed_body = substitute(expr.body, expr.var.name, new_var)
            return Lam(new_var, substitute(renamed_body, var_name, value))
        return Lam(expr.var, substitute(expr.body, var_name, value))
    if isinstance(expr, App):
        return App(substitute(expr.func, var_name, value),
                   substitute(expr.arg, var_name, value))
    return expr


def beta_reduce_step(expr):
    """Try one step of beta reduction. Returns (new_expr, changed)."""
    if isinstance(expr, App):
        # Beta redex: (\x. body) arg
        if isinstance(expr.func, Lam):
            result = substitute(expr.func.body, expr.func.var.name, expr.arg)
            return result, True
        # Try reducing func first
        new_func, changed = beta_reduce_step(expr.func)
        if changed:
            return App(new_func, expr.arg), True
        # Then try reducing arg
        new_arg, changed = beta_reduce_step(expr.arg)
        if changed:
            return App(expr.func, new_arg), True
    if isinstance(expr, Lam):
        new_body, changed = beta_reduce_step(expr.body)
        if changed:
            return Lam(expr.var, new_body), True
    return expr, False


def normalize(expr, max_steps=5000):
    """Fully beta-normalize an expression."""
    for i in range(max_steps):
        new_expr, changed = beta_reduce_step(expr)
        if not changed:
            return expr, i
        expr = new_expr
    return expr, max_steps


def expr_size(expr):
    """Count AST nodes."""
    if isinstance(expr, (Var, Atom)):
        return 1
    if isinstance(expr, Lam):
        return 1 + expr_size(expr.body)
    if isinstance(expr, App):
        return expr_size(expr.func) + expr_size(expr.arg)
    return 1


def normalize_safe(expr, max_steps=5000, max_size=50000):
    """Normalize with size limit to prevent blowup."""
    for i in range(max_steps):
        if expr_size(expr) > max_size:
            return expr, i, "SIZE_LIMIT"
        new_expr, changed = beta_reduce_step(expr)
        if not changed:
            return expr, i, "NORMAL"
        expr = new_expr
    return expr, max_steps, "STEP_LIMIT"


# === Constructors for test values ===

def make_true():
    """true = lam a. lam b. a"""
    a, b = Var("a"), Var("b")
    return Lam(a, Lam(b, a))

def make_false():
    """false = lam a. lam b. b"""
    a, b = Var("a"), Var("b")
    return Lam(a, Lam(b, b))

def make_pair(x, y):
    """pair(x,y) = lam f. f x y"""
    f = Var("f")
    return Lam(f, App(App(f, x), y))

def make_nil():
    """nil = false = lam a. lam b. b"""
    return make_false()

def make_scott_num(n):
    """Make Scott-encoded binary number (LSB first).
    0 = nil (false)
    n = pair(bit0, pair(bit1, ...))
    """
    if n == 0:
        return make_nil()
    result = make_nil()
    bits = []
    while n > 0:
        bits.append(n & 1)
        n >>= 1
    for bit in reversed(bits):
        result = make_pair(make_true() if bit else make_false(), result)
    return result

def make_list(items):
    """Make a list from items: cons(item0, cons(item1, ... nil))"""
    result = make_nil()
    for item in reversed(items):
        result = make_pair(item, result)
    return result

def make_church(n):
    """Make Church numeral n."""
    f, x = Var("f"), Var("x")
    body = x
    for _ in range(n):
        body = App(f, body)
    return Lam(f, Lam(x, body))


def decode_bool(expr):
    """Try to decode a normalized expression as a boolean."""
    s = lambda_to_str(expr)
    # Check for true pattern
    if isinstance(expr, Lam) and isinstance(expr.body, Lam):
        if isinstance(expr.body.body, Var):
            if expr.body.body.name == expr.var.name:
                return "TRUE"
            if expr.body.body.name == expr.body.var.name:
                return "FALSE"
    return None


def decode_scott_num(expr, max_depth=32):
    """Try to decode a Scott-encoded binary number."""
    bits = []
    current = expr
    for _ in range(max_depth):
        # Check if nil (false)
        b = decode_bool(current)
        if b == "FALSE":
            break
        # Check if pair(bit, rest)
        if isinstance(current, Lam):
            body = current.body
            if isinstance(body, App) and isinstance(body.func, App):
                if isinstance(body.func.func, Var) and body.func.func.name == current.var.name:
                    bit_expr = body.func.arg
                    rest_expr = body.arg
                    bit_val = decode_bool(bit_expr)
                    if bit_val == "TRUE":
                        bits.append(1)
                    elif bit_val == "FALSE":
                        bits.append(0)
                    else:
                        return None
                    current = rest_expr
                    continue
        return None

    if not bits:
        return 0
    # LSB first
    n = 0
    for i, b in enumerate(bits):
        n += b << i
    return n


def test_apply(func_expr, args, name="func"):
    """Apply function to args and normalize."""
    global _fresh
    _fresh = 10000  # avoid variable name collisions

    expr = func_expr
    for arg in args:
        expr = App(expr, arg)

    result, steps, status = normalize_safe(expr, max_steps=10000, max_size=100000)
    result_str = lambda_to_str(result)

    if len(result_str) > 300:
        result_str = result_str[:300] + "..."

    # Try to decode
    b = decode_bool(result)
    n = decode_scott_num(result)

    decoded = ""
    if b:
        decoded = f" = {b}"
    elif n is not None:
        decoded = f" = NUMBER({n})"

    print(f"  {name}({', '.join(str(a)[:30] for a in args)}) = {result_str}{decoded}  [{steps} steps, {status}]")
    return result


def main():
    print("=== Step 4d: Evaluate Operators ===\n")

    base = os.path.join(os.path.dirname(__file__), '..')
    ext_dir = os.path.join(base, 'extracted')

    # Read and parse X
    with open(os.path.join(ext_dir, 'left_x.txt'), 'r') as f:
        x_compact = f.read().strip()

    x_tree = compact_to_ski_tree(x_compact)

    # Extract args using the same method
    from step4c_extract_operators import extract_s_spine_args, tree_to_compact
    x_args = extract_s_spine_args(x_tree)
    print(f"Extracted {len(x_args)} operators from X\n")

    # Convert each arg to lambda
    operators = []
    for i, arg in enumerate(x_args):
        compact = tree_to_compact(arg)
        reset_var_counter()
        lam = revert(arg)
        operators.append((compact, lam))
        lam_str = lambda_to_str(lam)
        print(f"Operator {i}: {len(compact)} compact chars, lambda={len(lam_str)} chars")

    # === Test arg3 (57 chars - smallest, easiest) ===
    print(f"\n{'='*60}")
    print(f"=== Testing arg3 (57 compact chars) ===")
    print(f"Lambda: {lambda_to_str(operators[3][1])}")
    print()

    op3 = operators[3][1]

    # Test with booleans
    test_apply(op3, [make_true(), Var("default")], "arg3(true, default)")
    test_apply(op3, [make_false(), Var("default")], "arg3(false, default)")

    # Test with scott numbers
    test_apply(op3, [make_scott_num(0), Var("default")], "arg3(0, default)")
    test_apply(op3, [make_scott_num(1), Var("default")], "arg3(1, default)")
    test_apply(op3, [make_scott_num(2), Var("default")], "arg3(2, default)")
    test_apply(op3, [make_scott_num(3), Var("default")], "arg3(3, default)")
    test_apply(op3, [make_scott_num(5), Var("default")], "arg3(5, default)")

    # Test with pair of specific values
    test_apply(op3, [make_pair(make_true(), Var("rest")), Var("default")],
               "arg3(pair(T, rest), default)")
    test_apply(op3, [make_pair(make_false(), Var("rest")), Var("default")],
               "arg3(pair(F, rest), default)")

    # === Test arg2 (243 chars - Y combinator applied) ===
    print(f"\n{'='*60}")
    print(f"=== Testing arg2 (243 compact chars) ===")
    lam_str = lambda_to_str(operators[2][1])
    print(f"Lambda: {lam_str}")
    print()

    op2 = operators[2][1]

    # This is Y(f) applied to false (nil)
    # Looks like a recursive list operation starting with nil accumulator
    # Test with simple lists
    test_apply(op2, [make_nil()], "arg2(nil)")
    test_apply(op2, [make_pair(Var("h1"), make_nil())], "arg2([h1])")
    test_apply(op2, [make_pair(Var("h1"), make_pair(Var("h2"), make_nil()))],
               "arg2([h1, h2])")

    # Test with scott numbers
    test_apply(op2, [make_scott_num(0)], "arg2(num 0)")
    test_apply(op2, [make_scott_num(1)], "arg2(num 1)")
    test_apply(op2, [make_scott_num(3)], "arg2(num 3)")

    # === Test arg0 (20K chars - large) ===
    print(f"\n{'='*60}")
    print(f"=== Testing arg0 (20K compact chars - partial) ===")
    print(f"  arg0 is too large for full evaluation")
    print(f"  Lambda prefix: {lambda_to_str(operators[0][1])[:200]}...")

    # Try applying to simple args
    op0 = operators[0][1]
    print()
    test_apply(op0, [make_scott_num(0), make_scott_num(0)], "arg0(0, 0)")
    test_apply(op0, [make_scott_num(1), make_scott_num(0)], "arg0(1, 0)")
    test_apply(op0, [make_scott_num(0), make_scott_num(1)], "arg0(0, 1)")
    test_apply(op0, [make_scott_num(1), make_scott_num(1)], "arg0(1, 1)")
    test_apply(op0, [make_scott_num(2), make_scott_num(1)], "arg0(2, 1)")
    test_apply(op0, [make_scott_num(3), make_scott_num(2)], "arg0(3, 2)")
    test_apply(op0, [make_scott_num(5), make_scott_num(3)], "arg0(5, 3)")

    # === Test arg1 (10K chars) ===
    print(f"\n{'='*60}")
    print(f"=== Testing arg1 (10K compact chars - partial) ===")
    op1 = operators[1][1]
    print(f"  Lambda prefix: {lambda_to_str(op1)[:200]}...")
    print()

    # arg1 is Y-combinator applied - try with simple inputs
    test_apply(op1, [make_scott_num(0), make_scott_num(0)], "arg1(0, 0)")
    test_apply(op1, [make_scott_num(1), make_scott_num(0)], "arg1(1, 0)")
    test_apply(op1, [make_scott_num(0), make_scott_num(1)], "arg1(0, 1)")
    test_apply(op1, [make_scott_num(1), make_scott_num(1)], "arg1(1, 1)")
    test_apply(op1, [make_scott_num(2), make_scott_num(3)], "arg1(2, 3)")
    test_apply(op1, [make_scott_num(5), make_scott_num(3)], "arg1(5, 3)")

    # === Test arg4 ===
    print(f"\n{'='*60}")
    print(f"=== Testing arg4 (30K compact chars - partial) ===")
    op4 = operators[4][1]
    print(f"  Lambda prefix: {lambda_to_str(op4)[:200]}...")
    print()

    # arg4 takes 1 argument then more
    test_apply(op4, [make_scott_num(0)], "arg4(0)")
    test_apply(op4, [make_scott_num(1)], "arg4(1)")
    test_apply(op4, [make_scott_num(2)], "arg4(2)")
    test_apply(op4, [make_true()], "arg4(true)")
    test_apply(op4, [make_false()], "arg4(false)")

    # Test with 2 args
    test_apply(op4, [make_scott_num(1), make_scott_num(0)], "arg4(1, 0)")
    test_apply(op4, [make_scott_num(0), make_scott_num(1)], "arg4(0, 1)")


if __name__ == '__main__':
    main()
