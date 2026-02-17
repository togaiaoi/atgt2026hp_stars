#!/usr/bin/env python3
"""
Step 3: Reverse bracket abstraction (SKI -> lambda expressions)

Converts SKI combinator expressions back to lambda calculus using
the revert/unbracket algorithm described in the hints.

Algorithm:
  unbracket(x, I)         => x
  unbracket(x, (K e))     => e
  unbracket(x, (S e1 e2)) => (unbracket(x, e1) unbracket(x, e2))
  unbracket(x, other)     => error

  revert(e):
    Try unbracket(x, e) for fresh variable x
    -> Success: lambda x. revert(result)
    -> Failure: if atom, return as-is
                if (e1 e2), return (revert(e1) revert(e2))
"""

import os
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


class UnbracketError(Exception):
    pass


# Lambda expression representation
class Var:
    """Variable reference"""
    __slots__ = ['name']
    def __init__(self, name):
        self.name = name
    def __repr__(self):
        return self.name
    def __eq__(self, other):
        return isinstance(other, Var) and self.name == other.name
    def __hash__(self):
        return hash(('Var', self.name))


class Lam:
    """Lambda abstraction: lambda var. body"""
    __slots__ = ['var', 'body']
    def __init__(self, var, body):
        self.var = var
        self.body = body
    def __repr__(self):
        return f"(lambda {self.var}. {self.body})"


class App:
    """Application: (func arg)"""
    __slots__ = ['func', 'arg']
    def __init__(self, func, arg):
        self.func = func
        self.arg = arg
    def __repr__(self):
        return f"({self.func} {self.arg})"


class Atom:
    """Primitive combinator (S, K, I) or other constant"""
    __slots__ = ['name']
    def __init__(self, name):
        self.name = name
    def __repr__(self):
        return self.name
    def __eq__(self, other):
        return isinstance(other, Atom) and self.name == other.name
    def __hash__(self):
        return hash(('Atom', self.name))


# Fresh variable counter
_var_counter = 0

def fresh_var():
    global _var_counter
    _var_counter += 1
    return Var(f"x{_var_counter}")

def reset_var_counter():
    global _var_counter
    _var_counter = 0


def compact_to_ski_tree(s):
    """Parse compact string into SKI tree (using App/Atom nodes)"""
    stack = []
    for c in s:
        if c == 'k':
            stack.append(Atom('S'))
        elif c == 'D':
            stack.append(Atom('I'))
        elif c == 'X':
            stack.append(Atom('K'))
        elif c == '-':
            y = stack.pop()
            x = stack.pop()
            stack.append(App(x, y))
    assert len(stack) == 1, f"Expected 1 on stack, got {len(stack)}"
    return stack[0]


def unbracket(x, expr):
    """
    Try to remove one level of bracket abstraction.
    unbracket(x, I)         => x
    unbracket(x, (K e))     => e
    unbracket(x, (S e1 e2)) => (unbracket(x,e1) unbracket(x,e2))
    """
    # Case 1: I => x
    if isinstance(expr, Atom) and expr.name == 'I':
        return x

    # Case 2: (K e) => e
    if isinstance(expr, App):
        if isinstance(expr.func, Atom) and expr.func.name == 'K':
            return expr.arg

        # Case 3: ((S e1) e2) => (unbracket(x,e1) unbracket(x,e2))
        if isinstance(expr.func, App):
            inner = expr.func
            if isinstance(inner.func, Atom) and inner.func.name == 'S':
                e1 = inner.arg
                e2 = expr.arg
                return App(unbracket(x, e1), unbracket(x, e2))

    raise UnbracketError(f"Cannot unbracket: {expr}")


def revert(expr, depth=0, max_depth=200):
    """
    Convert SKI expression to lambda expression.
    Tries unbracket first; if successful, wraps in lambda.
    If not, recurses into sub-expressions.
    """
    if depth > max_depth:
        return expr  # Bail out at extreme depth

    # Try to unbracket
    try:
        x = fresh_var()
        result = unbracket(x, expr)
        return Lam(x, revert(result, depth + 1, max_depth))
    except UnbracketError:
        pass

    # Cannot unbracket: recurse on sub-expressions
    if isinstance(expr, Atom):
        return expr
    if isinstance(expr, App):
        return App(revert(expr.func, depth + 1, max_depth),
                   revert(expr.arg, depth + 1, max_depth))
    if isinstance(expr, Lam):
        return Lam(expr.var, revert(expr.body, depth + 1, max_depth))

    return expr


def lambda_to_str(expr, parens=False):
    """Pretty-print lambda expression"""
    if isinstance(expr, Var):
        return expr.name
    if isinstance(expr, Atom):
        return expr.name
    if isinstance(expr, Lam):
        body_str = lambda_to_str(expr.body)
        s = f"\\{expr.var.name}. {body_str}"
        return f"({s})" if parens else s
    if isinstance(expr, App):
        func_str = lambda_to_str(expr.func)
        arg_str = lambda_to_str(expr.arg, parens=True)
        s = f"{func_str} {arg_str}"
        return f"({s})" if parens else s
    return str(expr)


def lambda_depth(expr):
    """Count the maximum nesting depth of a lambda expression"""
    if isinstance(expr, (Var, Atom)):
        return 0
    if isinstance(expr, Lam):
        return 1 + lambda_depth(expr.body)
    if isinstance(expr, App):
        return 1 + max(lambda_depth(expr.func), lambda_depth(expr.arg))
    return 0


def count_lambdas(expr):
    """Count total number of lambda abstractions"""
    if isinstance(expr, (Var, Atom)):
        return 0
    if isinstance(expr, Lam):
        return 1 + count_lambdas(expr.body)
    if isinstance(expr, App):
        return count_lambdas(expr.func) + count_lambdas(expr.arg)
    return 0


# Known operator patterns for identification
KNOWN_OPERATORS = {
    # Boolean values
    'true': 'kXX--D-',       # S(KK)I - or just K (extensionally equivalent)
    'false': 'XD-',          # KI
    'K': 'X',                # K alone

    # Basic combinators
    'I': 'D',                # I
    'S': 'k',                # S
}


def identify_known_patterns(compact_str):
    """Try to identify known operator patterns in a compact string"""
    matches = []
    for name, pattern in KNOWN_OPERATORS.items():
        if compact_str == pattern:
            matches.append(name)
    return matches


def test_revert():
    """Test reverse bracket abstraction on known examples"""
    print("Testing reverse bracket abstraction...")

    # Test 1: S(KK)I = true = lambda x. lambda y. x
    reset_var_counter()
    tree = compact_to_ski_tree('kXX--D-')
    result = revert(tree)
    s = lambda_to_str(result)
    print(f"  S(KK)I -> {s}")
    # Should be lambda x. lambda y. x

    # Test 2: KI = false = lambda x. lambda y. y
    reset_var_counter()
    tree = compact_to_ski_tree('XD-')
    result = revert(tree)
    s = lambda_to_str(result)
    print(f"  KI -> {s}")
    # Should be lambda x. lambda y. y

    # Test 3: I = identity = lambda x. x
    reset_var_counter()
    tree = compact_to_ski_tree('D')
    result = revert(tree)
    s = lambda_to_str(result)
    print(f"  I -> {s}")

    # Test 4: SII = lambda x. x x (self-application)
    reset_var_counter()
    tree = compact_to_ski_tree('kDD--')
    result = revert(tree)
    s = lambda_to_str(result)
    print(f"  SII -> {s}")

    # Test 5: S(S(KS)(S(KK)I))(KI) from the hint example
    # Should revert to lambda x. lambda y. x y
    reset_var_counter()
    tree = compact_to_ski_tree('kkXk--kXX--D---XD--')
    result = revert(tree)
    s = lambda_to_str(result)
    print(f"  S(S(KS)(S(KK)I))(KI) -> {s}")

    # Test 6: l.2 answer: kkD-XXD----XXD---
    # S((SI)(K(KI)))(K(KI))
    reset_var_counter()
    tree = compact_to_ski_tree('kkD-XXD----XXD---')
    result = revert(tree)
    s = lambda_to_str(result)
    print(f"  l.2 answer -> {s}")

    print("Tests complete.\n")


def process_file(compact_path, output_dir):
    """Process a compact file and convert to lambda expressions"""
    print(f"Reading {compact_path}...")
    with open(compact_path, 'r') as f:
        compact = f.read().strip()

    print(f"Compact string: {len(compact):,} chars")

    if len(compact) > 500000:
        print("WARNING: Large input. Reverse bracket abstraction may be slow.")
        print("Processing first 1000 chars as preview...")
        # For large inputs, we'd need iterative/streaming approach
        # For now, just show info
        return

    print("Parsing to SKI tree...")
    reset_var_counter()
    tree = compact_to_ski_tree(compact)

    print("Running reverse bracket abstraction...")
    result = revert(tree)

    lambda_str = lambda_to_str(result)
    depth = lambda_depth(result)
    num_lambdas = count_lambdas(result)

    print(f"Result:")
    print(f"  Depth: {depth}")
    print(f"  Lambda count: {num_lambdas}")
    if len(lambda_str) <= 2000:
        print(f"  Expression: {lambda_str}")
    else:
        print(f"  Expression (first 500 chars): {lambda_str[:500]}...")
        print(f"  Expression (last 200 chars): ...{lambda_str[-200:]}")

    # Save result
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(compact_path))[0]
        output_path = os.path.join(output_dir, f"{base}_lambda.txt")
        with open(output_path, 'w') as f:
            f.write(lambda_str)
        print(f"  Saved to {output_path}")

    return result


if __name__ == '__main__':
    print("=== Step 3: Reverse Bracket Abstraction ===\n")

    test_revert()

    # Process extracted data items
    items_dir = os.path.join(os.path.dirname(__file__), '..', 'extracted', 'data_items')
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'extracted', 'lambda')

    if os.path.exists(items_dir):
        print("Processing extracted data items...")
        os.makedirs(output_dir, exist_ok=True)

        for fname in sorted(os.listdir(items_dir)):
            if not fname.endswith('.txt'):
                continue
            fpath = os.path.join(items_dir, fname)
            with open(fpath, 'r') as f:
                content = f.read().strip()

            if len(content) > 10000:
                print(f"\n  {fname}: {len(content):,} chars (skipping - too large)")
                continue

            print(f"\n  {fname}: {len(content):,} chars")
            try:
                reset_var_counter()
                tree = compact_to_ski_tree(content)
                result = revert(tree)
                lambda_str = lambda_to_str(result)

                if len(lambda_str) <= 500:
                    print(f"    Lambda: {lambda_str}")
                else:
                    print(f"    Lambda ({len(lambda_str)} chars): {lambda_str[:200]}...")

                base = os.path.splitext(fname)[0]
                with open(os.path.join(output_dir, f"{base}_lambda.txt"), 'w') as f:
                    f.write(lambda_str)
            except Exception as e:
                print(f"    Error: {e}")

        # Process LEFT
        left_path = os.path.join(os.path.dirname(__file__), '..', 'extracted', 'left.txt')
        if os.path.exists(left_path):
            print(f"\n\nProcessing LEFT ({os.path.getsize(left_path):,} bytes)...")
            process_file(left_path, output_dir)
    else:
        print(f"Extracted data directory not found: {items_dir}")
        print("Run step2_parse_structure.py first.")

        # Allow direct file processing
        if len(sys.argv) > 1:
            process_file(sys.argv[1], output_dir)
