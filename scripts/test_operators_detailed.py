#!/usr/bin/env python3
"""Detailed testing of arg0-arg4 operators to identify their exact behavior."""
import sys
sys.path.insert(0, 'scripts')
from pathlib import Path
from ski_eval import (compact_to_graph, make_scott_num_g, make_app, whnf, follow,
                      decode_bool_graph, decode_scott_num_graph, graph_to_compact,
                      make_false_g, make_true_g, make_k, make_i, make_s)

# Load operators
comp = []
for i in range(5):
    txt = Path(f'extracted/x_arg{i}.txt').read_text(encoding='utf-8')
    c = txt.split('compact: ', 1)[1].split('\n', 1)[0].strip()
    comp.append(c)
    print(f"arg{i}: {len(c)} chars")

def run2(op_compact, a, b, fuel=300000):
    """Test a 2-arg operator with scott-encoded numbers."""
    expr = compact_to_graph(op_compact)
    expr = make_app(expr, make_scott_num_g(a))
    expr = make_app(expr, make_scott_num_g(b))
    _, st = whnf(expr, fuel)
    r = follow(expr)
    bval = decode_bool_graph(r, 10000)
    n, _ = decode_scott_num_graph(r, 30000)
    pref = graph_to_compact(r, max_depth=40)
    return st, bval, n, pref

def run1(op_compact, a, fuel=300000):
    """Test a 1-arg operator."""
    expr = compact_to_graph(op_compact)
    expr = make_app(expr, make_scott_num_g(a))
    _, st = whnf(expr, fuel)
    r = follow(expr)
    bval = decode_bool_graph(r, 10000)
    n, _ = decode_scott_num_graph(r, 30000)
    pref = graph_to_compact(r, max_depth=40)
    return st, bval, n, pref

def fmt_result(bval, n, pref):
    if bval is True: return f"TRUE"
    if bval is False: return f"FALSE"
    if n is not None and len(pref) < 15: return f"NUM({n})"
    return f"?({pref[:20]})"

# ========== arg1: extensive truth table ==========
print("\n=== arg1(a, b) truth table ===")
print("     ", end="")
for b in range(9):
    print(f"{b:>7}", end="")
print()
for a in range(9):
    print(f"a={a}: ", end="")
    for b in range(9):
        try:
            st, bv, n, p = run2(comp[1], a, b, 200000)
            print(f"{fmt_result(bv, n, p):>7}", end="")
        except:
            print(f"  ERR  ", end="")
    print()

# ========== arg0: extensive truth table ==========
print("\n=== arg0(a, b) truth table ===")
print("     ", end="")
for b in range(9):
    print(f"{b:>7}", end="")
print()
for a in range(9):
    print(f"a={a}: ", end="")
    for b in range(9):
        try:
            st, bv, n, p = run2(comp[0], a, b, 200000)
            print(f"{fmt_result(bv, n, p):>7}", end="")
        except:
            print(f"  ERR  ", end="")
    print()

# ========== arg2: 1-arg function ==========
print("\n=== arg2(n) ===")
for a in range(16):
    try:
        st, bv, n, p = run1(comp[2], a, 200000)
        print(f"  arg2({a:2d}) = {fmt_result(bv, n, p):>12}  steps={st}")
    except Exception as e:
        print(f"  arg2({a:2d}) = ERR: {e}")

# ========== arg3: already known, but let's verify ==========
print("\n=== arg3(a, b) ===")
for a in range(5):
    for b in range(5):
        try:
            st, bv, n, p = run2(comp[3], a, b, 100000)
            print(f"  arg3({a},{b}) = {fmt_result(bv, n, p):>12}  steps={st}")
        except:
            print(f"  arg3({a},{b}) = ERR")

# ========== arg4: sample points ==========
print("\n=== arg4(a, b) sample ===")
for a in range(6):
    for b in range(6):
        try:
            st, bv, n, p = run2(comp[4], a, b, 500000)
            print(f"  arg4({a},{b}) = {fmt_result(bv, n, p):>12}  steps={st}")
        except:
            print(f"  arg4({a},{b}) = ERR")
