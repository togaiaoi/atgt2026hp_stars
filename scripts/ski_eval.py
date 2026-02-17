#!/usr/bin/env python3
"""
Lazy graph-reduction SKI combinator evaluator.

Uses thunks (mutable nodes) with sharing to avoid exponential blowup.
Supports S, K, I combinators and application.
"""

import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.setrecursionlimit(200000)


# Node types
APP = 0
S = 1
K = 2
I = 3
S1 = 4   # S applied to 1 arg
S2 = 5   # S applied to 2 args
K1 = 6   # K applied to 1 arg
IND = 7  # Indirection (for sharing/update)

class Node:
    """Mutable graph node for lazy evaluation."""
    __slots__ = ['tag', 'a', 'b']

    def __init__(self, tag, a=None, b=None):
        self.tag = tag
        self.a = a
        self.b = b

    def __repr__(self):
        if self.tag == S: return "S"
        if self.tag == K: return "K"
        if self.tag == I: return "I"
        if self.tag == APP: return f"({self.a} {self.b})"
        if self.tag == S1: return f"(S {self.a})"
        if self.tag == S2: return f"(S {self.a} {self.b})"
        if self.tag == K1: return f"(K {self.a})"
        if self.tag == IND: return f"[-> {self.a}]"
        return "?"


def make_s():
    return Node(S)

def make_k():
    return Node(K)

def make_i():
    return Node(I)

def make_app(f, x):
    return Node(APP, f, x)


def compact_to_graph(s):
    """Parse compact string to graph nodes."""
    stack = []
    for c in s:
        if c == 'k':
            stack.append(make_s())
        elif c == 'X':
            stack.append(make_k())
        elif c == 'D':
            stack.append(make_i())
        elif c == '-':
            y = stack.pop()
            x = stack.pop()
            stack.append(make_app(x, y))
    assert len(stack) == 1
    return stack[0]


def follow(node):
    """Follow indirection chain."""
    while node.tag == IND:
        node = node.a
    return node


def whnf(node, fuel=1000000):
    """Reduce to Weak Head Normal Form using iterative spine traversal."""
    steps = [0]

    def reduce(n):
        """Iterative WHNF reduction."""
        # Build spine (stack of application arguments)
        spine = []

        while True:
            if steps[0] >= fuel:
                return n

            n = follow(n)

            if n.tag == APP:
                spine.append(n)
                n = follow(n.a)
                continue

            if n.tag == I and len(spine) >= 1:
                # I x -> x
                steps[0] += 1
                app = spine.pop()
                x = follow(app.b)
                # Update app node to indirection
                app.tag = IND
                app.a = x
                n = x
                continue

            if n.tag == K and len(spine) >= 2:
                # K x y -> x
                steps[0] += 1
                app1 = spine.pop()  # K x
                app2 = spine.pop()  # (K x) y
                x = follow(app1.b)
                # Update app2 to indirection to x
                app2.tag = IND
                app2.a = x
                # Also update app1
                app1.tag = K1
                app1.a = x
                n = x
                continue

            if n.tag == K1 and len(spine) >= 1:
                # (K x) y -> x
                steps[0] += 1
                app = spine.pop()
                x = follow(n.a)
                app.tag = IND
                app.a = x
                n = x
                continue

            if n.tag == S and len(spine) >= 3:
                # S f g x -> f x (g x)
                steps[0] += 1
                app1 = spine.pop()  # S f
                app2 = spine.pop()  # (S f) g
                app3 = spine.pop()  # ((S f) g) x
                f = follow(app1.b)
                g = follow(app2.b)
                x = app3.b  # Don't follow - keep sharing
                fx = make_app(f, x)
                gx = make_app(g, x)
                result = make_app(fx, gx)
                # Update app3
                app3.tag = IND
                app3.a = result
                # Update partial apps
                app1.tag = S1
                app1.a = f
                app2.tag = S2
                app2.a = f
                app2.b = g
                n = result
                continue

            if n.tag == S1 and len(spine) >= 2:
                # (S f) g x -> f x (g x)
                steps[0] += 1
                app1 = spine.pop()  # (S f) g
                app2 = spine.pop()  # ((S f) g) x
                f = follow(n.a)
                g = follow(app1.b)
                x = app2.b
                fx = make_app(f, x)
                gx = make_app(g, x)
                result = make_app(fx, gx)
                app2.tag = IND
                app2.a = result
                app1.tag = S2
                app1.a = f
                app1.b = g
                n = result
                continue

            if n.tag == S2 and len(spine) >= 1:
                # (S f g) x -> f x (g x)
                steps[0] += 1
                app = spine.pop()
                f = follow(n.a)
                g = follow(n.b)
                x = app.b
                fx = make_app(f, x)
                gx = make_app(g, x)
                result = make_app(fx, gx)
                app.tag = IND
                app.a = result
                n = result
                continue

            # No reduction possible - return the outermost remaining node
            if spine:
                return spine[0]  # Root of WHNF application chain
            return n

    result = reduce(node)
    return result, steps[0]


def eval_app(func, arg, fuel=100000):
    """Apply func to arg, evaluate to WHNF, return result node and steps."""
    test = make_app(func, arg)
    _, steps = whnf(test, fuel)
    return follow(test), steps


def eval_apps(func, args, fuel=100000):
    """Apply func to multiple args, evaluate to WHNF."""
    node = func
    for arg in args:
        node = make_app(node, arg)
    _, steps = whnf(node, fuel)
    return follow(node), steps


def decode_bool_graph(node, fuel=100000):
    """Decode a boolean from graph node by applying to two unique markers."""
    # Use unique fresh nodes as markers (tag values that won't appear naturally)
    # We'll use special tag numbers to identify our markers
    marker_t = Node(99)  # Unique tag for true result
    marker_f = Node(98)  # Unique tag for false result

    test = make_app(make_app(node, marker_t), marker_f)
    _, steps = whnf(test, fuel)
    result = follow(test)

    if result.tag == 99:  # Got marker_t back
        return True
    if result.tag == 98:  # Got marker_f back
        return False
    return None


def decode_pair_fst(node, fuel=100000):
    """Extract first element: pair(K) = fst."""
    return eval_app(node, make_k(), fuel)


def decode_pair_snd(node, fuel=100000):
    """Extract second element: pair(KI) = snd."""
    ki = make_app(make_k(), make_i())
    return eval_app(node, ki, fuel)


def decode_scott_num_graph(node, fuel=500000):
    """Decode Scott-encoded binary number from graph.

    Numbers are encoded as pair chains:
    0 = false (KI)
    n = pair(bit0, pair(bit1, ... nil))
    where bit is true (1) or false (0), LSB first.
    """
    bits = []
    current = node
    total_steps = 0

    for _ in range(64):  # Max 64 bits
        if total_steps > fuel:
            break

        # First check: is current a boolean (nil/false)?
        is_nil = decode_bool_graph(current, fuel // 10)
        if is_nil is False:
            # It's false = nil = end of number
            break

        # Try to extract first (bit) and second (rest)
        first, steps1 = decode_pair_fst(current, fuel // 10)
        total_steps += steps1
        second, steps2 = decode_pair_snd(current, fuel // 10)
        total_steps += steps2

        # Decode the bit
        bit_val = decode_bool_graph(first, fuel // 10)

        if bit_val is True:
            bits.append(1)
            current = second
        elif bit_val is False:
            bits.append(0)
            current = second
        else:
            # Not a valid bit -> end of number or error
            break

    if not bits:
        return 0, total_steps

    n = 0
    for i, b in enumerate(bits):
        n += b << i
    return n, total_steps


def graph_to_compact(node, depth=0, max_depth=100):
    """Convert graph back to compact string (for debugging)."""
    if depth > max_depth:
        return "..."
    node = follow(node)
    if node.tag == S:
        return "k"
    if node.tag == K:
        return "X"
    if node.tag == I:
        return "D"
    if node.tag == APP:
        return graph_to_compact(node.a, depth+1, max_depth) + \
               graph_to_compact(node.b, depth+1, max_depth) + "-"
    if node.tag == K1:
        return "X" + graph_to_compact(node.a, depth+1, max_depth) + "-"
    if node.tag == S1:
        return "k" + graph_to_compact(node.a, depth+1, max_depth) + "-"
    if node.tag == S2:
        return "k" + graph_to_compact(node.a, depth+1, max_depth) + "-" + \
               graph_to_compact(node.b, depth+1, max_depth) + "-"
    return "?"


# === Convenience constructors ===

def make_true_g():
    """S(KK)I = true"""
    return compact_to_graph('kXX--D-')

def make_false_g():
    """KI = false"""
    return compact_to_graph('XD-')

def make_pair_g(a, b):
    """pair(a,b) = S(SI(Ka))(Kb)"""
    # S(S(I)(K a))(K b)
    # compact: kkD-X<a>---X<b>---  but we need to build it from nodes
    si_ka = make_app(make_app(make_s(), make_app(make_app(make_s(), make_i()), make_app(make_k(), a))),
                     make_app(make_k(), b))
    return si_ka

def make_nil_g():
    """nil = false"""
    return make_false_g()

def make_scott_num_g(n):
    """Make Scott-encoded binary number (LSB first)."""
    if n == 0:
        return make_nil_g()
    bits = []
    temp = n
    while temp > 0:
        bits.append(temp & 1)
        temp >>= 1
    result = make_nil_g()
    for bit in reversed(bits):
        result = make_pair_g(make_true_g() if bit else make_false_g(), result)
    return result


def test_basic():
    """Test basic SKI evaluation."""
    print("=== Basic SKI evaluation tests ===")

    # I(K) -> K
    n = make_app(make_i(), make_k())
    r, steps = whnf(n)
    r = follow(n)
    print(f"  I K = {r} ({steps} steps)")

    # K S I -> S
    n = make_app(make_app(make_k(), make_s()), make_i())
    r, steps = whnf(n)
    r = follow(n)
    print(f"  K S I = {r} ({steps} steps)")

    # S K K x -> x (for any x)
    x = make_i()
    n = make_app(make_app(make_app(make_s(), make_k()), make_k()), x)
    r, steps = whnf(n)
    r = follow(n)
    print(f"  S K K I = {r} ({steps} steps)")

    # true(A)(B) -> A
    n = make_app(make_app(make_true_g(), make_s()), make_i())
    _, steps = whnf(n)
    r = follow(n)
    print(f"  true(S)(I) = {r} (expect S) ({steps} steps)")

    # false(A)(B) -> B
    n = make_app(make_app(make_false_g(), make_s()), make_i())
    _, steps = whnf(n)
    r = follow(n)
    print(f"  false(S)(I) = {r} (expect I) ({steps} steps)")

    # pair(S, K)(true) -> S (fst)
    p = make_pair_g(make_s(), make_k())
    first, steps = decode_pair_fst(p)
    print(f"  pair(S,K) fst = {first} (expect S) ({steps} steps)")

    # pair(S, K)(false) -> K (snd)
    p2 = make_pair_g(make_s(), make_k())
    second, steps = decode_pair_snd(p2)
    print(f"  pair(S,K) snd = {second} (expect K) ({steps} steps)")

    # Boolean decode tests
    t = make_true_g()
    print(f"  decode_bool(true) = {decode_bool_graph(t)} (expect True)")
    f = make_false_g()
    print(f"  decode_bool(false) = {decode_bool_graph(f)} (expect False)")

    # Scott number tests
    for test_n in [0, 1, 2, 3, 5, 7, 10, 42]:
        num = make_scott_num_g(test_n)
        val, steps = decode_scott_num_graph(num)
        ok = "OK" if val == test_n else "FAIL"
        print(f"  decode(scott_num({test_n})) = {val} ({steps} steps) {ok}")

    print()


if __name__ == '__main__':
    test_basic()
