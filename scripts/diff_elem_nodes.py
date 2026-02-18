#!/usr/bin/env python3
"""
Analyze the structural differences between elem[0], elem[1], elem[2] of item_09.

Instead of serializing entire subtrees (which share most nodes), this script:
1. Extracts each element via pair_snd evaluation
2. Collects the set of reachable node indices for each element
3. Finds which nodes are unique to each element vs shared
4. Serializes ONLY the differing prefix (before the common tail)
"""

import sys
import os
import io
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ---- Inline arena-based SKI evaluator ----

APP = 0
TAG_S = 1
TAG_K = 2
TAG_I = 3
S1 = 4
S2 = 5
K1 = 6
IND = 7

TAG_NAMES = ['APP', 'S', 'K', 'I', 'S1', 'S2', 'K1', 'IND']

class Arena:
    def __init__(self, capacity=2_000_000):
        self.tag = bytearray(capacity)
        self.left = [0] * capacity
        self.right = [0] * capacity
        self.size = 0
        self.checkpoint_size = 0
        self.saved = {}

    def alloc(self, tag, a=0, b=0):
        idx = self.size
        if idx >= len(self.tag):
            new_cap = len(self.tag) * 2
            self.tag.extend(bytearray(new_cap - len(self.tag)))
            self.left.extend([0] * (new_cap - len(self.left)))
            self.right.extend([0] * (new_cap - len(self.right)))
        self.tag[idx] = tag
        self.left[idx] = a
        self.right[idx] = b
        self.size += 1
        return idx

    def follow(self, idx):
        while self.tag[idx] == IND:
            idx = self.left[idx]
        return idx

    def save_node(self, idx):
        if self.checkpoint_size > 0 and idx < self.checkpoint_size and idx not in self.saved:
            self.saved[idx] = (self.tag[idx], self.left[idx], self.right[idx])

    def set_checkpoint(self):
        self.checkpoint_size = self.size
        self.saved = {}

    def restore_checkpoint(self):
        for idx, (t, l, r) in self.saved.items():
            self.tag[idx] = t
            self.left[idx] = l
            self.right[idx] = r
        self.size = self.checkpoint_size
        self.saved = {}
        self.checkpoint_size = 0

    def whnf(self, idx, fuel):
        steps = 0
        spine = []
        n = idx
        while steps < fuel:
            n = self.follow(n)
            t = self.tag[n]
            if t == APP:
                spine.append(n)
                n = self.left[n]
                continue
            if t == TAG_I and len(spine) >= 1:
                steps += 1
                app = spine.pop()
                x = self.follow(self.right[app])
                self.save_node(app)
                self.tag[app] = IND
                self.left[app] = x
                n = x
                continue
            if t == TAG_K and len(spine) >= 2:
                steps += 1
                app1 = spine.pop()
                app2 = spine.pop()
                x = self.follow(self.right[app1])
                self.save_node(app2)
                self.tag[app2] = IND
                self.left[app2] = x
                self.save_node(app1)
                self.tag[app1] = K1
                self.left[app1] = x
                n = x
                continue
            if t == K1 and len(spine) >= 1:
                steps += 1
                app = spine.pop()
                x = self.follow(self.left[n])
                self.save_node(app)
                self.tag[app] = IND
                self.left[app] = x
                n = x
                continue
            if t == TAG_S and len(spine) >= 3:
                steps += 1
                app1 = spine.pop()
                app2 = spine.pop()
                app3 = spine.pop()
                f = self.follow(self.right[app1])
                g = self.follow(self.right[app2])
                x = self.right[app3]
                fx = self.alloc(APP, f, x)
                gx = self.alloc(APP, g, x)
                result = self.alloc(APP, fx, gx)
                self.save_node(app3)
                self.tag[app3] = IND
                self.left[app3] = result
                self.save_node(app1)
                self.tag[app1] = S1
                self.left[app1] = f
                self.save_node(app2)
                self.tag[app2] = S2
                self.left[app2] = f
                self.right[app2] = g
                n = result
                continue
            if t == S1 and len(spine) >= 2:
                steps += 1
                app1 = spine.pop()
                app2 = spine.pop()
                f = self.follow(self.left[n])
                g = self.follow(self.right[app1])
                x = self.right[app2]
                fx = self.alloc(APP, f, x)
                gx = self.alloc(APP, g, x)
                result = self.alloc(APP, fx, gx)
                self.save_node(app2)
                self.tag[app2] = IND
                self.left[app2] = result
                self.save_node(app1)
                self.tag[app1] = S2
                self.left[app1] = f
                self.right[app1] = g
                n = result
                continue
            if t == S2 and len(spine) >= 1:
                steps += 1
                app = spine.pop()
                f = self.follow(self.left[n])
                g = self.follow(self.right[n])
                x = self.right[app]
                fx = self.alloc(APP, f, x)
                gx = self.alloc(APP, g, x)
                result = self.alloc(APP, fx, gx)
                self.save_node(app)
                self.tag[app] = IND
                self.left[app] = result
                n = result
                continue
            if spine:
                return spine[0], steps
            return n, steps
        if spine:
            return spine[0], steps
        return n, steps


def parse_compact(arena, text):
    stack = []
    for ch in text:
        if ch == 'k':
            stack.append(arena.alloc(TAG_S))
        elif ch == 'X':
            stack.append(arena.alloc(TAG_K))
        elif ch == 'D':
            stack.append(arena.alloc(TAG_I))
        elif ch == '-':
            arg = stack.pop()
            func = stack.pop()
            stack.append(arena.alloc(APP, func, arg))
    assert len(stack) == 1
    return stack[0]


def make_false(arena):
    return arena.alloc(APP, arena.alloc(TAG_K), arena.alloc(TAG_I))


def pair_snd(arena, node, fuel=50_000_000):
    ki = arena.alloc(APP, arena.alloc(TAG_K), arena.alloc(TAG_I))
    dummy = make_false(arena)
    app1 = arena.alloc(APP, node, ki)
    app2 = arena.alloc(APP, app1, dummy)
    _, steps = arena.whnf(app2, fuel)
    return arena.follow(app2), steps


def pair_fst(arena, node, fuel=50_000_000):
    k = arena.alloc(TAG_K)
    dummy = make_false(arena)
    app1 = arena.alloc(APP, node, k)
    app2 = arena.alloc(APP, app1, dummy)
    _, steps = arena.whnf(app2, fuel)
    return arena.follow(app2), steps


def decode_bool(arena, node, fuel=10_000_000):
    marker_t = arena.alloc(99)
    marker_f = arena.alloc(98)
    app1 = arena.alloc(APP, node, marker_t)
    app2 = arena.alloc(APP, app1, marker_f)
    _, steps = arena.whnf(app2, fuel)
    result = arena.follow(app2)
    t = arena.tag[result]
    if t == 99: return True
    if t == 98: return False
    return None


def collect_reachable(arena, root, max_nodes=500000):
    """Collect all reachable node indices from root."""
    visited = set()
    stack = [arena.follow(root)]
    while stack and len(visited) < max_nodes:
        idx = stack.pop()
        idx = arena.follow(idx)
        if idx in visited:
            continue
        visited.add(idx)
        t = arena.tag[idx]
        if t == APP:
            stack.append(arena.left[idx])
            stack.append(arena.right[idx])
        elif t in (K1, S1):
            stack.append(arena.left[idx])
        elif t == S2:
            stack.append(arena.left[idx])
            stack.append(arena.right[idx])
    return visited


def describe(arena, idx, depth=3):
    idx = arena.follow(idx)
    if depth <= 0: return '...'
    t = arena.tag[idx]
    if t == TAG_S: return 'S'
    if t == TAG_K: return 'K'
    if t == TAG_I: return 'I'
    if t == APP:
        return f'({describe(arena, arena.left[idx], depth-1)} {describe(arena, arena.right[idx], depth-1)})'
    if t == K1: return f'K1({describe(arena, arena.left[idx], depth-1)})'
    if t == S1: return f'S1({describe(arena, arena.left[idx], depth-1)})'
    if t == S2: return f'S2({describe(arena, arena.left[idx], depth-1)}, {describe(arena, arena.right[idx], depth-1)})'
    if t == IND: return describe(arena, arena.left[idx], depth)
    return f'?{t}'


def structural_hash(arena, idx, memo=None, depth=0, max_depth=50):
    """Compute a structural hash for a subtree to identify identical structure."""
    if memo is None:
        memo = {}
    idx = arena.follow(idx)
    if idx in memo:
        return memo[idx]
    if depth >= max_depth:
        return hash(('deep', idx))

    t = arena.tag[idx]
    if t == TAG_S:
        h = hash(('S',))
    elif t == TAG_K:
        h = hash(('K',))
    elif t == TAG_I:
        h = hash(('I',))
    elif t == APP:
        lh = structural_hash(arena, arena.left[idx], memo, depth+1, max_depth)
        rh = structural_hash(arena, arena.right[idx], memo, depth+1, max_depth)
        h = hash(('APP', lh, rh))
    elif t == K1:
        lh = structural_hash(arena, arena.left[idx], memo, depth+1, max_depth)
        h = hash(('K1', lh))
    elif t == S1:
        lh = structural_hash(arena, arena.left[idx], memo, depth+1, max_depth)
        h = hash(('S1', lh))
    elif t == S2:
        lh = structural_hash(arena, arena.left[idx], memo, depth+1, max_depth)
        rh = structural_hash(arena, arena.right[idx], memo, depth+1, max_depth)
        h = hash(('S2', lh, rh))
    else:
        h = hash(('?', t, idx))

    memo[idx] = h
    return h


def tree_diff(arena, idx_a, idx_b, depth=0, max_depth=30, path=""):
    """Find structural differences between two subtrees. Returns list of (path, desc_a, desc_b)."""
    idx_a = arena.follow(idx_a)
    idx_b = arena.follow(idx_b)

    if idx_a == idx_b:
        return []  # Same node = identical

    if depth >= max_depth:
        return [(path, f'...@{idx_a}', f'...@{idx_b}')]

    ta = arena.tag[idx_a]
    tb = arena.tag[idx_b]

    # Different tags
    if ta != tb:
        return [(path, f'{TAG_NAMES[ta] if ta < 8 else "?"}@{idx_a}', f'{TAG_NAMES[tb] if tb < 8 else "?"}@{idx_b}')]

    if ta in (TAG_S, TAG_K, TAG_I):
        return []  # Same primitive

    diffs = []
    if ta == APP:
        diffs.extend(tree_diff(arena, arena.left[idx_a], arena.left[idx_b], depth+1, max_depth, path+".L"))
        diffs.extend(tree_diff(arena, arena.right[idx_a], arena.right[idx_b], depth+1, max_depth, path+".R"))
    elif ta in (K1, S1):
        diffs.extend(tree_diff(arena, arena.left[idx_a], arena.left[idx_b], depth+1, max_depth, path+".a"))
    elif ta == S2:
        diffs.extend(tree_diff(arena, arena.left[idx_a], arena.left[idx_b], depth+1, max_depth, path+".f"))
        diffs.extend(tree_diff(arena, arena.right[idx_a], arena.right[idx_b], depth+1, max_depth, path+".g"))

    return diffs


def main():
    input_file = r"d:\github\atgt2026hp_stars\extracted\data_items\item_09.txt"
    output_dir = r"d:\github\atgt2026hp_stars\extracted\data_items"

    print(f"Reading {input_file}...")
    with open(input_file, 'r') as f:
        text = f.read().strip()
    print(f"  Length: {len(text)} characters")

    print("Parsing into arena...")
    arena = Arena(capacity=len(text) * 4)
    root = parse_compact(arena, text)
    base_size = arena.size
    print(f"  Root: {root}, Arena size: {base_size}")

    # Extract elements (without checkpoint to keep results persistent)
    print("\nExtracting list elements...")
    elem_roots = []
    current = root

    for elem_idx in range(10):
        snd_idx, snd_steps = pair_snd(arena, current, fuel=100_000_000)
        snd_tag = TAG_NAMES[arena.tag[snd_idx]] if arena.tag[snd_idx] < 8 else f'?{arena.tag[snd_idx]}'
        print(f"  elem[{elem_idx}]: snd_idx={snd_idx}, tag={snd_tag}, steps={snd_steps}")
        print(f"    describe(4): {describe(arena, snd_idx, 4)}")
        elem_roots.append(snd_idx)

        fst_idx, fst_steps = pair_fst(arena, current, fuel=100_000_000)
        rest_bool = decode_bool(arena, fst_idx, fuel=10_000_000)
        print(f"    rest: idx={fst_idx}, bool={rest_bool}")

        if rest_bool is False:
            print(f"  -> End of list")
            break

        current = arena.follow(fst_idx)

    # Analyze reachable node sets for elem[0..2]
    if len(elem_roots) >= 3:
        print("\n" + "=" * 80)
        print("NODE SET ANALYSIS for elem[0], elem[1], elem[2]")
        print("=" * 80)

        sets = []
        for i in range(3):
            nodes = collect_reachable(arena, elem_roots[i])
            sets.append(nodes)
            # Count how many are in original base vs new
            in_base = sum(1 for n in nodes if n < base_size)
            in_new = sum(1 for n in nodes if n >= base_size)
            print(f"  elem[{i}]: {len(nodes)} reachable nodes ({in_base} in base, {in_new} new)")

        # Set operations
        common_01 = sets[0] & sets[1]
        common_02 = sets[0] & sets[2]
        common_12 = sets[1] & sets[2]
        common_all = sets[0] & sets[1] & sets[2]
        only_0 = sets[0] - sets[1] - sets[2]
        only_1 = sets[1] - sets[0] - sets[2]
        only_2 = sets[2] - sets[0] - sets[1]

        print(f"\n  Common to all 3: {len(common_all)} nodes")
        print(f"  Common 0&1: {len(common_01)}, 0&2: {len(common_02)}, 1&2: {len(common_12)}")
        print(f"  Only in elem[0]: {len(only_0)} nodes")
        print(f"  Only in elem[1]: {len(only_1)} nodes")
        print(f"  Only in elem[2]: {len(only_2)} nodes")

        # Show unique nodes
        for i, only_set in enumerate([only_0, only_1, only_2]):
            if only_set:
                print(f"\n  Unique to elem[{i}] ({len(only_set)} nodes):")
                sorted_nodes = sorted(only_set)
                for n in sorted_nodes[:20]:
                    t = arena.tag[n]
                    tn = TAG_NAMES[t] if t < 8 else f'?{t}'
                    if t == APP:
                        print(f"    node {n}: {tn}(L={arena.left[n]}, R={arena.right[n]}) desc: {describe(arena, n, 3)}")
                    elif t in (K1, S1):
                        print(f"    node {n}: {tn}(a={arena.left[n]}) desc: {describe(arena, n, 3)}")
                    elif t == S2:
                        print(f"    node {n}: {tn}(f={arena.left[n]}, g={arena.right[n]}) desc: {describe(arena, n, 3)}")
                    else:
                        print(f"    node {n}: {tn}")
                if len(sorted_nodes) > 20:
                    print(f"    ... and {len(sorted_nodes) - 20} more")

        # Tree structural diff
        print("\n" + "=" * 80)
        print("STRUCTURAL TREE DIFF")
        print("=" * 80)

        pairs = [(0, 1), (0, 2), (1, 2)]
        for a, b in pairs:
            print(f"\n--- elem[{a}] vs elem[{b}] ---")
            diffs = tree_diff(arena, elem_roots[a], elem_roots[b], max_depth=40)
            print(f"  {len(diffs)} differences found (at max_depth=40)")
            for i, (path, da, db) in enumerate(diffs[:50]):
                print(f"  diff #{i}: path={path}")
                print(f"    [{a}]: {da}")
                print(f"    [{b}]: {db}")
            if len(diffs) > 50:
                print(f"  ... and {len(diffs) - 50} more")

        # Also check: structural hashes at various depths
        print("\n" + "=" * 80)
        print("STRUCTURAL HASH COMPARISON")
        print("=" * 80)
        for depth in [5, 10, 20, 30]:
            hashes = []
            for i in range(3):
                h = structural_hash(arena, elem_roots[i], memo={}, max_depth=depth)
                hashes.append(h)
            same_01 = hashes[0] == hashes[1]
            same_02 = hashes[0] == hashes[2]
            same_12 = hashes[1] == hashes[2]
            print(f"  depth {depth}: 0==1: {same_01}, 0==2: {same_02}, 1==2: {same_12}")


if __name__ == '__main__':
    main()
