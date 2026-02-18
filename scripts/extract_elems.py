#!/usr/bin/env python3
"""
Extract individual list elements from item_09.txt using SKI evaluation.

item_09 is a pair2-encoded list. To extract elements, we need to evaluate:
  pair_fst(node) = node(K)(dummy)  -- gets the rest of the list
  pair_snd(node) = node(KI)(dummy) -- gets the current value

After extraction, serialize each element back to compact SKI notation
and compare elem[0], elem[1], elem[2].
"""

import sys
import os
import io
import gc
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.setrecursionlimit(500000)

# ---- Inline a minimal SKI evaluator (arena-based for speed) ----

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
    """Array-based arena for SKI graph nodes."""
    def __init__(self, capacity=2_000_000):
        self.tag = bytearray(capacity)
        self.left = [0] * capacity   # func/a field
        self.right = [0] * capacity  # arg/b field
        self.size = 0
        self.checkpoint_size = 0
        self.saved = {}  # idx -> (tag, left, right)

    def alloc(self, tag, a=0, b=0):
        idx = self.size
        if idx >= len(self.tag):
            # Grow
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
        """Save node state if within checkpoint base."""
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
        """Reduce to WHNF. Returns (result_idx, steps_used)."""
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

            # No reduction
            if spine:
                return spine[0], steps
            return n, steps

        # Out of fuel
        if spine:
            return spine[0], steps
        return n, steps


def parse_compact(arena, text):
    """Parse compact SKI notation into arena."""
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
    assert len(stack) == 1, f"Parse error: stack has {len(stack)} elements"
    return stack[0]


def make_true(arena):
    """true = S(KK)(I)"""
    kk = arena.alloc(APP, arena.alloc(TAG_K), arena.alloc(TAG_K))
    skk = arena.alloc(APP, arena.alloc(TAG_S), kk)
    return arena.alloc(APP, skk, arena.alloc(TAG_I))


def make_false(arena):
    """false = KI"""
    return arena.alloc(APP, arena.alloc(TAG_K), arena.alloc(TAG_I))


def pair_fst(arena, node, fuel=50_000_000):
    """pair2(a,b)(K)(dummy) = K(a)(b) = a"""
    k = arena.alloc(TAG_K)
    dummy = make_false(arena)
    app1 = arena.alloc(APP, node, k)
    app2 = arena.alloc(APP, app1, dummy)
    result_idx, steps = arena.whnf(app2, fuel)
    return arena.follow(app2), steps


def pair_snd(arena, node, fuel=50_000_000):
    """pair2(a,b)(KI)(dummy) = KI(a)(b) = I(b) = b"""
    ki = arena.alloc(APP, arena.alloc(TAG_K), arena.alloc(TAG_I))
    dummy = make_false(arena)
    app1 = arena.alloc(APP, node, ki)
    app2 = arena.alloc(APP, app1, dummy)
    result_idx, steps = arena.whnf(app2, fuel)
    return arena.follow(app2), steps


def decode_bool(arena, node, fuel=10_000_000):
    """Decode boolean by applying to two markers."""
    # true(x)(y) = x, false(x)(y) = y
    marker_t = arena.alloc(99)  # unique tag
    marker_f = arena.alloc(98)  # unique tag
    app1 = arena.alloc(APP, node, marker_t)
    app2 = arena.alloc(APP, app1, marker_f)
    _, steps = arena.whnf(app2, fuel)
    result = arena.follow(app2)
    t = arena.tag[result]
    if t == 99:
        return True
    if t == 98:
        return False
    return None


def serialize_compact(arena, root):
    """Serialize arena subtree to compact notation (iterative)."""
    result = []
    # Post-order traversal using stack
    stack = [(root, False)]

    while stack:
        idx, visited = stack.pop()
        idx = arena.follow(idx)
        t = arena.tag[idx]

        if t == TAG_S:
            result.append('k')
        elif t == TAG_K:
            result.append('X')
        elif t == TAG_I:
            result.append('D')
        elif t == APP:
            if not visited:
                stack.append((idx, True))
                stack.append((arena.right[idx], False))
                stack.append((arena.left[idx], False))
            else:
                result.append('-')
        elif t == K1:
            # K1(x) = App(K, x)
            if not visited:
                stack.append((idx, True))
                stack.append((arena.left[idx], False))  # x
            else:
                result.append('-')  # K applied to x
                result.insert(len(result) - 1, 'X')  # K before x...
                # Actually this is wrong, we need proper ordering
                # K1(x) should serialize as Xx- (= App(K, x))
                # Let me fix this by handling K1 properly
                pass
        elif t == S1:
            # S1(f) = App(S, f)
            if not visited:
                stack.append((idx, True))
                stack.append((arena.left[idx], False))  # f
            else:
                result.append('-')
        elif t == S2:
            # S2(f, g) = App(App(S, f), g)
            if not visited:
                stack.append((idx, True))
                stack.append((arena.right[idx], False))  # g
                stack.append((arena.left[idx], False))   # f  -- but we also need S and inner App
            else:
                result.append('-')
        elif t == IND:
            # Should be followed already
            pass
        else:
            result.append(f'?{t}')

    return ''.join(result)


def serialize_compact_v2(arena, root):
    """Serialize arena subtree to compact notation (iterative, handles K1/S1/S2)."""
    # Build a virtual tree and serialize it
    # To handle K1, S1, S2, we expand them:
    # K1(x) -> App(K, x) -> Xx-
    # S1(f) -> App(S, f) -> kf-
    # S2(f, g) -> App(App(S, f), g) -> kf-g-

    result = []
    # Stack: (idx, phase)
    # For APP: phase 0 = push children, phase 1 = emit '-'
    # For K1: expand to K then x
    # For S1: expand to S then f
    # For S2: expand to S, f, g (two apps)

    stack = [(arena.follow(root), 0)]

    while stack:
        idx, phase = stack.pop()
        idx = arena.follow(idx)
        t = arena.tag[idx]

        if t == TAG_S:
            result.append('k')
        elif t == TAG_K:
            result.append('X')
        elif t == TAG_I:
            result.append('D')
        elif t == APP:
            if phase == 0:
                stack.append((idx, 1))  # will emit '-'
                stack.append((arena.right[idx], 0))
                stack.append((arena.left[idx], 0))
            else:
                result.append('-')
        elif t == K1:
            # K1(x) = App(K, x)
            # Serialize as: [K] [x] -
            if phase == 0:
                stack.append((idx, 1))  # emit '-'
                stack.append((arena.left[idx], 0))  # x
                # emit K directly
                result.append('X')
            else:
                result.append('-')
        elif t == S1:
            # S1(f) = App(S, f)
            if phase == 0:
                stack.append((idx, 1))
                stack.append((arena.left[idx], 0))  # f
                result.append('k')
            else:
                result.append('-')
        elif t == S2:
            # S2(f, g) = App(App(S, f), g)
            # Serialize as: k [f] - [g] -
            if phase == 0:
                stack.append((idx, 1))  # outer '-'
                stack.append((arena.right[idx], 0))  # g
                # Inner: App(S, f) -> k [f] -
                stack.append((-1, 2))  # sentinel for inner '-'
                stack.append((arena.left[idx], 0))  # f
                result.append('k')
            elif phase == 1:
                result.append('-')
            elif phase == 2:
                result.append('-')
        elif t == IND:
            # Should have been followed
            stack.append((arena.left[idx], 0))
        else:
            result.append(f'?{t}')

    return ''.join(result)


def count_nodes(arena, root):
    """Count reachable nodes from root."""
    visited = set()
    stack = [arena.follow(root)]
    while stack:
        idx = stack.pop()
        idx = arena.follow(idx)
        if idx in visited:
            continue
        visited.add(idx)
        t = arena.tag[idx]
        if t == APP:
            stack.append(arena.left[idx])
            stack.append(arena.right[idx])
        elif t == K1 or t == S1:
            stack.append(arena.left[idx])
        elif t == S2:
            stack.append(arena.left[idx])
            stack.append(arena.right[idx])
    return len(visited)


def describe(arena, idx, depth=3):
    """Describe node structure."""
    idx = arena.follow(idx)
    if depth <= 0:
        return '...'
    t = arena.tag[idx]
    if t == TAG_S: return 'S'
    if t == TAG_K: return 'K'
    if t == TAG_I: return 'I'
    if t == APP:
        return f'({describe(arena, arena.left[idx], depth-1)} {describe(arena, arena.right[idx], depth-1)})'
    if t == K1:
        return f'K1({describe(arena, arena.left[idx], depth-1)})'
    if t == S1:
        return f'S1({describe(arena, arena.left[idx], depth-1)})'
    if t == S2:
        return f'S2({describe(arena, arena.left[idx], depth-1)}, {describe(arena, arena.right[idx], depth-1)})'
    if t == IND:
        return describe(arena, arena.left[idx], depth)
    return f'?{t}'


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
    print(f"  Root describe: {describe(arena, root, 4)}")

    # Extract list elements using evaluation
    print("\nExtracting list elements (pair2 via evaluation)...")
    elements = []  # List of (element_idx, snd_steps, fst_steps)
    current = root

    for elem_idx in range(10):  # Max 10 elements
        print(f"\n  --- elem[{elem_idx}] ---")

        # Extract pair_snd (value)
        arena.set_checkpoint()
        t0 = time.time()
        snd_idx, snd_steps = pair_snd(arena, current, fuel=100_000_000)
        t1 = time.time()
        snd_tag = TAG_NAMES[arena.tag[snd_idx]] if arena.tag[snd_idx] < len(TAG_NAMES) else f'?{arena.tag[snd_idx]}'
        print(f"    pair_snd: tag={snd_tag}, steps={snd_steps}, time={t1-t0:.2f}s")
        print(f"    describe: {describe(arena, snd_idx, 3)}")

        # Check if value is bool
        vbool = decode_bool(arena, snd_idx, fuel=10_000_000)
        print(f"    as bool: {vbool}")

        # Serialize the value subtree
        compact_snd = serialize_compact_v2(arena, snd_idx)
        print(f"    compact length: {len(compact_snd)}")

        # Save to file
        outpath = os.path.join(output_dir, f"elem_{elem_idx}.txt")
        with open(outpath, 'w') as f:
            f.write(compact_snd)
        print(f"    Saved to {outpath}")

        elements.append(compact_snd)
        arena.restore_checkpoint()

        # Extract pair_fst (rest of list)
        arena.set_checkpoint()
        fst_idx, fst_steps = pair_fst(arena, current, fuel=100_000_000)
        fst_tag = TAG_NAMES[arena.tag[fst_idx]] if arena.tag[fst_idx] < len(TAG_NAMES) else f'?{arena.tag[fst_idx]}'
        print(f"    pair_fst (rest): tag={fst_tag}, steps={fst_steps}")
        print(f"    describe: {describe(arena, fst_idx, 3)}")

        # Check if rest is nil (false = KI)
        rest_bool = decode_bool(arena, fst_idx, fuel=10_000_000)
        print(f"    rest as bool: {rest_bool}")

        arena.restore_checkpoint()

        if rest_bool is False:
            print(f"  -> End of list at elem[{elem_idx}]")
            break

        # For next iteration, we need persistent pair_fst result
        # Don't use checkpoint for this extraction
        fst_idx2, _ = pair_fst(arena, current, fuel=100_000_000)
        current = arena.follow(fst_idx2)
        print(f"    -> continuing with rest node {current}")

    # Diff elements
    if len(elements) >= 3:
        print("\n" + "=" * 80)
        print("DIFFING elem[0], elem[1], elem[2]")
        print("=" * 80)

        for i in range(min(3, len(elements))):
            print(f"  elem[{i}]: {len(elements[i])} chars")

        # Pairwise diffs
        pairs = [(0, 1), (0, 2), (1, 2)]
        for a, b in pairs:
            if a >= len(elements) or b >= len(elements):
                continue
            print(f"\n--- elem[{a}] vs elem[{b}] ---")
            sa, sb = elements[a], elements[b]
            min_len = min(len(sa), len(sb))
            max_len = max(len(sa), len(sb))

            diffs = []
            for pos in range(min_len):
                if sa[pos] != sb[pos]:
                    diffs.append(pos)

            if len(sa) != len(sb):
                print(f"  Length difference: {len(sa)} vs {len(sb)} (diff={abs(len(sa)-len(sb))})")

            if not diffs and len(sa) == len(sb):
                print("  IDENTICAL!")
                continue

            print(f"  {len(diffs)} differing positions (out of {min_len} common chars)")
            if len(sa) != len(sb):
                print(f"  + {max_len - min_len} chars extra in the longer one")

            # Show first 30 differences with context
            for idx_d, pos in enumerate(diffs[:30]):
                ctx_start = max(0, pos - 10)
                ctx_end = min(min_len, pos + 11)
                ctx_a = sa[ctx_start:ctx_end]
                ctx_b = sb[ctx_start:ctx_end]
                local_pos = pos - ctx_start
                marker = ' ' * local_pos + '^'
                print(f"  diff #{idx_d} at pos {pos}:")
                print(f"    [{a}]: {ctx_a}")
                print(f"    [{b}]: {ctx_b}")
                print(f"         {marker}")

            if len(diffs) > 30:
                print(f"  ... and {len(diffs) - 30} more differences")

        # 3-way comparison
        print(f"\n--- 3-way comparison ---")
        compacts = elements[:3]
        max_len = max(len(c) for c in compacts)
        padded = [c.ljust(max_len, '\0') for c in compacts]

        diff_positions = []
        for pos in range(max_len):
            chars = [p[pos] for p in padded]
            if len(set(chars)) > 1:
                diff_positions.append((pos, chars))

        print(f"Total positions with any difference: {len(diff_positions)}")

        if diff_positions:
            # Group consecutive diff regions
            regions = []
            start = diff_positions[0][0]
            end = start
            for pos, chars in diff_positions[1:]:
                if pos <= end + 5:
                    end = pos
                else:
                    regions.append((start, end))
                    start = pos
                    end = pos
            regions.append((start, end))

            print(f"Diff regions (merged within 5 chars): {len(regions)}")
            for i, (rs, re) in enumerate(regions[:20]):
                print(f"  Region #{i}: positions {rs}-{re} (length {re-rs+1})")
                context_start = max(0, rs - 5)
                context_end = min(max_len, re + 6)
                for j in range(3):
                    if context_end <= len(compacts[j]):
                        snippet = compacts[j][context_start:context_end]
                    else:
                        snippet = compacts[j][context_start:] + "<<<END"
                    print(f"    elem[{j}]: {snippet}")

    # Summary of all elements
    print(f"\n--- Summary ---")
    for i, comp in enumerate(elements):
        print(f"  elem[{i}]: {len(comp)} chars, first 80: {comp[:80]}")


if __name__ == '__main__':
    main()
