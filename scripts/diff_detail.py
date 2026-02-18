#!/usr/bin/env python3
"""Deep structural diff detail of elem[0], elem[1], elem[2] from item_09."""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

APP=0; TAG_S=1; TAG_K=2; TAG_I=3; S1=4; S2=5; K1=6; IND=7
TAG_NAMES=['APP','S','K','I','S1','S2','K1','IND']

class Arena:
    def __init__(self, cap=2000000):
        self.tag=bytearray(cap); self.left=[0]*cap; self.right=[0]*cap; self.size=0
    def alloc(self, tag, a=0, b=0):
        idx=self.size
        if idx>=len(self.tag):
            nc=len(self.tag)*2
            self.tag.extend(bytearray(nc-len(self.tag)))
            self.left.extend([0]*(nc-len(self.left)))
            self.right.extend([0]*(nc-len(self.right)))
        self.tag[idx]=tag; self.left[idx]=a; self.right[idx]=b; self.size+=1; return idx
    def follow(self, idx):
        while self.tag[idx]==IND: idx=self.left[idx]
        return idx
    def whnf(self, idx, fuel):
        steps=0; spine=[]; n=idx
        while steps<fuel:
            n=self.follow(n); t=self.tag[n]
            if t==APP: spine.append(n); n=self.left[n]; continue
            if t==TAG_I and len(spine)>=1:
                steps+=1; app=spine.pop(); x=self.follow(self.right[app]); self.tag[app]=IND; self.left[app]=x; n=x; continue
            if t==TAG_K and len(spine)>=2:
                steps+=1; a1=spine.pop(); a2=spine.pop(); x=self.follow(self.right[a1]); self.tag[a2]=IND; self.left[a2]=x; self.tag[a1]=K1; self.left[a1]=x; n=x; continue
            if t==K1 and len(spine)>=1:
                steps+=1; app=spine.pop(); x=self.follow(self.left[n]); self.tag[app]=IND; self.left[app]=x; n=x; continue
            if t==TAG_S and len(spine)>=3:
                steps+=1; a1=spine.pop(); a2=spine.pop(); a3=spine.pop()
                f=self.follow(self.right[a1]); g=self.follow(self.right[a2]); x=self.right[a3]
                fx=self.alloc(APP,f,x); gx=self.alloc(APP,g,x); r=self.alloc(APP,fx,gx)
                self.tag[a3]=IND; self.left[a3]=r; self.tag[a1]=S1; self.left[a1]=f; self.tag[a2]=S2; self.left[a2]=f; self.right[a2]=g; n=r; continue
            if t==S1 and len(spine)>=2:
                steps+=1; a1=spine.pop(); a2=spine.pop()
                f=self.follow(self.left[n]); g=self.follow(self.right[a1]); x=self.right[a2]
                fx=self.alloc(APP,f,x); gx=self.alloc(APP,g,x); r=self.alloc(APP,fx,gx)
                self.tag[a2]=IND; self.left[a2]=r; self.tag[a1]=S2; self.left[a1]=f; self.right[a1]=g; n=r; continue
            if t==S2 and len(spine)>=1:
                steps+=1; app=spine.pop()
                f=self.follow(self.left[n]); g=self.follow(self.right[n]); x=self.right[app]
                fx=self.alloc(APP,f,x); gx=self.alloc(APP,g,x); r=self.alloc(APP,fx,gx)
                self.tag[app]=IND; self.left[app]=r; n=r; continue
            if spine: return spine[0],steps
            return n,steps
        if spine: return spine[0],steps
        return n,steps

def parse_compact(arena, text):
    stack=[]
    for ch in text:
        if ch=='k': stack.append(arena.alloc(TAG_S))
        elif ch=='X': stack.append(arena.alloc(TAG_K))
        elif ch=='D': stack.append(arena.alloc(TAG_I))
        elif ch=='-': arg=stack.pop(); func=stack.pop(); stack.append(arena.alloc(APP,func,arg))
    return stack[0]

def make_false(a): return a.alloc(APP, a.alloc(TAG_K), a.alloc(TAG_I))

def pair_snd(a, node, fuel=50000000):
    ki=a.alloc(APP, a.alloc(TAG_K), a.alloc(TAG_I)); d=make_false(a)
    a1=a.alloc(APP,node,ki); a2=a.alloc(APP,a1,d)
    _,s=a.whnf(a2,fuel); return a.follow(a2),s

def pair_fst(a, node, fuel=50000000):
    k=a.alloc(TAG_K); d=make_false(a)
    a1=a.alloc(APP,node,k); a2=a.alloc(APP,a1,d)
    _,s=a.whnf(a2,fuel); return a.follow(a2),s

def describe(arena, idx, depth=3):
    idx=arena.follow(idx)
    if depth<=0: return '...'
    t=arena.tag[idx]
    if t==TAG_S: return 'S'
    if t==TAG_K: return 'K'
    if t==TAG_I: return 'I'
    if t==APP: return '({} {})'.format(describe(arena,arena.left[idx],depth-1), describe(arena,arena.right[idx],depth-1))
    if t==K1: return 'K1({})'.format(describe(arena,arena.left[idx],depth-1))
    if t==S1: return 'S1({})'.format(describe(arena,arena.left[idx],depth-1))
    if t==S2: return 'S2({},{})'.format(describe(arena,arena.left[idx],depth-1), describe(arena,arena.right[idx],depth-1))
    if t==IND: return describe(arena,arena.left[idx],depth)
    return '?{}'.format(t)

def nav(arena, idx, path):
    """Navigate a path like ['L','R','L'] through the tree."""
    idx = arena.follow(idx)
    for step in path:
        t = arena.tag[idx]
        if step == 'L':
            if t == APP: idx = arena.follow(arena.left[idx])
            elif t in (K1, S1): idx = arena.follow(arena.left[idx])
            elif t == S2: idx = arena.follow(arena.left[idx])
            else: return None
        elif step == 'R':
            if t == APP: idx = arena.follow(arena.right[idx])
            elif t == S2: idx = arena.follow(arena.right[idx])
            else: return None
        elif step == 'f':
            if t == S2: idx = arena.follow(arena.left[idx])
            else: return None
        elif step == 'g':
            if t == S2: idx = arena.follow(arena.right[idx])
            else: return None
        elif step == 'a':
            if t in (K1, S1): idx = arena.follow(arena.left[idx])
            else: return None
    return idx

def serialize_small(arena, idx, max_size=2000):
    """Serialize a small subtree to compact notation."""
    result = []
    stack = [(arena.follow(idx), 0)]
    count = 0
    while stack and count < max_size:
        idx2, phase = stack.pop()
        idx2 = arena.follow(idx2)
        t = arena.tag[idx2]
        if t == TAG_S: result.append('k'); count += 1
        elif t == TAG_K: result.append('X'); count += 1
        elif t == TAG_I: result.append('D'); count += 1
        elif t == APP:
            if phase == 0:
                stack.append((idx2, 1))
                stack.append((arena.right[idx2], 0))
                stack.append((arena.left[idx2], 0))
            else:
                result.append('-'); count += 1
        elif t == K1:
            if phase == 0:
                stack.append((idx2, 1))
                stack.append((arena.left[idx2], 0))
                result.append('X')  # K
                count += 1
            else:
                result.append('-'); count += 1
        elif t == S1:
            if phase == 0:
                stack.append((idx2, 1))
                stack.append((arena.left[idx2], 0))
                result.append('k')  # S
                count += 1
            else:
                result.append('-'); count += 1
        elif t == S2:
            if phase == 0:
                stack.append((idx2, 1))
                stack.append((arena.right[idx2], 0))
                stack.append((-1, 2))  # inner app marker
                stack.append((arena.left[idx2], 0))
                result.append('k')  # S
                count += 1
            elif phase == 1:
                result.append('-'); count += 1
            elif phase == 2:
                result.append('-'); count += 1
        elif t == IND:
            stack.append((arena.left[idx2], 0))
        else:
            result.append('?{}'.format(t)); count += 1
    if count >= max_size:
        result.append('...(truncated)')
    return ''.join(result)


def main():
    with open(r'd:\github\atgt2026hp_stars\extracted\data_items\item_09.txt') as f:
        text = f.read().strip()

    arena = Arena(cap=len(text)*4)
    root = parse_compact(arena, text)
    base_size = arena.size
    print("Arena base size:", base_size)

    # Extract elem[0..5]
    elems = []
    current = root
    for i in range(6):
        snd, _ = pair_snd(arena, current)
        fst, _ = pair_fst(arena, current)
        elems.append(snd)
        current = arena.follow(fst)

    print("Extracted {} elements".format(len(elems)))

    # Deep description of each elem[0..2]
    print("\n" + "=" * 80)
    print("DETAILED STRUCTURE OF elem[0], elem[1], elem[2]")
    print("=" * 80)

    for i in range(3):
        print("\n--- elem[{}] ---".format(i))
        print("  root idx={}, tag={}".format(elems[i], TAG_NAMES[arena.tag[elems[i]]]))
        print("  desc(6): {}".format(describe(arena, elems[i], 6)))

        # Navigate key paths
        paths = [
            ['L'],
            ['R'],
            ['L', 'L'],
            ['L', 'R'],
            ['L', 'R', 'L'],
            ['L', 'R', 'R'],
            ['R', 'L'],
            ['R', 'R'],
            ['R', 'L', 'L'],
            ['R', 'L', 'R'],
        ]
        for p in paths:
            n = nav(arena, elems[i], p)
            if n is not None:
                pstr = '.'.join(p)
                t = TAG_NAMES[arena.tag[n]] if arena.tag[n] < 8 else '?{}'.format(arena.tag[n])
                print("  .{}: idx={} tag={} desc={}".format(pstr, n, t, describe(arena, n, 5)))

    # The 3 differing locations between elem[0] and elem[1]:
    print("\n" + "=" * 80)
    print("DIFF DETAILS: exact content at differing paths")
    print("=" * 80)

    # path .L.R.L
    print("\n--- path .L.R.L (differs between all) ---")
    for i in range(3):
        n = nav(arena, elems[i], ['L', 'R', 'L'])
        if n is not None:
            print("  elem[{}]: idx={} tag={} desc(8)={}".format(i, n, TAG_NAMES[arena.tag[n]], describe(arena, n, 8)))
            compact = serialize_small(arena, n, max_size=500)
            print("    compact({} chars): {}".format(len(compact), compact[:200]))

    # path .R.L.L
    print("\n--- path .R.L.L (differs between all) ---")
    for i in range(3):
        n = nav(arena, elems[i], ['R', 'L', 'L'])
        if n is not None:
            print("  elem[{}]: idx={} tag={} desc(5)={}".format(i, n, TAG_NAMES[arena.tag[n]], describe(arena, n, 5)))

    # path .R.L.R
    print("\n--- path .R.L.R (differs between 0 vs 1,2) ---")
    for i in range(3):
        n = nav(arena, elems[i], ['R', 'L', 'R'])
        if n is not None:
            print("  elem[{}]: idx={} tag={} desc(5)={}".format(i, n, TAG_NAMES[arena.tag[n]], describe(arena, n, 5)))

    # Additional context: describe elem[3..5] briefly
    print("\n" + "=" * 80)
    print("BRIEF DESCRIPTION of elem[3..5]")
    print("=" * 80)
    for i in range(3, min(6, len(elems))):
        print("  elem[{}]: idx={} tag={} desc(5)={}".format(i, elems[i], TAG_NAMES[arena.tag[elems[i]]] if arena.tag[elems[i]]<8 else '?', describe(arena, elems[i], 5)))
        compact = serialize_small(arena, elems[i], max_size=200)
        print("    compact({} chars): {}".format(len(compact), compact))


if __name__ == '__main__':
    main()
