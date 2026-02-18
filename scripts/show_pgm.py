#!/usr/bin/env python3
"""Decode PGM images and show as ASCII art."""
import sys, os, glob

pattern = sys.argv[1] if len(sys.argv) > 1 else "d:/github/atgt2026hp_stars/images/zoom4x4_depth*.pgm"
files = sorted(glob.glob(pattern))

for f in files:
    name = os.path.basename(f)
    with open(f, 'rb') as fh:
        data = fh.read()

    # Parse P5 PGM
    parts = data.split(b'\n', 3)
    magic = parts[0]  # P5
    dims = parts[1].decode()  # "W H"
    maxval = parts[2]  # "255"
    pixels = parts[3]

    w, h = map(int, dims.split())

    print(f"=== {name} ({w}x{h}) ===")
    for y in range(h):
        row = ""
        for x in range(w):
            idx = y * w + x
            if idx < len(pixels):
                b = pixels[idx]
                row += "#" if b > 128 else "."
            else:
                row += "?"
        print(f"  {row}")
    print()
