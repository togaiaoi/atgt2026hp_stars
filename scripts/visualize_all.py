#!/usr/bin/env python3
"""
Visualize all zoom*.pgm files as ASCII art, grouped by resolution.
Also attempts to composite the 4x4 depth images (depths 9-25) into a larger view
to see if the pattern spells text or reveals a recognizable image.
"""
import os
import glob
import re


def read_pgm(filepath):
    """Read a P5 (binary) PGM file. Returns (width, height, maxval, pixels_as_list)."""
    with open(filepath, 'rb') as f:
        data = f.read()

    # Parse P5 header: "P5\nW H\nMAXVAL\n" then binary data
    parts = data.split(b'\n', 3)
    magic = parts[0].strip()
    assert magic == b'P5', f"Expected P5, got {magic}"
    w, h = map(int, parts[1].decode().split())
    maxval = int(parts[2].decode().strip())
    pixels = list(parts[3])
    return w, h, maxval, pixels


def pixel_char(val):
    """Convert pixel value to ASCII character."""
    if val == 255:
        return '#'
    elif val == 0:
        return '.'
    else:
        return '?'


def render_ascii(w, h, pixels, indent="  "):
    """Render pixel data as ASCII art grid."""
    lines = []
    for y in range(h):
        row = ""
        for x in range(w):
            idx = y * w + x
            if idx < len(pixels):
                row += pixel_char(pixels[idx])
            else:
                row += '!'
        lines.append(f"{indent}{row}")
    return "\n".join(lines)


def extract_depth(filename):
    """Extract depth number from filename."""
    m = re.search(r'depth(\d+)', filename)
    if m:
        return int(m.group(1))
    return 0


def main():
    base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "images")
    pattern = os.path.join(base_dir, "zoom*.pgm")
    files = sorted(glob.glob(pattern))

    if not files:
        print("No zoom*.pgm files found!")
        return

    # Categorize files by resolution
    categories = {}
    all_images = {}
    for f in files:
        name = os.path.basename(f)
        w, h, maxval, pixels = read_pgm(f)
        depth = extract_depth(name)
        res_key = f"{w}x{h}"
        if res_key not in categories:
            categories[res_key] = []
        categories[res_key].append((depth, name, w, h, pixels))
        all_images[name] = (w, h, pixels)

    # Sort each category by depth
    for key in categories:
        categories[key].sort(key=lambda x: x[0])

    # Display order: by resolution ascending, then special groups
    print("=" * 70)
    print("  ZOOM PGM IMAGE VISUALIZATION")
    print("=" * 70)

    # Phase 1 images (varying resolution, depth 1-6)
    phase1_keys = []
    phase2_4x4_key = "4x4"
    phase2_8x8_key = "8x8"

    # Separate Phase 1 (zoom_depth*) and Phase 2 (zoom4x4_depth*, zoom8x8_depth*)
    phase1_images = []
    phase2_4x4_images = []
    phase2_8x8_images = []

    for f in files:
        name = os.path.basename(f)
        depth = extract_depth(name)
        w, h, _, pixels = read_pgm(f)
        if name.startswith("zoom_depth"):
            phase1_images.append((depth, name, w, h, pixels))
        elif name.startswith("zoom4x4_depth"):
            phase2_4x4_images.append((depth, name, w, h, pixels))
        elif name.startswith("zoom8x8_depth"):
            phase2_8x8_images.append((depth, name, w, h, pixels))

    phase1_images.sort(key=lambda x: x[0])
    phase2_4x4_images.sort(key=lambda x: x[0])
    phase2_8x8_images.sort(key=lambda x: x[0])

    # --- Phase 1: Full image at various resolutions ---
    print("\n" + "=" * 70)
    print("  PHASE 1: Full image renders (depth 1-6)")
    print("=" * 70)
    for depth, name, w, h, pixels in phase1_images:
        print(f"\n--- {name} (depth={depth}, {w}x{h}) ---")
        print(render_ascii(w, h, pixels))
        # Show raw values for small images
        if w * h <= 16:
            vals = pixels[:w*h]
            print(f"  Raw values: {vals}")

    # --- Phase 2: 4x4 center zoom ---
    print("\n" + "=" * 70)
    print("  PHASE 2: 4x4 center zoom (depths 9-25)")
    print("=" * 70)
    for depth, name, w, h, pixels in phase2_4x4_images:
        print(f"\n--- {name} (depth={depth}, {w}x{h}) ---")
        print(render_ascii(w, h, pixels))
        vals = pixels[:w*h]
        print(f"  Raw values: {vals}")

    # --- Phase 2: 8x8 center zoom ---
    print("\n" + "=" * 70)
    print("  PHASE 2: 8x8 center zoom (depths 9-15)")
    print("=" * 70)
    for depth, name, w, h, pixels in phase2_8x8_images:
        print(f"\n--- {name} (depth={depth}, {w}x{h}) ---")
        print(render_ascii(w, h, pixels))

    # === COMPOSITE VIEW ===
    # The 4x4 images at depths 9-25 each show the center 4x4 pixels at that zoom level.
    # If we arrange them in order, we can see the pattern evolve with depth.
    #
    # More interestingly: each deeper depth reveals finer detail of the CENTER of the image.
    # The quadtree structure means:
    #   - depth N, 4x4 = center 4 quadrants at depth N
    #   - Going deeper = zooming into center
    #
    # Let's arrange them as a strip (side by side) and as a vertical stack.

    print("\n" + "=" * 70)
    print("  COMPOSITE: All 4x4 depth images side-by-side (depths 9-25)")
    print("=" * 70)

    if phase2_4x4_images:
        # Print depth numbers as header
        header = "     "
        for depth, name, w, h, pixels in phase2_4x4_images:
            header += f" d{depth:2d} "
        print(header)

        for row in range(4):
            line = f"  r{row}: "
            for depth, name, w, h, pixels in phase2_4x4_images:
                row_data = ""
                for col in range(4):
                    idx = row * w + col
                    if idx < len(pixels):
                        row_data += pixel_char(pixels[idx])
                    else:
                        row_data += '!'
                line += f" {row_data} "
            print(line)

    # === COMPOSITE: Arrange 4x4 grids in a larger pattern ===
    # 17 images (depth 9-25). Let's try arranging them in a grid.
    # 17 images -> try 4x5 grid (with 3 empty) or a single row
    print("\n" + "=" * 70)
    print("  COMPOSITE: 4x4 images arranged in grid layout")
    print("  (trying to find if the pattern spells something)")
    print("=" * 70)

    if phase2_4x4_images:
        n_images = len(phase2_4x4_images)

        # Try different grid arrangements
        for grid_cols in [4, 5, 6, 17]:
            grid_rows = (n_images + grid_cols - 1) // grid_cols
            print(f"\n--- Layout: {grid_cols} columns x {grid_rows} rows ---")

            # Print depth labels
            for gr in range(grid_rows):
                label_line = "  "
                for gc in range(grid_cols):
                    img_idx = gr * grid_cols + gc
                    if img_idx < n_images:
                        depth = phase2_4x4_images[img_idx][0]
                        label_line += f"[d{depth:2d}] "
                    else:
                        label_line += "      "
                print(label_line)

                # Print 4 rows of pixel data
                for pixel_row in range(4):
                    line = "  "
                    for gc in range(grid_cols):
                        img_idx = gr * grid_cols + gc
                        if img_idx < n_images:
                            _, _, w, h, pixels = phase2_4x4_images[img_idx]
                            row_data = ""
                            for col in range(4):
                                idx = pixel_row * w + col
                                if idx < len(pixels):
                                    row_data += pixel_char(pixels[idx])
                                else:
                                    row_data += '!'
                            line += f" {row_data} "
                        else:
                            line += "      "
                    print(line)
                print()

    # === BINARY ANALYSIS ===
    # Treat each 4x4 image as 16 bits and see if they encode something
    print("\n" + "=" * 70)
    print("  BINARY ANALYSIS: Each 4x4 as a 16-bit value")
    print("=" * 70)

    if phase2_4x4_images:
        for depth, name, w, h, pixels in phase2_4x4_images:
            bits = ""
            val = 0
            for i in range(min(16, len(pixels))):
                b = 1 if pixels[i] == 255 else 0
                bits += str(b)
                val = (val << 1) | b
            print(f"  depth {depth:2d}: {bits} = 0x{val:04X} = {val:5d}")

        # Try reading as bytes (8-bit): top half and bottom half of each 4x4
        print("\n  --- As 8-bit values (top half / bottom half) ---")
        all_bytes_top = []
        all_bytes_bot = []
        for depth, name, w, h, pixels in phase2_4x4_images:
            top_val = 0
            bot_val = 0
            for i in range(8):
                b = 1 if pixels[i] == 255 else 0
                top_val = (top_val << 1) | b
            for i in range(8, 16):
                b = 1 if pixels[i] == 255 else 0
                bot_val = (bot_val << 1) | b
            all_bytes_top.append(top_val)
            all_bytes_bot.append(bot_val)
            t_chr = chr(top_val) if 32 <= top_val < 127 else '?'
            b_chr = chr(bot_val) if 32 <= bot_val < 127 else '?'
            print(f"  depth {depth:2d}: top=0x{top_val:02X}({t_chr}) bot=0x{bot_val:02X}({b_chr})")

        # Try reading columns instead of rows
        print("\n  --- As 16-bit column-major ---")
        for depth, name, w, h, pixels in phase2_4x4_images:
            val = 0
            bits = ""
            for col in range(4):
                for row in range(4):
                    idx = row * w + col
                    b = 1 if (idx < len(pixels) and pixels[idx] == 255) else 0
                    bits += str(b)
                    val = (val << 1) | b
            print(f"  depth {depth:2d}: {bits} = 0x{val:04X} = {val:5d}")

    # === 8x8 COMPOSITE ===
    print("\n" + "=" * 70)
    print("  COMPOSITE: All 8x8 depth images side-by-side (depths 9-15)")
    print("=" * 70)

    if phase2_8x8_images:
        header = "     "
        for depth, name, w, h, pixels in phase2_8x8_images:
            header += f"  d{depth:2d}     "
        print(header)

        for row in range(8):
            line = f"  r{row}: "
            for depth, name, w, h, pixels in phase2_8x8_images:
                row_data = ""
                for col in range(8):
                    idx = row * w + col
                    if idx < len(pixels):
                        row_data += pixel_char(pixels[idx])
                    else:
                        row_data += '!'
                line += f" {row_data} "
            print(line)

    # === 8x8 BINARY ANALYSIS ===
    print("\n" + "=" * 70)
    print("  8x8 BINARY: Each 8x8 row as a byte")
    print("=" * 70)

    if phase2_8x8_images:
        for depth, name, w, h, pixels in phase2_8x8_images:
            print(f"\n  depth {depth}:")
            all_row_vals = []
            for row in range(8):
                val = 0
                bits = ""
                for col in range(8):
                    idx = row * w + col
                    b = 1 if (idx < len(pixels) and pixels[idx] == 255) else 0
                    bits += str(b)
                    val = (val << 1) | b
                c = chr(val) if 32 <= val < 127 else '?'
                all_row_vals.append(val)
                print(f"    row{row}: {bits} = 0x{val:02X} = {val:3d} '{c}'")
            # Check if it looks like a character bitmap
            print(f"    Row values: {all_row_vals}")

    # === SUMMARY ===
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  Total files: {len(files)}")
    print(f"  Phase 1 (full image): {len(phase1_images)} files, depths {[x[0] for x in phase1_images]}")
    print(f"  Phase 2 (4x4 zoom):  {len(phase2_4x4_images)} files, depths {[x[0] for x in phase2_4x4_images]}")
    print(f"  Phase 2 (8x8 zoom):  {len(phase2_8x8_images)} files, depths {[x[0] for x in phase2_8x8_images]}")

    # Count white pixels across depths
    print("\n  White pixel counts per depth (4x4):")
    for depth, name, w, h, pixels in phase2_4x4_images:
        whites = sum(1 for p in pixels[:w*h] if p == 255)
        print(f"    depth {depth:2d}: {whites}/16 white pixels")


if __name__ == "__main__":
    main()
