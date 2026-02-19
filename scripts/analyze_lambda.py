#!/usr/bin/env python3
"""Analyze the left_x_lambda.txt decoder structure."""

import re
import sys

def main():
    data = open('d:/github/atgt2026hp_stars/extracted/left_x_lambda.txt', 'r').read()
    print(f"Total length: {len(data)}")

    # The file is one long line: \x1. x1 (FIELD0) (FIELD1) (FIELD2) (FIELD3) (FIELD4)
    # This is a Church-encoded 5-tuple: lambda f. f(COND)(NW)(NE)(SW)(SE)

    # Find top-level field boundaries
    pos = 8  # after "\x1. x1 "
    depth = 0
    fields = []
    arg_start = None

    while pos < len(data):
        c = data[pos]
        if c == '(' and depth == 0:
            arg_start = pos
            depth = 1
        elif c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
            if depth == 0 and arg_start is not None:
                fields.append((arg_start, pos + 1))
                arg_start = None
        pos += 1

    field_names = ['COND', 'NW', 'NE', 'SW', 'SE']
    print("\n=== 5-Tuple Field Boundaries ===")
    for i, (s, e) in enumerate(fields):
        print(f"  {field_names[i]}: pos {s}-{e}, size {e-s} chars ({(e-s)*100/len(data):.1f}%)")

    # Print small fields entirely
    print("\n=== Field SW (complete) ===")
    print(data[fields[3][0]:fields[3][1]])

    print("\n=== Field NE (complete) ===")
    print(data[fields[2][0]:fields[2][1]])

    # Find Y-combinator patterns using simple string search
    # Pattern: "xN (xM xM) (\xP. xN (xP xP))"
    # This is Z-combinator: f(z z)(\w. f(w w))

    def find_y_combinators(text):
        """Find Y/Z combinator patterns in lambda text."""
        results = []
        # Look for pattern: xN (xM xM) (\xP. xN (xP xP))
        i = 0
        while i < len(text):
            # Look for "xNNN "
            if text[i] == 'x' and i > 0 and text[i-1] in ' .(':
                # Extract variable name
                j = i + 1
                while j < len(text) and text[j].isdigit():
                    j += 1
                var_f = text[i:j]

                # Check for " (xM xM) (\xP. xN (xP xP))"
                rest = text[j:]
                if rest.startswith(' ('):
                    # Find xM xM pattern
                    k = 2
                    if k < len(rest) and rest[k] == 'x':
                        m = k + 1
                        while m < len(rest) and rest[m].isdigit():
                            m += 1
                        var_z = rest[k:m]
                        expected = f" {var_z}) (\\x"
                        if rest[m:m+len(expected)] == expected:
                            # Find the \xP variable
                            p = m + len(expected)
                            q = p
                            while q < len(rest) and rest[q].isdigit():
                                q += 1
                            var_w = 'x' + rest[p:q]
                            expected2 = f". {var_f} ({var_w} {var_w}))"
                            if rest[q:q+len(expected2)] == expected2:
                                results.append((i, var_f, var_z, var_w))
                i = j
            else:
                i += 1
        return results

    print("\n=== Y-Combinator Locations ===")
    for fi, (s, e) in enumerate(fields):
        field_text = data[s:e]
        ys = find_y_combinators(field_text)
        print(f"\n{field_names[fi]} ({e-s} chars): {len(ys)} Y-combinators")
        for pos, f, z, w in ys:
            # Show context
            ctx_start = max(0, pos - 40)
            ctx_end = min(len(field_text), pos + 80)
            context = field_text[ctx_start:ctx_end]
            print(f"  at +{pos}: f={f}, z={z}, w={w}")
            print(f"    context: ...{context}...")

    # Now let's look at the structure of Field 4 (SE) - the biggest field
    # It should contain the main data processing logic with 4 Y-combinators
    print("\n=== Field SE structure (first 1000 chars) ===")
    se = data[fields[4][0]+1:fields[4][1]-1]  # strip outer parens
    print(se[:1000])
    print("\n=== Field SE structure (last 1000 chars) ===")
    print(se[-1000:])

    # Look for the terminal pattern: result(false)(false)(false)(prev)(1)
    # false = \x.\y. y = KI
    # In the lambda notation, false would be (\xN. \xM. xM)
    # Let's find all occurrences of (\xN. \xM. xM) patterns
    false_pattern_count = 0
    i = 0
    while i < len(se):
        # Match (\xN. \xM. xM)
        if se[i:i+2] == '(\\':
            j = i + 2
            while j < len(se) and se[j] != '.':
                j += 1
            if j < len(se) and se[j] == '.':
                # skip ". \"
                rest = se[j+1:].lstrip()
                if rest.startswith('\\'):
                    k = 1
                    while k < len(rest) and rest[k] != '.':
                        k += 1
                    if k < len(rest):
                        var2 = rest[1:k].strip()
                        after_dot = rest[k+1:].lstrip()
                        expected_end = var2 + ')'
                        if after_dot.startswith(expected_end):
                            false_pattern_count += 1
        i += 1

    print(f"\n'false' (KI) pattern count in SE: {false_pattern_count}")

    # Let's look at the end of SE for the step function pattern
    # step(step(step(data, 2048), 256), 32)
    # Numbers in Church encoding would appear as specific patterns
    print("\n=== Looking for numeric constants in SE ===")
    # Find all lambda abstractions at the end
    print("Last 2000 chars of SE:")
    print(se[-2000:])

if __name__ == '__main__':
    main()
