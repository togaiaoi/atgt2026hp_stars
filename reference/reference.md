# Stars 2026 Complete Reference Manual

---

## Part 1: Overview

### 1.1 Server Communication

All interaction is performed via POST requests to `https://stars-2026-hp.xyz/`.

**Sending commands:**
```bash
node send.js '<custom_language_command>'
```

`send.js` sends `{input: "<command>"}` as JSON and receives `{output: [{type: "string"|"image", value: ...}]}`.

### 1.2 Top-Level Commands

| Command | Custom | Syntax | Purpose |
|---------|--------|--------|---------|
| view | `zMnX` | `[view (chapter TITLE)]` | Display a problem description |
| answer | `znQPX` | `[answer (question ID) ANSWER]` | Submit a solution |
| print | `QnWPX` | `[print EXPR]` | Print/evaluate an expression |
| encode | `zlWPX` | `[encode EXPR]` | Encode expression to graph |
| what | `wnz` | `[what EXPR]` | Query information |

**Custom encoding of commands:**
```
[view (chapter TITLE)]    = czMnX e-nQPz TITLE8(
[answer (question ID) X]  = cznQPX ePX{nQ ID8 X(
[print EXPR]              = cQnWPX EXPR(
```

### 1.3 Server Response Format

**Correct answer:**
```
[result [question (question ID)] [answer ECHO] [correct [((chapter NEXT) is next chapter)]]]
```

**Wrong answer:**
```
[result [question (question ID)] [answer ECHO] [wrong]]
```

**View error (title not found):**
```
[error (wrong definition title (chapter TITLE))]
```

### 1.4 Problem IDs

| Problem | Question ID | Chapter Title (for view) |
|---------|------------|--------------------------|
| Q.0 | `QfDlu` | `QfDlu QnPX9M QnQ QfnQ&` |
| Q.1 | `QfXn&` | `QfXn& QnQPW9- PQ,P QPwF9zPX QnQ95P` |
| Q.2 | `QfPXz` | `QfPXz kPXzP Qnl-5P5` |
| Q.3 | `Qfn9u` | `Qfn9u CnQu9 QPW9-` |
| Q.4 | `QfzPQ` | `QfzPQ Qn9XPX` |
| Q.5 | `QfuPX` | `QfuPX wlQWPM Qn9XPX` |
| Q.6 | `QfuPXn&` | `QfuPXn& Qnz-n5` |
| Q.7 | `QfuPXPXz` | `QfuPXPXz Qnun-` |
| Q.8 | `QfuPXn9u` | `QfuPXn9u Qnz9XP5` |
| Q.9 | `QfuPXzlQ` | `QfuPXzlQ QPMFunPX PQ,P QPPuMnX` |
| Q.10 | `Qfwn-` | `Qfwn- kn<lW QPPuMnX` |
| Q.11 | `Qfwn-Xn&` | (end marker, not a problem) |
| l.0 | `lfDlu` | (title unknown — view rejected) |
| l.1 | `lfXn&` | `lfXn& QP5959PX QnQ Qn&n-QPznk` |
| l.2 | `lfPXz` | `lfPXz D5PMfXnkPXfQPkPM9X Qn&n-QPznk` |
| l.3 | `lfn9u` | `lfn9u D5PMfXnkPXfQPkPM9Xf&n-QPznk QPul-FX QnPXPn` |
| l.4 | `lfzPQ` | `lfzPQ D5PMfXnkPXfQPkPM9Xf&n-QPznk QPul-FX 9XPXfQPW9-` |
| l.5 | `lfuPX` | (end marker, not a problem) |

### 1.5 Answer Format by Problem Type

| Series | Problems | Answer Format |
|--------|----------|---------------|
| Q-series (simple) | Q.0–Q.3 | Number literal (e.g., `PXz` = 2) |
| Q-series (code) | Q.4–Q.10 | Code expression (e.g., `c9XPX ck( ...`) |
| l-series (graph) | l.0–l.3 | Graph symbol in parens: `e<graph_symbol>8` |
| l-series (code) | l.4 | Code expression |

---

## Part 2: Language Specification

### 2.1 Structural Characters

| Custom | Standard | Role |
|--------|----------|------|
| `c` | `[` | Open bracket (expression start) |
| `(` | `]` | Close bracket (expression end) |
| `e` | `(` | Open paren (grouping/graph symbol) |
| `8` | `)` | Close paren |
| `j` | `"` | String quote |
| `[` | `,` | Comma separator |
| `f` | `.` | Dot (namespace separator) |
| `o` | `?` | Question mark |

**Expression structure:** All expressions use bracket notation `[operator arg1 arg2 ...]`

Custom: `cOPERATOR ARG1 ARG2 ...(` = Standard: `[OPERATOR ARG1 ARG2 ...]`

Examples:
```
cQPM Xn& PXz(          = [add 1 2]
cCnQu9 kF9Q n9u zPQ(   = [if true 3 4]
c9XPX ck( k(            = [function [var] var]
```

### 2.2 Number System

**Base digits (0–9):**

| Custom | Value | | Custom | Value |
|--------|-------|-|--------|-------|
| `Dlu` | 0 | | `uPX` | 5 |
| `Xn&` | 1 | | `uPXn&` | 6 |
| `PXz` | 2 | | `uPXPXz` | 7 |
| `n9u` | 3 | | `uPXn9u` | 8 |
| `zPQ` | 4 | | `uPXzlQ` | 9 |

**Place values:**

| Custom | Value |
|--------|-------|
| `wn-` | 10 |
| `QPz` | 100 |
| `X9Q` | 1,000 |
| `0P,` | 10,000 |

**Compound number rules:**

| Rule | Pattern | Example |
|------|---------|---------|
| Addition | Concatenate | `wn-Xn&` = 10 + 1 = **11** |
| Multiplication | `<digit>n-<place>` | `PXzn-wn-` = 2 x 10 = **20** |
| Combined | Mult + Add | `PXzn-QPzn9un-wn-zPQ` = 200 + 30 + 4 = **234** |
| Negative | `MPQ<x>` = -(x + 1) | `MPQDlu` = -(0+1) = **-1** |

**Negative number examples:**

| Custom | Value |
|--------|-------|
| `MPQDlu` | -1 |
| `MPQXn&` | -2 |
| `MPQPXz` | -3 |
| `MPQzPQ` | -5 |
| `MPQuPXn&` | -7 |
| `MPQuPX` | -11 |
| `MPQuPXPXz` | -13 |

### 2.3 Booleans

| Custom | Value |
|--------|-------|
| `kF9Q` | true |
| `Q9Fk` | false |

---

## Part 3: Expression Evaluation System

Code is a tree of expressions. Evaluation proceeds by applying **reduction rules** — each rule rewrites a pattern into a simpler form. Reduction repeats until no more rules apply, yielding a final value.

### 3.1 Evaluation Fundamentals (Q.1)

**Key concepts:**
- Code consists of elements (atoms) or stacks of elements
- A **reduction rule** has the form: `result <- original`
- A **sub.code** is any subtree of the expression
- A **sub.match** applies a reduction rule to matching subtrees
- Variables `param`, `param.1`, `param.2` are pattern-matching placeholders

**How evaluation works:**
1. Find the leftmost reducible expression (redex)
2. Apply the matching reduction rule
3. Repeat until the expression is fully reduced

**Example reduction chain:**
```
[if true [add 1 2] [multiply 3 4]]
  rule: "param.1" <- "[if true param.1 param.2]"
→ [add 1 2]
  rule: "3" <- "[add 1 2]"
→ 3
```

### 3.2 Arithmetic Operations (Q.2)

| Operator | Custom | Syntax | Rule |
|----------|--------|--------|------|
| add | `QPM` | `[add A B]` | -> A + B |
| subtract | `MPQ` | `[subtract A B]` | -> A - B |
| multiply | `XP9X` | `[multiply A B]` | -> A x B |

**Reduction example:**
```
[multiply [add 1 2] 3]
→ [multiply 3 3]          (reduce inner: [add 1 2] → 3)
→ 9                        (reduce: [multiply 3 3] → 9)
```

### 3.3 Comparison Operators (Q.3)

| Operator | Custom | Syntax | Returns |
|----------|--------|--------|---------|
| greater | `,lQ{` | `[greater A B]` | true if A > B, else false |
| equals | `D5PM` | `[equals A B]` | true if A = B, else false |

### 3.4 Boolean Operators (Q.3)

| Operator | Custom | Syntax |
|----------|--------|--------|
| and | `PQ,P` | `[and A B]` |
| or | `MnW` | `[or A B]` |
| not | `PX,` | `[not A]` |

### 3.5 Conditional (Q.3)

```
[if CONDITION THEN ELSE]
= cCnQu9 COND THEN ELSE(
```

Reduction rules:
```
[if true  X Y] → X
[if false X Y] → Y
```

**Example (nested conditionals):**
```
[if [greater 2 1] [if [equals 2 1] 3 4] 5]
→ [if true [if [equals 2 1] 3 4] 5]     (greater 2 1 = true)
→ [if [equals 2 1] 3 4]                  (if true selects first branch)
→ [if false 3 4]                          (equals 2 1 = false)
→ 4                                       (if false selects second branch)
```

### 3.6 Functions (Q.4)

**Definition:**
```
[function [PARAMS] BODY]
= c9XPX cPARAMS( BODY(
```

**Call:**
```
[call FUNC ARGS]
= ckPM9X FUNC ARGS(
```

**Reduction:** Calling a function substitutes arguments into the body.
```
[call [function [var] [add var 1]] 5]
→ [add 5 1]
→ 6
```

**Multi-parameter functions** use section variables with dot-notation:
```
[function [z.0 z.1] [add z.0 z.1]]
= c9XPX czfDlu zfXn&( cQPM zfDlu zfXn&((
```

**Section variable naming:** `zfDlu` = z.0, `zfXn&` = z.1, `zfPXz` = z.2, `zfn9u` = z.3, `zfzPQ` = z.4

### 3.7 Recursion (Q.5)

```
[recursive [function [self] BODY]]
= cwlQWPM c9XPX c9( BODY((
```

The `recursive` wrapper allows the function to call itself via the `self` parameter (encoded as `9`).

**Reduction rule:**
```
[call param [recursive param]] <- [recursive param]
```
i.e., `[recursive F]` expands to `[call F [recursive F]]`, giving `F` access to itself.

**Standard recursive pattern:**
```
[recursive [function [self] [function [PARAMS]
  [if BASE_CONDITION
    BASE_VALUE
    RECURSIVE_EXPRESSION]]]]
```

### 3.8 Arrays & Destructuring (Q.6)

**Construction:**
```
[array ELEM0 ELEM1 ...]
= cz-n5 ELEM0 ELEM1 ...(
```

**Destructuring with input:**
```
[input [z.0 z.1] ARRAY_EXPR BODY]
= czPX czfDlu zfXn&( ARRAY_EXPR BODY(
```

**Reduction rule:**
```
[input [z.0 z.1] [array A B] BODY]
→ BODY with z.0=A, z.1=B
```

`input` (`zPX`) binds array elements to variables for use in the body expression.

### 3.9 Lists & Fold (Q.7)

**Constructors:**

| Operation | Custom | Syntax | Meaning |
|-----------|--------|--------|---------|
| nil (empty) | `kQPM` | `empty_stack` | Empty list |
| cons (push) | `Xn-PX` | `[push HEAD TAIL]` | Prepend element |

**List construction:**
```
[push 1 [push 2 [push 3 empty_stack]]]   → list [1, 2, 3]
```

**Fold (pop):**
```
[pop LIST [z.1 z.2 BODY] ACCUMULATOR]
= c-PuuPX LIST czfXn& zfPXz BODY( ACC(
```

**Reduction rules:**
```
[pop empty_stack          FUNC ACC] → ACC
[pop [push HEAD TAIL]     [z.1 z.2 BODY] ACC] → BODY with z.1=HEAD, z.2=TAIL
```

**IMPORTANT:** `z.2` receives the **raw tail** (remaining list), NOT a recursive fold result. For recursive processing, explicit recursion is required:

```
[recursive [function [self] [function [var]
  [pop var [z.1 z.2 [add z.1 [call self z.2]]] 0]]]]
```

### 3.10 Strings / Symbols (Q.8)

**Symbols** are string values enclosed in parentheses:
```
(example)  = e-5n5nQ8
(true)     = ekF9Q8
```

**Internal representation:** Strings are **stacks (lists) of character codes** — each character is stored as a number. The same `push`/`pop`/`empty_stack` operations used for lists work on strings.

**String operations:**
- `[format [output.string SYM end]]` outputs a symbol/string value
- Accessing first character: `[pop string [z.0 z.1 [push z.0 empty_stack]] empty_stack]`
- Strings can be iterated, reversed, and transformed using list operations (fold, cons, etc.)

**Key insight:** Since strings are lists of numbers, any list-processing function (fold, recursive traversal) works on strings. String reversal is identical to list reversal.

---

## Part 4: Format & Output System

### 4.1 Format Pipeline (Q.9)

The `format` system provides a pipeline for binding values and producing output.

```
[format PIPELINE]
= clXlM PIPELINE(
```

A pipeline chains substitution and output elements, terminated by `end` (`PQWnX`):
```
[format [substitute.number VAR [substitute.string SYM [output.string EXPR end]]]]
```

### 4.2 Pipeline Elements

| Element | Custom | Purpose |
|---------|--------|---------|
| `end` | `PQWnX` | Terminal — returns accumulated value |
| `substitute.number` | `QPMFunPXfkPXzP` | Binds a number variable |
| `substitute.string` | `QPMFunPXfz9XP5` | Binds a string/symbol variable |
| `output.number` | `QPPuMnXfkPXzP` | Outputs a number |
| `output.string` | `QPPuMnXfz9XP5` | Outputs a string/symbol |
| `output.image` | `QPPuMnXfkn<lW` | Outputs a bitmap image |

**Reduction behavior:**
```
[format end]                                → accumulated value
[format [output.number N REST]]             → outputs N, continues with REST
[format [output.string S REST]]             → outputs S, continues with REST
[format [substitute.number VAR REST]]       → binds number VAR, continues with REST
[format [substitute.string SYM REST]]       → binds string SYM, continues with REST
```

### 4.3 Print Command

```
[print EXPR]             → prints single value
[print EXPR [ARGS]]      → prints with arguments substituted
```

**Examples from server:**
```
[print [format [output.number 1 end]]]        → [result [1]]
[print [format [output.number 2 end]]]        → [result [2]]
```

When printing a format expression with arguments:
```
[print param [3 (definition)]]  → outputs "(definitiondefinitiondefinition)"
[print param [5 (true)]]        → outputs "(truetruetruetruetrue)"
```

### 4.4 Image Output — Diamond Structure (Q.10)

The `diamond` (`MPWnX-P`) is a 4-way branching construct for recursive image computation:

```
[diamond CONDITION QUADRANT_A QUADRANT_B QUADRANT_C QUADRANT_D]
= cMPWnX-P COND QA QB QC QD(
```

| Argument | Quadrant | Coordinate adjustment |
|----------|----------|-----------------------|
| CONDITION | Boolean determining pixel value | — |
| QUADRANT_A | | m-1, z-1 |
| QUADRANT_B | | m-1, z+1 |
| QUADRANT_C | | m+1, z-1 |
| QUADRANT_D | | m+1, z+1 |

**Image output element:**
```
[format [output.image EXPR end]]
```

Evaluates EXPR as an image generator function. **The resolution argument is mandatory** — `[print <format> [N]]` renders an NxN image. Printing without `[N]` returns an error.

**Verified examples:**
```
[print <format> [2]]  → 2x2 image (single # at z>m)
[print <format> [8]]  → 8x8 right triangle pattern
```

**Coordinate system:**
- `var`: scale factor (doubles each recursion level)
- `m`: row coordinate
- `z`: column coordinate
- Each recursion level: coordinates double via `[input [var m z] [array var*2 m*2 z*2] ...]`
- Initial call: `[call func 1 0 0]`

**Standard recursive pixel function pattern:**
```
[recursive [function [self] [function [var m z]
  [input [var m z]
    [array [multiply var 2] [multiply m 2] [multiply z 2]]
    [diamond CONDITION
      [call self var [subtract m 1] [subtract z 1]]
      [call self var [subtract m 1] [add z 1]]
      [call self var [add m 1] [subtract z 1]]
      [call self var [add m 1] [add z 1]]]]]]]
```

---

## Part 5: Graph Encoding System (l-series)

### 5.1 Binary Trees & Postorder Notation (l.0)

All graph structures are binary trees:
- **Leaf**: a single character (combinator type)
- **Application** `[case X Y]`: apply X to Y

**graph.symbol encoding (postorder):**
```
encode(leaf)       → single character for that leaf type
encode([case X Y]) → encode(X) + encode(Y) + '-'
```

**Decoding** uses a stack algorithm:
1. Read characters left to right
2. On leaf character → push to stack
3. On `-` → pop Y (top), pop X (second), push `(X Y)`
4. Final stack element is the decoded tree

**Worked example: `lll--ll--`**
```
Position 1: l → push l                          stack: [l]
Position 2: l → push l                          stack: [l, l]
Position 3: l → push l                          stack: [l, l, l]
Position 4: - → pop Y=l, X=l → (ll)             stack: [l, (ll)]
Position 5: - → pop Y=(ll), X=l → (l(ll))       stack: [(l(ll))]
Position 6: l → push l                          stack: [(l(ll)), l]
Position 7: l → push l                          stack: [(l(ll)), l, l]
Position 8: - → pop Y=l, X=l → (ll)             stack: [(l(ll)), (ll)]
Position 9: - → pop Y=(ll), X=(l(ll)) → ((l(ll))(ll))  stack: [((l(ll))(ll))]
```
Result: `[case [case l [case l l]] [case l l]]`

**Validity check:** A valid graph.symbol has exactly (N-1) dashes for N leaves, yielding a single tree.

### 5.2 Leaf Types & Combinator Rules

| Combinator | Custom Name | Leaf Char | Full graph.symbol | Length |
|-----------|-------------|-----------|-------------------|--------|
| edge_mark | `l` | `l` | `l` | 1 |
| K (named) | `XnkPX` | `X` | `lll--lll--l-l-l-l-l--` | 21 |
| I (equals) | `D5PM` | `D` | `ll-l-lll--lll--l-l-l-l-l---` | 27 |
| S (the_call) | `QPkPM9X` | `k` | `llll-ll-ll---llll-ll-------l-l-` | 31 |

### 5.3 Graph Reduction Rules (l.1)

Each combinator has a specific reduction rule:

**edge_mark (`l`) — 3-argument:**
```
[case [case [case l A] B] C]
→ [case [case A C] [case B [case named C]]]
```

**K / named (`XnkPX`) — 2-argument (select first):**
```
[case [case named A] B] → A
```

**I / equals (`D5PM`) — 1-argument (identity):**
```
[case equals A] → A
```

**S / the_call (`QPkPM9X`) — 3-argument (substitution):**
```
[case [case [case the_call F] G] X]
→ [case [case F X] [case G X]]
```

**In standard notation (SKI combinators):**
```
K x y   = x
I x     = x
S f g x = (f x)(g x)
```

### 5.4 Compact Encoding — equals.named.the_call.graph.symbol (l.2)

The compact encoding replaces full combinator graph.symbol patterns with single characters:

| Full Pattern (leaf) | Compact | Combinator |
|---------------------|---------|------------|
| `lll--lll--l-l-l-l-l--` (21 chars) | `X` | K |
| `ll-l-lll--lll--l-l-l-l-l---` (27 chars) | `D` | I |
| `llll-ll-ll---llll-ll-------l-l-` (31 chars) | `k` | S |

The `-` (application marker) is preserved. Compact strings use only: `k`, `D`, `X`, `-`.

**Conversion:**
- **Expand** (compact → graph.symbol): Replace each `X`/`D`/`k` with its full pattern. `-` stays `-`.
- **Compress** (graph.symbol → compact): Parse tree, match leaf subtrees to known patterns, replace with compact chars.

**Parsing compact notation** uses the same stack algorithm as graph.symbol:

Example: `kkD-XXD----XXD---`
```
k→S, k→S, D→I, -→(SI), X→K, X→K, D→I, -→(KI), -→K(KI),
-→(SI)(K(KI)), -→S((SI)(K(KI))), X→K, X→K, D→I, -→(KI), -→K(KI),
-→S((SI)(K(KI)))(K(KI))
```
Result: `S((SI)(K(KI)))(K(KI))`

### 5.5 Data Encoding in SKI (l.3)

**Scott-encoded booleans and pairs:**

| Value | SKI Expression | Compact Encoding |
|-------|---------------|-----------------|
| false / nil | `KI` | `XD-` |
| true | `S(KK)I` | `kXX--D-` |

**Scott encoding behavior:**
```
true   f g = f            (= K — selects first argument)
false  f g = g            (= KI — selects second argument)
pair a b f g = f a b      (applies first continuation to both elements)
nil    f g = g             (= KI — base case, selects second)
```

**Pair encoding:**
```
pair(A, B) = S(KK)(S(SI(KA))(KB))
```
Compact: `kXX--kkD-X<A>---X<B>---`

Where `<A>` and `<B>` are the compact encodings of A and B.

**Accessing elements:**
```
pair(a, b)(K)  = a     (first element)
pair(a, b)(KI) = b     (second element)
```

**Number encoding (binary, LSB-first):**

Numbers are linked lists of bits using pairs, with LSB first:
```
0 = pair(false, nil)                 = pair(KI, KI)
1 = pair(true,  nil)                 = pair(S(KK)I, KI)
2 = pair(false, pair(true, nil))
3 = pair(true,  pair(true, nil))
n = pair(n%2==1 ? true : false,  encode(n>>1))
```

Decoding: read bits from the pair chain: `bit0 + 2*bit1 + 4*bit2 + ...`

**Example — encoding 3 (binary 11, LSB-first [1,1]):**
```
3 = pair(true, pair(true, pair(false, false)))
  = pair(true, pair(true, 0))

Compact: kXX--kkD-XkXX--D----XkXX--kkD-XkXX--D----XkXX--kkD-XXD----XXD----------
```

Verification: bit0=true(1), bit1=true(1) → 1 + 2x1 = 3

**Operation encoding:**

The graph.symbol of an expression `[OP A B]` follows postorder:
```
encode([OP A B]) = encode(OP) + encode(A) + '-' + encode(B) + '--'
```

### 5.6 Bracket Abstraction (l.4)

Bracket abstraction converts lambda terms to SKI combinator expressions.

**Algorithm — three rules for abstracting variable z from a term:**

| Case | Condition | Rule | Compact Result |
|------|-----------|------|----------------|
| Identity | term IS z | `[z] z = I` | `D` |
| Constant | z not free in term M | `[z] M = K M` | `X<M>-` |
| Application | term is (M N) | `[z] (M N) = S([z]M)([z]N)` | `k<[z]M><[z]N>-` |

Apply recursively for nested lambdas (innermost variable first).

**Examples:**
```
[function [z] z]
  → [z]z = I
  Compact: D

[function [z] [call z z]]
  → [z](z z) = S([z]z)([z]z) = S I I
  Compact: kD-D-

[function [z.1] [function [z.2] [call z.1 [call z.1 z.2]]]]
  Inner: [z.2](z.1 (z.1 z.2))
       = S([z.2]z.1)([z.2](z.1 z.2))
       = S(K z.1)(S([z.2]z.1)([z.2]z.2))
       = S(K z.1)(S(K z.1) I)
  Outer: [z.1](S(K z.1)(S(K z.1) I))
       = ... (apply rules recursively)
```

**Function code representation in l-series:**

In the graph encoding system, lambda terms are represented using:
- `function` (`9XPX`) → lambda abstraction
- `call` (`kPM9X`) → application
- match variables → bound variables

The bracket abstraction process converts these to pure `l`/`-` graph.symbol strings (or compact `X`/`D`/`k`/`-` strings).

---

## Part 6: Problem Solutions

### Q.0: Printing (the_value of param)

**Title:** `QfDlu QnPX9M QnQ QfnQ&` = Q.0: the_value of param.nQ&

**Topic:** Learning the `print` command and the format/output system.

**Problem statement (from server):**
The server introduces the `print` command. Key points:
- `[print VALUE]` outputs the value. If valid: `[result VALUE]`. If invalid: `[error VALUE]`.
- Example: `[print [format [output.number 1 end]]]` is a command to print; `[format [output.number 1 end]]` is the value being printed, which evaluates to `1`.
- The result of `[print [format [output.number 1 end]]]` is `[result [1]]`, and `"1"` is what `[format [output.number 1 end]]` evaluates to.
- A 256x256 bitmap graph of the expression is displayed.

**Question:** "The result of printing the next value is what number?" with `[format [output.number 2 end]]`.

**Answer:** `PXz` (= **2**)

**Server Commands:**
```
View:   czMnX e-nQPz QfDlu QnPX9M QnQ QfnQ&8(
Answer: cznQPX ePX{nQ QfDlu8 PXz(
```

**Explanation:**
The format expression `[format [output.number 2 end]]` evaluates to `2`. This is confirmed by the server showing `[result [2]]` when printed. The user needed to understand that `output.number` outputs a number value, and the pipeline terminates at `end`.

---

### Q.1: Code and Reduction Rules

**Title:** `QfXn& QnQPW9- PQ,P QPwF9zPX QnQ95P` = Q.1: the_code and evaluation the_rule

**Topic:** Understanding how code expressions are reduced step by step.

**Problem statement (from server):**
- Code is a tree of elements. `[add 1 [multiply 2 3]]` consists of elements `1`, `add`, `multiply`, `2`, `3`.
- Evaluation rules have the form `result <- original`. Example: `"3" <- "[add 1 2]"`.
- Rules apply to sub-expressions: `"[multiply 3 4]" <- "[multiply [add 1 2] 4]"`.
- Sub.match applies all rules: `"param.1" <- "[if true param.1 param.2]"` means `"1" <- "[if true 1 2]"`.
- `[format [output.number EXPR end]]` evaluates EXPR and outputs its numeric value.

**Question:** "The result of evaluation of the next code is what number?" with `[if true [add 1 2] [multiply 3 4]]`.

**Answer:** `n9u` (= **3**)

**Server Commands:**
```
View:   czMnX e-nQPz QfXn& QnQPW9- PQ,P QPwF9zPX QnQ95P8(
Answer: cznQPX ePX{nQ QfXn&8 n9u(
```

**Explanation:**
```
[if true [add 1 2] [multiply 3 4]]
  rule: "param.1" <- "[if true param.1 param.2]"
→ [add 1 2]
  rule: "3" <- "[add 1 2]"
→ 3
```
Two reduction steps: first `if true` selects the first branch `[add 1 2]`, then `add` computes 1+2=3.

---

### Q.2: Number Operations

**Title:** `QfPXz kPXzP Qnl-5P5` = Q.2: number operation

**Topic:** Arithmetic operations: add, subtract, multiply.

**Problem statement (from server):**
Introduces the three arithmetic operators and their reduction rules. The example shows:
- `"3" <- "[add 1 2]"` (this is the evaluation rule)
- Applying this rule: `"[multiply 3 4]" <- "[multiply [add 1 2] 4]"`

**Question:** Evaluate `[multiply [add 1 2] [add 1 2]]` (or similar nested expression).

**Answer:** `uPXzlQ` (= **9**)

**Server Commands:**
```
View:   czMnX e-nQPz QfPXz kPXzP Qnl-5P58(
Answer: cznQPX ePX{nQ QfPXz8 uPXzlQ(
```

**Explanation:**
```
[multiply [add 1 2] 3]
→ [multiply 3 3]   (add 1 2 = 3)
→ 9                 (multiply 3 3 = 9)
```

---

### Q.3: Conditionals

**Title:** `Qfn9u CnQu9 QPW9-` = Q.3: if code

**Topic:** Conditional expressions, comparison operators.

**Problem statement (from server):**
Introduces `if`, `equals`, `greater` and demonstrates nested conditional evaluation.

**Question:** Evaluate the nested conditional expression.

**Answer:** `zPQ` (= **4**)

**Server Commands:**
```
View:   czMnX e-nQPz Qfn9u CnQu9 QPW9-8(
Answer: cznQPX ePX{nQ Qfn9u8 zPQ(
```

**Explanation:**
```
[if [greater 2 1] [if [equals 2 1] 3 4] 5]
→ [if true [if [equals 2 1] 3 4] 5]    (2 > 1 = true)
→ [if [equals 2 1] 3 4]                 (if true → first branch)
→ [if false 3 4]                         (2 = 1 is false)
→ 4                                      (if false → second branch)
```

---

### Q.4: Functions

**Title:** `QfzPQ Qn9XPX` = Q.4: the_function

**Topic:** Function definition (`function`) and application (`call`).

**Problem statement (from server):**
Write a function that computes absolute value. Functions are defined with `[function [var] body]` and called with `[call func arg]`. Examples show that calling the function with positive and negative inputs should return the positive value.

**Question:** What function computes absolute value?

**Answer:**
```
[function [var] [if [greater var 0] var [subtract 0 var]]]
```
Custom: `c9XPX ck( cCnQu9 c,lQ{ k Dlu( k cMPQ Dlu k((((`

**Server Commands:**
```
View:   czMnX e-nQPz QfzPQ Qn9XPX8(
Answer: cznQPX ePX{nQ QfzPQ8 c9XPX ck( cCnQu9 c,lQ{ k Dlu( k cMPQ Dlu k(((((
```

**Explanation:**
The absolute value function: if `var > 0`, return `var`; otherwise return `0 - var` (negation by subtraction).

---

### Q.5: Recursive Functions

**Title:** `QfuPX wlQWPM Qn9XPX` = Q.5: recursive the_function

**Topic:** Recursion via the `recursive` wrapper.

**Problem statement (from server):**
Write a recursive function for exponentiation (power). The `recursive` element enables self-reference: `[call param [recursive param]] <- [recursive param]`. The server provides an example of recursive summation:
```
[recursive [function [self] [function [var]
  [if [greater 0 var] 0 [add var [call self [subtract var 1]]]]]]]
```

**Question:** What function computes power (base^exponent)?

**Examples:** `[call param 2 3]` → 8, `[call param -5 3]` → -125, `[call param 4 1]` → 4, `[call param 0 2]` → 0. The function takes `[z.0 z.1]` as 2 arguments.

**Answer:**
```
[recursive [function [self] [function [z.0 z.1]
  [if [greater 1 z.1]
    1
    [multiply z.0 [call self z.0 [subtract z.1 1]]]]]]]
```
Custom: `cwlQWPM c9XPX c9( c9XPX czfDlu zfXn&( cCnQu9 c,lQ{ Xn& zfXn&( Xn& cXP9X zfDlu ckPM9X 9 zfDlu cMPQ zfXn& Xn&((((((((`

**Server Commands:**
```
View:   czMnX e-nQPz QfuPX wlQWPM Qn9XPX8(
Answer: cznQPX ePX{nQ QfuPX8 cwlQWPM c9XPX c9( c9XPX czfDlu zfXn&( cCnQu9 c,lQ{ Xn& zfXn&( Xn& cXP9X zfDlu ckPM9X 9 zfDlu cMPQ zfXn& Xn&(((((((((
```

**Explanation:**
Computes z.0 ^ z.1: base case `z.1 < 1` returns 1. Recursive case: `z.0 * self(z.0, z.1 - 1)`. This gives 2^3 = 2*2*2*1 = 8.

---

### Q.6: Arrays

**Title:** `QfuPXn& Qnz-n5` = Q.6: the_array

**Topic:** Array construction and destructuring.

**Problem statement (from server):**
Write a function that takes a 2-element array and returns the elements sorted in descending order (by greater). Examples:
- `[call param [array 1 2]]` → `[array 2 1]`
- `[call param [array 2 1]]` → `[array 2 1]`
- `[call param [array -2 3]]` → `[array 3 -2]`
- `[call param [array 0 -4]]` → `[array 0 -4]`

**Question:** What function compares numbers by greater and sorts?

**Answer:**
```
[function [var] [input [z.0 z.1] var
  [if [greater z.0 z.1]
    [array z.0 z.1]
    [array z.1 z.0]]]]
```
Custom: `c9XPX ck( czPX czfDlu zfXn&( k cCnQu9 c,lQ{ zfDlu zfXn&( cz-n5 zfDlu zfXn&( cz-n5 zfXn& zfDlu(((((`

**Server Commands:**
```
View:   czMnX e-nQPz QfuPXn& Qnz-n58(
Answer: cznQPX ePX{nQ QfuPXn&8 c9XPX ck( czPX czfDlu zfXn&( k cCnQu9 c,lQ{ zfDlu zfXn&( cz-n5 zfDlu zfXn&( cz-n5 zfXn& zfDlu(((((
```

**Explanation:**
Destructures the array into z.0 and z.1 using `input`. If z.0 > z.1, keep order; otherwise swap. This produces descending-order sorting for 2-element arrays.

---

### Q.7: Lists and Fold

**Title:** `QfuPXPXz Qnun-` = Q.7: list

**Topic:** List construction, fold operation.

**Problem statement (from server):**
- `empty_stack` = nil (0-element stack)
- `[push HEAD TAIL]` = cons (prepend to stack)
- `[pop LIST [z.1 z.2 BODY] ACC]` = fold (process list)
- Fold rules: `[pop empty_stack F ACC] → ACC`, `[pop [push H T] [z.1 z.2 BODY] ACC] → BODY[z.1=H, z.2=T]`

**Question:** What function computes the sum of all numbers in a list?

**Examples:** `[call param [push 1 [push 2 [push 3 empty_stack]]]]` → 6, `[call param [push 6 [push -5 [push 9 empty_stack]]]]` → 10.

**Answer:**
```
[recursive [function [self] [function [var]
  [pop var [z.1 z.2 [add z.1 [call self z.2]]] 0]]]]
```
Custom: `cwlQWPM c9XPX c9( c9XPX ck( c-PuuPX k czfXn& zfPXz cQPM zfXn& ckPM9X 9 zfPXz((( Dlu(((((`

**Server Commands:**
```
View:   czMnX e-nQPz QfuPXPXz Qnun-8(
Answer: cznQPX ePX{nQ QfuPXPXz8 cwlQWPM c9XPX c9( c9XPX ck( c-PuuPX k czfXn& zfPXz cQPM zfXn& ckPM9X 9 zfPXz((( Dlu(((((
```

**Explanation:**
Recursively sums a list: fold over the list, for each element add head `z.1` to `[call self z.2]` (recursive sum of tail). Base case (nil): accumulator 0. Note: fold gives raw tail in `z.2`, so `[call self z.2]` is needed for recursion.

---

### Q.8: Symbols and String Reversal

**Title:** `QfuPXn9u Qnz9XP5` = Q.8: the_symbol

**Topic:** Symbols/strings as stacks of character codes.

**Problem statement (from server):**
- `(` and `)` enclose a symbol value.
- `[format [output.string SYM end]]` evaluates to the symbol.
- Example: `(example)` ← `[print [format [output.string (example) end]]]`
- Strings are internally stacks of numbers (character codes).
- First character extraction: `[function [z9] [pop z9 [z.0 z.1 [push z.0 empty_stack]] empty_stack]]`

**Question:** What function computes the reverse of a string?

**Examples:** `[call param (definition)]` → `(DQlWlz)`, `[call param (true)]` → `(false)`. (Note: these reversed strings are in the custom character set.)

**Answer:**
```
[function [var]
  [call [recursive [function [self] [function [z.0 z.1]
    [pop z.0 [z.2 z.3 [call self z.3 [push z.2 z.1]]] z.1]]]]
  var nil]]
```
Custom: `c9XPX ck( ckPM9X cwlQWPM c9XPX c9( c9XPX czfDlu zfXn&( c-PuuPX zfDlu czfPXz zfn9u ckPM9X 9 zfn9u cXn-PX zfPXz zfXn&((( zfXn&(((( k kQPM(((`

**Server Commands:**
```
View:   czMnX e-nQPz QfuPXn9u Qnz9XP58(
Answer: cznQPX ePX{nQ QfuPXn9u8 c9XPX ck( ckPM9X cwlQWPM c9XPX c9( c9XPX czfDlu zfXn&( c-PuuPX zfDlu czfPXz zfn9u ckPM9X 9 zfn9u cXn-PX zfPXz zfXn&((( zfXn&(((( k kQPM(((
```

**Explanation:**
List reversal using an accumulator pattern:
1. Wrap the logic in a function that takes `var` (the string) and calls the recursive helper with `var` and `nil` (empty accumulator).
2. The recursive helper takes `z.0` (remaining list) and `z.1` (accumulator).
3. Fold: for each element, pop head `z.2` and tail `z.3`, then recurse on tail with `[push z.2 z.1]` (prepend head to accumulator).
4. Base case (empty list): return accumulator `z.1`.

Since strings are lists of character codes, list reversal = string reversal.

---

### Q.9: Substitute and Output

**Title:** `QfuPXzlQ QPMFunPX PQ,P QPPuMnX` = Q.9: substitute and output

**Topic:** The format pipeline with substitution and output elements.

**Problem statement (from server):**
Introduces `substitute.number`, `substitute.string`, `output.string`, `output.number`, and `end`. The question asks: given a number `var` and a symbol `su`, produce a format expression that repeats `su` exactly `var` times.

**Examples:**
- `[print param [3 (definition)]]` → `(definitiondefinitiondefinition)` — repeat "definition" 3 times
- `[print param [5 (true)]]` → `(truetruetruetruetrue)` — repeat "true" 5 times

**Question:** What format value, when printed with a number and a string, repeats the string that many times?

**Answer:**
```
[format [substitute.number var [substitute.string su [output.string
  [call [recursive [function [self.0] [function [var.0 su.0 su.1]
    [if [greater var.0 0]
      [call [recursive [function [self.1] [function [su.2 su.3]
        [pop su.2 [z.0 z.1 [push z.0 [call self.1 z.1 su.3]]] su.3]]]]
        su.0 [call self.0 [subtract var.0 1] su.0 su.1]]
      su.1]]]]
  var su nil] end]]]]]
```

Custom:
```
clXlM cQPMFunPXfkPXzP k cQPMFunPXfz9XP5 z9 cQPPuMnXfz9XP5
ckPM9X cwlQWPM c9XPX c9fDlu( c9XPX ckfDlu z9fDlu z9fXn&(
cCnQu9 c,lQ{ kfDlu Dlu(
ckPM9X cwlQWPM c9XPX c9fXn&( c9XPX cz9fPXz z9fn9u(
c-PuuPX z9fPXz czfDlu zfXn& cXn-PX zfDlu ckPM9X 9fXn& zfXn& z9fn9u(((
z9fn9u(((( z9fDlu ckPM9X 9fDlu cMPQ kfDlu Xn&( z9fDlu z9fXn&((
z9fXn&(((( k z9 kQPM( PQWnX(((((
```

**Server Commands:**
```
View:   czMnX e-nQPz QfuPXzlQ QPMFunPX PQ,P QPPuMnX8(
Answer: cznQPX ePX{nQ QfuPXzlQ8 clXlM cQPMFunPXfkPXzP k cQPMFunPXfz9XP5 z9 cQPPuMnXfz9XP5 ckPM9X cwlQWPM c9XPX c9fDlu( c9XPX ckfDlu z9fDlu z9fXn&( cCnQu9 c,lQ{ kfDlu Dlu( ckPM9X cwlQWPM c9XPX c9fXn&( c9XPX cz9fPXz z9fn9u( c-PuuPX z9fPXz czfDlu zfXn& cXn-PX zfDlu ckPM9X 9fXn& zfXn& z9fn9u((( z9fn9u(((( z9fDlu ckPM9X 9fDlu cMPQ kfDlu Xn&( z9fDlu z9fXn&(( z9fXn&(((( k z9 kQPM( PQWnX(((((
```

**Explanation:**
This uses two levels of recursion (double recursion pattern):
1. **Outer loop** (`self.0`): counts down `var.0` from N to 0. Each iteration appends one copy of the string.
2. **Inner loop** (`self.1`): copies the string `su.0` character by character (via fold/pop), appending each character to the front of the accumulated result from the recursive outer call.
3. **Base case**: when `var.0 <= 0`, return `su.1` (the accumulated result, initially `nil`).

The variable naming uses indexed prefixes:
- `var.0` (`kfDlu`): the number counter
- `su.0` (`z9fDlu`): the original string
- `su.1` (`z9fXn&`): the accumulated result string
- `self.0` (`9fDlu`): outer recursive self-reference
- `self.1` (`9fXn&`): inner recursive self-reference

---

### Q.10: Image Output

**Title:** `Qfwn- kn<lW QPPuMnX` = Q.10: image output

**Topic:** Bitmap image generation using the diamond branching structure.

**Problem statement (from server):**
Introduces `MPWnX-P` (diamond) — a 4-way branching structure for recursive pixel computation. Shows example images:
- 2x2 image: single `#` in top-left
- 8x8 image: right triangle pattern (staircase of `#`)

The diamond recursively subdivides into 4 quadrants, and the condition determines whether each pixel is on (`#`) or off (` `).

**Question:** What value produces the given triangle image pattern?

**Answer:**
```
[format [output.image
  [call [recursive [function [self] [function [var m z]
    [input [var m z]
      [array [multiply var 2] [multiply m 2] [multiply z 2]]
      [diamond [greater [subtract z m] 0]
        [call self var [subtract m 1] [subtract z 1]]
        [call self var [subtract m 1] [add z 1]]
        [call self var [add m 1] [subtract z 1]]
        [call self var [add m 1] [add z 1]]]]]]]
  1 0 0] end]]
```

Custom:
```
clXlM cQPPuMnXfkn<lW ckPM9X cwlQWPM c9XPX c9( c9XPX ck M z(
czPX ck M z( cz-n5 cXP9X k PXz( cXP9X M PXz( cXP9X z PXz((
cMPWnX-P c,lQ{ cMPQ z M( Dlu(
ckPM9X 9 k cMPQ M Xn&( cMPQ z Xn&((
ckPM9X 9 k cMPQ M Xn&( cQPM z Xn&((
ckPM9X 9 k cQPM M Xn&( cMPQ z Xn&((
ckPM9X 9 k cQPM M Xn&( cQPM z Xn&((((((( Xn& Dlu Dlu( PQWnX(((
```

**Server Commands:**
```
View:   czMnX e-nQPz Qfwn- kn<lW QPPuMnX8(
Answer: cznQPX ePX{nQ Qfwn-8 clXlM cQPPuMnXfkn<lW ckPM9X cwlQWPM c9XPX c9( c9XPX ck M z( czPX ck M z( cz-n5 cXP9X k PXz( cXP9X M PXz( cXP9X z PXz(( cMPWnX-P c,lQ{ cMPQ z M( Dlu( ckPM9X 9 k cMPQ M Xn&( cMPQ z Xn&(( ckPM9X 9 k cMPQ M Xn&( cQPM z Xn&(( ckPM9X 9 k cQPM M Xn&( cMPQ z Xn&(( ckPM9X 9 k cQPM M Xn&( cQPM z Xn&((((((( Xn& Dlu Dlu( PQWnX(((
```

**Explanation:**
The pixel function generates a right triangle where pixel is ON when `z - m > 0` (i.e., column > row).

How the diamond recursion works:
1. Start with `(var=1, m=0, z=0)`.
2. Each level: double all coordinates via `[input [var m z] [array var*2 m*2 z*2] ...]`.
3. The diamond branches into 4 sub-quadrants by adjusting m and z by +-1.
4. The condition `[greater [subtract z m] 0]` determines the pixel value.
5. The resolution is set by the print argument: `[print param [N]]` for NxN image.

Note: `var`, `m`, `z` here are used as raw variable names (single characters), not the standard `k`/`zf` indexed variables. `M` and `z` in the custom encoding happen to be single characters that aren't structural characters.

---

### l.0: Edge Mark (Graph Basics)

**Title:** l.0 (full chapter title unknown — view command with just `lfDlu` returns error)

**Topic:** Introduction to graph encoding.

**Problem statement:**
The server shows a 256x256 bitmap graph image. The user must identify the graph.symbol encoding of the displayed edge_mark node structure.

**Question:** What is the graph.symbol of the displayed graph?

**Answer:** `lll--ll--` (graph.symbol notation)

**Server Commands:**
```
View:   (chapter title unknown — czMnX e-nQPz lfDlu8( returns error)
Answer: cznQPX ePX{nQ lfDlu8 elll--ll--8(
```

**Explanation:**
Parsing `lll--ll--` via the stack algorithm:
```
l,l,l → [l,l,l]
-     → [l, (ll)]
-     → [(l(ll))]
l,l   → [(l(ll)), l, l]
-     → [(l(ll)), (ll)]
-     → [((l(ll))(ll))]
```
Result: `[case [case l [case l l]] [case l l]]`

This is the tree structure of the edge_mark node. The edge_mark (`l`) reduction rule is:
```
[case [case [case l A] B] C] → [case [case A C] [case B [case named C]]]
```

---

### l.1: Graph Rewriting

**Title:** `lfXn& QP5959PX QnQ Qn&n-QPznk` = l.1: rewriting of the_graph

**Topic:** Applying graph reduction rules.

**Problem statement (from server):**
Shows two large images (128x128 and 64x64) depicting graph structures. The question asks: given graph `[case [case [case [case edge_mark edge_mark] edge_mark] named] [case edge_mark edge_mark]]`, what is the graph.symbol after rewriting to remove the `named` type?

**Question:** What is the graph.symbol after rewriting?

**Answer:** `ll-` (graph.symbol)

**Server Commands:**
```
View:   czMnX e-nQPz lfXn& QP5959PX QnQ Qn&n-QPznk8(
Answer: cznQPX ePX{nQ lfXn&8 ell-8(
```

**Explanation:**
`ll-` = `[case l l]` = application of edge_mark to edge_mark. The rewriting process applies the named rule `[case [case named A] B] → A` and edge_mark rule to simplify the original complex graph down to this minimal form.

---

### l.2: Compact Graph Encoding

**Title:** `lfPXz D5PMfXnkPXfQPkPM9X Qn&n-QPznk` = l.2: equals.named.the_call the_graph

**Topic:** SKI combinator encoding and compact notation.

**Problem statement (from server):**
Defines three graph types with their full and compact graph.symbol encodings:
- **named** (K): `lll--lll--l-l-l-l-l--` → compact `X`
- **equals** (I): `ll-l-lll--lll--l-l-l-l-l---` → compact `D`
- **the_call** (S): `llll-ll-ll---llll-ll-------l-l-` → compact `k`

Each has its specific SKI reduction rule. The question asks for the `equals.named.the_call.graph.symbol` (compact encoding) of a given long graph.symbol string.

**Question:** What is the compact encoding of the given graph.symbol?

**Answer:** `kkD-XXD----XXD---` (compact encoding)

**Server Commands:**
```
View:   czMnX e-nQPz lfPXz D5PMfXnkPXfQPkPM9X Qn&n-QPznk8(
Answer: cznQPX ePX{nQ lfPXz8 ekkD-XXD----XXD---8(
```

**Explanation:**
Stack-based parsing of `kkD-XXD----XXD---`:
```
k → push S                    [S]
k → push S                    [S, S]
D → push I                    [S, S, I]
- → pop I,S → (SI)            [S, (SI)]
X → push K                    [S, (SI), K]
X → push K                    [S, (SI), K, K]
D → push I                    [S, (SI), K, K, I]
- → pop I,K → (KI)            [S, (SI), K, (KI)]
- → pop (KI),K → K(KI)        [S, (SI), K(KI)]
- → pop K(KI),(SI) → (SI)(K(KI))  [S, (SI)(K(KI))]
- → pop (SI)(K(KI)),S → S((SI)(K(KI)))  [S((SI)(K(KI)))]
X → push K                    [S((SI)(K(KI))), K]
X → push K                    [S((SI)(K(KI))), K, K]
D → push I                    [S((SI)(K(KI))), K, K, I]
- → pop I,K → (KI)            [S((SI)(K(KI))), K, (KI)]
- → pop (KI),K → K(KI)        [S((SI)(K(KI))), K(KI)]
- → pop K(KI),S((SI)(K(KI))) → S((SI)(K(KI)))(K(KI))
```
Result: `S((SI)(K(KI)))(K(KI))`

---

### l.3: Encoding Items in SKI

**Title:** `lfn9u D5PMfXnkPXfQPkPM9Xf&n-QPznk QPul-FX QnPXPn` = l.3: equals.named.the_call.graph apply_to item

**Topic:** Encoding values (numbers, expressions) as compact SKI graph strings.

**Problem statement (from server):**
Shows how to encode data items using the compact `equals.named.the_call.graph.symbol` notation. Examples demonstrate encoding expressions like `[add 2 3]` as `(compact of "add")(compact of "2")-(compact of "3")--`.

**Question:** What is the compact encoding of the number 3?

**Answer:** `kXX--kkD-XkXX--D----XkXX--kkD-XkXX--D----XkXX--kkD-XXD----XXD----------`

**Server Commands:**
```
View:   czMnX e-nQPz lfn9u D5PMfXnkPXfQPkPM9Xf&n-QPznk QPul-FX QnPXPn8(
Answer: cznQPX ePX{nQ lfn9u8 ekXX--kkD-XkXX--D----XkXX--kkD-XkXX--D----XkXX--kkD-XXD----XXD----------8(
```

**Explanation:**
The number 3 in binary (LSB-first): bits = [1, 1].

Encoding step by step:
```
3 = pair(true, pair(true, pair(false, false)))
  = pair(true, pair(true, 0))

Building from inside out:
  0      = pair(false, nil)  = pair(KI, KI)
         Compact: kXX--kkD-XXD----XXD----

  pair(true, 0)
         = pair(S(KK)I, 0)
         Compact: kXX--kkD-XkXX--D----X<0>---

  3 = pair(true, pair(true, 0))
    Compact: kXX--kkD-XkXX--D----X<pair(true,0)>---
```

The pair formula in compact: `pair(A,B) = kXX--kkD-X<A>---X<B>---`

Verify: bit0=true(1), bit1=true(1) → 1 + 2*1 = 3.

---

### l.4: Bracket Abstraction

**Title:** `lfzPQ D5PMfXnkPXfQPkPM9Xf&n-QPznk QPul-FX 9XPXfQPW9-` = l.4: equals.named.the_call.graph apply_to function.code

**Topic:** Converting function code (lambda terms) to SKI combinators.

**Problem statement (from server):**
Defines how `equals.named.the_call.graph` encodes function code. The `zPXkPQ` (translation) process converts `function`/`call` expressions into case-based graph representations using bracket abstraction rules.

**Question:** What function.code does the given compact encoding represent, when it encodes `[format end]`?

**Answer:**
```
[function [z.1]
  [call [call z.1
    [function [z.2]
      [call [call z.2
        [function [z.3] [function [z.4] z.4]]
      ] [function [z.3] [function [z.4] z.4]]
      ]
    ]]
  [function [z.3] [function [z.4] z.4]]
  ]]
```

Custom:
```
c9XPX czfXn&( ckPM9X ckPM9X zfXn&
c9XPX czfPXz( ckPM9X ckPM9X zfPXz
c9XPX czfn9u( c9XPX czfzPQ( zfzPQ(((
c9XPX czfn9u( c9XPX czfzPQ( zfzPQ(((((
c9XPX czfn9u( c9XPX czfzPQ( zfzPQ(((((
```

**Server Commands:**
```
View:   czMnX e-nQPz lfzPQ D5PMfXnkPXfQPkPM9Xf&n-QPznk QPul-FX 9XPXfQPW9-8(
Answer: cznQPX ePX{nQ lfzPQ8 c9XPX czfXn&( ckPM9X ckPM9X zfXn& c9XPX czfPXz( ckPM9X ckPM9X zfPXz c9XPX czfn9u( c9XPX czfzPQ( zfzPQ((( c9XPX czfn9u( c9XPX czfzPQ( zfzPQ((((( c9XPX czfn9u( c9XPX czfzPQ( zfzPQ(((((
```

**Explanation:**
This is the bracket abstraction algorithm expressed as a function in the custom language. The function `[function [z.3] [function [z.4] z.4]]` appears repeatedly — it represents `KI` (= false/nil), which is the identity selector for the second argument. The structure pattern-matches on the three cases of bracket abstraction:
1. Variable → I
2. Constant → K applied to the constant
3. Application → S composed with recursive abstraction of both sides

---

## Part 7: Appendices

### 7.1 Variable Naming Conventions

| Prefix | Custom | Meaning | Example |
|--------|--------|---------|---------|
| var | `k` | General variable | `k` = var |
| param | `Q` | Parameter reference | `QfDlu` = param.0 |
| edge | `l` | Edge/graph reference | `lfXn&` = edge.1 |
| section | `z` | Section/local variable | `zfDlu` = z.0 |
| self | `9` | Recursive self-reference | `9` = self |
| g | `&` | Graph variable | `&fDlu` = g.0 |
| symbol | `z9` | Symbol variable | `z9fDlu` = z9.0 |

**Indexed variables use dot-number notation:**
```
kfDlu  = var.0      kfXn&  = var.1
zfDlu  = z.0        zfXn&  = z.1        zfPXz  = z.2        zfn9u = z.3        zfzPQ = z.4
9fDlu  = self.0     9fXn&  = self.1
z9fDlu = z9.0       z9fXn& = z9.1       z9fPXz = z9.2       z9fn9u = z9.3
```

### 7.2 Complete Vocabulary

**Operators & Constructs:**

| Custom | English | Category | Introduced |
|--------|---------|----------|------------|
| `QPM` | add | Arithmetic | Q.2 |
| `MPQ` | subtract | Arithmetic | Q.2 |
| `XP9X` | multiply | Arithmetic | Q.2 |
| `QPQuP` | divide | Arithmetic | — |
| `D5PM` | equals | Comparison | Q.3 |
| `,lQ{` | greater | Comparison | Q.3 |
| `PQ,P` | and | Boolean | Q.3 |
| `MnW` | or | Boolean | Q.3 |
| `PX,` | not | Boolean | Q.3 |
| `CnQu9` | if | Conditional | Q.3 |
| `9XPX` | function | Function def | Q.4 |
| `kPM9X` | call | Function call | Q.4 |
| `wlQWPM` | recursive | Recursion | Q.5 |
| `z-n5` | array | Array | Q.6 |
| `zPX` | input (destructure) | Destructuring | Q.6 |
| `Xn-PX` | push (cons) | List | Q.7 |
| `kQPM` | empty_stack (nil) | List | Q.7 |
| `-PuuPX` | pop (fold) | List | Q.7 |
| `lXlM` | format | Format | Q.9 |
| `PQWnX` | end | Format terminal | Q.9 |
| `QPMFunPX` | substitute | Format | Q.9 |
| `QPMFunPXfkPXzP` | substitute.number | Format | Q.9 |
| `QPMFunPXfz9XP5` | substitute.string | Format | Q.9 |
| `QPPuMnX` | output | Format | Q.0 |
| `QPPuMnXfkPXzP` | output.number | Format | Q.0 |
| `QPPuMnXfz9XP5` | output.string | Format | Q.8 |
| `QPPuMnXfkn<lW` | output.image | Format | Q.10 |
| `MPWnX-P` | diamond | Image branching | Q.10 |

**Graph Combinators:**

| Custom | English | SKI | Arity | Rule |
|--------|---------|-----|-------|------|
| `l` | edge_mark | — | 3 | `l A B C → (A C)(B (named C))` |
| `XnkPX` | named | K | 2 | `K x y → x` |
| `D5PM` | equals | I | 1 | `I x → x` |
| `QPkPM9X` | the_call | S | 3 | `S f g x → (f x)(g x)` |

**Graph Terminology:**

| Custom | English |
|--------|---------|
| `QP-n9zPX` | case (application node) |
| `&n-QPznk` | graph |
| `&n-QPznkfz9XP5` | graph.symbol (postorder encoding) |
| `&n-QPznkfQPW9-` | graph.code |
| `D5PMfXnkPXfQPkPM9X` | equals.named.the_call (SKI system) |
| `D5PMfXnkPXfQPkPM9Xf&n-QPznkfz9XP5` | equals.named.the_call.graph.symbol (compact) |

**Commands & Keywords:**

| Custom | English | | Custom | English |
|--------|---------|--|--------|---------|
| `zMnX` | view | | `kPX` | is |
| `znQPX` | answer | | `PX,` | not |
| `QnWPX` | print | | `PQ,P` | and |
| `zlWPX` | encode | | `MnW` | or |
| `wnz` | what | | `,lz` | for |
| `Qn9zn` | error | | `QnQ` | of |
| `MPQP` | result | | `lQu` | the |
| `zWP5M` | correct | | `XPX` | it |
| `M5PWz` | wrong | | `-nW` | next |
| `*l&Pk` | yes | | `kPXzP` | number |
| `PkPMn` | info | | `PX9M` | value |
| `PX{nQ` | question | | `QPW9-` | code |
| `-nQPz` | chapter | | `z9XP5` | string |
| `-5n5nQ` | example | | `kn<lW` | image |
| `zMn5` | title | | `un-` | stack |
| `zlWlQD` | definition | | `Q95P` | rule |
| `Xnkl` | about | | `Xnkl` | about |

**Morphological words (key ones):**

| Custom | English | | Custom | English |
|--------|---------|--|--------|---------|
| `QPn&&PX` | called | | `QnPX9M` | the_value |
| `QP&PXMP5` | execute | | `QPznQPX` | the_answer |
| `QPQnWPX` | printing | | `QPwF9zPX` | evaluation |
| `QPXnWPX` | the_print | | `QnQ95P` | the_rule |
| `QnQPW9-` | the_code | | `QnXPk` | meaning |
| `Qn9XPX` | the_function | | `QnkPXzP` | the_number |
| `Qnz-n5` | the_array | | `Qnun-` | list |
| `Qnz9XP5` | the_symbol | | `Qn&n-QPznk` | the_graph |
| `CFn-9<P` | argument | | `XnD0P` | from |
| `P0DnX` | to | | `wlQWPM` | recursive |

### 7.3 Quick Reference — Encoding & Decoding

**Number → Compact SKI:**
1. Convert to binary (LSB first): `[b0, b1, ...]`
2. Build nested pairs from innermost:
   - Start with `nil` = `XD-`
   - Wrap each bit: `pair(bit, acc)` where `true` = `kXX--D-`, `false` = `XD-`
   - `pair(A, B)` = `kXX--kkD-X<A>---X<B>---`

**Compact SKI → Full graph.symbol:**
- `X` → `lll--lll--l-l-l-l-l--`
- `D` → `ll-l-lll--lll--l-l-l-l-l---`
- `k` → `llll-ll-ll---llll-ll-------l-l-`
- `-` → `-`

**Lambda → SKI (Bracket Abstraction):**
1. `[z] z = I` → compact `D`
2. `[z] M = K M` (z not free) → compact `X<M>-`
3. `[z] (M N) = S([z]M)([z]N)` → compact `k<[z]M><[z]N>-`

---

## Part 8: stars.txt Analysis (Final Challenge)

### 8.1 File Statistics

- Size: 419,677,398 bytes
- Contains only `l` (209,838,699) and `-` (209,838,698) characters
- Leaves - dashes = 1 → valid single binary tree
- Encoding: `leaf=l`, `[case X Y] = encode(X) + encode(Y) + '-'`

### 8.2 Compression

Using SKI combinator patterns from l.2:
- `k` pattern (S, 31 chars): 6,708,653 occurrences
- `D` pattern (I, 27 chars): 2,875,571 occurrences
- `X` pattern (K, 21 chars): 5,658,387 occurrences
- Result: `stars_compact.txt` (30,485,221 chars of `k`, `D`, `X`, `-`)

### 8.3 Top-level Structure

```
(LEFT RIGHT)
```

- **LEFT** (~152K chars): `((S X)(K Y))` — applied to R gives `X R Y`
  - X: 61,418 chars
  - Y: 90,513 chars
- **RIGHT** (~30.3M chars): `((B^6 H) DATA_CHAIN)`
  - B^6 composition tower: `S(K(S(K(S(K(S(K(S(K H)))))))))`
  - H: ~30M chars (the main program)
  - DATA_CHAIN: 307,125 chars (25 items)

### 8.4 DATA_CHAIN (25 items)

| Index | Type | Size | Decoded Value |
|-------|------|------|---------------|
| 0 | B^7 + SI(KV) | 87 | pair (complex) |
| 1 | B^7(K) | 18 | Separator (K = constant) |
| 2 | B^7 + SI(KV) | 1,345 | List of 33 numbers: [13, 6, 19, 8, 5, 22, 1, 17, 5, 7, 8, 1, 17, 19, 8, 23, 22, 24, 8, 21, 22, 19, 8, 22, 8, 19, 6, 20, 6, 5, 16, 8, 2] |
| 3 | B^7(K) | 18 | Separator |
| 4 | B^7 + SI(KV) | 87 | pair (complex) |
| 5 | B^7(K) | 18 | Separator |
| 6 | B^0 | 231 | Complex structure (no B wrapper) |
| 7 | K | 1 | K leaf |
| 8 | K | 1 | K leaf |
| 9 | (large) | 301,617 | Recursive self-similar structure (26 levels deep) |
| 10–16 | K | 1 each | K leaves |
| 17 | B^0 + SI(KV) | 87 | pair (complex) |
| 18 | K | 1 | K leaf |
| 19 | B^0 + SI(KV) | 229 | List of 5 numbers: [21, 9, 22, 20, 19] |
| 20 | K | 1 | K leaf |
| 21 | B^0 + SI(KV) | 45 | Number: 0 |
| 22–23 | K | 1 each | K leaves |
| 24 | I | 1 | Terminal (identity) |

### 8.5 Character Mapping (Unsolved)

Number sequences from items 2 and 19 have not been successfully mapped to readable text. Attempted:
- a=1 mapping: no recognizable words
- ASCII offsets: values out of printable range

### 8.6 Status

- Compression, structure analysis, data extraction: complete
- Full decoding of item 9's recursive structure (301K chars): incomplete
- Character mapping for number sequences: unknown
- Expected answer format: unknown
