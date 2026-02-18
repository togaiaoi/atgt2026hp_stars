/// Lazy graph-reduction SKI combinator evaluator.
///
/// Arena-based allocation with sharing via indirection nodes.
/// Reads compact format (k=S, X=K, D=I, -=application).

use std::env;
use std::fs;
use std::io::{self, Write};
use std::process;

// Node tags
const APP: u8 = 0;
const S: u8 = 1;
const K: u8 = 2;
const I: u8 = 3;
const S1: u8 = 4;  // S applied to 1 arg
const S2: u8 = 5;  // S applied to 2 args
const K1: u8 = 6;  // K applied to 1 arg
const IND: u8 = 7; // Indirection (sharing/update)

const NIL: u32 = u32::MAX;

#[derive(Clone, Copy)]
struct Node {
    tag: u8,
    a: u32, // left child / first arg / indirection target
    b: u32, // right child / second arg
}

struct Arena {
    nodes: Vec<Node>,
}

impl Arena {
    fn new(capacity: usize) -> Self {
        Arena {
            nodes: Vec::with_capacity(capacity),
        }
    }

    #[inline]
    fn alloc(&mut self, tag: u8, a: u32, b: u32) -> u32 {
        let idx = self.nodes.len() as u32;
        self.nodes.push(Node { tag, a, b });
        idx
    }

    #[inline]
    fn follow(&self, mut idx: u32) -> u32 {
        loop {
            let n = &self.nodes[idx as usize];
            if n.tag != IND {
                return idx;
            }
            idx = n.a;
        }
    }

    /// Follow and also do path compression.
    #[inline]
    fn follow_mut(&mut self, idx: u32) -> u32 {
        let root = self.follow(idx);
        // Path compression
        let mut cur = idx;
        while cur != root {
            let n = &self.nodes[cur as usize];
            if n.tag != IND { break; }
            let next = n.a;
            self.nodes[cur as usize].a = root;
            cur = next;
        }
        root
    }

    /// Reduce to Weak Head Normal Form.
    fn whnf(&mut self, node: u32, fuel: &mut u64) -> u32 {
        let mut spine: Vec<u32> = Vec::with_capacity(256);
        let mut n = self.follow_mut(node);

        loop {
            if *fuel == 0 {
                return n;
            }

            let tag = self.nodes[n as usize].tag;

            match tag {
                APP => {
                    spine.push(n);
                    let a = self.nodes[n as usize].a;
                    n = self.follow_mut(a);
                    continue;
                }
                I if !spine.is_empty() => {
                    // I x -> x
                    *fuel -= 1;
                    let app = spine.pop().unwrap();
                    let x = self.follow_mut(self.nodes[app as usize].b);
                    self.nodes[app as usize].tag = IND;
                    self.nodes[app as usize].a = x;
                    n = x;
                    continue;
                }
                K if spine.len() >= 2 => {
                    // K x y -> x
                    *fuel -= 1;
                    let app1 = spine.pop().unwrap(); // K x
                    let app2 = spine.pop().unwrap(); // (K x) y
                    let x = self.follow_mut(self.nodes[app1 as usize].b);
                    self.nodes[app2 as usize].tag = IND;
                    self.nodes[app2 as usize].a = x;
                    self.nodes[app1 as usize].tag = K1;
                    self.nodes[app1 as usize].a = x;
                    n = x;
                    continue;
                }
                K1 if !spine.is_empty() => {
                    // (K x) y -> x
                    *fuel -= 1;
                    let app = spine.pop().unwrap();
                    let x = self.follow_mut(self.nodes[n as usize].a);
                    self.nodes[app as usize].tag = IND;
                    self.nodes[app as usize].a = x;
                    n = x;
                    continue;
                }
                S if spine.len() >= 3 => {
                    // S f g x -> f x (g x)
                    *fuel -= 1;
                    let app1 = spine.pop().unwrap(); // S f
                    let app2 = spine.pop().unwrap(); // (S f) g
                    let app3 = spine.pop().unwrap(); // ((S f) g) x
                    let f = self.follow_mut(self.nodes[app1 as usize].b);
                    let g = self.follow_mut(self.nodes[app2 as usize].b);
                    let x = self.nodes[app3 as usize].b; // keep sharing
                    let fx = self.alloc(APP, f, x);
                    let gx = self.alloc(APP, g, x);
                    let result = self.alloc(APP, fx, gx);
                    self.nodes[app3 as usize].tag = IND;
                    self.nodes[app3 as usize].a = result;
                    self.nodes[app1 as usize].tag = S1;
                    self.nodes[app1 as usize].a = f;
                    self.nodes[app2 as usize].tag = S2;
                    self.nodes[app2 as usize].a = f;
                    self.nodes[app2 as usize].b = g;
                    n = result;
                    continue;
                }
                S1 if spine.len() >= 2 => {
                    // (S f) g x -> f x (g x)
                    *fuel -= 1;
                    let app1 = spine.pop().unwrap(); // (S f) g
                    let app2 = spine.pop().unwrap(); // ((S f) g) x
                    let f = self.follow_mut(self.nodes[n as usize].a);
                    let g = self.follow_mut(self.nodes[app1 as usize].b);
                    let x = self.nodes[app2 as usize].b;
                    let fx = self.alloc(APP, f, x);
                    let gx = self.alloc(APP, g, x);
                    let result = self.alloc(APP, fx, gx);
                    self.nodes[app2 as usize].tag = IND;
                    self.nodes[app2 as usize].a = result;
                    self.nodes[app1 as usize].tag = S2;
                    self.nodes[app1 as usize].a = f;
                    self.nodes[app1 as usize].b = g;
                    n = result;
                    continue;
                }
                S2 if !spine.is_empty() => {
                    // (S f g) x -> f x (g x)
                    *fuel -= 1;
                    let app = spine.pop().unwrap();
                    let f = self.follow_mut(self.nodes[n as usize].a);
                    let g = self.follow_mut(self.nodes[n as usize].b);
                    let x = self.nodes[app as usize].b;
                    let fx = self.alloc(APP, f, x);
                    let gx = self.alloc(APP, g, x);
                    let result = self.alloc(APP, fx, gx);
                    self.nodes[app as usize].tag = IND;
                    self.nodes[app as usize].a = result;
                    n = result;
                    continue;
                }
                _ => {
                    // No reduction possible - return outermost remaining node
                    if !spine.is_empty() {
                        return spine[0];
                    }
                    return n;
                }
            }
        }
    }
}

/// Parse compact string into arena.
fn parse_compact(arena: &mut Arena, input: &[u8]) -> u32 {
    let mut stack: Vec<u32> = Vec::with_capacity(1024);
    for &c in input {
        match c {
            b'k' => stack.push(arena.alloc(S, NIL, NIL)),
            b'X' => stack.push(arena.alloc(K, NIL, NIL)),
            b'D' => stack.push(arena.alloc(I, NIL, NIL)),
            b'-' => {
                let y = stack.pop().expect("stack underflow on '-'");
                let x = stack.pop().expect("stack underflow on '-'");
                stack.push(arena.alloc(APP, x, y));
            }
            b'\n' | b'\r' | b' ' => {} // skip whitespace
            _ => {} // skip unknown
        }
    }
    assert_eq!(stack.len(), 1, "parse error: stack has {} elements", stack.len());
    stack[0]
}

/// Build false = KI
fn make_false(arena: &mut Arena) -> u32 {
    let k = arena.alloc(K, NIL, NIL);
    let i = arena.alloc(I, NIL, NIL);
    arena.alloc(APP, k, i)
}

/// Build true = S(KK)I
fn make_true(arena: &mut Arena) -> u32 {
    let k1 = arena.alloc(K, NIL, NIL);
    let k2 = arena.alloc(K, NIL, NIL);
    let kk = arena.alloc(APP, k1, k2);
    let s = arena.alloc(S, NIL, NIL);
    let skk = arena.alloc(APP, s, kk);
    let i = arena.alloc(I, NIL, NIL);
    arena.alloc(APP, skk, i)
}

/// Build 2-arg Scott pair: pair(A, B) = S(KK)(S(SI(KA))(KB))
/// pair(f)(g) = f(A)(B) — takes 2 continuation args
fn make_pair(arena: &mut Arena, a: u32, b: u32) -> u32 {
    // inner = S(SI(KA))(KB)
    let s1 = arena.alloc(S, NIL, NIL);
    let i1 = arena.alloc(I, NIL, NIL);
    let si = arena.alloc(APP, s1, i1);
    let k_a = arena.alloc(K, NIL, NIL);
    let ka = arena.alloc(APP, k_a, a);
    let si_ka = arena.alloc(APP, si, ka);
    let s2 = arena.alloc(S, NIL, NIL);
    let s_si_ka = arena.alloc(APP, s2, si_ka);
    let k_b = arena.alloc(K, NIL, NIL);
    let kb = arena.alloc(APP, k_b, b);
    let inner = arena.alloc(APP, s_si_ka, kb);
    // outer = S(KK)(inner)
    let s3 = arena.alloc(S, NIL, NIL);
    let k3 = arena.alloc(K, NIL, NIL);
    let k4 = arena.alloc(K, NIL, NIL);
    let kk = arena.alloc(APP, k3, k4);
    let s_kk = arena.alloc(APP, s3, kk);
    arena.alloc(APP, s_kk, inner)
}

/// Decode boolean: apply to two unique markers.
fn decode_bool(arena: &mut Arena, node: u32, fuel: u64) -> Option<bool> {
    let marker_t = arena.alloc(100, NIL, NIL);
    let marker_f = arena.alloc(101, NIL, NIL);
    let app1 = arena.alloc(APP, node, marker_t);
    let app2 = arena.alloc(APP, app1, marker_f);
    let mut f = fuel;
    let result = arena.whnf(app2, &mut f);
    let result = arena.follow(result);
    let tag = arena.nodes[result as usize].tag;
    if tag == 100 { Some(true) }
    else if tag == 101 { Some(false) }
    else { None }
}

/// Decode Scott-encoded binary number.
/// Numbers are pair chains: pair(bit, rest) with 0 = pair(false, nil).
/// Each pair is a 2-arg Scott pair: pair(f)(g) = f(A)(B).
fn decode_scott_num(arena: &mut Arena, node: u32, fuel: u64) -> Option<u64> {
    let mut bits: Vec<u8> = Vec::new();
    let mut current = node;
    let mut remaining = fuel;
    // Each pair extraction/bool decode is cheap (~10-20 steps for pre-built pairs)
    let fuel_per_op = (fuel / 200).max(10000);

    for _ in 0..64 {
        if remaining < fuel_per_op * 4 { break; }

        // Extract fst (the bit) using 2-arg pair extraction
        let mut f1 = fuel_per_op;
        let first = pair_fst(arena, current, &mut f1);
        remaining = remaining.saturating_sub(fuel_per_op - f1);

        // Decode the bit
        let bit_val = decode_bool(arena, first, fuel_per_op);
        match bit_val {
            Some(false) => {
                // This could be 0-terminator pair(false, nil)
                // Check if snd is nil (false)
                let mut f2 = fuel_per_op;
                let second = pair_snd(arena, current, &mut f2);
                remaining = remaining.saturating_sub(fuel_per_op - f2);

                let snd_is_nil = decode_bool(arena, second, fuel_per_op);
                if snd_is_nil == Some(false) {
                    // pair(false, nil) → end of number (0 terminator)
                    break;
                } else {
                    // pair(false, rest) → bit 0, continue
                    bits.push(0);
                    current = second;
                }
            }
            Some(true) => {
                // pair(true, rest) → bit 1
                bits.push(1);
                let mut f2 = fuel_per_op;
                let second = pair_snd(arena, current, &mut f2);
                remaining = remaining.saturating_sub(fuel_per_op - f2);
                current = second;
            }
            None => {
                // Cannot decode as boolean - not a number
                break;
            }
        }
    }

    // 0 = pair(false, nil) → bits is empty, that's fine, return 0
    if bits.is_empty() {
        // Verify this is actually a pair by checking fst
        let mut vf = fuel_per_op;
        let fst_check = pair_fst(arena, node, &mut vf);
        let fst_bool = decode_bool(arena, fst_check, fuel_per_op);
        if fst_bool == Some(false) {
            return Some(0); // pair(false, ...) with no more bits = 0
        }
        return None; // not a number
    }
    let mut n: u64 = 0;
    for (i, &b) in bits.iter().enumerate() {
        n += (b as u64) << i;
    }
    Some(n)
}

/// Try to decode the result as a stream of bytes (list of numbers).
/// Uses 2-arg Scott pair extraction.
fn output_byte_stream(arena: &mut Arena, node: u32, fuel: u64) {
    let mut current = node;
    let mut remaining_fuel = fuel;
    let stdout = io::stdout();
    let mut out = stdout.lock();
    let mut count = 0u64;

    loop {
        if remaining_fuel == 0 {
            eprintln!("\n[Fuel exhausted after {} bytes]", count);
            break;
        }

        // Check if nil (end of list): decode_bool on the pair itself
        // nil = KI, which is false when applied to two args
        let is_nil = decode_bool(arena, current, remaining_fuel / 20);
        if is_nil == Some(false) {
            eprintln!("\n[End of stream after {} bytes]", count);
            break;
        }
        // Note: is_nil == None means it's a pair (not a simple boolean) - continue

        // Extract head and tail using 2-arg pair extraction
        let mut f1 = remaining_fuel / 20;
        let head = pair_fst(arena, current, &mut f1);

        let mut f2 = remaining_fuel / 20;
        let tail = pair_snd(arena, current, &mut f2);

        remaining_fuel = remaining_fuel.saturating_sub(fuel / 20 * 2);

        if let Some(n) = decode_scott_num(arena, head, remaining_fuel / 20) {
            if n < 256 {
                let _ = out.write_all(&[n as u8]);
            } else {
                eprintln!("\n[Value {} at position {} exceeds byte range]", n, count);
            }
            count += 1;
            if count % 1000 == 0 {
                let _ = out.flush();
                eprint!("\r[{} bytes output, {} nodes]", count, arena.nodes.len());
            }
        } else {
            eprintln!("\n[Failed to decode number at position {}]", count);
            break;
        }

        current = tail;
    }
    let _ = out.flush();
}

/// Describe a WHNF node for debugging.
fn describe(arena: &Arena, idx: u32, depth: usize) -> String {
    if depth > 8 { return "...".to_string(); }
    let idx = arena.follow(idx);
    let n = &arena.nodes[idx as usize];
    match n.tag {
        S => "S".to_string(),
        K => "K".to_string(),
        I => "I".to_string(),
        APP => {
            let f = describe(arena, n.a, depth + 1);
            let a = describe(arena, n.b, depth + 1);
            format!("({} {})", f, a)
        }
        S1 => format!("(S {})", describe(arena, n.a, depth + 1)),
        S2 => format!("(S {} {})", describe(arena, n.a, depth + 1), describe(arena, n.b, depth + 1)),
        K1 => format!("(K {})", describe(arena, n.a, depth + 1)),
        _ => format!("?{}", n.tag),
    }
}

/// Decode a list of Scott-encoded numbers.
/// Uses 2-arg Scott pair extraction.
fn decode_number_list(arena: &mut Arena, node: u32, fuel: u64, max_items: usize) -> Vec<u64> {
    let mut result = Vec::new();
    let mut current = node;
    let mut remaining_fuel = fuel;

    for _ in 0..max_items {
        if remaining_fuel == 0 { break; }

        let is_nil = decode_bool(arena, current, remaining_fuel / 20);
        if is_nil == Some(false) { break; }
        // None means pair (non-nil) - continue

        let mut f1 = remaining_fuel / 20;
        let head = pair_fst(arena, current, &mut f1);

        let mut f2 = remaining_fuel / 20;
        let tail = pair_snd(arena, current, &mut f2);

        remaining_fuel = remaining_fuel.saturating_sub(fuel / 20 * 2);

        if let Some(n) = decode_scott_num(arena, head, remaining_fuel / 20) {
            result.push(n);
        } else {
            break;
        }

        current = tail;
    }
    result
}

/// Decode a list of booleans (for image pixel data).
/// Uses 2-arg Scott pair extraction.
fn decode_bool_list(arena: &mut Arena, node: u32, fuel: u64, max_items: usize) -> Vec<bool> {
    let mut result = Vec::new();
    let mut current = node;
    let mut remaining_fuel = fuel;

    for _ in 0..max_items {
        if remaining_fuel == 0 { break; }

        let is_nil = decode_bool(arena, current, remaining_fuel / 20);
        if is_nil == Some(false) { break; }
        // None means pair (non-nil) - continue

        let mut f1 = remaining_fuel / 20;
        let head = pair_fst(arena, current, &mut f1);

        let mut f2 = remaining_fuel / 20;
        let tail = pair_snd(arena, current, &mut f2);

        remaining_fuel = remaining_fuel.saturating_sub(fuel / 20 * 2);

        // Decode head as boolean
        match decode_bool(arena, head, remaining_fuel / 20) {
            Some(b) => result.push(b),
            None => break,
        }

        current = tail;
    }
    result
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: ski-eval <compact-file> [--fuel N] [--decode list|stream|bool|num|boollist|describe]");
        process::exit(1);
    }

    let filename = &args[1];
    let mut fuel: u64 = 100_000_000;
    let mut decode_mode = "describe".to_string();
    let mut render_var: u64 = 4;
    let mut grid_size: u64 = 0; // 0 = use render_var as grid size
    let mut img_path = "d:/github/atgt2026hp_stars/images/rendered".to_string();

    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--fuel" => {
                i += 1;
                fuel = args[i].parse().expect("invalid fuel value");
            }
            "--decode" => {
                i += 1;
                decode_mode = args[i].clone();
            }
            "--var" => {
                i += 1;
                render_var = args[i].parse().expect("invalid var value");
            }
            "--img" => {
                i += 1;
                img_path = args[i].clone();
            }
            "--grid" => {
                i += 1;
                grid_size = args[i].parse().expect("invalid grid value");
            }
            _ => {
                eprintln!("Unknown option: {}", args[i]);
                process::exit(1);
            }
        }
        i += 1;
    }

    eprintln!("Reading {}...", filename);
    let input = fs::read(filename).expect("failed to read file");
    let input_len = input.len();
    eprintln!("  {} bytes", input_len);

    // Estimate node count
    let estimated_nodes = input_len * 2;
    let mut arena = Arena::new(estimated_nodes);

    eprintln!("Parsing...");
    let root = parse_compact(&mut arena, &input);
    eprintln!("  {} nodes", arena.nodes.len());

    eprintln!("Evaluating (fuel={})...", fuel);
    let mut remaining_fuel = fuel;
    let result = arena.whnf(root, &mut remaining_fuel);
    let steps = fuel - remaining_fuel;
    eprintln!("  {} reduction steps", steps);
    eprintln!("  {} total nodes (after reduction)", arena.nodes.len());

    let result = arena.follow(result);

    match decode_mode.as_str() {
        "describe" => {
            let desc = describe(&arena, result, 0);
            if desc.len() > 5000 {
                println!("{}", &desc[..5000]);
                println!("... (truncated)");
            } else {
                println!("{}", desc);
            }
        }
        "bool" => {
            match decode_bool(&mut arena, result, remaining_fuel) {
                Some(true) => println!("TRUE"),
                Some(false) => println!("FALSE"),
                None => println!("NOT A BOOLEAN"),
            }
        }
        "num" => {
            match decode_scott_num(&mut arena, result, remaining_fuel) {
                Some(n) => println!("{}", n),
                None => println!("NOT A NUMBER"),
            }
        }
        "list" => {
            let nums = decode_number_list(&mut arena, result, remaining_fuel, 100000);
            for n in &nums {
                print!("{} ", n);
            }
            println!();
            eprintln!("  {} items decoded", nums.len());
        }
        "boollist" => {
            let bools = decode_bool_list(&mut arena, result, remaining_fuel, 100000);
            for b in &bools {
                print!("{}", if *b { "1" } else { "0" });
            }
            println!();
            eprintln!("  {} items decoded", bools.len());
        }
        "stream" => {
            output_byte_stream(&mut arena, result, remaining_fuel);
        }
        "fst" => {
            // Extract first element of pair using 2-arg extraction
            let mut f = remaining_fuel;
            let fst = pair_fst(&mut arena, result, &mut f);
            let steps2 = remaining_fuel - f;
            eprintln!("  fst: {} additional steps", steps2);

            // Try various decodings
            let desc = describe(&arena, fst, 0);
            if desc.len() > 2000 {
                eprintln!("  describe: {}...", &desc[..2000]);
            } else {
                eprintln!("  describe: {}", desc);
            }
            match decode_bool(&mut arena, fst, f) {
                Some(true) => println!("fst = TRUE"),
                Some(false) => println!("fst = FALSE"),
                None => {
                    match decode_scott_num(&mut arena, fst, f) {
                        Some(n) => println!("fst = NUMBER({})", n),
                        None => println!("fst = {}", if desc.len() > 200 { &desc[..200] } else { &desc }),
                    }
                }
            }
        }
        "snd" => {
            // Extract second element using 2-arg extraction
            let mut f = remaining_fuel;
            let snd = pair_snd(&mut arena, result, &mut f);
            let steps2 = remaining_fuel - f;
            eprintln!("  snd: {} additional steps", steps2);

            let desc = describe(&arena, snd, 0);
            if desc.len() > 2000 {
                eprintln!("  describe: {}...", &desc[..2000]);
            } else {
                eprintln!("  describe: {}", desc);
            }
            match decode_bool(&mut arena, snd, f) {
                Some(true) => println!("snd = TRUE"),
                Some(false) => println!("snd = FALSE"),
                None => {
                    match decode_scott_num(&mut arena, snd, f) {
                        Some(n) => println!("snd = NUMBER({})", n),
                        None => println!("snd = {}", if desc.len() > 200 { &desc[..200] } else { &desc }),
                    }
                }
            }
        }
        "deep" => {
            // Recursively unpack pairs and try to decode structure
            eprintln!("Deep decoding...");
            deep_decode(&mut arena, result, remaining_fuel, 0, 20);
        }
        "qtree" => {
            // Interpret output and render as quadtree image
            eprintln!("Quadtree image rendering...");

            // The output is PAIR(header, image_data)
            // header = PAIR(1,1) might be format code 3 (image)
            // Extract image_data = snd(result)
            let image_data = pair_snd(&mut arena, result, &mut remaining_fuel);
            eprintln!("  Extracted image data");

            // === Method 1: Diamond encoding PAIR(cond, PAIR(qa, PAIR(qb, PAIR(qc, qd)))) ===
            for depth in &[8, 10] {
                let size = 1usize << depth;
                let mut pixels = vec![255u8; size * size]; // default white
                let mut pixel_count = 0u64;
                eprintln!("  Diamond {}x{} (depth {})...", size, size, depth);
                render_diamond(&mut arena, image_data, &mut pixels, 0, 0, size, size, &mut remaining_fuel, &mut pixel_count);
                eprintln!("    {} pixels rendered, {} nodes", pixel_count, arena.nodes.len());
                let fname = format!("d:/github/atgt2026hp_stars/images/diamond_{}x{}.pgm", size, size);
                write_pgm(&fname, size, size, &pixels);
                eprintln!("    Saved {}", fname);
            }

            // === Method 2: PAIR(PAIR(nw,ne), PAIR(sw,se)) with snd(result) ===
            for depth in &[8, 10] {
                let size = 1usize << depth;
                let mut pixels = vec![255u8; size * size];
                let mut pixel_count = 0u64;
                eprintln!("  Quadtree v2 {}x{} on snd(result)...", size, size);
                render_quadtree_v2(&mut arena, image_data, &mut pixels, 0, 0, size, size, &mut remaining_fuel, &mut pixel_count);
                eprintln!("    {} pixels rendered, {} nodes", pixel_count, arena.nodes.len());
                let fname = format!("d:/github/atgt2026hp_stars/images/qtree2_snd_{}x{}.pgm", size, size);
                write_pgm(&fname, size, size, &pixels);
                eprintln!("    Saved {}", fname);
            }

            // === Method 3: PAIR(PAIR(nw,ne), PAIR(sw,se)) with full result ===
            for depth in &[8, 10] {
                let size = 1usize << depth;
                let mut pixels = vec![255u8; size * size];
                let mut pixel_count = 0u64;
                eprintln!("  Quadtree v2 {}x{} on full result...", size, size);
                render_quadtree_v2(&mut arena, result, &mut pixels, 0, 0, size, size, &mut remaining_fuel, &mut pixel_count);
                eprintln!("    {} pixels rendered, {} nodes", pixel_count, arena.nodes.len());
                let fname = format!("d:/github/atgt2026hp_stars/images/qtree2_full_{}x{}.pgm", size, size);
                write_pgm(&fname, size, size, &pixels);
                eprintln!("    Saved {}", fname);
            }
        }
        "leaves" => {
            // Walk the output as a binary tree, collecting boolean leaves
            eprintln!("Collecting boolean leaves from output tree...");
            let snd_r = pair_snd(&mut arena, result, &mut remaining_fuel);
            let mut leaves: Vec<u8> = Vec::new();
            collect_bool_leaves(&mut arena, snd_r, &mut remaining_fuel, &mut leaves, 500000);
            eprintln!("  Collected {} boolean leaves", leaves.len());
            if leaves.len() > 100 {
                let sample: Vec<u8> = leaves[..100].to_vec();
                eprintln!("  First 100: {:?}", sample);
            }

            // Also from full result
            let mut leaves2: Vec<u8> = Vec::new();
            collect_bool_leaves(&mut arena, result, &mut remaining_fuel, &mut leaves2, 500000);
            eprintln!("  From full result: {} boolean leaves", leaves2.len());
            if leaves2.len() > 100 {
                let sample: Vec<u8> = leaves2[..100].to_vec();
                eprintln!("  First 100: {:?}", sample);
            }

            // Try rendering as image with width 4096
            for (name, lvs) in &[("snd", &leaves), ("full", &leaves2)] {
                let n = lvs.len();
                if n < 100 { continue; }
                for width in &[4096usize, 2048, 1024, 512, 256, 128] {
                    if n < *width { continue; }
                    let height = n / width;
                    if height < 10 { continue; }
                    let mut pixels: Vec<u8> = lvs[..width * height].iter().map(|&b| if b == 1 { 0u8 } else { 255u8 }).collect();
                    let fname = format!("d:/github/atgt2026hp_stars/images/leaves_{}_{}x{}.pgm", name, width, height);
                    write_pgm(&fname, *width, height, &pixels);
                    eprintln!("  Saved {}", fname);
                }
            }
        }
        "trace" => {
            // Trace the structure level by level
            eprintln!("Tracing output structure...");
            let mut f = remaining_fuel;

            eprintln!("\n=== Level 0: result ===");
            let desc0 = describe(&arena, result, 0);
            eprintln!("  {}", if desc0.len() > 300 { &desc0[..300] } else { &desc0 });

            let a = pair_fst(&mut arena, result, &mut f);
            let b = pair_snd(&mut arena, result, &mut f);

            eprintln!("\n=== Level 1a: fst(result) ===");
            let da = describe(&arena, a, 0);
            eprintln!("  {}", if da.len() > 300 { &da[..300] } else { &da });
            if let Some(bn) = decode_scott_num(&mut arena, a, f.min(1000000)) {
                eprintln!("  -> NUMBER({})", bn);
            }

            eprintln!("\n=== Level 1b: snd(result) ===");
            let db = describe(&arena, b, 0);
            eprintln!("  {}", if db.len() > 300 { &db[..300] } else { &db });
            if let Some(bb) = decode_bool(&mut arena, b, f.min(1000000)) {
                eprintln!("  -> BOOL: {}", bb);
            }

            let b_fst = pair_fst(&mut arena, b, &mut f);
            let b_snd = pair_snd(&mut arena, b, &mut f);

            eprintln!("\n=== Level 2a: fst(snd(result)) ===");
            let d2a = describe(&arena, b_fst, 0);
            eprintln!("  {}", if d2a.len() > 300 { &d2a[..300] } else { &d2a });
            if let Some(bn) = decode_scott_num(&mut arena, b_fst, f.min(1000000)) {
                eprintln!("  -> NUMBER({})", bn);
            }
            if let Some(bb) = decode_bool(&mut arena, b_fst, f.min(1000000)) {
                eprintln!("  -> BOOL: {}", bb);
            }

            eprintln!("\n=== Level 2b: snd(snd(result)) ===");
            let d2b = describe(&arena, b_snd, 0);
            eprintln!("  {}", if d2b.len() > 300 { &d2b[..300] } else { &d2b });
            if let Some(bb) = decode_bool(&mut arena, b_snd, f.min(1000000)) {
                eprintln!("  -> BOOL: {}", bb);
            }

            // Go deeper
            let b_fst_fst = pair_fst(&mut arena, b_fst, &mut f);
            let b_fst_snd = pair_snd(&mut arena, b_fst, &mut f);

            eprintln!("\n=== Level 3a: fst(fst(snd(result))) ===");
            let d3a = describe(&arena, b_fst_fst, 0);
            eprintln!("  {}", if d3a.len() > 300 { &d3a[..300] } else { &d3a });
            if let Some(bn) = decode_scott_num(&mut arena, b_fst_fst, f.min(1000000)) {
                eprintln!("  -> NUMBER({})", bn);
            }
            if let Some(bb) = decode_bool(&mut arena, b_fst_fst, f.min(1000000)) {
                eprintln!("  -> BOOL: {}", bb);
            }

            eprintln!("\n=== Level 3b: snd(fst(snd(result))) ===");
            let d3b = describe(&arena, b_fst_snd, 0);
            eprintln!("  {}", if d3b.len() > 300 { &d3b[..300] } else { &d3b });
            if let Some(bn) = decode_scott_num(&mut arena, b_fst_snd, f.min(1000000)) {
                eprintln!("  -> NUMBER({})", bn);
            }
            if let Some(bb) = decode_bool(&mut arena, b_fst_snd, f.min(1000000)) {
                eprintln!("  -> BOOL: {}", bb);
            }

            // Level 4: go into b_fst_snd (which should be the next level of nesting)
            let l4_fst = pair_fst(&mut arena, b_fst_snd, &mut f);
            let l4_snd = pair_snd(&mut arena, b_fst_snd, &mut f);
            eprintln!("\n=== Level 4a: fst(snd(fst(snd(r)))) ===");
            if let Some(bn) = decode_scott_num(&mut arena, l4_fst, f.min(1000000)) {
                eprintln!("  -> NUMBER({})", bn);
            }
            if let Some(bb) = decode_bool(&mut arena, l4_fst, f.min(1000000)) {
                eprintln!("  -> BOOL: {}", bb);
            }
            let d4a = describe(&arena, l4_fst, 0);
            eprintln!("  {}", if d4a.len() > 300 { &d4a[..300] } else { &d4a });

            eprintln!("\n=== Level 4b: snd(snd(fst(snd(r)))) ===");
            if let Some(bn) = decode_scott_num(&mut arena, l4_snd, f.min(1000000)) {
                eprintln!("  -> NUMBER({})", bn);
            }
            if let Some(bb) = decode_bool(&mut arena, l4_snd, f.min(1000000)) {
                eprintln!("  -> BOOL: {}", bb);
            }
            let d4b = describe(&arena, l4_snd, 0);
            eprintln!("  {}", if d4b.len() > 300 { &d4b[..300] } else { &d4b });
        }
        "apply" => {
            // Try applying result to various argument combinations to find pixel function
            eprintln!("Probing result with various argument patterns...");
            let mut f = remaining_fuel;

            // Pattern 1: result(row)(col) - 2 args
            eprintln!("\n--- Pattern: result(m)(z) ---");
            for (m, z) in &[(0u64,0u64), (0,1), (1,0), (1,1), (2,3)] {
                let mn = make_scott_num(&mut arena, *m);
                let zn = make_scott_num(&mut arena, *z);
                let app1 = arena.alloc(APP, result, mn);
                let app2 = arena.alloc(APP, app1, zn);
                let mut fuel = f.min(5000000);
                arena.whnf(app2, &mut fuel);
                let r = arena.follow(app2);
                let b = decode_bool(&mut arena, r, 1000000);
                let desc = describe(&arena, r, 0);
                let d = if desc.len() > 100 { &desc[..100] } else { &desc };
                eprintln!("  result({},{}) = {} bool={:?}", m, z, d, b);
            }

            // Pattern 2: result(N)(m)(z) - 3 args
            eprintln!("\n--- Pattern: result(N)(m)(z) ---");
            for (n, m, z) in &[(8u64,0u64,0u64), (8,0,1), (8,1,0), (8,1,1), (8,3,5)] {
                let nn = make_scott_num(&mut arena, *n);
                let mn = make_scott_num(&mut arena, *m);
                let zn = make_scott_num(&mut arena, *z);
                let app1 = arena.alloc(APP, result, nn);
                let app2 = arena.alloc(APP, app1, mn);
                let app3 = arena.alloc(APP, app2, zn);
                let mut fuel = f.min(10000000);
                arena.whnf(app3, &mut fuel);
                let r = arena.follow(app3);
                let b = decode_bool(&mut arena, r, 1000000);
                let desc = describe(&arena, r, 0);
                let d = if desc.len() > 100 { &desc[..100] } else { &desc };
                eprintln!("  result({},{},{}) = {} bool={:?}", n, m, z, d, b);
            }

            // Pattern 3: result(var)(m)(z) with var=1 (initial call)
            eprintln!("\n--- Pattern: result(1)(m)(z) ---");
            for (m, z) in &[(0u64,0u64), (0,1), (1,0), (1,1)] {
                let v1 = make_scott_num(&mut arena, 1);
                let mn = make_scott_num(&mut arena, *m);
                let zn = make_scott_num(&mut arena, *z);
                let app1 = arena.alloc(APP, result, v1);
                let app2 = arena.alloc(APP, app1, mn);
                let app3 = arena.alloc(APP, app2, zn);
                let mut fuel = f.min(10000000);
                arena.whnf(app3, &mut fuel);
                let r = arena.follow(app3);
                let b = decode_bool(&mut arena, r, 1000000);
                let desc = describe(&arena, r, 0);
                let d = if desc.len() > 100 { &desc[..100] } else { &desc };
                eprintln!("  result(1,{},{}) = {} bool={:?}", m, z, d, b);
            }

            // Pattern 4: snd(result)(args)
            let snd_r = pair_snd(&mut arena, result, &mut f);
            eprintln!("\n--- Pattern: snd(result)(m)(z) ---");
            for (m, z) in &[(0u64,0u64), (0,1), (1,0), (1,1)] {
                let mn = make_scott_num(&mut arena, *m);
                let zn = make_scott_num(&mut arena, *z);
                let app1 = arena.alloc(APP, snd_r, mn);
                let app2 = arena.alloc(APP, app1, zn);
                let mut fuel = f.min(10000000);
                arena.whnf(app2, &mut fuel);
                let r = arena.follow(app2);
                let b = decode_bool(&mut arena, r, 1000000);
                let desc = describe(&arena, r, 0);
                let d = if desc.len() > 100 { &desc[..100] } else { &desc };
                eprintln!("  snd(r)({},{}) = {} bool={:?}", m, z, d, b);
            }

            // Pattern 5: fst(snd(result))(args) - maybe the actual function is deeper
            let fst_snd = pair_fst(&mut arena, snd_r, &mut f);
            eprintln!("\n--- Pattern: fst(snd(result))(m)(z) ---");
            for (m, z) in &[(0u64,0u64), (0,1), (1,0), (1,1)] {
                let mn = make_scott_num(&mut arena, *m);
                let zn = make_scott_num(&mut arena, *z);
                let app1 = arena.alloc(APP, fst_snd, mn);
                let app2 = arena.alloc(APP, app1, zn);
                let mut fuel = f.min(10000000);
                arena.whnf(app2, &mut fuel);
                let r = arena.follow(app2);
                let b = decode_bool(&mut arena, r, 1000000);
                let desc = describe(&arena, r, 0);
                let d = if desc.len() > 100 { &desc[..100] } else { &desc };
                eprintln!("  fst(snd(r))({},{}) = {} bool={:?}", m, z, d, b);
            }
        }
        "render" => {
            // Render image: result(N)(m)(z) -> pixel value (binary 0/1)
            // N = resolution (image is NxN pixels), m = row, z = column
            let size = render_var; // N = image size directly (not 2^var)
            eprintln!("Rendering {}x{} image (N={})...", size, size, size);

            let mut pixels = vec![128u8; (size * size) as usize]; // default gray
            let mut decoded_count = 0u64;
            let mut bool_count = 0u64;
            let mut num_count = 0u64;
            let mut fail_count = 0u64;
            let fuel_per_pixel: u64 = 10_000_000;
            let mut fail_examples: Vec<(u64, u64, String)> = Vec::new();

            for m in 0..size {
                for z in 0..size {
                    let var_n = make_scott_num(&mut arena, size);
                    let m_n = make_scott_num(&mut arena, m);
                    let z_n = make_scott_num(&mut arena, z);
                    let app1 = arena.alloc(APP, result, var_n);
                    let app2 = arena.alloc(APP, app1, m_n);
                    let app3 = arena.alloc(APP, app2, z_n);
                    let mut pf = fuel_per_pixel;
                    arena.whnf(app3, &mut pf);
                    let r = arena.follow(app3);

                    // Try decode as Scott number, unwrapping K/S(KK) wrappers if needed
                    let mut val = r;
                    // Unwrap K(x) -> x: K1 node or APP(K, x) pattern
                    // Also unwrap S(KK)(g) -> g: S2 node where a = KK
                    for _ in 0..10 {
                        let vv = arena.follow(val);
                        let n = arena.nodes[vv as usize];
                        if n.tag == K1 {
                            val = arena.follow_mut(n.a);
                            continue;
                        }
                        if n.tag == APP {
                            let func = arena.follow(n.a);
                            if arena.nodes[func as usize].tag == K {
                                val = arena.follow_mut(n.b);
                                continue;
                            }
                        }
                        // S2(f, g) where f = KK → S(KK)(g) = K∘g → extract g
                        if n.tag == S2 {
                            let f = arena.follow(n.a);
                            let fn_node = arena.nodes[f as usize];
                            // Check if f = K1(K) i.e. KK
                            if fn_node.tag == K1 {
                                let inner = arena.follow(fn_node.a);
                                if arena.nodes[inner as usize].tag == K {
                                    val = arena.follow_mut(n.b);
                                    continue;
                                }
                            }
                            // Also check if f = APP(K, K)
                            if fn_node.tag == APP {
                                let fa = arena.follow(fn_node.a);
                                let fb = arena.follow(fn_node.b);
                                if arena.nodes[fa as usize].tag == K && arena.nodes[fb as usize].tag == K {
                                    val = arena.follow_mut(n.b);
                                    continue;
                                }
                            }
                        }
                        break;
                    }

                    if let Some(n) = decode_scott_num(&mut arena, val, 1_000_000) {
                        pixels[(m * size + z) as usize] = (n.min(255)) as u8;
                        num_count += 1;
                        decoded_count += 1;
                    } else if let Some(b) = decode_bool(&mut arena, val, 500_000) {
                        pixels[(m * size + z) as usize] = if b { 255 } else { 0 };
                        bool_count += 1;
                        decoded_count += 1;
                    } else {
                        fail_count += 1;
                        if fail_examples.len() < 5 {
                            let desc = describe(&arena, val, 0);
                            let d = if desc.len() > 200 { desc[..200].to_string() } else { desc };
                            fail_examples.push((m, z, d));
                        }
                    }
                }
                if (m + 1) % 4 == 0 || m == size - 1 {
                    eprint!("\r  row {}/{} ({} num, {} bool, {} fail, {} nodes)     ",
                        m + 1, size, num_count, bool_count, fail_count, arena.nodes.len());
                }
            }
            eprintln!();
            eprintln!("  Decoded: {} num, {} bool, {} fail out of {}", num_count, bool_count, fail_count, size * size);

            // Find max value for normalization
            let max_val = pixels.iter().copied().filter(|&p| p != 128).max().unwrap_or(1);
            eprintln!("  Max pixel value: {}", max_val);

            // Save raw image
            let fname = format!("{}_{}x{}.pgm", img_path, size, size);
            write_pgm(&fname, size as usize, size as usize, &pixels);
            eprintln!("  Saved {}", fname);

            // Also save normalized version if max > 1
            if max_val > 1 && max_val < 255 {
                let normalized: Vec<u8> = pixels.iter().map(|&p| {
                    if p == 128 { 128 } // keep gray for unknown
                    else { ((p as u32) * 255 / max_val as u32).min(255) as u8 }
                }).collect();
                let fname2 = format!("{}_norm_{}x{}.pgm", img_path, size, size);
                write_pgm(&fname2, size as usize, size as usize, &normalized);
                eprintln!("  Saved {}", fname2);
            }

            // Print fail examples
            if !fail_examples.is_empty() {
                eprintln!("  Failed pixel examples:");
                for (m, z, desc) in &fail_examples {
                    eprintln!("    ({},{}) = {}", m, z, desc);
                }
            }

            // Print sample pixel values (numeric)
            let sample = 16.min(size);
            eprintln!("  Sample pixel values (first {}x{}):", sample, sample);
            for m in 0..sample {
                eprint!("    ");
                for z in 0..sample {
                    let p = pixels[(m * size + z) as usize];
                    eprint!("{:3} ", p);
                }
                eprintln!();
            }
            // Also print as visual
            eprintln!("  Visual (0=. else #):");
            for m in 0..sample {
                eprint!("    ");
                for z in 0..sample {
                    let p = pixels[(m * size + z) as usize];
                    if p == 128 { eprint!("? "); }
                    else if p == 0 { eprint!(". "); }
                    else { eprint!("# "); }
                }
                eprintln!();
            }
        }
        "render2" => {
            // Render with N and render_size decoupled.
            // N (the format wrapper arg) = render_var
            // render_size = the actual pixel grid dimension
            // This lets us use N=4 or N=16 (which work) but render at higher resolution.
            let n_arg = render_var;
            let render_size = if grid_size > 0 { grid_size } else { n_arg };

            eprintln!("Rendering {}x{} image using N={} as format arg...", render_size, render_size, n_arg);

            let mut pixels = vec![128u8; (render_size * render_size) as usize];
            let mut num_count = 0u64;
            let mut bool_count = 0u64;
            let mut fail_count = 0u64;
            let fuel_per_pixel: u64 = 10_000_000;
            let mut fail_examples: Vec<(u64, u64, String)> = Vec::new();

            for m in 0..render_size {
                for z in 0..render_size {
                    let var_n = make_scott_num(&mut arena, n_arg); // N = fixed
                    let m_n = make_scott_num(&mut arena, m);
                    let z_n = make_scott_num(&mut arena, z);
                    let app1 = arena.alloc(APP, result, var_n);
                    let app2 = arena.alloc(APP, app1, m_n);
                    let app3 = arena.alloc(APP, app2, z_n);
                    let mut pf = fuel_per_pixel;
                    arena.whnf(app3, &mut pf);
                    let r = arena.follow(app3);

                    // Try decode as Scott number
                    let mut val = r;
                    for _ in 0..10 {
                        let vv = arena.follow(val);
                        let n = arena.nodes[vv as usize];
                        if n.tag == K1 {
                            val = arena.follow_mut(n.a);
                            continue;
                        }
                        if n.tag == APP {
                            let func = arena.follow(n.a);
                            if arena.nodes[func as usize].tag == K {
                                val = arena.follow_mut(n.b);
                                continue;
                            }
                        }
                        if n.tag == S2 {
                            let f = arena.follow(n.a);
                            let fn_node = arena.nodes[f as usize];
                            if fn_node.tag == K1 {
                                let inner = arena.follow(fn_node.a);
                                if arena.nodes[inner as usize].tag == K {
                                    val = arena.follow_mut(n.b);
                                    continue;
                                }
                            }
                            if fn_node.tag == APP {
                                let fa = arena.follow(fn_node.a);
                                let fb = arena.follow(fn_node.b);
                                if arena.nodes[fa as usize].tag == K && arena.nodes[fb as usize].tag == K {
                                    val = arena.follow_mut(n.b);
                                    continue;
                                }
                            }
                        }
                        break;
                    }

                    if let Some(n) = decode_scott_num(&mut arena, val, 1_000_000) {
                        pixels[(m * render_size + z) as usize] = (n.min(255)) as u8;
                        num_count += 1;
                    } else if let Some(b) = decode_bool(&mut arena, val, 500_000) {
                        pixels[(m * render_size + z) as usize] = if b { 255 } else { 0 };
                        bool_count += 1;
                    } else {
                        fail_count += 1;
                        if fail_examples.len() < 5 {
                            let desc = describe(&arena, val, 0);
                            let d = if desc.len() > 200 { desc[..200].to_string() } else { desc };
                            fail_examples.push((m, z, d));
                        }
                    }
                }
                if (m + 1) % 4 == 0 || m == render_size - 1 {
                    eprint!("\r  row {}/{} ({} num, {} bool, {} fail, {} nodes)     ",
                        m + 1, render_size, num_count, bool_count, fail_count, arena.nodes.len());
                }
            }
            eprintln!();
            eprintln!("  Decoded: {} num, {} bool, {} fail out of {}", num_count, bool_count, fail_count, render_size * render_size);

            let max_val = pixels.iter().copied().filter(|&p| p != 128).max().unwrap_or(1);
            eprintln!("  Max pixel value: {}", max_val);

            let fname = format!("{}_N{}_{}x{}.pgm", img_path, n_arg, render_size, render_size);
            write_pgm(&fname, render_size as usize, render_size as usize, &pixels);
            eprintln!("  Saved {}", fname);

            if max_val > 1 && max_val < 255 {
                let normalized: Vec<u8> = pixels.iter().map(|&p| {
                    if p == 128 { 128 }
                    else { ((p as u32) * 255 / max_val as u32).min(255) as u8 }
                }).collect();
                let fname2 = format!("{}_N{}_norm_{}x{}.pgm", img_path, n_arg, render_size, render_size);
                write_pgm(&fname2, render_size as usize, render_size as usize, &normalized);
                eprintln!("  Saved {}", fname2);
            }

            if !fail_examples.is_empty() {
                eprintln!("  Failed pixel examples:");
                for (m, z, desc) in &fail_examples {
                    eprintln!("    ({},{}) = {}", m, z, desc);
                }
            }

            // Print sample pixel values
            let sample = 16.min(render_size);
            eprintln!("  Sample pixel values (first {}x{}):", sample, sample);
            for m in 0..sample {
                eprint!("    ");
                for z in 0..sample {
                    let p = pixels[(m * render_size + z) as usize];
                    eprint!("{:3} ", p);
                }
                eprintln!();
            }
        }
        "structure" => {
            // Deep examination of result = S(SI(KA))(Y) structure
            eprintln!("Examining result structure...");
            let r = arena.follow(result);
            let rn = arena.nodes[r as usize];
            eprintln!("result: tag={} ({})", rn.tag, match rn.tag { 0=>"APP",1=>"S",2=>"K",3=>"I",4=>"S1",5=>"S2",6=>"K1",7=>"IND",_=>"?" });

            if rn.tag == S2 {
                // result = S2(f, Y)
                let f = arena.follow(rn.a);
                let y = arena.follow(rn.b);
                let fn_node = arena.nodes[f as usize];
                let yn = arena.nodes[y as usize];
                eprintln!("  f (result.a): tag={}", fn_node.tag);
                eprintln!("  Y (result.b): tag={}", yn.tag);

                // f should be S2(I, K(A))
                if fn_node.tag == S2 {
                    let f_a = arena.follow(fn_node.a); // should be I
                    let f_b = arena.follow(fn_node.b); // should be K1(A)
                    eprintln!("  f.a: tag={}", arena.nodes[f_a as usize].tag);
                    eprintln!("  f.b: tag={}", arena.nodes[f_b as usize].tag);

                    // f.b = K1(A)
                    let fb_node = arena.nodes[f_b as usize];
                    if fb_node.tag == K1 {
                        let a_node_idx = arena.follow(fb_node.a); // A
                        let a_node = arena.nodes[a_node_idx as usize];
                        eprintln!("  A: tag={}", a_node.tag);
                        let a_desc = describe(&arena, a_node_idx, 3);
                        eprintln!("  A describe(depth=3): {}", a_desc);

                        // If A = S2(SI(K(p1)), K(p2)) = pair(p1, p2)
                        if a_node.tag == S2 {
                            let a_a = arena.follow(a_node.a); // SI(K(p1))
                            let a_b = arena.follow(a_node.b); // K(p2)
                            eprintln!("  A.a = {}", describe(&arena, a_a, 3));
                            eprintln!("  A.b = {}", describe(&arena, a_b, 3));

                            // Extract p2 from K(p2) = K1(p2)
                            let ab_node = arena.nodes[a_b as usize];
                            if ab_node.tag == K1 {
                                let p2 = arena.follow(ab_node.a);
                                eprintln!("  p2 = {}", describe(&arena, p2, 5));
                                if let Some(n) = decode_scott_num(&mut arena, p2, 1_000_000) {
                                    eprintln!("  p2 = NUMBER({})", n);
                                }
                            }

                            // Extract p1 from SI(K(p1)) = S2(I, K(p1))
                            let aa_node = arena.nodes[a_a as usize];
                            if aa_node.tag == S2 {
                                let aa_b = arena.follow(aa_node.b); // K(p1) = K1(p1)
                                let aab_node = arena.nodes[aa_b as usize];
                                if aab_node.tag == K1 {
                                    let p1 = arena.follow(aab_node.a);
                                    eprintln!("  p1 = {}", describe(&arena, p1, 5));
                                    if let Some(n) = decode_scott_num(&mut arena, p1, 1_000_000) {
                                        eprintln!("  p1 = NUMBER({})", n);
                                    }
                                }
                            }
                        }
                    }
                }

                // Describe Y briefly
                let y_desc = describe(&arena, y, 2);
                eprintln!("  Y describe(depth=2): {}", y_desc);

                // Try Y as a pair
                let mut f = remaining_fuel;
                let y_fst = pair_fst(&mut arena, y, &mut f);
                let yf_desc = describe(&arena, y_fst, 2);
                eprintln!("  fst(Y) = {}", yf_desc);

                let y_snd = pair_snd(&mut arena, y, &mut f);
                let ys_desc = describe(&arena, y_snd, 2);
                eprintln!("  snd(Y) = {}", ys_desc);
            } else if rn.tag == APP {
                let la = arena.follow(rn.a);
                let lb = arena.follow(rn.b);
                eprintln!("  APP: left tag={} right tag={}", arena.nodes[la as usize].tag, arena.nodes[lb as usize].tag);

                // Navigate deeper if left is also APP
                let la_node = arena.nodes[la as usize];
                if la_node.tag == APP {
                    let lla = arena.follow(la_node.a);
                    let llb = arena.follow(la_node.b);
                    eprintln!("  Left is APP(tag={}, tag={})", arena.nodes[lla as usize].tag, arena.nodes[llb as usize].tag);
                    if arena.nodes[lla as usize].tag == S {
                        eprintln!("  → result = S(f)(Y) where f and Y are:");
                        eprintln!("  f = {}", describe(&arena, llb, 3));
                        eprintln!("  Y = {}", describe(&arena, lb, 2));
                    }
                }
            }
        }
        "examine" => {
            // Examine result(N) structure
            let n = make_scott_num(&mut arena, render_var);
            let app = arena.alloc(APP, result, n);
            let mut f = remaining_fuel;
            arena.whnf(app, &mut f);
            let rn = arena.follow(app);
            let desc = describe(&arena, rn, 0);
            eprintln!("result({}) WHNF:", render_var);
            eprintln!("  {}", if desc.len() > 3000 { &desc[..3000] } else { &desc });

            // Try as number list
            let nums = decode_number_list(&mut arena, rn, f.min(10_000_000), 200);
            if !nums.is_empty() {
                eprintln!("As number list ({} items): {:?}", nums.len(), &nums[..nums.len().min(50)]);
            }

            // Try as bool list
            let bools = decode_bool_list(&mut arena, rn, f.min(10_000_000), 200);
            if !bools.is_empty() {
                eprintln!("As bool list ({} items): {:?}", bools.len(), &bools[..bools.len().min(50)]);
            }

            // Try fst
            let fst_val = pair_fst(&mut arena, rn, &mut f);
            let fst_desc = describe(&arena, fst_val, 0);
            eprintln!("fst(result({})): {}", render_var, if fst_desc.len() > 1000 { &fst_desc[..1000] } else { &fst_desc });
            if let Some(n) = decode_scott_num(&mut arena, fst_val, f.min(5_000_000)) {
                eprintln!("  = NUMBER({})", n);
            }
            if let Some(b) = decode_bool(&mut arena, fst_val, f.min(5_000_000)) {
                eprintln!("  = BOOL({})", b);
            }

            // Try snd
            let snd_val = pair_snd(&mut arena, rn, &mut f);
            let snd_desc = describe(&arena, snd_val, 0);
            eprintln!("snd(result({})): {}", render_var, if snd_desc.len() > 1000 { &snd_desc[..1000] } else { &snd_desc });
            if let Some(n) = decode_scott_num(&mut arena, snd_val, f.min(5_000_000)) {
                eprintln!("  = NUMBER({})", n);
            }
            if let Some(b) = decode_bool(&mut arena, snd_val, f.min(5_000_000)) {
                eprintln!("  = BOOL({})", b);
            }

            // Try result(N)(0)
            let zero = make_scott_num(&mut arena, 0);
            let app_zero = arena.alloc(APP, rn, zero);
            let mut f2 = f.min(50_000_000);
            arena.whnf(app_zero, &mut f2);
            let r0 = arena.follow(app_zero);
            let r0_desc = describe(&arena, r0, 0);
            eprintln!("result({})(0): {}", render_var, if r0_desc.len() > 1000 { &r0_desc[..1000] } else { &r0_desc });

            // Try result(N)(0)(0)
            let zero2 = make_scott_num(&mut arena, 0);
            let app_00 = arena.alloc(APP, r0, zero2);
            let mut f3 = f.min(50_000_000);
            arena.whnf(app_00, &mut f3);
            let r00 = arena.follow(app_00);
            let r00_desc = describe(&arena, r00, 0);
            eprintln!("result({})(0)(0): {}", render_var, if r00_desc.len() > 500 { &r00_desc[..500] } else { &r00_desc });
            if let Some(n) = decode_scott_num(&mut arena, r00, f.min(5_000_000)) {
                eprintln!("  = NUMBER({})", n);
            }

            // Try result(N) as stream (output bytes)
            eprintln!("\nTrying result({}) as byte stream...", render_var);
            let rn2 = arena.follow(app); // re-follow
            output_byte_stream(&mut arena, rn2, f.min(50_000_000));
        }
        "nsweep" => {
            // Quick sweep of N values to find which produce meaningful images
            // Tests a few pixels per N to distinguish meaningful vs gradient
            eprintln!("Sweeping N values to find meaningful images...");
            let n_start = if render_var > 2 { render_var } else { 2 };
            let n_end = if grid_size > 0 { grid_size } else { 512 };

            let test_coords: Vec<(u64, u64)> = vec![
                (0, 0), (0, 1), (1, 0), (1, 1),
                (0, 2), (2, 0), (2, 1), (1, 2),
            ];

            for n in n_start..=n_end {
                let mut pixel_vals: Vec<(u64, u64, String)> = Vec::new();
                let mut all_ok = true;

                for &(m, z) in &test_coords {
                    if m >= n || z >= n { continue; }
                    let var_n = make_scott_num(&mut arena, n);
                    let m_n = make_scott_num(&mut arena, m);
                    let z_n = make_scott_num(&mut arena, z);
                    let app1 = arena.alloc(APP, result, var_n);
                    let app2 = arena.alloc(APP, app1, m_n);
                    let app3 = arena.alloc(APP, app2, z_n);
                    let mut pf: u64 = 5_000_000;
                    arena.whnf(app3, &mut pf);
                    let r = arena.follow(app3);

                    // Unwrap K wrappers
                    let mut val = r;
                    for _ in 0..10 {
                        let vv = arena.follow(val);
                        let nn = arena.nodes[vv as usize];
                        if nn.tag == K1 {
                            val = arena.follow_mut(nn.a);
                            continue;
                        }
                        if nn.tag == APP {
                            let func = arena.follow(nn.a);
                            if arena.nodes[func as usize].tag == K {
                                val = arena.follow_mut(nn.b);
                                continue;
                            }
                        }
                        break;
                    }

                    if let Some(num) = decode_scott_num(&mut arena, val, 500_000) {
                        pixel_vals.push((m, z, format!("{}", num)));
                    } else if let Some(b) = decode_bool(&mut arena, val, 200_000) {
                        pixel_vals.push((m, z, format!("{}", if b { "T" } else { "F" })));
                    } else {
                        let desc = describe(&arena, val, 0);
                        let d = if desc.len() > 60 { desc[..60].to_string() } else { desc };
                        pixel_vals.push((m, z, format!("?{}", d)));
                        all_ok = false;
                    }
                }

                // Check if row 0 and row 1 differ (gradient test)
                let r0c1 = pixel_vals.iter().find(|(m,z,_)| *m == 0 && *z == 1).map(|(_,_,v)| v.as_str()).unwrap_or("");
                let r1c1 = pixel_vals.iter().find(|(m,z,_)| *m == 1 && *z == 1).map(|(_,_,v)| v.as_str()).unwrap_or("");
                let r0c0 = pixel_vals.iter().find(|(m,z,_)| *m == 0 && *z == 0).map(|(_,_,v)| v.as_str()).unwrap_or("");
                let r1c0 = pixel_vals.iter().find(|(m,z,_)| *m == 1 && *z == 0).map(|(_,_,v)| v.as_str()).unwrap_or("");
                let r2c0 = pixel_vals.iter().find(|(m,z,_)| *m == 2 && *z == 0).map(|(_,_,v)| v.as_str()).unwrap_or("");
                let r2c1 = pixel_vals.iter().find(|(m,z,_)| *m == 2 && *z == 1).map(|(_,_,v)| v.as_str()).unwrap_or("");

                let rows_differ = r0c1 != r1c1 || r0c0 != r1c0;
                let marker = if rows_differ { "OK" } else { "GRAD?" };

                eprintln!("  N={:4}: [{}] (0,0)={} (0,1)={} (1,0)={} (1,1)={} (2,0)={} (2,1)={} nodes={}",
                    n, marker, r0c0, r0c1, r1c0, r1c1, r2c0, r2c1, arena.nodes.len());
            }
        }
        "probe3" => {
            // Systematic probe: try result(a)(b)(c) for small values
            eprintln!("Probing result(a)(b)(c) systematically...");
            let mut f = remaining_fuel;

            // Test with var=4 (16x16), small coords
            eprintln!("\n--- result(var)(m)(z) with var=4 ---");
            for m in 0..4u64 {
                for z in 0..4u64 {
                    let var_n = make_scott_num(&mut arena, 4);
                    let m_n = make_scott_num(&mut arena, m);
                    let z_n = make_scott_num(&mut arena, z);
                    let app1 = arena.alloc(APP, result, var_n);
                    let app2 = arena.alloc(APP, app1, m_n);
                    let app3 = arena.alloc(APP, app2, z_n);
                    let mut pf = f.min(5_000_000);
                    arena.whnf(app3, &mut pf);
                    let r = arena.follow(app3);

                    let num = decode_scott_num(&mut arena, r, 1_000_000);
                    let b = if num.is_none() { decode_bool(&mut arena, r, 500_000) } else { None };
                    eprint!("  ({},{})=", m, z);
                    if let Some(n) = num { eprint!("N{}", n); }
                    else if let Some(b) = b { eprint!("B{}", b as u8); }
                    else { eprint!("??"); }
                }
                eprintln!();
            }

            // Also test with var=1 (2x2)
            eprintln!("\n--- result(var)(m)(z) with var=1 ---");
            for m in 0..2u64 {
                for z in 0..2u64 {
                    let var_n = make_scott_num(&mut arena, 1);
                    let m_n = make_scott_num(&mut arena, m);
                    let z_n = make_scott_num(&mut arena, z);
                    let app1 = arena.alloc(APP, result, var_n);
                    let app2 = arena.alloc(APP, app1, m_n);
                    let app3 = arena.alloc(APP, app2, z_n);
                    let mut pf = f.min(10_000_000);
                    arena.whnf(app3, &mut pf);
                    let r = arena.follow(app3);
                    let num = decode_scott_num(&mut arena, r, 1_000_000);
                    let b = if num.is_none() { decode_bool(&mut arena, r, 500_000) } else { None };
                    eprint!("  ({},{})=", m, z);
                    if let Some(n) = num { eprint!("N{}", n); }
                    else if let Some(b) = b { eprint!("B{}", b as u8); }
                    else {
                        let desc = describe(&arena, r, 0);
                        let d = if desc.len() > 60 { &desc[..60] } else { &desc };
                        eprint!("[{}]", d);
                    }
                }
                eprintln!();
            }

            // Also try: what if we DON'T give var? result(m)(z)
            eprintln!("\n--- result(m)(z) with 2 args ---");
            for m in 0..4u64 {
                for z in 0..4u64 {
                    let m_n = make_scott_num(&mut arena, m);
                    let z_n = make_scott_num(&mut arena, z);
                    let app1 = arena.alloc(APP, result, m_n);
                    let app2 = arena.alloc(APP, app1, z_n);
                    let mut pf = f.min(5_000_000);
                    arena.whnf(app2, &mut pf);
                    let r = arena.follow(app2);
                    let num = decode_scott_num(&mut arena, r, 1_000_000);
                    let b = if num.is_none() { decode_bool(&mut arena, r, 500_000) } else { None };
                    eprint!("  ({},{})=", m, z);
                    if let Some(n) = num { eprint!("N{}", n); }
                    else if let Some(b) = b { eprint!("B{}", b as u8); }
                    else { eprint!("??"); }
                }
                eprintln!();
            }
        }
        "extract" => {
            // Extract EXPR from format(output.image(EXPR)(end))
            // Theory: result = \z. z(A)(B(z))
            //   where A = fst(result) = pipeline ITEM
            //   and B = format(end) (rest of pipeline)
            //
            // For format(end) from l.4:
            //   format(end) = \z. z(pair(FALSE,FALSE))(FALSE)
            //
            // So: fst(result) = A = ITEM containing EXPR
            //     snd(result) = B(KI) should = FALSE (end marker)

            let mut f = remaining_fuel;

            // Step 1: Get A = fst(result)
            eprintln!("\n=== Step 1: A = fst(result) ===");
            let a = pair_fst(&mut arena, result, &mut f);
            let desc_a = describe(&arena, a, 0);
            eprintln!("  A = {}", if desc_a.len() > 500 { &desc_a[..500] } else { &desc_a });

            // Step 1b: Verify snd(result) = FALSE (end marker)
            eprintln!("\n=== Step 1b: snd(result) - should be FALSE ===");
            let b_ki = pair_snd(&mut arena, result, &mut f);
            match decode_bool(&mut arena, b_ki, f.min(1_000_000)) {
                Some(true) => eprintln!("  snd(result) = TRUE"),
                Some(false) => eprintln!("  snd(result) = FALSE  ✓ (end marker)"),
                None => {
                    let desc = describe(&arena, b_ki, 0);
                    eprintln!("  snd(result) = NOT BOOL: {}", if desc.len() > 200 { &desc[..200] } else { &desc });
                }
            }

            // Step 2: Decompose A = pair(TAG, DATA)?
            eprintln!("\n=== Step 2: Decompose A ===");
            let a_fst = pair_fst(&mut arena, a, &mut f);
            let a_snd = pair_snd(&mut arena, a, &mut f);

            eprintln!("  fst(A) =");
            match decode_bool(&mut arena, a_fst, f.min(1_000_000)) {
                Some(b) => eprintln!("    BOOL({})", b),
                None => match decode_scott_num(&mut arena, a_fst, f.min(1_000_000)) {
                    Some(n) => eprintln!("    NUMBER({})", n),
                    None => {
                        let desc = describe(&arena, a_fst, 0);
                        eprintln!("    {}", if desc.len() > 300 { &desc[..300] } else { &desc });
                    }
                }
            }

            eprintln!("  snd(A) =");
            match decode_bool(&mut arena, a_snd, f.min(1_000_000)) {
                Some(b) => eprintln!("    BOOL({})", b),
                None => match decode_scott_num(&mut arena, a_snd, f.min(1_000_000)) {
                    Some(n) => eprintln!("    NUMBER({})", n),
                    None => {
                        let desc = describe(&arena, a_snd, 0);
                        eprintln!("    {}", if desc.len() > 300 { &desc[..300] } else { &desc });
                    }
                }
            }

            // Step 3: Go deeper - decompose fst(A) and snd(A)
            eprintln!("\n=== Step 3: Deeper decomposition ===");

            // fst(fst(A))
            let aa_fst = pair_fst(&mut arena, a_fst, &mut f);
            eprintln!("  fst(fst(A)) =");
            match decode_bool(&mut arena, aa_fst, f.min(1_000_000)) {
                Some(b) => eprintln!("    BOOL({})", b),
                None => match decode_scott_num(&mut arena, aa_fst, f.min(1_000_000)) {
                    Some(n) => eprintln!("    NUMBER({})", n),
                    None => {
                        let desc = describe(&arena, aa_fst, 0);
                        eprintln!("    {}", if desc.len() > 200 { &desc[..200] } else { &desc });
                    }
                }
            }

            // snd(fst(A))
            let aa_snd = pair_snd(&mut arena, a_fst, &mut f);
            eprintln!("  snd(fst(A)) =");
            match decode_bool(&mut arena, aa_snd, f.min(1_000_000)) {
                Some(b) => eprintln!("    BOOL({})", b),
                None => match decode_scott_num(&mut arena, aa_snd, f.min(1_000_000)) {
                    Some(n) => eprintln!("    NUMBER({})", n),
                    None => {
                        let desc = describe(&arena, aa_snd, 0);
                        eprintln!("    {}", if desc.len() > 200 { &desc[..200] } else { &desc });
                    }
                }
            }

            // fst(snd(A))
            let ab_fst = pair_fst(&mut arena, a_snd, &mut f);
            eprintln!("  fst(snd(A)) =");
            match decode_bool(&mut arena, ab_fst, f.min(1_000_000)) {
                Some(b) => eprintln!("    BOOL({})", b),
                None => match decode_scott_num(&mut arena, ab_fst, f.min(1_000_000)) {
                    Some(n) => eprintln!("    NUMBER({})", n),
                    None => {
                        let desc = describe(&arena, ab_fst, 0);
                        eprintln!("    {}", if desc.len() > 200 { &desc[..200] } else { &desc });
                    }
                }
            }

            // snd(snd(A))
            let ab_snd = pair_snd(&mut arena, a_snd, &mut f);
            eprintln!("  snd(snd(A)) =");
            match decode_bool(&mut arena, ab_snd, f.min(1_000_000)) {
                Some(b) => eprintln!("    BOOL({})", b),
                None => match decode_scott_num(&mut arena, ab_snd, f.min(1_000_000)) {
                    Some(n) => eprintln!("    NUMBER({})", n),
                    None => {
                        let desc = describe(&arena, ab_snd, 0);
                        eprintln!("    {}", if desc.len() > 200 { &desc[..200] } else { &desc });
                    }
                }
            }

            // Step 4: Try various candidates as EXPR - call with (1)(0)(0) and check for diamond
            eprintln!("\n=== Step 4: Try calling candidates as EXPR(1)(0)(0) ===");
            let candidates: Vec<(&str, u32)> = vec![
                ("A=fst(result)", a),
                ("snd(A)", a_snd),
                ("fst(A)", a_fst),
                ("fst(fst(A))", aa_fst),
                ("snd(fst(A))", aa_snd),
                ("fst(snd(A))", ab_fst),
                ("snd(snd(A))", ab_snd),
            ];

            for (name, node) in &candidates {
                let one = make_scott_num(&mut arena, 1);
                let zero1 = make_scott_num(&mut arena, 0);
                let zero2 = make_scott_num(&mut arena, 0);
                let app1 = arena.alloc(APP, *node, one);
                let app2 = arena.alloc(APP, app1, zero1);
                let app3 = arena.alloc(APP, app2, zero2);
                let mut pf = f.min(50_000_000);
                arena.whnf(app3, &mut pf);
                let r = arena.follow(app3);

                let b = decode_bool(&mut arena, r, 1_000_000);
                let desc = describe(&arena, r, 0);
                let d = if desc.len() > 200 { &desc[..200] } else { &desc };
                eprintln!("  {}(1)(0)(0) = {} bool={:?}", name, d, b);

                // If it's a non-boolean, check if it has diamond structure
                if b.is_none() {
                    // Try 5-element: fst = cond, snd = pair(qa, pair(qb, pair(qc, qd)))
                    let cond = pair_fst(&mut arena, r, &mut f);
                    let cond_bool = decode_bool(&mut arena, cond, 500_000);
                    let rest = pair_snd(&mut arena, r, &mut f);
                    let qa = pair_fst(&mut arena, rest, &mut f);
                    let qa_bool = decode_bool(&mut arena, qa, 500_000);
                    eprintln!("    diamond? cond={:?} qa_is_pair={}", cond_bool, qa_bool.is_none());
                }
            }

            // Step 5: Also try result(I) directly
            eprintln!("\n=== Step 5: result(I) ===");
            let i_node = arena.alloc(I, NIL, NIL);
            let ri = arena.alloc(APP, result, i_node);
            let mut pf = f.min(50_000_000);
            arena.whnf(ri, &mut pf);
            let ri_val = arena.follow(ri);
            let desc_ri = describe(&arena, ri_val, 0);
            eprintln!("  result(I) = {}", if desc_ri.len() > 500 { &desc_ri[..500] } else { &desc_ri });

            // Try result(I)(1)(0)(0)
            let one = make_scott_num(&mut arena, 1);
            let zero1 = make_scott_num(&mut arena, 0);
            let zero2 = make_scott_num(&mut arena, 0);
            let app1 = arena.alloc(APP, ri_val, one);
            let app2 = arena.alloc(APP, app1, zero1);
            let app3 = arena.alloc(APP, app2, zero2);
            let mut pf = f.min(50_000_000);
            arena.whnf(app3, &mut pf);
            let r = arena.follow(app3);
            let b = decode_bool(&mut arena, r, 1_000_000);
            let desc = describe(&arena, r, 0);
            let d = if desc.len() > 200 { &desc[..200] } else { &desc };
            eprintln!("  result(I)(1)(0)(0) = {} bool={:?}", d, b);

            // Step 6: Try passing a handler that captures the second arg
            // If format = \pipeline. pipeline(\tag. \data. data), extract data
            eprintln!("\n=== Step 6: Pass extractor handlers ===");

            // Handler: \x.\y. y (select second)
            let ki_handler = make_false(&mut arena);
            let rh = arena.alloc(APP, result, ki_handler);
            let mut pf = f.min(50_000_000);
            arena.whnf(rh, &mut pf);
            let rh_val = arena.follow(rh);
            let desc_rh = describe(&arena, rh_val, 0);
            eprintln!("  result(KI) = {}", if desc_rh.len() > 300 { &desc_rh[..300] } else { &desc_rh });
            let rh_bool = decode_bool(&mut arena, rh_val, 1_000_000);
            eprintln!("    bool={:?}", rh_bool);

            // Handler: \x.\y. x (select first) = K
            let k_handler = arena.alloc(K, NIL, NIL);
            let rk = arena.alloc(APP, result, k_handler);
            let mut pf = f.min(50_000_000);
            arena.whnf(rk, &mut pf);
            let rk_val = arena.follow(rk);
            let desc_rk = describe(&arena, rk_val, 0);
            eprintln!("  result(K) = {}", if desc_rk.len() > 300 { &desc_rk[..300] } else { &desc_rk });

            // Try calling result(K)(1)(0)(0) - maybe result(K) IS the EXPR?
            let one = make_scott_num(&mut arena, 1);
            let zero1 = make_scott_num(&mut arena, 0);
            let zero2 = make_scott_num(&mut arena, 0);
            let app1 = arena.alloc(APP, rk_val, one);
            let app2 = arena.alloc(APP, app1, zero1);
            let app3 = arena.alloc(APP, app2, zero2);
            let mut pf = f.min(50_000_000);
            arena.whnf(app3, &mut pf);
            let r = arena.follow(app3);
            let b_val = decode_bool(&mut arena, r, 1_000_000);
            let desc = describe(&arena, r, 0);
            let d = if desc.len() > 200 { &desc[..200] } else { &desc };
            eprintln!("  result(K)(1)(0)(0) = {} bool={:?}", d, b_val);

            // Step 7: Extract B from S2 node directly
            // result = S2(f, B) where result(z) = f(z)(B(z))
            // f = SI(K TAG), B = the pipeline body function
            eprintln!("\n=== Step 7: Extract B from S2 node ===");
            let r = arena.follow(result);
            let rn = arena.nodes[r as usize];
            eprintln!("  result tag = {}", rn.tag);
            if rn.tag == S2 {
                let f_part = arena.follow(rn.a);
                let b_part = arena.follow(rn.b);
                eprintln!("  f = {}", {
                    let d = describe(&arena, f_part, 0);
                    if d.len() > 200 { d[..200].to_string() } else { d }
                });
                eprintln!("  B = {}", {
                    let d = describe(&arena, b_part, 0);
                    if d.len() > 200 { d[..200].to_string() } else { d }
                });

                // B(K) = fst of the next pipeline level
                let k_node = arena.alloc(K, NIL, NIL);
                let bk = arena.alloc(APP, b_part, k_node);
                let mut pf = f.min(50_000_000);
                arena.whnf(bk, &mut pf);
                let bk_val = arena.follow(bk);
                eprintln!("  B(K) =");
                match decode_bool(&mut arena, bk_val, 1_000_000) {
                    Some(b) => eprintln!("    BOOL({})", b),
                    None => match decode_scott_num(&mut arena, bk_val, 1_000_000) {
                        Some(n) => eprintln!("    NUMBER({})", n),
                        None => {
                            let d = describe(&arena, bk_val, 0);
                            eprintln!("    {}", if d.len() > 500 { &d[..500] } else { &d });
                        }
                    }
                }

                // B(KI) = snd of the next pipeline level
                let ki_node = make_false(&mut arena);
                let bki = arena.alloc(APP, b_part, ki_node);
                let mut pf = f.min(50_000_000);
                arena.whnf(bki, &mut pf);
                let bki_val = arena.follow(bki);
                eprintln!("  B(KI) =");
                match decode_bool(&mut arena, bki_val, 1_000_000) {
                    Some(b) => eprintln!("    BOOL({})", b),
                    None => match decode_scott_num(&mut arena, bki_val, 1_000_000) {
                        Some(n) => eprintln!("    NUMBER({})", n),
                        None => {
                            let d = describe(&arena, bki_val, 0);
                            eprintln!("    {}", if d.len() > 500 { &d[..500] } else { &d });
                        }
                    }
                }

                // Now check if B is ALSO an S2 node (nested pipeline)
                let bn = arena.nodes[arena.follow(b_part) as usize];
                eprintln!("  B tag = {}", bn.tag);
                if bn.tag == S2 {
                    let b_f = arena.follow(bn.a);
                    let b_b = arena.follow(bn.b); // This is C, the next level
                    eprintln!("  B.f = {}", {
                        let d = describe(&arena, b_f, 0);
                        if d.len() > 200 { d[..200].to_string() } else { d }
                    });
                    eprintln!("  B.B (=C) = {}", {
                        let d = describe(&arena, b_b, 0);
                        if d.len() > 200 { d[..200].to_string() } else { d }
                    });

                    // C(K) = fst of NEXT next level
                    let k2 = arena.alloc(K, NIL, NIL);
                    let ck = arena.alloc(APP, b_b, k2);
                    let mut pf = f.min(50_000_000);
                    arena.whnf(ck, &mut pf);
                    let ck_val = arena.follow(ck);
                    eprintln!("  C(K) =");
                    match decode_bool(&mut arena, ck_val, 1_000_000) {
                        Some(b) => eprintln!("    BOOL({})", b),
                        None => match decode_scott_num(&mut arena, ck_val, 1_000_000) {
                            Some(n) => eprintln!("    NUMBER({})", n),
                            None => {
                                let d = describe(&arena, ck_val, 0);
                                eprintln!("    {}", if d.len() > 500 { &d[..500] } else { &d });
                            }
                        }
                    }

                    // C(KI) = snd
                    let ki2 = make_false(&mut arena);
                    let cki = arena.alloc(APP, b_b, ki2);
                    let mut pf = f.min(50_000_000);
                    arena.whnf(cki, &mut pf);
                    let cki_val = arena.follow(cki);
                    eprintln!("  C(KI) =");
                    match decode_bool(&mut arena, cki_val, 1_000_000) {
                        Some(b) => eprintln!("    BOOL({})", b),
                        None => {
                            let d = describe(&arena, cki_val, 0);
                            eprintln!("    {}", if d.len() > 200 { &d[..200] } else { &d });
                        }
                    }
                }

                // Step 8: Try B(K) as EXPR - call with (1)(0)(0)
                eprintln!("\n=== Step 8: Try B(K) as EXPR(1)(0)(0) ===");
                let one = make_scott_num(&mut arena, 1);
                let zero1 = make_scott_num(&mut arena, 0);
                let zero2 = make_scott_num(&mut arena, 0);
                let app1 = arena.alloc(APP, bk_val, one);
                let app2 = arena.alloc(APP, app1, zero1);
                let app3 = arena.alloc(APP, app2, zero2);
                let mut pf = f.min(100_000_000);
                arena.whnf(app3, &mut pf);
                let r = arena.follow(app3);
                let b_check = decode_bool(&mut arena, r, 1_000_000);
                let desc = describe(&arena, r, 0);
                let d = if desc.len() > 500 { &desc[..500] } else { &desc };
                eprintln!("  B(K)(1)(0)(0) = {} bool={:?}", d, b_check);
                // Check diamond structure
                if b_check.is_none() {
                    let cond = pair_fst(&mut arena, r, &mut f);
                    let cb = decode_bool(&mut arena, cond, 500_000);
                    eprintln!("    fst (cond?) = {:?}", cb);
                    let rest = pair_snd(&mut arena, r, &mut f);
                    let rb = decode_bool(&mut arena, rest, 500_000);
                    eprintln!("    snd = bool={:?}", rb);
                    if rb.is_none() {
                        let qa = pair_fst(&mut arena, rest, &mut f);
                        let qa_b = decode_bool(&mut arena, qa, 500_000);
                        eprintln!("    fst(snd) (qa?) = bool={:?}", qa_b);
                    }
                }

                // Step 9: Try B(K)(K) as EXPR(K) to see structure
                eprintln!("\n=== Step 9: Explore B(K) structure ===");
                let bk_fst = pair_fst(&mut arena, bk_val, &mut f);
                let bk_snd = pair_snd(&mut arena, bk_val, &mut f);
                eprintln!("  fst(B(K)) =");
                match decode_bool(&mut arena, bk_fst, 1_000_000) {
                    Some(b) => eprintln!("    BOOL({})", b),
                    None => match decode_scott_num(&mut arena, bk_fst, 1_000_000) {
                        Some(n) => eprintln!("    NUMBER({})", n),
                        None => {
                            let d = describe(&arena, bk_fst, 0);
                            eprintln!("    {}", if d.len() > 300 { &d[..300] } else { &d });
                        }
                    }
                }
                eprintln!("  snd(B(K)) =");
                match decode_bool(&mut arena, bk_snd, 1_000_000) {
                    Some(b) => eprintln!("    BOOL({})", b),
                    None => match decode_scott_num(&mut arena, bk_snd, 1_000_000) {
                        Some(n) => eprintln!("    NUMBER({})", n),
                        None => {
                            let d = describe(&arena, bk_snd, 0);
                            eprintln!("    {}", if d.len() > 300 { &d[..300] } else { &d });
                        }
                    }
                }
            } else {
                eprintln!("  Result is NOT S2 - tag = {}", rn.tag);
            }
        }
        "payload" => {
            // Extract payload from format wrapper and render as diamond tree.
            // result = S2(f, K1(payload)). The payload is the image EXPR thunk.
            // When forced, it should produce a diamond quadtree structure.
            eprintln!("Extracting payload from format wrapper...");

            let r = arena.follow(result);
            let rn = arena.nodes[r as usize];

            // result can be:
            //   S2(f, K1(payload)) - after being applied once
            //   APP(APP(S, f), K1(payload)) - before application (WHNF form)
            //   APP(S1(f), K1(payload)) - intermediate
            let b_part_opt: Option<u32> = if rn.tag == S2 {
                Some(arena.follow(rn.b))
            } else if rn.tag == APP {
                // result = APP(something, B). B is the second component.
                let b_raw = arena.follow(rn.b);
                eprintln!("  result is APP: .a tag={}, .b tag={}", arena.nodes[arena.follow(rn.a) as usize].tag, arena.nodes[b_raw as usize].tag);
                Some(b_raw)
            } else {
                None
            };

            if b_part_opt.is_none() {
                eprintln!("ERROR: result has unexpected tag={}", rn.tag);
            } else {
                // Y = result.b — the image encoder function
                // result = S(f)(Y) where f = SI(K·type_tag)
                // result(N) = f(N)(Y(N)) = N(type_tag)(Y(N))
                // For N where Scott encoding passes through: result(N) = Y(N)
                // Y(N) should be the diamond tree for resolution N
                let y_func = b_part_opt.unwrap();
                let yn = arena.nodes[y_func as usize];
                eprintln!("  Y tag = {}", yn.tag);
                let yd = describe(&arena, y_func, 0);
                eprintln!("  Y = {}", if yd.len() > 300 { &yd[..300] } else { &yd });

                // Apply Y to various N values and examine
                for n_val in [2u64, 4, 8, 16] {
                    let n_scott = make_scott_num(&mut arena, n_val);
                    let app = arena.alloc(APP, y_func, n_scott);
                    let mut pf = remaining_fuel.min(100_000_000);
                    arena.whnf(app, &mut pf);
                    let yn_result = arena.follow(app);
                    let fuel_used = 100_000_000u64.min(remaining_fuel) - pf;

                    let is_bool = decode_bool(&mut arena, yn_result, 1_000_000);
                    let tag = arena.nodes[yn_result as usize].tag;
                    eprintln!("\n  Y({}) tag={}, fuel_used={}", n_val, tag, fuel_used);
                    if let Some(b) = is_bool {
                        eprintln!("    = BOOL({})", b);
                    } else {
                        let d = describe(&arena, yn_result, 0);
                        eprintln!("    = {}", if d.len() > 300 { &d[..300] } else { &d });

                        // Check diamond structure: pair(cond, pair(qa, pair(qb, pair(qc, qd))))
                        let cond = pair_fst(&mut arena, yn_result, &mut remaining_fuel);
                        let cond_bool = decode_bool(&mut arena, cond, 1_000_000);
                        eprintln!("    cond = {:?}", cond_bool);

                        if cond_bool.is_some() {
                            // It's a diamond! Try rendering
                            eprintln!("    → Valid diamond root! Rendering...");
                            let size = (n_val as usize).next_power_of_two().max(16);
                            let mut pixels = vec![255u8; size * size];
                            let mut pixel_count = 0u64;
                            let mut rf = remaining_fuel.min(500_000_000);
                            render_diamond(&mut arena, yn_result, &mut pixels, 0, 0, size, size, &mut rf, &mut pixel_count);
                            let black = pixels.iter().filter(|&&p| p == 0).count();
                            let white = pixels.iter().filter(|&&p| p == 255).count();
                            let gray = pixels.iter().filter(|&&p| p == 128).count();
                            eprintln!("    {}x{}: {} pix rendered, black={}, white={}, gray={}",
                                size, size, pixel_count, black, white, gray);
                            let fname = format!("{}_Yn{}_diamond_{}x{}.pgm", img_path, n_val, size, size);
                            write_pgm(&fname, size, size, &pixels);
                            eprintln!("    Saved {}", fname);
                        }
                    }
                }

                // Also try: extract Y(N) for N=4 and render as diamond at various sizes
                eprintln!("\n  === Rendering Y(4) as diamond at multiple sizes ===");
                let n_scott_4 = make_scott_num(&mut arena, 4);
                let y4_app = arena.alloc(APP, y_func, n_scott_4);
                let mut pf_y4 = remaining_fuel.min(100_000_000);
                arena.whnf(y4_app, &mut pf_y4);
                let y4 = arena.follow(y4_app);
                eprintln!("  Y(4) tag={}", arena.nodes[y4 as usize].tag);

                for depth in &[4usize, 6, 8, 10] {
                    let size = 1usize << depth;
                    let mut pixels = vec![255u8; size * size];
                    let mut pixel_count = 0u64;
                    let mut rf = remaining_fuel.min(500_000_000);
                    eprintln!("  Diamond {}x{} from Y(4)...", size, size);
                    render_diamond(&mut arena, y4, &mut pixels, 0, 0, size, size, &mut rf, &mut pixel_count);
                    let black = pixels.iter().filter(|&&p| p == 0).count();
                    let white = pixels.iter().filter(|&&p| p == 255).count();
                    let gray = pixels.iter().filter(|&&p| p == 128).count();
                    eprintln!("    {} pix, black={}, white={}, gray={}, nodes={}",
                        pixel_count, black, white, gray, arena.nodes.len());
                    let fname = format!("{}_Y4_diamond_{}x{}.pgm", img_path, size, size);
                    write_pgm(&fname, size, size, &pixels);
                    eprintln!("    Saved {}", fname);
                }

                // And Y(16) at higher res
                eprintln!("\n  === Rendering Y(16) as diamond at multiple sizes ===");
                let n_scott_16 = make_scott_num(&mut arena, 16);
                let y16_app = arena.alloc(APP, y_func, n_scott_16);
                let mut pf_y16 = remaining_fuel.min(100_000_000);
                arena.whnf(y16_app, &mut pf_y16);
                let y16 = arena.follow(y16_app);
                eprintln!("  Y(16) tag={}", arena.nodes[y16 as usize].tag);

                for depth in &[4usize, 6, 8, 10] {
                    let size = 1usize << depth;
                    let mut pixels = vec![255u8; size * size];
                    let mut pixel_count = 0u64;
                    let mut rf = remaining_fuel.min(500_000_000);
                    eprintln!("  Diamond {}x{} from Y(16)...", size, size);
                    render_diamond(&mut arena, y16, &mut pixels, 0, 0, size, size, &mut rf, &mut pixel_count);
                    let black = pixels.iter().filter(|&&p| p == 0).count();
                    let white = pixels.iter().filter(|&&p| p == 255).count();
                    let gray = pixels.iter().filter(|&&p| p == 128).count();
                    eprintln!("    {} pix, black={}, white={}, gray={}, nodes={}",
                        pixel_count, black, white, gray, arena.nodes.len());
                    let fname = format!("{}_Y16_diamond_{}x{}.pgm", img_path, size, size);
                    write_pgm(&fname, size, size, &pixels);
                    eprintln!("    Saved {}", fname);
                }

                // Also try full result(N) for N=4,16 and render as diamond
                eprintln!("\n  === Rendering result(4) and result(16) as diamond ===");
                for n_val in [4u64, 16] {
                    let n_scott = make_scott_num(&mut arena, n_val);
                    let rn_app = arena.alloc(APP, result, n_scott);
                    let mut pf_rn = remaining_fuel.min(100_000_000);
                    arena.whnf(rn_app, &mut pf_rn);
                    let rn_result = arena.follow(rn_app);
                    let rn_tag = arena.nodes[rn_result as usize].tag;
                    eprintln!("  result({}) tag={}", n_val, rn_tag);

                    let is_bool = decode_bool(&mut arena, rn_result, 1_000_000);
                    if let Some(b) = is_bool {
                        eprintln!("    = BOOL({})", b);
                    } else {
                        // Try as diamond tree
                        let cond = pair_fst(&mut arena, rn_result, &mut remaining_fuel);
                        let cond_bool = decode_bool(&mut arena, cond, 1_000_000);
                        eprintln!("    cond = {:?}", cond_bool);

                        let size = 256usize;
                        let mut pixels = vec![255u8; size * size];
                        let mut pixel_count = 0u64;
                        let mut rf = remaining_fuel.min(500_000_000);
                        eprintln!("    Diamond {}x{} from result({})...", size, size, n_val);
                        render_diamond(&mut arena, rn_result, &mut pixels, 0, 0, size, size, &mut rf, &mut pixel_count);
                        let black = pixels.iter().filter(|&&p| p == 0).count();
                        let white = pixels.iter().filter(|&&p| p == 255).count();
                        let gray = pixels.iter().filter(|&&p| p == 128).count();
                        eprintln!("    {} pix, black={}, white={}, gray={}",
                            pixel_count, black, white, gray);
                        let fname = format!("{}_result{}_diamond_{}x{}.pgm", img_path, n_val, size, size);
                        write_pgm(&fname, size, size, &pixels);
                        eprintln!("    Saved {}", fname);
                    }
                }
            }
        }
        "ntest" => {
            // Quick test: for N=1..64, try result(N)(0)(1) and check if it decodes
            eprintln!("Testing which N values work for result(N)(0)(1)...");
            for n in 1..=64u64 {
                let var_n = make_scott_num(&mut arena, n);
                let m_n = make_scott_num(&mut arena, 0);
                let z_n = make_scott_num(&mut arena, 1);
                let app1 = arena.alloc(APP, result, var_n);
                let app2 = arena.alloc(APP, app1, m_n);
                let app3 = arena.alloc(APP, app2, z_n);
                let mut pf = remaining_fuel.min(10_000_000);
                arena.whnf(app3, &mut pf);
                let r = arena.follow(app3);

                // Try unwrapping
                let mut val = r;
                for _ in 0..10 {
                    let vv = arena.follow(val);
                    let nd = arena.nodes[vv as usize];
                    if nd.tag == K1 {
                        val = arena.follow_mut(nd.a);
                        continue;
                    }
                    if nd.tag == APP {
                        let func = arena.follow(nd.a);
                        if arena.nodes[func as usize].tag == K {
                            val = arena.follow_mut(nd.b);
                            continue;
                        }
                    }
                    break;
                }

                let num = decode_scott_num(&mut arena, val, 1_000_000);
                let b = if num.is_none() { decode_bool(&mut arena, val, 500_000) } else { None };
                let tag = arena.nodes[arena.follow(val) as usize].tag;
                if let Some(n_val) = num {
                    eprint!("N={:3}: num={:<5}  ", n, n_val);
                } else if let Some(bv) = b {
                    eprint!("N={:3}: bool={}   ", n, bv);
                } else {
                    eprint!("N={:3}: FAIL t={}  ", n, tag);
                }
                if n % 4 == 0 { eprintln!(); }
            }
            eprintln!();
        }
        "selftest" => {
            // Self-test: verify pair encoding, number encoding, and extraction
            eprintln!("=== Self-test: pair/number encoding ===\n");
            let mut ok = true;
            let mut test_fuel: u64 = 1_000_000;

            // Test 1: true(x)(y) = x
            {
                let t = make_true(&mut arena);
                let marker_x = arena.alloc(100, NIL, NIL);
                let marker_y = arena.alloc(101, NIL, NIL);
                let app1 = arena.alloc(APP, t, marker_x);
                let app2 = arena.alloc(APP, app1, marker_y);
                let mut f = test_fuel;
                arena.whnf(app2, &mut f);
                let r = arena.follow(app2);
                if arena.nodes[r as usize].tag == 100 {
                    eprintln!("  [OK] true(x)(y) = x");
                } else {
                    eprintln!("  [FAIL] true(x)(y) = tag {}, expected 100", arena.nodes[r as usize].tag);
                    ok = false;
                }
            }

            // Test 2: false(x)(y) = y
            {
                let f_node = make_false(&mut arena);
                let marker_x = arena.alloc(100, NIL, NIL);
                let marker_y = arena.alloc(101, NIL, NIL);
                let app1 = arena.alloc(APP, f_node, marker_x);
                let app2 = arena.alloc(APP, app1, marker_y);
                let mut f = test_fuel;
                arena.whnf(app2, &mut f);
                let r = arena.follow(app2);
                if arena.nodes[r as usize].tag == 101 {
                    eprintln!("  [OK] false(x)(y) = y");
                } else {
                    eprintln!("  [FAIL] false(x)(y) = tag {}, expected 101", arena.nodes[r as usize].tag);
                    ok = false;
                }
            }

            // Test 3: pair(a,b)(K)(dummy) = a
            {
                let marker_a = arena.alloc(100, NIL, NIL);
                let marker_b = arena.alloc(101, NIL, NIL);
                let p = make_pair(&mut arena, marker_a, marker_b);
                let mut f = test_fuel;
                let fst = pair_fst(&mut arena, p, &mut f);
                if arena.nodes[fst as usize].tag == 100 {
                    eprintln!("  [OK] fst(pair(a,b)) = a");
                } else {
                    eprintln!("  [FAIL] fst(pair(a,b)) = tag {}, expected 100", arena.nodes[fst as usize].tag);
                    ok = false;
                }
            }

            // Test 4: pair(a,b)(KI)(dummy) = b
            {
                let marker_a = arena.alloc(100, NIL, NIL);
                let marker_b = arena.alloc(101, NIL, NIL);
                let p = make_pair(&mut arena, marker_a, marker_b);
                let mut f = test_fuel;
                let snd = pair_snd(&mut arena, p, &mut f);
                if arena.nodes[snd as usize].tag == 101 {
                    eprintln!("  [OK] snd(pair(a,b)) = b");
                } else {
                    eprintln!("  [FAIL] snd(pair(a,b)) = tag {}, expected 101", arena.nodes[snd as usize].tag);
                    ok = false;
                }
            }

            // Test 5: decode_bool on true/false
            {
                let t = make_true(&mut arena);
                let b = decode_bool(&mut arena, t, test_fuel);
                if b == Some(true) {
                    eprintln!("  [OK] decode_bool(true) = true");
                } else {
                    eprintln!("  [FAIL] decode_bool(true) = {:?}", b);
                    ok = false;
                }
                let f = make_false(&mut arena);
                let b2 = decode_bool(&mut arena, f, test_fuel);
                if b2 == Some(false) {
                    eprintln!("  [OK] decode_bool(false) = false");
                } else {
                    eprintln!("  [FAIL] decode_bool(false) = {:?}", b2);
                    ok = false;
                }
            }

            // Test 6: encode/decode numbers 0..15
            eprintln!();
            for n in 0..=15u64 {
                let num_node = make_scott_num(&mut arena, n);
                let decoded = decode_scott_num(&mut arena, num_node, test_fuel);
                if decoded == Some(n) {
                    eprint!("  [OK] num({})={} ", n, n);
                } else {
                    eprint!("  [FAIL] num({})={:?} ", n, decoded);
                    ok = false;
                }
                if (n + 1) % 8 == 0 { eprintln!(); }
            }

            // Test 7: Verify number 3 compact matches server-verified encoding
            {
                let expected = "kXX--kkD-XkXX--D----XkXX--kkD-XkXX--D----XkXX--kkD-XXD----XXD----------";
                let n3_from_compact = parse_compact(&mut arena, expected.as_bytes());
                let dec = decode_scott_num(&mut arena, n3_from_compact, test_fuel);
                if dec == Some(3) {
                    eprintln!("  [OK] Server-verified compact of 3 decodes to 3");
                } else {
                    eprintln!("  [FAIL] Server-verified compact of 3 decodes to {:?}", dec);
                    ok = false;
                }
            }

            // Test 8: Verify large numbers
            for n in &[100u64, 255, 1000, 65535] {
                let num_node = make_scott_num(&mut arena, *n);
                let decoded = decode_scott_num(&mut arena, num_node, 10_000_000);
                if decoded == Some(*n) {
                    eprintln!("  [OK] num({}) round-trips", n);
                } else {
                    eprintln!("  [FAIL] num({}) decoded as {:?}", n, decoded);
                    ok = false;
                }
            }

            eprintln!();
            if ok {
                eprintln!("=== All self-tests PASSED ===");
            } else {
                eprintln!("=== Some self-tests FAILED ===");
            }
        }
        "walk1" => {
            // Walk the output structure using 1-arg pair extraction.
            // The program output uses 1-arg Scott pairs:
            //   pair1(A, B) = S(SI(KA))(KB)
            //   pair1(A, B)(handler) = handler(A)(B)
            // Extract: node(K) = A, node(KI) = B
            eprintln!("Walking output with 1-arg pair extraction...");
            let mut f = remaining_fuel;
            let mut current = result;

            for i in 0..20 {
                eprintln!("\n=== List item {} ===", i);

                // Check if current is a boolean (nil/end marker)
                let is_bool = decode_bool(&mut arena, current, f.min(1_000_000));
                if let Some(b) = is_bool {
                    eprintln!("  = BOOL({}) → end of list", b);
                    break;
                }

                // Extract head (1-arg)
                let head = pair1_fst(&mut arena, current, &mut f);
                let tail = pair1_snd(&mut arena, current, &mut f);

                // Describe head
                let desc = describe(&arena, head, 0);
                eprintln!("  head = {}", if desc.len() > 300 { &desc[..300] } else { &desc });

                // Try decode head as various types
                if let Some(b) = decode_bool(&mut arena, head, f.min(1_000_000)) {
                    eprintln!("  head = BOOL({})", b);
                } else if let Some(n) = decode_scott_num(&mut arena, head, f.min(1_000_000)) {
                    eprintln!("  head = NUMBER({})", n);
                } else {
                    // Head might be a 1-arg pair (nested structure)
                    let h_fst = pair1_fst(&mut arena, head, &mut f);
                    let h_snd = pair1_snd(&mut arena, head, &mut f);
                    let hf_bool = decode_bool(&mut arena, h_fst, f.min(500_000));
                    let hf_num = if hf_bool.is_none() { decode_scott_num(&mut arena, h_fst, f.min(500_000)) } else { None };
                    let hs_bool = decode_bool(&mut arena, h_snd, f.min(500_000));
                    let hs_num = if hs_bool.is_none() { decode_scott_num(&mut arena, h_snd, f.min(500_000)) } else { None };
                    eprintln!("  head.fst = bool={:?} num={:?}", hf_bool, hf_num);
                    eprintln!("  head.snd = bool={:?} num={:?}", hs_bool, hs_num);

                    // If head is a 5-element (5-argument image symbol):
                    // head(handler) = handler(a1)(a2)(a3)(a4)(a5)
                    // Try extracting 5 fields by applying a 5-arg extractor
                    eprintln!("  Trying head as 5-arg constructor...");
                    for arg_idx in 0..5 {
                        // Build extractor: λa1..a5. a_{arg_idx+1}
                        // For arg 0: K(K(K(K)))  → gets 1st (K⁴)
                        // For arg 1: K(K(K(KI))) → gets 2nd ...
                        // Actually: extractors for 5-arg:
                        // arg0: λa.λb.λc.λd.λe. a  = difficult in pure SKI
                        // Instead, apply head to a sequence of K/KI extractors
                        // head(K)(K)(K)(K)(K) gets us: K(a1)(a2)(a3)(a4)(a5) = a1(a3)(a4)(a5)
                        // That's wrong. We need Church-style extractors.

                        // Simpler: just apply head to 5 unique markers and see what comes out
                        let markers: Vec<u32> = (0..5).map(|j| arena.alloc(110 + j, NIL, NIL)).collect();
                        let mut app = arena.alloc(APP, head, markers[0]);
                        for &m in &markers[1..] {
                            app = arena.alloc(APP, app, m);
                        }
                        let mut pf = f.min(10_000_000);
                        arena.whnf(app, &mut pf);
                        let r = arena.follow(app);
                        let tag = arena.nodes[r as usize].tag;
                        let r_desc = describe(&arena, r, 0);
                        let rd = if r_desc.len() > 100 { &r_desc[..100] } else { &r_desc };
                        eprintln!("    head(m0)(m1)(m2)(m3)(m4) tag={} = {}", tag, rd);
                        break; // only need to test once
                    }

                    // Also try: head as 1-arg pair and walk deeper
                    eprintln!("  Walking head as 1-arg pair list:");
                    let mut hcur = head;
                    for j in 0..8 {
                        let hb = decode_bool(&mut arena, hcur, f.min(500_000));
                        if let Some(b) = hb {
                            eprintln!("    [{}] BOOL({})", j, b);
                            break;
                        }
                        let hn = decode_scott_num(&mut arena, hcur, f.min(500_000));
                        if let Some(n) = hn {
                            eprintln!("    [{}] NUMBER({})", j, n);
                            break;
                        }
                        let hh = pair1_fst(&mut arena, hcur, &mut f);
                        let ht = pair1_snd(&mut arena, hcur, &mut f);
                        let hh_b = decode_bool(&mut arena, hh, f.min(500_000));
                        let hh_n = if hh_b.is_none() { decode_scott_num(&mut arena, hh, f.min(500_000)) } else { None };
                        eprintln!("    [{}] fst=bool={:?} num={:?}", j, hh_b, hh_n);
                        hcur = ht;
                    }
                }

                // Describe tail briefly
                let td = describe(&arena, tail, 0);
                eprintln!("  tail = {}", if td.len() > 200 { &td[..200] } else { &td });

                // Check if tail is boolean (end of list)
                let tail_bool = decode_bool(&mut arena, tail, f.min(1_000_000));
                if let Some(b) = tail_bool {
                    eprintln!("  tail = BOOL({}) → last item", b);
                    break;
                }

                current = tail;
            }
        }
        "render1" => {
            // Render using 1-arg pair structure.
            // Theory: result is a 1-arg pair list. The first item contains
            // the image data or pixel function. Try rendering various ways.
            eprintln!("Extracting from 1-arg pair structure...");
            let mut f = remaining_fuel;

            // Get first item and rest
            let item = pair1_fst(&mut arena, result, &mut f);
            let rest = pair1_snd(&mut arena, result, &mut f);
            eprintln!("  item extracted");

            let rest_bool = decode_bool(&mut arena, rest, f.min(1_000_000));
            eprintln!("  rest is bool: {:?}", rest_bool);

            // Try item as pixel function: item(N)(m)(z)
            eprintln!("\n--- item(N)(m)(z) with N=16 ---");
            let size = render_var;
            let mut pixels = vec![128u8; (size * size) as usize];
            let mut num_count = 0u64;
            let mut bool_count = 0u64;
            let mut fail_count = 0u64;
            let fuel_per_pixel: u64 = 10_000_000;

            for m in 0..size {
                for z in 0..size {
                    let var_n = make_scott_num(&mut arena, size);
                    let m_n = make_scott_num(&mut arena, m);
                    let z_n = make_scott_num(&mut arena, z);
                    let app1 = arena.alloc(APP, item, var_n);
                    let app2 = arena.alloc(APP, app1, m_n);
                    let app3 = arena.alloc(APP, app2, z_n);
                    let mut pf = fuel_per_pixel;
                    arena.whnf(app3, &mut pf);
                    let r = arena.follow(app3);

                    if let Some(n) = decode_scott_num(&mut arena, r, 1_000_000) {
                        pixels[(m * size + z) as usize] = (n.min(255)) as u8;
                        num_count += 1;
                    } else if let Some(b) = decode_bool(&mut arena, r, 500_000) {
                        pixels[(m * size + z) as usize] = if b { 255 } else { 0 };
                        bool_count += 1;
                    } else {
                        fail_count += 1;
                    }
                }
                if (m + 1) % 4 == 0 {
                    eprint!("\r  row {}/{} ({} num, {} bool, {} fail)     ", m + 1, size, num_count, bool_count, fail_count);
                }
            }
            eprintln!();
            eprintln!("  Decoded: {} num, {} bool, {} fail", num_count, bool_count, fail_count);

            let max_val = pixels.iter().copied().filter(|&p| p != 128).max().unwrap_or(1);
            let fname = format!("{}_item_{}x{}.pgm", img_path, size, size);
            write_pgm(&fname, size as usize, size as usize, &pixels);
            eprintln!("  Saved {}", fname);

            if max_val > 1 && max_val < 255 {
                let normalized: Vec<u8> = pixels.iter().map(|&p| {
                    if p == 128 { 128 }
                    else { ((p as u32) * 255 / max_val as u32).min(255) as u8 }
                }).collect();
                let fname2 = format!("{}_item_norm_{}x{}.pgm", img_path, size, size);
                write_pgm(&fname2, size as usize, size as usize, &normalized);
                eprintln!("  Saved {}", fname2);
            }

            // Print sample
            let sample = 8.min(size);
            eprintln!("  Sample pixels:");
            for m in 0..sample {
                eprint!("    ");
                for z in 0..sample {
                    eprint!("{:3} ", pixels[(m * size + z) as usize]);
                }
                eprintln!();
            }

            // Also try: item(m)(z) without N
            eprintln!("\n--- item(m)(z) without N ---");
            for m in 0..4u64 {
                for z in 0..4u64 {
                    let m_n = make_scott_num(&mut arena, m);
                    let z_n = make_scott_num(&mut arena, z);
                    let app1 = arena.alloc(APP, item, m_n);
                    let app2 = arena.alloc(APP, app1, z_n);
                    let mut pf = f.min(10_000_000);
                    arena.whnf(app2, &mut pf);
                    let r = arena.follow(app2);
                    let num = decode_scott_num(&mut arena, r, 1_000_000);
                    let b = if num.is_none() { decode_bool(&mut arena, r, 500_000) } else { None };
                    eprint!("  ({},{})=", m, z);
                    if let Some(n) = num { eprint!("N{} ", n); }
                    else if let Some(b) = b { eprint!("B{} ", b as u8); }
                    else { eprint!("?? "); }
                }
                eprintln!();
            }

            // Also try rendering the diamond tree from item
            eprintln!("\n--- item as diamond tree ---");
            for depth in &[4usize, 8] {
                let sz = 1usize << depth;
                let mut pix = vec![255u8; sz * sz];
                let mut pc = 0u64;
                let mut rf = f.min(100_000_000);
                render_diamond(&mut arena, item, &mut pix, 0, 0, sz, sz, &mut rf, &mut pc);
                let black = pix.iter().filter(|&&p| p == 0).count();
                let white = pix.iter().filter(|&&p| p == 255).count();
                eprintln!("  {}x{}: {} rendered, black={}, white={}", sz, sz, pc, black, white);
                let fname = format!("{}_item_diamond_{}x{}.pgm", img_path, sz, sz);
                write_pgm(&fname, sz, sz, &pix);
            }
        }
        _ => {
            eprintln!("Unknown decode mode: {}", decode_mode);
        }
    }
}

/// Build Church numeral in arena: Church n = succ^n(zero)
/// where succ = S(S(KS)K), zero = KI
fn make_church_num(arena: &mut Arena, n: u64) -> u32 {
    // zero = KI
    let ki = make_false(arena);
    if n == 0 {
        return ki;
    }
    // succ = S(S(KS)K)
    let k = arena.alloc(K, NIL, NIL);
    let s = arena.alloc(S, NIL, NIL);
    let ks = arena.alloc(APP, k.clone(), s.clone());
    let s2 = arena.alloc(S, NIL, NIL);
    let s_ks = arena.alloc(APP, s2, ks);
    let k2 = arena.alloc(K, NIL, NIL);
    let succ = arena.alloc(APP, s_ks, k2);

    let mut result = ki;
    for _ in 0..n {
        // Need fresh succ each time since lazy eval shares nodes
        let k = arena.alloc(K, NIL, NIL);
        let s = arena.alloc(S, NIL, NIL);
        let ks = arena.alloc(APP, k, s);
        let s2 = arena.alloc(S, NIL, NIL);
        let s_ks = arena.alloc(APP, s2, ks);
        let k2 = arena.alloc(K, NIL, NIL);
        let succ = arena.alloc(APP, s_ks, k2);
        result = arena.alloc(APP, succ, result);
    }
    result
}

/// Build Scott-encoded number in arena.
fn make_scott_num(arena: &mut Arena, n: u64) -> u32 {
    // 0 = pair(false, nil), NOT bare nil
    let nil = make_false(arena);
    let false_node = make_false(arena);
    let zero = make_pair(arena, false_node, nil);

    if n == 0 {
        return zero;
    }

    let mut bits = Vec::new();
    let mut temp = n;
    while temp > 0 {
        bits.push(temp & 1);
        temp >>= 1;
    }
    // Build from MSB to LSB (reversed bits, build pair chain)
    // Terminate with pair(false, nil) = zero, NOT bare nil
    let mut result = zero;
    for &bit in bits.iter().rev() {
        let bit_node = if bit == 1 {
            make_true(arena)
        } else {
            make_false(arena)
        };
        result = make_pair(arena, bit_node, result);
    }
    result
}

/// Recursively decode a pair structure.
fn deep_decode(arena: &mut Arena, node: u32, fuel: u64, depth: usize, max_depth: usize) {
    if depth > max_depth || fuel == 0 { return; }

    let indent = "  ".repeat(depth);

    // Check if boolean
    match decode_bool(arena, node, fuel / 10) {
        Some(true) => { println!("{}TRUE", indent); return; }
        Some(false) => { println!("{}FALSE (nil)", indent); return; }
        None => {}
    }

    // Check if number
    if let Some(n) = decode_scott_num(arena, node, fuel / 10) {
        if n > 0 {
            println!("{}NUMBER({})", indent, n);
            return;
        }
    }

    // Try as pair (2-arg Scott pair extraction)
    let mut f1 = fuel / 4;
    let fst = pair_fst(arena, node, &mut f1);

    let mut f2 = fuel / 4;
    let snd = pair_snd(arena, node, &mut f2);

    println!("{}PAIR(", indent);
    deep_decode(arena, fst, fuel / 4, depth + 1, max_depth);
    println!("{},", indent);
    deep_decode(arena, snd, fuel / 4, depth + 1, max_depth);
    println!("{})", indent);
}

/// Collect boolean leaves from a pair tree by DFS.
/// Treats pairs as internal nodes, booleans as leaves.
fn collect_bool_leaves(
    arena: &mut Arena,
    node: u32,
    fuel: &mut u64,
    leaves: &mut Vec<u8>,
    max_leaves: usize,
) {
    if leaves.len() >= max_leaves || *fuel == 0 { return; }

    // Check if boolean leaf
    let b = decode_bool(arena, node, (*fuel).min(100000));
    match b {
        Some(true) => { leaves.push(1); return; }
        Some(false) => { leaves.push(0); return; }
        None => {}
    }

    // Not a boolean - treat as pair and recurse
    let fst = pair_fst(arena, node, fuel);
    collect_bool_leaves(arena, fst, fuel, leaves, max_leaves);

    if leaves.len() >= max_leaves { return; }

    let snd = pair_snd(arena, node, fuel);
    collect_bool_leaves(arena, snd, fuel, leaves, max_leaves);
}

/// Write PGM image file.
fn write_pgm(filename: &str, width: usize, height: usize, pixels: &[u8]) {
    let mut f = fs::File::create(filename).expect("failed to create PGM file");
    let header = format!("P5\n{} {}\n255\n", width, height);
    f.write_all(header.as_bytes()).expect("write header");
    f.write_all(pixels).expect("write pixels");
}

/// Extract pair's first element: pair(K)(dummy) → A
/// 2-arg Scott pair needs TWO arguments to extract.
fn pair_fst(arena: &mut Arena, node: u32, fuel: &mut u64) -> u32 {
    let k_sel = arena.alloc(K, NIL, NIL);
    let app1 = arena.alloc(APP, node, k_sel);
    let dummy = arena.alloc(I, NIL, NIL); // dummy second arg (ignored by pair)
    let app2 = arena.alloc(APP, app1, dummy);
    arena.whnf(app2, fuel);
    arena.follow(app2)
}

/// Extract pair's second element: pair(KI)(dummy) → B
/// 2-arg Scott pair needs TWO arguments to extract.
fn pair_snd(arena: &mut Arena, node: u32, fuel: &mut u64) -> u32 {
    let ki = make_false(arena);
    let app1 = arena.alloc(APP, node, ki);
    let dummy = arena.alloc(I, NIL, NIL); // dummy second arg (ignored by pair)
    let app2 = arena.alloc(APP, app1, dummy);
    arena.whnf(app2, fuel);
    arena.follow(app2)
}

/// 1-arg pair extraction: node(K) → first element
/// For 1-arg Scott pairs: S(SI(KA))(KB)(K) = K(A)(B) = A
fn pair1_fst(arena: &mut Arena, node: u32, fuel: &mut u64) -> u32 {
    let k_sel = arena.alloc(K, NIL, NIL);
    let app = arena.alloc(APP, node, k_sel);
    arena.whnf(app, fuel);
    arena.follow(app)
}

/// 1-arg pair extraction: node(KI) → second element
/// For 1-arg Scott pairs: S(SI(KA))(KB)(KI) = KI(A)(B) = B
fn pair1_snd(arena: &mut Arena, node: u32, fuel: &mut u64) -> u32 {
    let ki = make_false(arena);
    let app = arena.alloc(APP, node, ki);
    arena.whnf(app, fuel);
    arena.follow(app)
}

/// Render diamond quadtree to pixel buffer.
/// Diamond encoding (5-element):
///   PAIR(cond, PAIR(qa, PAIR(qb, PAIR(qc, qd))))
///   cond = boolean pixel value at this level
///   qa = m-1, z-1 (top-left/NW)
///   qb = m-1, z+1 (top-right/NE)
///   qc = m+1, z-1 (bottom-left/SW)
///   qd = m+1, z+1 (bottom-right/SE)
/// Leaf: FALSE (white) or TRUE (black)
fn render_diamond(
    arena: &mut Arena,
    node: u32,
    pixels: &mut [u8],
    x: usize,
    y: usize,
    size: usize,
    img_width: usize,
    fuel: &mut u64,
    count: &mut u64,
) {
    if *fuel == 0 || size == 0 { return; }

    // Check if it's a boolean leaf
    let is_bool = decode_bool(arena, node, (*fuel).min(200000));
    match is_bool {
        Some(false) => {
            fill_rect(pixels, x, y, size, 255, img_width); // white
            *count += (size * size) as u64;
            return;
        }
        Some(true) => {
            fill_rect(pixels, x, y, size, 0, img_width); // black
            *count += (size * size) as u64;
            return;
        }
        None => {}
    }

    // At pixel level, extract condition for color
    if size <= 1 {
        let cond = pair_fst(arena, node, fuel);
        let b = decode_bool(arena, cond, (*fuel).min(200000));
        let color = match b {
            Some(true) => 0u8,   // black
            Some(false) => 255u8, // white
            None => 128u8,        // gray (unknown)
        };
        if x < img_width && y < img_width {
            pixels[y * img_width + x] = color;
        }
        *count += 1;
        return;
    }

    // Diamond structure: PAIR(cond, PAIR(qa, PAIR(qb, PAIR(qc, qd))))
    let rest = pair_snd(arena, node, fuel);       // PAIR(qa, PAIR(qb, PAIR(qc, qd)))
    let qa = pair_fst(arena, rest, fuel);          // qa (NW: m-1, z-1)
    let rest2 = pair_snd(arena, rest, fuel);       // PAIR(qb, PAIR(qc, qd))
    let qb = pair_fst(arena, rest2, fuel);         // qb (NE: m-1, z+1)
    let rest3 = pair_snd(arena, rest2, fuel);      // PAIR(qc, qd)
    let qc = pair_fst(arena, rest3, fuel);         // qc (SW: m+1, z-1)
    let qd = pair_snd(arena, rest3, fuel);         // qd (SE: m+1, z+1)

    let half = size / 2;
    render_diamond(arena, qa, pixels, x, y, half, img_width, fuel, count);
    render_diamond(arena, qb, pixels, x + half, y, half, img_width, fuel, count);
    render_diamond(arena, qc, pixels, x, y + half, half, img_width, fuel, count);
    render_diamond(arena, qd, pixels, x + half, y + half, half, img_width, fuel, count);
}

/// Alternate interpretation: PAIR(PAIR(nw, ne), PAIR(sw, se))
fn render_quadtree_v2(
    arena: &mut Arena,
    node: u32,
    pixels: &mut [u8],
    x: usize,
    y: usize,
    size: usize,
    img_width: usize,
    fuel: &mut u64,
    count: &mut u64,
) {
    if *fuel == 0 || size == 0 { return; }

    let is_bool = decode_bool(arena, node, (*fuel).min(200000));
    match is_bool {
        Some(false) => {
            fill_rect(pixels, x, y, size, 255, img_width);
            *count += (size * size) as u64;
            return;
        }
        Some(true) => {
            fill_rect(pixels, x, y, size, 0, img_width);
            *count += (size * size) as u64;
            return;
        }
        None => {}
    }

    if size <= 1 {
        if x < img_width && y < img_width {
            pixels[y * img_width + x] = 128;
        }
        *count += 1;
        return;
    }

    let top = pair_fst(arena, node, fuel);
    let bottom = pair_snd(arena, node, fuel);
    let nw = pair_fst(arena, top, fuel);
    let ne = pair_snd(arena, top, fuel);
    let sw = pair_fst(arena, bottom, fuel);
    let se = pair_snd(arena, bottom, fuel);

    let half = size / 2;
    render_quadtree_v2(arena, nw, pixels, x, y, half, img_width, fuel, count);
    render_quadtree_v2(arena, ne, pixels, x + half, y, half, img_width, fuel, count);
    render_quadtree_v2(arena, sw, pixels, x, y + half, half, img_width, fuel, count);
    render_quadtree_v2(arena, se, pixels, x + half, y + half, half, img_width, fuel, count);
}

/// Fill a rectangular region with a color.
fn fill_rect(pixels: &mut [u8], x: usize, y: usize, size: usize, color: u8, img_width: usize) {
    for dy in 0..size {
        for dx in 0..size {
            let px = x + dx;
            let py = y + dy;
            if px < img_width && py < img_width {
                pixels[py * img_width + px] = color;
            }
        }
    }
}
