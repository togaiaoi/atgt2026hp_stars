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
fn decode_scott_num(arena: &mut Arena, node: u32, fuel: u64) -> Option<u64> {
    let mut bits: Vec<u8> = Vec::new();
    let mut current = node;
    let mut total_fuel = fuel;

    for _ in 0..64 {
        if total_fuel == 0 { break; }

        let is_nil = decode_bool(arena, current, total_fuel / 10);
        if is_nil == Some(false) {
            break;
        }

        let k_sel = arena.alloc(K, NIL, NIL);
        let fst_app = arena.alloc(APP, current, k_sel);
        let mut f1 = total_fuel / 10;
        arena.whnf(fst_app, &mut f1);
        let first = arena.follow(fst_app);

        let ki = make_false(arena);
        let snd_app = arena.alloc(APP, current, ki);
        let mut f2 = total_fuel / 10;
        arena.whnf(snd_app, &mut f2);
        let second = arena.follow(snd_app);

        total_fuel = total_fuel.saturating_sub(fuel / 10 * 2);

        let bit_val = decode_bool(arena, first, total_fuel / 10);
        match bit_val {
            Some(true) => { bits.push(1); current = second; }
            Some(false) => { bits.push(0); current = second; }
            None => break,
        }
    }

    if bits.is_empty() {
        return Some(0);
    }
    let mut n: u64 = 0;
    for (i, &b) in bits.iter().enumerate() {
        n += (b as u64) << i;
    }
    Some(n)
}

/// Try to decode the result as a stream of bytes (list of numbers).
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

        let is_nil = decode_bool(arena, current, remaining_fuel / 20);
        if is_nil == Some(false) {
            eprintln!("\n[End of stream after {} bytes]", count);
            break;
        }
        if is_nil.is_none() {
            eprintln!("\n[Not a boolean at position {}, cannot decode further]", count);
            break;
        }

        let k_sel = arena.alloc(K, NIL, NIL);
        let fst_app = arena.alloc(APP, current, k_sel);
        let mut f1 = remaining_fuel / 20;
        arena.whnf(fst_app, &mut f1);
        let head = arena.follow(fst_app);

        let ki = make_false(arena);
        let snd_app = arena.alloc(APP, current, ki);
        let mut f2 = remaining_fuel / 20;
        arena.whnf(snd_app, &mut f2);
        let tail = arena.follow(snd_app);

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
fn decode_number_list(arena: &mut Arena, node: u32, fuel: u64, max_items: usize) -> Vec<u64> {
    let mut result = Vec::new();
    let mut current = node;
    let mut remaining_fuel = fuel;

    for _ in 0..max_items {
        if remaining_fuel == 0 { break; }

        let is_nil = decode_bool(arena, current, remaining_fuel / 20);
        if is_nil == Some(false) { break; }
        if is_nil.is_none() { break; }

        let k_sel = arena.alloc(K, NIL, NIL);
        let fst_app = arena.alloc(APP, current, k_sel);
        let mut f1 = remaining_fuel / 20;
        arena.whnf(fst_app, &mut f1);
        let head = arena.follow(fst_app);

        let ki = make_false(arena);
        let snd_app = arena.alloc(APP, current, ki);
        let mut f2 = remaining_fuel / 20;
        arena.whnf(snd_app, &mut f2);
        let tail = arena.follow(snd_app);

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
fn decode_bool_list(arena: &mut Arena, node: u32, fuel: u64, max_items: usize) -> Vec<bool> {
    let mut result = Vec::new();
    let mut current = node;
    let mut remaining_fuel = fuel;

    for _ in 0..max_items {
        if remaining_fuel == 0 { break; }

        let is_nil = decode_bool(arena, current, remaining_fuel / 20);
        if is_nil == Some(false) { break; }
        if is_nil.is_none() { break; }

        // Extract head
        let k_sel = arena.alloc(K, NIL, NIL);
        let fst_app = arena.alloc(APP, current, k_sel);
        let mut f1 = remaining_fuel / 20;
        arena.whnf(fst_app, &mut f1);
        let head = arena.follow(fst_app);

        // Extract tail
        let ki = make_false(arena);
        let snd_app = arena.alloc(APP, current, ki);
        let mut f2 = remaining_fuel / 20;
        arena.whnf(snd_app, &mut f2);
        let tail = arena.follow(snd_app);

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
        _ => {
            eprintln!("Unknown decode mode: {}", decode_mode);
        }
    }
}
