//! HDH — DFA-Hardened Hash Function
//!
//! A post-quantum-resistant sponge hash built on the HDH χ-core permutation.
//!
//! USAGE
//!   hdh [OPTIONS] [INPUT...]
//!
//! OPTIONS
//!   --256          Output 256-bit digest  (default)
//!   --512          Output 512-bit digest
//!   --1024         Output 1024-bit digest
//!   --prf <KEY>    Keyed PRF mode (KEY = 64 hex bytes)
//!   --demo         Print demonstration vectors and permutation stats
//!
//! EXAMPLES
//!   hdh "hello world"
//!   hdh --512 "the quick brown fox"
//!   hdh --demo
//!   echo -n "abc" | hdh
//!
//! ALGORITHM PARAMETERS
//!   State  : 6400 bits  (4 DFA-resistant shares × 25 lanes × 64 bits)
//!   Rate   : 5888 bits  (736 bytes absorbed/squeezed per permutation call)
//!   Capacity: 512 bits  (hidden; birthday collision security = 256 bits)
//!   Padding : pad10*1   (prefix-free, length-extension immune)
//!   Rounds  : 6 per permutation call

use hdh::algorithm::{
    hdh_hash, hdh_hash_256, hdh_hash_512, hdh_prf,
    RATE_BYTES, CAPACITY_BYTES,
};
use std::io::Read;

fn to_hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn print_digest(label: &str, digest: &[u8]) {
    println!("{label}: {}", to_hex(digest));
}

fn run_demo() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║          HDH — DFA-Hardened Hash Function  (demo)           ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    println!("── Algorithm parameters ────────────────────────────────────────");
    println!("  State bits   : {}", 4 * 25 * 64);
    println!("  Rate bytes   : {RATE_BYTES}");
    println!("  Capacity bits: {}", CAPACITY_BYTES * 8);
    println!("  Rounds       : {}", hdh::algorithm::ROUNDS);
    println!("  Security     : 256-bit collision / 256-bit preimage");
    println!("  Post-quantum : ~171-bit quantum collision (BHT bound)");
    println!();

    println!("── Known-answer test vectors ───────────────────────────────────");
    let vectors: &[(&str, &[u8])] = &[
        ("empty",     b""),
        ("\"abc\"",   b"abc"),
        ("\"hello world\"", b"hello world"),
        ("\"The quick brown fox\"", b"The quick brown fox jumps over the lazy dog"),
    ];
    for (label, msg) in vectors {
        let d256 = hdh_hash_256(msg);
        let d512 = hdh_hash_512(msg);
        println!("  {label}");
        println!("    HDH-256 : {}", to_hex(&d256));
        println!("    HDH-512 : {}", to_hex(&d512));
    }
    println!();

    println!("── Avalanche (single-bit input flip → output change) ───────────");
    let base_msg = b"avalanche test message";
    let base = hdh_hash_256(base_msg);
    let mut total_flipped = 0u32;
    for byte_idx in 0..base_msg.len() {
        let mut msg2 = base_msg.to_vec();
        msg2[byte_idx] ^= 0x80;
        let h2 = hdh_hash_256(&msg2);
        total_flipped += base
            .iter()
            .zip(h2.iter())
            .map(|(a, b)| (a ^ b).count_ones())
            .sum::<u32>();
    }
    let avg_pct = total_flipped as f64 / (base_msg.len() as f64 * 256.0) * 100.0;
    println!("  Average bits flipped per 1-bit input change: {avg_pct:.1}%  (ideal 50%)");
    println!();

    println!("── Streaming API ───────────────────────────────────────────────");
    use hdh::algorithm::HdhSponge;
    let mut sponge = HdhSponge::new();
    sponge.absorb(b"hello ");
    sponge.absorb(b"world");
    let mut digest = [0u8; 32];
    sponge.squeeze(&mut digest);
    let oneshot = hdh_hash_256(b"hello world");
    println!("  Streaming digest  : {}", to_hex(&digest));
    println!("  One-shot digest   : {}", to_hex(&oneshot));
    println!("  Match             : {}", digest == oneshot);
    println!();

    println!("── Keyed PRF mode ──────────────────────────────────────────────");
    let key = [0xABu8; 64];
    let prf_out = hdh_prf(&key, b"hello world", 32);
    let hash_out = hdh_hash(b"hello world", 32);
    println!("  PRF (key=0xAB×64) : {}", to_hex(&prf_out));
    println!("  Unkeyed hash      : {}", to_hex(&hash_out));
    println!("  Distinct          : {}", prf_out != hash_out);
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();

    if args.is_empty() {
        // No arguments: read stdin.
        let mut input = Vec::new();
        if std::io::stdin().read_to_end(&mut input).is_ok() && !input.is_empty() {
            print_digest("HDH-256", &hdh_hash_256(&input));
        } else {
            run_demo();
        }
        return;
    }

    let mut output_bits = 256usize;
    let mut prf_key: Option<[u8; 64]> = None;
    let mut inputs: Vec<String> = Vec::new();
    let mut i = 0;

    while i < args.len() {
        match args[i].as_str() {
            "--demo" => {
                run_demo();
                return;
            }
            "--256"  => output_bits = 256,
            "--512"  => output_bits = 512,
            "--1024" => output_bits = 1024,
            "--prf"  => {
                i += 1;
                if i >= args.len() {
                    eprintln!("error: --prf requires a 128-hex-char key argument");
                    std::process::exit(1);
                }
                let hex_key = &args[i];
                if hex_key.len() != 128 {
                    eprintln!("error: PRF key must be exactly 64 bytes (128 hex chars), got {} chars", hex_key.len());
                    std::process::exit(1);
                }
                let mut key = [0u8; 64];
                for (j, chunk) in hex_key.as_bytes().chunks(2).enumerate() {
                    let hi = char::from(chunk[0]).to_digit(16).unwrap_or(0) as u8;
                    let lo = char::from(chunk[1]).to_digit(16).unwrap_or(0) as u8;
                    key[j] = (hi << 4) | lo;
                }
                prf_key = Some(key);
            }
            arg => inputs.push(arg.to_string()),
        }
        i += 1;
    }

    for input_str in &inputs {
        let data = input_str.as_bytes();
        let tag = format!("HDH-{output_bits}({input_str:?})");

        let digest = match &prf_key {
            Some(key) => hdh_prf(key, data, output_bits / 8),
            None      => hdh_hash(data, output_bits / 8),
        };
        print_digest(&tag, &digest);
    }

    if inputs.is_empty() {
        run_demo();
    }
}
