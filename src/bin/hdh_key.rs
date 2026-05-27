/// HDH Key Generation — Visual Trace
///
/// Walks every stage of the HDH sponge hash for a user-chosen message and
/// prints a formatted trace so you can see exactly how the 512-bit key is
/// constructed from the input.

use hdh::hash::{iv_state, RATE_BYTES, OUTPUT_BYTES};
use hdh::{chi, theta, phi};
use hdh::state::State;
use blake3::hash as blake3_hash;

// ── helpers ───────────────────────────────────────────────────────────────────

fn bar(ch: char, width: usize) -> String {
    std::iter::repeat(ch).take(width).collect()
}

fn section(title: &str) {
    let w = 72;
    let pad = if title.len() + 4 < w { (w - title.len() - 4) / 2 } else { 0 };
    println!();
    println!("┌{}┐", bar('─', w));
    println!("│{} {} {}│", bar(' ', pad), title, bar(' ', w - pad - title.len() - 2));
    println!("└{}┘", bar('─', w));
}

fn lane_snapshot(label: &str, s: &State) {
    // Show lane 0 of each share and the parity.
    println!("  {label}");
    println!("    s1[0] = {:016x}  s2[0] = {:016x}", s.s1[0], s.s2[0]);
    println!("    s3[0] = {:016x}  s4[0] = {:016x}", s.s3[0], s.s4[0]);
    println!("    parity[0] = {:016x}", s.parity[0]);
}

fn print_bytes_row(data: &[u8]) {
    for (i, b) in data.iter().enumerate() {
        if i % 16 == 0 { print!("    {:3}: ", i); }
        print!("{:02x} ", b);
        if (i + 1) % 16 == 0 { println!(); }
    }
    if data.len() % 16 != 0 { println!(); }
}

fn hamming_diff(a: &[u8; OUTPUT_BYTES], b: &[u8; OUTPUT_BYTES]) -> u32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x ^ y).count_ones()).sum()
}

// ── padding (mirrors hash.rs) ─────────────────────────────────────────────────

fn pad_blocks(data: &[u8]) -> Vec<[u8; RATE_BYTES]> {
    let mut buf: Vec<u8> = data.to_vec();
    buf.push(0x01);
    let rem = buf.len() % RATE_BYTES;
    let extra = if rem == 0 { RATE_BYTES } else { RATE_BYTES - rem };
    buf.resize(buf.len() + extra, 0x00);
    *buf.last_mut().unwrap() ^= 0x80;
    buf.chunks_exact(RATE_BYTES)
        .map(|c| c.try_into().unwrap())
        .collect()
}

// ── absorb one block (mirrors hash.rs, but returns the pre-permute state too) ─

fn absorb_block_traced(mut state: State, block: &[u8; RATE_BYTES]) -> (State, State) {
    let mut pos = 0;
    macro_rules! xor_share {
        ($share:ident, $range:expr) => {
            for i in $range {
                state.$share[i] ^= u64::from_le_bytes(
                    block[pos..pos + 8].try_into().unwrap(),
                );
                pos += 8;
            }
        };
    }
    xor_share!(s1, 0..25);
    xor_share!(s2, 0..25);
    xor_share!(s3, 0..25);
    xor_share!(s4, 0..17);
    for i in 0..25 {
        state.parity[i] = state.s1[i] ^ state.s2[i] ^ state.s3[i] ^ state.s4[i];
    }
    let after_xor = state.clone();
    // Apply 6-round permutation step-by-step so we can log each round.
    (after_xor, state) // returned: (state after XOR, ready to permute)
}

fn permute_traced(mut s: State) -> State {
    for round_idx in 0u64..6 {
        // Derive round constant (same as lib.rs).
        let digest = blake3_hash(&round_idx.to_le_bytes());
        let b = digest.as_bytes();
        let r0 = u64::from_le_bytes(b[ 0.. 8].try_into().unwrap());
        let r1 = u64::from_le_bytes(b[ 8..16].try_into().unwrap());
        let r2 = u64::from_le_bytes(b[16..24].try_into().unwrap());
        let r3 = u64::from_le_bytes(b[24..32].try_into().unwrap());

        println!();
        println!("  ── Round {} ─────────────────────────────────────────────────────", round_idx);
        println!("    RC: r0={:016x}  r1={:016x}", r0, r1);
        println!("         r2={:016x}  r3={:016x}", r2, r3);

        // Add round constant.
        for i in 0..25 {
            s.s1[i] ^= r0;
            s.s2[i] ^= r1;
            s.s3[i] ^= r2;
            s.s4[i] ^= r3;
            s.parity[i] = s.s1[i] ^ s.s2[i] ^ s.s3[i] ^ s.s4[i];
        }
        lane_snapshot("After ⊕ RC:", &s);

        // χ — nonlinear layer.
        s = chi::chi(&s);
        lane_snapshot("After χ  :", &s);

        // θ — linear diffusion.
        s = theta::theta(&s);
        lane_snapshot("After θ  :", &s);

        // Φ — state-dependent routing.
        s = phi::phi(&s);
        lane_snapshot("After Φ  :", &s);
    }
    s
}

// ── main ──────────────────────────────────────────────────────────────────────

fn main() {
    let message = b"HDH Key Demo: hello world";

    // ── 1. Input ──────────────────────────────────────────────────────────────
    section("1 · INPUT MESSAGE");
    println!("  Text  : {:?}", std::str::from_utf8(message).unwrap_or("<binary>"));
    println!("  Bytes : {} byte(s)", message.len());
    print_bytes_row(message);

    // ── 2. Padding ────────────────────────────────────────────────────────────
    section("2 · PAD10*1 PADDING");
    let blocks = pad_blocks(message);
    println!("  Rate  : {} bytes ({} bits)", RATE_BYTES, RATE_BYTES * 8);
    println!("  Blocks: {} (each {} bytes)", blocks.len(), RATE_BYTES);
    println!();
    println!("  Pad rule: append 0x01, zero-fill to block boundary, XOR 0x80 into last byte.");
    println!("  First/last 16 bytes of block 0:");
    print_bytes_row(&blocks[0][..16]);
    print!("  ...");
    let tail = &blocks[0][RATE_BYTES - 16..];
    for b in tail { print!(" {:02x}", b); }
    println!("  ← 0x80 at end");

    // ── 3. IV ─────────────────────────────────────────────────────────────────
    section("3 · BLAKE3-DERIVED IV STATE");
    let iv = iv_state();
    println!("  Each of the 100 lanes (4 shares × 25) gets a unique 64-bit IV.");
    println!("  Derived via BLAKE3(\"HDH-IV-s<share>-l<lane>\")[0..8].");
    println!();
    println!("  IV lane samples:");
    for i in 0..4 {
        println!("    s1[{i}]={:016x}  s2[{i}]={:016x}  s3[{i}]={:016x}  s4[{i}]={:016x}",
            iv.s1[i], iv.s2[i], iv.s3[i], iv.s4[i]);
    }
    println!("    ... (25 lanes per share, 4 shares = 100 unique IVs)");

    // ── 4. Absorption ─────────────────────────────────────────────────────────
    section("4 · ABSORPTION  (message ⊕ rate lanes → 6-round permutation)");
    let mut state = iv;

    for (blk_idx, block) in blocks.iter().enumerate() {
        println!();
        println!("  ╔══ Block {} of {} ══════════════════════════════════════════════════╗",
            blk_idx + 1, blocks.len());

        // Show state before XOR.
        lane_snapshot("State BEFORE ⊕ block:", &state);

        // XOR the rate lanes.
        let (after_xor, pre_perm) = absorb_block_traced(state.clone(), block);
        let _ = pre_perm; // same as after_xor, we just use after_xor
        lane_snapshot("State AFTER  ⊕ block:", &after_xor);

        println!();
        println!("  ─── Permutation: 6 rounds of RC ⊕ χ ⊕ θ ⊕ Φ ────────────────────");

        // Run the traced 6-round permutation.
        state = permute_traced(after_xor);

        lane_snapshot("State AFTER permutation:", &state);
        println!("  ╚══════════════════════════════════════════════════════════════════╝");
    }

    // ── 5. Squeezing ──────────────────────────────────────────────────────────
    section("5 · SQUEEZE  (extract s1[0..8] = 512-bit key)");
    let mut key = [0u8; OUTPUT_BYTES];
    for (i, &lane) in state.s1[..8].iter().enumerate() {
        key[i * 8..(i + 1) * 8].copy_from_slice(&lane.to_le_bytes());
    }
    println!("  Reading 8 lanes × 64 bits = 512 bits from s1[0..8] of the final state.");
    println!();
    println!("  s1[0]={:016x}  s1[1]={:016x}", state.s1[0], state.s1[1]);
    println!("  s1[2]={:016x}  s1[3]={:016x}", state.s1[2], state.s1[3]);
    println!("  s1[4]={:016x}  s1[5]={:016x}", state.s1[4], state.s1[5]);
    println!("  s1[6]={:016x}  s1[7]={:016x}", state.s1[6], state.s1[7]);

    // ── 6. The Key ────────────────────────────────────────────────────────────
    section("6 · HDH KEY  (512-bit / 64-byte digest)");
    println!("  Input : {:?}", std::str::from_utf8(message).unwrap_or("<binary>"));
    println!();
    println!("  ┌─ HDH Key (hex, 64 bytes) ─────────────────────────────────────┐");
    for row in 0..8 {
        let slice = &key[row * 8..(row + 1) * 8];
        print!("  │  {:3}–{:3}: ", row * 8, row * 8 + 7);
        for b in slice { print!("{:02x} ", b); }
        // also show ASCII column
        print!("  │  ");
        for b in slice { print!("{}", if b.is_ascii_graphic() { *b as char } else { '·' }); }
        println!();
    }
    println!("  └───────────────────────────────────────────────────────────────┘");

    // ── 7. Avalanche demo ─────────────────────────────────────────────────────
    section("7 · AVALANCHE CHECK  (flip 1 input bit → ~50% output change)");
    let mut flipped = message.to_vec();
    flipped[0] ^= 0x01;  // flip lowest bit of first byte
    let key2 = hdh::hash::hash(&flipped);

    let changed_bits = hamming_diff(&key, &key2);
    let pct = changed_bits as f64 / 512.0 * 100.0;

    println!("  Original  : {:?}", std::str::from_utf8(message).unwrap());
    println!("  Flipped   : {:?}", std::str::from_utf8(&flipped).unwrap_or("<binary>"));
    println!();
    println!("  Original key[0..8] : {:02x?}", &key[..8]);
    println!("  Flipped  key[0..8] : {:02x?}", &key2[..8]);
    println!();
    println!("  Bits changed: {changed_bits} / 512  ({pct:.1}%)");

    let bar_len = (changed_bits as usize * 40 / 512).max(1);
    let ideal   = 20usize; // 50% of 40
    println!("  Change  : [{}{}] {:.1}%", "█".repeat(bar_len), "░".repeat(40 - bar_len), pct);
    println!("  Ideal≈50%: [{}{}]",       "│".repeat(ideal),   " ".repeat(40 - ideal));
    println!();
    if pct > 40.0 && pct < 60.0 {
        println!("  ✓ Avalanche satisfied: a 1-bit input change avalanches to ~50% of key bits.");
    } else {
        println!("  ✗ Avalanche outside expected range — check algorithm.");
    }

    // ── 8. Summary ────────────────────────────────────────────────────────────
    section("8 · ALGORITHM SUMMARY");
    println!("  State   : 6400 bits  (4 shares × 25 lanes × 64 bits)");
    println!("  Rate    : 5888 bits  (rate lanes absorbed per block)");
    println!("  Capacity: 512  bits  (s4[17..25], never exposed)");
    println!("  Rounds  : 6 per block  (RC ⊕ χ ⊕ θ ⊕ Φ)");
    println!("  Output  : 512 bits  (s1[0..8] after final permutation)");
    println!();
    println!("  Layer roles:");
    println!("    χ (chi)   — nonlinear: g = s1·s2 ⊕ s3·s4; rotations 0,7,13,31");
    println!("    θ (theta) — linear diffusion: ±1, ±7 strides on Z₂₅  (B(θ)=6)");
    println!("    Φ (phi)   — state-driven routing: j=(s1⊕s2)%25, k=(s3⊕s4)%25");
    println!("    RC        — BLAKE3(round_index)[0..32] → 4 independent 64-bit words");
    println!();
    println!("  The 512-bit output above IS the HDH key for the given input.");
}
