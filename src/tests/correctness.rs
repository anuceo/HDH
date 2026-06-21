/// Correctness tests C1–C5.

use crate::framework::TestCase;
use crate::hdh::{hash, pad_blocks, xof, OUTPUT_BYTES, RATE_BYTES};

const CAT: &str = "Correctness";

// ── C1: Deterministic output ──────────────────────────────────────────────────

pub fn c1_determinism() -> TestCase {
    let inputs: &[&[u8]] = &[b"", b"abc", b"HDH test vector", &[0u8; 736], &[0xFFu8; 1500]];
    let mut all_ok = true;
    let mut tries = 0usize;

    for &data in inputs {
        for _ in 0..20 {
            if hash(data) != hash(data) {
                all_ok = false;
            }
            tries += 1;
        }
    }

    if all_ok {
        TestCase::pass("C1", CAT, "Deterministic output",
            format!("hash(x) == hash(x) verified across {tries} trials on {} inputs", inputs.len()))
    } else {
        TestCase::fail("C1", CAT, "Deterministic output",
            "Mismatch detected",
            "Non-determinism: same input produced different outputs — implementation bug")
    }
}

// ── C2: Avalanche effect ──────────────────────────────────────────────────────

pub fn c2_avalanche() -> TestCase {
    // For a 64-byte input, flip each of the 512 input bits and count output bit changes.
    let base = [0x5Au8; 64];
    let base_hash = hash(&base);

    let mut total_flips = 0u64;
    let mut total_bits = 0u64;

    for byte_idx in 0..64 {
        for bit in 0..8u8 {
            let mut perturbed = base;
            perturbed[byte_idx] ^= 1 << bit;
            let perturbed_hash = hash(&perturbed);

            let flipped: u32 = base_hash
                .iter()
                .zip(perturbed_hash.iter())
                .map(|(a, b)| (a ^ b).count_ones())
                .sum();

            total_flips += flipped as u64;
            total_bits += 512;
        }
    }

    let avg_pct = total_flips as f64 / total_bits as f64 * 100.0;
    // Expect 50% ± 5% (tight criterion: reject if outside 45–55%).
    let ok = avg_pct >= 45.0 && avg_pct <= 55.0;

    if ok {
        TestCase::pass("C2", CAT, "Avalanche effect: single-bit flip → ~50% output change",
            format!("avg {:.2}% of output bits changed across 512 single-bit flips ({total_flips}/{total_bits})",
                avg_pct))
    } else {
        TestCase::fail("C2", CAT, "Avalanche effect: single-bit flip → ~50% output change",
            format!("avg {avg_pct:.2}% bits changed"),
            format!("expected 45–55%; got {avg_pct:.2}% — diffusion is too weak or too strong"))
    }
}

// ── C3: Fixed-length output ───────────────────────────────────────────────────

pub fn c3_output_length() -> TestCase {
    let lengths = [0, 1, 10, 63, 64, 65, 735, 736, 737, 1472, 10_000];
    let mut wrong = Vec::new();

    for &len in &lengths {
        let data = vec![0xABu8; len];
        let out = hash(&data);
        if out.len() != OUTPUT_BYTES {
            wrong.push(len);
        }
    }

    if wrong.is_empty() {
        TestCase::pass("C3", CAT, "Fixed-length output: always 512 bits",
            format!("hash() returns exactly {OUTPUT_BYTES} bytes for all {} tested input sizes", lengths.len()))
    } else {
        TestCase::fail("C3", CAT, "Fixed-length output: always 512 bits",
            format!("wrong output length for input sizes: {wrong:?}"),
            "output byte count != 64 — squeeze logic is broken")
    }
}

// ── C4: Sponge compliance — absorb/squeeze boundary ──────────────────────────

pub fn c4_sponge_boundary() -> TestCase {
    let mut issues = Vec::new();

    // 4a. Multi-block produces a distinct result from single-block.
    let single = hash(&[0xABu8; 100]);
    let multi  = hash(&[0xABu8; 837]); // crosses 736-byte block boundary
    if single == multi {
        issues.push("single-block == multi-block hash (colliding inputs)");
    }

    // 4b. Prepending a zero byte changes the hash (prefix sensitivity).
    let mut prefixed = vec![0u8];
    prefixed.extend_from_slice(&[0xABu8; 100]);
    if hash(&prefixed) == single {
        issues.push("hash(0x00 || msg) == hash(msg) — prefix ignored");
    }

    // 4c. Capacity bytes must differ between two distinct inputs
    //     (indirect: if the sponge capacity was leaked into rate, two hashes
    //      would show a systematic relationship; we just check distinctness).
    let a = hash(b"message A");
    let b = hash(b"message B");
    if a == b {
        issues.push("hash('message A') == hash('message B') — trivial collision");
    }

    // 4d. XOF prefix consistency: first 64 bytes of XOF == hash().
    let h  = hash(b"xof-test");
    let xf = xof(b"xof-test", 128);
    if h != xf[..OUTPUT_BYTES] {
        issues.push("xof first 64 bytes != hash() — squeeze inconsistency");
    }

    if issues.is_empty() {
        TestCase::pass("C4", CAT, "Sponge compliance: absorb/squeeze boundary",
            "multi-block distinct, prefix-sensitive, XOF first 64B == hash()")
    } else {
        TestCase::fail("C4", CAT, "Sponge compliance: absorb/squeeze boundary",
            issues.join("; "),
            "sponge construction violates at least one boundary property")
    }
}

// ── C5: Padding correctness (pad10*1) ────────────────────────────────────────

pub fn c5_padding() -> TestCase {
    let mut issues = Vec::new();

    // Test lengths at and around each block boundary (RATE_BYTES = 736).
    for &len in &[0usize, 1, 734, 735, 736, 737, 1471, 1472, 1473] {
        let data = vec![0x42u8; len];
        let blocks = pad_blocks(&data);

        // Each block must be exactly RATE_BYTES.
        for b in &blocks {
            if b.len() != RATE_BYTES {
                issues.push(format!("input {len}B: block size {} != {RATE_BYTES}", b.len()));
            }
        }

        // Must produce at least 1 block (even for empty input).
        if blocks.is_empty() {
            issues.push(format!("input {len}B: zero blocks produced"));
            continue;
        }

        // Last byte of last block must have bit 7 set (0x80 marker).
        let last = *blocks.last().unwrap();
        if last[RATE_BYTES - 1] & 0x80 == 0 {
            issues.push(format!("input {len}B: last byte 0x{:02x} — missing 0x80 marker",
                last[RATE_BYTES - 1]));
        }

        // First byte after message must be 0x01.
        let first_pad_byte = last_pad_start(&data, &blocks);
        if let Some(b) = first_pad_byte {
            if b != 0x01 {
                issues.push(format!("input {len}B: first padding byte 0x{b:02x} != 0x01"));
            }
        }

        // No two different lengths should produce identical padded outputs.
    }

    // Prefix-freeness: distinct lengths → distinct padded byte sequences
    let p1 = pad_blocks(&vec![0u8; 0]);
    let p2 = pad_blocks(&vec![0u8; 1]);
    if p1 == p2 {
        issues.push("empty and 1-byte inputs produce identical padding — not prefix-free".into());
    }

    if issues.is_empty() {
        TestCase::pass("C5", CAT, "Padding correctness (pad10*1)",
            "block size, 0x01 start marker, 0x80 end marker all correct for 9 boundary lengths")
    } else {
        TestCase::fail("C5", CAT, "Padding correctness (pad10*1)",
            issues.join("; "),
            "pad10*1 rule violated — sponge absorb-boundary is malformed")
    }
}

/// Locate the first padding byte in the padded output.
fn last_pad_start(original: &[u8], blocks: &[[u8; RATE_BYTES]]) -> Option<u8> {
    let total_rate_bytes = blocks.len() * RATE_BYTES;
    let flat: Vec<u8> = blocks.iter().flat_map(|b| b.iter().copied()).collect();
    // The first padding byte is at position original.len() in the flat buffer.
    if original.len() < total_rate_bytes {
        Some(flat[original.len()])
    } else {
        None
    }
}

// ── Runner ────────────────────────────────────────────────────────────────────

pub fn run(_seed: u64) -> Vec<TestCase> {
    vec![
        c1_determinism(),
        c2_avalanche(),
        c3_output_length(),
        c4_sponge_boundary(),
        c5_padding(),
    ]
}
